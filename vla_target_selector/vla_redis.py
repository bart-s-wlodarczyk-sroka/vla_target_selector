import re
import json
import yaml
import time
import threading
import math
import random
import pytz
import scipy.constants as con
import pandas as pd
import numpy as np
from io import StringIO
from functools import reduce
from datetime import datetime
from geometry import Target, great_circle_distance, cosine_attenuation
# from target_selector_variables import priority_decay, primary_sensitivity_exponent
from optimizer import Optimizer
from test_plot import test_plot

try:
    from .logger import log as logger
    from .vla_db import Triage
    from .redis_tools import (publish,
                              get_redis_key,
                              write_pair_redis,
                              connect_to_redis,
                              delete_key)

except ImportError:
    from logger import log as logger
    from vla_db import Triage
    from redis_tools import (publish,
                             get_redis_key,
                             write_pair_redis,
                             connect_to_redis,
                             delete_key)


class ProcessingStatus(object):
    def __init__(self, value):
        self._proc_status = value

    @property
    def proc_status(self):
        return self._proc_status

    @proc_status.setter
    def proc_status(self, value):
        self._proc_status = value

    @proc_status.deleter
    def proc_status(self):
        del self._proc_status


pStatus = ProcessingStatus("ready")


successfully_processed = []


class Listen(threading.Thread):
    """

    ADD IN DOCUMENTATION

    Examples:
        >>> client = Listen(['alerts', 'sensor_alerts'])
        >>> client.start()

    When start() is called, a loop is started that subscribes to the "alerts" and
    "sensor_alerts" channels on the Redis server. Depending on the which message
    that passes over which channel, various processes are run:

    Alerts:
        1. Configure:
            -
        2. Deconfigure
            -

    Sensor Alerts:
        1. data_suspect:
            -
        2. schedule_blocks:
            -
        3.

    Things left to do:
        1. Listen for a success message from the processing nodes. Once this
           success/failure message has been returned, then add to the database.
    """

    def __init__(self, chan=None, config_file='target_selector.yml'):

        if not chan:
            chan = ['sensor_alerts', 'alerts']

        threading.Thread.__init__(self)

        # Initialize redis connection
        self.redis_server = connect_to_redis()

        # Subscribe to channel
        self.p = self.redis_server.pubsub(ignore_subscribe_messages=True)
        self.p.psubscribe(chan)

        # Database connection and triaging
        self.engine = Triage(config_file)

        self.channel_actions = {
            'alerts': self._alerts,
            'sensor_alerts': self._sensor_alerts,
        }

        self.alerts_actions = {
            'deconfigure': self._pass,
            'configure': self._pass,
            # 'deconfigure': self._deconfigure,
            # 'configure': self._configure,
            'conf_complete': self._pass,
            'capture-init': self._pass,
            'capture-start': self._pass,
            'capture-stop': self._pass,
            'capture-done': self._pass
        }

        self.sensor_actions = {
            'data_suspect': self._pass,
            'schedule_blocks': self._pass,
            'pool_resources': self._pass,
            'observation_status': self._pass,
            'target': self._target_query,
            'frequency': self._frequency,
            'processing_success': self._pass,
            'acknowledge': self._pass,
            'new_obs': self._pass,
            'current_obs': self._pass,
            'duration': self._duration
        }

    def run(self):
        """Runs continuously to listen for messages that come in from specific
           redis channels. Main function that handles the processing of the
           messages that come through redis.
        """

        for item in self.p.listen():
            self._message_to_func(item['channel'], self.channel_actions)(item['data'])

    def fetch_data(self, product_id, mode):
        """Fetches telescope status data and selects targets when telescope status data is stored

        Parameters:
            product_id: (str)
                Product ID parsed from redis message
            mode: (str)
                Mode in which function is to be run; "recalculating" is used to update parameters for the currently
                processing sample, "current_obs" fetches the data and writes it to current_obs redis keys, while
                "new_obs" fetches the data for comparison against the currently processing block, without overwriting
                 keys relating to the currently processing observations
        Returns:
            None
        """
        if ("None" not in str(self._get_sensor_value(product_id, "new_obs:coords")))\
                and ("None" not in str(self._get_sensor_value(product_id, "new_obs:frequency"))):
            try:
                # create redis key-val pairs to store current observation data & current telescope status data
                new_coords = self._get_sensor_value(product_id, "new_obs:coords")
                coords_ra = float(new_coords.split(", ")[0])
                coords_dec = float(new_coords.split(", ")[1])
                new_freq = self._get_sensor_value(product_id, "new_obs:frequency")
                new_duration = self._get_sensor_value(product_id, "new_obs:duration")

                if mode == "current_obs":
                    # logger.info("Writing values of [{}:new_obs:*] to [{}:current_obs:*]"
                    #             .format(product_id, product_id))
                    write_pair_redis(self.redis_server, "{}:{}:coords".format(product_id, mode), new_coords)
                    # logger.info("Fetched [{}:{}:coords]: [{}]"
                    #             .format(product_id, mode, self._get_sensor_value(product_id, "{}:coords"
                    #                                                              .format(mode))))
                    write_pair_redis(self.redis_server, "{}:{}:frequency".format(product_id, mode), new_freq)
                    # logger.info("Fetched [{}:{}:frequency]: [{}]"
                    #             .format(product_id, mode, self._get_sensor_value(product_id, "{}:frequency"
                    #                                                              .format(mode))))
                    write_pair_redis(self.redis_server, "{}:{}:duration".format(product_id, mode), new_duration)
                    # logger.info("Fetched [{}:{}:duration]: [{}]"
                    #             .format(product_id, mode, self._get_sensor_value(product_id, "{}:duration"
                    #                                                              .format(mode))))

                targets = self\
                    .engine.select_targets(np.deg2rad(coords_ra),
                                           np.deg2rad(coords_dec),
                                           current_freq=new_freq,
                                           beam_rad=self._beam_radius(new_freq))

                columns = ['ra', 'decl', 'source_id', 'dist_c', 'table_name', 'priority']
                targ_64_dict = targets.head(64).loc[:, columns].to_dict('list')

                if len(targets.index) == 0:
                    self.coord_error(coords=self._get_sensor_value(product_id, "{}:coords".format(mode)),
                                     frequency=self._get_sensor_value(product_id, "{}:frequency".format(mode)),
                                     mode=mode,
                                     product_id=product_id)

                    # write null table to redis
                    key = 'None'
                    channel = "bluse:///set"
                    write_pair_redis(self.redis_server, key, "None")
                    publish(self.redis_server, channel, key)

                else:
                    write_pair_redis(self.redis_server, "{}:{}:target_list"
                                     .format(product_id, mode), json.dumps(targ_64_dict))

            except KeyError:
                pass

    """

    Alerts Functions

    """

    def _alerts(self, message):
        """Response to message from the Alerts channel. Runs a function depending
        on the message sent.

        Parameters:
            message: (str)
                Message passed over the alerts channel

        Returns:
            None
        """
        sensor, product_id = self._parse_sensor_name(message)
        self._message_to_func(sensor, self.alerts_actions)(product_id)

    def _pass(self, item):
        """Temporary function to handle alerts that we don't care about responding
        to at the moment
        """
        return 0

    def _capture_start(self, message):
        """Function that responds to capture start updates. Upon receipt the target list is generated, triaged and
         published & observation start time taken.

       Parameters:
            message: (str)
                Message passed over the alerts channel
        Returns:
            None
        """

        logger.info("Capture start message received: {}".format(message))
        product_id = message

        if pStatus.proc_status == "ready":
            self.fetch_data(product_id, mode="current_obs")
            if "None" not in str(self._get_sensor_value(product_id, "current_obs:target_list")):
                sub_arr_id = "0"  # TODO: CHANGE TO HANDLE SUB-ARRAYS
                pulled_targets = json.loads(self._get_sensor_value(product_id, "current_obs:target_list"))
                self._publish_targets(pulled_targets, product_id, sub_arr_id)

    """

    Sensor Alerts Functions

    """

    def _sensor_alerts(self, message):
        """Response to sensor_alerts channel. Runs a function based on the input
        message

        Parameters:
            message: (str)
                Message received from listening to the sensor_alerts channel

        Returns:
            None
        """
        product_id, sensor = self._parse_sensor_name(message)

        if sensor.endswith('frequency'):
            sensor = 'frequency'

        if sensor.endswith('duration'):
            sensor = 'duration'

        if sensor.startswith('new_obs'):
            sensor = 'new_obs'

        if sensor.startswith('current_obs'):
            sensor = 'current_obs'

        self._message_to_func(sensor, self.sensor_actions)(message)

    def _target_query(self, message):
        """Response to message from the Sensor Alerts channel. If both the right
        ascension and declination are stored, then the database is queried
        for

        Parameters:
            message: (str)
                Message passed over the sensor alerts channel

        Returns:
            None
        """
        coord_value = get_redis_key(self.redis_server, message)

        logger.info("Target coordinate message received: {}, {}".format(message, coord_value))
        product_id, sensor_name = message.split(':', 2)[:2]
        coord_key = "{}:new_obs:coords".format(product_id)
        write_pair_redis(self.redis_server, coord_key, coord_value)
        # logger.info("Wrote [{}] to [{}]".format(coord_value, coord_key))

    def _frequency(self, message):
        """Response to a frequency message from the sensor_alerts channel.

        Parameters:
            message: (str)
                Message passed over sensor_alerts channels. Acts as the key to
                query Redis in the case of this function.

        Returns:
            None
        """
        frequency_value = get_redis_key(self.redis_server, message)
        logger.info("Frequency message received: {}, {}".format(message, frequency_value))

        product_id, sensor_name = message.split(':')
        frequency_key = "{}:new_obs:frequency".format(product_id)
        write_pair_redis(self.redis_server, frequency_key, frequency_value)
        # logger.info("Wrote [{}] to [{}]".format(frequency_value, frequency_key))

    def _duration(self, message):
        """Response to a duration message from the sensor_alerts channel.

        Parameters:
            message: (str)
                Message passed over sensor_alerts channels. Acts as the key to
                query Redis in the case of this function.

        Returns:
            None
        """
        duration_value = get_redis_key(self.redis_server, message)
        logger.info("Duration value message received: {}, {}".format(message, duration_value))

        product_id, sensor_name = message.split(':')
        duration_key = "{}:new_obs:duration".format(product_id)
        write_pair_redis(self.redis_server, duration_key, duration_value)
        # logger.info("Wrote [{}] to [{}]".format(frequency_value, frequency_key))
        self._capture_start(product_id)
        self.store_metadata(product_id, mode="new_obs")

    def _processing_success(self, message):
        """Response to a successful processing success message from the sensor_alerts channel.

        Parameters:
            message: (str)
                Message passed over sensor_alerts channel.

        Returns:
            None
        """

        # message = "array_1:success_XX.XXXX_YY.YYYY"
        product_id = message.split(':')[0]
        ra = "{:0.4f}".format(float(message.split("_")[2]))
        decl = "{:0.4f}".format(float(message.split("_")[3]))

        self.engine.update_obs_status(duration=str(self._get_sensor_value(product_id, "current_obs:duration")),
                                      beamform_ra=ra,
                                      beamform_decl=decl,
                                      processed='1')

    """

    Internal Methods

    """

    def hms_dms_decimal_convert(self, coords):
        """Converts H:M:S, D:M:S RA/Dec values to decimal for processing

        Parameters:
            coords: (str)
                Coordinate string

        Returns:
            coords_ra_deg, coords_dec_deg: (float, float)
                Tuple of RA/Dec coordinates in decimal format
        """

        coords_ra_hms, coords_dec_dms = coords.split(', ')
        coords_ra_h = int(coords_ra_hms.split(':')[0])
        coords_ra_m = int(coords_ra_hms.split(':')[-2])
        coords_ra_s = float(coords_ra_hms.split(':')[-1])
        coords_dec_d = int(coords_dec_dms.split(':')[0])
        coords_dec_m = int(coords_dec_dms.split(':')[-2])
        coords_dec_s = float(coords_dec_dms.split(':')[-1])

        if coords_ra_h < 0:
            coords_ra_deg = float(15 * (coords_ra_h - (coords_ra_m / 60) - (coords_ra_s / 3600)))
        else:
            coords_ra_deg = float(15 * (coords_ra_h + (coords_ra_m / 60) + (coords_ra_s / 3600)))
        if coords_dec_d < 0:
            coords_dec_deg = float(coords_dec_d - (coords_dec_m / 60) - (coords_dec_s / 3600))
        else:
            coords_dec_deg = float(coords_dec_d + (coords_dec_m / 60) + (coords_dec_s / 3600))

        if coords_ra_deg < 0:
            coords_ra_deg = 360 + coords_ra_deg

        return coords_ra_deg, coords_dec_deg

    def append_tbdfm(self, table, coords, frequency):
        """Function to calculate and append TBDFM values to tables containing targets

        TBDFM = To Be Determined Figure of Merit

        Parameters:
            table: (dataframe)
                Table from which to calculate TBDFM and to which the values are appended
            coords: TODO: ADD DESCRIPTION
                TODO: ADD DESCRIPTION
            frequency: TODO: ADD DESCRIPTION
                TODO: ADD DESCRIPTION
        Returns:
            table: (dataframe)
                Dataframe with appended TBDFM values for each row
        """
        # create empty array for TBDFM parameter values in the target list
        tbdfm_param = np.full(table.shape[0], 0, dtype=float)

        for q in table.index:
            # calculate primary beam sensitivity for each target based on cosine attenuation pattern
            point_1 = table['ra'][q], table['decl'][q]
            point_2 = float(coords.split(", ")[0]), float(coords.split(", ")[1])
            gcd = great_circle_distance(point_1, point_2)
            beam_fwhm = np.rad2deg((con.c / float(frequency)) / 13.5)
            proportional_offset = gcd / beam_fwhm
            primary_sensitivity = cosine_attenuation(proportional_offset)
            # One target of priority n is worth priority_decay targets of priority n+1.

            priority_decay = get_redis_key(self.redis_server, "product_id:variables:priority_decay")
            primary_sensitivity_exponent = \
                get_redis_key(self.redis_server, "product_id:variables:primary_sensitivity_exponent")

            tbdfm_param[q] = int((primary_sensitivity ** primary_sensitivity_exponent())
                                 * priority_decay() ** (7 - table['priority'][q]))
        # append this array to the target list dataframe
        table['tbdfm_param'] = tbdfm_param
        return table

    def abort_criteria(self, product_id, time_elapsed=None, observation_time=None, fraction_processed=None):
        """Function to abort processing if certain conditions are met

        Parameters:
            product_id: (str)
                Subarray ID received from redis message
            time_elapsed: (str)
                Total time elapsed since start of processing
            observation_time: (str)
                Total recorded observation time from the processing nodes (t_obs)
            fraction_processed: (str)
                Fraction of sources successfully processed from the currently processing block

        Returns:
            None
        """
        if (not fraction_processed) and (not observation_time):
            # processing aborted based on priority of new sources & optimising TBDFM values
            # (larger TBDFM = better)
            logger.info("New pointing contains sources with a higher total TBDFM parameter. Aborting")
            # self._deconfigure(product_id)
            pStatus.proc_status = "ready"
            logger.info(
                "-------------------------------------------------------------------------------------------------")
            logger.info("Processing state set to \'ready\'")

        elif not fraction_processed:  # processing aborted based on observation time (t_obs)
            if (time_elapsed > 1200) and (time_elapsed > (2 * observation_time) - 300):
                logger.info("Processing time has exceeded both 20 and (2t_obs - 5) minutes."
                            " Aborting")
                self._deconfigure(product_id)
                pStatus.proc_status = "ready"
                logger.info(
                    "-------------------------------------------------------------------------------------------------")
                logger.info("Processing state set to \'ready\'")

        elif not observation_time:  # processing aborted based on absolute processing time
            if (fraction_processed > 0.9) and (time_elapsed > 600):
                logger.info("Processing time has exceeded 10 minutes, with >90% of targets processed successfully."
                            " Aborting")
                self._deconfigure(product_id)
                pStatus.proc_status = "ready"
                logger.info(
                    "-------------------------------------------------------------------------------------------------")
                logger.info("Processing state set to \'ready\'")

    def reformat_table(self, table):
        """Function to reformat the table of targets pushed to the backend

        Parameters:
            table: (str)
                A pandas DataFrame containing target list information, parsed as a string

        Returns:
            targets_to_process: (pandas.DataFrame)
                Reformatted pandas DataFrame containing information from the given table
        """
        replace_chars = ("\"", ""), (":", ","), ("[", ""), ("], ", "\n"), ("]", ""), \
                        ("{", ""), ("}", "")
        formatted = reduce(lambda a, kv: a.replace(*kv), replace_chars, table)
        data = StringIO(formatted)
        df = pd.read_csv(data, header=None, index_col=0, float_precision='round_trip')
        targets_to_process = df.transpose()
        return targets_to_process

    def coord_error(self, product_id, coords, frequency, mode):
        """Function to handle errors due to coordinate values (empty pointings or out of range)

        Parameters:
            product_id: (str)
                subarray ID
            coords: (str)
                string to parse containing erroneous coordinates
            frequency: (str)
                the central frequency of observation
            mode: (str)
                current_obs or new_obs, the keys concerning erroneous or unavailable coordinates, to be deleted

        Returns:
            None
        """
        arr_ra_dec = coords.split(', ')
        dec_coord = arr_ra_dec[1]
        formatted_freq = self.engine.freq_format(frequency)
        logger.info('No targets visible for coordinates ({}) at {}. Waiting for new coordinates'
                    .format(coords, formatted_freq))

        # DELETE mode:target_list redis key
        target_key = ('{}:{}:target_list'.format(product_id, mode))
        if "None" not in str(self._get_sensor_value(product_id, "{}:target_list".format(mode))):
            delete_key(self.redis_server, target_key)
        pStatus.proc_status = "ready"

    def load_schedule_block(self, message):
        """Reformats schedule block messages and reformats them into dictionary format

        Parameters:
            message: (str)
                asdf

        Returns:
            None
        """

        message = message.replace('"[', '[')
        message = message.replace(']"', ']')
        return yaml.safe_load(message)

    def _get_sensor_value(self, product_id, sensor_name):
        """Returns the value for a given sensor and product id number

        Parameters:
            product_id: (str)
                ID received from redis message
            sensor_name: (str)
                Name of the sensor to be queried

        Returns:
            value: (str, int)
                Value attached to the key in the redis database
        """

        key = '{}:{}'.format(product_id, sensor_name)
        value = get_redis_key(self.redis_server, key)
        return value

    def _message_to_func(self, channel, action):
        """Function that selects a function to run based on the channel entered

        Parameters:
            channel: (str)
                channel/sensor name

        Returns:
            Function attached to a particular sensor_name

        """

        return action.get(channel, self._other)

    def _other(self, channel):
        """Function that handles unrecognized requests from redis server

        Parameters:
            channel: (dict)
                channel over which message is passed

        Returns:
            None
        """
        logger.info('Unrecognized channel style: {}'.format(channel))

    def _status_update(self, msg):
        """Function to test the status_update from the processing nodes.

        Parameters:
            msg: (str)
                string formatted like a

        Returns:
            None
        """
        status_msg = self.load_schedule_block(msg)
        if status_msg['success']:
            self.engine.update_obs_status(**status_msg)

    def _unsubscribe(self, channels=None):
        """Unsubscribe from the redis server

        Parameters:
            channels: (str, list)
                List of channels you wish to unsubscribe to

        Returns:
            None
        """

        if channels is None:
            self.p.unsubscribe()
        else:
            self.p.unsubscribe(channels)

        logger.info('Unsubscribed from channel(s)')

    def store_metadata(self, product_id, mode):
        """Stores observation metadata in database.

        Parameters:
            product_id: (str)
                Product ID of the subarray from which status metadata is pulled to add to the table of previously
                 completed observations
            mode: (str)
                Either new_sample or next_64; if mode=new_sample, the observation end time is set, else the previously
                stored value is used
        Returns:
            None
        """

        if "None" not in str(self._get_sensor_value(product_id, "current_obs:target_list")):
            if (pStatus.proc_status == "ready") or (mode == "next_64"):

                current_freq = self._get_sensor_value(product_id, "current_obs:frequency")
                bands = current_freq
                # TODO: ask Daniel/Dave about unique file-id
                file_id = 'filler_file_id'

                # TODO: Change this to handle specific pointing in subarray
                targets = pd.DataFrame.from_dict(
                    json.loads(self._get_sensor_value(product_id, "current_obs:target_list")))
                duration = self._get_sensor_value(product_id, "current_obs:duration")

                coords = self._get_sensor_value(product_id, "current_obs:coords")
                processed = "1"
                self.engine.add_sources_to_db(targets, coords, duration, file_id, bands, processed)

    def _beam_radius(self, current_freq, dish_size=25):
        """Returns the beam radius based on the frequency band used in the
           observation

       Parameters:
            current_freq: (float)
                Central frequency for the current observation in Hertz
        Returns:
            beam_rad: (float)
                Radius of the beam in radians
        """

        # TODO: change this to the real name
        beam_rad = 0.5 * (con.c / float(current_freq)) / dish_size
        return beam_rad

    def _publish_targets(self, targets, product_id, columns=None,
                         sub_arr_id=0, sensor_name='targets'):
        """Reformat the table returned from target searching

        Parameters:
            targets: (dict)
                dict containing data for the triaged target list to be processed
            product_id: (str)
                product_id for the given subarray
            sub_arr_id: (int)
                ASDF pointing number for the given schedule block
            sensor_name: (str)
                name of the sensor to be queried; in this case the targets sensor
            columns: (list)
                list of columns for the published list of targets to have
            channel: (str)
                channel over which to publish the targets

        Returns:
            None
        """

        # fetch list of beamforming coordinates
        str_ra = [str(i) for i in targets['ra']]
        str_decl = [str(i) for i in targets['decl']]

        key = '{}:pointing_{}:{}'.format(product_id, sub_arr_id, sensor_name)
        channel = "bluse:///set"
        # write tables to redis
        write_pair_redis(self.redis_server, key, json.dumps(targets))

        pd.DataFrame.to_csv(pd.DataFrame.from_dict(targets), "targets.csv")
        logger.info('{} beamforming coordinates published to {}'
                    .format(len(targets['source_id']), channel))
        publish(self.redis_server, channel, key)

    def _parse_sensor_name(self, message):
        """Parse channel name sent over redis channel

        Parameters:
            message: (str)
                Message received over the ASDF channel

        Returns:
            product_id: (str)
                product_id of the given subarray
            sensor: (str)
                Name of the particular sensor
        """

        try:
            if len(message.split(':')) == 3:
                product_id, sensor, _ = message.split(':')

            elif len(message.split(':')) == 2:
                product_id, sensor = message.split(':')

            else:
                temp = message.split(", ", maxsplit=1)  # TRYING TO FIGURE OUT ERROR
                temp[0] = temp[0].split(":")  # TRYING TO FIGURE OUT ERROR
                message = temp[0] + [temp[1]]  # TRYING TO FIGURE OUT ERROR

                product_id = message[0]
                sensor = message[1]

                # product_id, sensor = message.split(':', 2)[:2]

            return product_id, sensor

        except:
            logger.warning('Parsing sensor name failed. Unrecognized message '
                           'style: {}'.format(message))
            return False

    def _found_aliens(self):
        """You found aliens! Alerting slack

        Parameters:
            None

        Returns:
            None
        """
        try:
            from .slack_tools import notify_slack
        except ImportError:
            from slack_tools import notify_slack

        notify_slack("Congratulations! You found aliens!")
