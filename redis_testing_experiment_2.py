import re
import json
import yaml
import time
import threading
import redis
import csv
import itertools
import math

from vla_target_selector.logger import log as logger
from vla_target_selector.vla_db import Triage
from vla_target_selector.redis_tools import (publish,
                                             get_redis_key,
                                             write_pair_redis,
                                             connect_to_redis,
                                             delete_key)


class Listen(threading.Thread):
    """

    ADD IN DOCUMENTATION

    Examples:
        >>> client = Listen(['bluse:///set'])
        >>> client.start()

    When start() is called, a loop is started that subscribes to the "alerts" and
    "sensor_alerts" channels on the Redis server. Depending on the which message
    that passes over which channel, various processes are run.
    """

    # vla_pointings = '/Users/Bart/Downloads/Telegram Desktop/Allcoords_META_2014-11-25_2020-11-24.csv'
    vla_pointings = '/Users/Bart/Downloads/VLA stuff/with_antenna_configs.csv'
    file = open(vla_pointings)
    reader = csv.reader(file)
    lines = len(list(reader))

    def __init__(self, chan=None):

        if not chan:
            chan = ['bluse:///set']

        threading.Thread.__init__(self)

        # Initialize redis connection
        self.redis_server = connect_to_redis()

        # Subscribe to channel
        self.p = self.redis_server.pubsub(ignore_subscribe_messages=True)
        self.p.psubscribe(chan)

    def get_csv_line(self, path, line_number):
        with open(path) as f:
            return next(itertools.islice(csv.reader(f), line_number, None))

    def run(self):
        """Runs continuously to listen for messages that come in from specific
           redis channels. Main function that handles the processing of the
           messages that come through redis.
        """

        # Various parameter values
        priority_decay = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        min_local_attenuation = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        min_include_attenuation = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        primary_sensitivity_exponent = [1/4, 1/3, 1/2, 1, 2, 3, 4]

        # Iterate over parameter values
        for pd in priority_decay:
            for mla in min_local_attenuation:
                for mia in min_include_attenuation:
                    for pse in primary_sensitivity_exponent:
                        # Change values in target_selector_variables
                        write_pair_redis(self.redis_server, "product_id:variables:priority_decay", pd)
                        write_pair_redis(self.redis_server, "product_id:variables:min_local_attenuation", mla)
                        write_pair_redis(self.redis_server, "product_id:variables:min_include_attenuation", mia)
                        write_pair_redis(self.redis_server, "product_id:variables:primary_sensitivity_exponent", pse)
                        write_pair_redis(self.redis_server, "product_id:variables:number_beams", 64)

                        logger.info("Priority decay: {}".format(pd))
                        logger.info("Min. local attenuation: {}".format(mla))
                        logger.info("Min. included attenuation: {}".format(mia))
                        logger.info("Primary sensitivity exponent: {}".format(pse))

                        # Clear observation_status SQL table
                        # delete(observation_status)

                        # query = """
                        #         DELETE FROM breakthrough_db.observation_status
                        #         """

                        # tb = pd.read_sql(query, con=self.conn)

                        row_index = 1
                        while row_index <= self.lines:
                            print("Row {} / {} ({} %)"
                                  .format(row_index, self.lines,
                                          round(((row_index / float(self.lines)) * 100), 3)))
                            row = self.get_csv_line(self.vla_pointings, row_index)
                            # ra in deg
                            ra = float(row[0])
                            # dec in deg
                            dec = float(row[1])
                            # duration in s
                            duration = float(row[2])
                            mjdstart = float(row[4])
                            # frequency in Hz
                            frequency = float(row[10])
                            # tracking rates in rad/day
                            # 1 rad/day = (180/PI) deg/day = ((180/PI) / (24 * 60 * 60)) deg/s
                            ra_tracking_rate = float(row[17]) * ((180/math.pi) / (24 * 60 * 60))
                            dec_tracking_rate = float(row[18]) * ((180/math.pi) / (24 * 60 * 60))

                            with open('test/channels.txt', 'r') as f:
                                channel = f.read()
                            with open('test/messages.txt', 'r') as f:
                                messages = f.read()

                            chnls = channel.split('\n')
                            msgs = messages.split('\n')

                            def publish_key(chan, key, val):
                                r.set(key, val)
                                r.publish(chan, key)
                                return True

                            r = redis.StrictRedis()
                            if (ra_tracking_rate != 0) or (dec_tracking_rate != 0):
                                time_remaining = duration
                                time_step = 5  # seconds
                                part_chunk = duration % time_step
                                while time_remaining >= time_step:
                                    logger.info("RA: {}".format(ra))
                                    logger.info("Dec: {}".format(dec))
                                    logger.info("Total pointing duration: {}".format(duration))
                                    logger.info("Split pointing duration: {}".format(time_step))
                                    logger.info("MJD start: {}".format(mjdstart))
                                    logger.info("Frequency: {}".format(frequency))
                                    logger.info("RA tracking rate: {}".format(ra_tracking_rate))
                                    logger.info("Dec tracking rate: {}".format(dec_tracking_rate))
                                    ra_new = ra + (ra_tracking_rate * (duration - time_remaining))
                                    dec_new = dec + (dec_tracking_rate * (duration - time_remaining))
                                    # run target selector with time_step observation time
                                    publish_key('sensor_alerts', 'array_1:target:radec',
                                                "{}, {}".format(ra_new, dec_new))
                                    publish_key('sensor_alerts',
                                                'array_1:subarray_1_streams_wide_antenna_channelised_voltage_centre_frequency',
                                                frequency)
                                    publish_key('sensor_alerts', 'array_1:duration',
                                                time_step)
                                    # publish other relevant keys
                                    time_remaining = time_remaining - time_step
                                    for item in self.p.listen():
                                        while time_remaining >= time_step:
                                            logger.info("RA: {}".format(ra))
                                            logger.info("Dec: {}".format(dec))
                                            logger.info("Total pointing duration: {}".format(duration))
                                            logger.info("Time remaining: {}".format(time_remaining))
                                            logger.info("Split pointing duration: {}".format(time_step))
                                            logger.info("MJD start: {}".format(mjdstart))
                                            logger.info("Frequency: {}".format(frequency))
                                            logger.info("RA tracking rate: {}".format(ra_tracking_rate))
                                            logger.info("Dec tracking rate: {}".format(dec_tracking_rate))
                                            ra_new = ra + (ra_tracking_rate * (duration - time_remaining))
                                            dec_new = dec + (dec_tracking_rate * (duration - time_remaining))
                                            publish_key('sensor_alerts', 'array_1:target:radec',
                                                        "{}, {}".format(ra_new, dec_new))
                                            publish_key('sensor_alerts',
                                                        'array_1:subarray_1_streams_wide_antenna_channelised_voltage_centre_frequency',
                                                        frequency)
                                            publish_key('sensor_alerts', 'array_1:duration',
                                                        time_step)
                                            # publish other relevant keys
                                            time_remaining = time_remaining - time_step
                                        if time_remaining < time_step:
                                            logger.info("RA: {}".format(ra))
                                            logger.info("Dec: {}".format(dec))
                                            logger.info("Total pointing duration: {}".format(duration))
                                            logger.info("Time remaining: {}".format(time_remaining))
                                            logger.info("Split pointing duration: {}".format(time_remaining))
                                            logger.info("MJD start: {}".format(mjdstart))
                                            logger.info("Frequency: {}".format(frequency))
                                            logger.info("RA tracking rate: {}".format(ra_tracking_rate))
                                            logger.info("Dec tracking rate: {}".format(dec_tracking_rate))
                                            ra_new = ra + (ra_tracking_rate * (duration - time_remaining))
                                            dec_new = dec + (dec_tracking_rate * (duration - time_remaining))
                                            publish_key('sensor_alerts', 'array_1:target:radec',
                                                        "{}, {}".format(ra_new, dec_new))
                                            publish_key('sensor_alerts',
                                                        'array_1:subarray_1_streams_wide_antenna_channelised_voltage_centre_frequency',
                                                        frequency)
                                            publish_key('sensor_alerts', 'array_1:duration',
                                                        time_remaining)
                                            # publish other relevant keys
                                            break

                                for item in self.p.listen():
                                    ra_new = ra + (ra_tracking_rate * (duration - part_chunk))
                                    dec_new = dec + (dec_tracking_rate * (duration - part_chunk))
                                    # run target selector with part_chunk observation time
                                    publish_key('sensor_alerts', 'array_1:target:radec',
                                                "{}, {}".format(ra_new, dec_new))
                                    publish_key('sensor_alerts',
                                                'array_1:subarray_1_streams_wide_antenna_channelised_voltage_centre_frequency',
                                                frequency)
                                    publish_key('sensor_alerts', 'array_1:duration',
                                                duration)
                                    # publish other relevant keys
                                    break
                            elif duration > 300:
                                time_remaining = duration
                                part_chunk = duration % 300
                                while time_remaining >= 300:
                                    logger.info("RA: {}".format(ra))
                                    logger.info("Dec: {}".format(dec))
                                    logger.info("Total pointing duration: {}".format(duration))
                                    logger.info("Split pointing duration: {}".format(300))
                                    logger.info("MJD start: {}".format(mjdstart))
                                    logger.info("Frequency: {}".format(frequency))
                                    logger.info("RA tracking rate: {}".format(ra_tracking_rate))
                                    logger.info("Dec tracking rate: {}".format(dec_tracking_rate))
                                    # run target selector with 300 second observation time
                                    publish_key('sensor_alerts', 'array_1:target:radec',
                                                "{}, {}".format(ra, dec))
                                    publish_key('sensor_alerts',
                                                'array_1:subarray_1_streams_wide_antenna_channelised_voltage_centre_frequency',
                                                frequency)
                                    publish_key('sensor_alerts', 'array_1:duration', 300)
                                    # publish other relevant keys
                                    time_remaining = time_remaining - 300

                                    if time_remaining > part_chunk:
                                        for item in self.p.listen():
                                            logger.info("RA: {}".format(ra))
                                            logger.info("Dec: {}".format(dec))
                                            logger.info("Total pointing duration: {}".format(duration))
                                            logger.info("Split pointing duration: {}".format(300))
                                            logger.info("MJD start: {}".format(mjdstart))
                                            logger.info("Frequency: {}".format(frequency))
                                            logger.info("RA tracking rate: {}".format(ra_tracking_rate))
                                            logger.info("Dec tracking rate: {}".format(dec_tracking_rate))
                                            publish_key('sensor_alerts', 'array_1:target:radec',
                                                        "{}, {}".format(ra, dec))
                                            publish_key('sensor_alerts',
                                                        'array_1:subarray_1_streams_wide_antenna_channelised_voltage_centre_frequency',
                                                        frequency)
                                            publish_key('sensor_alerts', 'array_1:duration',
                                                        300)
                                            # publish other relevant keys
                                            time_remaining = time_remaining - 300
                                            # logger.info("Time remaining: {}".format(time_remaining))

                                if part_chunk != 0:
                                    for item in self.p.listen():
                                        logger.info("RA: {}".format(ra))
                                        logger.info("Dec: {}".format(dec))
                                        logger.info("Total pointing duration: {}".format(duration))
                                        logger.info("Split pointing duration: {}".format(part_chunk))
                                        logger.info("MJD start: {}".format(mjdstart))
                                        logger.info("Frequency: {}".format(frequency))
                                        logger.info("RA tracking rate: {}".format(ra_tracking_rate))
                                        logger.info("Dec tracking rate: {}".format(dec_tracking_rate))
                                        # run target selector with part_chunk observation time
                                        publish_key('sensor_alerts', 'array_1:target:radec',
                                                    "{}, {}".format(ra, dec))
                                        publish_key('sensor_alerts',
                                                    'array_1:subarray_1_streams_wide_antenna_channelised_voltage_centre_frequency',
                                                    frequency)
                                        publish_key('sensor_alerts', 'array_1:duration',
                                                    part_chunk)
                                        break
                            else:
                                logger.info("RA: {}".format(ra))
                                logger.info("Dec: {}".format(dec))
                                logger.info("Total pointing duration: {}".format(duration))
                                logger.info("Split pointing duration: {}".format(duration))
                                logger.info("MJD start: {}".format(mjdstart))
                                logger.info("Frequency: {}".format(frequency))
                                logger.info("RA tracking rate: {}".format(ra_tracking_rate))
                                logger.info("Dec tracking rate: {}".format(dec_tracking_rate))
                                # run target selector with duration observation time
                                publish_key('sensor_alerts', 'array_1:target:radec',
                                            "{}, {}".format(ra, dec))
                                publish_key('sensor_alerts',
                                            'array_1:subarray_1_streams_wide_antenna_channelised_voltage_centre_frequency',
                                            frequency)
                                publish_key('sensor_alerts', 'array_1:duration',
                                            duration)

                            for item in self.p.listen():
                                row_index += 1
                                break
                            # for item in self.p.listen():
                            #     while row_index <= self.lines:
                            #         print("Row {} / {} ({} %)"
                            #               .format(row_index, self.lines,
                            #                       round(((row_index / float(self.lines)) * 100), 3)))
                            #         row = self.get_csv_line(self.vla_pointings, row_index)
                            #         ra = float(row[0])
                            #         dec = float(row[1])
                            #         duration = float(row[2])
                            #         mjdstart = float(row[4])
                            #         frequency = float(row[10])
                            #         ra_tracking_rate = float(row[17])
                            #         dec_tracking_rate = float(row[18])
                            #
                            #         logger.info("RA: {}".format(ra))
                            #         logger.info("Dec: {}".format(dec))
                            #         logger.info("Duration: {}".format(duration))
                            #         logger.info("MJD start: {}".format(mjdstart))
                            #         logger.info("Frequency: {}".format(frequency))
                            #         logger.info("RA tracking rate: {}".format(ra_tracking_rate))
                            #         logger.info("Dec tracking rate: {}".format(dec_tracking_rate))
                            #
                            #         with open('test/channels.txt', 'r') as f:
                            #             channel = f.read()
                            #         with open('test/messages.txt', 'r') as f:
                            #             messages = f.read()
                            #
                            #         chnls = channel.split('\n')
                            #         msgs = messages.split('\n')
                            #
                            #         def publish_key(chan, key, val):
                            #             r.set(key, val)
                            #             r.publish(chan, key)
                            #             return True
                            #
                            #         r = redis.StrictRedis()
                            #         if (ra_tracking_rate != 0) or (dec_tracking_rate != 0):
                            #             time_remaining = duration
                            #             time_step = 5  # seconds
                            #             part_chunk = duration % time_step
                            #             while time_remaining >= time_step:
                            #                 ra_new = ra + (ra_tracking_rate * (duration - time_remaining))
                            #                 dec_new = dec + (dec_tracking_rate * (duration - time_remaining))
                            #                 # run target selector with time_step observation time
                            #                 publish_key('sensor_alerts', 'array_1:target:radec',
                            #                             "{}, {}".format(ra_new, dec_new))
                            #                 publish_key('sensor_alerts',
                            #                             'array_1:subarray_1_streams_wide_antenna_channelised_voltage_centre_frequency',
                            #                             frequency)
                            #                 publish_key('sensor_alerts', 'array_1:duration',
                            #                             duration)
                            #                 # publish other relevant keys
                            #                 time_remaining = time_remaining - time_step
                            #                 for item in self.p.listen():
                            #                     ra_new = ra + (ra_tracking_rate * (duration - time_remaining))
                            #                     dec_new = dec + (dec_tracking_rate * (duration - time_remaining))
                            #                     publish_key('sensor_alerts', 'array_1:target:radec',
                            #                                 "{}, {}".format(ra_new, dec_new))
                            #                     publish_key('sensor_alerts',
                            #                                 'array_1:subarray_1_streams_wide_antenna_channelised_voltage_centre_frequency',
                            #                                 frequency)
                            #                     publish_key('sensor_alerts', 'array_1:duration',
                            #                                 duration)
                            #                     # publish other relevant keys
                            #             ra_new = ra + (ra_tracking_rate * (duration - part_chunk))
                            #             dec_new = dec + (dec_tracking_rate * (duration - part_chunk))
                            #             # run target selector with part_chunk observation time
                            #             publish_key('sensor_alerts', 'array_1:target:radec',
                            #                         "{}, {}".format(ra_new, dec_new))
                            #             publish_key('sensor_alerts',
                            #                         'array_1:subarray_1_streams_wide_antenna_channelised_voltage_centre_frequency',
                            #                         frequency)
                            #             publish_key('sensor_alerts', 'array_1:duration',
                            #                         duration)
                            #             # publish other relevant keys
                            #         elif duration > 300:
                            #             time_remaining = duration
                            #             part_chunk = duration % 300
                            #             while time_remaining >= 300:
                            #                 # run target selector with 300 second observation time
                            #                 publish_key('sensor_alerts', 'array_1:target:radec',
                            #                             "{}, {}".format(ra, dec))
                            #                 publish_key('sensor_alerts',
                            #                             'array_1:subarray_1_streams_wide_antenna_channelised_voltage_centre_frequency',
                            #                             frequency)
                            #                 publish_key('sensor_alerts', 'array_1:duration',
                            #                             duration)
                            #                 # publish other relevant keys
                            #                 time_remaining = time_remaining - 300
                            #                 for item in self.p.listen():
                            #                     publish_key('sensor_alerts', 'array_1:target:radec',
                            #                                 "{}, {}".format(ra, dec))
                            #                     publish_key('sensor_alerts',
                            #                                 'array_1:subarray_1_streams_wide_antenna_channelised_voltage_centre_frequency',
                            #                                 frequency)
                            #                     publish_key('sensor_alerts', 'array_1:duration',
                            #                                 duration)
                            #                     # publish other relevant keys
                            #                     time_remaining = time_remaining - 300
                            #             # run target selector with part_chunk observation time
                            #             publish_key('sensor_alerts', 'array_1:target:radec',
                            #                         "{}, {}".format(ra, dec))
                            #             publish_key('sensor_alerts',
                            #                         'array_1:subarray_1_streams_wide_antenna_channelised_voltage_centre_frequency',
                            #                         frequency)
                            #             publish_key('sensor_alerts', 'array_1:duration',
                            #                         duration)
                            #         else:
                            #             # run target selector with duration observation time
                            #             publish_key('sensor_alerts', 'array_1:target:radec',
                            #                         "{}, {}".format(ra, dec))
                            #             publish_key('sensor_alerts',
                            #                         'array_1:subarray_1_streams_wide_antenna_channelised_voltage_centre_frequency',
                            #                         frequency)
                            #             publish_key('sensor_alerts', 'array_1:duration',
                            #                         duration)
                        # Format filename
                        name_string = '{}_pd_{}_mla_{}_mia_{}_pse'.format(pd, mla, mia, pse)
                        # Output SQL table to file
                        pass
