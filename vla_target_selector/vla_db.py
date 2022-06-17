from dateutil import parser
from datetime import datetime
from sqlalchemy import create_engine, delete
from sqlalchemy.engine.url import URL
import yaml
import math
import numpy as np
import pandas as pd

try:
    from .logger import log as logger
    from .redis_tools import get_redis_key

except ImportError:
    from logger import log as logger
    from redis_tools import get_redis_key


class DatabaseHandler(object):
    """
    Class to handle the connection to the source database as well as querying
    the database for astronomical sources within the field of view.

    Examples:
        # db = DatabaseHandler()
        # db.select_targets(c_ra, c_dec, beam_rad)
    """

    def __init__(self, config_file):
        """
        __init__ function for the DatabaseHandler class

        Parameters:
            config_file: (str)
                asdf

        Returns:
            None
        """
        self.cfg = self.configure_settings(config_file)
        self.conn = self.connect_to_db(self.cfg['mysql'])

    def configure_settings(self, config_file):
        """Sets configuration settings

        Parameters:
            config_file: (str)
                Name of the yaml configuration file to be opened

        Returns:
            cfg: (dict)
                Dictionary containing the values of the configuration file
        """
        try:
            with open(config_file, 'r') as f:
                try:
                    cfg = yaml.safe_load(f)
                    return cfg
                except yaml.YAMLError as E:
                    logger.error(E)
        except IOError:
            logger.error('Config file not found')

    def connect_to_db(self, cred):
        """
        Connects to the Breakthrough Listen database

        Parameters:
            cred: (dict)
                Dictionary containing information on the source list database

        Returns:
            conn : sqlalchemy connection
                SQLalchemy connection to the database containing sources for
                triaging
        """
        url = URL(**cred)
        # self.engine = create_engine(name_or_url = url)
        self.engine = create_engine(url)
        return self.engine.connect()

    def close_conn(self):
        """Close the connection to the database

        Parameters:
            None

        Returns:
            None
        """
        self.conn.close()
        self.engine.dispose()


class Triage(DatabaseHandler):
    """

    ADD IN DOCUMENTATION

    Examples:
        # conn = Triage()
        # conn.select_targets(ra, dec, beam_rad)

    When start() is called, a loop is started that subscribes to the "alerts" and
    "sensor_alerts" channels on the Redis server. Depending on the which message
    that passes over which channel, various processes are run:
    """

    def __init__(self, config_file):
        super(Triage, self).__init__(config_file)

    def add_sources_to_db(self, df, coords, duration, file_id, bands, processed="1",
                          table='observation_status'):
        """
        Adds a pandas DataFrame to a specified table

        Parameters:
            df: (pandas.DataFrame)
                DataFrame containing information on the sources within the field
                of views
            coords: (str)
                Boresight coordinates for the primary beam
            file_id: (str)
                unique identifier for the file containing observation data ASDF
            bands: (str)
                frequency band of the observation
            table: (str)
                name of the observation metadata table to add sources to

        Returns
            bool:
                If sources were successfully added to the database, returns True.
                Else, returns False.
        """

        source_ids = []

        c_ra = float(coords.split(", ")[0])
        c_dec = float(coords.split(", ")[1])

        for n in df.index:
            indiv_ids = df['source_id'][n].split(", ")
            for item in indiv_ids:
                source_ids.append(item)

        source_tb = pd.DataFrame(
            {
                'source_id': source_ids,
             })

        source_tb['ra'] = c_ra
        source_tb['dec'] = c_dec
        source_tb['duration'] = duration
        source_tb['file_id'] = file_id
        source_tb['bands'] = bands
        source_tb['processed'] = processed

        try:
            source_tb.to_sql(table, self.conn, if_exists='append', index=False)
            return True

        except Exception as e:
            logger.info(e)
            logger.warning('Was not able to add sources to the database!')
            return False

    def _box_filter(self, c_ra, c_dec, beam_rad, table, cols, current_freq):
        """Returns a string which acts as a pre-filter for the more computationally
        intensive search

        Reference:
            http://janmatuschek.de/LatitudeLongitudeBoundingCoordinates

        Parameters:
            c_ra, c_dec: (float)
                Pointing coordinates of the telescope in radians
            beam_rad: (float)
                Angular radius of the primary beam in radians
            table: (str)
                Table within database where
            cols: (list)
                Columns to select within the table
            current_freq: (str)
                Current central frequency of observation in Hz
        Returns:
            query: (str)
                SQL query string

        """

        beam_rad_arcmin = beam_rad * (180 / math.pi) * 60

        logger.info("Primary beam radius at {}: {} radians = {} arc minutes"
                    .format(self.freq_format(current_freq), beam_rad, beam_rad_arcmin))

        if c_dec - beam_rad <= - np.pi / 2.0:
            ra_min, ra_max = 0.0, 2.0 * np.pi
            dec_min = -np.pi / 2.0
            dec_max = c_dec + beam_rad

        elif c_dec + beam_rad >= np.pi / 2.0:
            ra_min, ra_max = 0.0, 2.0 * np.pi
            dec_min = c_dec - beam_rad
            dec_max = np.pi / 2.0

        else:
            ra_offset = np.arcsin(np.sin(beam_rad) / np.cos(c_dec))
            ra_min = c_ra - ra_offset
            ra_max = c_ra + ra_offset
            dec_min = c_dec - beam_rad
            dec_max = c_dec + beam_rad

        bounds = np.rad2deg([ra_min, ra_max, dec_min, dec_max])

        if bounds[1] >= 360:
            ra_360 = bounds[1] - 360
            ra_str = "(({} < ra AND ra < 360) OR (0 < ra AND ra < {}))".format(bounds[0], ra_360)
        elif bounds[0] <= 0:
            ra_360 = bounds[0] + 360
            ra_str = "((0 < ra AND ra < {}) OR ({} < ra AND ra < 360))".format(bounds[1], ra_360)
        else:
            ra_str = "({} < ra  AND ra < {})".format(bounds[0], bounds[1])

        query = """
                SELECT {cols}
                FROM exotica_list
                UNION ALL
                SELECT {cols}
                FROM adhoc_list
                UNION ALL
                SELECT {cols}
                FROM {table}
                WHERE {ra_str} AND
                      ({dec_min} < decl AND decl < {dec_max})
                """.format(cols=', '.join(cols), table=table,
                           ra_str=ra_str, dec_min=bounds[2], dec_max=bounds[3])
        return query

    def triage(self, tb, current_freq, table='observation_status'):
        """
        Returns an array of priority values (or maybe the table with priority values
        appended)

        Parameters:
            tb: (pandas.DataFrame)
                table containing sources within the field of view of MeerKAT's pointing
            current_freq: (str)
                Current central frequency of observation in Hz
            table: (str)
                Name of the MySQL table of previous observations to be used for triaging

        Returns:
            tb: (pandas.DataFrame)
                table containing the sources to be beamformed on
        """

        # initially, all sources assigned a priority of 2
        priority = np.full(tb.shape[0], 2, dtype=int)
        # priority = np.random.randint(1, 7, size=tb.shape[0])

        query = """
                SELECT source_id, processed
                FROM {}
                GROUP BY source_id, processed
                """.format(table)

        # TODO replace these with sqlalchemy queries

        # list of previous observations
        prev_obs = pd.read_sql(query, con=self.conn)
        # logger.info("Previous observations:\n{}\n".format(prev_obs))
        # prev_obs.to_csv('prev_obs.csv')
        successfully_processed = \
            prev_obs.astype({'source_id': 'str'}).loc[prev_obs['processed'].isin(['1'])]

        # # exotica sources
        # priority[tb['table_name'].str.contains('exotica')] = 3

        # sources previously observed & successfully processed
        priority[tb['source_id'].isin(successfully_processed['source_id'])] = 6

        # # sources previously successfully processed with their frequency bands
        # prev_freq = successfully_processed\
        #     .drop(['duration'], axis=1)\
        #     .groupby('source_id')\
        #     .agg(lambda x: ', '.join(x.values))
        # # sources previously successfully processed with their maximum durations
        # longest_obs = successfully_processed.groupby('source_id')['duration'].max()
        #
        # for p in tb['source_id']:
        #     # sources previously observed, but at a different frequency
        #     try:
        #         if current_freq in prev_freq.loc[p]['bands']:
        #             pass
        #         else:
        #             priority[tb['source_id'] == p] = 5
        #     except KeyError:  # chosen source is not in prev_freq table
        #         pass
        #     except IndexError:  # prev_freq table is empty
        #         pass
        #     # sources previously observed, but for < 5 minutes
        #     try:
        #         if longest_obs[p] < 300:
        #             priority[tb['source_id'] == p] = 4
        #     except KeyError:  # chosen source is not in prev_obs table
        #         pass
        #     except IndexError:  # prev_obs table is empty
        #         pass
        #
        # # ad-hoc sources
        # priority[tb['table_name'].str.contains('adhoc')] = 1

        tb['priority'] = priority
        logger.info("\n\n{}\n".format(tb.sort_values(by=['priority', 'dist_c']).reset_index()))
        return tb.sort_values(by=['priority', 'dist_c']).reset_index()

    def select_targets(self, c_ra, c_dec, beam_rad, current_freq='Unknown', table='target_list',
                       cols=None):
        """Returns a string to query the 1 million star database to find sources
           within some primary beam area
        Parameters:
            c_ra : (float)
                Pointing coordinates of the telescope in radians (right ascension)
            c_dec : (float)
                Pointing coordinates of the telescope in radians (declination)
            beam_rad: (float)
                Angular radius of the primary beam in radians
            current_freq: (str)
                Current central frequency of observation in Hz
            table: (str)
                Name of the MySQL table that is being queried
            cols: (list)
                Columns of table to output
        Returns:
            target_list: (DataFrame)
                Returns a pandas DataFrame containing the objects meeting the filter
                criteria, sorted in order of priority
        """

        if not cols:
            cols = ['ra', 'decl', 'source_id', 'dist_c', 'table_name']

        mask = self._box_filter(c_ra, c_dec, beam_rad, table, cols, current_freq)

        query = """
                SELECT *
                FROM ({mask}) as T
                WHERE ACOS( SIN(RADIANS(decl)) * SIN({c_dec}) + COS(RADIANS(decl)) *
                COS({c_dec}) * COS({c_ra} - RADIANS(ra))) < {beam_rad}; \
                """.format(mask=mask, c_ra=c_ra,
                           c_dec=c_dec, beam_rad=beam_rad)

        logger.info("\n{}\n".format(query))

        # TODO: replace with sqlalchemy queries
        tb = pd.read_sql(query, con=self.conn)
        # logger.info("\n{}\n".format(tb))

        sorting_priority = self.triage(tb, current_freq)
        target_list = sorting_priority.sort_values(by=['priority', 'dist_c']).reset_index()
        # self.output_targets(target_list, c_ra, c_dec, current_freq)
        return target_list

    def freq_format(self, current_freq):
        """Function to format current frequency to either MHz or GHz for output

        Parameters:
            current_freq: (str)
                Current central frequency of observation in Hz
        Returns:
            freq_formatted: (str)
                Formatted central frequency of observation, in either MHz or GHz
        """

        if float(current_freq) > 1000000000:
            gigahertz = float(current_freq) * 10**-9
            freq_formatted = "{} GHz".format(gigahertz)
        else:
            megahertz = float(current_freq) * 10**-6
            freq_formatted = "{} MHz".format(megahertz)

        return freq_formatted
