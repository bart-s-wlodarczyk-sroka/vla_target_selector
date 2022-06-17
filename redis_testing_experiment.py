import re
import json
import yaml
import time
import threading
import redis
import csv
import itertools

from vla_target_selector.logger import log as logger
from vla_target_selector.vla_db import Triage
from vla_target_selector.redis_tools import (publish,
                                             get_redis_key,
                                             write_pair_redis,
                                             connect_to_redis,
                                             delete_key)


vla_pointings = "vla_nonvlass_subset.csv"


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
        row_index = 1  # start at first entry
        print("Row {} / {} ({} %)".format(row_index, self.lines, round(((row_index / float(self.lines)) * 100), 3)))
        row = self.get_csv_line(vla_pointings, row_index)
        ra = row[0]
        print("RA: {}".format(ra))
        dec = row[1]
        print("DEC: {}".format(dec))
        duration = row[2]
        print("Duration: {} s".format(duration))
        frequency = row[5]
        print("Frequency: {} Hz".format(frequency))
        print("----------------------------------------")

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

        publish_key('sensor_alerts', 'array_1:target:radec',
                    "{}, {}".format(ra, dec))

        publish_key('sensor_alerts', 'array_1:subarray_1_streams_wide_antenna_channelised_voltage_centre_frequency',
                    frequency)

        publish_key('sensor_alerts', 'array_1:duration',
                    duration)

        row_index += 1

        for item in self.p.listen():
            print("Row {} / {} ({} %)".format(row_index, self.lines, round(((row_index / float(self.lines)) * 100), 3)))
            row = self.get_csv_line(vla_pointings, row_index)
            ra = row[0]
            print("RA: {}".format(ra))
            dec = row[1]
            print("DEC: {}".format(dec))
            duration = row[2]
            print("Duration: {} s".format(duration))
            frequency = row[5]
            print("Frequency: {} Hz".format(frequency))
            print("----------------------------------------")

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

            publish_key('sensor_alerts', 'array_1:target:radec',
                        "{}, {}".format(ra, dec))

            publish_key('sensor_alerts', 'array_1:subarray_1_streams_wide_antenna_channelised_voltage_centre_frequency',
                        frequency)

            publish_key('sensor_alerts', 'array_1:duration',
                        duration)

            row_index += 1
