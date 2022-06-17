from io import StringIO
import json
import redis
import csv
import numpy as np
import pandas as pd
import yaml
import time
import re

vla_pointings = "vla_117k.csv"

with open(vla_pointings, "r") as csvfile:
    datareader = csv.reader(csvfile)
    next(datareader)
    for row in datareader:
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
        time.sleep(5)
