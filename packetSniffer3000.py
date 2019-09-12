# Matthew Phelan and Trent Woods
# CS 371
# packetSniffer3000
from scapy.all import *
from scapy.layers.inet import IP, UDP, TCP, ICMP
import pandas as pd
import numpy as np
import sys
import socket
import os
# used to compare packets for flow detection
previous_flow = tuple()
# id for flow of packets
flow_id = 0
# number of packets
packet_count = 1
# features (some may be added or dropped)
lenSum = 0
ttlSum = 0
windowSum = 0
protocol = 0
init_time = 0
elap_time = 0
tuple_count = 0
# Labels for machine learning
# 1.Web browsing
# 2.Video streaming (e.g., YouTube)
# 3.Video conference (e.g., Skype)
# 4.File download 
# just change label when you are testing one of the above
print('Enter a label 1-4 for what type of traffic data you are acquiring: ')
label = input()
# callback for packet sniffing
def fields_extraction(x):
    # assign current flow tuple (wrapped in try/except because some parameters may be missing at times)
    try:
        flow = (x.src, x.sport, x.dst, x.dport, x.proto)
    except:
        flow = (0, 0, 0, 0, 0)
    # variable to keep track of previous flow tuple
    global previous_flow
    # flow ID for each set of flows
    global flow_id
    # for features (mostly averages):
    global packet_count
    global lenSum
    global ttlSum
    global windowSum
    global label
    global protocol
    global init_time
    global tuple_count
    # bidirectional flow (can find averages, whatever at end of flow while keeping track of bidirectionally equiv. packets)
    if previous_flow != () and flow != ():
        if flow == previous_flow or (flow[2] == previous_flow[0] and flow[3] == previous_flow[1]
                                     and flow[0] == previous_flow[2] and flow[1] == previous_flow[3] and flow[4] ==
                                     previous_flow[4]) and tuple_count < 200:
            if packet_count == 1:
                init_time = x.time
            packet_count += 1  # add to number of packets
            try:
                lenSum += x.len  # sum lengths for average
                ttlSum += x.ttl  # sum time to live for average
                protocol = x.proto
            except:
                lenSum += 0
                ttlSum += 0
        elif packet_count >= 3:  # filter for flows that have 3 or more packets
            try:
                tuple_count += 1
                flow_id += 1
                packet_count = 1
                elap_time = x.time - init_time
                row = str(x.sport) + ',' + str(x.dport) + ',' + str(protocol) + ',' + str(flow_id) + ',' + str(packet_count) + ',' + str(
                    lenSum / packet_count) + ',' + str(ttlSum / packet_count) + ',' + str(elap_time) + ',' + str(label) + '\n'
                # append row with data to csv file
                with open("data.csv", 'a') as fd:
                    fd.write(row)

                if packet_count != 0:
                    print(x.sport, x.dport, protocol, flow_id, packet_count, lenSum / packet_count, ttlSum / packet_count, elap_time, label)
            except:
                tuple_count += 0

    # change current flow to previous for comparison in next loop through
    previous_flow = flow

pkts = sniff(prn=fields_extraction, count=30000)

sys.exit()
