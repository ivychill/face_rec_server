# Task sink
# Binds PULL socket to tcp://localhost:5558
# Collects results from workers via that socket
#
# Author: Lev Givon <lev(at)columbia(dot)edu>

import time
import zmq
import sys,signal
import threading
import os
import msgpack
import msgpack_numpy as m
sys.path.append('../')
import logging
import copy
import json

class TaskSink(threading.Thread):
    def __init__(self,recv_addr,sendqueue):
        threading.Thread.__init__(self)
        self.context = zmq.Context()
        # Socket to receive messages on
        self.receiver = self.context.socket(zmq.PULL)
        self.recv_addr = recv_addr
        self.poller = zmq.Poller()
        self.publisher = self.context.socket(zmq.PUB)
        self.sendqueue = sendqueue
        self.thread_state = True



    def stop(self):
        print("TaskSink thread stop!")
        self.thread_state = False
        self.receiver.close()
        self.publisher.close()
        self.context.term()

    def run(self):
        self.receiver.bind(self.recv_addr)
        print("sink bind {} for recv face feature!".format(self.recv_addr))
        logging.debug("sink bind {} for recv face feature!".format(self.recv_addr))
        self.poller.register(self.receiver, zmq.POLLIN)
        while self.thread_state:
            try:
            #if True:
                socks = dict(self.poller.poll())
                if self.receiver in socks and socks[self.receiver] == zmq.POLLIN:
                    message = self.receiver.recv()
                    #print('sink RECV MESSAGE:', len(message))
                    message = msgpack.unpackb(message, object_hook=m.decode)
                    self.sendqueue.put(message)

            except Exception as e:
                logging.error(e)
                print e



def sigint_handler(signum, frame):
    print("main-thread exit")
    global taskSink
    taskSink.stop()
    time.sleep(2)
    sys.exit()




if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    taskSink = TaskSink("tcp://127.0.0.1:12301")
    taskSink.start()
    taskSink.join()