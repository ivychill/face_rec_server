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

class ServerSend(threading.Thread):
    def __init__(self,pub_mix_addr,sendqueue):
        threading.Thread.__init__(self)
        self.context = zmq.Context()

        self.sendqueue = sendqueue

        self.publisher = self.context.socket(zmq.PUB)
        self.pub_mix_addr = pub_mix_addr
        self.thread_state = True



    def stop(self):
        print("TaskSink thread stop!")
        self.thread_state = False
        self.receiver.close()
        self.publisher.close()
        self.context.term()


    def run(self):

        self.publisher.bind(self.pub_mix_addr)
        print("sink bind {} for send track face feature!".format(self.pub_mix_addr))
        logging.debug("sink bind {} for send track face feature!".format(self.pub_mix_addr))
        while self.thread_state:
            try:
                message = self.sendqueue.get()
                self.publisher.send_json(message)
                vid = message[0]
                print(' send video {} MESSAGE====>mix'.format(vid))
                logging.debug(' send video {} MESSAGE====>mix'.format(vid))
                #logging.debug('sink send MESSAGE====>mix')

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