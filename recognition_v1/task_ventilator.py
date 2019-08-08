import time
import zmq
import sys, signal
import threading
import os
import msgpack
import msgpack_numpy as m
import logging
from multiprocessing import Queue
import json
class TaskVentilator(threading.Thread):
    def __init__(self, msgqueue,ventilator_addr):
        threading.Thread.__init__(self)

        self.context = zmq.Context()
        self.msgqueue = msgqueue
        self.ventilator_addr = ventilator_addr

        self.thread_state = True



    def stop(self):
        print("ObjectTracker thread stop!")
        self.thread_state = False
        self.receiver.close()
        self.context.term()

    def run(self):

        self.sender = self.context.socket(zmq.PUSH)
        self.sender.bind(self.ventilator_addr)
        print("ventilator bind {} for send  facenet msg!".format(self.ventilator_addr))
        logging.debug("ventilator bind {} for send facenet msg!".format(self.ventilator_addr))

        while self.thread_state:
            try:
                vsendmsg = self.msgqueue.get()
                serialized = msgpack.packb(vsendmsg, default=m.encode)
                self.sender.send(serialized)
            except Exception as e:
               print 'object tracker error:',e
               logging.error(e)
        self.stop()