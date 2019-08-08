import time
import zmq
import sys, signal
import threading
import os
import msgpack
import msgpack_numpy as m
import logging
sys.path.append('../')
import json
class ServerRecv(threading.Thread):
    def __init__(self, recv_addr,msgqueue):
        threading.Thread.__init__(self)
        self.poller = zmq.Poller()
        self.context = zmq.Context()
        self.recv_addr = recv_addr
        self.msgqueue = msgqueue
        self.thread_state = True



    def stop(self):
        print("ObjectTracker thread stop!")
        self.thread_state = False
        self.receiver.close()
        self.context.term()

    def run(self):
        # Socket to receive messages on
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.bind(self.recv_addr)
        print("ObjectTracker bind {} for recv yolo msg!".format(self.recv_addr))
        logging.debug("ObjectTracker bind {} for recv yolo msg!".format(self.recv_addr))
        self.poller.register(self.receiver, zmq.POLLIN)

        while self.thread_state:
            try:
            #if True:
                socks = dict(self.poller.poll())
                if self.receiver in socks and socks[self.receiver] == zmq.POLLIN:
                    message = self.receiver.recv()
                    message = msgpack.unpackb(message, object_hook=m.decode)
                    # Do the work
                    #vsendmsg=[vid,image]
                    self.msgqueue.put(message)
            except Exception as e:
               print 'object tracker error:',e
               logging.error(e)
        self.stop()