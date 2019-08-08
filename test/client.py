# coding=utf-8
'''
Created on 2015-10-13
发起请求
@author: kwsy2015
'''
import zmq
import random
import time
import sys,signal
import threading
from multiprocessing import Queue
from scipy import misc
import numpy as np
from sklearn.datasets import load_files
import msgpack
import msgpack_numpy as m
#  Prepare our context and sockets
import random

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image

def load_data(image_paths):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, 160, 160, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        img = crop(img, True, 160)

        images[i,:,:,:] = img
    return images


import os

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

class Client(threading.Thread):
    def __init__(self,recv_addr,facedir):
        threading.Thread.__init__(self)

        self.context = zmq.Context()
        # Socket to receive messages on
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(recv_addr)

        self.image_paths = get_image_paths(facedir)

        self.thread_state = True

    def stop(self):
        print("TaskSink thread stop!")
        self.thread_state = False
        self.socket.close()
        self.context.term()


    def run(self):
        while self.thread_state:
            nrof_samples = len(self.image_paths)
            no = random.randint(0,nrof_samples);
            img = misc.imread(self.image_paths[no])
            #img = crop(img, True, 160)
            message = [no,img]
            try:
                serialized = msgpack.packb(message, default=m.encode)
                self.socket.send(serialized)
                message = self.socket.recv()
                message = msgpack.unpackb(message, object_hook=m.decode)
                print(message)
            except Exception as e:
                print e
            time.sleep(5)
        self.stop()




def sigint_handler(signum, frame):
    print("main-thread exit")
    global taskSink
    taskSink.stop()
    time.sleep(2)
    sys.exit()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    #facedir = '/data/yanhong.jia/datasets/ID_out'
    facedir = '/data/333'
    client = Client("tcp://192.168.10.124:12111",facedir)
    client.start()
    client.join()
