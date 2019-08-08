# coding=utf-8
'''
Created on 2015-10-13
发起请求
@author: kwsy2015
'''
from HashIdentifierAutoTrain import *
import zmq
import random
import time
import sys,signal
import threading
from multiprocessing import Queue
from scipy import misc
import numpy as np
import msgpack
import msgpack_numpy as m
#  Prepare our context and sockets
import random
import os,subprocess
import sys
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(path,'..'))
from recognition import facenet

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

class Client(threading.Thread):
    def __init__(self,addr,facedir,npydir):
        threading.Thread.__init__(self)
        self.context = zmq.Context()
        # Socket to receive messages on
        self.client = self.context.socket(zmq.REQ)
        self.client.connect(addr)
        self.poll = zmq.Poller()
        self.poll.register(self.client, zmq.POLLIN)
        self.image_paths = facenet.get_image_paths(facedir)
        self.facedir =  facedir
        self.thread_state = True

        self.curlabels = []
        for i, image_path in enumerate(self.image_paths):
            base_name = os.path.splitext(os.path.split(image_path)[1])[0]
            if '_' in base_name:
                image_id = base_name.split('_')[1]
                self.curlabels.append(image_id)
            else:
                print('{} does not conform to Comparison face database naming conventions'.format(base_name))
        self.features_file = os.path.join(npydir,'features.npy')
        self.labels_file = os.path.join(npydir,'labels.npy')

        self.npydir = npydir
        self.old_labels = []
        self.old_features = []

        if os.path.exists(self.labels_file) and os.path.exists(self.features_file):
            #self.hashIdentifier = HashIdentifierAutoTrain(featuresPath=self.features_file, labelsPath=self.labels_file)
            self.old_labels = list(np.load(self.labels_file))
            self.old_features = list(np.load(self.features_file))
            print('restart lsh_server')
            cmd = 'cd {}; ./lsh_server.sh restart;cd ..'.format(self.npydir)
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        else:
            #self.hashIdentifier = HashIdentifierAutoTrain()
            pass


    def stop(self):
        print("TaskSink thread stop!")
        self.thread_state = False
        self.client.close()
        self.context.term()


    def run(self):
        REQUEST_TIMEOUT = 2500
        while self.thread_state:
            new_features = []
            new_labels = []

            self.curlabels = []
            self.image_paths = facenet.get_image_paths(self.facedir)
            for i, image_path in enumerate(self.image_paths):
                base_name = os.path.splitext(os.path.split(image_path)[1])[0]
                if '_' in base_name:
                    image_id = base_name.split('_')[1]
                    self.curlabels.append(image_id)
                else:
                    print('{} does not conform to Comparison face database naming conventions'.format(base_name))


            same_label = []
            same_features = []
            for k,label in enumerate(self.old_labels):
                if label in self.curlabels:
                    same_label.append(self.old_labels[k])
                    same_features.append(self.old_features[k])
            for k,label in enumerate(self.curlabels):
                if label not in self.old_labels:
                    img = misc.imread(self.image_paths[k])
                    #img = crop(img, True, 160)
                    print(self.image_paths[k])
                    message = [-1,img,1]
                    try:
                        serialized = msgpack.packb(message, default=m.encode)

                        self.client.send(serialized)
                        #socks = dict(self.poll.poll(REQUEST_TIMEOUT))
                        socks = dict(self.poll.poll())
                        if socks.get(self.client) == zmq.POLLIN:
                            message = self.client.recv()
                            message = msgpack.unpackb(message, object_hook=m.decode)
                            if len(message[1])>0:
                                new_features.append(message[1])
                                new_labels.append(label)
                            print('R: recv message from server')
                        else:
                            print "W: No response from server, retrying…"
                            # Socket is confused. Close and remove it.
                            self.client.setsockopt(zmq.LINGER, 0)
                            self.client.close()
                            self.poll.unregister(self.client)

                            print "I: Reconnecting and resending"
                            # Create new connection
                            self.client = self.context.socket(zmq.REQ)
                            self.client.connect("tcp://%s:%s" % (self.lshserver, self.port))
                            self.poll.register(self.client, zmq.POLLIN)
                            self.client.send(serialized)
                    except Exception as e:
                        print e
            if len(new_labels)>0 or len(same_label) != len(self.old_labels):
                #self.hashIdentifier.train_hash_model(np.array(new_features), np.array(new_labels))
                self.old_labels = same_label+new_labels
                self.old_features = same_features+new_features

                np.save(self.features_file,self.old_features)
                np.save(self.labels_file,self.old_labels)
                print('resave {} {}'.format(self.features_file, self.labels_file))
                print('restart lsh_server')
                cmd = 'cd {}; ./lsh_server.sh restart;cd ..'.format(self.npydir)
                p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)



            time.sleep(1)
        self.stop()




def sigint_handler(signum, frame):
    print("main-thread exit")
    global taskSink
    taskSink.stop()
    time.sleep(2)
    sys.exit()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    facedir = '/data/yanhong.jia/datasets/ai_challenger/caption/' \
              'ai_challenger_caption_test1_20170923/' \
              'caption_test1_images_20170923'
    client = Client("tcp://127.0.0.1:12811",facedir)
    client.start()
    client.join()
