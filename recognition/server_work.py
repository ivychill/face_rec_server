# coding=utf-8
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
from multiprocessing import Process
from  face import Recognition
import time
from npy.hash_client import Identifier
import facenet
class TaskRoute(threading.Thread):
    def __init__(self,frontAddr,backAddr):
        threading.Thread.__init__(self)
        self.context = zmq.Context()
        self.frontAddr = frontAddr
        self.backAddr = backAddr
        self.frontend =  self.context.socket(zmq.ROUTER)
        self.backend =  self.context.socket(zmq.DEALER)

        # Initialize poll set
        self.poller = zmq.Poller()
        self.thread_state = True


    def stop(self):
        print("TaskSink thread stop!")
        self.thread_state = False
        self.frontend.close()
        self.backend.close()
        self.context.term()

    def run(self):

        self.frontend.bind(self.frontAddr)
        self.backend.bind(self.backAddr)
        # Initialize poll set
        self.poller.register(self.frontend, zmq.POLLIN)
        self.poller.register(self.backend, zmq.POLLIN)
        while self.thread_state:
            try:
                socks = dict(self.poller.poll())

                # frontend 收到了提问后，由backend发送给REP端
                if socks.get(self.frontend) == zmq.POLLIN:
                    message = self.frontend.recv_multipart()
                    self.backend.send_multipart(message)

                # backend 收到了回答后，由frontend发送给REQ端
                if socks.get(self.backend) == zmq.POLLIN:
                    serialized = self.backend.recv_multipart()
                    #message = msgpack.unpackb(serialized, object_hook=m.decode)
                    #name = self.hashIdentifier.hash_search(message[1],EdisLimit=10,cosDisLimit=0,k=3)
                    #message.append(name)
                    #serialized = msgpack.packb(message, default=m.encode)
                    self.frontend.send_multipart(serialized)
            except Exception as e:
                logging.error(e)
                print e
'''
收到请求后回复
@author: jyh
'''
SUPPORT_FEATURE_DIMENSION = 512
from scipy import misc
class Worker(Process):
    def __init__(self,load_mode_finish_q,modedir,
                 backAddr,hash_server_addr, debugflag,saveflag,savedir,face_lib):
        #threading.Thread.__init__(self)
        Process.__init__(self)
        self.load_mode_finish_q = load_mode_finish_q
        self.thread_state = True
        self.faceFeature = None
        self.modedir=modedir
        self.hash_server_addr = hash_server_addr
        self.face_lib_dir = face_lib

        self.debugflag=debugflag
        self.savedir=savedir
        self.saveflag = saveflag
        self.recv_addr = backAddr

    def stop(self):
        print("TaskWorker Process stop!")
        self.thread_state = False

    def update_idtoname(self):
        self.image_paths = facenet.get_image_paths(self.face_lib_dir)
        for i, image_path in enumerate(self.image_paths):
            image_id = os.path.splitext(os.path.split(image_path)[1])[0].split('_')[1]
            image_name = os.path.splitext(os.path.split(image_path)[1])[0].split('_')[0]
            self.idtoname[image_id] = [image_name, image_path]

    def initParam(self):
        print("{}start init facenet param".format(os.getpid()))
        logging.debug("{}start init facenet param".format(os.getpid()))
        self.recognition = Recognition(self.modedir,self.debugflag,self.savedir)
        print("{}finish init facenet param".format(os.getpid()))
        logging.debug("{}finish init facenet param".format(os.getpid()))
        self.poller = zmq.Poller()
        self.context = zmq.Context()
        # Socket to receive messages on
        self.receiver = self.context.socket(zmq.REP)
        self.receiver.connect(self.recv_addr)
        print("facenet work connect {} for recv face image!".format(self.recv_addr))
        logging.debug("facenet work connect {} for recv face image!".format(self.recv_addr))

        self.hash_client = Identifier(self.hash_server_addr)
        self.poller.register(self.receiver, zmq.POLLIN)
        self.idtoname = {}
        self.update_idtoname()

    def run(self):
        self.initParam()
        self.load_mode_finish_q.put(os.getgid())
        while self.thread_state:
            #try:
            #if True:
                socks = dict(self.poller.poll())
                if self.receiver in socks and socks[self.receiver] == zmq.POLLIN:
                    message = self.receiver.recv()
                    print("worker send =====> message")
                    start_time = time.time()
                    message = msgpack.unpackb(message, object_hook=m.decode)
                    image = message[1]
                    flag = message[0]

                    if flag<0:
                        faces = self.recognition.identify(image, message[2])
                        message[1] = faces[0].embedding
                        self.update_idtoname()
                    else:
                        faces = self.recognition.identify(image)
                        if len(faces)>0:
                            #message[1]= faces[0].embedding
                            images_id = self.hash_client.search(faces[0].embedding.reshape(1, SUPPORT_FEATURE_DIMENSION), [1], [1])
                            images_name = []
                            images = []
                            if 0:
                                for id in images_id[0]:
                                    images_name.append(self.idtoname[id][0])
                                    img = misc.imread(self.idtoname[id][1])
                                    images.append(img)
                                #message.append(images_name)
                                message[1] = images_name
                                message.append(images_id[0])
                                message.append(images_id[1])
                                message.append(images)
                            else:
                                PID = []
                                confidences = []

                                for k, id in enumerate(images_id[0]):
                                    if images_id[1][k] > 0.78:
                                        images_name.append(self.idtoname[id][0])
                                        img = misc.imread(self.idtoname[id][1])
                                        images.append(img)
                                        PID.append(id)
                                        confidences.append(images_id[1][k])
                                # message.append(images_name)
                                message[1] = images_name
                                message.append(PID)
                                message.append(confidences)
                                message.append(images)
                        else:
                            message[1] =[]

                    #message.append(faces[0].name)
                    serialized = msgpack.packb(message, default=m.encode)
                    self.receiver.send(serialized)
                    print("worker send =====> message")
            #except Exception as e:
            #    logging.error(e)
            #    print e
                #break
        self.receiver.close()

        self.context.term()
