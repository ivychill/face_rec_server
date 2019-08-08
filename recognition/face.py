# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import os
import math
import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
import facenet
import align.detect_face
import time
import sys
import logging
from datetime import datetime
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(path,'../'))

class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
        self.align_time = 0
        self.emb_time = 0
        self.ident_time = 0



import importlib
class Encoder:
    def __init__(self,facenet_model_checkpoint):
        network = importlib.import_module('recognition.models.inception_resnet_v1')
        print('Pre-trained model: %s' % os.path.expanduser(facenet_model_checkpoint))
        self.image_size = 160

        with tf.Graph().as_default():
            self.images_placeholder =tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='input')
            self.phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
            # Build the inference graph
            prelogits, _ = network.inference(self.images_placeholder, 1.0,
                                             phase_train=self.phase_train_placeholder, bottleneck_layer_size=128,
                                             weight_decay=0.0)
            model_exp = os.path.expanduser(facenet_model_checkpoint)
            self.embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
            #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            meta_file, ckpt_file = facenet.get_model_filenames(model_exp)
            saver = tf.train.Saver(tf.trainable_variables())
            print('Restoring pretrained model: %s' %facenet_model_checkpoint)
            saver.restore(self.sess, os.path.join(model_exp, ckpt_file))
        self.embedding_size = self.embeddings.get_shape()[1]
        # self.sess = tf.Session()
        # with self.sess.as_default():
        #     facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, images):
        with tf.Graph().as_default():
            prewhiten_face = facenet.prewhiten(images)

            # Run forward pass to calculate embeddings
            feed_dict = {self.images_placeholder: [prewhiten_face], self.phase_train_placeholder: False}
            return self.sess.run(self.embeddings, feed_dict=feed_dict)[0]




class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=24):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(allow_growth=True)
            #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            #with sess.as_default():

            return align.detect_face.create_mtcnn(self.sess, None)

    def find_faces(self, image):
        faces = []
        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                        self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(face)

        return faces

class Recognition:
    def __init__(self,modedir,debugflag=False,outdir=None):
        self.detect = Detection()
        self.encoder = Encoder(modedir)
        #self.identifier = Identifier()
        self.output_dir = outdir
        if self.output_dir !=None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        self.debug = debugflag
        self.raw_dir = None
        self.align_dir = None
        if self.debug:
            self.raw_dir= os.path.join(self.output_dir, 'raw')
            if not os.path.exists(self.raw_dir):
                os.makedirs(self.raw_dir)
            self.align_dir = os.path.join(self.output_dir, 'align')
            if not os.path.exists(self.align_dir):
                os.makedirs(self.align_dir)

    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)
        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def identify(self, image,alignflag=1):

        start_time = time.time()

        if alignflag==1:
            faces = self.detect.find_faces(image)
        else:
            faces = [Face()]
            faces[0].image = image

        align_time = time.time()-start_time
        t1 = time.time()
        if self.debug:
            subname = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

            filename = os.path.join(self.raw_dir,
                                    '_' + subname + '.png')
            # cv2.imwrite(filename, cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
            misc.imsave(filename, faces[0].container_image)
            filename = os.path.join(self.align_dir,
                                    '_' + subname + '.png')
            # cv2.imwrite(filename, cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
            misc.imsave(filename, faces[0].image)
        debug_time = time.time()-t1
        for i, face in enumerate(faces):
            face.align_time = align_time
            t1 = time.time()
            face.embedding = self.encoder.generate_embedding(face.image)
            face.emb_time = time.time()-t1
            t1 = time.time()
            #face.name = self.identifier.identify(face)
            face.ident_time = time.time()-t1

            if self.debug:
                subname = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
                dirname = os.path.join(self.output_dir, 'raw')
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                filename = os.path.join(dirname,
                                        '_' + subname + '_' + str(i).zfill(4) + '.png')
                # cv2.imwrite(filename, cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
                misc.imsave(filename, face.container_image)
        if len(faces)>0:
            print('{} proc all time:{:.3f} align:{:.3f} ebm:{:.3f} debug:{:.3f}'.format(
                os.getpid(), time.time() - start_time, align_time,
                faces[0].emb_time, debug_time))
            logging.debug('{} proc all time:{:.3f} align:{:.3f} facenet:{:.3f} debug:{:.3f}'.format(
                os.getpid(), time.time() - start_time, align_time,
                faces[0].emb_time, debug_time))
        else:
            print('{} proc all time:{:.3f} align:{:.3f} debug:{:.3f}'.format(
                os.getpid(), time.time() - start_time, align_time,debug_time))
            logging.debug('{} proc all time:{:.3f} align:{:.3f} debug:{:.3f}'.format(
                os.getpid(), time.time() - start_time, align_time,debug_time))
        return faces