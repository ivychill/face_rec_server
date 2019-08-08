# -*- coding: utf8 -*-
# ! /usr/bin/python

from multiprocessing import Process
import tensorflow as tf
import numpy as np
import os
import math
from scipy import misc
import time
import sys
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(path,'../recognition'))
import face
import facenet
class Embedding:
    def __init__(self,facenet_model_checkpoint):
        self.batch_size = 64
        self.image_size = 160
        self.detect = None
        self.facenet_model_checkpoint = facenet_model_checkpoint
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(self.facenet_model_checkpoint)
            self.detect = face.Detection()

    def get_image_paths_and_labels(self, dataset):
        image_paths_flat = []
        class_labels_flat = []
        labels = []
        for i in range(len(dataset)):
            image_paths_flat += dataset[i].image_paths
            labels += [i] * len(dataset[i].image_paths)
            class_labels_flat += [dataset[i].name.replace('_', ' ')] * len(dataset[i].image_paths)
        #print('Number of classes: %d' % len(dataset))
        #print('Number of images: %d' % len(image_paths_flat))
        return image_paths_flat, labels, class_labels_flat

    def load_data(self, image_paths,  do_prewhiten=True):
        nrof_samples = len(image_paths)
        images = np.zeros((nrof_samples, self.image_size, self.image_size, 3))
        for i in range(nrof_samples):
            if os.path.isfile(image_paths[i]):
                img = misc.imread(image_paths[i])
                img = img[:, :, 0:3]
                faces = self.detect.find_faces(img)
                if (len(faces) < 1):
                    print('unable align:{}'.format(image_paths[i]))
                else:
                    img = facenet.prewhiten(faces[0].image)
                    images[i, :, :, :] = img
                #img = misc.imresize(img, (self.image_size, self.image_size,3), interp='bilinear')
                #images[i, :, :, :] = img
        return images

    def generate_embedding(self, paths):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = self.load_data(paths_batch)
            if len(images)<1:
                print ('There must be at least one image can not align for each class in the dataset ')
                continue
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = self.sess.run(embeddings, feed_dict=feed_dict)

        return emb_array


class ClassifierModel(Process):
    def __init__(self, q,benchmark_dir):
        Process.__init__(self)
        self.thread_state = False
        self.q = q
        self.embedding = None
        self.model = None
        self.class_names = None
        self.benchmark_dir =benchmark_dir
        self.classifier_dir =  os.path.dirname(__file__)+"/../models"
        self.classifier_filenames = face.get_classifier_model_filenames(self.classifier_dir)


    def stop(self):
        self.thread_state = False
        print("ClassifierModel %d Process stop!" % (os.getpid()))

    def run(self):
        self.thread_state = True
        self.embedding = Embedding()

        oldlabels = np.load(os.path.join(self.classifier_dir, "labels.npy"))

        self.q.put(os.getpid())
        while self.thread_state:
            dataset = facenet.get_dataset(self.benchmark_dir)
            paths, labels, class_labels = self.embedding.get_image_paths_and_labels(dataset)

            benchmark_classes_set = set(class_labels)
            model_classes_set = set(oldlabels)
            diff_set = benchmark_classes_set.symmetric_difference(model_classes_set)
            if len(diff_set) > 0:
                print('Number of classes: %d' % len(dataset))
                print('Number of images: %d' % len(paths))
                start_time = time.time()
                emb_array = self.embedding.generate_embedding(paths)
                # Create a list of class names
                class_names = [cls.name.replace('_', ' ') for cls in dataset]
                # subname = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
                # self.classifier_filenames = os.path.join(self.classifier_dir, 'knn'+subname+'.pkl')

                # Saving classifier model
                # with open(self.classifier_filenames, 'wb') as outfile:
                #     pickle.dump((self.model, class_names), outfile)
                np.save(os.path.join(self.classifier_dir, "features.npy"), emb_array)
                np.save(os.path.join(self.classifier_dir, "labels.npy"), class_labels)

                # print('Saved classifier model to file "%s" %4.4fs' % (self.classifier_filenames,time.time()-start_time))
                self.class_names = class_names
                self.q.put(os.getpid())
            time.sleep(1)




