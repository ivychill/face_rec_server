# -*- coding: utf8 -*-
# ! /usr/bin/python
import sys,signal
from classifier_model import ClassifierModel
from feature_recognition import get_classifier_model_filenames,Identifier
from multiprocessing import Queue
import os
import numpy as np
import time
def sigint_handler(signum,frame):
    print("main-thread exit")
    sys.exit()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)

    load_mode_finish_q = Queue()
    class_model = ClassifierModel(load_mode_finish_q, '/home/asus/ai/bench_image_classes')
    class_model.start()
    face_recognition = Identifier()


    #labels_name = np.load(os.path.join(os.path.dirname(__file__), "labels_name.npy"))

    emb = np.load(os.path.join(os.path.dirname(__file__), "signatures.npy"))

    while True:
        #print(load_mode_finish_q.get())
        if(load_mode_finish_q.qsize()>0):
            load_mode_finish_q.get()
            face_recognition.load_identifier()
        print(get_classifier_model_filenames(os.path.join(os.path.dirname(__file__), "../models")), '!!!!!!')


        names, confidence =face_recognition.identifys(emb)
        time.sleep(2)
        print(names)
        print(confidence)

    class_model.join()










