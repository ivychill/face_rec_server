# -*- coding: utf8 -*-
# ! /usr/bin/python
import signal
import configparser
from recognition.server_work import Worker,TaskRoute
from datetime import datetime
from classifier.classifier_client import *
from multiprocessing import Queue
import logging
import fcntl, sys, os
from web.WebInterfaceServer import WebServer
pidfile = 0
def sigint_handler(signum, frame):
    print("main-thread exit")
    global face_proc
    for i in range(len(face_proc)):
        face_proc[i].stop()
    sys.exit()

def ApplicationInstance():
    global pidfile
    pidfile = open(os.path.realpath(__file__), "r")
    try:
        fcntl.flock(pidfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except:
        print "another instance is running..."
        logging.critical("another instance is running...")
        sys.exit(1)


if __name__ == '__main__':
    ApplicationInstance()
    signal.signal(signal.SIGINT, sigint_handler)
    face_proc = []
    load_mode_finish_q = Queue()
    config = configparser.ConfigParser()
    config.read("cfg/configfile.cfg")
    debugflag = bool(int(config.get("facenet", "debugflag")))
    LOG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'log')
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    LOG_SUB = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    LOG_FILE = os.path.split(__file__)[-1].split('.')[0] + '_' + LOG_SUB + '.log'
    format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
    if debugflag==True:
        logging.basicConfig(format=format, level=logging.DEBUG, filename=os.path.join(LOG_PATH, LOG_FILE), filemode='w')
    else:
        logging.basicConfig(format=format, level=logging.INFO, filename=os.path.join(LOG_PATH, LOG_FILE), filemode='w')

    front_addr = config.get("facenet", "front_addr")
    backend_addr = config.get("facenet", "backend_addr")
    hash_server_addr = config.get("facenet", "hash_server_addr")
    web_addr = config.get("facenet", "web_addr")
    web_port = config.get("facenet", "web_port")
    procnum = int(config.get("facenet", "procnum"))

    model = config.get("facenet", "model")
    devstr = config.get("facenet", "CUDA_VISIBLE_DEVICES")
    sdev = devstr.split(',')


    saveflag = bool(int(config.get("facenet", "saveflag")))

    save_dir = config.get("facenet", "save_dir")
    face_lib = config.get("facenet", "face_lib")
    face_npy = config.get("facenet", "face_npy")
    for i in range(procnum):
        proc_worker = Worker(load_mode_finish_q,model,
                           backend_addr,hash_server_addr,debugflag,saveflag,save_dir,face_lib)
        logging.debug("near proc {} run in CUDA {}".format(i, str(sdev[i % len(sdev)])))
        print("near proc {} run in CUDA {}".format(i, str(sdev[i % len(sdev)])))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(sdev[i % len(sdev)])
        proc_worker.start()
        face_proc.append(proc_worker)

    for i in range(len(face_proc)):
        load_mode_finish_q.get()


    task_route = TaskRoute(front_addr,backend_addr)
    task_route.start()

    web_srv = WebServer(web_addr,int(web_port))
    web_srv.start()

    compare_client = Client(front_addr, face_lib, face_npy)
    compare_client.start()
    task_route.join()
    compare_client.join()
    web_srv.join()
    for i in range(procnum):
        proc_worker[i].join()