# -*- coding: utf8 -*-
# ! /usr/bin/python
import signal
import configparser
import fcntl, sys, os
from multiprocessing import Queue
from recognition.task_worker import TaskWorker
from recognition.task_ventilator import  TaskVentilator
from recognition.task_sink import TaskSink
from recognition.server_recv import ServerRecv
from recognition.server_send import ServerSend
import logging
from datetime import datetime


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
    recvQueue = Queue()
    sendQueue = Queue()
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



    server_recv_addr = config.get("facenet", "server_recv_addr")
    server_send_addr = config.get("facenet", "server_send_addr")
    sink_addr = config.get("facenet", "sink_addr")
    ventilator_addr = config.get("facenet", "ventilator_addr")

    procnum = int(config.get("facenet", "procnum"))

    model = config.get("facenet", "model")
    devstr = config.get("facenet", "CUDA_VISIBLE_DEVICES")
    sdev = devstr.split(',')

    alignflag = bool(int(config.get("facenet", "alignflag")))

    saveflag = bool(int(config.get("facenet", "saveflag")))
    batchflag = bool(int(config.get("facenet", "batchflag")))
    debugoutdir = config.get("facenet", "debug_out_dir")

    for i in range(procnum):

        face_feat = TaskWorker(load_mode_finish_q,model,
                               ventilator_addr,sink_addr,
                               alignflag,saveflag,debugoutdir)
        logging.debug("near proc {} run in CUDA {}".format(i, str(sdev[i % len(sdev)])))
        print("near proc {} run in CUDA {}".format(i, str(sdev[i % len(sdev)])))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(sdev[i % len(sdev)])
        face_feat.start()
        face_proc.append(face_feat)

    for i in range(len(face_proc)):
        load_mode_finish_q.get()

    task_recv = ServerRecv(server_recv_addr,recvQueue)
    task_ventilator = TaskVentilator(ventilator_addr,recvQueue)
    task_sink = TaskSink(sink_addr,sendQueue)

    task_recv.start()
    task_sink.start()
    task_ventilator.start()
    task_recv.join()
    task_sink.join()
    task_ventilator.join()
    for i in range(procnum):
        face_proc[i].join()




