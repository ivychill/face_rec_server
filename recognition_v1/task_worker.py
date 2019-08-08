# Task worker
# Connects PULL socket to tcp://localhost:5557
# Collects workloads from ventilator via that socket
# Connects PUSH socket to tcp://localhost:5558
# Sends results to sink via that socket
#
# Author: Lev Givon <lev(at)columbia(dot)edu>


import time
import zmq
import sys,signal
import os
import logging
import msgpack
import msgpack_numpy as m
from  face import Recognition
import configparser
from multiprocessing import Process
import threading
#class TaskWorker(threading.Thread):
class TaskWorker(Process):
    def __init__(self,load_mode_finish_q,modedir,
                 ventilator_addr, sink_mix_addr,
                 alignflag,debugflag,debugoutdir):
        #threading.Thread.__init__(self)
        Process.__init__(self)
        self.load_mode_finish_q = load_mode_finish_q
        self.thread_state = True
        self.faceFeature = None
        self.modedir=modedir

        self.alignflag=alignflag
        self.debugflag=debugflag
        self.debugoutdir=debugoutdir

        self.recv_addr = ventilator_addr
        self.send_addr = sink_mix_addr
    def stop(self):
        print("TaskWorker Process stop!")
        self.thread_state = False


    def initParam(self):
        print("{}start init facenet param".format(os.getpid()))
        logging.debug("{}start init facenet param".format(os.getpid()))
        self.recognition = Recognition(self.modedir,self.debugflag,self.debugoutdir)
        print("{}finish init facenet param".format(os.getpid()))
        logging.debug("{}finish init facenet param".format(os.getpid()))
        self.poller = zmq.Poller()
        self.context = zmq.Context()
        # Socket to receive messages on
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.connect(self.recv_addr)
        print("facenet work connect {} for recv face image!".format(self.recv_addr))
        logging.debug("facenet work connect {} for recv face image!".format(self.recv_addr))
        self.poller.register(self.receiver, zmq.POLLIN)

        # Socket to send messages to
        self.sender = self.context.socket(zmq.PUSH)
        self.sender.connect(self.send_addr)

        print("facenet work connect {} for send face feature!".format(self.send_addr))
        logging.debug("facenet work connect {} for send face feature!".format(self.send_addr))



    def run(self):
        self.initParam()
        self.load_mode_finish_q.put(os.getgid())
        while self.thread_state:
            try:
            #if True:
                socks = dict(self.poller.poll())
                if self.receiver in socks and socks[self.receiver] == zmq.POLLIN:
                    message = self.receiver.recv()
                    start_time = time.time()
                    message = msgpack.unpackb(message, object_hook=m.decode)
                    image = message[1]
                    faces = self.recognition.identify(image)
                    message[1]= faces[0].embedding
                    message.append(faces[0].name)
                    serialized = msgpack.packb(message, default=m.encode)
                    self.sender.send(serialized)
                    #print("worker send =====> message")
            except Exception as e:
                logging.error(e)
                print e
                #break
        self.receiver.close()
        self.sender.close()
        self.context.term()


def sigint_handler(signum, frame):
    print("main-thread exit")
    global taskWorker
    taskWorker.stop()
    time.sleep(2)
    sys.exit()
def logconfig():
    sys.path.append('../')
    import logging
    from datetime import datetime
    LOG_PATH = os.path.dirname(os.path.realpath(__file__))
    LOG_SUB = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    LOG_FILE = '/../log/' + os.path.split(__file__)[-1].split('.')[0] + LOG_SUB + '.log'
    format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
    logging.basicConfig(format=format, level=logging.DEBUG, filename=LOG_PATH + LOG_FILE, filemode='w')


from multiprocessing import Queue
if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    config = configparser.ConfigParser()
    load_mode_finish_q = Queue()
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..','cfg','configfile.cfg')

    sink_mix_addr = config.get("facenet", "sink_mix_addr")

    near_model = os.path.join('..',config.get("facenet", "near_model"))
    devstr = config.get("facenet", "NEAR_CUDA_VISIBLE_DEVICES")
    sdev = devstr.split(',')
    srflag = bool(int(config.get("facenet", "AI3flag")))
    alignflag = bool(int(config.get("facenet", "alignflag")))
    debugflag = bool(int(config.get("facenet", "debugflag")))
    debugoutdir = config.get("facenet", "debug_out_dir")
    ventilator_near_addr = config.get("facenet", "ventilator_near_addr")

    taskWorker = TaskWorker(load_mode_finish_q, near_model,
                           ventilator_near_addr, sink_mix_addr,
                           srflag, alignflag, debugflag, debugoutdir)

    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    taskWorker.start()
    load_mode_finish_q.get()

    taskWorker.join()


