# -*- coding: utf8 -*-
# ! /usr/bin/python
'''
ChangeLog:
#2018-7-19 为工作线程加保护，避免异常情况下，无法响应
#2018-08-02 根据Hash Server接口设计V1.0文档修改
'''

from __future__ import division
import numpy as np
import falconn as fc
import time
import threading
import zmq
import sys, getopt
import pickle
import collections
import traceback
import msgpack
# import msgpack_numpy as m

import logging
import logging.handlers

eDistanceLimit = 2
LOG_FILE = 'lsh_server.log'

handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=100 * 1024 * 1024, backupCount=5)  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'

formatter = logging.Formatter(fmt)  # 实例化formatter
handler.setFormatter(formatter)  # 为handler添加formatter

logger = logging.getLogger('lsh_server')  # 获取名为lsh_server的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.INFO)

def getsimi(v1,v2):
    # print np.linalg.norm(v2-v1)
    if np.linalg.norm(v2-v1) <eDistanceLimit:
        return getcosine(v1,v2)
    else:
        return 0.0
	
def getcosine(v1,v2):
        cos_angle = ( np.dot(v1, v2) / ((np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2))) + 1) / 2
        return cos_angle


class lsh_server:
    def __init__(self,feature_file,label_file,port,worker_num=10):
        self.url_worker = 'inproc://ping-workers'
        url_router = "tcp://*:%s"%port
        self.worker_num = worker_num
        self.worker_counts=[0L]*worker_num
        self.context = zmq.Context()
        self.router = self.context.socket(zmq.ROUTER)
        self.router.bind(url_router)
        self.workers = self.context.socket(zmq.DEALER)
        self.workers.bind(self.url_worker)

        self.label = np.load(label_file)
        logger.info ("start load feature data")
        t1 = time.time()
        self.feature = np.load(feature_file)
        t2 = time.time()
        logger.info ("load cost time:%f" % (t2 - t1))
        dp = fc.get_default_parameters(self.feature.shape[0], self.feature.shape[1],
                                       fc.DistanceFunction.EuclideanSquared)
        ds = fc.LSHIndex(dp)
        train_st = time.time()
        ds.setup(self.feature)
        train_et = time.time()
        logger.info ("train cost time:%f" % (train_et - train_st))
        # self.qo = ds.construct_query_object()
        self.qp=ds.construct_query_pool()


    def construct_name_confidence(self,query_array,indexs):
        names=[]
        confidences=[]
        for i in range(len(indexs)):
            if indexs[i] == None:
                names.append(None)
            else:
                names.append(self.label[indexs[i]])
            cos = np.dot(query_array[i],self.feature[indexs[i]])/(np.linalg.norm(query_array[i])*np.linalg.norm(self.feature[indexs[i]]))
            confidences.append((cos+1)*0.5)
            
        # freq = collections.Counter(names)
        # s=sum(freq.values())
        # for k,v in freq.items():
        #     freq[k]=v/s
        return names,confidences

    def loop(self):
        threading.Thread(target=self.stater).start()
        for i in range(self.worker_num):
            thread = threading.Thread(target=self.worker, args=(i, self.url_worker, self.context,))
            thread.start()
        zmq.device(zmq.QUEUE, self.router, self.workers)
        self.router.close()
        self.workers.close()
        self.context.term()
    
    def stater(self):
        last=sum(self.worker_counts)
        sleep_base=5
        while True:
            time.sleep(sleep_base)
            logger.info("%s"%self.worker_counts)
            cur=sum(self.worker_counts)
            logger.info("lsh_server Concurrency:%f"%((cur-last)/sleep_base))
            last=cur


    def worker(self,name, url_worker, context):
        logger.info('worker {0} start'.format(name))
        worker = context.socket(zmq.REP)
        worker.connect(url_worker)
        count=0
        while True:
            try:
                errorcode = 0
                message = worker.recv()
                logger.info("##################\nworker %d receive count:%d" %(name,count))
                t1 = time.time()
                #query = pickle.loads(message)
                #2018-08-02 根据Hash Server接口设计V1.0文档修改
                # query_arr=msgpack.unpackb(message,object_hook=m.decode)
                search_arg=msgpack.unpackb(message,raw=True)
                cmd=search_arg[0]
                map_start=1
                find_k=5
                if cmd == 0x8001:
                    map_start=1
                elif cmd == 0x8002:
                    find_k=search_arg[1]
                    map_start=2
                else:
                    errorcode=-5
                    raise ValueError("Invalid command")
                query_arr=np.fromstring(search_arg[map_start]['data'],dtype=np.dtype(search_arg[map_start]['type'])).reshape(search_arg[map_start]['shape'])
                h_angle=search_arg[map_start+1]
                v_angle=search_arg[map_start+2] 
                logger.debug ("cmd:%x, %s---%s" %( cmd, h_angle,  v_angle   ))          
                if len(query_arr.shape) != 2:
                    errorcode = -3
                    raise ValueError('emb_array must be a two-dimensional array')
                if len(query_arr) != len(h_angle) or len(query_arr) != len(v_angle):
                    errorcode=-2
                    raise ValueError('len of all array must be same')
                # print query_arr
                # print query_arr.dtype
                t2 = time.time()
                logger.debug ("unpack cost time:%f"%(t2-t1))
                # print ("this is query:%s"%query)
                # print query
                #indexs=[]
                names=[]
                confidences=[]
                ts=time.time()
                for query in query_arr:
                    result=None
                    t1 = time.time()
                    print('.........................')
                    print('datatype:',query.dtype)
                    print('datasize:',query.shape)
                    # result = self.qo.find_nearest_neighbor(query)
                    if cmd == 0x8001:
                        result = self.qp.find_nearest_neighbor(query)
                        t2 = time.time()
                        logger.debug ("\nresult:[%d],cost time:%f"%(result,t2-t1))
                        names.append(self.label[result] if result != None else None)
                        confidences.append(getsimi(query,self.feature[result]) if result != None else 0.0)
                    elif cmd == 0x8002:
                        result = self.qp.find_k_nearest_neighbors(query,find_k)
                        t2 = time.time()
                        logger.debug ("\nresult:[%s],cost time:%f"%(",".join([str(x) for x in result]),t2-t1))
                        for x in result:
                            #print x                            
                            names.append(self.label[x])
                            confidences.append(getsimi(query,self.feature[x]))
                te=time.time()
                logger.debug ("Total compute time:%f,mean compute time:%f"%((te-ts),(te-ts)/len(query_arr)))
                #print indexs
                #name_confidence=self.construct_name_confidence(query_arr,indexs)
                name_confidence=(errorcode,names,confidences)
                t1 = time.time()
                #serialized = pickle.dumps(name_confidence, protocol=0)
                serialized=msgpack.packb(name_confidence,use_bin_type=True)
                t2 = time.time()
                logger.debug ("before send,msg pack cost time:%f"%(t2-t1))
                worker.send(serialized)
                logger.debug ("after send")
                logger.info("#########################\nworker %d send count:%d"%(name,count))
                count+=1
                self.worker_counts[name]+=1
            except Exception, e:
                logger.error ("e.message:\t%s"%e.message)
                logger.error ("trace info:\n%s" % traceback.format_exc())
                print ("e.message:\t%s"%e.message)
                print ("trace info:\n%s" % traceback.format_exc())
                #2018-7-19 为工作线程加保护，避免异常情况下，无法响应
                # break 
                if errorcode == 0:
                    if e.message.find('query dimension mismatch') != -1:
                        errorcode = -1
                    elif e.message.find('dataset and query must have the same dtype') != -1:
                        errorcode = -4
                    else:
                        errorcode = -1000               
                name_confidence=(errorcode,['Fatal Error:'+e.message],[0.0])
                serialized=msgpack.packb(name_confidence,use_bin_type=True)
                worker.send(serialized)
        worker.close()

def usage():
    print "python lsh_server -f feature.npy -l labels.npy -k 5 "

def get_arg():
    opts, args = getopt.getopt(sys.argv[1:], "hf:k:l:p:t:")
    feature_file = ""
    label_file = ""
    k = 5
    port = "5555"
    worker_num=10
    for op, value in opts:
        if op == "-f":
            feature_file = value
        elif op == "-l":
            label_file = value
        elif op == "-k":
            k = int(value)
        elif op == "-p":
            port = value
        elif op == "-t":
            worker_num = int(value)
        elif op == "-h":
            usage()
            sys.exit()
    if feature_file == "" or label_file == "":
            usage()
            sys.exit()

    return feature_file,label_file,k,port,worker_num


if __name__ == '__main__':
    feature_file,label_file,k,port,worker_num=get_arg()
    server=lsh_server(feature_file,label_file,port,worker_num)
    server.loop()

# python lsh_server.py -f features.npy -l labels.npy -k 5