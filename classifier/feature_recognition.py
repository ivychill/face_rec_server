# -*- coding: utf8 -*-
# ! /usr/bin/python
from __future__ import division
import numpy as np
import falconn as fc
import time
import collections
import os
import pickle
DISTANCE_THRESHOLD=1.0
class Identifier:
    def __init__(self,labelsPath,featuresPath):
        self.load_identifier(labelsPath,featuresPath)

    def construct_name_confidence(self,array):
        names=[self.label[i] for i in array]
        freq = collections.Counter(names)
        s=sum(freq.values())
        for k,v in freq.items():
            freq[k]=v/s
        return freq.keys(),freq.values()

    def lsh_hash5(self, query):
        t1 = time.time()
        r = self.qo.find_k_nearest_neighbors(query,5)
        t2 = time.time()
        #print ("\nresult:[%s],cost time:%f" % (",".join([str(x) for x in r]), t2 - t1))
        return self.construct_name_confidence(r)

    def compute_norm(self, f, a_query):
        dic = {}
        for j in range(1):
            q = a_query
            dis_min = DISTANCE_THRESHOLD
            index = 0
            for i in range(f.shape[0]):
                dis = np.linalg.norm(f[i] - q)
                # dis = self.getcosine(f[i] , q)
                if dis < dis_min:
                    dis_min = dis
                    index = i
            if dis_min < DISTANCE_THRESHOLD:
                dic[j] = (index, dis_min)
        dis_min = DISTANCE_THRESHOLD
        # min_v=None
        if len(dic)== 0:
            return None
        for k, v in dic.items():
            if dic[k][1] < dis_min:
                dis_min = v[1]
                min_index = v[0]
                # min_v=a_query[k]
        return min_index

    def identifys5(self, emb_array,probLimit,e_disLimit):
        resName = []
        resProb = []
        for i in range(emb_array.shape[0]):
            min_index = self.compute_norm(self.idfeature, emb_array[i])
            if min_index == None:
                resName.append('unknown')
                resProb.append(0)
            else:
                r = self.qo.find_nearest_neighbor(self.idfeature[min_index])
                # print("index:....",i,"-----",r,'ind:',min_index)
                # cosd = self.getcosine(self.feature[r],emb_array[i])
                cosd = self.getcosine(self.feature[r],emb_array[i])
                resName.append(self.label[r])
                resProb.append(cosd)
        return resName,resProb

    def identifys5s(self, emb_array,probLimit,e_disLimit):
        resName = []
        resProb = []
        resDis = []
        for i in range(emb_array.shape[0]):
            r = self.qo.find_nearest_neighbor(emb_array[i])
            if r == None or r == []:
                resName.append('unknown')
                resProb.append(0)
                resDis.append(0)
            else:
                dis = np.linalg.norm(self.feature[r] - emb_array[i])
                cosd = self.getcosine(self.feature[r], emb_array[i])
                print('dis:',dis,'cos:',cosd)
                if dis < e_disLimit and cosd >probLimit : # 最小欧式距离限
                    resName.append(self.label[r])
                    resProb.append(cosd)
                    resDis.append(dis)
                else:
                    resName.append('unknown')
                    resProb.append(0)
                    resDis.append(0)
                # print("index:....",i,"-----",r,'ind:',min_index)
        return resName,resProb,resDis

    def lsh_hash(self, query):
        t1 = time.time()
        #print('11111111111111',query)
        r = self.qo.find_nearest_neighbor(query)
        t2 = time.time()
        #print ("\nresult:[%s],cost time:%f" % (",".join([str(x) for x in r]), t2 - t1))
        return r

    def identifys(self, emb_array):
        names = []
        for i in range(len(emb_array)):
            name = self.lsh_hash(emb_array[i])
            names += [name]
        return names

    def load_identifier(self,labelFile,featuresFile):
        self.label = np.load( labelFile)
        print "start load feature data"
        print(labelFile)
        t1 = time.time()
        self.feature = np.load(featuresFile)
        self.embs = self.feature
        print ("feature dtype:%d", self.feature.dtype)
        t2 = time.time()
        print ("load cost time:%f" % (t2 - t1))
        self.dp = fc.get_default_parameters(self.feature.shape[0], self.feature.shape[1],
                                            fc.DistanceFunction.EuclideanSquared)
        self.dp.l = 30
        self.ds = fc.LSHIndex(self.dp)
        train_st = time.time()
        self.ds.setup(self.feature)
        train_et = time.time()
        print ("train cost time:%f" % (train_et - train_st))
        self.qo = self.ds.construct_query_object()

    def getcosine(self,fea1,fea2):

        # Lx = np.sqrt(embeddings1.dot(embeddings1.T))
        # Lx = np.sqrt(np.sum(np.square(embeddings1), 1)
        # # np.dot(embeddings1.T, embeddings1)
        # # Ly = np.sqrt(embeddings2.dot(embeddings2.T))
        # Ly = np.sqrt(np.sum(np.square(embeddings2), 1))
        # cos_angle = np.sum(embeddings1 * embeddings2, 1) / (Lx * Ly)
        # cos_angle = (cos_angle+1)*0.5
        cos_angle = ( np.dot(fea1, fea2) / ((np.linalg.norm(fea1, 2) * np.linalg.norm(fea2, 2))) + 1) / 2
        return cos_angle

    def similarity(self,names,embeddings):
        simil = np.zeros(len(names))
        for k in range(len(names)):
            for i in range(len(self.label)):
                if self.label[i] == names[k]:
                    # temp = cosine_similarity(faces[0].embedding.reshape(1,-1), self.embs[i].reshape(1,-1))
                    temp = self.getcosine(embeddings[k].reshape(1, -1), self.embs[i].reshape(1, -1))
                    if simil[k] < temp[0]:
                        simil[k] = temp[0]
        return simil









