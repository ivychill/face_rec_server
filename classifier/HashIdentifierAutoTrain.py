# -*- coding=utf-8 -*-
import time
import falconn as fc
import threading
import numpy as np
import Queue
import logging
import os


class HashIdentifierAutoTrain(object):
    def __init__(self,featuresPath=None,labelsPath=None,loadDefaultModel=False):
        self.features = None
        self.labels = None
        self.tempFeatures = None
        self.tempLabels = None
        self.qo_hash = None
        self.EdisLimit = 1.
        self.cosDisLimit = 0.9
        self.loadDefaultModel = loadDefaultModel
        self.isSearching = False

        self.inpQ = Queue.Queue()
        self.outQ = Queue.Queue()
        self.trainHashThread = TrainHashModel(self.inpQ,self.outQ)
        self.trainHashThread.start()
        self.hash_init(featuresPath, labelsPath)

    def hash_init(self,featuresPath,labelsPath):
        if (featuresPath is not None) and (labelsPath is not None):
            if not(os.path.exists(featuresPath)) or not(os.path.exists(labelsPath)):
                print('feature %s or label %s path is not exists!!'%(featuresPath,labelsPath))
                logging.debug('feature or label path is not exists!!')
                return None
            features = np.load(featuresPath)
            labels = np.load(labelsPath)
            print('load feature %s and label %s'%(featuresPath,labelsPath))
            logging.debug('load feature:{} and label:{}'.format(featuresPath,labelsPath))
            self.inpQ.put([features,labels])
            self.qo_hash,self.features,self.labels = self.outQ.get()
            self.tempFeatures = self.features
            self.tempLabels = self.labels
        elif self.loadDefaultModel:
            try:
                features = np.load('temp/features.npy')
                labels = np.load('temp/labels.npy')
                self.inpQ.put([features, labels])
                self.qo_hash, self.features, self.labels = self.outQ.get()
                self.tempFeatures = self.features
                self.tempLabels = self.labels
            except Exception as e:
                logging.debug('hash init by default fail ')
        else:
            return None

    def train_hash_model(self,newFeatures,newLabels):
        logging.debug('add feature and lable %s' % newLabels)
        print('add to lib :',newLabels)
        if self.inpQ.qsize()<1 and self.outQ.qsize()<2:
            if self.features is None:
                if newFeatures.shape[0] == 1:
                    self.features = np.concatenate((newFeatures, newFeatures), 0)
                    self.labels = np.append(newLabels,newLabels)
                    self.tempFeatures = newFeatures
                    self.tempLabels = newLabels
                else:
                    self.features = newFeatures
                    self.labels = newLabels
                    self.tempFeatures = newFeatures
                    self.tempLabels = newLabels
                self.inpQ.put([self.features, self.labels])
                time.sleep(0.01)
            else:
                self.tempFeatures = np.concatenate((self.tempFeatures,newFeatures),0)
                self.tempLabels = np.append(self.tempLabels,newLabels)
                self.inpQ.put([self.tempFeatures, self.tempLabels])
        else:
            self.tempFeatures = np.concatenate((self.tempFeatures, newFeatures), 0)
            self.tempLabels = np.append(self.tempLabels, newLabels)
        if self.outQ.qsize()>0 and not(self.isSearching):
            self.qo_hash,self.features,self.labels = self.outQ.get()
            aa = 1

    def hasModel(self):
        if self.outQ.qsize()>0 and not(self.isSearching):
            self.qo_hash,self.features,self.labels = self.outQ.get()
            return True
        if self.qo_hash is not None:
            return True
        else:
            return False

    def save_model(self):
        if not(os.path.exists('temp')):
            os.mkdir('temp')
        np.save('temp/features.npy',self.features)
        np.save('temp/labels.npy',self.labels)

    def hash_search(self,searchData,EdisLimit=None,cosDisLimit=None,k=3):
        self.isSearching = True
        self.hasModel()
        nameL = []
        simiL = []
        edisL = []
        indL = []
        bestMatchedFeature = None
        if self.qo_hash is None:
            self.isSearching = False
            return nameL, simiL, edisL, indL, bestMatchedFeature
        if EdisLimit is None:
            EdisLimit = self.EdisLimit
        if cosDisLimit is None:
            cosDisLimit = self.cosDisLimit
        for i in range(searchData.shape[0]):
            queryFeature = searchData[i]
            if isinstance(queryFeature,list):
                self.isSearching = False
                return nameL, simiL, edisL, indL, bestMatchedFeature
            # res_ind = self.qo_hash.find_nearest_neighbor(queryFeature)
            indList = self.qo_hash.find_k_nearest_neighbors(queryFeature,k)
            if len(indList) == 0:
                continue
            for ind in indList:
                fdis = self.cal_e_dis(self.features[ind], queryFeature)
                if fdis <= EdisLimit:
                    cosDis = self.cal_cos_dis(self.features[ind], queryFeature)
                    if cosDis > cosDisLimit:
                        if ind == indList[0]:
                            bestMatchedFeature = self.features[indList[0]]
                        nameL.append(self.labels[ind])
                        simiL.append(cosDis)
                        edisL.append(fdis)
                        indL.append(i)
        self.isSearching = False
        return nameL, simiL, edisL, indL, bestMatchedFeature

    def cal_e_dis(self,feature1, feature2):
        # 计算欧式距离
        dis = np.linalg.norm(feature1 - feature2, 2)
        return dis

    def cal_cos_dis(self,fea1, fea2):
        # 计算余弦距离
        cos_angle = (np.dot(fea1, fea2) / ((np.linalg.norm(fea1, 2) * np.linalg.norm(fea2, 2))) + 1) / 2
        return cos_angle

class TrainHashModel(threading.Thread):
    def __init__(self,inpQ,outQ):
        super(TrainHashModel,self).__init__()
        self.inpQ = inpQ
        self.outQ = outQ
        self.isTraing = False
        self.setDaemon(True)

    def hash_construct(self,features):
        dp = fc.get_default_parameters(features.shape[0], features.shape[1], fc.DistanceFunction.EuclideanSquared)
        dp.l = 20
        ds = fc.LSHIndex(dp)
        train_st = time.time()
        ds.setup(features)
        train_et = time.time()
        print ("### hash train time:%f" % (train_et - train_st))
        return ds.construct_query_object()

    def run(self):
        while True:
            # if self.outQ.qsize()<1:#由于features一般比较大，必须控制存储在内存中数据的�?            
            features, labels = self.inpQ.get()
            assert features.shape[0]==len(labels), 'Err:feature length should be the same with labels'
            self.isTraing = True
            qo_hash = self.hash_construct(features)
            res = [qo_hash,features,labels]
            self.outQ.put(res)
            self.isTraing = False

if __name__ == "__main__":
    # init.
    featuresPath='./features.npy'
    labelsPath='./label.npy'
    hashIdentifier = HashIdentifierAutoTrain(featuresPath=featuresPath,labelsPath=labelsPath)

    # add feature and label,输入都要是numpy格式数据
    newFeatures = np.random.normal(0,1,[3,10])
    newLabels = np.array(['test','aa','cc'])
    hashIdentifier.train_hash_model(newFeatures,newLabels)

    # feature query
    queryFeature = np.random.normal(0,1,[1,10])
    res = hashIdentifier.hash_search(queryFeature,EdisLimit=10,cosDisLimit=0,k=3)
    print('result',res)
    # 输出[['name1','name2','name3'],['相似度1','相似度2','相似度3'],['距离1','距离2','距离3'],['用于匹配的数据的下标1','用于匹配的数据的下标2',''],用于匹配的最优特征]
