# -*- coding: UTF-8 -*-
import numpy as np
from sklearn import svm
import math
import random
import operator
import csv
from sklearn import mixture
from sklearn.metrics import adjusted_rand_score 


'''
@author:hunter
@time:2022.01.03
'''

class GMM(object):
    def __init__(self):
        pass

    def loadDataset(self,filename, split, trainingSet, testSet):  # 加载数据集  split以某个值为界限分类train和test
        with open(filename, 'r') as csvfile:
            lines = csv.reader(csvfile)   #读取所有的行
            dataset = list(lines)     #转化成列表
            for x in range(len(dataset)-1):
                for y in range(5):
                    dataset[x][y] = float(dataset[x][y])
                 
                #if x < split*len(dataset) :   # 将所有数据加载到train和test中
                #    trainingSet.append(dataset[x])
                #else:
                #    testSet.append(dataset[x])
                        
                if random.random() < split:   # 将所有数据加载到train和test中
                    trainingSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])
   
    def Run(self):
        trainingSet = []
        testSet = []
        split = 0.7
        self.loadDataset(r'GSE72056_melanoma_single_cell_revised_v2_gene_filtering_ben3388mal1257_deg_ben_mal_top2.csv', split, trainingSet, testSet)   #数据划分
        print('Train set: ' + str(len(trainingSet)))
        print('Test set: ' + str(len(testSet)))

        x_train, y_train = np.split(trainingSet, (4, ), axis=1)
        x_test, y_test = np.split(testSet, (4, ), axis=1)

        GMM_model = mixture.GaussianMixture(n_components=2, random_state=0)
        GMM_model.fit(x_train) 
        GMM_labels = GMM_model.predict(x_test)

        y_pred = GMM_labels
        y_test = y_test.flatten()
        #print('y_pred.shape: ', y_pred)
        #print('y_train.shape: ', y_train)
               
        ARI = adjusted_rand_score(y_test,y_pred)
        print('ARI: ', ARI)
        return ARI

def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
    with open(file_name,'w',newline ='') as f:
        mywrite = csv.writer(f)
        mywrite.writerow(datas)    


if __name__ == '__main__':
    b = []
    cnt = 0
    c = 0
    for x in range(100):  
        cnt = cnt + 1
        a = GMM()
        tmp = a.Run()
        b.append(tmp)
        c = c + tmp
        
    file_name_data1 = './ARI_top20-1_gaussian_mixture_models/ARI_top2_100_times_gaussian_mixture_models.csv'
    
    data_write_csv(file_name_data1, b)   
    ari_data1 = c/cnt
    
    #求均值
    ari_mean_d1 = np.mean(b)
    #求方差
    ari_var_d1 = np.var(b)
    #求标准差
    ari_std_d1 = np.std(b,ddof=1)

    print("均值为: %f" %(ari_mean_d1))
    print("方差为: %f" %(ari_var_d1))
    print("标准差为: %f" %(ari_std_d1))

    print('Ave_ari_top20_100_times_gaussian_mixture_models: ' + repr(ari_data1))
 

