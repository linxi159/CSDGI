# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import math
import random
import operator
import csv
from sklearn.metrics import f1_score

'''
@author:hm
@time:2022.01.03
'''

class RF(object):
    def __init__(self):
        pass

    def loadDataset(self,filename, split, trainingSet, testSet):  # 加载数据集  split以某个值为界限分类train和test
        with open(filename, 'r') as csvfile:
            lines = csv.reader(csvfile)   #读取所有的行
            dataset = list(lines)     #转化成列表
            for x in range(len(dataset)):
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
        self.loadDataset(r'GSE75688_GEO_processed_Breast_Cancer_raw_TPM_matrix_gene_filtering_log_nontumor198tumor317_deg_nontumor_tumor_top2.csv', split, trainingSet, testSet)   #数据划分
        print('Train set: ' + str(len(trainingSet)))
        print('Test set: ' + str(len(testSet)))

        x_train, y_train = np.split(trainingSet, (4, ), axis=1)
        x_test, y_test = np.split(testSet, (4, ), axis=1)
            
        clf = RandomForestClassifier()
        clf.fit(x_train, y_train)
        #print(x_train.shape)
        acc = clf.predict(x_test) == y_test.flat
        accuracy = np.mean(acc)*100.0
        print('Accuracy: ' + repr(accuracy) + '%')
        #print (y_test)
        
        y_pred = np.squeeze(clf.predict(x_test))
        y_true = np.squeeze(y_test.flat)
        f1 = f1_score(y_true, y_pred,average = 'macro')
        
        print('f1: ' + repr(f1) + '%')
        return accuracy, f1

def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
    with open(file_name,'w',newline ='') as f:
        mywrite = csv.writer(f)
        mywrite.writerow(datas)    


if __name__ == '__main__':
    b1 = []
    b2 = []
    cnt = 0
    c1 = 0
    c2 = 0
    for x in range(100):  
        cnt = cnt + 1
        a = RF()
        tmp1,tmp2 = a.Run()
        b1.append(tmp1)
        b2.append(tmp2)
        c1 = c1 + tmp1
        c2 = c2 + tmp2
        
    file_name_data1 = './Accuracy_top20-1_rf/Accuracy_top2_100_times_rf.csv'
    file_name_data2 = './F1_top20-1_rf/F1_top2_100_times_rf.csv'
    
    data_write_csv(file_name_data1, b1)   
    data_write_csv(file_name_data2, b2)
    accuracy_data = c1/cnt
    f1_data = c2/cnt
    
    #求均值
    acc_mean_d1 = np.mean(b1)
    #求方差
    acc_var_d1 = np.var(b1)
    #求标准差
    acc_std_d1 = np.std(b1,ddof=1)
    print("均值为: %f" %(acc_mean_d1))
    print("方差为: %f" %(acc_var_d1))
    print("标准差为: %f" %(acc_std_d1))
    print('Ave_accuracy_top10_100_times_rf: ' + repr(accuracy_data) + '%')

    #求均值
    acc_mean_d2 = np.mean(b2)
    #求方差
    acc_var_d2 = np.var(b2)
    #求标准差
    acc_std_d2 = np.std(b2,ddof=1)
    print("均值为: %f" %(acc_mean_d2))
    print("方差为: %f" %(acc_var_d2))
    print("标准差为: %f" %(acc_std_d2))
    print('Ave_f1_top10_100_times_rf: ' + repr(f1_data) + '%')
 
























