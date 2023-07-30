# -*- coding: UTF-8 -*-
import numpy as np
import math
import random
import operator
import csv
from array import array
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from minepy import MINE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
#from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans 
from sklearn import mixture
from sklearn.metrics import adjusted_rand_score 
from sklearn.metrics import f1_score

'''
@author:huang
@time:2022.01.03
'''

class FS_cla(object):
    def __init__(self,n_features):
        self.n_features = n_features
        self.result = [] 
        self.target = []

    def loadDataset(self,filename, data, target):  # 加载数据集  split以某个值为界限分类train和test
        with open(filename, 'r') as csvfile:
            lines = csv.reader(csvfile)   #读取所有的行
            dataset = list(lines)     #转化成列表
            dat = []
            for x in range(len(dataset)):
                for y in range(1049):
                    dataset[x][y] = float(dataset[x][y])
                
                dat.append(dataset[x])
                
            dat = np.array(dat)
            data, target = np.split(dat, (1048, ), axis=1)            
            data[data == ''] = 0.0
            #data = data.astype(np.float)
            target[target == ''] = 0.0
            #target = target.astype(np.float)
        
        return data, target
    
    def loadDataset_(self, data, target, split):  # 加载数据集  split以某个值为界限分类train和test
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for x in range(len(data)):                 
            #if x < split*len(data):   # 将所有有序数据加载到train和test中
            #    x_train.append(data[x])
            #    y_train.append(target[x])
            #else:
            #    x_test.append(data[x])
            #    y_test.append(target[x])
            
            if random.random() < split :   # 将所有数据随机加载到train和test中
                x_train.append(data[x])
                y_train.append(target[x])
            else:
                x_test.append(data[x])
                y_test.append(target[x])
                 
        return x_train, y_train, x_test, y_test
       
    def Run(self):
        k_fea = self.n_features
        data = []
        target = []
        data, target = self.loadDataset(r'GSE76312_Single_cell_HSCs_expression_RPKM_table_gene_filtering_log_normal232tumor902_deg1048.csv', data, target)   #数据划分
        print('data set: ' + str(len(data)))
        print('target set: ' + str(len(target)))

        #[1]方差选择法，返回值为特征选择后的数据
        #参数threshold为方差的阈值
        #result1 = VarianceThreshold(threshold=3).fit_transform(data)

        #[2]相关系数法,选择K个最好的特征，返回选择特征后的数据
        #第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，
        #输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
        #参数k为选择的特征个数
        #result2 = SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(data, target)

        #[3][可用]卡方检验,选择K个最好的特征，返回选择特征后的数据
        #result = SelectKBest(chi2, k=k_fea).fit_transform(data, target)
        
        #[4]互信息法,由于MINE的设计不是函数式的，定义mic方法将其为函数式的，
        #返回一个二元组，二元组的第2项设置成固定的P值0.5
        #def mic(x, y):
        #    m = MINE()
        #    m.compute_score(x, y)
        #    return (m.mic(), 0.5)
 
        #选择K个最好的特征，返回特征选择后的数据
        #result4 = SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(data, target)
        #result = mutual_info_classif(data, target)

        #[5][可用]递归特征消除法，返回特征选择后的数据
        #参数estimator为基模型
        #参数n_features_to_select为选择的特征个数
        result = RFE(estimator=LogisticRegression(), n_features_to_select=k_fea).fit_transform(data, target)

        #[6][可用]基于惩罚项的特征选择法,带L1惩罚项的逻辑回归作为基模型的特征选择
        #result = SelectFromModel(LogisticRegression(penalty="l2", C=0.1),max_features=2).fit_transform(data, target)

        #[7][可用]基于树模型的特征选择法,GBDT作为基模型的特征选择
        #result = SelectFromModel(GradientBoostingClassifier(),max_features=2).fit_transform(data, target)
        
        #results=np.matrix(results)
      
        #print('result set: ' + str(len(result)))
        
        self.result = result
        self.target = target
        

    def cla(self):
                
        split = 0.7#0.604#
        x_train, y_train, x_test, y_test = self.loadDataset_(self.result, self.target, split)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
 
        print('Train set: ', x_train.shape)
        print('Test set: ' , x_test.shape)
   
        #SVM         
        clf = svm.SVC(C=0.50, kernel='linear', decision_function_shape='ovr')
        clf.fit(x_train, y_train, sample_weight=None)
        
        #RF
        #clf = RandomForestClassifier()
        #clf.fit(x_train, y_train)
        """          
        # accuracy
        acc = clf.predict(x_test) == y_test.flat
        accuracy = np.mean(acc)*100.0
        #print('Accuracy: ' + repr(accuracy) + '%')
        
        print('y_test: ', y_test)
        return accuracy
        """
        # F1
        y_pred = np.squeeze(clf.predict(x_test))
        y_true = np.squeeze(y_test.flat)
        f1 = f1_score(y_true, y_pred,average = 'macro')
        
        print('f1: ' + repr(f1) + '%')
        return f1
        
    
    def clustering(self):
                    
        split = 0.7#
        x_train, y_train, x_test, y_test = self.loadDataset_(self.result, self.target, split)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
 
        print('Train set: ', x_train.shape)
        print('Test set: ' , x_test.shape)
    
        #x_train y_train, ARI, k-means and gaussian_mixture_models           
        #从sklearn导入KMeans，初始化模型，并设置聚类中心数量为2
        #kmeans = KMeans(n_clusters=2) 
        #kmeans.fit(x_train) 
        #判断每个样本的聚类中心 
        #y_pred = kmeans.predict(x_test) 
        
        #GMM             
        GMM_model = mixture.GaussianMixture(n_components=2, random_state=0)
        GMM_model.fit(x_train) 
        GMM_labels = GMM_model.predict(x_test)
        y_pred = GMM_labels
        
        y_test = y_test.flatten()
        y_test = y_test - 1
        #使用ARI进行K-means聚类性能评估(每个数据都有所属的类别) 
        ARI = adjusted_rand_score(y_test,y_pred)
        print('ARI: ', ARI)

        return ARI    
        
   
def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
    with open(file_name,'w',newline ='') as f:
        mywrite = csv.writer(f)
        mywrite.writerow(datas)    


if __name__ == '__main__':
    ave_ = []
    var_ = []
    std_ = []    
    for i in range(2,21,2):
        b=[]
        cnt = 0
        c = 0
        a = FS_cla(n_features = i)
        a.Run()
        for x in range(100):  
            cnt = cnt + 1
            #tmp = a.cla()
            tmp = a.clustering()
            b.append(tmp)
            c = c + tmp
            
        accuracy_data = c/cnt
        ave_.append(accuracy_data)
               
        #print('Ave_Accuracy_RFE_f%s_svm_100_times: '%(i) + repr(accuracy_data) + '%')
        #print('Ave_F1_chi2_f%s_svm_100_times: '%(i) + repr(accuracy_data) + '%')
        print('Ave_Ari_RFE_f%s_GMM_100_times: ' %(i) + repr(accuracy_data))
    
        #求均值
        acc_mean = np.mean(b)
        #d.append(acc_mean_d1)
        #求方差
        acc_var = np.var(b)
        #求标准差
        acc_std = np.std(b,ddof=1)

        var_.append(acc_var)
        std_.append(acc_std)
        
        print("平均值为：%f" % acc_mean)
        print("方差为：%f" % acc_var)
        print("标准差为:%f" % acc_std)

    #file_name_data1 = 'Ave_Accuracy_chi2_f1-20_svm_100_times.csv'
    #file_name_data2 = 'Var_Accuracy_chi2_f1-20_svm_100_times.csv'
    #file_name_data3 = 'Std_Accuracy_chi2_f1-20_svm_100_times.csv'
    #file_name_data1 = 'Ave_F1_chi2_f1-20_svm_100_times.csv'
    #file_name_data2 = 'Var_F1_chi2_f1-20_svm_100_times.csv'
    #file_name_data3 = 'Std_F1_chi2_f1-20_svm_100_times.csv'
    file_name_data1 = 'Ave_Ari_RFE_f1-20_GMM_100_times.csv'
    file_name_data2 = 'Var_Ari_RFE_f1-20_GMM_100_times.csv'
    file_name_data3 = 'Std_Ari_RFE_f1-20_GMM_100_times.csv'
    data_write_csv(file_name_data1, ave_)
    data_write_csv(file_name_data2, var_)
    data_write_csv(file_name_data3, std_)
    
    
    