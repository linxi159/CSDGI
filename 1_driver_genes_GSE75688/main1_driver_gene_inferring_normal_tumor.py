import math
import os
import time
import torch
import torch.nn as nn
import numpy as np;
from torch.autograd import Variable
import scipy
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
import pickle
import torch.nn.functional as F
from sklearn import mixture
import datetime
from torch.nn.parameter import Parameter
import torch.optim as optim
import random
import warnings
warnings.filterwarnings("ignore")

class Optim(object):
    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay = self.weight_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay = self.weight_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay = self.weight_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay = self.weight_decay)
        elif self.method == 'adamW':
            self.optimizer = optim.AdamW(self.params, lr=self.lr, weight_decay = self.weight_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, max_grad_norm, lr_decay=1, start_decay_at=None, weight_decay = 0.):
        self.params = list(params)  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self.weight_decay = weight_decay;

        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        for param in self.params:
            grad_norm += math.pow(param.grad.data.norm(), 2)

        grad_norm = math.sqrt(grad_norm)
      
        if grad_norm > 0:
            shrinkage = self.max_grad_norm / grad_norm
        else:
            shrinkage = 1.

        for param in self.params:
            if shrinkage < 1:
                param.grad.data.mul_(shrinkage)

        self.optimizer.step()
        return grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False
        self.last_ppl = ppl
        self._makeOptimizer()


class FR_Model(nn.Module):
    def __init__(self, args):
        super(FR_Model, self).__init__()
        self.pre_win = args.pre_win
        self.m = args.m
        self.p_list = (args.p_list) 
        self.len_p_list = len(args.p_list) 
        self.compress_p_list = args.compress_p_list
        self.p_allsum = np.sum(self.p_list)
        self.len_compress_p_list = len(self.compress_p_list)
        if self.len_compress_p_list>0:
            
            self.compress_p = args.compress_p_list[-1]
            self.weight = nn.Parameter(1e-10*torch.ones([self.m, self.compress_p, self.pre_win]))
        else:
            self.weight = nn.Parameter(1e-10*torch.ones([self.m, self.p_allsum, self.pre_win]))
        self.bias = nn.Parameter(1e-12*torch.ones(self.m,self.pre_win)) 
                
    def forward(self, x):
        if self.pre_win ==1:
            final_y = torch.empty(x.shape[0], self.m) 
        else :
            final_y = torch.empty(x.shape[0], self.pre_win, self.m) 
        
        for j in range(self.m):           
            if self.pre_win ==1:   
                final_y[:,j] = F.linear(x[:,j,:], self.weight[j,:].view(1, self.weight.shape[1]), self.bias[j,:]).view(-1);               
            else:
                final_y[:,:,j] = F.linear(x[:,j,:], self.weight[j,:].transpose(1,0), self.bias[j,:]);                       
        return final_y;


class Graph_Model(nn.Module):
    def __init__(self, in_features, low_rank):
        super(Graph_Model, self).__init__()
        self.in_features = in_features
        self.low_rank = low_rank
        self.weight_graph_left = Parameter((1/self.in_features)*torch.ones(low_rank, in_features))
        self.bias = Parameter(1e-12*torch.ones(in_features))    
                
    def forward(self, inputs):
        self.weight_graph_left = self.weight_graph_left.to(args.device)
        self.bias = self.bias.to(args.device)
        inputs = inputs.to(args.device)
        self.weight_graph_left.data = F.normalize(self.weight_graph_left, p=2, dim=1, eps=1e-10)
        whole_graph = torch.abs(torch.matmul(self.weight_graph_left.transpose(0,1), self.weight_graph_left))
        whole_graph = whole_graph.to(args.device)
        whole_graph = whole_graph / whole_graph.sum(1, keepdim=True)
        whole_graph = whole_graph * (1 - torch.eye(self.in_features, self.in_features).to(args.device))
        whole_graph = whole_graph.to(args.device)
        return (F.linear(inputs, whole_graph.transpose(0,1), bias = self.bias))
        

class CSDGI_EnDecoder(nn.Module):# CSDGI_EnDecoder
    def __init__(self, args):
        super(CSDGI_EnDecoder, self).__init__()
        self.m = args.m
        self.low_rank = args.low_rank
        self.w = args.window
        self.batch_size = args.batch_size
        self.scale_alpha = args.scale_alpha
        self.p_list = args.p_list
        self.p_allsum = np.sum(self.p_list)
        self.len_p_list = len(self.p_list)  
        self.len_compress_p_list = 0
        self.num_cluster = args.num_cluster
        
        self.linears = [ (nn.Linear(self.w, self.p_list[0]))]; #w->hid
        if self.len_p_list>1:
            for p_i in np.arange(1,self.len_p_list):
                self.linears.append( (nn.Linear(self.p_list[p_i-1], self.p_list[p_i], bias=True))); #w->hid
        ## graph layers
        for cluster_i in range(self.num_cluster):
            #for graph_i in range(self.num_graphs):
            self.linears.append( Graph_Model(self.m, self.low_rank)); #m->m, supervised    
            self.linears.append(FR_Model(args)); #k->k          
        self.linears = nn.ModuleList(self.linears);
        self.dropout = nn.Dropout(args.dropout);

        for p_i in np.arange(0,self.len_p_list):           
            nn.init.normal_(self.linears[p_i].weight, mean=0.0, std=1e-10)
            nn.init.normal_(self.linears[p_i].bias, mean=0.0, std=1e-10)
                
    def forward(self, inputs):  
        inputs  = inputs.to(args.device)
        x_org = inputs.clone()
        x_org = x_org.to(args.device)
        x_p = []
        x_0n = (x_org).repeat(1,1,self.p_list[0])
        x_0n = x_0n.to(args.device)
        x_0 = x_org.clone()
        x_0 = x_0.to(args.device)
        for layer_i in range(self.len_p_list):  
            x_i = self.linears[layer_i](x_0);
            x_i = F.relu(x_i + x_0n)
            x_0n = x_i
            x_0 = x_i
            x_p.append(x_i)
        
        x_p_all = torch.cat(x_p, dim = 2) 
        x_p_all = self.dropout(x_p_all)
        x_p_all = x_p_all.to(args.device)
        
        final_y_cluster = [[] for idx_class in range(self.num_cluster)] 
        for cluster_i in range(self.num_cluster):
            x_sp =  x_p_all.transpose(2,1).contiguous(); ## read the data piece  
            x_sp = x_sp.to(args.device)
            x_sp = self.linears[self.len_p_list+cluster_i*(2+self.len_compress_p_list)+0](x_sp);  #lxk 
            x_sp = F.tanh(x_sp/self.scale_alpha);
            x_sp = self.dropout(x_sp)
            
            x_sp = x_sp.transpose(2,1).contiguous(); #mxl

            x_sp = self.linears[self.len_p_list+cluster_i*(2+self.len_compress_p_list)+1+self.len_compress_p_list](x_sp); #mx2
            x_sp = x_sp.to(args.device)
            final_y_cluster[cluster_i] = (x_sp).squeeze().to(args.device)

        return final_y_cluster      

    def predict_relationship_inside(self):        
        G_all = []
        fea_weight_all = []  
        Left_all  = []
        Right_all = [] 
        Final_all = []       
        for cluster_i in range(self.num_cluster):
            Left = self.linears[self.len_p_list+cluster_i*(2+self.len_compress_p_list)+0].weight_graph_left.transpose(0,1)#.detach()
            Right = self.linears[self.len_p_list+cluster_i*(2+self.len_compress_p_list)+0].weight_graph_left
            Left = Left.to(args.device)
            Right = Right.to(args.device)
            
            A = torch.matmul(Left, Right)
            A = A.to(args.device)
            A_nodiag = torch.abs(A * (1 - torch.eye(A.shape[0], A.shape[1]).to(args.device)))
            A_nodiag = A_nodiag.to(args.device)
            G_all.append(torch.abs(A_nodiag))#.detach().numpy())
            
            Left_all.append(torch.abs(Left))
            Right_all.append(torch.abs(Right))
                       
            final_layer = (self.linears[self.len_p_list+cluster_i*(2+self.len_compress_p_list)+1+self.len_compress_p_list].weight[:,:,0])
            final_layer = final_layer.to(args.device)
            Final_all.append(final_layer)
            
            tmp = F.normalize(Left, p=2, dim=0, eps=1e-10).to(args.device)
            tmp = torch.sum(torch.abs(tmp), dim=1).to(args.device) #+ torch.sum(torch.abs(Right), dim=0)
            tmp = 1- F.normalize(tmp, p=2, dim=0, eps=1e-10).to(args.device)
            tmp = F.normalize(tmp, p=2, dim=0, eps=1e-10).to(args.device)
            fea_weight_all.append(tmp)
            
        return G_all, fea_weight_all, Left_all, Right_all, Final_all

def find_permutation(n_clusters, real_labels, labels):
    permutation=[]
    for i in range(n_clusters):
        idx = labels == i
        new_label=scipy.stats.mode(real_labels[idx])[0][0]  # Choose the most common label among data points in the cluster
        permutation.append(new_label)
    return permutation
        
    
def train(data, model, criterion, optim, batch_size):
    model.train();
    total_loss = 0;
    n_samples = 0;       
    total_time = 0
    counter = 0    
    
    for inputs in get_batches(data, batch_size, True):     
        begin_time1 = time.time()
        X, Y = inputs[0], inputs[1]
        X = X.to(args.device)
        Y = Y.to(args.device)  
        model.zero_grad();
        output = model(X);  
        
        G_all, fea_scaler, Left_all, Right_all, Final_all = model.predict_relationship_inside()
        residuals_weighted_agg_batch = torch.zeros((len(Y), len(output))).to(args.device)
        residuals_weighted_agg_batch = residuals_weighted_agg_batch.to(args.device)
        for cluster_i in range(model.num_cluster):
            residuals_tmp = criterion(output[cluster_i], Y) ### raw residuals: bxm
            residuals_tmp = residuals_tmp.to(args.device)
            weight_tmp = fea_scaler[cluster_i][None, :].repeat(len(Y),1) ### weighting vector: bxm
            weight_tmp = weight_tmp.to(args.device)
            residuals_tmp = torch.mul(residuals_tmp, weight_tmp)  ## weighting
            residuals_tmp = residuals_tmp.to(args.device)
            residuals_weighted_agg_batch[:, cluster_i] = torch.sum( residuals_tmp, dim=1).to(args.device);   ### sum_residual: b 
   
        loss_org = torch.sum(torch.min(residuals_weighted_agg_batch, dim=1)[0])   
        
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-30)  
               
        loss_org.backward(retain_graph=True)
        total_loss += loss_org.data.item();
        optim.step();
        #n_samples += output[0].size(0) * output[0].size(1); 
        n_samples += args.batch_size * args.m
        counter = counter + 1        
        total_time = total_time + time.time() - begin_time1        

    return total_loss / n_samples, total_time


def test(data, model, criterion, batch_size):
    total_loss = 0;
    n_samples = 0;       
    total_time = 0
    counter = 0
    labels_predict = []             ## predict clustering label
    residuals_weighted_AGG = []     ## record all weighted residuals aggregated (no feature side)
    residuals_raw_Full = []         ## record all weighted residuals 
    residuals_weighted_FULL = []    ## record all weighted residuals full (with feature side)
    predict_FULL = []               ## ????
    
    for inputs in get_batches(data, batch_size, False):     
        begin_time1 = time.time()
        X, Y = inputs[0], inputs[1]
        X = X.to(args.device)
        Y = Y.to(args.device)
        output = model(X);        
        G_all, fea_scaler, Left_all, Right_all, Final_all  = model.predict_relationship_inside()
        residuals_weighted_agg_batch = torch.zeros((len(Y), len(output))).to(args.device)                       ## bxc
        residuals_weighted_agg_batch = residuals_weighted_agg_batch.to(args.device)
        residuals_raw_batch = torch.zeros((len(Y), X.shape[1], len(output))).to(args.device)                    ## bxmxc
        residuals_raw_batch = residuals_raw_batch.to(args.device)
        residuals_weighted_batch =  torch.zeros((len(Y), X.shape[1], len(output))).to(args.device)              ## bxmxc
        residuals_weighted_batch = residuals_weighted_batch.to(args.device) 
        predict_FULL_batch =  torch.zeros((len(Y), X.shape[1], len(output))).to(args.device)                    ## bxmxc
        predict_FULL_batch = predict_FULL_batch.to(args.device) 
        for cluster_i in range(model.num_cluster):
            predict_FULL_batch[:, :, cluster_i] = output[cluster_i]                             ## record raw prediction: bxm(xc)
            residuals_tmp = criterion(output[cluster_i], Y)                                     ## raw residuals: bxm
            residuals_tmp = residuals_tmp.to(args.device)      
            residuals_raw_batch[:, :, cluster_i] = residuals_tmp                                ## record raw residual: bxm(xc)
            weight_tmp = fea_scaler[cluster_i][None, :].repeat(len(Y),1)                        ## weighting: bxm
            weight_tmp = weight_tmp.to(args.device)
            residuals_tmp = torch.mul(residuals_tmp, weight_tmp).to(args.device)                                ## weighting
            residuals_tmp = residuals_tmp.to(args.device)
            residuals_weighted_batch[:,:,cluster_i] = residuals_tmp                             ## record weigthed residual: bxm(xc)
            residuals_weighted_agg_batch[:, cluster_i] = torch.sum( residuals_tmp, dim=1);      ## record aggregated weighted residual: b(xc)  

        residuals_weighted_AGG.append(residuals_weighted_agg_batch)                             ## record aggregated weighted residual: bxc   
        residuals_raw_Full.append(residuals_raw_batch)                                          ## record raw residual: bxmxc
        residuals_weighted_FULL.append(residuals_weighted_batch)                                ## record weigthed residual: bxmxc
        predict_FULL.append(predict_FULL_batch)                                                 ## record raw prediction: bxmxc
    
        loss_org = torch.sum(torch.min(residuals_weighted_agg_batch, dim=1)[0])                 ## sum of min (bxc -> b -> 1)     
        labels_predict.append(torch.min(residuals_weighted_agg_batch, dim=1)[1])                ## label from min (bxc): bx1 
        
        total_loss += loss_org.data.item();
        #n_samples += (output[0].size(0) * output[0].size(1)); 
        n_samples += args.batch_size * args.m 
        counter = counter + 1
        total_time = total_time + time.time() - begin_time1
    
    labels_predict = torch.cat(labels_predict, dim = 0)                                         ## label: nx1  
    labels_predict = labels_predict.cpu()
    predict_FULL = (torch.cat(predict_FULL, dim = 0)).detach().cpu().numpy()                          ## record raw prediction: nxmxc
    residuals_raw_Full = (torch.cat(residuals_raw_Full, dim = 0)).detach().cpu().numpy()              ## record raw residual: nxmxc
    residuals_weighted_FULL = (torch.cat(residuals_weighted_FULL, dim = 0)).detach().cpu().numpy()    ## record weigthed residual: nxmxc   
    residuals_weighted_AGG = (torch.cat(residuals_weighted_AGG, dim = 0)).detach().cpu().numpy()      ## record aggregated weighted residual: nxc   
    
    residual_min_tmp = np.repeat(np.min(residuals_weighted_AGG, axis=1)[:, np.newaxis], model.num_cluster, axis=1)
    residuals_weighted_AGG = residuals_weighted_AGG - residual_min_tmp                                          
    residual_max_tmp = np.repeat(np.max(residuals_weighted_AGG, axis=1)[:, np.newaxis], model.num_cluster, axis=1)
    residuals_weighted_AGG = np.divide(residuals_weighted_AGG, residual_max_tmp)
       
    return total_loss/n_samples, total_time, labels_predict, predict_FULL, residuals_raw_Full, residuals_weighted_FULL, residuals_weighted_AGG, G_all, fea_scaler, Left_all, Right_all, Final_all

def get_batches(data, batch_size, shuffle = False):    
    inputs = data[0]
    targets = data[1]
    length = len(inputs)
    if shuffle:
        index = torch.randperm(length)
    else:
        index = torch.LongTensor(range(length))
    start_idx = 0
        
    while (start_idx < length):
        end_idx = min(length, start_idx + batch_size)
        excerpt = index[start_idx:end_idx]
        X = inputs[excerpt]; 
        Y = targets[excerpt];              
        data = [Variable(X), Variable(Y)]
        yield data;
        start_idx += batch_size

if __name__ == '__main__':
    
    # hyper - parameter
    class args:
        scale_alpha = 1
        window = 1
        pre_win = 1
        low_rank = 1

        p_list = [10]*10
        L1Loss = False#False
        clip = 1.
        epochs = 1000 # 100
        batch_size = 100
        dropout = 0.00001
        seed = 12345
        gpu = 0
        optim = 'adamW'#'adam'
        lr = 1e-3#1e-7
        weight_decay = 5e-5#lr/10.
        horizon = 1

        random_shuffle = True
        train = True # True
        test = False
        disable_cuda = True
        random_seed = 123
        device = None
        save_path = './result'

    args = args
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(args.random_seed)
    else:
        args.device = torch.device('cpu')

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False,reduce=False);
    else:
        criterion = nn.MSELoss(size_average=True,reduce=False); #SmoothL1Loss    

    best_val = 100000;
    print('buliding model')
    
    print(args.lr)

    filename   = './data/GSE75688_GEO_processed_Breast_Cancer_raw_TPM_matrix_gene_filtering_log_nontumor198tumor317_deg.csv'
 
    with open(filename,encoding = 'utf-8') as f:
        data_org = np.loadtxt(f,str,delimiter = ",")
    truth_labels = np.array(data_org[1,1:],dtype='uint8')
    data_org = data_org[2:,1:].T
    data_org = data_org.astype(np.float)#.toarray()
    
    args.m = data_org.shape[1]
    args.num_cluster = 2 
    
    truth_labels = truth_labels.squeeze()

    model = CSDGI_EnDecoder(args);
    model = model.to(args.device)
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, weight_decay = args.weight_decay,
        )

    print(model.linears)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    data_org_tensor = torch.from_numpy(data_org).float()
    train_data_org = [ data_org_tensor[:, :, None],  data_org_tensor]
    train_data = train_data_org.copy()
    test_data = [ data_org_tensor[:, :, None],  data_org_tensor]

    print('data org shape: ', data_org_tensor.shape)

    train_loss_all = []
    test_loss_all = []
    NMI_all = []
    ACC_all = []        

    test_truth_labels = truth_labels

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.train == True:
        print ("1-train")
        for epoch in range(args.epochs): 
            train_loss, epoch_time = train(train_data, model, criterion, optim, args.batch_size)  
            torch.save(model.state_dict(), os.path.join(args.save_path, 'epoch_%d_num_cluster_%d_best.model' % (epoch, args.num_cluster)))
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best.model'))
    
            test_loss, epoch_time, labels_predict, predict_FULL, residuals_raw_Full, residuals_weighted_FULL, residuals_weighted_AGG, G_all, fea_scaler, Left_all, Right_all, Final_all = test(test_data, model, criterion, args.batch_size)
            labels_predict = labels_predict.detach().numpy() 
    
            CSDGI_EnDecoder_NMI = normalized_mutual_info_score(test_truth_labels, labels_predict)    
            if len(np.unique(labels_predict))>=args.num_cluster:
                permutation = find_permutation(args.num_cluster, test_truth_labels, labels_predict)
                new_labels = [ permutation[label] for label in labels_predict]   # permute the labels
                CSDGI_EnDecoder_ACC = accuracy_score(test_truth_labels, new_labels)   
                permutation_reorder = np.argsort(permutation)
            else:
                CSDGI_EnDecoder_ACC = -1
                permutation_reorder = range(args.num_cluster)
    
            now = datetime.datetime.now()
            print ('Epoch: ', epoch, ' ', now.strftime("%Y-%m-%d %H:%M:%S"))
            print('CSDGI_EnDecoder_NMI:', "{:5.3f}".format(CSDGI_EnDecoder_NMI), ' CSDGI_EnDecoder_ACC:', "{:5.3f}".format(CSDGI_EnDecoder_ACC), ' trn_loss:', "{:5.7f}".format(train_loss), ' tst_loss:', "{:5.7f}".format(test_loss))
            train_loss_all.append(train_loss) 
            test_loss_all.append(test_loss)  
            NMI_all.append(CSDGI_EnDecoder_NMI)
            ACC_all.append(CSDGI_EnDecoder_ACC)        
    else:
        print ("2-test")
        model.load_state_dict(torch.load(os.path.join(args.save_path, "best.model")))
        
        batch_size = 100
        test_loss, epoch_time, labels_predict, predict_FULL, residuals_raw_Full, residuals_weighted_FULL, residuals_weighted_AGG, G_all, fea_scaler, Left_all, Right_all, Final_all = test(test_data, model, criterion, batch_size)
        labels_predict = labels_predict.detach().numpy()   
        print(labels_predict)
        
        labels_predict_ = labels_predict + 1
        print(labels_predict_)
        print(test_truth_labels)
        
        CSDGI_EnDecoder_NMI = normalized_mutual_info_score(test_truth_labels, labels_predict)    
        if len(np.unique(labels_predict))>=args.num_cluster:
            permutation = find_permutation(args.num_cluster, test_truth_labels, labels_predict)
            new_labels = [ permutation[label] for label in labels_predict]   # permute the labels
            CSDGI_EnDecoder_ACC = accuracy_score(test_truth_labels, new_labels)   
            permutation_reorder = np.argsort(permutation)
        else:
            CSDGI_EnDecoder_ACC = -1
            permutation_reorder = range(args.num_cluster)
    
        now = datetime.datetime.now()

        print('CSDGI_EnDecoder_NMI:', "{:5.3f}".format(CSDGI_EnDecoder_NMI), ' CSDGI_EnDecoder_ACC:', "{:5.3f}".format(CSDGI_EnDecoder_ACC))
        NMI_all.append(CSDGI_EnDecoder_NMI)
        ACC_all.append(CSDGI_EnDecoder_ACC)        
                       
        print('fea_scaler[0]: ', fea_scaler[0].detach().cpu().numpy())
        print('fea_scaler[0]_sum: ', np.sum(fea_scaler[0].detach().cpu().numpy()))
        print('fea_scaler[0].shape: ', fea_scaler[0].shape)
        fea_weight_0 = fea_scaler[0].detach().cpu().numpy()
        np.savetxt('./result/feature_weight_0_n2_label.csv', fea_weight_0, delimiter = ',')
 
        print('fea_scaler[1]: ', fea_scaler[1].detach().cpu().numpy())
        print('fea_scaler[1]_sum: ', np.sum(fea_scaler[1].detach().cpu().numpy()))
        print('fea_scaler[1].shape: ', fea_scaler[1].shape)
        fea_weight_1 = fea_scaler[1].detach().cpu().numpy()
        np.savetxt('./result/feature_weight_1_n2_label.csv', fea_weight_1, delimiter = ',')
        
        print('fea_scaler: ', fea_scaler)
             
        print('Left_all[0]: ', Left_all[0].detach().cpu().numpy())
        print('Left_all[0]_sum: ', np.sum(Left_all[0].detach().cpu().numpy()))
        print('Left_all[0].shape: ', Left_all[0].shape)
        left_0 = Left_all[0].detach().cpu().numpy()
        np.savetxt('./result/left_0_n2_label.csv', left_0, delimiter = ',')
       
        print('Left_all[1]: ', Left_all[1].detach().cpu().numpy())
        print('Left_all[1]_sum: ', np.sum(Left_all[1].detach().cpu().numpy()))
        print('Left_all[1].shape: ', Left_all[1].shape)
        left_1 = Left_all[1].detach().cpu().numpy()
        np.savetxt('./result/left_1_n2_label.csv', left_1, delimiter = ',')
       
        print('Left_all: ', Left_all)        
        
        np.savetxt('./result/labels_predict_n2_label.csv', labels_predict_, delimiter = ',')
        
        print(' tst_loss:', "{:5.7f}".format(test_loss))
        test_loss_all.append(test_loss)  
    


        
