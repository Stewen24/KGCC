import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from typing import List, Dict
from sklearn import metrics
from torch.nn import init


class feature_extractor(nn.Module):
    def __init__(self, hidden_1, hidden_2):
        super(feature_extractor, self).__init__()
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 2 * 2, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


class discriminator(nn.Module):
    def __init__(self, hidden_1):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(hidden_1, hidden_1)
        self.fc2 = nn.Linear(hidden_1, 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Domain_adaption_model(nn.Module):
    def __init__(self,hidden_1=32,hidden_2=32,hidden_3=32,hidden_4=32,num_of_class=3,low_rank=32,max_iter=1000,upper_threshold=0.9,lower_threshold=0.5, device="cuda"):
        super(Domain_adaption_model,self).__init__()
        self.fea_extrator_f= feature_extractor(hidden_1,hidden_2)
        self.fea_extrator_g= feature_extractor(hidden_3,hidden_4)
        self.U=nn.Parameter(torch.randn(low_rank,hidden_2),requires_grad=True)
        self.V=nn.Parameter(torch.randn(low_rank,hidden_4),requires_grad=True)
        self.P=torch.randn(num_of_class,hidden_4)
        self.stored_mat=torch.matmul(self.V,self.P.T)
        self.max_iter=max_iter
        self.upper_threshold=upper_threshold
        self.lower_threshold=lower_threshold
 #       self.diff=(upper_threshold-lower_threshold)
        self.threshold=upper_threshold
        self.cluster_label=np.zeros(num_of_class)
        self.num_of_class=num_of_class
        self.device=device
    def forward(self,source,target,source_label):
        feature_source_f=self.fea_extrator_f(source)
        feature_target_f=self.fea_extrator_f(target)
        feature_source_g=self.fea_extrator_f(source)
        ## Update P through some algebra computations for the convenice of broadcast
        self.P=torch.matmul(
            torch.inverse(torch.diag(source_label.sum(axis=0))+torch.eye(self.num_of_class).to(self.device)),
            torch.matmul(source_label.T,feature_source_g)
        )
        self.stored_mat=torch.matmul(self.V,self.P.T)
        source_predict=torch.matmul(torch.matmul(self.U,feature_source_f.T).T,self.stored_mat)
        target_predict=torch.matmul(torch.matmul(self.U,feature_target_f.T).T,self.stored_mat)
        source_label_feature=torch.nn.functional.softmax(source_predict, dim=1)
        target_label_feature=torch.nn.functional.softmax(target_predict, dim=1)
        ## DAC part
        sim_matrix=self.get_cos_similarity_distance(source_label_feature)
        sim_matrix_target=self.get_cos_similarity_distance(target_label_feature)
        return source_predict,feature_source_f,feature_target_f,sim_matrix,sim_matrix_target
    def target_domain_evaluation(self,test_features,test_labels):
        self.eval()
        feature_target_f=self.fea_extrator_f(test_features)
        test_logit=torch.matmul(torch.matmul(self.U,feature_target_f.T).T,self.stored_mat.to(self.device))
        test_cluster=torch.nn.functional.softmax(test_logit, dim=1)
        test_cluster=np.argmax(test_cluster.cpu().detach().numpy(),axis=1)
        test_labels=np.argmax(test_labels.cpu().detach().numpy(),axis=1)
        test_predict=np.zeros_like(test_labels)
        for i in range(len(self.cluster_label)):
            cluster_index=np.where(test_cluster==i)[0]
            test_predict[cluster_index]=self.cluster_label[i]
 #       acc=np.sum(label_smooth(test_predict)==test_labels)/len(test_predict)
        acc=np.sum(test_predict==test_labels)/len(test_predict)
        nmi=metrics.normalized_mutual_info_score(test_predict,test_labels)
        return acc,nmi   
    def cluster_label_update(self,source_features,source_labels):
        self.eval()
        feature_target_f=self.fea_extrator_f(source_features)
        source_logit=torch.matmul(torch.matmul(self.U,feature_target_f.T).T,self.stored_mat.to(self.device))
        source_cluster=np.argmax(torch.nn.functional.softmax(source_logit, dim=1).cpu().detach().numpy(),axis=1)
        source_labels=np.argmax(source_labels.cpu().detach().numpy(),axis=1)
        for i in range(len(self.cluster_label)):
            samples_in_cluster_index=np.where(source_cluster==i)[0]
            label_for_samples=source_labels[samples_in_cluster_index]
            if len(label_for_samples)==0:
               self.cluster_label[i]=0
            else:
               label_for_current_cluster=np.argmax(np.bincount(label_for_samples))
               self.cluster_label[i]=label_for_current_cluster
        source_predict=np.zeros_like(source_labels)
        for i in range(len(self.cluster_label)):
            cluster_index=np.where(source_cluster==i)[0]
            source_predict[cluster_index]=self.cluster_label[i]
        acc=np.sum(source_predict==source_labels)/len(source_predict)
        nmi=metrics.normalized_mutual_info_score(source_predict,source_labels)
        return acc,nmi
    def predict(self,target):
        with torch.no_grad():
            self.eval()         
            feature_target_f=self.fea_extrator_f(target)
            test_logit=torch.matmul(torch.matmul(self.U,feature_target_f.T).T,self.stored_mat.to(self.device))/8
            test_cluster=torch.nn.functional.softmax(test_logit, dim=1)
            test_cluster=np.argmax(test_cluster.cpu().detach().numpy(),axis=1)
            cluster_0_index,cluster_1_index,cluster_2_index=np.where(test_cluster==0)[0],np.where(test_cluster==1)[0],np.where(test_cluster==2)[0]
            test_cluster[cluster_0_index]=self.cluster_label[0]
            test_cluster[cluster_1_index]=self.cluster_label[1]
            test_cluster[cluster_2_index]=self.cluster_label[2]
        return test_cluster
    def get_cos_similarity_distance(self, features):
         """Get distance in cosine similarity
         :param features: features of samples, (batch_size, num_clusters)
         :return: distance matrix between features, (batch_size, batch_size)
         """
         # (batch_size, num_clusters)
         features_norm = torch.norm(features, dim=1, keepdim=True)
         # (batch_size, num_clusters)
         features = features / features_norm
         # (batch_size, batch_size)
         cos_dist_matrix = torch.mm(features, features.transpose(0, 1))
         return cos_dist_matrix
    def get_cos_similarity_by_threshold(self, cos_dist_matrix):
         """Get similarity by threshold
         :param cos_dist_matrix: cosine distance in matrix,
         (batch_size, batch_size)
         :param threshold: threshold, scalar
         :return: distance matrix between features, (batch_size, batch_size)
         """
         device = cos_dist_matrix.device
         dtype = cos_dist_matrix.dtype
         similar = torch.tensor(1, dtype=dtype, device=device)
         dissimilar = torch.tensor(0, dtype=dtype, device=device)
         sim_matrix = torch.where(cos_dist_matrix > self.threshold, similar,
                                  dissimilar)
         return sim_matrix
    def compute_indicator(self,cos_dist_matrix):
        device = cos_dist_matrix.device
        dtype = cos_dist_matrix.dtype
        selected = torch.tensor(1, dtype=dtype, device=device)
        not_selected = torch.tensor(0, dtype=dtype, device=device)
        w2=torch.where(cos_dist_matrix < self.lower_threshold,selected,not_selected)
        w1=torch.where(cos_dist_matrix > self.upper_threshold,selected,not_selected)
        w = w1 + w2
        nb_selected=torch.sum(w)
        return w,nb_selected
    def update_threshold(self, epoch: int):
         """Update threshold
         :param threshold: scalar
         :param epoch: scalar
         :return: new_threshold: scalar
         """
         n_epochs = self.max_iter
         diff = self.upper_threshold - self.lower_threshold
         eta = diff / n_epochs
         # First epoch doesn't update threshold
         if epoch != 0:
             self.upper_threshold = self.upper_threshold-eta
             self.lower_threshold = self.lower_threshold+eta
         else:
             self.upper_threshold = self.upper_threshold
             self.lower_threshold = self.lower_threshold
         self.threshold=(self.upper_threshold+self.lower_threshold)/2

def weight_init(m):  ## model parameter intialization
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()    
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.03)
#        torch.nn.init.kaiming_normal_(m.weight.data,a=0,mode='fan_in',nonlinearity='relu')
        m.bias.data.zero_()
