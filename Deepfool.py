from time import sleep
import numpy as np
import torch
import torch.nn.functional as F
import math
import torch.optim as optim
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN
from torch.autograd import Variable
import copy
from scipy.sparse import csr_matrix

def transform_csr_to_tensor(adj):
    adj = adj.A
    return torch.from_numpy(adj.astype(np.float32))

class RGCN:
    def __init__(self, model,adj,features,labels,device):
        self.features = transform_csr_to_tensor(features)
        self.labels = labels
        self.model = model
        self.model.to(device)
        self.adj = transform_csr_to_tensor(adj)
        # train_iters
        self.output = []

    def calculate_grad(self, pert_adj, idx_train, idx_test, num_classes):
        x = Variable(pert_adj, requires_grad=True)
        output = self.model(self.features, x)
        self.output = output
        acc_test, p, l = utils.accuracy(output[idx_train], self.labels[idx_train])
        grad = [[],[]]
        train_length = len(idx_train)
        test_length = len(idx_test)

        for i, idx in enumerate(idx_train):
            grad[0].append([])
            progress = i / train_length * 100
            print('train gradient::{:.2f}%!!!'.format(progress))
            for i in range(num_classes):
                output[idx:idx+1][0][i].backward(retain_graph=True)
                grad[0][-1].append(copy.deepcopy(x.grad[idx].cpu().numpy()))

        for i, idx in enumerate(idx_test):
            grad[1].append([])
            progress = i / test_length * 100
            print('test gradient::{:.2f}%!!!'.format(progress))
            for i in range(num_classes):
                cls = torch.LongTensor(np.array(i).reshape(1)).cpu()
                loss = F.nll_loss(output[idx:idx+1], cls)
                loss.backward(retain_graph=True)
                grad[1][-1].append(copy.deepcopy(x.grad[idx].cpu().numpy()))
        return np.array(grad),acc_test

    def order_node_by_deepfool(self, idx_train, idx_test, num_classes, dataname, ptb_rate):
        def takeSecond(elem):
            return elem['distance']
        list = [[],[]]
        train_list = []
        pos_list = []
        standard = []
        distance_list = [[],[]]

        for i in range(num_classes):
            list[0].append([])
            list[1].append([])
            train_list.append([])
            pos_list.append([])
            distance_list[0].append([])
            distance_list[1].append([])
            standard.append([])
        try:
            gradients = np.load('{}_gradient_{}.npy'.format(dataname,ptb_rate))
        except:
            gradients,train_acc = self.calculate_grad(self.adj, idx_train, idx_test, num_classes)
            np.save('{}_gradient_{}'.format(dataname, ptb_rate), gradients)

        try:
            list = np.load('{}_sum_list_{}.npy'.format(dataname,ptb_rate))
        except:
            for i, ele in enumerate(idx_train):
                distance, label, confident = self.deepfool(i, ele, gradients, num_classes)
                list[0][label].append({'node': ele, 'distance': distance, 'pos': i, 'label': label, 'trueLabel': self.labels[ele],
                                    'isCorrect': True if label == self.labels[ele] else False, 'confident': confident})
            for i, ele in enumerate(idx_test):
                distance, label, confident = self.deepfool(i, ele, gradients, num_classes,isTrain=False)
                list[1][label].append({'node': ele, 'distance': distance, 'pos': i,  'label': label,'trueLabel': self.labels[ele],
                                    'isCorrect': True if label == self.labels[ele] else False, 'confident': confident})
            np.save('{}_sum_list_{}'.format(dataname, ptb_rate), list)
        for i in range(num_classes):
            list[0][i].sort(key=takeSecond, reverse=True)
            for ele in list[0][i]:
                distance_list[0][i].append(ele['distance'])
            standard[i] = distance_list[0][i][-1]
        for i in range(num_classes):
            list[1][i].sort(key=takeSecond, reverse=True)
            for ele in list[1][i]:
                train_list[i].append(ele['node'])
                pos_list[i].append(ele['pos'])
                distance_list[1][i].append(ele['distance'])
        for i in range(num_classes):
            for j, ele in enumerate(distance_list[1][i]):
                if ele <= standard[i]:
                    train_list[i] = train_list[i][0:j]
                    pos_list[i] = pos_list[i][0:j]
                    distance_list[1][i] = distance_list[1][i][0:j]
                    break

        return np.array(train_list),np.array(pos_list), np.array(list), self.output,train_acc

    def order_node_by_accuracy(self, idx_test, output, labels, standard=0.99):
        train_list = []
        pos_list = []
        sum_list = []

        for i, ele in enumerate(idx_test):
            confident = output[ele].max()
            label = labels[i]
            if confident > math.log(standard,2):
                pos_list.append(i)
                train_list.append(ele)
                sum_list.append({'node': ele, 'pos': i,  'label': label,'trueLabel': self.labels[ele],
                                    'isCorrect': True if label == self.labels[ele] else False, 'confident': confident})

        return np.array(train_list), np.array(pos_list), np.array(sum_list)

    def deepfool(self, idx , ele, gradients, num_classes, isTrain = True):
        # innormal_adj: the perturbed adjacency matrix not normalized
        # ori_adj: the normalized perturbed adjacency matrix
        baseIndex = 0
        if not isTrain:
            baseIndex = 1

        self.model.eval()
        pred = self.output[ele]
        pred = pred.detach().cpu().numpy()

        I = pred.argsort()[::-1]
        I = I[0:num_classes]
        label = I[0]
        f_i = np.array(pred).flatten()

        pert = np.inf
        for i in range(num_classes):
            # set new w_k and new f_k
            if i != label:
                w_k = gradients[baseIndex][idx][i] - gradients[baseIndex][idx][label]
                f_k = f_i[i] - f_i[label]
                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
                if pert_k == np.nan:
                    pert_k = np.inf
                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
        return pert, label, pred[label]