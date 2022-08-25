import torch
from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.global_attack import Random
from deeprobust.graph.targeted_attack import nettack
from deeprobust.graph.defense import GCN,GAT
from deeprobust.graph import utils
from utils import Standard
from test import GCNJaccardDefend
import numpy as np
from time import *
import copy
from Deepfool import RGCN
import math

runTimes = 0
while runTimes < 1:
    paramter = [0.0,1.0,2.0,3.0,4.0,5.0]
    dataset = 'pubmed'
    direction = "nettack"
    threshold = 0.1
    pos = 0
    while pos <= 5:
        with open('{}-{}-{}.txt'.format(dataset,direction,threshold), 'a') as f:
            f.write('{}\n'.format(paramter[pos]))
        data = Dataset(root='E:\\temp',name=dataset,setting='nettack',seed=15)
        # adj:csr_matrix features:csr_matrix labels:ndarray
        adj,features,labels = data.adj,data.features,data.labels
        trueLabels = copy.deepcopy(labels)
        idx_train,idx_val,idx_test = data.idx_train,data.idx_val,data.idx_test
        if pos == -1:
            perturbed_adj = data.adj
            perturbed_data = PrePtbDataset(root='E:\\temp',name=dataset,attack_method=direction, ptb_rate=1.0)
            target_nodes = perturbed_data.target_nodes
        else:
            perturbed_data = PrePtbDataset(root='E:\\temp',name=dataset,attack_method=direction, ptb_rate=paramter[pos])
            perturbed_adj = perturbed_data.adj
            target_nodes = perturbed_data.target_nodes
        device = torch.torch.device("cpu")
        GCNJaccardDefendList = [[],[],[],[]]
        handle_list_train = idx_train.copy()
        handle_list_test = idx_test.copy()
        init_add_size = len(idx_train)
        class_number = labels.max().item() + 1

        def average(list):
            return sum(list) / len(list)

        def std(list,avg):
            stdList = []
            for i in range(len(list)):
                stdList.append((list[i]-avg)**2)
            return math.sqrt(sum(stdList)/len(list))

        def printy(list,name):
            with open('{}-{}-{}.txt'.format(dataset,direction,threshold), 'a') as f:
                f.write("{}--accuracy:{:.4f}--std:{:.4f},precision:{:.4f},recall:{:.4f},f1:{:.4f}\n".format(name,average(list[0]),std(list[0],average(list[0])), average(list[1]),average(list[2]),average(list[3])))

        def spread(list):
            size = len(list)
            l = []
            for i in range(size):
                l.extend(list[i])
            return l
        times = 10
        for i in range(times):
            print('--------------GCNJaccardDefend-----------------')
            print('----------------{} iteration----------------'.format(i+1))

            model = GCNJaccardDefend(nfeat=features.shape[1],nclass=class_number,nhid=16,
                               device=device)
            model = model.to(device)
            low_edge1,low_edge2,handle_new_matrix = model.fit(features, perturbed_adj, labels, idx_train, idx_val, threshold=threshold)
            model.eval()
            # accuracy,pred ,oLabels,output = model.test(idx_test, trueLabels)
            accuracy, pred, oLabels, output = model.test(idx_test,labels=trueLabels,target_node=target_nodes)
            # 计算其余指标
            s = Standard(pred,oLabels)
            precision = s.precision()
            recall = s.recall()
            f1 = s.f1()
            GCNJaccardDefendList[0].append(accuracy)
            GCNJaccardDefendList[1].append(precision)
            GCNJaccardDefendList[2].append(recall)
            GCNJaccardDefendList[3].append(f1)
        printy(GCNJaccardDefendList,"GCNJaccardDefend-init")

        times = 1
        for iter in range(times):

            print('----------------{} iteration----------------'.format(iter + 1))
            print("idx_train_length:{}".format(len(idx_train)))
            print("idx_test_length:{}".format(len(idx_test)))
            model = GCNJaccardDefend(nfeat=features.shape[1],nclass=class_number,nhid=16,
                               device=device)
            model = model.to(device)
            model.fit(features, perturbed_adj, labels, idx_train, idx_val, threshold=threshold)
            model.eval()
            # accuracy,pred ,oLabels,output = model.test(idx_test, trueLabels)
            accuracy, pred, oLabels, output = model.test(idx_test,labels=trueLabels,target_node=target_nodes)

            begin_time = time()
            nGCN = RGCN(model, adj, features, labels, device)

            train_list, pos_list, sum_list, output, train_acc = \
                nGCN.order_node_by_deepfool(idx_train, idx_test, class_number, dataset, paramter[pos])
            if len(np.array(spread(train_list))) != 0:
                acc_test, p, l = utils.accuracy(output[np.array(spread(train_list))], trueLabels[np.array(spread(train_list))])
                print("correctRate:{:.4f}".format(acc_test))
                idx_train = np.concatenate((handle_list_train, np.array(spread(train_list))))
                print("idx_train_length:{}".format(len(idx_train)))

                for i in range(class_number):
                    for j, item in enumerate(train_list[i]):
                        labels[item] = i

            end_time = time()
            with open('{}-{}-{}.txt'.format(dataset,direction,threshold), 'a') as f:
                f.write('trainAccuracy:{:.4f}--psudo:{:.4f}-{:.4f}--runtime:{:.4f}\n'.format(train_acc,acc_test,len(idx_train),end_time - begin_time))


        GCNJaccardDefendList = [[],[],[],[]]
        print("idx_train_length:{}".format(len(idx_train)))
        times = 10
        for i in range(times):
            print('--------------GCNJaccardDefend-----------------')
            print('----------------{} iteration----------------'.format(i + 1))

            model = GCNJaccardDefend(nfeat=features.shape[1], nclass=class_number, nhid=16,
                                     device=device)
            model = model.to(device)
            low_edge1, low_edge2, handle_new_matrix = model.fit(features, perturbed_adj, labels, idx_train, idx_val, threshold=threshold)
            model.eval()
            accuracy, pred, oLabels, output = model.test(idx_test, labels=trueLabels,target_node=target_nodes)
            # accuracy, pred, oLabels, output = model.test(idx_test, trueLabels)
            s = Standard(pred, oLabels)
            precision = s.precision()
            recall = s.recall()
            f1 = s.f1()
            print("precision:{:.4f},recall:{:.4f},f1:{:.4f}".format(precision, recall, f1))
            GCNJaccardDefendList[0].append(accuracy)
            GCNJaccardDefendList[1].append(precision)
            GCNJaccardDefendList[2].append(recall)
            GCNJaccardDefendList[3].append(f1)
        printy(GCNJaccardDefendList, "GCNJaccardDefend")
        pos += 1
    runTimes += 1
