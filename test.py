import torch
from deeprobust.graph import utils
from mGCN import GCN
import scipy.sparse as sp
import numpy as np



class GCNJaccardDefend(GCN):

    def __init__(self, nfeat, nhid, nclass, binary_feature=True, dropout=0.5, lr=0.01, weight_decay=5e-4,
                 with_relu=True, with_bias=True, device='cpu'):

        super(GCNJaccardDefend, self).__init__(nfeat, nhid, nclass, dropout, lr, weight_decay, with_relu, with_bias,
                                               device=device)
        self.device = device
        self.binary_feature = binary_feature
        self.recordRate1 = [0,0,0,0,0,0,0,0,0,0,0]
        self.recordRate2 = [0,0,0,0,0,0,0,0,0,0,0]

    def fit(self, features, adj, labels, idx_train, idx_val=None, threshold=0.05, train_iters=200, pertubed_num=3,
            initialize=True, verbose=True, **kwargs):

        self.threshold = threshold
        low_edge1,low_edge2,jaccard_matrix = self.drop_dissimilar_edges(features, adj)
        features, pertubed_adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        self.pertubed_adj = pertubed_adj
        self.features = features
        self.labels = labels
        super().fit(features, pertubed_adj, labels, idx_train, jaccard_matrix, idx_val,train_iters=train_iters, initialize=initialize,
                    verbose=verbose)
        return low_edge1,low_edge2,jaccard_matrix

    def drop_dissimilar_edges(self, features, adj,  metric='similarity'):
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)

        adj_triu = sp.triu(adj, format='csr')
        low_edge1,low_edge2,jaccard_matrix = self.dropedge_second_jaccard(adj_triu.data, adj_triu.indptr, adj_triu.indices, features,threshold=self.threshold)
        print('Jaccard{},{}'.format(self.recordRate1,self.recordRate2))
        return low_edge1,low_edge2,jaccard_matrix


    def dropedge_second_jaccard(self,A, iA, jA, features, threshold):
        size = len(iA) - 1
        low_edge1 = 0
        low_edge2 = 0

        jaccard_matrix = torch.ones((size,size))
        temp_matrix = torch.zeros((size, size))
        # one Hop
        for row in range(len(iA) - 1):
            for i in range(iA[row], iA[row + 1]):
                if A[i] == 0:
                    continue
                n1 = row
                n2 = jA[i]
                a, b = features[n1], features[n2]
                intersection = a.multiply(b).count_nonzero()
                J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
                if A[i] == 1:
                    self.recordRate1[int(J*10)] += 1
                if J <= threshold and A[i] == 1 and jaccard_matrix[n1][n2].item() != J:
                    jaccard_matrix[n1][n2] = J
                    jaccard_matrix[n2][n1] = J
                    low_edge1 += 1
        # Two Hop
        for row in range(len(iA) - 1):
            for i in range(iA[row], iA[row + 1]):
                if A[i] == 0:
                    continue
                n1 = row
                n2 = jA[i]
                if jaccard_matrix[n1][n2] != 1:
                    continue
                a, b = features[n1], features[n2]
                intersection = a.multiply(b).count_nonzero()
                J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
                M11 = intersection
                M10 = a.count_nonzero() - intersection
                M01 = b.count_nonzero() - intersection
                low_edge2 += self.second_jaccard(iA, jA, features,J, n1, n2,threshold,jaccard_matrix, M11, M10, M01)
        return low_edge1,low_edge2,jaccard_matrix

    def second_jaccard_calc(self,iA, jA, features, c, d,matrix):
        low_edge = 0
        num = len(range(iA[c], iA[c + 1]))+len(range(iA[d], iA[d + 1]))-2
        if num > 0:
            for i in range(iA[c], iA[c + 1]):
                n1 = d
                n2 = jA[i]
                if n2 == n1 or matrix[n1][n2] == 1:
                    continue
                a, b = features[n1], features[n2]
                intersection = a.multiply(b).count_nonzero()
                J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
                self.recordRate2[int(J * 10)] += 1
                matrix[n1][n2] = matrix[n2][n1] = 1
            for j in range(iA[d], iA[d + 1]):
                n1 = c
                n2 = jA[j]
                if n2 == n1 or matrix[n1][n2] == 1:
                    continue
                a, b = features[n1], features[n2]
                intersection = a.multiply(b).count_nonzero()
                J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
                self.recordRate2[int(J * 10)] += 1
                matrix[n1][n2] = matrix[n2][n1] = 1
        return low_edge

    def second_jaccard(self,iA, jA, features,J, c, d, threshold,jaccard_matrix,M11,M10,M01):
        low_edge = 0
        num = len(range(iA[c], iA[c + 1])) + len(range(iA[d], iA[d + 1]))-2
        m = c
        J1 = J
        if num > 0:
            for i in range(iA[c], iA[c + 1]):
                n1 = d
                n2 = jA[i]
                if n2 == n1 or jaccard_matrix[c][n2] !=1:
                    continue

                a, b = features[c], features[n2]
                intersection = a.multiply(b).count_nonzero()
                nM11 = intersection
                nM01 = b.count_nonzero() - intersection
                mM11 = (min(nM11, M11) + min(nM01, M01))
                mSum = abs(nM11 - M11) + abs(nM01-M01) + mM11
                # 计算上界
                if mSum == 0:
                    overLimit = 0
                else:
                    overLimit = mM11 / mSum
                threshold2 = threshold * overLimit
                J2 = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
                a, b = features[n1], features[n2]
                intersection = a.multiply(b).count_nonzero()
                J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
                if J <= threshold2:
                    if J1 < J2:
                        n = d
                    else:
                        n = n2

                    if jaccard_matrix[m][n].item() != 1:
                        jaccard_matrix[m][n] = jaccard_matrix[m][n] if J > jaccard_matrix[m][n] else J
                        jaccard_matrix[n][m] = jaccard_matrix[m][n]
                    else:
                        jaccard_matrix[m][n] = J
                        jaccard_matrix[n][m] = J
                        low_edge += 1
            m = d
            for j in range(iA[d], iA[d + 1]):
                n1 = c
                n2 = jA[j]
                if n2 == n1 or jaccard_matrix[d][n2] !=1:
                    continue

                a, b = features[d], features[n2]
                intersection = a.multiply(b).count_nonzero()
                nM11 = intersection
                nM01 = b.count_nonzero() - intersection
                mM11 = (min(nM11, M11) + min(nM01,M10))
                mSum = abs(nM11 - M11) + abs(nM01-M10) + mM11
                if mM11 == 0:
                    overLimit = 0
                else:
                    overLimit = mM11 / mSum
                threshold2 = threshold * overLimit
                J2 = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)

                a, b = features[n1], features[n2]
                intersection = a.multiply(b).count_nonzero()
                J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)

                if J <= threshold2:
                    if J1 < J2:
                        n = c
                    else:
                        n = n2
                    if jaccard_matrix[m][n].item() != 1:
                        jaccard_matrix[m][n] = jaccard_matrix[m][n] if J > jaccard_matrix[m][n] else J
                        jaccard_matrix[n][m] = jaccard_matrix[m][n]
                    else:
                        jaccard_matrix[m][n] = J
                        jaccard_matrix[n][m] = J
                        low_edge += 1
        return low_edge



