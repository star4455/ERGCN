import torch

class Standard:
    def __init__(self,preds,labels):
        self.nclass = labels.max().item()+1
        self.labelClassNum = []
        self.correctClassNum = []
        self.predsClassNum = []
        for i in range(self.nclass):
            labelClassIndex = (labels == i).sum().item()
            predsClassIndex = (preds == i).sum().item()
            self.labelClassNum.append(labelClassIndex if labelClassIndex != 0 else 1)
            self.predsClassNum.append(predsClassIndex if predsClassIndex != 0 else 1)
            self.correctClassNum.append(((labels == i) & (preds == i)).sum().item())
        self.labelClassNum = torch.tensor(self.labelClassNum)
        self.correctClassNum = torch.tensor(self.correctClassNum)
        self.predsClassNum = torch.tensor(self.predsClassNum)

    def precision(self):
        return (self.correctClassNum / self.predsClassNum).mean().item()
    def recall(self):
        return (self.correctClassNum / self.labelClassNum).mean().item()
    def f1(self):
        return 2 / (1/self.precision() + 1 / self.recall())