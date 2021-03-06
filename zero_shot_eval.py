

import random
import numpy as np 
import torch
import torch.nn as nn 

import torch.nn.functional as f
# probFea = torch.randn(900, 2048)
# galFea = torch.randn(9000, 2048)
# centers = torch.randn(625, 2048)

# eq2 : implementation in pytroch
class w_calc(nn.Module):
    def __init__(self, num_classes1=900, num_classes2=625, feat_dim=2048, p=0.000005,lmbd=0.0000005 ,use_gpu=True):
        super(w_calc, self).__init__()
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.p = p
        self.lmbd =  lmbd
        if self.use_gpu:
            self.w = nn.Parameter(torch.randn(num_classes2, num_classes1).cuda())
        else:
            self.w = nn.Parameter(torch.randn(num_classes2, num_classes1))
    def forward(self, x, centers):
        # centers.shape =  625 , 2048
        # x.shape =  900 , 2048
        Dx = torch.mm(centers.t() , self.w ).t() #2048 x 900
        T1 = 0.5 * f.mse_loss(Dx, x,reduce=False).sum(-1).mean()
        # T1 = 0.5 * torch.pow((Dx - x),2).sum(-1).mean() #900x2048=> 900 x1 
        # .sum(1).mean(0) 
        T2 = (self.p / 2) * self.w.norm(p=2)
        # .mean() 
        T3 = self.lmbd * torch.norm(self.w, p=1)
        # .mean()
        loss= T1 + T2 + T3
        return loss



num_classes1 = probFea.shape[0]
num_classes2 = centers.shape[0]
l1 = w_calc(num_classes1,num_classes2,use_gpu=False , p=0.5 , lmbd=0.5)
optimizer_l1 = torch.optim.Rprop(l1.parameters())
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_l1, gamma=0.1, last_epoch=-1 )
for i in range(60):
    optimizer_l1.zero_grad()
    loss= l1(probFea, centers)
    loss.backward()
    print(loss)
    optimizer_l1.step()
    if i % 10 == 0 :
        scheduler.step()    


Dx = torch.mm(centers.t() , l1.w ).t()
T1 =torch.pow((Dx - probFea),2).sum(-1).sqrt().mean()
print(T1)
num_classes1 = galFea.shape[0]
l2 = w_calc(num_classes1,num_classes2,use_gpu=False , lmbd=0.5  , p=0.5)
optimizer_l2 = torch.optim.Rprop(l2.parameters())
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_l2, gamma=0.1, last_epoch=-1 )
for i in range(60):
    optimizer_l2.zero_grad()
    loss= l2(galFea, centers)
    loss.backward()
    print(loss)
    optimizer_l2.step()
    if i % 10 == 0 :
        scheduler.step()


Dx = torch.mm(centers.t() , l2.w ).t()
T1 =torch.pow((Dx - galFea),2).sum(-1).sqrt().mean()
print(T1)

# eq3 : implementation in pytroch
w_query = (l1.w.t() > 0).float()  
w_gallery = (l2.w > 0).float() 
score = torch.mm(w_query, w_gallery)
f_cross_view_support_consistency = torch.pow(1 / (1 +  score), 1)

# eq4: intersection of neighbours
m, n , o = probFea.size(0), galFea.size(0) , centers.size(0)
distmat_q = torch.pow(probFea, 2).sum(dim=1, keepdim=True).expand(m, o) + \
          torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(o, m).t()
distmat_q = distmat_q.addmm_(1, -2, probFea, centers.t()).sqrt()

distmat_g = torch.pow(galFea, 2).sum(dim=1, keepdim=True).expand(n, o) + \
          torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(o, n).t()
distmat_g = distmat_g.addmm_(1, -2, galFea, centers.t()).sqrt()
_, indices_g = torch.topk(distmat_g, 20)
_, indices_q = torch.topk(distmat_q, 20)
z1 = torch.zeros(n, o).scatter_(1, indices_g, 1)
z2 = torch.zeros(m, o).scatter_(1, indices_q, 1)
P = torch.mm(z2, z1.t())

#eq4
f_cross_view_projection_consistency = torch.pow(1 / (1 +  P), 1)

# used \alpha and \beta = 1 because the values are most distincitve there 
