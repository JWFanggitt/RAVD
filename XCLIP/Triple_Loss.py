import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self,margin):
        super(TripletLoss,self).__init__()
        self.margin=margin

    def forward(self,anchor,positive,negative):
        pos_dist=torch.sum((anchor-positive)**2,dim=-1)
        neg_dist=torch.sum((anchor-negative)**2,dim=-1)

        basic_loss=pos_dist-neg_dist+self.margin
        loss=torch.mean(torch.max(basic_loss,torch.zeros_like(basic_loss)))
        return loss


# 定义温度系数的InfoNCE损失函数
def InfoNCE_loss(clip1, clip2, clip3, clip4, temperature=1):
    # 计算内积
    inner_product_clip1_clip3 = torch.matmul(clip1, clip3.t()) / temperature
    inner_product_clip2_clip4 = torch.matmul(clip2, clip4.t()) / temperature
    inner_product_clip2_clip3 = torch.matmul(clip2, clip3.t()) / temperature
    inner_product_clip1_clip4 = torch.matmul(clip1, clip4.t()) / temperature

    # 计算对角线项
    diag_clip1_clip3 = torch.diag(inner_product_clip1_clip3)
    diag_clip2_clip4 = torch.diag(inner_product_clip2_clip4)
    diag_clip2_clip3 = torch.diag(inner_product_clip2_clip3)
    diag_clip1_clip4 = torch.diag(inner_product_clip1_clip4)

    # 计算InfoNCE损失
    loss = -torch.mean(torch.log(torch.exp(diag_clip1_clip3) / (
                torch.sum(torch.exp(inner_product_clip1_clip3), dim=1) - diag_clip1_clip3 + 1e-8)) +
                       torch.log(torch.exp(diag_clip2_clip4) / (
                                   torch.sum(torch.exp(inner_product_clip2_clip4), dim=1) - diag_clip2_clip4 + 1e-8)) +
                       torch.log(torch.exp(inner_product_clip2_clip3 - diag_clip2_clip3).sum(dim=1) / (
                                   torch.sum(torch.exp(inner_product_clip2_clip3), dim=1) - diag_clip2_clip3 + 1e-8)) +
                       torch.log(torch.exp(inner_product_clip1_clip4 - diag_clip1_clip4).sum(dim=1) / (
                                   torch.sum(torch.exp(inner_product_clip1_clip4), dim=1) - diag_clip1_clip4 + 1e-8)))

    return loss