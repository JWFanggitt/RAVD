import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from inspect import isfunction
import xformers
from different_topk import STVisualTokenSelection
from diffusers.models.attention_processor import Attention
import einops
import numpy as np
import random
from einops.layers.torch import Rearrange
class up_down_sampling(nn.Module):
    def __init__(self,in_dim,out_dim,):
        super(up_down_sampling,self).__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.conv=nn.Conv2d(in_channels=in_dim, out_channels=out_dim,kernel_size=1)

    def forward(self, A, B,):

        batch_frames, h_w, channels_B = B.shape
        H, W = int(A.shape[1] ** 0.5), int(A.shape[1] ** 0.5)
        h, w = int(B.shape[1] ** 0.5), int(B.shape[1] ** 0.5)
        B = B.permute(0,2,1).contiguous().reshape(batch_frames,channels_B,h,w)
        B= F.interpolate(B, size=(H, W), mode='bilinear', align_corners=False)
        B = self.conv(B)
        # B=torch.randn(16,256,1024).to(A.device,A.dtype)
        batch_frames, new_channels,new_w,new_h  =B.shape
        B_resized=B.permute(0,2,3,1).contiguous().reshape(batch_frames,new_w*new_h,new_channels).contiguous()
        return B_resized

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# class CrossAttention(nn.Module):
#     def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
#         super().__init__()
#         inner_dim = dim_head * heads
#         context_dim = default(context_dim, query_dim)
#
#         self.scale = dim_head**-0.5
#         self.heads = heads
#
#         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
#
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
#         )
#
#     def forward(self, x, text, mask=None):
#         B, L, C = x.shape
#         q = self.to_q(x)
#         # text = default(text, x)
#         k = self.to_k(text)
#         v = self.to_v(text)
#
#         q, k, v = map(
#             lambda t: rearrange(t, "B L (H D) -> B H L D", H=self.heads), (q, k, v)
#         )  # B H L D
#         # if ATTENTION_MODE == "flash":
#         x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
#         x = einops.rearrange(x, "B H L D -> B L (H D)")
#         # elif ATTENTION_MODE == "xformers":
#         # x = xformers.ops.memory_efficient_attention(q, k, v)
#         # x = einops.rearrange(x, "B L H D -> B L (H D)", H=self.heads)
#         # elif ATTENTION_MODE == "math":
#         #     attn = (q @ k.transpose(-2, -1)) * self.scale
#         #     attn = attn.softmax(dim=-1)
#         #     attn = self.attn_drop(attn)
#         #     x = (attn @ v).transpose(1, 2).reshape(B, L, C)
#         # else:
#         #     raise NotImplemented
#         return self.to_out(x)
#

class InfoGate(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InfoGate, self).__init__()
        self.a = input_dim
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=input_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=input_dim // 2, out_channels=output_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, input):
        x = torch.cat((input[0], input[1]), 2).permute(0, 2, 1)  # 变换维度为 [batch, channels, length]
        x = self.conv1d(x).permute(0, 2, 1)  # 输入应为 [batch, input_dim, length]
        # h, w = x.shape[2], x.shape[2]  # 假设 h 和 w 相等，若不相等需调整
        # gate = Rearrange('b c h w -> b (h w) c', h=h, w=w)(x)
        gate = F.gumbel_softmax(x, tau=0.3)
        gate = input[0]*gate

        # gate = Rearrange('b (h w) c -> b c h w', h=h, w=w)(gate)
        # out = input[0] * gate[:, 0, :, :].view(-1, 1, h, w)
        # out1 = input[1] * gate[:, 1, :, :].view(-1, 1, h, w)
        return gate







#yuan shi qa
# class QA(nn.Module):
#     # def __init__(self, query_dim, context_dim, video_frames, embed_dim, topk, non_topk, num_class):
#     def __init__(self,in_dim,out_dim):
#         super(QA, self).__init__()  # 确保这一行在构造函数中
#
#         # self.CrossAttention=Attention(query_dim=1024,cross_attention_dim=1024,heads=8,dim_head =64,)
#         self.token_selection=STVisualTokenSelection(max_frames=16)
#         self.up_down=up_down_sampling(in_dim,out_dim)
#         self.info_gated=InfoGate(input_dim=2048,output_dim=1024)
#         # self.max_pool_topk = nn.MaxPool1d(kernel_size=video_frames*topk)
#         # self.max_pool_non_topk = nn.MaxPool1d(kernel_size=video_frames * non_topk)
#
#         # self.decoder = nn.Sequential(
#         #     nn.Linear(query_dim, query_dim // 2),
#         #     nn.Tanh(),
#         #     nn.Linear(query_dim // 2, num_class))
#         # # self.project=nn.Linear(query_dim,256)
#         #
#         # self.text_weight_fc = nn.Sequential(
#         #     nn.Linear(query_dim, query_dim //2), nn.ReLU(inplace=True),
#         #     nn.Linear(query_dim // 2, 1))
#         # self.video_weight_fc = nn.Sequential(
#         #     nn.Linear(query_dim,query_dim // 2), nn.ReLU(inplace=True),
#         #     nn.Linear(query_dim // 2, 1))
#         # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#
#         # self.CrossEn=CrossEn()
#
#
#     # def get_similarity_logits(self,text_feat, video_feat, shaped=False):
#     #     text_weight = self.text_weight_fc(text_feat).squeeze(2)  # B x N_t x D -> B x N_t
#     #     text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t
#     #
#     #     video_weight =self.video_weight_fc(video_feat).squeeze(2)  # B x N_v x D -> B x N_v
#     #     video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v
#     #
#     #     text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
#     #     video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
#     #
#     #     retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])
#     #     t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
#     #     t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])
#     #
#     #     v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
#     #     v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
#     #
#     #     retrieve_logits = (t2v_logits + v2t_logits) / 2.0
#     #
#     #     return retrieve_logits, retrieve_logits.T
#
#
#     # def forward(self, video,question,option,noise,answer_id,non_aff):
#     def forward(self,fusion,video_noise,map):
#         batch_size=fusion.shape[0]//16
#         # option=fusion[1]
#         # non_option=fusion[2]
#         # r_option = torch.cat([option, non_option], dim=1)
#         # B, N, L, C = r_option.shape
#         # r_option = r_option.view(B, N * L, C)
#         # B_n, N_n, L_n, C_n = non_option.shape
#         # non_option = non_option.view(B_n,N_n * L_n, C_n)
#         #
#         # question=fusion[3]
#         bhw,frames,channels=video_noise.shape
#         hw=bhw//batch_size
#         h, w = int(hw ** 0.5), int(hw ** 0.5)
#         video_noise=rearrange(video_noise,'(b h w) f c -> (b f) (h w) c',b=batch_size,h=h,w=w)
#         # fusion=self.info_gated((fusion,map))
#         # up_down=up_down_sampling(video_noise.shape[-1],fusion.shape[-1]).to(device=video_noise.device,dtype=torch.float16)
#         video_noise= self.up_down(fusion,video_noise)
#         # video=self.gated_conv(map)
#         # video_noise=torch.randn(1,256,1024).to(fusion.device,fusion.dtype)
#         video=fusion+video_noise
#         # video = fusion
#         im_token, un_im_token,un_im_do_token = self.token_selection(video)
#         return im_token, un_im_token, un_im_do_token
#
#         # a_f = torch.cat([a_ff, non_a_ff], dim=1)
#         # fusion_token=torch.cat([im_token, un_im_token,un_im_do_token], dim=1)
#         # im_q_v1 = torch.cat([question,im_token], dim=1)
#         # un_im_q_v2 = torch.cat([question,un_im_token],dim=1)
#         # un_im_q_v3 = torch.cat([question,un_im_do_token], dim=1)
#         # im_q_out = self.CrossAttention(im_q_v1,r_option)
#         # un_im_q_out=self.CrossAttention(un_im_q_v2,non_option)
#         # un_im_do_out=self.CrossAttention(un_im_q_v3,non_option)
#
#
#
#
#
#
#         # all_ids = list(range(5))
#         # gt_option=option[:,answer_id,:,:].squeeze(1)
#
#         # all_ids.remove(answer_id)
#         # un_option = option[:, all_ids, :, :]
#         # B,N,L,C=non_aff.shape
#         # un_option=non_aff.view(B, N*L,C)
#         #
#         # t2v_logits, v2t_logits=self.get_similarity_logits(option,im_q_v1)
#         # t2v_logits=t2v_logits * self.logit_scale
#         # v2t_logits=v2t_logits * self.logit_scale
#         # discrimination_loss_t2v = self.CrossEn(t2v_logits * self.logit_scale)
#         # discrimination_loss_v2t = self.CrossEn(v2t_logits * self.logit_scale)
#         # discrimination_loss = (discrimination_loss_t2v + discrimination_loss_v2t) / 2
#         # im_out= self.max_pool_topk(self.CrossAttention(im_q_v1,option).transpose(1, 2)).squeeze(-1)
#         # max_pool_non_topk= nn.MaxPool1d(kernel_size=un_im_q_v2.shape[1])
#         # un_im_out = max_pool_non_topk(self.CrossAttention(un_im_q_v2,un_option).transpose(1, 2)).squeeze(-1)
#         # max_pool_non_do_topk = nn.MaxPool1d(kernel_size=un_im_q_v3.shape[1])
#         # un_do_im_out = max_pool_non_do_topk(self.CrossAttention(un_im_q_v3, un_option).transpose(1, 2)).squeeze(-1)
#         # un_im_out=self.decoder(un_im_out)
#         # un_im_do_out=self.decoder(un_do_im_out)
#
#         # return t2v_logits, v2t_logits,un_im_out,un_im_do_out
#         # return im_q_out,un_im_q_out,un_im_do_out
#



class QA(nn.Module):
    # def __init__(self, query_dim, context_dim, video_frames, embed_dim, topk, non_topk, num_class):
    def __init__(self,in_dim,out_dim):
        super(QA, self).__init__()  # 确保这一行在构造函数中

        # self.CrossAttention=Attention(query_dim=1024,cross_attention_dim=1024,heads=8,dim_head =64,)
        self.token_selection=STVisualTokenSelection(max_frames=16)
        self.up_down=up_down_sampling(in_dim,out_dim)
        self.info_gated=InfoGate(input_dim=2048,output_dim=1024)
        # self.max_pool_topk = nn.MaxPool1d(kernel_size=video_frames*topk)
        # self.max_pool_non_topk = nn.MaxPool1d(kernel_size=video_frames * non_topk)

        # self.decoder = nn.Sequential(
        #     nn.Linear(query_dim, query_dim // 2),
        #     nn.Tanh(),
        #     nn.Linear(query_dim // 2, num_class))
        # # self.project=nn.Linear(query_dim,256)
        #
        # self.text_weight_fc = nn.Sequential(
        #     nn.Linear(query_dim, query_dim //2), nn.ReLU(inplace=True),
        #     nn.Linear(query_dim // 2, 1))
        # self.video_weight_fc = nn.Sequential(
        #     nn.Linear(query_dim,query_dim // 2), nn.ReLU(inplace=True),
        #     nn.Linear(query_dim // 2, 1))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # self.CrossEn=CrossEn()


    # def get_similarity_logits(self,text_feat, video_feat, shaped=False):
    #     text_weight = self.text_weight_fc(text_feat).squeeze(2)  # B x N_t x D -> B x N_t
    #     text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t
    #
    #     video_weight =self.video_weight_fc(video_feat).squeeze(2)  # B x N_v x D -> B x N_v
    #     video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v
    #
    #     text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    #     video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
    #
    #     retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])
    #     t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
    #     t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])
    #
    #     v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
    #     v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
    #
    #     retrieve_logits = (t2v_logits + v2t_logits) / 2.0
    #
    #     return retrieve_logits, retrieve_logits.T


    # def forward(self, video,question,option,noise,answer_id,non_aff):
    def forward(self,fusion,map):
        fusion = self.info_gated((fusion, map))

        im_token, un_im_token,un_im_do_token = self.token_selection(fusion)
        return im_token, un_im_token, un_im_do_token

        # a_f = torch.cat([a_ff, non_a_ff], dim=1)
        # fusion_token=torch.cat([im_token, un_im_token,un_im_do_token], dim=1)
        # im_q_v1 = torch.cat([question,im_token], dim=1)
        # un_im_q_v2 = torch.cat([question,un_im_token],dim=1)
        # un_im_q_v3 = torch.cat([question,un_im_do_token], dim=1)
        # im_q_out = self.CrossAttention(im_q_v1,r_option)
        # un_im_q_out=self.CrossAttention(un_im_q_v2,non_option)
        # un_im_do_out=self.CrossAttention(un_im_q_v3,non_option)






        # all_ids = list(range(5))
        # gt_option=option[:,answer_id,:,:].squeeze(1)

        # all_ids.remove(answer_id)
        # un_option = option[:, all_ids, :, :]
        # B,N,L,C=non_aff.shape
        # un_option=non_aff.view(B, N*L,C)
        #
        # t2v_logits, v2t_logits=self.get_similarity_logits(option,im_q_v1)
        # t2v_logits=t2v_logits * self.logit_scale
        # v2t_logits=v2t_logits * self.logit_scale
        # discrimination_loss_t2v = self.CrossEn(t2v_logits * self.logit_scale)
        # discrimination_loss_v2t = self.CrossEn(v2t_logits * self.logit_scale)
        # discrimination_loss = (discrimination_loss_t2v + discrimination_loss_v2t) / 2
        # im_out= self.max_pool_topk(self.CrossAttention(im_q_v1,option).transpose(1, 2)).squeeze(-1)
        # max_pool_non_topk= nn.MaxPool1d(kernel_size=un_im_q_v2.shape[1])
        # un_im_out = max_pool_non_topk(self.CrossAttention(un_im_q_v2,un_option).transpose(1, 2)).squeeze(-1)
        # max_pool_non_do_topk = nn.MaxPool1d(kernel_size=un_im_q_v3.shape[1])
        # un_do_im_out = max_pool_non_do_topk(self.CrossAttention(un_im_q_v3, un_option).transpose(1, 2)).squeeze(-1)
        # un_im_out=self.decoder(un_im_out)
        # un_im_do_out=self.decoder(un_do_im_out)

        # return t2v_logits, v2t_logits,un_im_out,un_im_do_out
        # return im_q_out,un_im_q_out,un_im_do_out




class P_QA(nn.Module):
    # def __init__(self, query_dim, context_dim, video_frames, embed_dim, topk, non_topk, num_class):
    def __init__(self,query_dim,num_class1,num_class2):
        super(P_QA, self).__init__()  # 确保这一行在构造函数中
        self.query_dim=query_dim
        # self.context_dim=context_dim
        self.CrossAttention=Attention(query_dim=1024,cross_attention_dim=1024,heads=8,dim_head =64,)
        self.max_frames=16
        # self.max_pool_topk = nn.MaxPool1d(kernel_size=video_frames*topk)
        # self.max_pool_non_topk = nn.MaxPool1d(kernel_size=video_frames * non_topk)

        self.decoder = nn.Sequential(
            nn.Linear(query_dim, query_dim // 2),
            nn.Tanh(),
            nn.Linear(query_dim // 2, num_class1))

        self.decoder1 = nn.Sequential(
            nn.Linear(query_dim, query_dim // 2),
            nn.Tanh(),
            nn.Linear(query_dim // 2, num_class2))
        self.avg_p=nn.AdaptiveAvgPool1d(1)
    # def forward(self, video,question,option,noise,answer_id,non_aff):
    def forward(self,im_token,un_im_token,un_im_do_token,question,r_option, non_option,answer):
        # output1 =selected_patch_feature.reshape(B, self.max_frames, -1, D).reshape(B, -1,
        #                                                            D)
        B, L, D = im_token.shape
        b=B//self.max_frames
        im_token=im_token.reshape(b, self.max_frames, -1, D).reshape(b, -1, D)
        # un_im_token = un_im_token.reshape(b, self.max_frames, -1, D).reshape(b, -1, D)
        # un_im_do_token =un_im_do_token.reshape(b, self.max_frames, -1, D).reshape(b, -1, D)
        #

        im_q_v1 = torch.cat([question,im_token], dim=1)
        # un_im_q_v2 = torch.cat([question,un_im_token],dim=1)
        # un_im_q_v3 = torch.cat([question,un_im_do_token], dim=1)
        # max_pool_topk = nn.MaxPool1d(kernel_size=im_q_v1.shape[1])
        # max_pool_non_topk = nn.MaxPool1d(kernel_size=un_im_q_v2.shape[1])
        # max_pool_do_non_topk = nn.MaxPool1d(kernel_size=un_im_q_v3.shape[1])
        im_q_out = self.decoder(self.avg_p(self.CrossAttention(im_q_v1,r_option).transpose(1, 2)).squeeze(2))
        # un_im_q_out=self.decoder1(self.avg_p(self.CrossAttention(un_im_q_v2,non_option).transpose(1, 2)).squeeze(2))
        # un_im_do_out=self.decoder1(self.avg_p(self.CrossAttention(un_im_q_v3,non_option).transpose(1, 2)).squeeze(2))

        # im_q_out =self.CrossAttention(im_q_v1,r_option).transpose(1,2)
        # # max_pooling = nn.MaxPool1d(kernel_size=im_q_out.shape[2])
        # im_q_out=torch.mean(im_q_out,dim=2).unsqueeze(2)
        # predicts=torch.bmm(answer,im_q_out).squeeze()
        predicted=torch.max(im_q_out,dim=1).indices
        return  predicted







if __name__=="__main__":
    model=InfoGate(input_dim=2048,output_dim=1024)
    x1=torch.randn(16,256,1024)
    x2=torch.randn(16,256,1024)
    y=(x1,x2)
    out=model(y)
    print(out.shape)





#     model=QA(query_dim=1024, context_dim=1024,video_frames=16,
#              embed_dim=1024,topk=32,non_topk=224,num_class=5)
#     video = torch.randn(16, 256, 1024)
#     noise= torch.randn(16, 256, 1024)
#     question = torch.randn(1, 10, 1024)
#     answer = torch.randn(1, 5, 10, 1024)
#     im_out,un_im_out=model(video,question,answer,noise)
#     print(im_out.shape,un_im_out.shape)



    # net=CrossAttention(query_dim=1024,context_dim=1024)
    # net_=torch.nn.Linear(1024,256)
    # noise = torch.randn(16, 256, 1024)
    # oo = net_(noise)
    # # print(oo.shape)
    # video = torch.randn(16, 256, 1024)
    # token_slection=STVisualTokenSelection(max_frames=16,embed_dim=1024,topk=32)
    # out1,out2=token_slection(video,oo)
    # question = torch.randn(1, 10, 1024)
    # q_v1 = torch.cat([question,out1],dim=1)
    # q_v2=torch.cat([question,out2],dim=1)
    # answer = torch.randn(1, 5, 10, 1024)
    # answer=answer.view(1,50,1024)
    # max_pool = nn.MaxPool1d(kernel_size=522)
    # max_pool1=nn.MaxPool1d(kernel_size=3594)
    #
    # out3=net(q_v1,answer)
    # out4=net(q_v2,answer)
    # out3_transposed = out3.transpose(1, 2)
    # out4_transposed = out4.transpose(1, 2)
    # ooo1=max_pool( out3_transposed).squeeze(-1)
    # ooo2=max_pool1(out4_transposed).squeeze(-1)
    #
    # atten_pool = nn.Sequential(
    #     nn.Linear(1024, 1024 // 2),
    #     nn.Tanh(),
    #     nn.Linear(1024 // 2 , 5))
    #
    # r1=atten_pool(ooo1)
    # r2=atten_pool(ooo2)
    # print(r1.shape)
    # print(r2.shape)
    # ce = nn.CrossEntropyLoss()
    # kl_mb = nn.KLDivLoss(reduction='batchmean')
    # kl_b = nn.KLDivLoss(reduction='batchmean')
    # ans_targets=torch.Tensor(1)
    # ans_targets=ans_targets.to(dtype=torch.long)
    #
    # ce_loss = ce(r1, ans_targets)
    # print(ce_loss)
    # klb_loss = kl_b(F.log_softmax(r2, dim=1), r2.new_ones(r2.size()) / 5)  # outb是补充视频预测
    # print(klb_loss)



