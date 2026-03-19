import torch
import torch.nn as nn

# class Decoder2D(nn.Module):
#     def __init__(self, in_channels=1024, out_channels=1,
#                  features=[512, 256, 128, 64]):
#         super().__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.decoder_1 = nn.Sequential(
#             nn.Conv2d(in_channels, features[0], 3, padding=1),
#             nn.BatchNorm2d(features[0]),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         )
#         self.decoder_2 = nn.Sequential(
#             nn.Conv2d(features[0], features[1], 3, padding=1),
#             nn.BatchNorm2d(features[1]),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         )
#         self.decoder_3 = nn.Sequential(
#             nn.Conv2d(features[1], features[2], 3, padding=1),
#             nn.BatchNorm2d(features[2]),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         )
#         self.decoder_4 = nn.Sequential(
#             nn.Conv2d(features[2], features[3], 3, padding=1),
#             nn.BatchNorm2d(features[3]),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=1.75, mode="bilinear", align_corners=True)
#         )
#         self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)
#
#     def forward(self, x):
#         # x=torch.cat([x,y],dim=1)
#         height = width = int(x.shape[1] ** 0.5)
#         x = x.reshape(
#             shape=(-1, 1024, height, width)
#         )
#         x=x.mean(dim=0).unsqueeze(0)
#         x = self.decoder_1(x)
#         x = self.decoder_2(x)
#         x = self.decoder_3(x)
#         x = self.decoder_4(x)
#         x = self.final_out(x)
#         x = self.sigmoid(x)
#         return x
#

def kl_loss(y_true, y_pred, eps=1e-07):
    P = y_pred
    P = P / (eps + torch.sum(P, dim=(0, 1, 2, 3), keepdim=True))
    Q = y_true
    Q = Q / (eps + torch.sum(Q, dim=(0, 1, 2, 3), keepdim=True))
    kld = torch.sum(Q * torch.log(eps + Q / (eps + P)), dim=(0, 1, 2, 3))
    return kld




if __name__=="__main__":
    x=torch.randn(16,256,1024)
    height = width = int(x.shape[1] ** 0.5)
    x = x.reshape(
        shape=(-1, 1024, height, width )
    )
    net=Decoder2D()
    out=net(x)
    print(out.shape)
    # y=torch.randn(16,1,256,256)
    # loss=kl_loss(y,out)
    #
    # print(loss)
    # unpatchify
