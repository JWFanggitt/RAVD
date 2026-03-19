import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from Triple_Loss import TripletLoss
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, \
    generate_text, auto_resume_helper
# from datasets.build1 import build_dataloader
from datasets.build2 import build_dataloader
from utils.logger import *
import time
import numpy as np
import random
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from models import xclip
import torch.nn.functional as F
from sklearn.metrics import *
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datasets.CAP import DADA2KS
from tqdm import tqdm

train_dataset= DADA2KS(root_path=r"/media/ubuntu/My Passport/CAPDATA", interval=1,phase="train")
train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        pin_memory=True, drop_last=True)

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str,
                        default=r'/home/ubuntu/lileilei/X-CLIP/configs/k400/16_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="./results")
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)

    parser.add_argument("--local_rank", type=int, default=-1,
                        help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    # print(config)
    # train_data, val_data, train_loader, val_loader = build_dataloader(config)
    model, _ = xclip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH,
                          device="cpu", jit=False,
                          T=config.DATA.NUM_FRAMES,
                          droppath=0.5,
                          use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                          use_cache=config.MODEL.FIX_TEXT,
                          logger=logger,
                          )
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(2)
    model = model.to(device)
    loss=TripletLoss(0.5)
    loop = tqdm(train_dataloader, total=len(train_dataloader), leave=True)
    for epoch in range(0, config.TRAIN.EPOCHS):
        for step, batch in enumerate(loop):

                nv=batch["nv"].to(device)
                rv=batch["rv"].to(device)
                av=batch["av"].to(device)
                N_t=batch["N_t"]
                R_t=batch["R_t"]
                P_t=batch["P_t"]
                C_t=batch["C_t"]
                N_t=generate_text(N_t).to(device)
                R_t = generate_text(R_t).to(device)
                P_t=generate_text(P_t).to(device)
                C_t=generate_text(C_t).to(device)
                clip1=model(nv,N_t)
                clip2=model(rv,R_t)
                clip3=model(rv,P_t)
                clip4=model(av,C_t)






        #     total_loss = criterion(output, label_id)
        #     total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
        #
        #     if mixup_fn is not None:
        #         label_id = hard_label
        #     values_1, indices_1 = output.topk(1, dim=-1)
        #     num += images.shape[0]
        #     for i in range(images.shape[0]):
        #         if indices_1[i] == label_id[i]:
        #             corr_1 += 1
        #
        #     if config.TRAIN.ACCUMULATION_STEPS == 1:
        #         optimizer.zero_grad()
        #     if config.TRAIN.OPT_LEVEL != 'O0':
        #         with amp.scale_loss(total_loss, optimizer) as scaled_loss:
        #             scaled_loss.backward()
        #     else:
        #         total_loss.backward()
        #     if config.TRAIN.ACCUMULATION_STEPS > 1:
        #         if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
        #             optimizer.step()
        #             optimizer.zero_grad()
        #             lr_scheduler.step_update(epoch * num_steps + idx + float(1e-8))
        #     else:
        #         optimizer.step()
        #         lr_scheduler.step_update(epoch * num_steps + idx)
        #
        #     torch.cuda.synchronize()
        #
        #     tot_loss_meter.update(total_loss.item(), len(label_id))
        #     batch_time.update(time.time() - end)
        #     end = time.time()
        #
        #     if idx % config.PRINT_FREQ == 0:
        #         lr = optimizer.param_groups[0]['lr']
        #         memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        #         etas = batch_time.avg * (num_steps - idx)
        #         logger.info(
        #             f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
        #             f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
        #             f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
        #             f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
        #             f'mem {memory_used:.0f}MB')
        # epoch_time = time.time() - start
        # top1 = float(corr_1) / num * 100
        # logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
        # if tb_writer is not None:
        #     tb_writer.add_scalar('train/loss', tot_loss_meter.val, epoch)
        #     tb_writer.add_scalar('train/top1', top1, epoch)
        #     tb_writer.add_scalar('train/lr', optimizer.state_dict()['param_groups'][0]['lr']
        #                          , epoch)







#     if config.AUG.MIXUP > 0:
#         criterion = SoftTargetCrossEntropy()
#         mixup_fn = CutmixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
#                                        smoothing=config.AUG.LABEL_SMOOTH,
#                                        mixup_alpha=config.AUG.MIXUP,
#                                        cutmix_alpha=config.AUG.CUTMIX,
#                                        switch_prob=config.AUG.MIXUP_SWITCH_PROB)
#     elif config.AUG.LABEL_SMOOTH > 0:
#         criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
#     else:
#         mixup_fn = None
#         criterion = nn.CrossEntropyLoss()
#
#     optimizer = build_optimizer(config, model)
#     lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
#     if config.TRAIN.OPT_LEVEL != 'O0':
#         model, optimizer = amp.initialize(models=model, optimizers=optimizer,
#                                           opt_level=config.TRAIN.OPT_LEVEL)
#     # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK],
#     # broadcast_buffers=False, find_unused_parameters=False)
#
#     start_epoch, max_accuracy = 0, 0.0
#
#     if config.TRAIN.AUTO_RESUME:
#         resume_file = auto_resume_helper(config.OUTPUT)
#         if resume_file:
#             config.defrost()
#             config.MODEL.RESUME = resume_file
#             config.freeze()
#             logger.info(f'auto resuming from {resume_file}')
#         else:
#             logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
#
#     if config.MODEL.RESUME:
#         start_epoch, max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
#
#     text_labels = generate_text(train_data)
#
#     tb_writer = SummaryWriter(log_dir=args.output)
#
#     if config.TEST.ONLY_TEST:
#         acc1, val_logger = validate(start_epoch, val_loader, text_labels, model, config, device, tb_writer)
#         logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.2f}%")
#         val_logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.2f}%")
#         return
#
#     for epoch in range(start_epoch, config.TRAIN.EPOCHS):
#         # train_loader.sampler.set_epoch(epoch)
#         train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels,
#                         config, mixup_fn, device, tb_writer)
#
#         acc1, val_logger = validate(epoch, val_loader, text_labels, model, config, device, tb_writer)
#         # plt.savefig(f'./results/confusion_matrix_{epoch}.jpg')
#         logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.2f}%")
#         is_best = acc1 > max_accuracy
#         max_accuracy = max(max_accuracy, acc1)
#         if is_best:
#             best_epoch = epoch
#             logger.info(f'Max accuracy: {max_accuracy:.2f}%, epoch:{best_epoch}')
#             val_logger.info(f'Max accuracy: {max_accuracy:.2f}%, epoch:{best_epoch}')
#         if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
#             epoch_saving(config, epoch, model, max_accuracy, optimizer, lr_scheduler,
#                          logger, config.OUTPUT, is_best)
#
#     config.defrost()
#     config.TEST.NUM_CLIP = 4
#     config.TEST.NUM_CROP = 3
#     config.freeze()
#     print(config)
#
#     train_data, val_data, train_loader, val_loader = build_dataloader(config)
#     acc1, val_logger = validate(epoch, val_loader, text_labels, model, config, device, tb_writer)
#     logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.2f}%")
#
#
# def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels,
#                     config, mixup_fn, device, tb_writer):
#     model.train()
#     # print(model)
#     optimizer.zero_grad()
#
#     num_steps = len(train_loader)
#     batch_time = AverageMeter()
#     tot_loss_meter = AverageMeter()
#
#     start = time.time()
#     end = time.time()
#     num = 0
#     corr_1 = 0
#     texts = text_labels.cuda(non_blocking=True)
#
#     for idx, batch_data in enumerate(train_loader):
#         # optimizer.zero_grad()
#         images, label_id = batch_data[0], batch_data[1]
#         if mixup_fn is not None:
#             hard_label = label_id
#             images, label_id = mixup_fn(images, label_id)
#         images = images.to(device)  # b,t,c,h,w
#         label_id = label_id.to(device)  # b, num_classes
#         if texts.shape[0] == 1:
#             texts = texts.view(1, -1)
#         texts = texts.to(device)  # len(text), 77
#         output = model(images, texts)  # b,len(text)
#
#         total_loss = criterion(output, label_id)
#         total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
#
#         if mixup_fn is not None:
#             label_id = hard_label
#         values_1, indices_1 = output.topk(1, dim=-1)
#         num += images.shape[0]
#         for i in range(images.shape[0]):
#             if indices_1[i] == label_id[i]:
#                 corr_1 += 1
#
#         if config.TRAIN.ACCUMULATION_STEPS == 1:
#             optimizer.zero_grad()
#         if config.TRAIN.OPT_LEVEL != 'O0':
#             with amp.scale_loss(total_loss, optimizer) as scaled_loss:
#                 scaled_loss.backward()
#         else:
#             total_loss.backward()
#         if config.TRAIN.ACCUMULATION_STEPS > 1:
#             if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 lr_scheduler.step_update(epoch * num_steps + idx + float(1e-8))
#         else:
#             optimizer.step()
#             lr_scheduler.step_update(epoch * num_steps + idx)
#
#         torch.cuda.synchronize()
#
#         tot_loss_meter.update(total_loss.item(), len(label_id))
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if idx % config.PRINT_FREQ == 0:
#             lr = optimizer.param_groups[0]['lr']
#             memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
#             etas = batch_time.avg * (num_steps - idx)
#             logger.info(
#                 f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
#                 f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
#                 f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
#                 f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
#                 f'mem {memory_used:.0f}MB')
#     epoch_time = time.time() - start
#     top1 = float(corr_1) / num * 100
#     logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
#     if tb_writer is not None:
#         tb_writer.add_scalar('train/loss', tot_loss_meter.val, epoch)
#         tb_writer.add_scalar('train/top1', top1, epoch)
#         tb_writer.add_scalar('train/lr', optimizer.state_dict()['param_groups'][0]['lr']
#                              , epoch)
#
#
# @torch.no_grad()
# def validate(epoch, val_loader, text_labels, model, config, device, tb_writer):
#     model.eval()
#     val_logger = get_logger('results/val.log')
#     acc1_meter, acc5_meter, acc10_meter = AverageMeter(), AverageMeter(), AverageMeter()
#     y_true = np.empty((0))
#     y_scores = np.empty((0, 58))
#     corr_1 = 0
#     corr_5 = 0
#     corr_10 = 0
#     num = 0
#     all_preds = []
#     all_labels = []
#     total_time = []
#     with torch.no_grad():
#         text_inputs = text_labels.cuda()
#         logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
#         target = []
#         indices_1_all = []
#         pred_list = []
#         classes = [i for i in range(58)]
#         for idx, batch_data in enumerate(val_loader):
#             _image, label_id = batch_data[0], batch_data[1]
#             label_id = label_id.reshape(-1)
#
#             # b, tn, c, h, w = _image.size()
#             # t = config.DATA.NUM_FRAMES
#             # n = tn // t
#             # _image = _image.view(b, n, t, c, h, w)
#             b, n, t, c, h, w = _image.size()
#             tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
#             for i in range(n):
#                 image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
#                 label_id = label_id.cuda(non_blocking=True)
#                 image_input = image.cuda(non_blocking=True)
#
#                 if config.TRAIN.OPT_LEVEL == 'O2':
#                     image_input = image_input.half()
#                 start = time.time()
#                 output = model(image_input, text_inputs)
#                 end = time.time()
#                 expend = end - start
#                 total_time.append(expend)
#                 # pred_softmax = torch.softmax(output, 1).cpu().numpy()
#                 # pred_list.append(pred_softmax.tolist()[0])
#
#                 similarity = output.view(b, -1).softmax(dim=-1)
#                 tot_similarity += similarity
#
#             pred_softmax = similarity  # b,58
#             all_preds.append(tot_similarity.cpu().numpy())
#             all_labels.append(label_id.cpu().numpy())
#             pred_ = torch.softmax(tot_similarity, 1).cpu().numpy()
#             pred_list.append(pred_.tolist()[0])
#             values_1, indices_1 = tot_similarity.topk(1, dim=-1)
#             values_5, indices_5 = tot_similarity.topk(5, dim=-1)
#             values_10, indices_10 = tot_similarity.topk(10, dim=-1)
#             acc1, acc5, acc10 = 0, 0, 0
#             num += b
#             for i in range(b):
#                 if indices_1[i] == label_id[i]:
#                     acc1 += 1
#                     corr_1 +=1
#                 if label_id[i] in indices_5[i]:
#                     acc5 += 1
#                     corr_5 += 1
#                 if label_id[i] in indices_10[i]:
#                     acc10 += 1
#                     corr_10 += 1
#
#             label_id = label_id.to("cpu")
#             indices_1 = indices_1.to("cpu")
#             for j in label_id:
#                 j = j.cpu().detach().numpy()
#                 # p = p.tolist()
#                 target.append(j)
#             targets = np.array(target)
#             for k in indices_1:
#                 indices_1_all.append(k)
#             indices_1_list = [arr[0].item() for arr in indices_1_all]
#             indices_1_array = np.array(indices_1_list)
#
#             acc1_meter.update(float(acc1) / b * 100, b)
#             acc5_meter.update(float(acc5) / b * 100, b)
#             acc10_meter.update(float(acc10) / b * 100, b)
#             if idx % config.PRINT_FREQ == 0:
#                 logger.info(
#                     f'Test: [{idx}/{len(val_loader)}]\t'
#                     f'Acc@1: {acc1_meter.avg:.2f}\t'
#                     f'Acc@5: {acc5_meter.avg:.2f}\t'
#                     f'Acc@10: {acc10_meter.avg:.2f}\t'
#                 )
#     # acc1_meter.sync()
#     # acc5_meter.sync()
#     # acc10_meter.sync()
#     top1 = float(corr_1) / num * 100
#     top5 = float(corr_5) / num * 100
#     top10 = float(corr_10) / num * 100
#     logger.info(f'top1:{top1:.2f},top5:{top5:.2f},top10:{top10:.2f}\n')
#
#     all_preds = np.concatenate(all_preds, axis=0)  # (1941, 58)
#     all_labels = np.concatenate(all_labels, axis=0)  # (1941,)
#
#     # probabilities = np.exp(all_preds) / np.sum(np.exp(all_preds), axis=1, keepdims=True)
#     thresholds = np.linspace(0, 1, 41)
#     precision_v = []
#     recall_v = []
#     fpr_v = []
#     tpr_v = []
#     for threshold in thresholds:
#         pred = np.where(all_preds > threshold, 1, 0)
#         tp = np.zeros(58)
#         fp = np.zeros(58)
#         fn = np.zeros(58)
#         tn = np.zeros(58)
#         for i in range(58):
#             tp[i] = np.sum((pred[:, i] == 1) & (all_labels == i))
#             fp[i] = np.sum((pred[:, i] == 1) & (all_labels != i))
#             fn[i] = np.sum((pred[:, i] == 0) & (all_labels == i))
#             tn[i] = np.sum((pred[:, i] == 0) & (all_labels != i))
#         class_precision = tp / np.maximum((tp + fp), 1)
#         class_recall = tp / np.maximum((tp + fn), 1)
#         class_fpr = fp / np.maximum((fp + tn), 1)
#         class_tpr = tp / np.maximum((tp + fn), 1)
#         # class_precision = tp / (tp + fp + 1e-8)
#         # class_recall = tp / (tp + fn + 1e-8)
#
#         precision = np.mean(class_precision)
#         recall = np.mean(class_recall)
#         fpr = np.mean(class_fpr)
#         tpr = np.mean(class_tpr)
#
#         precision_v.append(precision)
#         recall_v.append(recall)
#         fpr_v.append(fpr)
#         tpr_v.append(tpr)
#     data_pr = np.column_stack((recall_v, precision_v))
#     data_roc = np.column_stack((fpr_v, tpr_v))
#     np.savetxt('./results/pr-xclip.txt', data_pr)
#     np.savetxt('./results/roc-xclip.txt', data_roc)
#     plt.plot(recall_v, precision_v)
#     plt.xlabel("Recall")
#     plt.ylabel('Precision')
#     plt.show()
#     plt.plot(fpr_v, tpr_v)
#     plt.xlabel("False positive rate")
#     plt.ylabel('True positive rate')
#     plt.show()
#     # num_classes = 10
#     # colors = plt.cm.tab20(np.linspace(0, 1, 20))
#     # best_classes = np.argsort(np.average(y_scores, axis=0))[::-1][:num_classes]
#     # # 绘制PR曲线
#     # plt.figure(figsize=(12, 8))
#     # for class_idx in range(0, 58):
#     #     class_true = (y_true == class_idx).astype(int)
#     #     class_scores = y_scores[:, class_idx]
#     #     class_precision, class_recall, class_thresholds = precision_recall_curve(class_true,
#     #                                                                              class_scores)
#     #     if class_idx in best_classes:
#     #         plt.plot(class_recall, class_precision, color=colors[class_idx % 20], linewidth=3,
#     #                  linestyle='-', label=f'Class-{class_idx}')
#     #     else:
#     #         plt.plot(class_recall, class_precision, color=colors[class_idx % 20], linewidth=3,
#     #                  linestyle='--', alpha=0.5)
#     #     # average_precision[i] = average_precision_score(class_true, class_scores)
#     #     # plt.plot(recall, precision, label=f'Class-{class_idx}')
#     # # map_score = np.mean(list(average_precision.values()))
#     # # plt.title('Precision-Recall Curve\n Average Precision={0:0.2f}'.format(map_score))
#     # plt.xlabel('Recall', fontsize=20)
#     # plt.ylabel('Precision', fontsize=20)
#     # plt.ylim([0.0, 1.05])
#     # plt.xlim([0.0, 1.0])
#     # plt.grid(True)
#     # # plt.legend(loc='upper right', fontsize='xx-small', ncol=5)  # ncol=5代表图例分成5列
#     # plt.legend(loc='best', fontsize='large')
#     # plt.xticks(fontsize=18)
#     # plt.yticks(fontsize=18)
#     # if config.TEST.ONLY_TEST:
#     #     plt.savefig('./results/PR曲线-test.pdf', dpi=300, bbox_inchess='tight')
#     # else:
#     #     plt.savefig(f'./results/PR曲线-e{epoch}.pdf', dpi=300, bbox_inchess='tight')
#     # plt.clf()
#     #
#     # # 绘制ROC曲线
#     # plt.figure(figsize=(14, 10))
#     # for class_idx in range(0, 58):
#     #     class_true = (y_true == class_idx).astype(int)
#     #     class_scores = y_scores[:, class_idx]
#     #     class_fpr, class_tpr, class_thresholds = roc_curve(class_true, class_scores)
#     #     if class_idx in best_classes:
#     #         plt.plot(class_fpr, class_tpr, color=colors[class_idx % 20], linewidth=3,
#     #                  linestyle='-', label=f'Class-{class_idx}')
#     #     else:
#     #         plt.plot(class_fpr, class_tpr, color=colors[class_idx % 20], linewidth=3,
#     #                  linestyle='--', alpha=0.4)
#     # plt.xlabel('False Positive Rate', fontsize=20)
#     # plt.ylabel('True Positive Rate', fontsize=20)
#     # plt.ylim([0.0, 1.05])
#     # plt.xlim([0.0, 1.0])
#     # plt.grid(True)
#     # # plt.legend(loc='upper right', fontsize='xx-small', ncol=5)  # ncol=5代表图例分成5列
#     # plt.legend(loc='best', fontsize='large')
#     # plt.xticks(fontsize=18)
#     # plt.yticks(fontsize=18)
#     # if config.TEST.ONLY_TEST:
#     #     plt.savefig('./results/ROC曲线-test.pdf', dpi=300, bbox_inchess='tight')
#     # else:
#     #     plt.savefig(f'./results/ROC曲线-e{epoch}.pdf', dpi=300, bbox_inchess='tight')
#     # plt.clf()
#
#     report = classification_report(targets, indices_1_array, digits=4, output_dict=False)
#     val_logger.info('Classification report: \n%s:\n' % report)
#     val_logger.info('Precision: %.2f' % precision_score(targets, indices_1_array, average="macro"))
#     val_logger.info('Recall: %.2f' % recall_score(targets, indices_1_array, average="macro"))
#     val_logger.info('F1-score: %.2f' % f1_score(targets, indices_1_array, average="macro"))
#
#     # conf_mx = confusion_matrix(targets, indices_1_array, labels=classes)
#     # if config.TEST.ONLY_TEST:
#     #     np.savetxt(f'./results/混淆矩阵-test.csv', conf_mx, delimiter=',')
#     # else:
#     #     np.savetxt(f'./results/混淆矩阵-e{epoch}.csv', conf_mx, delimiter=',')
#
#     # df = pd.DataFrame(report).transpose()
#     # df.to_csv('各类别准确率评估指标.csv', index_label='类别')
#
#     # conf_mx = confusion_matrix(targets, indices_1_array, labels=classes)
#     # # plot_confusion_matrix(conf_mx, classes, normalize=True)
#     # cmp = ConfusionMatrixDisplay(confusion_matrix=conf_mx, display_labels=classes)
#     # # cmp.plot(cmap=plt.cm.Blues)
#     # fig, ax = plt.subplots(figsize=(12.8, 12.8), dpi=120)
#     # cmp.plot(ax=ax, colorbar=False, cmap=plt.cm.Blues)
#     # plt.tick_params(labelsize=10)
#     # plt.xticks(classes, rotation=45)
#     # plt.ylabel('True label', fontsize=20, labelpad=20)
#     # plt.xlabel('Predicted label', fontsize=20, labelpad=20)
#     # cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
#     # plt.colorbar(cmp.im_, cax=cax)
#     # plt.savefig('./results/confusion_matrix.jpg')
#     # plt.show()
#
#     df_pred = pd.DataFrame(data=pred_list, columns=classes)
#     df_pred.to_csv('./results/pred_result.csv', encoding='gbk', index=False)
#
#     logger.info(f'* Acc@1 {acc1_meter.avg:.2f}, Acc@5 {acc5_meter.avg:.2f},  '
#                 f'Acc@10 {acc10_meter.avg:.2f}')
#     val_logger.info(f'* Acc@1 {acc1_meter.avg:.2f}, Acc@5 {acc5_meter.avg:.2f}, '
#                     f'Acc@10 {acc10_meter.avg:.2f}')
#
#     total_time = np.sum(total_time)
#     print(total_time/1940)
#     return acc1_meter.avg, val_logger


if __name__ == '__main__':
    # prepare config
    args, config = parse_option()

    # init_distributed
    # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    #     rank = int(os.environ["RANK"])
    #     world_size = int(os.environ['WORLD_SIZE'])
    #     print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    # else:
    #     rank = -1
    #     world_size = -1
    # torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # torch.distributed.barrier(device_ids=[args.local_rank])

    # seed = config.SEED + dist.get_rank()
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)

    # logger
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")

    # save config 
    # if dist.get_rank() == 0:
    logger.info(config)
    shutil.copy(args.config, config.OUTPUT)

    main(config)
