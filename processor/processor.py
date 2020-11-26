import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
import cv2
from utils.meter import AverageMeter
from utils.metrics import R1_mAP, R1_mAP_Pseudo
import json
import datetime
from solver import make_optimizer, WarmupMultiStepLR
import torch.distributed as dist
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def pcb_loss_forward(pcb_feat,targets):
    # print("inputs.device",inputs.device)
    # print("next(model.parameters()).device",next(model.parameters()).device)
    #
    # outputs = model(inputs)
    criterion = nn.CrossEntropyLoss()
        #
        # global loss0
        # global loss1
        # global loss2
        # global loss3
        # global loss4
        # global loss5
    loss0 = criterion(pcb_feat[0],targets)
    loss1 = criterion(pcb_feat[1],targets)
    loss2 = criterion(pcb_feat[2],targets)
    loss3 = criterion(pcb_feat[3],targets)
    loss4 = criterion(pcb_feat[4],targets)
    loss5 = criterion(pcb_feat[5],targets)
    loss_merge = criterion(pcb_feat[6],targets)
    return loss0,loss1,loss2,loss3,loss4,loss5,loss_merge

def get_pcb_optimizer(model):
    if hasattr(model.module,'base'):
        base_param_ids = set(map(id,model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(),'lr_mult': 0.1},
            {'params': new_params,'lr_mult': 1.0}
        ]
    else:
        param_groups = model.parameters()
    optimizers = torch.optim.SGD(param_groups,lr=0.1,momentum=0.9,weight_decay=5e-4,nesterov=True)
    return optimizers




def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info('start training')

    if device:
       # dist.init_process_group(backend='nccl',init_method='env://')

        model.to(device)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

            model = nn.DataParallel(model)
            # model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
        else:
            if cfg.SOLVER.FP16:
                model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    loss_meter = AverageMeter()
    all_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    pcb_losses = AverageMeter()
    pcb_merge_losses = AverageMeter()

    pcb_optimizer = get_pcb_optimizer(model)
    pcb_scheduler = WarmupMultiStepLR(pcb_optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                  cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        all_loss_meter.reset()
        acc_meter.reset()
        pcb_losses.reset()
        pcb_merge_losses.reset()

        model.train()
        for n_iter, (img, vid) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)

            if cfg.MODEL.IF_USE_PCB:
                score, feat, pcb_out = model(img, target)

                loss = loss_fn(score, feat, target)
                loss0, loss1, loss2, loss3, loss4, loss5,loss_merge = pcb_loss_forward(pcb_feat=pcb_out, targets=target)
                pcb_loss = (loss0 + loss1 + loss2 + loss3 + loss4 + loss5) / 6
                all_loss = loss + 0.5 * pcb_loss + 0.5 * loss_merge

                if cfg.SOLVER.FP16:
                    with amp.scale_loss(all_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    all_loss.backward()


                optimizer.step()
                if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                    for param in center_criterion.parameters():
                        param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                    optimizer_center.step()

                acc = (score.max(1)[1] == target).float().mean()
                loss_meter.update(loss.item(), img.shape[0])
                all_loss_meter.update(all_loss.item(), img.shape[0])
                pcb_losses.update(pcb_loss.item(), img.shape[0])
                pcb_merge_losses.update(loss_merge.item(),img.shape[0])
                acc_meter.update(acc, 1)

                if (n_iter + 1) % log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] All_Loss: {:.3f},Global_Loss: {:.3f},PCB_Loss: {:.3f},Merge_Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        all_loss_meter.avg,loss_meter.avg, pcb_losses.avg,pcb_merge_losses.avg,acc_meter.avg, scheduler.get_lr()[0]))
            else:
                score, feat = model(img, target)

            
                loss = loss_fn(score, feat, target)
                if cfg.SOLVER.FP16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                                    scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                    for param in center_criterion.parameters():
                        param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                    optimizer_center.step()

                acc = (score.max(1)[1] == target).float().mean()
                loss_meter.update(loss.item(), img.shape[0])
                # all_loss_meter.update(all_loss.item(), img.shape[0])
                # pcb_losses.update(pcb_loss.item(), img.shape[0])
                acc_meter.update(acc, 1)

                if (n_iter + 1) % log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] Global_Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg,acc_meter.avg, scheduler.get_lr()[0]))
            

            # if cfg.SOLVER.FP16:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            #     loss.backward(retain_graph=True)

            # loss0, loss1, loss2, loss3, loss4, loss5 = pcb_loss_forward(pcb_feat=pcb_out, targets=target)
            # pcb_loss = (loss0 + loss1 + loss2 + loss3 + loss4 + loss5) / 6


            # all_loss = 0.1 * loss + 0.9 * pcb_loss

            # if cfg.SOLVER.FP16:
            #     with amp.scale_loss(all_loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            #     all_loss.backward()

            # wenli:if use mulit task to train ,may overfit.Deprecated use this method
            # pcb_optimizer.zero_grad()
            # torch.autograd.backward([loss0, loss1, loss2, loss3, loss4, loss5],
            #                         [torch.ones(1)[0].cuda(), torch.ones(1)[0].cuda(), torch.ones(1)[0].cuda(),
            #                          torch.ones(1)[0].cuda(), torch.ones(1)[0].cuda(), torch.ones(1)[0].cuda(),
            #                          torch.ones(1)[0].cuda()])
            # pcb_optimizer.step()


            # optimizer.step()
            # if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
            #     for param in center_criterion.parameters():
            #         param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
            #     optimizer_center.step()

            # acc = (score.max(1)[1] == target).float().mean()
            # loss_meter.update(loss.item(), img.shape[0])
            # all_loss_meter.update(all_loss.item(), img.shape[0])
            # pcb_losses.update(pcb_loss.item(), img.shape[0])
            # acc_meter.update(acc, 1)

            # if (n_iter + 1) % log_period == 0:
            #     logger.info("Epoch[{}] Iteration[{}/{}] All_Loss: {:.3f},Global_Loss: {:.3f},PCB_Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
            #                 .format(epoch, (n_iter + 1), len(train_loader),
            #                         all_loss_meter.avg,loss_meter.avg, pcb_losses.avg,acc_meter.avg, scheduler.get_lr()[0]))
        
        #pcb_scheduler.step()
        scheduler.step()
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))


def do_inference(cfg,
                 model,
                 val_loader_green,
                val_loader_normal,
                 num_query_green,
                 num_query_normal):
    device = "cuda"
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    val_loader = [val_loader_green, val_loader_normal]
    for index, loader in enumerate(val_loader):
        if index == 0:
            subfix = '1'
            reranking_parameter = [30, 2, 0.8]
            evaluator = R1_mAP(cfg,num_query_green, max_rank=200, feat_norm=cfg.TEST.FEAT_NORM,
                               reranking=cfg.TEST.RE_RANKING)
        else:
            subfix = '2'
            reranking_parameter = [30, 2, 0.8]
            evaluator = R1_mAP(cfg,num_query_normal, max_rank=200, feat_norm=cfg.TEST.FEAT_NORM,
                               reranking=cfg.TEST.RE_RANKING)

        evaluator.reset()
        DISTMAT_PATH = os.path.join(cfg.OUTPUT_DIR, "distmat_{}.npy".format(subfix))
        QUERY_PATH = os.path.join(cfg.OUTPUT_DIR, "query_path_{}.npy".format(subfix))
        GALLERY_PATH = os.path.join(cfg.OUTPUT_DIR, "gallery_path_{}.npy".format(subfix))

        for n_iter, (img, pid, camid, imgpath) in enumerate(loader):
            with torch.no_grad():
                img = img.to(device)

                if cfg.TEST.FLIP_FEATS == 'on':
                    # if model_name != efficientnet_b* ,use this feat
                    # feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                    feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                    for i in range(2):
                        if i == 1:
                            inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                            img = img.index_select(3, inv_idx)
                        if cfg.MODEL.IF_USE_PCB:
                            #print("image shape is ", img.shape)
                            f,_ = model(img)
                        else:
                            f = model(img)
                        feat = feat + f
                else:
                    feat,_ = model(img)
                if cfg.MODEL.IF_USE_PCB:
                    _,pcb_feat = model(img)
                    if cfg.TEST.USE_PCB_MERGE_FEAT:
                        evaluator.update_pcb((feat,pcb_feat, imgpath))
                    else:
                        evaluator.update_pcb_split((feat,pcb_feat,imgpath))
                else:
                    evaluator.update((feat, imgpath))
        # if cfg.MODEL.IF_USE_PCB:
        #     if not cfg.TEST.USE_PCB_MERGE_FEAT:
        #         evaluator.update_split()

        data, distmat, img_name_q, img_name_g = evaluator.compute(reranking_parameter)
        np.save(DISTMAT_PATH, distmat)
        np.save(QUERY_PATH, img_name_q)
        np.save(GALLERY_PATH, img_name_g)

        if index == 0:
            data_1 = data

    data_all = {**data_1, **data}
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with open(os.path.join(cfg.OUTPUT_DIR, 'result_{}.json'.format(nowTime)), 'w',encoding='utf-8') as fp:
        json.dump(data_all, fp)


def do_inference_Pseudo(cfg,
                 model,
                val_loader,
                num_query
                 ):
    device = "cuda"

    evaluator = R1_mAP_Pseudo(num_query, max_rank=200, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    reranking_parameter = [14, 4, 0.4]

    model.eval()
    for n_iter, (img, pid, camid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.TEST.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)
                    feat = feat + f
            else:
                feat = model(img)

            evaluator.update((feat, imgpath))

    distmat, img_name_q, img_name_g = evaluator.compute(reranking_parameter)

    return distmat, img_name_q, img_name_g