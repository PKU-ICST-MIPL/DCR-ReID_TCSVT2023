import time
import datetime
import logging
import torch
import numpy as np
from apex import amp
from tools.utils import AverageMeter


def train_cal(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
    criterion_clothes, criterion_adv, criterion_shuffle, recon_uncloth, recon_contour, recon_cloth, optimizer, optimizer_cc, trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_clo_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    batch_rec_loss = AverageMeter()
    
    batch_clo_rec_loss = AverageMeter()
    batch_unclo_rec_loss = AverageMeter()
    batch_cont_rec_loss = AverageMeter()

    corrects = AverageMeter()
    clothes_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    clothes_classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids, cloth, _cloth, contour) in enumerate(trainloader):

        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        cloth, _cloth, contour = cloth.cuda(), _cloth.cuda(), contour.cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        features, unclo_img, cont_img, clo_img = model(imgs)
        features_shuffle = features.clone()
        ori_lst = np.arange(0, int(config.DATA.TRAIN_BATCH/config.DATA.NUM_INSTANCES))
        rdn_lst = np.arange(0, int(config.DATA.TRAIN_BATCH/config.DATA.NUM_INSTANCES))
        np.random.shuffle(rdn_lst)
        while np.sum(ori_lst == rdn_lst) > 0:
            np.random.shuffle(rdn_lst)
        ori_lst = np.arange(0, int(config.DATA.TRAIN_BATCH))
        rdn_lst = np.transpose(np.array([rdn_lst*4+i for i in range(0, config.DATA.NUM_INSTANCES)])).reshape(-1)
        features_shuffle[ori_lst, config.MODEL.FEATURE_DIM-config.MODEL.CLOTHES_DIM:config.MODEL.FEATURE_DIM] = features_shuffle[rdn_lst, config.MODEL.FEATURE_DIM-config.MODEL.CLOTHES_DIM:config.MODEL.FEATURE_DIM]
        # Classification
        outputs = classifier(features)
        outputs_shuffle = classifier(features_shuffle)
        pred_clothes = clothes_classifier(features.detach())
        _, preds = torch.max(outputs.data, 1)
        # Update the clothes discriminator
        clothes_loss = criterion_clothes(pred_clothes, clothes_ids)
        if epoch >= config.TRAIN.START_EPOCH_CC:
            optimizer_cc.zero_grad()
            if config.TRAIN.AMP:
                with amp.scale_loss(clothes_loss, optimizer_cc) as scaled_loss:
                    scaled_loss.backward()
            else:
                clothes_loss.backward()
            optimizer_cc.step()
        # Update the backbone
        new_pred_clothes = clothes_classifier(features)
        new_pred_clothes_shuffle = clothes_classifier(features_shuffle)
        _, clothes_preds = torch.max(new_pred_clothes.data, 1)
        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        cla_loss_shuffle = criterion_cla(outputs_shuffle, pids)
        pair_loss = criterion_pair(features, pids)
        pair_loss_shuffle = criterion_pair(features_shuffle, pids)
        adv_loss = criterion_adv(new_pred_clothes, clothes_ids, pos_mask)
        adv_loss_shuffle = criterion_shuffle(new_pred_clothes_shuffle, clothes_ids, pos_mask)
        
        cla_loss = (cla_loss + config.PARA.SHUF_PID_RATIO * cla_loss_shuffle)
        pair_loss= (pair_loss + pair_loss_shuffle)
        adv_loss = (adv_loss + config.PARA.SHUF_ADV_CLO_RATIO * adv_loss_shuffle)

        unclo_loss = recon_uncloth(unclo_img, _cloth)
        cont_loss = recon_contour(cont_img, contour)
        clo_loss = recon_cloth(clo_img, cloth)

        rec_loss = (unclo_loss + cont_loss + clo_loss)
        
        if epoch >= config.TRAIN.START_EPOCH_ADV:
            loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss + config.PARA.RECON_RATIO * rec_loss
        else:
            loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss + config.PARA.RECON_RATIO * rec_loss
        optimizer.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        clothes_corrects.update(torch.sum(clothes_preds == clothes_ids.data).float()/clothes_ids.size(0), clothes_ids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_clo_loss.update(clothes_loss.item(), clothes_ids.size(0))
        batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        batch_rec_loss.update(rec_loss.item(), pids.size(0))

        batch_unclo_rec_loss.update(unclo_loss.item(), pids.size(0))
        batch_clo_rec_loss.update(clo_loss.item(), pids.size(0))
        batch_cont_rec_loss.update(cont_loss.item(), pids.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                  'Time:{batch_time.sum:.1f}s '
                  'Data:{data_time.sum:.1f}s '
                  'ClaLoss:{cla_loss.avg:.4f} '
                  'PairLoss:{pair_loss.avg:.4f} '
                  'CloLoss:{clo_loss.avg:.4f} '
                  'AdvLoss:{adv_loss.avg:.4f} '
                  'RecLoss:{rec_loss.avg:.4f} '
                  'Acc:{acc.avg:.2%} '
                  'CloAcc:{clo_acc.avg:.2%} '
                  'clo_loss:{clo__loss.avg:.4f} '
                  'unclo_loss:{unclo_loss.avg:.4f} '
                  'cont_loss:{cont_loss.avg:.4f} '.format(
                   epoch+1, batch_time=batch_time, data_time=data_time, 
                   cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, 
                   clo_loss=batch_clo_loss, adv_loss=batch_adv_loss, rec_loss=batch_rec_loss,
                   acc=corrects, clo_acc=clothes_corrects,
                   clo__loss=batch_clo_rec_loss, unclo_loss=batch_unclo_rec_loss,cont_loss=batch_cont_rec_loss))


def train_cal_with_memory(config, epoch, model, classifier, criterion_cla, criterion_pair, 
    criterion_adv, optimizer, trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        features = model(imgs)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(features, pids)

        if epoch >= config.TRAIN.START_EPOCH_ADV:
            adv_loss = criterion_adv(features, clothes_ids, pos_mask)
            loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss   
        else:
            loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss  

        optimizer.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        if epoch >= config.TRAIN.START_EPOCH_ADV: 
            batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                'Time:{batch_time.sum:.1f}s '
                'Data:{data_time.sum:.1f}s '
                'ClaLoss:{cla_loss.avg:.4f} '
                'PairLoss:{pair_loss.avg:.4f} '
                'AdvLoss:{adv_loss.avg:.4f} '
                'Acc:{acc.avg:.2%} '.format(
                epoch+1, batch_time=batch_time, data_time=data_time, 
                cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, 
                adv_loss=batch_adv_loss, acc=corrects))