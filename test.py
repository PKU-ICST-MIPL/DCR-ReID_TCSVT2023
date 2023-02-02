import time
import datetime
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
from tools.eval_metrics import evaluate, evaluate_with_clothes
import cv2
import os


VID_DATASET = ['ccvid']

global_config = None

def concat_all_gather(tensors, num_total_examples):
    '''
    Performs all_gather operation on the provided tensor list.
    '''
    outputs = []
    for tensor in tensors:
        tensor = tensor.cuda()
        tensors_gather = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor)
        output = torch.cat(tensors_gather, dim=0).cpu()
        # truncate the dummy elements added by DistributedInferenceSampler
        outputs.append(output[:num_total_examples])
    return outputs

def to_rgb(img, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
    # print ('-------------------------------------')
    # print (img.shape)
    # print (mean, std)
    mean = np.tile(mean, (img.size(1), img.size(2), 1))
    std = np.tile(std, (img.size(1), img.size(2), 1))
    # print (mean.shape, std.shape, img.shape)
    img = torch.transpose(torch.transpose(img, 0, 1), 1, 2).numpy()
    
    # img = img.view(img.size(1), img.size(2), img.size(0)).numpy()
    # print (img.shape)
    img = (((img * std) + mean) * 255.0)
    img = img[:,:,::-1]
    return img

def show_res(img_s, clos, unclos, conts, aa_s, bb_s, cc_s):
    # print (img_s.shape, clos.shape, unclos.shape, conts.shape)
    for idx,(a,b,c,d, aa, bb, cc) in enumerate(zip(img_s, clos, unclos, conts, aa_s, bb_s, cc_s)):
        
        # print (torch.min(a), torch.max(a), '-------------------', torch.min(b), torch.max(b))
        
        a = to_rgb(a)
        
        # b = to_rgb(b, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
        # c = to_rgb(c, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
        # d = to_rgb(d, np.array([0]), np.array([1]))
        # aa = to_rgb(aa)
        # bb = to_rgb(bb)
        # cc = to_rgb(cc, np.array([0]), np.array([1]))

        b = to_rgb(b, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
        c = to_rgb(c, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
        d = to_rgb(d, np.array([0]), np.array([1]))
        aa = to_rgb(aa, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
        bb = to_rgb(bb, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
        cc = to_rgb(cc, np.array([0]), np.array([1]))
        global global_config
        # print (global_config)
        if not os.path.exists(global_config+'/show/'):
            os.makedirs(global_config+'/show/')
        cv2.imwrite(global_config+'/show/'+str(idx)+'_origin'+'.jpg', a)
        cv2.imwrite(global_config+'/show/'+str(idx)+'_clo'+'.jpg', b)
        cv2.imwrite(global_config+'/show/'+str(idx)+'_unclo'+'.jpg', c)
        cv2.imwrite(global_config+'/show/'+str(idx)+'_cont'+'.jpg', d)

        cv2.imwrite(global_config+'/show/'+str(idx)+'_clo_tar'+'.jpg', aa)
        cv2.imwrite(global_config+'/show/'+str(idx)+'_unclo_tar'+'.jpg', bb)
        cv2.imwrite(global_config+'/show/'+str(idx)+'_cont_tar'+'.jpg', cc)


@torch.no_grad()
def extract_img_feature(model, dataloader, show = False):
    features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    img_s, clos, unclos, conts = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    aa_s, bb_s, cc_s = [], [], []
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids, aa, bb, cc) in enumerate(dataloader):

        flip_imgs = torch.flip(imgs, [3])
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        aa, bb, cc = aa.cuda(), bb.cuda(), cc.cuda()
        batch_features, unclo, cont, clo = model(imgs)
        batch_features_flip, _, _, _ = model(flip_imgs)
        batch_features += batch_features_flip
        batch_features = F.normalize(batch_features, p=2, dim=1)

        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
        
        img_s.append(imgs.cpu())
        aa_s.append(aa.cpu())
        bb_s.append(bb.cpu())
        cc_s.append(cc.cpu())
        clos = torch.cat((clos, clo.cpu()), dim=0)
        unclos = torch.cat((unclos, unclo.cpu()), dim=0)
        conts = torch.cat((conts, cont.cpu()), dim=0)
    features = torch.cat(features, 0)

    img_s = torch.cat(img_s, 0)
    aa_s = torch.cat(aa_s, 0)
    bb_s = torch.cat(bb_s, 0)
    cc_s = torch.cat(cc_s, 0)
    ##########################################################
    # for i in img_s:

    # exit(0)
    ##########################################################
    if show:
        show_res(img_s, clos, unclos, conts, aa_s, bb_s, cc_s)

    return features, pids, camids, clothes_ids


@torch.no_grad()
def extract_vid_feature(model, dataloader, vid2clip_index, data_length):
    # In build_dataloader, each original test video is split into a series of equilong clips.
    # During test, we first extact features for all clips
    clip_features, clip_pids, clip_camids, clip_clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (vids, batch_pids, batch_camids, batch_clothes_ids, aa, bb, cc) in enumerate(dataloader):
        if (batch_idx + 1) % 200==0:
            logger.info("{}/{}".format(batch_idx+1, len(dataloader)))
        vids = vids.cuda()
        batch_features, aa_, bb_, cc_ = model(vids)
        clip_features.append(batch_features.cpu())
        clip_pids = torch.cat((clip_pids, batch_pids.cpu()), dim=0)
        clip_camids = torch.cat((clip_camids, batch_camids.cpu()), dim=0)
        clip_clothes_ids = torch.cat((clip_clothes_ids, batch_clothes_ids.cpu()), dim=0)
    clip_features = torch.cat(clip_features, 0)

    # Gather samples from different GPUs
    clip_features, clip_pids, clip_camids, clip_clothes_ids = \
        concat_all_gather([clip_features, clip_pids, clip_camids, clip_clothes_ids], data_length)

    # Use the averaged feature of all clips split from a video as the representation of this original full-length video
    features = torch.zeros(len(vid2clip_index), clip_features.size(1)).cuda()
    clip_features = clip_features.cuda()
    pids = torch.zeros(len(vid2clip_index))
    camids = torch.zeros(len(vid2clip_index))
    clothes_ids = torch.zeros(len(vid2clip_index))
    for i, idx in enumerate(vid2clip_index):
        features[i] = clip_features[idx[0] : idx[1], :].mean(0)
        features[i] = F.normalize(features[i], p=2, dim=0)
        pids[i] = clip_pids[idx[0]]
        camids[i] = clip_camids[idx[0]]
        clothes_ids[i] = clip_clothes_ids[idx[0]]
    features = features.cpu()

    return features, pids, camids, clothes_ids


def test(config, model, queryloader, galleryloader, dataset):
    logger = logging.getLogger('reid.test')
    global global_config
    global_config = config.OUTPUT
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    # Extract features 
    if config.DATA.DATASET in VID_DATASET:
        qf, q_pids, q_camids, q_clothes_ids = extract_vid_feature(model, queryloader, 
                                                                  dataset.query_vid2clip_index,
                                                                  len(dataset.recombined_query))
        gf, g_pids, g_camids, g_clothes_ids = extract_vid_feature(model, galleryloader, 
                                                                  dataset.gallery_vid2clip_index,
                                                                  len(dataset.recombined_gallery))
    else:
        qf, q_pids, q_camids, q_clothes_ids = extract_img_feature(model, queryloader)
        gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(model, galleryloader)
        # Gather samples from different GPUs
        torch.cuda.empty_cache()
        qf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids], len(dataset.query))
        gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    qf = qf[:, 0:config.MODEL.FEATURE_DIM-config.MODEL.CLOTHES_DIM]
    gf = gf[:, 0:config.MODEL.FEATURE_DIM-config.MODEL.CLOTHES_DIM]
    torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    
    logger.info("Extracted features for query set, obtained {} matrix".format(qf.shape))    
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    since = time.time()
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))
    qf, gf = qf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat[i] = (- torch.mm(qf[i:i+1], gf.t())).cpu()
    distmat = distmat.numpy()
    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    time_elapsed = time.time() - since
    logger.info('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    since = time.time()
    logger.info("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    logger.info('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if config.DATA.DATASET in ['last', 'deepchange', 'vcclothes_sc', 'vcclothes_cc']: return cmc[0]

    logger.info("Computing CMC and mAP only for the same clothes setting")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    logger.info("Computing CMC and mAP only for clothes-changing")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    return cmc[0], mAP


def test_prcc(config, model, queryloader_same, queryloader_diff, galleryloader, dataset):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    # Extract features for query set
    qsf, qs_pids, qs_camids, qs_clothes_ids = extract_img_feature(model, queryloader_same)
    qdf, qd_pids, qd_camids, qd_clothes_ids = extract_img_feature(model, queryloader_diff)
    # Extract features for gallery set
    gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(model, galleryloader)
    # Gather samples from different GPUs
    torch.cuda.empty_cache()
    qsf, qs_pids, qs_camids, qs_clothes_ids = concat_all_gather([qsf, qs_pids, qs_camids, qs_clothes_ids], len(dataset.query_same))
    qdf, qd_pids, qd_camids, qd_clothes_ids = concat_all_gather([qdf, qd_pids, qd_camids, qd_clothes_ids], len(dataset.query_diff))
    gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    qsf = qsf[:, 0:config.MODEL.FEATURE_DIM-config.MODEL.CLOTHES_DIM]
    qdf = qdf[:, 0:config.MODEL.FEATURE_DIM-config.MODEL.CLOTHES_DIM]
    gf = gf[:, 0:config.MODEL.FEATURE_DIM-config.MODEL.CLOTHES_DIM]
    time_elapsed = time.time() - since
    
    logger.info("Extracted features for query set (with same clothes), obtained {} matrix".format(qsf.shape))
    logger.info("Extracted features for query set (with different clothes), obtained {} matrix".format(qdf.shape))
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    m, n, k = qsf.size(0), qdf.size(0), gf.size(0)
    distmat_same = torch.zeros((m, k))
    distmat_diff = torch.zeros((n, k))
    qsf, qdf, gf = qsf.cuda(), qdf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat_same[i] = (- torch.mm(qsf[i:i+1], gf.t())).cpu()
    for i in range(n):
        distmat_diff[i] = (- torch.mm(qdf[i:i+1], gf.t())).cpu()
    distmat_same = distmat_same.numpy()
    distmat_diff = distmat_diff.numpy()
    qs_pids, qs_camids, qs_clothes_ids = qs_pids.numpy(), qs_camids.numpy(), qs_clothes_ids.numpy()
    qd_pids, qd_camids, qd_clothes_ids = qd_pids.numpy(), qd_camids.numpy(), qd_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    logger.info("Computing CMC and mAP for the same clothes setting")
    cmc, mAP = evaluate(distmat_same, qs_pids, g_pids, qs_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    logger.info("Computing CMC and mAP only for clothes changing")
    cmc, mAP = evaluate(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    return cmc[0], mAP