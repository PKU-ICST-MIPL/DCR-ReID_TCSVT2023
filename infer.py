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
import re
from PIL import Image
import tqdm

ltcc_cc_ids = [6,8,9,11,57,58,137,105,106,78,29,79,82,83,31,90,46,59,60,16,17,143,110,118,24,25,52,28,33,63,101,134,142,145,65,20,32,37,74,125,84,104,141,120,72]
ltcc_sc_ids = [1,12,80,139,107,108,112,114,49,55,14,62,67,26,99,126,102,19,23,121,77,119,73,68,38,147,148,113,41,42]
cam2label = {'A': 0, 'B': 1, 'C': 2}

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
    mean = np.tile(mean, (img.size(1), img.size(2), 1))
    std = np.tile(std, (img.size(1), img.size(2), 1))
    # print (mean.shape, std.shape, img.shape)
    img = torch.transpose(torch.transpose(img, 0, 1), 1, 2).numpy()
    
    img = (((img * std) + mean) * 255.0)
    img = img[:,:,::-1]
    return img

def show_res(img_s, clos, unclos, conts, aa_s, bb_s, cc_s):
    for idx,(a,b,c,d, aa, bb, cc) in enumerate(zip(img_s, clos, unclos, conts, aa_s, bb_s, cc_s)):
        a = to_rgb(a)
        b = to_rgb(b, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
        c = to_rgb(c, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
        d = to_rgb(d, np.array([0]), np.array([1]))
        aa = to_rgb(aa, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
        bb = to_rgb(bb, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
        cc = to_rgb(cc, np.array([0]), np.array([1]))
        global global_config
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
def extract_img_feature(model, dataloader):
    features, pids, camids, clothes_ids, paths = [], torch.tensor([]), torch.tensor([]), torch.tensor([]), []

    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids, _, _, _) in enumerate(dataloader):

        flip_imgs = torch.flip(imgs, [3])
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()

        batch_features, unclo, cont, clo = model(imgs)
        batch_features_flip, _, _, _ = model(flip_imgs)
        
        batch_features += batch_features_flip
        batch_features = F.normalize(batch_features, p=2, dim=1)

        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)

    features = torch.cat(features, 0)

    return features, pids, camids, clothes_ids#, paths


@torch.no_grad()
def extract_vid_feature(model, dataloader, vid2clip_index, data_length):
    # In build_dataloader, each original test video is split into a series of equilong clips.
    # During test, we first extact features for all clips
    clip_features, clip_pids, clip_camids, clip_clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (vids, batch_pids, batch_camids, batch_clothes_ids) in enumerate(dataloader):
        if (batch_idx + 1) % 200==0:
            logger.info("{}/{}".format(batch_idx+1, len(dataloader)))
        vids = vids.cuda()
        batch_features = model(vids)
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


def infer(config, model, queryloader, galleryloader, dataset, topk=10):
    logger = logging.getLogger('reid.infer')
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

    q_paths = [data for data, _, _, _ in dataset.query]
    g_paths = [data for data, _, _, _ in dataset.gallery]
    indices = np.argsort(distmat, axis=1)
    for q_idx, query_path in enumerate(q_paths):
        out_file = query_path.replace(config.DATA.ROOT[0:-1], config.OUTPUT).replace('.png', '.txt').replace('.jpg', '.txt')
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
        if q_idx == 0:
            print ("Save to:", os.path.dirname(out_file))
        assert out_file.endswith('.txt')

        with open(out_file, 'w') as fp:
            for g_idx in indices[q_idx, :]:
                fp.write(g_paths[g_idx]+'\n')
        #Visulization
    visualize_ranked_results_inference(distmat, q_paths, g_paths, config, topk=topk)
    return

def visualize_ranked_results_inference(distmat, q_paths, g_paths, config, width=128, height=256, topk=10, GRID_SPACING=10, QUERY_EXTRA_SPACING=90, BW=5):
    """Visualizes ranked results.

    Supports both image-reid and video-reid.

    For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
    saved in folders each containing a tracklet.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    num_q, num_g = distmat.shape

    indices = np.argsort(distmat, axis=1)

    num_cols = topk + 1
    grid_img = 255 * np.ones((height, num_cols * width + (topk - 1) * GRID_SPACING + QUERY_EXTRA_SPACING, 3),
                             dtype=np.uint8)

    for q_idx, qimg_path in tqdm.tqdm(enumerate(q_paths)):
        qimg = cv2.imread(qimg_path)
        qimg = Image.fromarray(cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB))
        qimg = cv2.cvtColor(np.asarray(qimg), cv2.COLOR_RGB2BGR)

        qimg = cv2.resize(qimg, (width, height))
        grid_img[:height, :width, :] = qimg

        if config.DATA.DATASET == 'ltcc':
            pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
            pattern2 = re.compile(r'(\w+)_c')
            q_pid, _, q_camid = map(int, pattern1.search(qimg_path).groups())
            q_cloid = pattern2.search(qimg_path).group(1)
        elif config.DATA.DATASET == 'prcc':
            q_pid = os.path.basename(os.path.dirname(qimg_path))
            q_camid = os.path.basename(os.path.dirname(os.path.dirname(qimg_path)))
            q_camid = cam2label[q_camid]
            q_cloid = q_camid

        if config.INFER.SHOW_CC == True:
            if config.DATA.DATASET == 'ltcc' and q_pid not in ltcc_cc_ids:
                continue
        
        if config.INFER.SHOW_CC == False:
            if config.DATA.DATASET == 'ltcc' and q_pid not in ltcc_sc_ids:
                continue
        
        cnt = 0
        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            gimg_path = g_paths[g_idx]

            if config.DATA.DATASET == 'ltcc':
                
                g_pid, _, g_camid = map(int, pattern1.search(gimg_path).groups())
                g_cloid = pattern2.search(gimg_path).group(1)
            elif config.DATA.DATASET == 'prcc':
                g_pid = os.path.basename(os.path.dirname(gimg_path))
                g_camid = os.path.basename(os.path.dirname(os.path.dirname(gimg_path)))
                g_camid = cam2label[g_camid]
                g_cloid = g_camid
            
            ## ALL show
            if q_pid == g_pid and q_camid == g_camid:
                continue

            ## CC show
            if config.INFER.SHOW_CC == True:
                if q_pid == g_pid and q_cloid == g_cloid:
                    continue
            

            if q_pid == g_pid:
                border_color = (0, 255, 0)
            else:
                border_color = (0, 0, 255)
            
            gimg = cv2.imread(gimg_path)
            gimg = Image.fromarray(cv2.cvtColor(gimg, cv2.COLOR_BGR2RGB))
            gimg = cv2.cvtColor(np.asarray(gimg), cv2.COLOR_RGB2BGR)

            gimg = cv2.resize(gimg, (width, height))
            gimg = cv2.copyMakeBorder(gimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color)
            gimg = cv2.resize(gimg, (width, height))

            start = rank_idx * width + (rank_idx - 1) * GRID_SPACING + QUERY_EXTRA_SPACING
            end = (rank_idx + 1) * width + (rank_idx - 1) * GRID_SPACING + QUERY_EXTRA_SPACING
            grid_img[:height, start:end, :] = gimg

            rank_idx += 1
            if rank_idx > topk:
                break
        if cnt == 0:
            cnt += 1
        cv2.imwrite(qimg_path.replace(config.DATA.ROOT[0:-1], config.OUTPUT), grid_img)


def infer_prcc(config, model, queryloader_same, queryloader_diff, galleryloader, dataset, topk=10):
    logger = logging.getLogger('reid.infer')
    global global_config
    global_config = config.OUTPUT
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



    if config.INFER.SHOW_CC:
        distmat = distmat_diff
        q_paths = [data for data, _, _, _ in dataset.query_diff]
        indices = np.argsort(distmat, axis=1)
    else:
        distmat = distmat_same
        q_paths = [data for data, _, _, _ in dataset.query_same]
        indices = np.argsort(distmat, axis=1)
    g_paths = [data for data, _, _, _ in dataset.gallery]

    for q_idx, query_path in enumerate(q_paths):
        out_file = query_path.replace(config.DATA.ROOT[0:-1], config.OUTPUT).replace('.png', '.txt').replace('.jpg', '.txt')
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
        if q_idx == 0:
            print ("Save to:", os.path.dirname(out_file))
        assert out_file.endswith('.txt')

        with open(out_file, 'w') as fp:
            for g_idx in indices[q_idx, :]:
                fp.write(g_paths[g_idx]+'\n')
        #Visulization
    visualize_ranked_results_inference(distmat, q_paths, g_paths, config, topk=topk)
    return