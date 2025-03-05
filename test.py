import numpy as np
import torch
import yaml
from easydict import EasyDict
import argparse
import cv2
import copy
#from collections import OrderedDict
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import tensorflow as tf

from datasets import CustomDataset
from models import (ConvertLayout, Detector, DisplayLayout,_DisplayLayout, display2Dseg, Loss,
                    Reconstruction, _validate_colormap, post_process,MakeLayoutImage)
from scipy.optimize import linear_sum_assignment
from func import (calculate_intersection)
from line_detection import get_lines

import psutil
import GPUtil


def match_by_Hungarian(gt, pred):
    n = len(gt)
    m = len(pred)
    gt = np.array(gt)
    pred = np.array(pred)
    valid = (gt.sum(0) > 0).sum()
    if m == 0:
        raise IOError
    else:
        gt = gt[:, np.newaxis, :, :]
        pred = pred[np.newaxis, :, :, :]
        cost = np.sum((gt+pred) == 2, axis=(2, 3))  # n*m
        row, col = linear_sum_assignment(-1 * cost)
        inter = cost[row, col].sum()
        PE = inter / valid
        return 1 - PE


def evaluate(gtseg, gtdepth, preseg, predepth, evaluate_2D=True, evaluate_3D=False):
    image_iou, image_pe, merror_edge, rmse, us_rmse = 0, 0, 0, 0, 0

    if evaluate_2D:
        # Parse GT polys
        gt_polys_masks = []
        h, w = gtseg.shape
        gt_polys_edges_mask = np.zeros((h, w))
        edge_thickness = 1
        gt_valid_seg = np.ones((h, w))
        labels = np.unique(gtseg)
        for i, label in enumerate(labels):
            gt_poly_mask = gtseg == label
            if label == -1:
                gt_valid_seg[gt_poly_mask] = 0  # zero pad region
            else:
                contours_, hierarchy = cv2.findContours(gt_poly_mask.astype(
                    np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.polylines(gt_polys_edges_mask, contours_, isClosed=True, color=[
                              1.], thickness=edge_thickness)
                gt_polys_masks.append(gt_poly_mask.astype(np.int32))

        def sortPolyBySize(mask):
            return mask.sum()
        gt_polys_masks.sort(key=sortPolyBySize, reverse=True)

        # Parse predictions
        pred_polys_masks = []
        pred_polys_edges_mask = np.zeros((h, w))
        pre_invalid_seg = np.zeros((h, w))
        labels = np.unique(preseg)
        for i, label in enumerate(labels):
            pred_poly_mask = np.logical_and(preseg == label, gt_valid_seg == 1)
            if pred_poly_mask.sum() == 0:
                continue
            if label == -1:
                # zero pad and infinity region
                pre_invalid_seg[pred_poly_mask] = 1
            else:
                contours_, hierarchy = cv2.findContours(pred_poly_mask.astype(
                    np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_SIMPLE
                cv2.polylines(pred_polys_edges_mask, contours_, isClosed=True, color=[
                              1.], thickness=edge_thickness)
                pred_polys_masks.append(pred_poly_mask.astype(np.int32))
        if len(pred_polys_masks) == 0.:
            pred_polys_edges_mask[edge_thickness:-
                                  edge_thickness, edge_thickness:-edge_thickness] = 1
            pred_polys_edges_mask = 1 - pred_polys_edges_mask
            pred_poly_mask = np.ones((h, w))
            pred_polys_masks = [pred_poly_mask]

        pred_polys_masks_cand = copy.copy(pred_polys_masks)

        #print("gt", gt_polys_masks)
        #print("pre", pred_polys_masks)

        # Assign predictions to ground truth polygons
        ordered_preds = []
        for gt_ind, gt_poly_mask in enumerate(gt_polys_masks):
            best_iou_score = 0.3
            best_pred_ind = None
            best_pred_poly_mask = None
            if len(pred_polys_masks_cand) == 0:
                break
            for pred_ind, pred_poly_mask in enumerate(pred_polys_masks_cand):
                gt_pred_add = gt_poly_mask + pred_poly_mask
                inter = np.equal(gt_pred_add, 2.).sum()
                union = np.greater(gt_pred_add, 0.).sum()
                iou_score = inter / union

                if iou_score > best_iou_score:
                    best_iou_score = iou_score
                    best_pred_ind = pred_ind
                    best_pred_poly_mask = pred_poly_mask
            ordered_preds.append(best_pred_poly_mask)

            pred_polys_masks_cand = [pred_poly_mask for pred_ind, pred_poly_mask in enumerate(pred_polys_masks_cand)
                                     if pred_ind != best_pred_ind]
            if best_pred_poly_mask is None:
                continue

        ordered_preds += pred_polys_masks_cand
        class_num = max(len(ordered_preds), len(gt_polys_masks))
        colormap = _validate_colormap(None, class_num + 1)

        # Generate GT poly mask
        gt_layout_mask = np.zeros((h, w))
        gt_layout_mask_colored = np.zeros((h, w, 3))
        for gt_ind, gt_poly_mask in enumerate(gt_polys_masks):
            gt_layout_mask = np.maximum(
                gt_layout_mask, gt_poly_mask * (gt_ind + 1))
            gt_layout_mask_colored += gt_poly_mask[:,
                                                   :, None] * colormap[gt_ind + 1]

        # Generate pred poly mask
        pred_layout_mask = np.zeros((h, w))
        pred_layout_mask_colored = np.zeros((h, w, 3))
        for pred_ind, pred_poly_mask in enumerate(ordered_preds):
            if pred_poly_mask is not None:
                pred_layout_mask = np.maximum(
                    pred_layout_mask, pred_poly_mask * (pred_ind + 1))
                pred_layout_mask_colored += pred_poly_mask[:,
                                                           :, None] * colormap[pred_ind + 1]

        # Calc IOU
        ious = []
        for layout_comp_ind in range(1, len(gt_polys_masks) + 1):
            inter = np.logical_and(np.equal(gt_layout_mask, layout_comp_ind),
                                   np.equal(pred_layout_mask, layout_comp_ind)).sum()
            fp = np.logical_and(np.not_equal(gt_layout_mask, layout_comp_ind),
                                np.equal(pred_layout_mask, layout_comp_ind)).sum()
            fn = np.logical_and(np.equal(gt_layout_mask, layout_comp_ind),
                                np.not_equal(pred_layout_mask, layout_comp_ind)).sum()
            union = inter + fp + fn
            iou = inter / union
            ious.append(iou)

        image_iou = sum(ious) / class_num

        # Calc PE
        image_pe = 1 - np.equal(gt_layout_mask[gt_valid_seg == 1],
                                pred_layout_mask[gt_valid_seg == 1]).sum() / (np.sum(gt_valid_seg == 1))
        # Calc PE by Hungarian
        image_pe_hung = match_by_Hungarian(gt_polys_masks, pred_polys_masks)
        # Calc edge error
        # ignore edges at image borders
        img_bound_mask = np.zeros_like(pred_polys_edges_mask)
        img_bound_mask[10:-10, 10:-10] = 1

        pred_dist_trans = cv2.distanceTransform((img_bound_mask * (1 - pred_polys_edges_mask)).astype(np.uint8),
                                                cv2.DIST_L2, 3)
        gt_dist_trans = cv2.distanceTransform((img_bound_mask * (1 - gt_polys_edges_mask)).astype(np.uint8),
                                              cv2.DIST_L2, 3)

        chamfer_dist = pred_polys_edges_mask * gt_dist_trans + \
            gt_polys_edges_mask * pred_dist_trans
        merror_edge = 0.5 * np.sum(chamfer_dist) / np.sum(
            np.greater(img_bound_mask * (gt_polys_edges_mask), 0))

    # Evaluate in 3D
    if evaluate_3D:
        max_depth = 50
        gt_layout_depth_img_mask = np.greater(gtdepth, 0.)
        gt_layout_depth_img = 1. / gtdepth[gt_layout_depth_img_mask]
        gt_layout_depth_img = np.clip(gt_layout_depth_img, 0, max_depth)
        gt_layout_depth_med = np.median(gt_layout_depth_img)
        # max_depth = np.max(gt_layout_depth_img)
        # may be max_depth should be max depth of all scene
        predepth[predepth == 0] = 1 / max_depth
        pred_layout_depth_img = 1. / predepth[gt_layout_depth_img_mask]
        pred_layout_depth_img = np.clip(pred_layout_depth_img, 0, max_depth)
        pred_layout_depth_med = np.median(pred_layout_depth_img)

        # Calc MSE
        ms_error_image = (pred_layout_depth_img - gt_layout_depth_img) ** 2
        rmse = np.sqrt(np.sum(ms_error_image) /
                       np.sum(gt_layout_depth_img_mask))

        # Calc up to scale MSE
        if np.isnan(pred_layout_depth_med) or pred_layout_depth_med == 0:
            d_scale = 1.
        else:
            d_scale = gt_layout_depth_med / pred_layout_depth_med
        us_ms_error_image = (
            d_scale * pred_layout_depth_img - gt_layout_depth_img) ** 2
        us_rmse = np.sqrt(np.sum(us_ms_error_image) /
                          np.sum(gt_layout_depth_img_mask))

    return image_iou, image_pe, merror_edge, rmse, us_rmse, image_pe_hung




def test_custom_original(model, criterion, iters, inputs, device, cfg, interpreter, input_details, output_details):


    
    # forward
    x = model(inputs['img'])

    loss, loss_stats = criterion(x)
    #print("region", x['line_region'])
    #print("params", x['line_params'])


    # post process on output feature map size, and extract planes, lines, plane params instance and plane params pixelwise
    dt_planes, dt_lines, dt_params3d_instance, dt_params3d_pixelwise,img_height = post_process(x, Mnms=1)
    
    # reconstruction according to detection results
    for i in range(1):
        (_ups, _downs, _attribution, _params_layout), (ups, downs, attribution, params_layout), ( pfloor, pceiling) = Reconstruction(
            dt_planes[i],
            dt_params3d_instance[i],
            dt_lines[i],
            K=inputs['intri'][i].cpu().numpy(),
            size=(720, 1280),
            threshold=(0.3, 0.05, 0.05, 0.3))
        
        #print("param_layout", params_layout)


        _input_ups, _input_downs= np.copy(_ups), np.copy(_downs)
        input_ups, input_downs= np.copy(ups), np.copy(downs)
        #mlsd_lines_sphere, mlsd_lines, width, height, line_img=get_lines(inputs['img'], interpreter, input_details, output_details)



        #re_ups, re_downs, flag=reline(input_ups, input_downs,input_ups, input_downs,mlsd_lines,ver_lines)
        #print("flag", flag)


        re_ups, re_downs, flag=ups,downs, 0
        _re_ups, _re_downs, _flag=_ups, _downs, 0
        # convert no opt results to segmentation and depth map and evaluate results
        _seg, _depth, _, _polys = ConvertLayout(
            inputs['img'][i], _ups, _downs, _attribution,
            K=inputs['intri'][i].cpu().numpy(), pwalls=_params_layout,
            pfloor=pfloor, pceiling=pceiling,
            ixy1map=inputs['ixy1map'][i].cpu().numpy(),
            valid=inputs['iseg'][i].cpu().numpy(),
            oxy1map=inputs['oxy1map'][i].cpu().numpy(), pixelwise=None)
        
        _re_seg, _re_depth, re_img, re_polys = ConvertLayout(
            inputs['img'][i], _re_ups, _re_downs, _attribution,
            K=inputs['intri'][i].cpu().numpy(), pwalls=params_layout,
            pfloor=pfloor, pceiling=pceiling,
            ixy1map=inputs['ixy1map'][i].cpu().numpy(),
            valid=inputs['iseg'][i].cpu().numpy(),
            oxy1map=inputs['oxy1map'][i].cpu().numpy(), pixelwise=None)

        #attribution=3

        # convert opt results to segmentation and depth map and evaluate results
        seg, depth, img, polys = ConvertLayout(
            inputs['img'][i], ups, downs, attribution,
            K=inputs['intri'][i].cpu().numpy(), pwalls=params_layout,
            pfloor=pfloor, pceiling=pceiling,
            ixy1map=inputs['ixy1map'][i].cpu().numpy(),
            valid=inputs['iseg'][i].cpu().numpy(),
            oxy1map=inputs['oxy1map'][i].cpu().numpy(), pixelwise=None)
        
                    # convert opt results to segmentation and depth map and evaluate results
        re_seg, re_depth, re_img, re_polys = ConvertLayout(
            inputs['img'][i], re_ups, re_downs, attribution,
            K=inputs['intri'][i].cpu().numpy(), pwalls=params_layout,
            pfloor=pfloor, pceiling=pceiling,
            ixy1map=inputs['ixy1map'][i].cpu().numpy(),
            valid=inputs['iseg'][i].cpu().numpy(),
            oxy1map=inputs['oxy1map'][i].cpu().numpy(), pixelwise=None)

        
        res = evaluate(inputs['iseg'][i].cpu().numpy(), None, seg, None)
        #print("4")



        torch.set_printoptions(threshold=float('inf'))

        # スケーリング
        scaled_depth_map = scale_depth_map(depth)


        width = inputs['img'].shape[1]
        height = inputs['img'].shape[0]

        # if cfg.visual:
            # # display layout
            # _DisplayLayout(img, seg, depth, polys, _seg, _depth, _polys, inputs['iseg'][i].cpu().numpy(),
            #     inputs['ilbox'][i].cpu().numpy(), iters)
            # print("custom")
        if True:
            # display layout
            input_img1=np.copy(img)
            input_img2=np.copy(img)

            gt_img=MakeLayoutImage(input_img1,  inputs['iseg'][i].cpu().numpy(), inputs['ilbox'][i].cpu().numpy(), iters)
            opt_img=MakeLayoutImage(input_img2, seg, inputs['ilbox'][i].cpu().numpy(), iters)



        return res, opt_img, img


def test_custom_line(model, criterion, iters, inputs, device, cfg, interpreter, input_details, output_details):

    input_img = inputs['img'][0].cpu().numpy().transpose([1, 2, 0])
    mean, std = np.array([0.485, 0.456, 0.406]), np.array(
        [0.229, 0.224, 0.225])
    input_img = ((input_img * std) + mean)*255
    input_img = input_img[:, :, ::-1]
    
    #線を入れてみる
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    mlsd_lines_sphere, mlsd_lines, width, height, line_img=get_lines(inputs['img'], interpreter, input_details, output_details)
    # 1. 画像を浮動小数点数に変換し、[0, 1] の範囲に戻す
    image = line_img.astype(np.float32) / 255.0
    # 2. 標準化の逆変換を行う
    image = (image - mean) / std
    # 3. バッチ次元を追加する
    image = np.expand_dims(image, axis=0)  # [1, height, width, channels]
    # 4. テンソルの次元を [batch_size, channels, height, width] に戻す
    image = np.transpose(image, (0, 3, 1, 2))
    # 5. numpy から torch テンソルに戻す
    image = torch.from_numpy(image).float()
    # 6. GPUに再度移動させる場合
    inputs['img'] = image.cuda()

    #x = model(inputs_line_img)

    # forward
    x = model(inputs['img'])

    loss, loss_stats = criterion(x)

    # post process on output feature map size, and extract planes, lines, plane params instance and plane params pixelwise
    dt_planes, dt_lines, dt_params3d_instance, dt_params3d_pixelwise,img_height = post_process(x, Mnms=1)
    
    # dt_line : [xs, ys, reg_alpha, scores]
                #dt_linesを描写してみる
    ver_lines=[]
    height=line_img.shape[0]
    width=line_img.shape[1]

    for line in dt_lines[0]:
        if line[3]==1:
            #x=my+b
            rate=height/img_height
            b=line[1]*rate
            m=line[0]

            # 線の始点と終点を計算
            # x = 0 のときの y 座標 (b)
            x_start, y_start = int(b), 0

            # x = 画像の幅 のときの y 座標 (x=my+b)
            x_end, y_end = int(m*height+b), height
            ver_lines.append([x_start, y_start,x_end, y_end])

            for mlsd_line in mlsd_lines:
                min_intersection=[0,-float('inf')]
                minflag=False
                max_intersection=[0,  float('inf')]
                maxflag=False
                intersection = calculate_intersection((x_start, y_start), (x_end, y_end), (mlsd_line[0],mlsd_line[1] ),(mlsd_line[2],mlsd_line[3]))
                if intersection:
                    if (min_intersection[1]>intersection[1]):
                        min_intersection=intersection
                        minflag=True
                    if (max_intersection[1]>intersection[1]):
                        max_intersection=intersection
                        maxflag=True
                if minflag:
                    x_start, y_start=min_intersection
                if maxflag:
                    x_end, y_end=max_intersection

            np.append(mlsd_lines,[x_start, y_start,x_end, y_end])


    # 1. 画像を浮動小数点数に変換し、[0, 1] の範囲に戻す
    image = line_img.astype(np.float32) / 255.0
    # 2. 標準化の逆変換を行う
    image = (image - mean) / std
    # 3. バッチ次元を追加する
    image = np.expand_dims(image, axis=0)  # [1, height, width, channels]
    # 4. テンソルの次元を [batch_size, channels, height, width] に戻す
    image = np.transpose(image, (0, 3, 1, 2))
    # 5. numpy から torch テンソルに戻す
    image = torch.from_numpy(image).float()
    # 6. GPUに再度移動させる場合
    inputs['img'] = image.cuda()

    line_img = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'results/line/{iters}_lines.png', line_img.astype(np.uint8) )





    # reconstruction according to detection results
    for i in range(1):
        (_ups, _downs, _attribution, _params_layout), (ups, downs, attribution, params_layout), ( pfloor, pceiling) = Reconstruction(
            dt_planes[i],
            dt_params3d_instance[i],
            dt_lines[i],
            K=inputs['intri'][i].cpu().numpy(),
            size=(720, 1280),
            threshold=(0.3, 0.05, 0.05, 0.3))
        
        #print("param_layout", params_layout)


        _input_ups, _input_downs= np.copy(_ups), np.copy(_downs)
        input_ups, input_downs= np.copy(ups), np.copy(downs)
        #mlsd_lines_sphere, mlsd_lines, width, height, line_img=get_lines(inputs['img'], interpreter, input_details, output_details)



        #re_ups, re_downs, flag=reline(input_ups, input_downs,input_ups, input_downs,mlsd_lines,ver_lines)
        #print("flag", flag)


        re_ups, re_downs, flag=ups,downs, 0
        _re_ups, _re_downs, _flag=_ups, _downs, 0
        # convert no opt results to segmentation and depth map and evaluate results
        _seg, _depth, _, _polys = ConvertLayout(
            inputs['img'][i], _ups, _downs, _attribution,
            K=inputs['intri'][i].cpu().numpy(), pwalls=_params_layout,
            pfloor=pfloor, pceiling=pceiling,
            ixy1map=inputs['ixy1map'][i].cpu().numpy(),
            valid=inputs['iseg'][i].cpu().numpy(),
            oxy1map=inputs['oxy1map'][i].cpu().numpy(), pixelwise=None)
        
        _re_seg, _re_depth, re_img, re_polys = ConvertLayout(
            inputs['img'][i], _re_ups, _re_downs, _attribution,
            K=inputs['intri'][i].cpu().numpy(), pwalls=params_layout,
            pfloor=pfloor, pceiling=pceiling,
            ixy1map=inputs['ixy1map'][i].cpu().numpy(),
            valid=inputs['iseg'][i].cpu().numpy(),
            oxy1map=inputs['oxy1map'][i].cpu().numpy(), pixelwise=None)

        #attribution=3

        # convert opt results to segmentation and depth map and evaluate results
        seg, depth, img, polys = ConvertLayout(
            inputs['img'][i], ups, downs, attribution,
            K=inputs['intri'][i].cpu().numpy(), pwalls=params_layout,
            pfloor=pfloor, pceiling=pceiling,
            ixy1map=inputs['ixy1map'][i].cpu().numpy(),
            valid=inputs['iseg'][i].cpu().numpy(),
            oxy1map=inputs['oxy1map'][i].cpu().numpy(), pixelwise=None)
        
                    # convert opt results to segmentation and depth map and evaluate results
        re_seg, re_depth, re_img, re_polys = ConvertLayout(
            inputs['img'][i], re_ups, re_downs, attribution,
            K=inputs['intri'][i].cpu().numpy(), pwalls=params_layout,
            pfloor=pfloor, pceiling=pceiling,
            ixy1map=inputs['ixy1map'][i].cpu().numpy(),
            valid=inputs['iseg'][i].cpu().numpy(),
            oxy1map=inputs['oxy1map'][i].cpu().numpy(), pixelwise=None)

        
        res = evaluate(inputs['iseg'][i].cpu().numpy(), None, seg, None)
        #print("4")

        #print("polys", polys)
        torch.set_printoptions(threshold=float('inf'))
        #print(inputs['iseg'][i])

        #print("depth size", len(depth))
        # スケーリング
        scaled_depth_map = scale_depth_map(depth)



        width = inputs['img'].shape[1]
        height = inputs['img'].shape[0]

        # 画像として保存
        #cv2.imwrite('depth_map.png', colored_depth_map)
        cv2.imwrite(f'results/depth/{iters}_depth_map.png', scaled_depth_map)

        #visualize_3d_model(input_img*255, scaled_depth_map)
        #generate_3d_model_from_depth(scaled_depth_map, re_seg, input_img*255)

        # if cfg.visual:
            # # display layout
            # _DisplayLayout(img, seg, depth, polys, _seg, _depth, _polys, inputs['iseg'][i].cpu().numpy(),
            #     inputs['ilbox'][i].cpu().numpy(), iters)
            # print("custom")
        if cfg.visual:
            # display layout
            line_seg=MakeLayoutImage(input_img, seg,inputs['ilbox'][i].cpu().numpy(), iters)

 
    return res, line_seg, line_img


import time  # 処理のタイミングを計測するために使用

# GPUメモリ使用量を取得する関数
def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # 使用中のメモリ (MB単位)
        reserved = torch.cuda.memory_reserved() / 1024**2  # 確保済みのメモリ (MB単位)
        return allocated, reserved
    return 0, 0

import psutil

# メモリ使用量を取得する関数
def get_cpu_memory_usage():
    process = psutil.Process()  # 現在のプロセスを取得
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)  # 使用メモリ量 (MB単位)


def test_painting(model, criterion, dataloader, device, cfg, interpreter, input_details, output_details):
    model.eval()
    #print("data", dataloader)
    results=[]
    #print("c")
    IoU_ori=[]
    PE_ori=[]
    EE_ori=[]

    IoU=[]
    PE=[]
    EE=[]
    # 結果記録用リスト
    cpu_memories = []
    gpu_allocated_memories = []
    gpu_reserved_memories = []


    
    max_cpu, max_gpu = 0, 0  # 最大使用量
    total_cpu, total_gpu = 0, 0  # 合計使用量
    
    for iters, inputs in enumerate(dataloader):
        #print("input",inputs)
        print(f'{iters}/{len(dataloader)}')
 

        #print("type", type(inputs))
        #results=[]
        # set device
        for key, value in inputs.items():
            inputs[key] = value.to(device)
        

        original_res, original_img, img=test_custom_original(model, criterion, iters, inputs, device, cfg, interpreter, input_details, output_details)
        line_res, lineres_img, line_img = test_custom_line(model, criterion, iters, inputs, device, cfg, interpreter, input_details, output_details) 
        
        # メモリ使用量を取得
        allocated, reserved = get_gpu_memory_usage()
        cpu_memory = get_cpu_memory_usage()

        # メモリ使用量をリストに記録
        gpu_allocated_memories.append(allocated)
        gpu_reserved_memories.append(reserved)
        cpu_memories.append(cpu_memory)

        print(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
        print(f"CPU Memory - Used: {cpu_memory:.2f} MB")
    

        
        # 一時停止 (例: 実行の確認のため)
        time.sleep(1)


        input_img=np.copy(img)
        gt_img=MakeLayoutImage(input_img,  inputs['iseg'][0].cpu().numpy(), inputs['ilbox'][0].cpu().numpy(), iters)
        
        results.append([original_res, line_res])
        mean_res=np.mean(np.array(results), axis=0)
        print("result", mean_res)

        #cv2.imwrite(f'figures/line/{iters}.png', line_img)

        IoU.append(line_res[0])
        PE.append(line_res[1])
        EE.append(line_res[2])

        IoU_ori.append(original_res[0])
        PE_ori.append(original_res[1])
        EE_ori.append(original_res[2])

        cv2.imwrite(f'results/{iters}_output.png',
            np.concatenate([img[:360]/255, line_img[:360]/255, gt_img[:360], original_img[:360], lineres_img[:360]], axis=1) * 255)
        # cv2.imwrite(f'figures/surveyA/{iters}.png',np.concatenate([gt_img[:360], original_img[:360]], axis=1) * 255)
        # cv2.imwrite(f'figures/surveyB/{iters}.png',np.concatenate([gt_img[:360], lineres_img[:360]], axis=1) * 255)
        # cv2.imwrite(f'figures/{iters}_p.png',np.concatenate([img[:360]/255, gt_img[:360]], axis=1) * 255)

        # cv2.imwrite(f'figures/images/{iters}.png',np.concatenate([img[:360]], axis=1))
        # cv2.imwrite(f'figures/images/{iters}_gt.png',np.concatenate([gt_img[:360]], axis=1) * 255)
        # cv2.imwrite(f'figures/images/{iters}_A.png',np.concatenate([original_img[:360]], axis=1) * 255)
        # cv2.imwrite(f'figures/images/{iters}_B.png',np.concatenate([lineres_img[:360]], axis=1) * 255)
 
        #計算結果の後処理
        del original_res, original_img, img, line_res, lineres_img, line_img

        torch.cuda.empty_cache()    

    # 処理全体の統計を計算
    max_gpu_allocated = max(gpu_allocated_memories)
    max_gpu_reserved = max(gpu_reserved_memories)
    avg_gpu_allocated = sum(gpu_allocated_memories) / len(gpu_allocated_memories)
    avg_gpu_reserved = sum(gpu_reserved_memories) / len(gpu_reserved_memories)
    max_cpu_memory = max(cpu_memories)
    avg_cpu_memory = sum(cpu_memories) / len(cpu_memories)

    # 結果を出力
    print("\nSummary:")
    print(f"Maximum GPU Allocated Memory: {max_gpu_allocated:.2f} MB")
    print(f"Maximum GPU Reserved Memory: {max_gpu_reserved:.2f} MB")
    print(f"Average GPU Allocated Memory per image: {avg_gpu_allocated:.2f} MB")
    print(f"Average GPU Reserved Memory per image: {avg_gpu_reserved:.2f} MB")
    print(f"Maximum CPU Memory Used: {max_cpu_memory:.2f} MB")
    print(f"Average CPU Memory Used per image: {avg_cpu_memory:.2f} MB")

    calculate_statistics(IoU)
    calculate_statistics(PE)
    calculate_statistics(EE)

import scipy.stats as stats

def calculate_statistics(data, confidence=0.95):
    """
    Calculate mean, variance, standard deviation, and confidence interval for given data.

    Parameters:
        data (list or numpy array): Input data array.
        confidence (float): Confidence level for the interval, default is 0.95 (95%).

    Returns:
        dict: A dictionary with mean, variance, std_dev, and confidence_interval.
    """
    # Convert data to numpy array if not already
    data = np.array(data)
    
    # Calculate basic statistics
    mean = np.mean(data)
    variance = np.var(data, ddof=1)  # ddof=1 for sample variance
    std_dev = np.std(data, ddof=1)
    
    # Calculate confidence interval
    n = len(data)
    se = std_dev / np.sqrt(n)  # Standard error of the mean
    margin_of_error = stats.t.ppf((1 + confidence) / 2, df=n-1) * se
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)

    # Print results
    print("Statistics:")
    print(f"Mean: {mean:.3f}")
    print(f"Variance: {variance:.3f}")
    print(f"Standard Deviation: {std_dev:.3f}")
    print(f"{int(confidence * 100)}% Confidence Interval: {confidence_interval}")

    
    # Return all results as a dictionary
    return {
        "mean": mean,
        "variance": variance,
        "std_dev": std_dev,
        "confidence_interval": confidence_interval
    }



def scale_depth_map(depth_map):
        min_val = np.min(depth_map)
        max_val = np.max(depth_map)
        scaled_depth_map = 255 * (depth_map - min_val) / (max_val - min_val)
        return scaled_depth_map.astype(np.uint8)

    # 深度マップのカラーマッピングを行う関数
def colorize_depth_map(scaled_depth_map):
        scaled_depth_map = scaled_depth_map.astype(np.float32)
        gray_depth_map = cv2.cvtColor(scaled_depth_map, cv2.COLOR_BGR2GRAY)
        return gray_depth_map

def get_edge(segs_gt, label, input_img):
    _segs_gt = []
    for i in np.unique(segs_gt):
        if i == -1:
            continue
        else:
            _segs_gt.append(segs_gt==i)

    _segs_gt = np.array(_segs_gt).astype(np.int)
    print("_segs", _segs_gt)
    img = cv2.applyColorMap(input_img, cv2.COLORMAP_HOT)
    #img /= 255
    # plt.imshow(img)
    # plt.title("img")
    # plt.axis("off")
    # plt.show()

    color1 = np.arange(2, len(_segs_gt)+2)
    for i in range(len(color1)):
        if label[i] == 1:
            color1[i] = 0
        elif label[i] == 2:
            color1[i] = 1

    print(color1)


    for i in range(len(color1)):
        print("i", i)
        alpha_fill = (_segs_gt[i]==1)[..., None].astype(np.float32)
        sx = cv2.Sobel(alpha_fill, cv2.CV_32F, 1, 0, ksize=5)
        sy = cv2.Sobel(alpha_fill, cv2.CV_32F, 0, 1, ksize=5)
        alpha_edge = (sx ** 2 + sy ** 2) ** 0.5
        alpha_edge /= max(0.001, np.max(alpha_edge))
        alpha_edge = alpha_edge[..., None]

        #ssprint("edge",alpha_edge)
        #if alpha_edge==[[[0]]]:
        #     continue
        alpha_fill *= 0.5

        #print("no")
        color = (0,0,0)
        #img = img * (1 - alpha_fill) + alpha_fill * color
        img = img * (1 - alpha_edge) + alpha_edge * color

    # plt.imshow(img)
    # plt.title("img")
    # plt.axis("off")
    # plt.show()

    cv2.imwrite('depth_map_edge.png', img)

    return img




def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--room-layout-model', type=str, required=True, help='the room layout estimation model')
    parser.add_argument('--visual', default=True, action='store_true', help='whether to visual the results')
    parser.add_argument('--exam', action='store_true', help='test one example on nyu303 dataset')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument(
    "-lm",
    "--line_model_path",
    default="tflite_models/M-LSD_512_large_fp32.tflite",
    type=str,
    help="path to tflite model",
    )
    parser.add_argument(
    "-i", "--image-path", help="Path to the input folder", required=True
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    with open('cfg.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

        cfg = EasyDict(config)
    args = parse()
    cfg.update(vars(args))

    input_path = args.image_path

    if cfg.exam:
        assert cfg.data == 'NYU303', 'provide one example of nyu303 to test'
    #  dataset
    dataset = CustomDataset(cfg.Dataset.CUSTOM, 'test', input_path)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=cfg.num_workers)

    # create network
    model = Detector()


    # compute loss
    criterion = Loss(cfg.Weights)

    # set data parallel
    # if cfg.num_gpus > 1 and torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model)

    # reload weights
    if cfg.room_layout_model:
        state_dict = torch.load(cfg.room_layout_model,
                                map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)

    model_path = args.line_model_path

    # Load tflite model
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    test_painting(model, criterion, dataloader, device, cfg,interpreter, input_details, output_details)

