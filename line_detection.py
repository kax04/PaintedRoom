import os

import cv2
import numpy as np
import torch
from utils_our import pred_lines
from PIL import Image, ImageDraw


def get_lines(
    image, interpreter, input_details, output_details, num_of_img=None, num_of_per=None
):
    """MLSDで線分を検出する関数

    Args:
        image_path (str): 画像のパス
        interpreter : tfliteモデル
        input_details : tfliteモデルの入力
        output_details : tfliteモデルの出力
        num_of_img : 画像の番号
        num_of_per : 透視画像の番号(0,1,2)

    Returns:
        mlsd_lines_sphere : 球面座標系で表された線分のリスト
        mlsd_lines : 画像座標系で表された線分のリスト
        width : 画像の幅
        height : 画像の高さ
    """
    #print("image", image)
    #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #print(image)
    if torch.max(image) <= 1:
        image *= 255

    width = image.shape[1]
    height = image.shape[0]
    scale_w = np.maximum(width, height)
    scale_h = scale_w



    mlsd_lines, line_img = pred_lines(
        image,
        interpreter,
        input_details,
        output_details,
        num_of_img=None, 
        num_of_per=None,
        input_shape=[512, 512],
        score_thr=0.55,
        dist_thr=0.0,
    )

    if len(mlsd_lines)==0:
        return [], mlsd_lines, width, height, line_img

    #print("mlsd_line", mlsd_lines)
    mlsd_lines_sphere = mlsd_lines.copy()

    # 画像の大きさに対して線分を正規化
    mlsd_lines_sphere[:, 0] -= width / 2.0
    mlsd_lines_sphere[:, 1] -= height / 2.0
    mlsd_lines_sphere[:, 2] -= width / 2.0
    mlsd_lines_sphere[:, 3] -= height / 2.0
    mlsd_lines_sphere[:, 0] /= scale_w / 2.0
    mlsd_lines_sphere[:, 1] /= scale_h / 2.0
    mlsd_lines_sphere[:, 2] /= scale_w / 2.0
    mlsd_lines_sphere[:, 3] /= scale_h / 2.0
    mlsd_lines_sphere[:, 1] *= -1
    mlsd_lines_sphere[:, 3] *= -1
    return mlsd_lines_sphere, mlsd_lines, width, height, line_img


