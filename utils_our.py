'''
M-LSD
Copyright 2021-present NAVER Corp.
Apache License v2.0
'''
import os
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt


def pred_lines(image, interpreter, input_details, output_details, num_of_img=None, num_of_per=None,input_shape=[512, 512], score_thr=0.2, dist_thr=10.0):
    #image = cv2.imread(image, -1)
    #print("type", type(image))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = image.cpu().numpy()  # GPU上のテンソルの場合
    image_rearranged = np.transpose(image, (0, 2, 3, 1))  # [batch_size, height, width, channels]
    # バッチ次元を削除して [height, width, channels] にする
    image = image_rearranged[0]

    image = (image * std) + mean
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        
    # # 画像を表示
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()


    h, w = image.shape[:2]
    size = (h + w)/2
    h_ratio, w_ratio = [h / input_shape[0], w / input_shape[1]]
    resized_img = cv2.resize(image, (input_shape[0],input_shape[1]), interpolation=cv2.INTER_AREA)
    one = np.ones([input_shape[0], input_shape[1], 1])
    # print("resized_img shape:", resized_img)
    # print("one shape:", one.shape)

    if len(resized_img.shape)!=3:
        # print("colorchenge")
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
        resized_image = np.concatenate([resized_img, one], axis=-1)
    else:
        resized_image = np.concatenate([resized_img, one], axis=-1)

    batch_image = np.expand_dims(resized_image, axis=0).astype('float32')

    # オーバーフローチャンネルを除外する
    image_data = batch_image[:, :, :, :3]

    # 新しいオーバーフローチャンネルを作成する（全ての要素が1）
    overflow_channel = np.ones((input_shape[0], input_shape[1], 1), dtype=np.float32)

    # オーバーフローチャンネルを次元を追加して4次元の配列にする
    overflow_channel = np.expand_dims(overflow_channel, axis=0)

    # オーバーフローチャンネルを画像データに連結する
    batch_image_processed = np.concatenate([image_data, overflow_channel], axis=-1)

    # Interpreterにテンソルを設定する
    interpreter.set_tensor(input_details[0]['index'], batch_image_processed)

    #interpreter.set_tensor(input_details[0]['index'], batch_image)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    pts = interpreter.get_tensor(output_details[0]['index'])[0]
    pts_score = interpreter.get_tensor(output_details[1]['index'])[0]
    vmap = interpreter.get_tensor(output_details[2]['index'])[0]

    start = vmap[:,:,:2] #(x1, y1) 256x256x2
    end = vmap[:,:,2:] #(x2, y2) 256x256x2
    test = np.sum((start - end) ** 2, axis=-1)
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1)) # 256x256(maybe size of image) take abs

    segments_list = []
    name_list = []

    for center, score in zip(pts, pts_score):
        y, x = center # =pts
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])
            #img_name = "per{0}_image{1:04d}".format(num_of_per + 1, num_of_img)

            #name_list.append(img_name)


    lines = 2 * np.array(segments_list) # 256 > 512
    #print("until_our", w_ratio, lines)
    if len(lines)==0:
        return lines, image
    lines[:,0] = lines[:,0] * w_ratio #x1
    lines[:,1] = lines[:,1] * h_ratio #y1
    lines[:,2] = lines[:,2] * w_ratio #x2
    lines[:,3] = lines[:,3] * h_ratio #y2

    line_img=draw_lines_on_image(image, lines)
    #line_img=lines_img_save(image, lines)

    #return lines, name_list
    return lines, line_img

def lines_img_save(image, lines):
    h, w = image.shape[:2]
    #lines=_line_nms(lines, h, w)
    img_with_lines = image.copy()
    

    # 各線分を画像に描画
    for line in lines:
        x_start, y_start, x_end, y_end = map(int, line)
        cv2.line(img_with_lines, (x_start, y_start), (x_end, y_end), (255, 0,0), 2)  # 緑色の線で描画   

    return img_with_lines



def draw_lines_on_image(image, lines):
    """
    画像に線分を描画して表示する関数
    :param image: 入力画像 (height, width, channels) の形状
    :param lines: pred_lines 関数で得られた線分のリスト
    """

    shadow_offset=3
    shadow_thickness=3
    line_thickness=2
    #shadow_color=(50,50,50)
    background_color = np.mean(image, axis=(0, 1)).astype(int)
    shadow_color = (background_color * 0.9).astype(int)

    h, w = image.shape[:2]
    

    # 元の画像をコピーする
    img_with_lines = image.copy()
    luminance = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]

    # 最も暗い輝度を持つピクセルのインデックスを取得
    min_luminance_idx = np.unravel_index(np.argmin(luminance), luminance.shape)

    # そのピクセルのRGB値を取得
    darkest_color = tuple(map(int, image[min_luminance_idx]))   
    #print("color", shadow_color)

    # 影の色を画像の平均色から決定する
    avg_color = np.mean(image, axis=(0, 1)).astype(int)  # 画像全体の平均色を取得
    
    # 影の色を少し暗めにする（明度を落とす）
    shadow_color = (avg_color * 0.5).astype(int)  # 50%暗くする
    shadow_color = tuple(shadow_color.tolist())
    
    shadow_color2 = (avg_color * 0.8).astype(int)  # 50%暗くする
    shadow_color2 = tuple(shadow_color2.tolist())    # タプル形式に変換
  

    # for line in lines:
    #     x_start, y_start, x_end, y_end = map(int, line)
    #     cv2.line(img_with_lines, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
 
    lines=_line_nms(lines, h, w)

    # 各線分を画像に描画
    for line in lines:
        x_start, y_start, x_end, y_end, score = map(int, line)
        img_with_lines=draw_shadowed_line(img_with_lines, (x_start, y_start, x_end, y_end), shadow_offset, shadow_thickness+3, shadow_color)      
        cv2.line(img_with_lines, (x_start, y_start), (x_end, y_end), darkest_color, line_thickness)  # 緑色の線で描画   


    return img_with_lines

def _line_nms(lines, h, w):
    # lines is 2D array with shape (n, 4) representing (x1, y1, x2, y2)
    n = lines.shape[0]
    lines_ = np.zeros((n, 5), dtype=np.float32)  # (x1, y1, x2, y2, score)
    select = np.ones(n, dtype=np.float32)
    
    for j in range(n):
        if select[j] == 0:
            continue
        l1 = lines[j, :4]  # (x1, y1, x2, y2)

        # 傾き m1 と切片 b1 を計算
        if l1[2] != l1[0]:  # x1 != x2 の場合
            m1 = (l1[3] - l1[1]) / (l1[2] - l1[0])  # (y2 - y1) / (x2 - x1)
            b1 = l1[1] - m1 * l1[0]  # y1 - m * x1
        else:
            m1 = np.inf  # 垂直線の場合
            b1 = l1[0]  # x = x1 = x2

        score = 1.0  # とりあえず固定のスコアを設定
        lines_[j, :4] = l1
        lines_[j, 4] = score  # スコアも一応保持

        for k in range(j+1, n):
            clear = 0
            l2 = lines[k, :4]  # (x1, y1, x2, y2)

            # 傾き m2 と切片 b2 を計算
            if l2[2] != l2[0]:  # x1 != x2 の場合
                m2 = (l2[3] - l2[1]) / (l2[2] - l2[0])
                b2 = l2[1] - m2 * l2[0]
            else:
                m2 = np.inf  # 垂直線の場合
                b2 = l2[0]  # x = x1 = x2

            # もし2つの線が近ければ、削除
            dist = np.sqrt((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2 + (l1[2] - l2[2])**2 + (l1[3] - l2[3])**2)
            if dist < 20:  # 距離が近ければ削除
                clear = 1

            if clear == 1:
                select[k] = 0

    #print(lines_)

    return lines_

def draw_shadowed_line(image, line, shadow_offset=3, shadow_thickness=3, shadow_color=(30,30,30) ,shadow_blur=5, line_color=(0, 0, 0), line_thickness=2):
    """
    画像に線を描画し、その上と下と横に影をつける。
    
    Parameters:
    - image: 処理対象の画像
    - line: 単一の線 (x1, y1, x2, y2)
    - shadow_offset: 線の周りに影を描画する際のオフセット量
    - shadow_thickness: 影の太さ
    - shadow_blur: 影のぼかし具合
    - line_color: 線の色（デフォルトは黒）
    - line_thickness: 線の太さ
    """
    x1, y1, x2, y2 = line

    # 線の方向ベクトルを計算
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)

    # 線に垂直な方向に影をオフセットするベクトルを計算
    offset_x = -dy / length * shadow_offset
    offset_y = dx / length * shadow_offset

    # 影を描画するための座標 (上側)

    if length==0:
        return image
    shadow_line_upper = (
        int(x1 + offset_x), int(y1 + offset_y),
        int(x2 + offset_x), int(y2 + offset_y)
    )

    # 影を描画するための座標 (下側)
    shadow_line_lower = (
        int(x1 - offset_x), int(y1 - offset_y),
        int(x2 - offset_x), int(y2 - offset_y)
    )

    # 影用の画像を作成（黒の背景）
    shadow_image = np.zeros_like(image)

            # 上側の影の線を描画
    cv2.line(shadow_image, (shadow_line_upper[0], shadow_line_upper[1]), 
             (shadow_line_upper[2], shadow_line_upper[3]), shadow_color, shadow_thickness+3)

    # 下側の影の線を描画
    cv2.line(shadow_image, (shadow_line_lower[0], shadow_line_lower[1]), 
             (shadow_line_lower[2], shadow_line_lower[3]), shadow_color, shadow_thickness+3)


    # 影をぼかす
    shadow_image = cv2.GaussianBlur(shadow_image, (shadow_blur, shadow_blur), 0)


    # 元の画像から影を引いて暗くする
    result_image = cv2.subtract(image, shadow_image)

    # 元の線を描画
    cv2.line(result_image, (x1, y1), (x2, y2), line_color, line_thickness)


       # 上側の影の線を描画
    cv2.line(shadow_image, (shadow_line_upper[0], shadow_line_upper[1]), 
             (shadow_line_upper[2], shadow_line_upper[3]), (100, 100, 100), shadow_thickness)

    # 下側の影の線を描画
    cv2.line(shadow_image, (shadow_line_lower[0], shadow_line_lower[1]), 
             (shadow_line_lower[2], shadow_line_lower[3]), (100, 100, 100), shadow_thickness)



    # 影をぼかす
    shadow_image = cv2.GaussianBlur(shadow_image, (shadow_blur, shadow_blur), 0)

    # 元の画像から影を引いて暗くする
    result_image = cv2.subtract(image, shadow_image)

    # 元の線を描画
    cv2.line(result_image, (x1, y1), (x2, y2), line_color, line_thickness)

    return result_image

