import math
import ast
import numpy as np
import cv2


def calculate_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # 線分の傾きと切片を計算
    m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else None  # 垂直線の処理
    m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else None

    if m1 is not None and m2 is not None:  # 両方の線が垂直でない場合
        b1 = y1 - m1 * x1
        b2 = y3 - m2 * x3

        # 傾きが同じ場合、平行
        if m1 == m2:
            return None  # 平行な場合、交点はない

        # 交点の x 座標
        x_intersect = (b2 - b1) / (m1 - m2)
        
        # x_intersect が両方の線分内にあるかチェック
        if (min(x1, x2) <= x_intersect <= max(x1, x2)) and (min(x3, x4) <= x_intersect <= max(x3, x4)):
            # 交点の y 座標
            y_intersect = m1 * x_intersect + b1
            return (x_intersect, y_intersect)
        else:
            return None  # 交点が線分の範囲外

    elif m1 is None:  # 線分1が垂直
        x_intersect = x1
        y_intersect = m2 * x_intersect + (y3 - m2 * x3)
        if min(y1, y2) <= y_intersect <= max(y1, y2):
            return (x_intersect, y_intersect)
        else:
            return None  # 交点が線分の範囲外

    elif m2 is None:  # 線分2が垂直
        x_intersect = x3
        y_intersect = m1 * x_intersect + (y1 - m1 * x1)
        if min(y3, y4) <= y_intersect <= max(y3, y4):
            return (x_intersect, y_intersect)
        else:
            return None  # 交点が線分の範囲外

