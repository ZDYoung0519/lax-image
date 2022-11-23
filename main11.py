import os
import math
import cv2
import numpy as np
from glob import glob
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


n_theta = 48        # 划线的个数
R = 200             # 划线的长度，单位是像素的长度
h_cm, w_cm = 22.61, 32.21       # 图片的高和宽，单位是cm
h, w = 641, 913                 # 图片的高和宽，单位是像素
xc_cm, yc_cm = 15.63, 11.32     # 中心点，单位是cm

vis = True
save = True
src_path = './imgs'
save_path = './results'


if not os.path.exists(src_path):
    os.mkdir(save_path)


def get_start_end_pt(ptx, pty, xc, yc):
    if len(ptx) == 0:
        return None, None, None, None, 0
    dist = np.sqrt((ptx-xc)**2 + (pty-yc)**2)
    idx = np.argsort(-dist)
    ptx, pty = ptx[idx], pty[idx]
    thickness = np.sqrt((ptx[-1]-ptx[0])**2 + (pty[-1]-pty[0])**2)
    return ptx[0], pty[0], ptx[-1], pty[-1], thickness


def img_kernel_filter(img, kernel_size=7, threshold=None):
    dst = img.copy()
    kernel = np.ones((kernel_size, kernel_size), np.float32)
    img = img.copy() / 255
    img = 1 - img
    mask = cv2.filter2D(img, -1, kernel)

    if threshold is None:
        threshold = kernel_size**2 / 2
    dst[mask >= threshold] = 0.
    dst[mask <= threshold] = 255
    return dst.astype(np.uint8)


def generate_mask(h, w, R):
    mask = np.empty([h, w], dtype=bool)
    for i in range(h):
        for j in range(w):
            dist = np.sqrt((xc - j)**2 + (yc - i)**2)
            if dist > R:
                mask[i, j] = True
            else:
                mask[i, j] = False
    return mask


def process_points(ptx, pty, xc, yc, shift_x=50, shift_y=0):
    idx = np.argsort(ptx)
    ptx = ptx[idx]
    pty = pty[idx]
    if ptx.std() >= 20:
        temp = ptx - xc + shift_x
    else:
        temp = yc - pty - shift_y
    positive = temp >= 0
    negative = temp < 0
    if positive.sum() > negative.sum():
        return ptx[positive], pty[positive]
    else:
        return ptx[negative], pty[negative]


if __name__ == '__main__':
    img_dirs = src_path
    scale_factor = np.mean([h_cm / h, w_cm / w])

    xc, yc = xc_cm / w_cm * w - 1, yc_cm / h_cm * h - 1
    xc, yc = int(np.round(xc)), int(np.round(yc))

    theta_min, theta_max, n_theta = 0, 2 * math.pi, n_theta
    THETA = np.arange(theta_min, theta_max, (theta_max - theta_min) / n_theta)
    img_threshold = 150

    mask = generate_mask(h, w, R)
    img_list = glob(os.path.join(src_path, '*.bmp'))


    results = pd.DataFrame(columns=['IMG', 'thickness'])
    for img_file in img_list[1500:]:
        img_ori = cv2.imread(img_file)
        img = img_ori.copy()

        # rgb to gray
        img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        # keep the main img
        img_gray[mask] = 255

        # Thresholding
        img_gray[img_gray <= img_threshold] = 0
        img_gray[img_gray > img_threshold] = 255

        # filter img
        img_gray = img_kernel_filter(img_gray)
        pty, ptx = np.where(img_gray <= img_threshold)

        pt_theta = []
        for i in range(len(ptx)):
            b = yc - pty[i]
            a = ptx[i] - xc
            theta = math.atan(b/a)
            if a > 0 and b > 0:
                pt_theta.append(theta)
            elif a > 0 and b < 0:
                pt_theta.append(theta + 2 * math.pi)
            else:
                pt_theta.append(theta + math.pi)
        pt_theta = np.array(pt_theta)

        thickness = []
        pts = []
        for theta in THETA:
            idx = np.abs(pt_theta - theta) <= 0.01
            ptx_, pty_ = ptx[idx], pty[idx]
            ptx_, pty_ = process_points(ptx_, pty_, xc, yc)
            ptx1, pty1, ptx2, pty2, th = get_start_end_pt(ptx_, pty_, xc, yc)
            pts.append([ptx1, pty1, ptx2, pty2])
            thickness.append(th * scale_factor)

        thickness = np.array(thickness)
        thickness_avg = np.mean(thickness)
        pts = np.array(pts)
        for i, theta in enumerate(THETA):
            if vis:
                # draw line
                x_theta, y_theta = int(xc + R * math.cos(theta)), int(yc - R * math.sin(theta))
                cv2.line(img, (xc, yc), (x_theta, y_theta), (155, 126, 69), 1, cv2.LINE_AA)

                # draw thickness
                ptx1, pty1, ptx2, pty2 = pts[i]
                cv2.line(img, (ptx1, pty1), (ptx2, pty2), (0, 0, 255), 2, cv2.LINE_AA)

        if vis:
            cv2.imshow('img', img)
            cv2.imshow('img gray', img_gray)
            cv2.waitKey(1)
        if save:
            basename = os.path.basename(img_file)
            cv2.imwrite(os.path.join(save_path, basename), img)

            results = results.append(pd.Series([basename, thickness_avg], index=results.columns), ignore_index=True)
    results.to_csv('./res.csv', index=0)
