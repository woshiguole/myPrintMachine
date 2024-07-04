# -*-coding:gbk*-
import math
import os
import cv2
import time
import numpy as np
# import line_profiler
import matplotlib.pyplot as plt
import threading
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk  # ͼ��ؼ�
from pypylon import pylon
from pypylon import genicam
import modbus_tk.defines as cst
from modbus_tk import modbus_tcp

# ����Ϊȫ�ֱ�����������ɫ������̺�ͼƬչʾ������ͬʱʹ��
# �б���Ϊ2����ʾ2�����
ImageAcquired = [np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.uint8)]  #rgb3ͨ��10mal10
ResultMessage = ['', '']
ClientToServer = []
CompletedAcquisitionCnt = [0, 0]
FailedAcquisitionCnt = [0, 0]
CompletedDetectionCnt = [0, 0]
FailedDetectionCnt = [0, 0]
IpListOld = []
cameras = []
converter = []
connect = 0
event = threading.Event()
camera_run_type = 0  # 0��Ӳ�������У�1��������
image_save = 0
img_view = 1
reconnectCmd = 0
head_tail_distance_mm = 14 #�����β��
C1 = 0
C2 = 0

### ����IP
ipStr = '192.168.1.66'
# ipStr = '127.0.0.1' #����
# ipStr = '192.168.137.66' #���ػ�
def num2register(num):  # һ������ת��Ϊmodbustcp�е������Ĵ���[��16λ����16λ]
    return [num & 65535, (num >> 16) & 65535]


def register2num(register0, register1):  # modbustcp�е������Ĵ���ת��Ϊһ������[��16λ����16λ]
    return register0 + (register1 << 16)


class MarkDetect:
    def __init__(self):
        global head_tail_distance_mm
        # ���Կ��Ʋ���
        self.plt_show = 0  # �Ƿ���м����̵�ͼƬ��ʾ
        self.profile_enable = 0  # �Ƿ���������ʱ

        ### ���ó�ʼ״̬�µĳ��ȷ�Χ����Ӧ���ؿ�������
        self.field_of_view_X_mm = 50  #���� ���
        self.resolution_X = 2048
        self.mark_num = 5
        # self.head_tail_pixel = 700
        self.head_tail_x_pixel = 0
        self.head_tail_y_pixel = 0
        self.mark_detect_num = 0
        self.detect_step = 0
        self.mark_type_ix = -1



        # ���α�+Բ�α�

        self.mark_type = 0  # 0�����α꣬1��Բ�α�
        self.mark_width = [1, 1.5] # ���ζԽ��� Բ��ֱ��
        self.mark_height = [1, 1.5]
        self.rectangularity = [1, 0.785]
        self.limit = [0.6, 1.2]  # [0.7, 1.2]��ɸѡ����ͳ���������С������Χ

        # ����㷨��ز���
        self.scaling = 3  # 3������С3���ּ�⣬ʱ�仨�Ѻ;��ȶ��ȽϺ���
        self.blur_kernel = 35  # 9������֮ǰ��˹�˲����ڴ�С��9�ȽϺ���
        self.adaptive_block = 67  # 67������Ӧ��ֵ���Ĵ��ڴ�С������Ŀǰ���ԵĴ�С���α�����Ǳ궼��Ϊ����
        self.C = 3  # 12������Ӧ��ֵ������ֵƫ��, ����ֱ����أ���������
        self.par = []
        self.mark_area = []

        # �������������Ŀǰ�ú�����ֻʹ��һ���������������������߳��зֱ��øú���ʵ�֣���������self.camera_ixֱ������Ϊ0��2������У�
        self.global_ymin = -1
        self.global_ymax = -1
        self.global_ROI_enable = 1
        self.camera_ix = 0 #�ģ�
        self.par_init()

    def par_init(self):
        self.mark_width = [i * self.resolution_X / self.field_of_view_X_mm for i in self.mark_width]
        self.mark_height = [i * self.resolution_X / self.field_of_view_X_mm for i in self.mark_height]
        self.mark_area = [i * j * k for i, j, k in zip(self.mark_width, self.mark_height, self.rectangularity)]
        self.par = []
        for n in [1, self.scaling]:  # ����Ԥ�ȼ���
            n2 = n * n
            area_limit = [[area * self.limit[0] / n2, area * self.limit[1] / n2] for area in self.mark_area]
            width_limit = [[w * self.limit[0] / n, w * self.limit[1] / n] for w in self.mark_width]
            height_limit = [[h * self.limit[0] / n, h * self.limit[1] / n] for h in self.mark_height]
            rectangularity_limit = [[rec * self.limit[0], rec * self.limit[1]] for rec in self.rectangularity]
            blur_kernel = int(round(self.blur_kernel / n, 0))
            blur_kernel = blur_kernel if blur_kernel % 2 else blur_kernel + 1
            adaptive_block = int(round(self.adaptive_block / n, 0))
            adaptive_block = adaptive_block if adaptive_block % 2 else adaptive_block + 1
            self.par.append(
                [n, n2, area_limit, width_limit, height_limit, rectangularity_limit, blur_kernel, adaptive_block])

    def mark_detect_single_channel(self, img, img_HSV, detect_step,n=1, xmin=0, ymin=0):
        if n == 1:
            n, n2, area_limit, width_limit, height_limit, rectangularity_limit, blur_kernel, adaptive_block = self.par[
                0]
        else:
            n, n2, area_limit, width_limit, height_limit, rectangularity_limit, blur_kernel, adaptive_block = self.par[
                1]

        # img = img[0:-1:n, 0:-1:n]
        # img_bilateralFilter = cv2.bilateralFilter(img, blur_kernel, 15, 5)
        # img_medianBlur = cv2.medianBlur(img, blur_kernel)
        img_Gaussian = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
        binary_img = cv2.adaptiveThreshold(img_Gaussian, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                           adaptive_block, self.C)
        # binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_ERODE, np.ones((3, 3), np.uint8), iterations=1)
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]  # Ѱ������
        ### ����ڰ�ͼ���û�����CV����
        # cv2.imshow("draw",binary_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # ���������������������ж�
        result = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # ����ּ�,����ɫ��
            area_ok = 0
            for [a1, a2] in area_limit:
                if a1 < area < a2:
                    area_ok = 1
                    break
            if not area_ok:
                continue

            rect = cv2.minAreaRect(cnt)  # ������С��Ӿ���
            w, h = rect[1]

            size_ok = 0
            for [w1, w2], [h1, h2] in zip(width_limit, height_limit):  #��Ӿ��ο���ж�
                if (w1 < w < w2 and h1 < h < h2) or (w1 < h < w2 and h1 < w < h2):
                    size_ok = 1
                    break
            if not size_ok: continue

            # if not 30 < abs(rect[2]) < 60:
            #     continue  # ���νǶ��ж�

            # ���ۺ��ж�һ�£�ͬʱȷ�����������͵�ɫ�꣬���������ζ�ɫ�������ж�
            mark_ok = 0
            self.mark_type_ix = -1
            for [w1, w2], [h1, h2], [a1, a2], [r1, r2], ix in zip(width_limit, height_limit, area_limit,
                                                                  rectangularity_limit, range(len(width_limit))):
                if (a1 < area < a2) and ((w1 < w < w2 and h1 < h < h2) or (w1 < h < w2 and h1 < w < h2)) and (
                        r1 < area / (w * h) < r2):
                    mark_ok = 1
                    self.mark_type_ix = ix
            if mark_ok == 0:
                continue
            if self.mark_type_ix != self.mark_type:
                continue

            # ����ɫ�����ĵ�
            x_center, y_center = rect[0]
            print(x_center, y_center)
            hValue = img_HSV[int(y_center),int(x_center)][0]
            hValue_Center = img_HSV[int(y_center), int(x_center)][0]
            hValues = []
            hValue_min = img_HSV[int(y_center),int(x_center)][0]
            hValue_max = img_HSV[int(y_center),int(x_center)][0]
            for y_mark in range(int(max((y_center - h/4),0)),int(min((y_center + h/4 + 1),img.shape[0]))):
                for x_mark in range(int(max((x_center - w/4),0)), int(min((x_center + w/4 + 1),img.shape[1]))):
                    if img_HSV[int(y_mark ),int(x_mark)][0] > hValue_max:
                        hValue_max = img_HSV[int(y_mark ),int(x_mark)][0]
                    elif img_HSV[int(y_mark ),int(x_mark)][0] < hValue_min:
                        hValue_min = img_HSV[int(y_mark), int(x_mark)][0]
                    hValues.append(img_HSV[int(y_mark ),int(x_mark)][0])
            hValues = sorted(hValues)
            hValue = hValues[int(len(hValues)/2)]



            vValue = img_HSV[int(y_center),int(x_center)][2]
            vValues = []
            for y_mark in range(int(max((y_center - h/4),0)),int(min((y_center + h/4 + 1),img.shape[0]))):
                for x_mark in range(int(max((x_center - w/4),0)), int(min((x_center + w/4 + 1),img.shape[1]))):
                    vValues.append(img_HSV[int(y_mark ),int(x_mark)][2])
            vValues = sorted(vValues)
            vValue = vValues[int(len(vValues)/2)]
            # vValue = min(vValues)
            mark_V = int(vValue)

            color = 3  # 0-��ɫ�� 1-�죬 2-�ƣ� 3-��

            # if mark_V < 95:
            #     # color = 3
            #     hValue = max(hValues)
            # else:
            #     hValues = sorted(hValues)
            #     hValue = hValues[int(len(hValues)/2)]


            mark_Value = int(hValue)
            if 5 < mark_Value < 30:
                color = 0 #��
            elif 75 <= mark_Value <= 120:
                color = 2 #��
            elif 140 < mark_Value < 169:
                color = 1 #��
            # elif 173 < mark_Value <= 180:
            #     color = 3 #��

            if (mark_V < 90) and ((int(hValue_min) < 6) or (int(hValue_max) > 173)): #����Ƭ������أ���Ȧ�����գ��ع�ʱ��
                 color = 3

            x_center, y_center = x_center + xmin, y_center + ymin

            # ���������С���ж���
            box = cv2.boxPoints(rect)  # ��Ӿ��εĶ�������
            box = np.int0(box)  # �������ͻ�

            cal_other_output = 0  # area ���ζȣ��ҶȶԱ�
            if cal_other_output:
                # ����ʵ�ʳߴ�ͱ�׼�ߴ�ı�ֵ
                area_rate = area * n2 / self.mark_area[self.mark_type_ix]
                w_rate = w * n / self.mark_width[self.mark_type_ix]
                h_rate = h * n / self.mark_height[self.mark_type_ix]

                # ������ζ�
                rectangularity = area / (w * h)

                # ����ɫ���뱳���ҶȲ��
                xmin1, xmax1 = box[:, 0].min(), box[:, 0].max()
                ymin1, ymax1 = box[:, 1].min(), box[:, 1].max()
                xmin2 = max(xmin1 - (xmax1 - xmin1), 0)
                xmax2 = min(xmax1 + (xmax1 - xmin1), img.shape[1])
                ymin2 = max(ymin1 - (ymax1 - ymin1), 0)
                ymax2 = min(ymax1 + (ymax1 - ymin1), img.shape[0])
                meangray1 = img[ymin1:ymax1, xmin1:xmax1].mean()
                meangray2 = img[ymin2:ymax2, xmin2:xmax2].mean()
                meangray_dis = abs(meangray1 - meangray2)
            else:
                meangray_dis, rectangularity, area_rate, w_rate, h_rate = 0, 0, 0, 0, 0

            result.append([x_center, y_center, cnt, rect, box, color, mark_Value, detect_step, meangray_dis, rectangularity, area_rate, w_rate, h_rate])

            # for i in range(0, len(result)-1):
            #     if (result[:, 7].min() <= result[i, 7] <= (result[:, 7].min() + 20)) and (result[i, 7] < 90):
                    # result[i, 5] = 3

            # for i, r in enumerate(result):  # i��� r����
            #     if (result[:, 7].min() <= result[i, 7] <= (result[:, 7].min() + 20)) and (result[i, 7] < 90):
            #         result[i, 5] = 3
                # x_distance = result[:, 0] - r[0]  # �ϴκ����x���
                # if np.abs(x_distance).min() > (
                #         self.mark_height[self.mark_type_ix] + self.mark_width[self.mark_type_ix]) * 0.5:
                #     result_pos = np.concatenate((result_pos, result_pos_R_channel[i:i + 1]), axis=0)


        if self.plt_show:
            binary_img_plt = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
            img_plt = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_plt2 = img_plt.copy()
            plt.figure(figsize=(23, 12))
            for x_center, y_center, cnt, rect, *o in result:
                cv2.drawContours(img_plt, [cnt], -1, (0, 0, 255), 2)
                cv2.drawContours(binary_img_plt, [cnt], -1, (0, 0, 255), 2)
                box = cv2.boxPoints(rect)  # ��Ӿ��εĶ�������
                box = np.int0(box)  # �������ͻ�
                cv2.drawContours(img_plt2, [box], -1, (0, 0, 255), 2)
            plt.subplot(2, 2, 1), plt.imshow(img_plt, 'gray'), plt.title("img_color")
            plt.subplot(2, 2, 2), plt.imshow(binary_img_plt, 'gray'), plt.title('binary_img')
            plt.subplot(2, 2, 3), plt.imshow(img_plt2, 'gray'), plt.title('binary_img')
            plt.show()

        return result, img, binary_img


    def mark_detect(self, img):
        global  head_tail_distance_mm, C1, C2
        self.head_tail_pixel = float(head_tail_distance_mm/self.field_of_view_X_mm) * 2048

        scaling = self.scaling

        # ����ԭʼ��С
        origin_img_width, origin_img_height = img.shape[1], img.shape[0]

        _ymin, _ymax, _enable = self.global_ymin, self.global_ymax, self.global_ROI_enable
        if _ymin >= 0 and _ymax >= 0 and _enable:
            img = img[self.global_ymin:self.global_ymax, :]
            # ����ͼresizeһ�£��ּ��
            img_in = cv2.cvtColor(img[0:-1:scaling, 0:-1:scaling, :], cv2.COLOR_RGB2GRAY)  # ��ʼ��ֹ������resize���´���
            img_HSV = cv2.cvtColor(img[0:-1:scaling, 0:-1:scaling, :], cv2.COLOR_RGB2HSV)  # ��ʼ��ֹ������resize���´���
            # img_in = cv2.cvtColor(img[0:-1:scaling, 0:-1:scaling, :], cv2.COLOR_RGB2GRAY)  # ��ʼ��ֹ������resize���´���
            # img_HSV = cv2.cvtColor(img[0:-1:scaling, 0:-1:scaling, :], cv2.COLOR_RGB2HSV)  # ��ʼ��ֹ������resize���´���
        else:
            # ����ͼresizeһ�£��ּ��
            img_in = cv2.cvtColor(img[0:-1:scaling, 0:-1:scaling, :], cv2.COLOR_RGB2GRAY)  # ��ʼ��ֹ������resize���´���
            img_HSV = cv2.cvtColor(img[0:-1:scaling, 0:-1:scaling, :], cv2.COLOR_RGB2HSV)  # ��ʼ��ֹ������resize���´���

        self.detect_step = 0
        result_gray, img_, binary_img = self.mark_detect_single_channel(img_in,img_HSV,self.detect_step, n=scaling)


        # �����
        if scaling != 1:
            accurate_result = []
            for x_center, y_center, cnt, rect, box, *o in result_gray:
                x, y = box[:, 0].min(), box[:, 1].min()
                w, h = box[:, 0].max() - x, box[:, 1].max() - y
                x, y, w, h = x * scaling, y * scaling, w * scaling, h * scaling
                edge = 20
                xmin = max(x - edge, 0)
                xmax = min(x + w + edge, img.shape[1])
                ymin = max(y - edge, 0)
                ymax = min(y + h + edge, img.shape[0])
                im_in = img[ymin:ymax, xmin:xmax]
                im_in = cv2.cvtColor(im_in, cv2.COLOR_RGB2GRAY)
                img_HSV = img[ymin:ymax, xmin:xmax]
                img_HSV = cv2.cvtColor(img_HSV, cv2.COLOR_RGB2HSV)
                self.detect_step = 1
                result_, img_, binary_img_ = self.mark_detect_single_channel(im_in,img_HSV,self.detect_step, n=1, xmin=xmin, ymin=ymin)
                if len(result_) > 0:
                    accurate_result.append(result_[0])
            result_gray = accurate_result
        result_pos_gray = np.array([[x_center, y_center, rect[2], color, mark_Value, detect_step, meangray_dis, rectangularity] for
                                    x_center, y_center, cnt, rect, box, color, mark_Value, detect_step, meangray_dis, rectangularity, *o in
                                    result_gray])  #����
        result_pos = result_pos_gray.copy()  #copy

        # ���ֻ�����˲���ɫ�꣨����һ����С��ȫ���������Ե���ȡ0ͨ���������ɫ��
        if 1 <= len(result_gray) < self.mark_num:
            # ������Ҷ�ͼ������ɫ�������£���������ɫ��λ�ã���ȡɫ�������ܱߵ�ͼ��ȡ�ý�ȡͼ���ĳ������ͨ�����ټ��һ��
            if C2 != 0:
                self.C = C2
            else:
                self.C = 4
            ymax, ymin = result_pos_gray[:, 1].max(), result_pos_gray[:, 1].min()
            ymin = int(max(ymin - self.mark_height[self.mark_type_ix] - 20, 0))
            ymax = int(min(ymax + self.mark_height[self.mark_type_ix] + 20, img.shape[0]))
            img_R_channel = img[ymin:ymax, :, 0]
            img_HSV = img[ymin:ymax, :]
            img_HSV = cv2.cvtColor(img_HSV, cv2.COLOR_RGB2HSV)
            self.detect_step = 2
            result_R_channel, img_, binary_img = self.mark_detect_single_channel(img_R_channel,img_HSV,self.detect_step, n=1, xmin=0, ymin=ymin)
            result_pos_R_channel = np.array([[x_center, y_center, rect[2], color, mark_Value, detect_step, meangray_dis, rectangularity] for
                                             x_center, y_center, cnt, rect, box, color, mark_Value, detect_step, meangray_dis, rectangularity, *o in
                                             result_R_channel])
            if C1 != 0:
                self.C = C1
            else:
                self.C = 3

                # �ڶ��μ�������һ��ɫ�������£����Ժϲ�
            if len(result_R_channel) >= 1:
                # result_pos_R_channel = result_pos_R_channel[np.argsort(result_pos_R_channel[:, 0])]  # ����
                for i, r in enumerate(result_pos_R_channel):  #i��� r����
                    x_distance = result_pos[:, 0] - r[0]  #�ϴκ����x���
                    if np.abs(x_distance).min() > (self.mark_height[self.mark_type_ix] + self.mark_width[self.mark_type_ix]) * 0.5:
                        result_pos = np.concatenate((result_pos, result_pos_R_channel[i:i + 1]), axis=0)

        if result_pos.shape[0] >= 3:  # ���ټ��3��ʱ������һ�μ��ɫ����ݱ���ɫ���ȡROI
            # ��ɫ����ͼ�еĴ���λ�ü�¼�������´μ��ʱ��Ĭ�ϼ��ô���λ���ڵ�ͼƬ
            _ymin, _ymax, _enable = self.global_ymin, self.global_ymax, self.global_ROI_enable
            if self.global_ymin >= 0 and self.global_ymax >= 0 and _enable:
                result_pos[:, 1] = result_pos[:, 1] + self.global_ymin
            if abs(result_pos[:, 1].max() - result_pos[:, 1].min()) < self.mark_height[self.mark_type_ix] * 2:
                self.global_ymin = int(max(result_pos[:, 1].min() - self.mark_height[self.mark_type_ix] * 2, 0))
                self.global_ymax = int(min(result_pos[:, 1].max() + self.mark_height[self.mark_type_ix] * 2, origin_img_height))


                # if abs(min(result_pos[:, 1].max()) - max(result_pos[:, 1].min())) < self.mark_height[self.mark_type_ix] * 4:
                #     self.global_ymin = int(max(result_pos[:, 1].min() - self.mark_height[self.mark_type_ix] * 4, 0))
                #     self.global_ymax = int(min(result_pos[:, 1].max() + self.mark_height[self.mark_type_ix] * 4, origin_img_height))

            # if _enable:
            #     if abs(result_pos[:, 1].max() - result_pos[:, 1].min()) < self.mark_height[self.mark_type_ix] * 4:
            #         result_pos[:, 1] = result_pos[:, 1] + _ymin
            #         self.global_ymin = int(max(result_pos[:, 1].min() - self.mark_height[self.mark_type_ix] * 4, 0))
            #         self.global_ymax = int(min(result_pos[:, 1].max() + self.mark_height[self.mark_type_ix] * 4, origin_img_height))
        else:
            self.global_ymin = -1
            self.global_ymax = -1



        detection_success = 0
        if result_pos.shape[0] >= 2:  # ���ټ��2��
            detection_success = 1
            self.mark_detect_num = result_pos.shape[0]
            result_pos = result_pos[np.argsort(result_pos[:, 0])]  # ����
            # ������β��У׼���� result_pos[:, :2] = result_pos[:, :2] * self.head_tail_distance_mm / (result_pos[-1,
            # 0] - result_pos[0, 0])  #��������

            #������β����
            breakflag = 0
            head_tail = 0
            # print(self.head_tail_pixel)
            for i in range(0, result_pos.shape[0]-1):
                for j in range(i + 1, result_pos.shape[0]):
                    if result_pos[i,3] == result_pos[j,3]:
                        self.head_tail_x_pixel = abs(result_pos[j, 0] - result_pos[i, 0])
                        self.head_tail_y_pixel = abs(result_pos[j, 1] - result_pos[i, 1])
                        head_tail = math.sqrt(self.head_tail_x_pixel ** 2 + self.head_tail_y_pixel ** 2)
                        if 0.8 * self.head_tail_pixel < head_tail < 1.2 * self.head_tail_pixel:
                            # print(head_tail, self.head_tail_pixel)
                            if abs(j - i) == 4:#��β��࣬��ֹʶ�����ɫ
                                self.head_tail_pixel = head_tail
                                breakflag = 1
                                break
                if breakflag == 1:
                    break
            # print(detection_success)
        else:
            detection_success = 0


        return detection_success, result_pos


class ImageAcquistionAndDetect():
    def __init__(self):
        """
        ���������ֹ���
        1.��ʼ�����
        2.��ʼ��modbustcp
        """
        global ImageAcquired, IpListOld, cameras, converter, connect

        randomByteArray = bytearray(os.urandom(100000))
        flatNumpyArray = np.array(randomByteArray)

        grayImage = flatNumpyArray.reshape(200,500)


        self.img_default = grayImage
        ### ʹ�ñ���ͼƬ���ǵ���ȡ��ͼƬ�����ڵ���
        # self.img_default = cv2.imdecode(np.fromfile("image_14us.jpg", dtype=np.uint8), 1)
        ImageAcquired = [self.img_default, self.img_default]
        # head_tail_distance_mm = 14  # ���

        cameras, converter, IpListOld = self.camera_init()
        connect = 0
        self.modbus_server = self.modbus_init()  #��
        ### Ĭ����Ұ�µ�������Ұ���
        self.filed_of_view_X_mm = 50 #���

    def image_acquistion_and_detect(self, camera_ix):
        global ImageAcquired, CompletedAcquisitionCnt, FailedAcquisitionCnt, ResultMessage, CompletedDetectionCnt,\
            FailedDetectionCnt, IpListOld, cameras, converter, connect, event, camera_run_type,image_save, img_view,\
            reconnectCmd, head_tail_distance_mm

        ix = camera_ix
        camera = cameras[ix]
        mark_detect = MarkDetect()


        exposure_time_old_0 = 0  #���
        exposure_time_old_1 = 0  #���

        camera_run_type_old_0 = 0
        camera_run_type_old_1 = 0

        t_heartbeat_ctrl = time.time()
        num_heartbeat_ctrl = 0
        t00 = time.time()
        xxx = 0

        ArrayHead_tail_pixel = []
        Head_tail_pixel_ErrorSum = 0
        Head_tail_pixel_ErrorMean = 0

        while True:
            time_retrieve = 0
            time_converter = 0
            time_detection = 0
            time_save = 0
            values = 0

            t1 = time.time()
            t0 = time.time()

            # ��modbusʵʱ��ȡ����
            [mark_shape, mark_size, size_limit_min, size_limit_max, exposure_time, head_tail_distance_mm, image_save,
             img_view, camera_run_type, reconnectCmd] = self.get_modbus_data(ix)
            # print(self.get_modbus_data(ix))
            # ���״̬λ��0
            #self.modbus_server.set_values(block_name='0', address=100 + ix * 100 + 1, values=[0])  #��������
            self.modbus_server.set_values(block_name='1', address=100 + ix * 2 + 2, values=num2register(0))
            # ÿ1s��������1
            if time.time() - t_heartbeat_ctrl > 1:
                t_heartbeat_ctrl = time.time()
                num_heartbeat_ctrl = int(num_heartbeat_ctrl + 1) % 255
                #self.modbus_server.set_values(block_name='0', address=100 + ix * 100, values=[num_heartbeat_ctrl])  #��������
                self.modbus_server.set_values(block_name='1', address=100 + ix * 1, values=[num_heartbeat_ctrl])

            detection_success = 0
            try:
                # �ع�ʱ��ı��ʱ���д�����
                if (exposure_time_old_0 != exposure_time) and (camera_ix == 0):
                    camera.ExposureTimeAbs.SetValue(exposure_time)
                    exposure_time_old_0 = exposure_time
                elif (exposure_time_old_1 != exposure_time) and (camera_ix == 1):
                    camera.ExposureTimeAbs.SetValue(exposure_time)
                    exposure_time_old_1 = exposure_time

                if (camera_run_type_old_0 != camera_run_type) and (camera_ix == 0):
                    camera_run_type_old_0 = camera_run_type
                    if camera_run_type == 0:
                        # �趨Acquisition Mode
                        camera.AcquisitionMode.SetValue("Continuous")
                        # �趨TriggerSelector
                        camera.TriggerSelector.SetValue("FrameStart")
                        # �趨TriggerMode
                        camera.TriggerMode.SetValue("On")
                        # �趨TriggerSource
                        camera.TriggerSource.SetValue("Line1")
                        camera.TriggerActivation.SetValue("RisingEdge")
                        camera.TriggerDelayAbs.SetValue(0)
                    else:
                        # �趨Acquisition Mode
                        camera.AcquisitionMode.SetValue("Continuous")
                        # �趨TriggerSelector
                        camera.TriggerSelector.SetValue("FrameStart")
                        # �趨TriggerMode
                        camera.TriggerMode.SetValue("Off")
                elif (camera_run_type_old_1 != camera_run_type) and (camera_ix == 1):
                    camera_run_type_old_1 = camera_run_type
                    if camera_run_type == 0:
                        # �趨Acquisition Mode
                        camera.AcquisitionMode.SetValue("Continuous")
                        # �趨TriggerSelector
                        camera.TriggerSelector.SetValue("FrameStart")
                        # �趨TriggerMode
                        camera.TriggerMode.SetValue("On")
                        # �趨TriggerSource
                        camera.TriggerSource.SetValue("Line1")
                        camera.TriggerActivation.SetValue("RisingEdge")
                        camera.TriggerDelayAbs.SetValue(0)
                    else:
                        # �趨Acquisition Mode
                        camera.AcquisitionMode.SetValue("Continuous")
                        # �趨TriggerSelector
                        camera.TriggerSelector.SetValue("FrameStart")
                        # �趨TriggerMode
                        camera.TriggerMode.SetValue("Off")

                # ��ȡͼƬ
                grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                time_retrieve = time.time() - t1  #�����ͼʱ��
                t1 = time.time()

                # �ж�ͼƬ��ȡ�Ƿ�ɹ�
                if grabResult.GrabSucceeded():
                    # ���״̬λ��2
                    #self.modbus_server.set_values(block_name='0', address=100 + ix * 100 + 1, values=[2])  #��������
                    self.modbus_server.set_values(block_name='1', address=100 + ix * 2 + 2, values=num2register(2))
                    # ���ͼƬ�����־λ��1
                    #self.modbus_server.set_values(block_name='0', address=100 + ix * 100 + 3, values=[1])  #��������
                    self.modbus_server.set_values(block_name='1', address=100 + ix * 1 + 8, values=[1])

                    CompletedAcquisitionCnt[ix] = CompletedAcquisitionCnt[ix] + 1
                    # print(CompletedAcquisitionCnt[ix])
                    self.modbus_server.set_values(block_name='1', address=100 + ix * 2 + 70,
                                                  values=num2register(CompletedAcquisitionCnt[ix]))
                    image = converter.Convert(grabResult)  #format
                    ImageAcquired[ix] = image.GetArray()
                    ### ʹ�ñ���ͼƬ���ǵ���ȡ��ͼƬ�����ڵ���
                    # ImageAcquired[ix] = self.img_default

                    time_converter = time.time() - t1
                    t1 = time.time()


                    mark_detect.mark_num = 5
                    mark_detect.mark_type = mark_shape  # 0�����α꣬1��Բ�α�
                    mark_detect.mark_width = [0, 0]
                    mark_detect.mark_height = [0, 0]
                    mark_detect.mark_width[mark_shape] = mark_size
                    mark_detect.mark_height[mark_shape] = mark_size
                    mark_detect.limit = [size_limit_min, size_limit_max]
                    mark_detect.par_init()

                    # detection_success�����ټ������ɫ�꼴��Ϊ����ɹ���������ʧ��
                    # result_pos��m��n�е�numpy���飬mΪ�����ɫ����Ŀ����ÿһ�о���һ��ɫ��Ľ������һ����X����ֵ���ڶ�����Y����ֵ��
                    # ��������ɫ��������С��Ӿ��εĽǶȣ�֮�����Ϊ�������������Ҫ���ڳ������޸�
                    # result_pos�����е����ݶ��Ǹ�����
                    detection_success, result_pos = mark_detect.mark_detect(ImageAcquired[ix])

                    if detection_success:
                        self.filed_of_view_X_mm = (float(head_tail_distance_mm / mark_detect.head_tail_pixel)) * 2048
                    # ɫ����
                    mark_detect.field_of_view_X_mm = self.filed_of_view_X_mm

                    # ����һ�������λ�þ������
                    ### 24.5.21 ��һ�ο���ע�Ϳ�ע��
                    if result_pos.shape[0] == 5:
                        Pix12 = math.sqrt((result_pos[1, 1] - result_pos[0, 1]) ** 2 + (result_pos[1, 0] - result_pos[0, 0]) ** 2)
                        Pix12 = float(Pix12 * self.filed_of_view_X_mm / 2048)
                        ArrayHead_tail_pixel.append(Pix12)
                        sorted(ArrayHead_tail_pixel)
                        if 25 <= len(ArrayHead_tail_pixel):
                            Head_tail_pixel_ErrorSum = 0
                            Head_tail_pixel_Mean = np.mean(ArrayHead_tail_pixel[
                                                           len(ArrayHead_tail_pixel) - 21:len(
                                                               ArrayHead_tail_pixel) - 1])
                            for k in range(len(ArrayHead_tail_pixel) - 21, len(ArrayHead_tail_pixel) - 1):
                                Head_tail_pixel_ErrorSum = Head_tail_pixel_ErrorSum + abs(
                                    ArrayHead_tail_pixel[k] - Head_tail_pixel_Mean)
                            Head_tail_pixel_ErrorMean_mm = Head_tail_pixel_ErrorSum / 20
                            # print(self.ArrayHead_tail_pixel)
                            print(Head_tail_pixel_Mean, Head_tail_pixel_ErrorMean_mm)


                    self.modbus_server.set_values(block_name='1', address=100 + ix * 2 + 66,
                                                  values=num2register(int(self.filed_of_view_X_mm * 10000)))

                    if detection_success:
                        CompletedDetectionCnt[ix] = CompletedDetectionCnt[ix] + 1
                        self.modbus_server.set_values(block_name='1', address=100 + ix * 2 + 10, values=num2register(CompletedDetectionCnt[ix]))
                        values = self.set_modbus_data(result_pos, ix)
                    else:
                        FailedDetectionCnt[ix] = FailedDetectionCnt[ix] + 1
                    np.set_printoptions(suppress=True)
                    # ��ӡ����
                    print('����������' + str(ix),
                          [mark_shape, mark_size, size_limit_min, size_limit_max, self.filed_of_view_X_mm,
                           exposure_time])
                    print('����ɹ��������' + str(ix), CompletedDetectionCnt[ix])
                    print('����' + str(ix), num_heartbeat_ctrl)
                    print('��������' + str(ix), result_pos)
                    Period = 0
                    if CompletedAcquisitionCnt[ix] > 1:
                        Period = (time.time() - t00) / CompletedAcquisitionCnt[ix]
                    print('��ʱ' + str(ix), round(time.time() - t0, 3))
                    print('�ۼƾ�ʱ' + str(ix), round((time.time() - t00) / CompletedAcquisitionCnt[ix], 3))


                    time_detection = time.time() - t1
                    t1 = time.time()

                    # ������Ƭ�洢
                    if CompletedAcquisitionCnt[ix] < -10:
                        os.makedirs("Data20230519\\Basler", exist_ok=True)
                        cv2.imwrite("Data20230519\\Basler\\" + str(time.time()) + ".jpg", ImageAcquired[1])
                    if camera_ix == 1 and detection_success == 1 and xxx < -10:
                        xxx = xxx + 1
                        os.makedirs("Data20230525\\Basler", exist_ok=True)
                        cv2.imwrite("Data20230525\\Basler\\" + str(time.time()) + ".jpg", ImageAcquired[1])

                    time_save = time.time() - t1

                    # ���״̬λ��0
                    #self.modbus_server.set_values(block_name='0', address=100 + ix * 100 + 1, values=[0])  #��������
                    self.modbus_server.set_values(block_name='1', address=100 + ix * 2 + 2, values=num2register(0))
                    # trigger��0
                    # self.modbus_server.set_values(block_name='1', address=100 + ix * 1 + 6, values=[0])  #��������
                    # ���ͼƬ�����־λ��0
                    #self.modbus_server.set_values(block_name='0', address=100 + ix * 100 + 3, values=[0])  #��������
                    self.modbus_server.set_values(block_name='1', address=100 + ix * 1 + 8, values=[0])

                else:
                    # ���״̬λ��65534
                    #self.modbus_server.set_values(block_name='0', address=100 + ix * 100 + 1, values=[65534])  #��������
                    self.modbus_server.set_values(block_name='1', address=100 + ix * 2 + 2, values=[65534])
                    FailedAcquisitionCnt[ix] = FailedAcquisitionCnt[ix] + 1
                    ImageAcquired[ix] = self.img_default
                    detection_success = 0
                # time.sleep(min(max(0.1 - Period,0),0.05))
            except genicam.TimeoutException:
                # connect = 1
                # event.wait()
                print("���" + str(ix) + "��ȡͼƬ��ʱ")



            print_str = "���" + str(camera_ix) + ''
            if detection_success:
                print_str = print_str + '���ɹ�' + '��'
            else:
                print_str + '���ʧ��' + ','
            print_str = print_str + '��ʱ' + str(round(time.time() - t0, 3)) + ','
            print_str = print_str + '��ȡ' + str(round(time_retrieve, 3)) + ','
            print_str = print_str + 'ת��' + str(round(time_converter, 3)) + ','
            print_str = print_str + '���' + str(round(time_detection, 3)) + ','
            print_str = print_str + '����' + str(round(time_save, 3)) + ','
            print_str = print_str + '�ۼƾ�ʱ' + str(round((time.time() - t00) / max(CompletedAcquisitionCnt[ix],1), 3))
            print_str = print_str + '���ʱ��' + str(round(time_detection, 3))

            if detection_success:
                print_str = print_str + '\n' + str(values)
            # print(print_str)
            ResultMessage[ix] = print_str  # ��gui����ʾ


    def camera_init(self):
        global cameras, IpListOld, converter
        cameras = []
        IpList = []
        tl_factory = pylon.TlFactory.GetInstance()  #�������
        for ix, dev_info in enumerate(tl_factory.EnumerateDevices()):
            if not dev_info.GetDeviceClass() == 'BaslerGigE': continue
            cam_info = dev_info
            print(
                "using %s @ %s (%s)" % (
                    cam_info.GetModelName(),
                    cam_info.GetIpAddress(),
                    cam_info.GetMacAddress()
                )
            )
            while True:
                try:
                    camera = pylon.InstantCamera(tl_factory.CreateDevice(dev_info))
                    camera.Open()
                    camera = self.camera_setting(camera)
                    cameras.append(camera)
                except:
                    continue
                else:
                    break

            # time.sleep(1)
            # if cam_info.GetIpAddress() not in IpListOld:
            #     camera.Open()
            #     camera = self.camera_setting(camera)
            #     time.sleep(1)
            # cameras.append(camera)

        # converting to opencv bgr format
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        IpListOld = IpList.copy()
        return cameras, converter, IpListOld

    def camera_setting(self, camera):
        global camera_run_type
        # �ع��������
        camera.ExposureMode.SetValue("Timed")
        camera.ExposureTimeMode.SetValue("Standard")
        ### �����ع�ʱ��
        camera.ExposureTimeAbs.SetValue(30)

        # AOI����
        if camera.Width.GetValue() > 1500:  # �����
            camera.Width.SetValue(2048)
            camera.Height.SetValue(800)
            camera.OffsetX.SetValue(0)
            camera.OffsetY.SetValue(300)

        # �趨MaxNumBuffer
        camera.MaxNumBuffer = 2

        if camera_run_type == 0:
            # �趨Acquisition Mode
            camera.AcquisitionMode.SetValue("Continuous")
            # �趨TriggerSelector
            camera.TriggerSelector.SetValue("FrameStart")
            # �趨TriggerMode
            camera.TriggerMode.SetValue("On")
            # �趨TriggerSource
            camera.TriggerSource.SetValue("Line1")
            camera.TriggerActivation.SetValue("RisingEdge")
            camera.TriggerDelayAbs.SetValue(0)
        else:
            # �趨Acquisition Mode
            camera.AcquisitionMode.SetValue("Continuous")
            # �趨TriggerSelector
            camera.TriggerSelector.SetValue("FrameStart")
            # �趨TriggerMode
            camera.TriggerMode.SetValue("Off")


        camera.StartGrabbing()

        return camera

    def modbus_init(self):
        global ipStr
        """
        ��ʹ�õĺ���:
        ������վ: server.add_slave(slave_id)
            slave_id(int):��վid
        Ϊ��վ��Ӵ洢��: slave.add_block(block_name, block_type, starting_address, size)
            block_name(str):block��
            block_type(int):block����,COILS = 1,DISCRETE_INPUTS = 2,HOLDING_REGISTERS = 3,ANALOG_INPUTS = 4
            starting_address(int):��ʼ��ַ
            size(int):block��С
        ����blockֵ:slave.set_values(block_name, address, values)
            block_name(str):block��
            address(int):��ʼ�޸ĵĵ�ַ
            values(a list or a tuple or a number):Ҫ�޸ĵ�һ��(a number)����(a list or a tuple)ֵ
        ��ȡblockֵ:slave.get_values(block_name, address, size)
            block_name(str):block��
            address(int):��ʼ��ȡ�ĵ�ַ
            size(int):Ҫ��ȡ��ֵ������
        """
        # ������վ�ܷ�����
        # server = modbus_tcp.TcpServer(address='127.0.0.1')  # address��������,portĬ��Ϊ502
        # server = modbus_tcp.TcpServer(address='192.168.137.66')  # address��������,portĬ��Ϊ502 ���ػ�
        server = modbus_tcp.TcpServer(address=ipStr)  # address��������,portĬ��Ϊ502
        server.start()
        # ������վ
        slave_1 = server.add_slave(1)  # slave_id = 1
        # Ϊ��վ��Ӵ洢��
        slave_1.add_block(block_name='0', block_type=cst.HOLDING_REGISTERS, starting_address=0, size=17)
        slave_1.add_block(block_name='1', block_type=cst.ANALOG_INPUTS, starting_address=100, size=74)
        print("Modbus tcp running...")

        return slave_1

    def get_modbus_data(self, camera_ix):
        global head_tail_distance_mm, ClientToServer, C1, C2
        modbusdata = self.modbus_server.get_values(block_name='0', address=0, size=17)  #��������
        ClientToServer = modbusdata
        mark_shape = modbusdata[0]
        mark_size = modbusdata[1] * 0.001  # um->mm
        #size_limit_min = modbusdata[3] * 0.01  #��������
        #size_limit_max = modbusdata[4] * 0.01  #��������
        size_limit_min = 0.6
        size_limit_max = 1.2
        # �������������ѡ���Ӧ������
        if camera_ix == 0:
            #filed_of_view_X_mm = modbusdata[5]  #��������

            ### ���ó�ʼ״̬�µĳ��ȷ�Χ����Ӧ���ؿ�������
            filed_of_view_X_mm = 50
            #exposure_time = modbusdata[7]  #��������

            ### �����ع�ʱ��
            exposure_time = max(modbusdata[2],30) #standard 26��������
        elif camera_ix == 1:
            #filed_of_view_X_mm = modbusdata[6]  #��������

            ### ���ó�ʼ״̬�µĳ��ȷ�Χ����Ӧ���ؿ�������
            filed_of_view_X_mm = 50
            #exposure_time = modbusdata[8]  #��������

            ### �����ع�ʱ��
            exposure_time = max(modbusdata[3],30)
        head_tail_distance_mm = modbusdata[4]
        # image_save = modbusdata[5]
        image_save = 0
        # img_view = modbusdata[6]
        img_view = 1
        camera_run_type = modbusdata[7]
        reconnectCmd = modbusdata[8]
        C1 = modbusdata[15]
        C2 = modbusdata[16]

        # ��������һЩ����
        if mark_shape not in [0, 1]:
            mark_shape = 0
        if mark_size <= 0:
            mark_size = 0.1
        if size_limit_min < 0:
            size_limit_min = 0
        if size_limit_max < 0:
            size_limit_max = 0
        if size_limit_max < size_limit_min:
            size_limit_max, size_limit_min = size_limit_min, size_limit_max  # ��������
        # if filed_of_view_X_mm <= 0:
        #     filed_of_view_X_mm = 0.1
        exposure_time = max(10, exposure_time)
        exposure_time = min(exposure_time, 100) #ultrashort 26����
        head_tail_distance_mm = max(10, head_tail_distance_mm)
        head_tail_distance_mm = min(40, head_tail_distance_mm)
        if image_save not in [0, 1]:
            image_save = 0
        if img_view not in [0, 1]:
            img_view = 0
        if camera_run_type not in [0, 1]:
            camera_run_type = 0
        if reconnectCmd not in [0, 1]:
            reconnectCmd = 0

        return [mark_shape, mark_size, size_limit_min, size_limit_max, exposure_time, head_tail_distance_mm,
                image_save, img_view, camera_run_type, reconnectCmd]

    def set_modbus_data(self, result_pos, ix):
        values = []
        mark_num = min(result_pos.shape[0], 5)
        self.modbus_server.set_values(block_name='1', address=100 + ix + 14, values=[mark_num])
        pixel_pos_x = [0, 0, 0, 0, 0]
        pixel_pos_y = [0, 0, 0, 0, 0]
        color_mark = [0, 0, 0, 0, 0]
        for i, r in enumerate(result_pos):
            if i > 4:
                break
            pixel_pos_x[i] = int(r[0] * 100)
            pixel_pos_y[i] = int(r[1] * 100)
            color_mark[i] = int(r[3])


        pixel_pos_x_register, pixel_pos_y_register, color_register = [], [], []
        for x, y, z in zip(pixel_pos_x, pixel_pos_y, color_mark):
            pixel_pos_x_register = pixel_pos_x_register + num2register(x)
            pixel_pos_y_register = pixel_pos_y_register + num2register(y)
            color_register = color_register + [z]
        # ��������겹�봦��ȫ������5ɫ�꣬��λӦ����Ҫ��
        values = values + pixel_pos_x_register + pixel_pos_y_register + color_register
        # self.modbus_server.set_values(block_name='0', address=100 + ix * 100 + 4, values=values)  #��������
        self.modbus_server.set_values(block_name='1', address=100 + ix * 25 + 16, values = values)
        # print(self.modbus_server.get_values(block_name='1', address=100, size=74))
        # self.modbus_server.set_values(block_name='1', address=100 + ix * 2 + 11, values = [mark_num])
        # self.modbus_server.set_values(block_name='1', address=100 + ix * 1 + 15, values = [mark_num])
        # self.modbus_server.set_values(block_name='1', address=100 + ix * 25 + 16, values = pixel_pos_x_register)
        return values

class camera_connect(threading.Thread):
    # �������̵߳�ԭ���ǣ������ɫ�����߳��б���ͼƬ����ù��̻���ռ��ʵʱ�����ʱ�䣬�ʴ˴��������߳�
    def __init__(self):
        threading.Thread.__init__(self)


    def run(self):
        global IpListOld, cameras, converter, connect, event, camera_run_type
        cameras = []
        IpList = []
        tl_factory = pylon.TlFactory.GetInstance()  # �������
        for ix, dev_info in enumerate(tl_factory.EnumerateDevices()):
            if not dev_info.GetDeviceClass() == 'BaslerGigE':
                continue
            cam_info = dev_info
            print(
                "using %s @ %s (%s)" % (
                    cam_info.GetModelName(),
                    cam_info.GetIpAddress(),
                    cam_info.GetMacAddress()
                )
            )
            IpList.append(cam_info.GetIpAddress())
            camera = pylon.InstantCamera(tl_factory.CreateDevice(dev_info))
            if cam_info.GetIpAddress() not in IpListOld:
                camera.Open()
                # 0����������  https://docs.baslerweb.com/free-run-image-acquisition
                # 1��Ӳ�������� https://docs.baslerweb.com/triggered-image-acquisition
                camera.ExposureMode.SetValue("Timed")
                camera.ExposureTimeMode.SetValue("Standard")
                camera.ExposureTimeAbs.SetValue(30)  # ��� ####

                # AOI����
                if camera.Width.GetValue() > 1500:  # �����
                    camera.Width.SetValue(2048)
                    camera.Height.SetValue(800)
                    camera.OffsetX.SetValue(0)
                    camera.OffsetY.SetValue(300)

                # �趨MaxNumBuffer
                camera.MaxNumBuffer = 2

                if camera_run_type == 0:
                    # �趨Acquisition Mode
                    camera.AcquisitionMode.SetValue("Continuous")
                    # �趨TriggerSelector
                    camera.TriggerSelector.SetValue("FrameStart")
                    # �趨TriggerMode
                    camera.TriggerMode.SetValue("On")
                    # �趨TriggerSource
                    camera.TriggerSource.SetValue("Line1")
                    camera.TriggerActivation.SetValue("RisingEdge")
                    camera.TriggerDelayAbs.SetValue(0)
                else:
                    # �趨Acquisition Mode
                    camera.AcquisitionMode.SetValue("Continuous")
                    # �趨TriggerSelector
                    camera.TriggerSelector.SetValue("FrameStart")
                    # �趨TriggerMode
                    camera.TriggerMode.SetValue("Off")


                camera.StartGrabbing()
            cameras.append(camera)

        # converting to opencv bgr format
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        IpListOld = IpList.copy()
        connect = 0
        # event.set()


class ImagePresentation(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global ImageAcquired, CompletedAcquisitionCnt, FailedAcquisitionCnt, ResultMessage, ClientToServer, ipStr

        def test_func():
            global CompletedAcquisitionCnt, FailedAcquisitionCnt
            CompletedAcquisitionCnt, FailedAcquisitionCnt = [0, 0], [0, 0]

        def tkImage(img):
            cvimage = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            w, h = cvimage.shape[1], cvimage.shape[0]
            w_new = max(w, h) / 500
            image_width, image_height = int(w / w_new), int(h / w_new)
            pilImage = Image.fromarray(cvimage)
            pilImage = pilImage.resize((image_width, image_height), Image.ANTIALIAS)
            tkImage = ImageTk.PhotoImage(image=pilImage)
            return tkImage

        # ����ť�������ʱ��ִ��click_button()����
        def click_button():
            # ʹ����Ϣ�Ի���ؼ���showinfo()��ʾ��ܰ��ʾ
            try:
                cv2.imwrite("image0" + str(CompletedAcquisitionCnt[0]) + ".jpg", ImageAcquired[0])
                cv2.imwrite("image1" + str(CompletedAcquisitionCnt[1]) + ".jpg", ImageAcquired[1])
            except:
                print('ͼƬ����ʧ�ܣ��������Ա���')

        # def on_closing():
        #     pass

        # ���滭������ͼ��
        top = tk.Tk()
        top.title('��Ƶ����')
        top.geometry('1050x300+800+175')
        top.overrideredirect(True)
        # top.protocol("WM_DELETE_WINDOW", on_closing)
        top.lift()
        image_width = 2048
        image_height = 800
        canvas = Canvas(top, bg='white', width=image_width, height=image_height)  # ���ƻ���
        Label(top, text='���ͼƬ', font=("����", 13), width=15, height=1).place(x=10, y=10, anchor='nw')
        canvas.place(x=0, y=0)
        x_offset = 500

        # �����ťʱִ�еĺ���
        button = tk.Button(top, text='����ͼƬ', bg='#CC33CC', width=8, height=1, command=click_button).place(x=150,
                                                                                                               y=10,
                                                                                                               anchor='nw')
        test_btn = tk.Button(text='��������', master=top, bg='#CC33CC', command=lambda: test_func())
        test_btn.pack()
        test_btn.place(x=150+x_offset, y=10)

        UpdateMessage_0 = tk.StringVar()
        UpdateMessage_1 = tk.StringVar()
        UpdateMessage_2 = tk.StringVar()
        UpdateMessage_3 = tk.StringVar()


        StrClient = 'ip��ַ ' + ipStr
        Label(top, text=StrClient, font=("����", 13), fg="red", width=100, height=2).place(
            x=50,
            y=50,
            anchor='nw')




        # top.update()
        # top.after(1)


        while True:
            # time.sleep(1)
            # print(CompletedAcquisitionCnt[0])
            UpdateMessage_0.set('�ɹ�' + str(CompletedAcquisitionCnt[0]))
            Label(top, textvariable=UpdateMessage_0, font=("����", 13), fg="red", width=12,
                  height=2).place(x=250,
                                  y=10,
                                  anchor='nw')
            UpdateMessage_1.set('ʧ��' + str(FailedAcquisitionCnt[0]))
            Label(top, textvariable=UpdateMessage_1, font=("����", 13), fg="red", width=12, height=2).place(
                x=350,
                y=10,
                anchor='nw')
            # Label(top, text=ResultMessage[0], font=("����", 13), fg="red", width=100, height=3).place(
            #     x=50,
            #     y=50,
            #     anchor='nw')



            # top.update()
            # top.after(1)
            UpdateMessage_2.set('�ɹ�' + str(CompletedAcquisitionCnt[1]))
            Label(top, textvariable=UpdateMessage_2, font=("����", 13), fg="red", width=12,
                  height=2).place(x=250 + x_offset,
                                  y=10,
                                  anchor='nw')
            UpdateMessage_3.set('ʧ��' + str(FailedAcquisitionCnt[1]))
            Label(top, textvariable=UpdateMessage_3, font=("����", 13), fg="red", width=12,
                  height=2).place(x=350 + x_offset,
                                  y=10,
                                  anchor='nw')
            pic1 = tkImage(ImageAcquired[0])
            canvas.create_image(10, 100, anchor='nw', image=pic1)
            pic = tkImage(ImageAcquired[1])
            canvas.create_image(10 + x_offset, 100, anchor='nw', image=pic)
            top.update()


            # Label(top, text=ResultMessage[1], font=("����", 13), fg="red", width=100, height=3).place(
            #     x=50 + x_offset // 2,
            #     y=120,
            #     anchor='nw')

        top.mainloop()


class ImageSave(threading.Thread):
    # �������̵߳�ԭ���ǣ������ɫ�����߳��б���ͼƬ����ù��̻���ռ��ʵʱ�����ʱ�䣬�ʴ˴��������߳�
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global ImageAcquired, CompletedAcquisitionCnt, FailedAcquisitionCnt, ResultMessage

        CompletedAcquisitionCntOld = [0, 0]

        while True:
            for i in range(len(CompletedAcquisitionCntOld)):
                if CompletedAcquisitionCntOld[i] != CompletedAcquisitionCnt[i]:
                    try:
                        CompletedAcquisitionCntOld[i] = CompletedAcquisitionCnt[i]
                        cv2.imwrite("image" + str(i) + ".jpg", ImageAcquired[i])
                    except:
                        print('ͼƬ����ʧ�ܣ��������Ա���')


if __name__ == "__main__":
    globals()
    register_ctrl = ImageAcquistionAndDetect()
    thread1 = threading.Thread(target=register_ctrl.image_acquistion_and_detect, args=(0,))
    thread2 = threading.Thread(target=register_ctrl.image_acquistion_and_detect, args=(1,))



    # �����߳�����
    thread1.start()
    thread2.start()


    if image_save:
        thread4 = ImageSave()
        thread4.start()
    #     thread4.setDaemon(True)

    if img_view:
        thread3 = ImagePresentation()
        thread3.start()

    # if connect:
    #     thread5 = camera_connect()
    #     thread5.start()



    #�ȴ������߳�ִ�����
    if image_save:
        thread4.join()
    if img_view:
        thread3.join()
    # if connect:
    #     thread5.join()
    #
    thread1.join()
    thread2.join()
