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
from PIL import Image, ImageTk  # 图像控件
from pypylon import pylon
from pypylon import genicam
import modbus_tk.defines as cst
from modbus_tk import modbus_tcp

# 定义为全局变量，方便在色标检测进程和图片展示进程中同时使用
# 列表长度为2，表示2个相机
ImageAcquired = [np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.uint8)]  #rgb3通道10mal10
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
camera_run_type = 0  # 0：硬触发运行，1：自由运
image_save = 0
img_view = 1
reconnectCmd = 0
head_tail_distance_mm = 14 #落地首尾标
C1 = 0
C2 = 0

### 设置IP
ipStr = '192.168.1.66'
# ipStr = '127.0.0.1' #仿真
# ipStr = '192.168.137.66' #工控机
def num2register(num):  # 一个数字转换为modbustcp中的两个寄存器[低16位，高16位]
    return [num & 65535, (num >> 16) & 65535]


def register2num(register0, register1):  # modbustcp中的两个寄存器转换为一个数字[低16位，高16位]
    return register0 + (register1 << 16)


class MarkDetect:
    def __init__(self):
        global head_tail_distance_mm
        # 调试控制参数
        self.plt_show = 0  # 是否进行检测过程的图片显示
        self.profile_enable = 0  # 是否分析代码耗时

        ### 设置初始状态下的长度范围，对应像素宽的物理宽
        self.field_of_view_X_mm = 50  #长度 落地
        self.resolution_X = 2048
        self.mark_num = 5
        # self.head_tail_pixel = 700
        self.head_tail_x_pixel = 0
        self.head_tail_y_pixel = 0
        self.mark_detect_num = 0
        self.detect_step = 0
        self.mark_type_ix = -1



        # 菱形标+圆形标

        self.mark_type = 0  # 0：菱形标，1：圆形标
        self.mark_width = [1, 1.5] # 菱形对角线 圆形直径
        self.mark_height = [1, 1.5]
        self.rectangularity = [1, 0.785]
        self.limit = [0.6, 1.2]  # [0.7, 1.2]，筛选面积和长宽的最大最小倍数范围

        # 检测算法相关参数
        self.scaling = 3  # 3，先缩小3倍粗检测，时间花费和精度都比较合适
        self.blur_kernel = 35  # 9，处理之前高斯滤波窗口大小，9比较合适
        self.adaptive_block = 67  # 67，自适应二值化的窗口大小，对于目前测试的大小矩形标和三角标都较为合适
        self.C = 3  # 12，自适应二值化的阈值偏差, 光亮直接相关！！！！！
        self.par = []
        self.mark_area = []

        # 两个相机，但是目前该函数中只使用一个相机，两个相机在两个线程中分别用该函数实现，所以这里self.camera_ix直接设置为0，2相机可行？
        self.global_ymin = -1
        self.global_ymax = -1
        self.global_ROI_enable = 1
        self.camera_ix = 0 #改！
        self.par_init()

    def par_init(self):
        self.mark_width = [i * self.resolution_X / self.field_of_view_X_mm for i in self.mark_width]
        self.mark_height = [i * self.resolution_X / self.field_of_view_X_mm for i in self.mark_height]
        self.mark_area = [i * j * k for i, j, k in zip(self.mark_width, self.mark_height, self.rectangularity)]
        self.par = []
        for n in [1, self.scaling]:  # 参数预先计算
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
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]  # 寻找轮廓
        ### 输出黑白图，用户调试CV参数
        # cv2.imshow("draw",binary_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # 遍历所有轮廓，并进行判断
        result = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # 面积粗检,多种色标
            area_ok = 0
            for [a1, a2] in area_limit:
                if a1 < area < a2:
                    area_ok = 1
                    break
            if not area_ok:
                continue

            rect = cv2.minAreaRect(cnt)  # 轮廓最小外接矩形
            w, h = rect[1]

            size_ok = 0
            for [w1, w2], [h1, h2] in zip(width_limit, height_limit):  #外接矩形宽高判断
                if (w1 < w < w2 and h1 < h < h2) or (w1 < h < w2 and h1 < w < h2):
                    size_ok = 1
                    break
            if not size_ok: continue

            # if not 30 < abs(rect[2]) < 60:
            #     continue  # 菱形角度判断

            # 再综合判断一下，同时确定是哪种类型的色标，面积长宽矩形度色标类型判断
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

            # 计算色标中心点
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

            color = 3  # 0-青色， 1-红， 2-黄， 3-黑

            # if mark_V < 95:
            #     # color = 3
            #     hValue = max(hValues)
            # else:
            #     hValues = sorted(hValues)
            #     hValue = hValues[int(len(hValues)/2)]


            mark_Value = int(hValue)
            if 5 < mark_Value < 30:
                color = 0 #青
            elif 75 <= mark_Value <= 120:
                color = 2 #黄
            elif 140 < mark_Value < 169:
                color = 1 #红
            # elif 173 < mark_Value <= 180:
            #     color = 3 #黑

            if (mark_V < 90) and ((int(hValue_min) < 6) or (int(hValue_max) > 173)): #和照片亮度相关！光圈，光照，曝光时间
                 color = 3

            x_center, y_center = x_center + xmin, y_center + ymin

            # 计算外接最小举行顶点
            box = cv2.boxPoints(rect)  # 外接矩形的定点坐标
            box = np.int0(box)  # 坐标整型化

            cal_other_output = 0  # area 矩形度，灰度对比
            if cal_other_output:
                # 计算实际尺寸和标准尺寸的比值
                area_rate = area * n2 / self.mark_area[self.mark_type_ix]
                w_rate = w * n / self.mark_width[self.mark_type_ix]
                h_rate = h * n / self.mark_height[self.mark_type_ix]

                # 计算矩形度
                rectangularity = area / (w * h)

                # 计算色标与背景灰度差别
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

            # for i, r in enumerate(result):  # i序号 r内容
            #     if (result[:, 7].min() <= result[i, 7] <= (result[:, 7].min() + 20)) and (result[i, 7] < 90):
            #         result[i, 5] = 3
                # x_distance = result[:, 0] - r[0]  # 上次和这次x差别
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
                box = cv2.boxPoints(rect)  # 外接矩形的定点坐标
                box = np.int0(box)  # 坐标整型化
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

        # 计算原始大小
        origin_img_width, origin_img_height = img.shape[1], img.shape[0]

        _ymin, _ymax, _enable = self.global_ymin, self.global_ymax, self.global_ROI_enable
        if _ymin >= 0 and _ymax >= 0 and _enable:
            img = img[self.global_ymin:self.global_ymax, :]
            # 整张图resize一下，粗检测
            img_in = cv2.cvtColor(img[0:-1:scaling, 0:-1:scaling, :], cv2.COLOR_RGB2GRAY)  # 起始终止步长，resize重新处理
            img_HSV = cv2.cvtColor(img[0:-1:scaling, 0:-1:scaling, :], cv2.COLOR_RGB2HSV)  # 起始终止步长，resize重新处理
            # img_in = cv2.cvtColor(img[0:-1:scaling, 0:-1:scaling, :], cv2.COLOR_RGB2GRAY)  # 起始终止步长，resize重新处理
            # img_HSV = cv2.cvtColor(img[0:-1:scaling, 0:-1:scaling, :], cv2.COLOR_RGB2HSV)  # 起始终止步长，resize重新处理
        else:
            # 整张图resize一下，粗检测
            img_in = cv2.cvtColor(img[0:-1:scaling, 0:-1:scaling, :], cv2.COLOR_RGB2GRAY)  # 起始终止步长，resize重新处理
            img_HSV = cv2.cvtColor(img[0:-1:scaling, 0:-1:scaling, :], cv2.COLOR_RGB2HSV)  # 起始终止步长，resize重新处理

        self.detect_step = 0
        result_gray, img_, binary_img = self.mark_detect_single_channel(img_in,img_HSV,self.detect_step, n=scaling)


        # 精检测
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
                                    result_gray])  #矩阵
        result_pos = result_pos_gray.copy()  #copy

        # 如果只检测出了部分色标（大于一个，小于全部），则尝试单独取0通道检测其他色标
        if 1 <= len(result_gray) < self.mark_num:
            # 在上面灰度图检测出来色标的情况下，根据所有色标位置，截取色标所在周边的图像，取该截取图像的某个单独通道，再检测一遍
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

                # 第二次检测出至少一个色标的情况下，尝试合并
            if len(result_R_channel) >= 1:
                # result_pos_R_channel = result_pos_R_channel[np.argsort(result_pos_R_channel[:, 0])]  # 排序
                for i, r in enumerate(result_pos_R_channel):  #i序号 r内容
                    x_distance = result_pos[:, 0] - r[0]  #上次和这次x差别
                    if np.abs(x_distance).min() > (self.mark_height[self.mark_type_ix] + self.mark_width[self.mark_type_ix]) * 0.5:
                        result_pos = np.concatenate((result_pos, result_pos_R_channel[i:i + 1]), axis=0)

        if result_pos.shape[0] >= 3:  # 至少检出3个时，则下一次检测色标根据本次色标截取ROI
            # 将色标在图中的大致位置记录下来，下次检测时先默认检测该大致位置内的图片
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
        if result_pos.shape[0] >= 2:  # 至少检出2个
            detection_success = 1
            self.mark_detect_num = result_pos.shape[0]
            result_pos = result_pos[np.argsort(result_pos[:, 0])]  # 排序
            # 根据首尾标校准距离 result_pos[:, :2] = result_pos[:, :2] * self.head_tail_distance_mm / (result_pos[-1,
            # 0] - result_pos[0, 0])  #更换参数

            #计算首尾标间距
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
                            if abs(j - i) == 4:#首尾间距，防止识别错颜色
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
        包含两部分工作
        1.初始化相机
        2.初始化modbustcp
        """
        global ImageAcquired, IpListOld, cameras, converter, connect

        randomByteArray = bytearray(os.urandom(100000))
        flatNumpyArray = np.array(randomByteArray)

        grayImage = flatNumpyArray.reshape(200,500)


        self.img_default = grayImage
        ### 使用本地图片覆盖掉获取的图片，用于调试
        # self.img_default = cv2.imdecode(np.fromfile("image_14us.jpg", dtype=np.uint8), 1)
        ImageAcquired = [self.img_default, self.img_default]
        # head_tail_distance_mm = 14  # 落地

        cameras, converter, IpListOld = self.camera_init()
        connect = 0
        self.modbus_server = self.modbus_init()  #！
        ### 默认视野下的物理视野宽度
        self.filed_of_view_X_mm = 50 #落地

    def image_acquistion_and_detect(self, camera_ix):
        global ImageAcquired, CompletedAcquisitionCnt, FailedAcquisitionCnt, ResultMessage, CompletedDetectionCnt,\
            FailedDetectionCnt, IpListOld, cameras, converter, connect, event, camera_run_type,image_save, img_view,\
            reconnectCmd, head_tail_distance_mm

        ix = camera_ix
        camera = cameras[ix]
        mark_detect = MarkDetect()


        exposure_time_old_0 = 0  #落地
        exposure_time_old_1 = 0  #落地

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

            # 从modbus实时获取数据
            [mark_shape, mark_size, size_limit_min, size_limit_max, exposure_time, head_tail_distance_mm, image_save,
             img_view, camera_run_type, reconnectCmd] = self.get_modbus_data(ix)
            # print(self.get_modbus_data(ix))
            # 相机状态位置0
            #self.modbus_server.set_values(block_name='0', address=100 + ix * 100 + 1, values=[0])  #更换参数
            self.modbus_server.set_values(block_name='1', address=100 + ix * 2 + 2, values=num2register(0))
            # 每1s心跳增加1
            if time.time() - t_heartbeat_ctrl > 1:
                t_heartbeat_ctrl = time.time()
                num_heartbeat_ctrl = int(num_heartbeat_ctrl + 1) % 255
                #self.modbus_server.set_values(block_name='0', address=100 + ix * 100, values=[num_heartbeat_ctrl])  #更换参数
                self.modbus_server.set_values(block_name='1', address=100 + ix * 1, values=[num_heartbeat_ctrl])

            detection_success = 0
            try:
                # 曝光时间改变的时候才写入相机
                if (exposure_time_old_0 != exposure_time) and (camera_ix == 0):
                    camera.ExposureTimeAbs.SetValue(exposure_time)
                    exposure_time_old_0 = exposure_time
                elif (exposure_time_old_1 != exposure_time) and (camera_ix == 1):
                    camera.ExposureTimeAbs.SetValue(exposure_time)
                    exposure_time_old_1 = exposure_time

                if (camera_run_type_old_0 != camera_run_type) and (camera_ix == 0):
                    camera_run_type_old_0 = camera_run_type
                    if camera_run_type == 0:
                        # 设定Acquisition Mode
                        camera.AcquisitionMode.SetValue("Continuous")
                        # 设定TriggerSelector
                        camera.TriggerSelector.SetValue("FrameStart")
                        # 设定TriggerMode
                        camera.TriggerMode.SetValue("On")
                        # 设定TriggerSource
                        camera.TriggerSource.SetValue("Line1")
                        camera.TriggerActivation.SetValue("RisingEdge")
                        camera.TriggerDelayAbs.SetValue(0)
                    else:
                        # 设定Acquisition Mode
                        camera.AcquisitionMode.SetValue("Continuous")
                        # 设定TriggerSelector
                        camera.TriggerSelector.SetValue("FrameStart")
                        # 设定TriggerMode
                        camera.TriggerMode.SetValue("Off")
                elif (camera_run_type_old_1 != camera_run_type) and (camera_ix == 1):
                    camera_run_type_old_1 = camera_run_type
                    if camera_run_type == 0:
                        # 设定Acquisition Mode
                        camera.AcquisitionMode.SetValue("Continuous")
                        # 设定TriggerSelector
                        camera.TriggerSelector.SetValue("FrameStart")
                        # 设定TriggerMode
                        camera.TriggerMode.SetValue("On")
                        # 设定TriggerSource
                        camera.TriggerSource.SetValue("Line1")
                        camera.TriggerActivation.SetValue("RisingEdge")
                        camera.TriggerDelayAbs.SetValue(0)
                    else:
                        # 设定Acquisition Mode
                        camera.AcquisitionMode.SetValue("Continuous")
                        # 设定TriggerSelector
                        camera.TriggerSelector.SetValue("FrameStart")
                        # 设定TriggerMode
                        camera.TriggerMode.SetValue("Off")

                # 获取图片
                grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                time_retrieve = time.time() - t1  #可算采图时间
                t1 = time.time()

                # 判断图片获取是否成功
                if grabResult.GrabSucceeded():
                    # 相机状态位置2
                    #self.modbus_server.set_values(block_name='0', address=100 + ix * 100 + 1, values=[2])  #更换参数
                    self.modbus_server.set_values(block_name='1', address=100 + ix * 2 + 2, values=num2register(2))
                    # 相机图片处理标志位置1
                    #self.modbus_server.set_values(block_name='0', address=100 + ix * 100 + 3, values=[1])  #更换参数
                    self.modbus_server.set_values(block_name='1', address=100 + ix * 1 + 8, values=[1])

                    CompletedAcquisitionCnt[ix] = CompletedAcquisitionCnt[ix] + 1
                    # print(CompletedAcquisitionCnt[ix])
                    self.modbus_server.set_values(block_name='1', address=100 + ix * 2 + 70,
                                                  values=num2register(CompletedAcquisitionCnt[ix]))
                    image = converter.Convert(grabResult)  #format
                    ImageAcquired[ix] = image.GetArray()
                    ### 使用本地图片覆盖掉获取的图片，用于调试
                    # ImageAcquired[ix] = self.img_default

                    time_converter = time.time() - t1
                    t1 = time.time()


                    mark_detect.mark_num = 5
                    mark_detect.mark_type = mark_shape  # 0：菱形标，1：圆形标
                    mark_detect.mark_width = [0, 0]
                    mark_detect.mark_height = [0, 0]
                    mark_detect.mark_width[mark_shape] = mark_size
                    mark_detect.mark_height[mark_shape] = mark_size
                    mark_detect.limit = [size_limit_min, size_limit_max]
                    mark_detect.par_init()

                    # detection_success：至少检出两个色标即认为检出成功，否则检出失败
                    # result_pos：m行n列的numpy数组，m为检出的色标数目。即每一行就是一个色标的结果，第一列是X像素值，第二列是Y像素值，
                    # 第三列是色标轮廓最小外接矩形的角度，之后的列为其它输出，如需要可在程序中修改
                    # result_pos中所有的数据都是浮点型
                    detection_success, result_pos = mark_detect.mark_detect(ImageAcquired[ix])

                    if detection_success:
                        self.filed_of_view_X_mm = (float(head_tail_distance_mm / mark_detect.head_tail_pixel)) * 2048
                    # 色标检测
                    mark_detect.field_of_view_X_mm = self.filed_of_view_X_mm

                    # 计算一二标相对位置精度误差
                    ### 24.5.21 这一段可以注释可注释
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
                    # 打印数据
                    print('相机输入参数' + str(ix),
                          [mark_shape, mark_size, size_limit_min, size_limit_max, self.filed_of_view_X_mm,
                           exposure_time])
                    print('相机成功计数结果' + str(ix), CompletedDetectionCnt[ix])
                    print('心跳' + str(ix), num_heartbeat_ctrl)
                    print('相机检测结果' + str(ix), result_pos)
                    Period = 0
                    if CompletedAcquisitionCnt[ix] > 1:
                        Period = (time.time() - t00) / CompletedAcquisitionCnt[ix]
                    print('耗时' + str(ix), round(time.time() - t0, 3))
                    print('累计均时' + str(ix), round((time.time() - t00) / CompletedAcquisitionCnt[ix], 3))


                    time_detection = time.time() - t1
                    t1 = time.time()

                    # 测试照片存储
                    if CompletedAcquisitionCnt[ix] < -10:
                        os.makedirs("Data20230519\\Basler", exist_ok=True)
                        cv2.imwrite("Data20230519\\Basler\\" + str(time.time()) + ".jpg", ImageAcquired[1])
                    if camera_ix == 1 and detection_success == 1 and xxx < -10:
                        xxx = xxx + 1
                        os.makedirs("Data20230525\\Basler", exist_ok=True)
                        cv2.imwrite("Data20230525\\Basler\\" + str(time.time()) + ".jpg", ImageAcquired[1])

                    time_save = time.time() - t1

                    # 相机状态位置0
                    #self.modbus_server.set_values(block_name='0', address=100 + ix * 100 + 1, values=[0])  #更换参数
                    self.modbus_server.set_values(block_name='1', address=100 + ix * 2 + 2, values=num2register(0))
                    # trigger置0
                    # self.modbus_server.set_values(block_name='1', address=100 + ix * 1 + 6, values=[0])  #更换参数
                    # 相机图片处理标志位置0
                    #self.modbus_server.set_values(block_name='0', address=100 + ix * 100 + 3, values=[0])  #更换参数
                    self.modbus_server.set_values(block_name='1', address=100 + ix * 1 + 8, values=[0])

                else:
                    # 相机状态位置65534
                    #self.modbus_server.set_values(block_name='0', address=100 + ix * 100 + 1, values=[65534])  #更换参数
                    self.modbus_server.set_values(block_name='1', address=100 + ix * 2 + 2, values=[65534])
                    FailedAcquisitionCnt[ix] = FailedAcquisitionCnt[ix] + 1
                    ImageAcquired[ix] = self.img_default
                    detection_success = 0
                # time.sleep(min(max(0.1 - Period,0),0.05))
            except genicam.TimeoutException:
                # connect = 1
                # event.wait()
                print("相机" + str(ix) + "获取图片超时")



            print_str = "相机" + str(camera_ix) + ''
            if detection_success:
                print_str = print_str + '检测成功' + '，'
            else:
                print_str + '检测失败' + ','
            print_str = print_str + '耗时' + str(round(time.time() - t0, 3)) + ','
            print_str = print_str + '获取' + str(round(time_retrieve, 3)) + ','
            print_str = print_str + '转换' + str(round(time_converter, 3)) + ','
            print_str = print_str + '检测' + str(round(time_detection, 3)) + ','
            print_str = print_str + '保存' + str(round(time_save, 3)) + ','
            print_str = print_str + '累计均时' + str(round((time.time() - t00) / max(CompletedAcquisitionCnt[ix],1), 3))
            print_str = print_str + '检测时间' + str(round(time_detection, 3))

            if detection_success:
                print_str = print_str + '\n' + str(values)
            # print(print_str)
            ResultMessage[ix] = print_str  # 在gui上显示


    def camera_init(self):
        global cameras, IpListOld, converter
        cameras = []
        IpList = []
        tl_factory = pylon.TlFactory.GetInstance()  #遍历相机
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
        # 曝光参数设置
        camera.ExposureMode.SetValue("Timed")
        camera.ExposureTimeMode.SetValue("Standard")
        ### 设置曝光时间
        camera.ExposureTimeAbs.SetValue(30)

        # AOI设置
        if camera.Width.GetValue() > 1500:  # 大相机
            camera.Width.SetValue(2048)
            camera.Height.SetValue(800)
            camera.OffsetX.SetValue(0)
            camera.OffsetY.SetValue(300)

        # 设定MaxNumBuffer
        camera.MaxNumBuffer = 2

        if camera_run_type == 0:
            # 设定Acquisition Mode
            camera.AcquisitionMode.SetValue("Continuous")
            # 设定TriggerSelector
            camera.TriggerSelector.SetValue("FrameStart")
            # 设定TriggerMode
            camera.TriggerMode.SetValue("On")
            # 设定TriggerSource
            camera.TriggerSource.SetValue("Line1")
            camera.TriggerActivation.SetValue("RisingEdge")
            camera.TriggerDelayAbs.SetValue(0)
        else:
            # 设定Acquisition Mode
            camera.AcquisitionMode.SetValue("Continuous")
            # 设定TriggerSelector
            camera.TriggerSelector.SetValue("FrameStart")
            # 设定TriggerMode
            camera.TriggerMode.SetValue("Off")


        camera.StartGrabbing()

        return camera

    def modbus_init(self):
        global ipStr
        """
        可使用的函数:
        创建从站: server.add_slave(slave_id)
            slave_id(int):从站id
        为从站添加存储区: slave.add_block(block_name, block_type, starting_address, size)
            block_name(str):block名
            block_type(int):block类型,COILS = 1,DISCRETE_INPUTS = 2,HOLDING_REGISTERS = 3,ANALOG_INPUTS = 4
            starting_address(int):起始地址
            size(int):block大小
        设置block值:slave.set_values(block_name, address, values)
            block_name(str):block名
            address(int):开始修改的地址
            values(a list or a tuple or a number):要修改的一个(a number)或多个(a list or a tuple)值
        获取block值:slave.get_values(block_name, address, size)
            block_name(str):block名
            address(int):开始获取的地址
            size(int):要获取的值的数量
        """
        # 创建从站总服务器
        # server = modbus_tcp.TcpServer(address='127.0.0.1')  # address必须设置,port默认为502
        # server = modbus_tcp.TcpServer(address='192.168.137.66')  # address必须设置,port默认为502 工控机
        server = modbus_tcp.TcpServer(address=ipStr)  # address必须设置,port默认为502
        server.start()
        # 创建从站
        slave_1 = server.add_slave(1)  # slave_id = 1
        # 为从站添加存储区
        slave_1.add_block(block_name='0', block_type=cst.HOLDING_REGISTERS, starting_address=0, size=17)
        slave_1.add_block(block_name='1', block_type=cst.ANALOG_INPUTS, starting_address=100, size=74)
        print("Modbus tcp running...")

        return slave_1

    def get_modbus_data(self, camera_ix):
        global head_tail_distance_mm, ClientToServer, C1, C2
        modbusdata = self.modbus_server.get_values(block_name='0', address=0, size=17)  #更换参数
        ClientToServer = modbusdata
        mark_shape = modbusdata[0]
        mark_size = modbusdata[1] * 0.001  # um->mm
        #size_limit_min = modbusdata[3] * 0.01  #更换参数
        #size_limit_max = modbusdata[4] * 0.01  #更换参数
        size_limit_min = 0.6
        size_limit_max = 1.2
        # 根据相机索引号选择对应的数据
        if camera_ix == 0:
            #filed_of_view_X_mm = modbusdata[5]  #更换参数

            ### 设置初始状态下的长度范围，对应像素宽的物理宽
            filed_of_view_X_mm = 50
            #exposure_time = modbusdata[7]  #更换参数

            ### 设置曝光时间
            exposure_time = max(modbusdata[2],30) #standard 26以上限制
        elif camera_ix == 1:
            #filed_of_view_X_mm = modbusdata[6]  #更换参数

            ### 设置初始状态下的长度范围，对应像素宽的物理宽
            filed_of_view_X_mm = 50
            #exposure_time = modbusdata[8]  #更换参数

            ### 设置曝光时间
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

        # 对数据做一些限制
        if mark_shape not in [0, 1]:
            mark_shape = 0
        if mark_size <= 0:
            mark_size = 0.1
        if size_limit_min < 0:
            size_limit_min = 0
        if size_limit_max < 0:
            size_limit_max = 0
        if size_limit_max < size_limit_min:
            size_limit_max, size_limit_min = size_limit_min, size_limit_max  # 交换数据
        # if filed_of_view_X_mm <= 0:
        #     filed_of_view_X_mm = 0.1
        exposure_time = max(10, exposure_time)
        exposure_time = min(exposure_time, 100) #ultrashort 26以下
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
        # 需添加坐标补齐处理，全部对至5色标，对位应满足要求
        values = values + pixel_pos_x_register + pixel_pos_y_register + color_register
        # self.modbus_server.set_values(block_name='0', address=100 + ix * 100 + 4, values=values)  #更换参数
        self.modbus_server.set_values(block_name='1', address=100 + ix * 25 + 16, values = values)
        # print(self.modbus_server.get_values(block_name='1', address=100, size=74))
        # self.modbus_server.set_values(block_name='1', address=100 + ix * 2 + 11, values = [mark_num])
        # self.modbus_server.set_values(block_name='1', address=100 + ix * 1 + 15, values = [mark_num])
        # self.modbus_server.set_values(block_name='1', address=100 + ix * 25 + 16, values = pixel_pos_x_register)
        return values

class camera_connect(threading.Thread):
    # 开启此线程的原因是：如果在色标检测线程中保存图片，则该过程会大大占用实时任务的时间，故此处开启新线程
    def __init__(self):
        threading.Thread.__init__(self)


    def run(self):
        global IpListOld, cameras, converter, connect, event, camera_run_type
        cameras = []
        IpList = []
        tl_factory = pylon.TlFactory.GetInstance()  # 遍历相机
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
                # 0：自由运行  https://docs.baslerweb.com/free-run-image-acquisition
                # 1：硬触发运行 https://docs.baslerweb.com/triggered-image-acquisition
                camera.ExposureMode.SetValue("Timed")
                camera.ExposureTimeMode.SetValue("Standard")
                camera.ExposureTimeAbs.SetValue(30)  # 落地 ####

                # AOI设置
                if camera.Width.GetValue() > 1500:  # 大相机
                    camera.Width.SetValue(2048)
                    camera.Height.SetValue(800)
                    camera.OffsetX.SetValue(0)
                    camera.OffsetY.SetValue(300)

                # 设定MaxNumBuffer
                camera.MaxNumBuffer = 2

                if camera_run_type == 0:
                    # 设定Acquisition Mode
                    camera.AcquisitionMode.SetValue("Continuous")
                    # 设定TriggerSelector
                    camera.TriggerSelector.SetValue("FrameStart")
                    # 设定TriggerMode
                    camera.TriggerMode.SetValue("On")
                    # 设定TriggerSource
                    camera.TriggerSource.SetValue("Line1")
                    camera.TriggerActivation.SetValue("RisingEdge")
                    camera.TriggerDelayAbs.SetValue(0)
                else:
                    # 设定Acquisition Mode
                    camera.AcquisitionMode.SetValue("Continuous")
                    # 设定TriggerSelector
                    camera.TriggerSelector.SetValue("FrameStart")
                    # 设定TriggerMode
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

        # 当按钮被点击的时候执行click_button()函数
        def click_button():
            # 使用消息对话框控件，showinfo()表示温馨提示
            try:
                cv2.imwrite("image0" + str(CompletedAcquisitionCnt[0]) + ".jpg", ImageAcquired[0])
                cv2.imwrite("image1" + str(CompletedAcquisitionCnt[1]) + ".jpg", ImageAcquired[1])
            except:
                print('图片保存失败，继续尝试保存')

        # def on_closing():
        #     pass

        # 界面画布更新图像
        top = tk.Tk()
        top.title('视频窗口')
        top.geometry('1050x300+800+175')
        top.overrideredirect(True)
        # top.protocol("WM_DELETE_WINDOW", on_closing)
        top.lift()
        image_width = 2048
        image_height = 800
        canvas = Canvas(top, bg='white', width=image_width, height=image_height)  # 绘制画布
        Label(top, text='相机图片', font=("黑体", 13), width=15, height=1).place(x=10, y=10, anchor='nw')
        canvas.place(x=0, y=0)
        x_offset = 500

        # 点击按钮时执行的函数
        button = tk.Button(top, text='保存图片', bg='#CC33CC', width=8, height=1, command=click_button).place(x=150,
                                                                                                               y=10,
                                                                                                               anchor='nw')
        test_btn = tk.Button(text='数据清零', master=top, bg='#CC33CC', command=lambda: test_func())
        test_btn.pack()
        test_btn.place(x=150+x_offset, y=10)

        UpdateMessage_0 = tk.StringVar()
        UpdateMessage_1 = tk.StringVar()
        UpdateMessage_2 = tk.StringVar()
        UpdateMessage_3 = tk.StringVar()


        StrClient = 'ip地址 ' + ipStr
        Label(top, text=StrClient, font=("黑体", 13), fg="red", width=100, height=2).place(
            x=50,
            y=50,
            anchor='nw')




        # top.update()
        # top.after(1)


        while True:
            # time.sleep(1)
            # print(CompletedAcquisitionCnt[0])
            UpdateMessage_0.set('成功' + str(CompletedAcquisitionCnt[0]))
            Label(top, textvariable=UpdateMessage_0, font=("黑体", 13), fg="red", width=12,
                  height=2).place(x=250,
                                  y=10,
                                  anchor='nw')
            UpdateMessage_1.set('失败' + str(FailedAcquisitionCnt[0]))
            Label(top, textvariable=UpdateMessage_1, font=("黑体", 13), fg="red", width=12, height=2).place(
                x=350,
                y=10,
                anchor='nw')
            # Label(top, text=ResultMessage[0], font=("黑体", 13), fg="red", width=100, height=3).place(
            #     x=50,
            #     y=50,
            #     anchor='nw')



            # top.update()
            # top.after(1)
            UpdateMessage_2.set('成功' + str(CompletedAcquisitionCnt[1]))
            Label(top, textvariable=UpdateMessage_2, font=("黑体", 13), fg="red", width=12,
                  height=2).place(x=250 + x_offset,
                                  y=10,
                                  anchor='nw')
            UpdateMessage_3.set('失败' + str(FailedAcquisitionCnt[1]))
            Label(top, textvariable=UpdateMessage_3, font=("黑体", 13), fg="red", width=12,
                  height=2).place(x=350 + x_offset,
                                  y=10,
                                  anchor='nw')
            pic1 = tkImage(ImageAcquired[0])
            canvas.create_image(10, 100, anchor='nw', image=pic1)
            pic = tkImage(ImageAcquired[1])
            canvas.create_image(10 + x_offset, 100, anchor='nw', image=pic)
            top.update()


            # Label(top, text=ResultMessage[1], font=("黑体", 13), fg="red", width=100, height=3).place(
            #     x=50 + x_offset // 2,
            #     y=120,
            #     anchor='nw')

        top.mainloop()


class ImageSave(threading.Thread):
    # 开启此线程的原因是：如果在色标检测线程中保存图片，则该过程会大大占用实时任务的时间，故此处开启新线程
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
                        print('图片保存失败，继续尝试保存')


if __name__ == "__main__":
    globals()
    register_ctrl = ImageAcquistionAndDetect()
    thread1 = threading.Thread(target=register_ctrl.image_acquistion_and_detect, args=(0,))
    thread2 = threading.Thread(target=register_ctrl.image_acquistion_and_detect, args=(1,))



    # 启动线程运行
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



    #等待所有线程执行完毕
    if image_save:
        thread4.join()
    if img_view:
        thread3.join()
    # if connect:
    #     thread5.join()
    #
    thread1.join()
    thread2.join()
