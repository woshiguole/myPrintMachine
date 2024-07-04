# -*- coding: utf-8 -*-
"""
文件描述：用于检测色标图片的中心点位置，颜色等信息
作者：Wenjie Cui，Dr. Zhu
创建日期：2024.5.23
最后修改日期：2024.5.28

快速检索标志
##    代表日常修改注释
###   代表初次接收程序时的中点标注
####  代表主要工艺注释
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from array import array
import yaml
import tkinter as tk
from tkinter import filedialog
import os
from collections import deque

class MarkDetect:
    # ROI位置
    global_ROI_ymin = -1
    global_ROI_ymax = -1
    global_ROI_enable = 1
        
    def __init__(self, yaml_path: str):
        # 相机设置        
        self.head_tail_distance_mm = None
        self.field_of_view_X_mm = None
        self.resolution_X = None
        self.resolution_Y = None
        
        # 检测算法相关参数
        self.mark_num = None
        self.mark_type = None
        self.mark_width = None
        self.mark_height = None
        self.rectangularity = None
        self.limit = None
        self.scaling = None
        self.blur_kernel = None
        self.adaptive_block = None
        self.C = None
        self.C1 = None
        self.C2 = None

        self.load_para(yaml_path)

        # 调试控制参数
        self.__profile_enable = 0  # 是否分析代码耗时

        # 内部变量
        self.__mark_width_pixel = None # 圆标直径1-1.5mm，宽度转换为对应的像素大小
        self.__mark_height_pixel = None # 圆标直径1-1.5mm，高度转换为对应的像素大小
        self.__mark_area_pixel = 0 # 色标面积
        self.__head_tail_x_pixel = 0
        self.__head_tail_y_pixel = 0
        self.__head_tail_pixel = 0 # 首尾两个色标的像素距离 
        self.__mark_type_ix = -1

        self.__par = []

        # 两个相机，但是目前该函数中只使用一个相机，两个相机在两个线程中分别用该函数实现，所以这里self.__camera_ix直接设置为0，2相机可行？
        self.__camera_ix = 0  # 改！
        self.par_init()

    def load_para(self, yaml_path: str) -> None:
        # 加载并读取 YAML 文件
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        ### 设置初始状态下的长度范围，对应像素宽的物理宽
        self.field_of_view_X_mm = data['camera_setting']['field_of_view_X_mm']  # 视野X（mm）
        self.head_tail_distance_mm = data['camera_setting']['head_tail_distance_mm'] # 首尾两个色标的实际间距（mm）
        self.resolution_X = data['camera_setting']['resolution_X'] # 像素X
        self.resolution_Y = data['camera_setting']['resolution_Y'] # 像素Y

        # 检测算法相关参数
        self.mark_num = data['blob_setting']['mark_num'] # 检测标记数量
        self.mark_type = data['blob_setting']['mark_type']  # 0：菱形标，1：圆形标
        self.mark_width = data['blob_setting']['mark_width']  # 菱形对角线 圆形直径
        self.mark_height = data['blob_setting']['mark_height'] # 菱形对角线 圆形直径
        self.rectangularity = data['blob_setting']['rectangularity']  # PI/4 = 0.785
        self.limit = data['blob_setting']['limit']  # [0.7, 1.2]，筛选面积和长宽的最大最小倍数范围
        self.scaling = data['blob_setting']['scaling']  # 3，先缩小3倍粗检测，时间花费和精度都比较合适
        self.blur_kernel = data['blob_setting']['blur_kernel']  # 9，处理之前高斯滤波窗口大小，9比较合适
        self.adaptive_block = data['blob_setting']['adaptive_block']  # 67，自适应二值化的窗口大小，对于目前测试的大小矩形标和三角标都较为合适
        self.C = data['blob_setting']['C']  # 12，自适应二值化的阈值偏差, 光亮直接相关！！！！！
        self.C1 = data['blob_setting']['C1']
        self.C2 = data['blob_setting']['C2']

    def update_para(self, para: array) -> None:
        self.head_tail_distance_mm = para[0]
        self.C1 = para[1]
        self.C2 = para[2]

    def par_init(self) -> None:
        # 圆标直径1-1.5mm，转换为对应的像素大小，（2048/50）* [1, 1.5]
        # 检测圆形标时只计算圆形标参数，检测菱形标时只计算菱形标参数
        self.__mark_width_pixel = self.mark_width[self.mark_type] * self.resolution_X / self.field_of_view_X_mm 
        self.__mark_height_pixel = self.mark_height[self.mark_type] * self.resolution_X / self.field_of_view_X_mm 
        self.__mark_area_pixel = self.__mark_width_pixel * self.__mark_height_pixel * self.rectangularity[self.mark_type]
        
        # 根据预先设定的筛选大小范围self.limit，分别计算scaling和不scaling参数，计算长、宽、面积的限制
        for n in [1, self.scaling]:  # 参数预先计算
            n2 = n * n
            area_limit = [self.__mark_area_pixel * self.limit[0] / n2, self.__mark_area_pixel * self.limit[1] / n2] 
            width_limit_pixel = [self.__mark_width_pixel * self.limit[0] / n, self.__mark_width_pixel * self.limit[1] / n]
            height_limit_pixel = [self.__mark_height_pixel * self.limit[0] / n, self.__mark_height_pixel * self.limit[1] / n] 
            # 矩形度，菱形是1，圆形是0.785，根据self.limit设置放缩的范围
            rectangularity_limit = [self.rectangularity[self.mark_type] * self.limit[0], self.rectangularity[self.mark_type] * self.limit[1]]
            blur_kernel = int(round(self.blur_kernel / n, 0))
            blur_kernel = blur_kernel if blur_kernel % 2 else blur_kernel + 1
            adaptive_block = int(round(self.adaptive_block / n, 0))
            adaptive_block = adaptive_block if adaptive_block % 2 else adaptive_block + 1
            
            self.__par.append([n, n2, area_limit, width_limit_pixel, height_limit_pixel, rectangularity_limit, blur_kernel, adaptive_block])

    def mark_detect_main(self, img, detect_func):
        # 计算原始大小
        origin_img_width, origin_img_height = img.shape[1], img.shape[0]
            
        # 工艺一：对整图resize，进行第一次粗检测  
        img_resize, result_rough = self.mark_detect_rough(img, detect_func)    

        # 工艺二：找出粗检测的范围，进行精检测
        if self.scaling != 1:
            result_pos = self.mark_detect_fine(img_resize, result_rough, detect_func)
        
        # 工艺三，测试结果数量不足就补测 
        if 1 <= result_pos.shape[0] < self.mark_num:
            result_pos = self.mark_detect_num_too_few(img_resize, result_pos, detect_func)
            
        # 工艺四，如果检测结果过多，对检测结果中明显存在角度偏差的菱形对象进行删除    
        if self.mark_type == 0 and result_pos.shape[0] > self.mark_num:
            result_pos = self.mark_detect_num_too_many(result_pos) 
        
        # 计算首尾两个色标的像素距离 
        if result_pos.shape[0] >= 2:
            result_pos = result_pos[np.argsort(result_pos[:, 0])]  # 排序
            detection_success = True
            self.mark_detect_calc_head_tail_pixel(result_pos)
        else:
            detection_success = False  
        
        # 将检测范围ROI加回检测结果
        if MarkDetect.global_ROI_ymin >= 0 and MarkDetect.global_ROI_ymax >= 0 and MarkDetect.global_ROI_enable:
            result_pos[:, 1] = result_pos[:, 1] + MarkDetect.global_ROI_ymin
            
        if result_pos.shape[0] >= 3:  
            # 将色标在图中的大致位置记录下来，下次检测时先默认检测该大致位置内的图片
            self.mark_detect_calc_next_roi(origin_img_height, result_pos)
        else:
            MarkDetect.global_ROI_ymin = -1
            MarkDetect.global_ROI_ymax = -1
            
        return detection_success, result_pos
    
    def mark_detect_rough(self, img, detect_func):        
        """
        ####  工艺一：对整图resize，进行第一次粗检测        
        """
        if MarkDetect.global_ROI_ymin >= 0 and MarkDetect.global_ROI_ymax >= 0 and MarkDetect.global_ROI_enable:
            # 整张图resize一下，粗检测
            img_resize = img[MarkDetect.global_ROI_ymin:MarkDetect.global_ROI_ymax, :]
            img_in = cv2.cvtColor(img_resize[0:-1:self.scaling, 0:-1:self.scaling, :], cv2.COLOR_RGB2GRAY)  # 起始终止步长，resize重新处理
            img_HSV = cv2.cvtColor(img_resize[0:-1:self.scaling, 0:-1:self.scaling, :], cv2.COLOR_RGB2HSV)  # 起始终止步长，resize重新处理
        else:
            # 0:-1:scaling这个切片的含义是从第一个像素开始（索引0），到倒数第二个像素结束（索引-1，不包含-1）
            # 步长为scaling。这样就实现了对图像进行缩放的效果。
            img_resize = img
            img_in = cv2.cvtColor(img_resize[0:-1:self.scaling, 0:-1:self.scaling, :], cv2.COLOR_RGB2GRAY)  # 将RGB图像转换为灰度图像
            img_HSV = cv2.cvtColor(img_resize[0:-1:self.scaling, 0:-1:self.scaling, :], cv2.COLOR_RGB2HSV)  # 将RGB图像转换为HSV颜色空间
        result_rough = detect_func(img_in, img_HSV, n=self.scaling, detect_step=0)
        
        return img_resize, result_rough
       
    def mark_detect_fine(self, img, result_rough, detect_func):
        """
        #### 工艺二：找出粗检测的范围，进行精检测
        """
        accurate_result = []
        for x_center, y_center, angle, radius, range, *o in result_rough:
            xmin, xmax = range[0][0], range[0][1] # 计算四个点的XY最小值
            ymin, ymax = range[1][0], range[1][1]
            w, h = xmax - xmin, ymax- ymin #计算外接最小矩形的XY正方向宽度和高度
            x, y, w, h = xmin * self.scaling, ymin * self.scaling, w * self.scaling, h * self.scaling
            edge = 30
            xmin = max(x - edge, 0)
            xmax = min(x + w + edge, img.shape[1])
            ymin = max(y - edge, 0)
            ymax = min(y + h + edge, img.shape[0])
            img_in = img[ymin:ymax, xmin:xmax]
            img_in = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)
            img_HSV = img[ymin:ymax, xmin:xmax]
            img_HSV = cv2.cvtColor(img_HSV, cv2.COLOR_RGB2HSV)
            result_ = detect_func(img_in, img_HSV, n=1, detect_step=1, xmin=xmin, ymin=ymin)
            if len(result_) > 0:
                accurate_result.append(result_[0])
        result_gray_fine = accurate_result
        
        result_pos = []
        result_pos = np.array(
            [[x_center, y_center, angle, radius, color, mark_Value, detect_step, meangray_dis, rectangularity] for
             x_center, y_center, angle, radius, range, color, mark_Value, detect_step, meangray_dis, rectangularity, *o in
             result_gray_fine])  # 矩阵
                
        return result_pos

    def mark_detect_num_too_few(self, img, result_pos, detect_func):
        """
        #### 工艺三，测试结果数量不足就补测        
        #### 如果只检测出了部分色标（大于一个，小于全部），则尝试单独取0通道检测其他色标
        """
        # 在上面灰度图检测出来色标的情况下，根据所有色标位置，截取色标所在周边的图像，取该截取图像的某个单独通道，再检测一遍
        if self.C2 != 0:
            self.C = self.C2
        else:
            self.C = 4

        # 将检测范围改为y方向的小范围内，大概100多的像素范围
        ymax, ymin = result_pos[:, 1].max(), result_pos[:, 1].min()
        ymin = int(max(ymin - self.__mark_height_pixel - 20, 0))
        ymax = int(min(ymax + self.__mark_height_pixel + 20, img.shape[0]))

        img_R_channel = img[ymin:ymax, :, 0]
        img_HSV = img[ymin:ymax, :]
        img_HSV = cv2.cvtColor(img_HSV, cv2.COLOR_RGB2HSV)

        result_R_channel = detect_func(img_R_channel, img_HSV, n=1, detect_step=2, xmin=0, ymin=ymin)
        result_pos_R_channel = np.array(
            [[x_center, y_center, angle, radius, color, mark_Value, detect_step, meangray_dis, rectangularity] for
                x_center, y_center, angle, radius, range, color, mark_Value, detect_step, meangray_dis, rectangularity, *o in
                result_R_channel])

        if self.C1 != 0:
            self.C = self.C1
        else:
            self.C = 3

        # 第二次检测出至少一个色标的情况下，尝试合并
        if len(result_R_channel) >= 1:
            # result_pos_R_channel = result_pos_R_channel[np.argsort(result_pos_R_channel[:, 0])]  # 排序
            for i, r in enumerate(result_pos_R_channel):  # i序号 r内容
                x_distance = result_pos[:, 0] - r[0]  # 上次和这次x差别
                if np.abs(x_distance).min() > (self.__mark_height_pixel + self.__mark_width_pixel) * 0.5:
                    result_pos = np.concatenate((result_pos, result_pos_R_channel[i:i + 1]), axis=0)
                    
        return result_pos
    
    def mark_detect_num_too_many(self, result_pos):
        """
        ## 工艺四，2024.5.23 对检测结果中明显存在角度偏差的菱形对象进行删除
        ## 只有在检测结果result_pos的数量大于self.mark_num时，才会执行该操作。
        """  
        # 计算中点（平均值）
        midpoint = np.mean(result_pos[:, 2])
        # 计算每个数据点与中点的绝对偏差
        deviations = np.abs(result_pos[:, 2] - midpoint)
        # 找到偏离度较大的值
        n_outliers = result_pos.shape[0] - self.mark_num
        outlier_indices = np.argsort(deviations)[-n_outliers:]  # 如果n_outliers=0， 则所有索引都会包含，result_pos会被删光
        result_pos = np.delete(result_pos, outlier_indices, axis=0)    
        
        return result_pos
            
    def mark_detect_blob(self, img, img_HSV, n=1, detect_step=0, xmin=0, ymin=0):
        if n == 1:
            n, n2, self.area_limit, width_limit_pixel, height_limit_pixel, rectangularity_limit, blur_kernel, adaptive_block = self.__par[0]
        else:  # n为scaling，如果scaling不为1，则执行 self.par[1]
            n, n2, self.area_limit, width_limit_pixel, height_limit_pixel, rectangularity_limit, blur_kernel, adaptive_block = self.__par[1]

        # img = img[0:-1:n, 0:-1:n]
        # img_bilateralFilter = cv2.bilateralFilter(img, blur_kernel, 15, 5)
        # img_medianBlur = cv2.medianBlur(img, blur_kernel)
        img_Gaussian = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
        binary_img = cv2.adaptiveThreshold(img_Gaussian, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, adaptive_block, self.C)
        # self.binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_ERODE, np.ones((3, 3), np.uint8), iterations=1)
        self.contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]  # 寻找轮廓
        # contours 在这种情况下是ndarray，(N, 1, 2) 的形式表示：
        # N：表示轮廓上的点的数量。
        # 1：表示每个点有一个维度。
        # 2：表示每个点有两个值，分别是 x 和 y 坐标。
            
        # 遍历所有轮廓，并进行判断
        result = []
        for i, cnt in enumerate(self.contours):
            area = cv2.contourArea(cnt) # 主要用于计算图像轮廓的面积,通常搭配findContours()函数使用
            rect = cv2.minAreaRect(cnt)  # 轮廓最小外接矩形, rect[0]表示center(x,y), rect[1]表示width,height, rect[2]表示angle of rotation            
            box = cv2.boxPoints(rect)  # 计算外接最小矩形顶点， 外接矩形的定点坐标
            box = np.intp(box)  # 坐标整型化
            
            x_range = [box[:, 0].min(), box[:, 0].max()]
            y_range = [box[:, 1].min(), box[:, 1].max()]
            range = [x_range, y_range]
            radius = (box[:, 0].max() - box[:, 0].min()+box[:, 1].max() - box[:, 1].min())/4
            
            (x_center, y_center), (w, h), angle = rect
                        
            # 面积判断、边长判断、矩形度判断
            mark_ok = self.mark_detect_blob_markOk(area, w, h, width_limit_pixel, height_limit_pixel, rectangularity_limit)
            if mark_ok == 1:
                hValue, vValue, hValue_min, hValue_max = self.mark_detect_calc_h_v(img_HSV, y_center, x_center, h, w, img.shape)
                color, mark_Value = self.mark_detect_calc_color(hValue, vValue, hValue_min, hValue_max)
                meangray_dis, rectangularity, area_rate, w_rate, h_rate = self.mark_detect_calc_others(img, range, area, w, h, n, n2)

                x_center, y_center = x_center + xmin, y_center + ymin

                result.append([x_center, y_center, angle, radius, range, color, mark_Value, detect_step, meangray_dis, rectangularity, area_rate, w_rate, h_rate])

            # for i in range(0, len(result)-1):
            #     if (result[:, 7].min() <= result[i, 7] <= (result[:, 7].min() + 20)) and (result[i, 7] < 90):
            # result[i, 5] = 3

            # for i, r in enumerate(result):  # i序号 r内容
            #     if (result[:, 7].min() <= result[i, 7] <= (result[:, 7].min() + 20)) and (result[i, 7] < 90):
            #         result[i, 5] = 3
            # x_distance = result[:, 0] - r[0]  # 上次和这次x差别
            # if np.abs(x_distance).min() > (
            #         self.__mark_height_pixel + self.__mark_width_pixel) * 0.5:
            #     result_pos = np.concatenate((result_pos, result_pos_R_channel[i:i + 1]), axis=0)

        return result

    def mark_detect_hough(self, img, img_HSV, n=1, detect_step=0, xmin=0, ymin=0):
        if n == 1:
            n, n2, self.area_limit, width_limit_pixel, height_limit_pixel, rectangularity_limit, blur_kernel, adaptive_block = self.__par[0]
        else:  # n为scaling，如果scaling不为1，则执行 self.par[1]
            n, n2, self.area_limit, width_limit_pixel, height_limit_pixel, rectangularity_limit, blur_kernel, adaptive_block = self.__par[1]
            
        img_Gaussian = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
        
        TL, TH = 25, 50        
        img_Canny = cv2.Canny(img_Gaussian, TL, TH, 0)
        
        # Hough变换圆检测
        circles = cv2.HoughCircles(img_Canny, cv2.HOUGH_GRADIENT, 1, 30, param1=TH, param2=10, minRadius=int(width_limit_pixel[0]), maxRadius=int(width_limit_pixel[1]))
        img_Fade = cv2.convertScaleAbs(img, alpha=0.5, beta=128)
            
        result = []
        if circles is not None:
            circles = np.uint16(np.around(circles))            
            for i, circle in enumerate(circles[0, :]):
                x_center, y_center, radius = circle[0], circle[1], circle[2]
                print("i={}, x={}, y={}, r={}".format(i, x_center, y_center, radius))
                cv2.circle(img_Fade, (x_center,y_center), radius, (255,0,0), 2) #绘制圆
                cv2.circle(img_Fade, (x_center,y_center), 2, (0,0,255), 2) #绘制圆心                
                
                h = radius
                w = radius
                area = 3.14*radius*radius
                x_range = [x_center-radius, x_center+radius]
                y_range = [y_center-radius, y_center+radius]
                range = [x_range, y_range]
                angle = 0
            
                hValue, vValue, hValue_min, hValue_max = self.mark_detect_calc_h_v(img_HSV, y_center, x_center, h, w, img.shape)
                color, mark_Value = self.mark_detect_calc_color(hValue, vValue, hValue_min, hValue_max)
                meangray_dis, rectangularity, area_rate, w_rate, h_rate = self.mark_detect_calc_others(img, range, area, w, h, n, n2)

                x_center, y_center = x_center + xmin, y_center + ymin

                result.append([x_center, y_center, angle, radius, range, color, mark_Value, detect_step, meangray_dis, rectangularity, area_rate, w_rate, h_rate])
            plt.figure(figsize=(9,4))
            plt.subplot(131)
            plt.imshow(img, cmap='gray')
            plt.subplot(132)
            plt.imshow(cv2.bitwise_not(img_Canny), cmap='gray')
            plt.subplot(133)
            plt.imshow(img_Fade, cmap='gray')
            plt.show()    
        return result
    
    def mark_detect_match(self, img, img_HSV, n=1, detect_step=0, xmin=0, ymin=0):
        if n == 1:
            n, n2, self.area_limit, width_limit_pixel, height_limit_pixel, rectangularity_limit, blur_kernel, adaptive_block = self.__par[0]
        else:  # n为scaling，如果scaling不为1，则执行 self.par[1]
            n, n2, self.area_limit, width_limit_pixel, height_limit_pixel, rectangularity_limit, blur_kernel, adaptive_block = self.__par[1]
            
        # 创建一张白色背景的图片
        image = np.ones((100, 100, 3), dtype=np.uint8)*255

        # 使用cv2.circle()生成圆形
        cv2.circle(image, center=(50, 50), radius=13, color=(255, 0, 0), thickness=-1)
        refer = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        sift = cv2.SIFT.create()
        kpRef, desRef = sift.detectAndCompute(refer, None)
        kpObj, desObj = sift.detectAndCompute(img, None)
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desRef, desObj, k=2)
        goodMatches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                goodMatches.append([m])
                
        imgMatches = cv2.drawMatchesKnn(refer, kpRef, img, kpObj, goodMatches, None, matchColor=(0,255,0))
        print("(2) bf.knnMatch:{}, goodMatch:{}".format(len(matches), len(goodMatches)))
        print(type(matches), type(matches[0]), type(matches[0][0]))
        print(matches[0][0].distance)
        plt.figure(figsize=(9, 6))
        plt.imshow(cv2.cvtColor(imgMatches, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.show()
        
    def mark_detect_template(self, img, img_HSV, n=1, detect_step=0, xmin=0, ymin=0):
        if n == 1:
            n, n2, self.area_limit, width_limit_pixel, height_limit_pixel, rectangularity_limit, blur_kernel, adaptive_block = self.__par[0]
        else:  # n为scaling，如果scaling不为1，则执行 self.par[1]
            n, n2, self.area_limit, width_limit_pixel, height_limit_pixel, rectangularity_limit, blur_kernel, adaptive_block = self.__par[1]   

        # 寻找圆的半径，同时用于生成圆模板
        radius = int(self.__mark_width_pixel/n/2)
        
        # 创建一张白色背景的图片
        image = np.ones((int(radius*2+30),int(radius*2+30), 3), dtype=np.uint8)*255

        if self.mark_type == 1:
            # 圆形标，使用cv2.circle()生成圆形
            cv2.circle(image, center=(int(radius+15), int(radius+15)), radius=int(radius), color=(255, 0, 0), thickness=-1)
        else:            
            # 菱形标，菱形的顶点
            pts = np.array([[int(radius+15), int(2*radius+15)], [int(2*radius+15), int(radius+15)], [int(radius+15), 15], [15, int(radius+15)]], np.int32)
            
            # 绘制菱形轮廓
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # 填充菱形形成实心菱形
            cv2.fillPoly(image, [pts], color=(0, 255, 0))
        
        refer = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)       
        
        # 获取查询图像的尺寸
        wRefer, hRefer = refer.shape[::-1]
        
        # 进行模板匹配
        res = cv2.matchTemplate(img, refer, cv2.TM_CCOEFF_NORMED)
        
        # 找到最佳匹配位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
        # 如果找到多个匹配结果，需要遍历所有位置
        threshold = 0.8  # 设置一个阈值，根据需要调整
        locations = np.where(res >= max(0.5,threshold*max_val))  # 找到所有大于阈值的位置
        
        # 使用列表推导式计算每个坐标的平方和
        coord_sums = [(x**2 + y**2) for x, y in zip(locations[0], locations[1])]
        
        # 对坐标平方和进行排序
        sorted_indices = sorted(range(len(coord_sums)), key=coord_sums.__getitem__)
        
        # 根据排序后的索引返回排序后的坐标
        sorted_locations = [(locations[0][i], locations[1][i]) for i in sorted_indices]
        
        result = []
        # 根据位置绘制矩形框      
        groups = deque()  # 使用队列来存储分组    
        for (centerY, centerX) in (sorted_locations):  # 逆序是因为loc得到的是列首先的坐标            
            disOK = False
            if not groups:
                groups.append([(centerX, centerY)])
            else:
                for grouped_point in groups[-1]:
                    if ((grouped_point[0] - centerX) ** 2 + (grouped_point[1] - centerY) ** 2) ** 0.5 <= radius:
                        disOK = True
                        break
                if disOK:
                    groups[-1].append((centerX, centerY))
                else:
                    groups.append([(centerX, centerY)])
        
        for group in groups:
            # 根据匹配得分，加权计算匹配中心
            x_center, y_center = 0, 0 
            resMark = 0
            for (centerX, centerY) in group:
                x_center = x_center+centerX*res[centerY][centerX]
                y_center = y_center+centerY*res[centerY][centerX]
                resMark = resMark + res[centerY][centerX]
            x_center =  x_center/resMark+wRefer/2
            y_center =  y_center/resMark+hRefer/2
            # cv2.rectangle(img, (int(x_center-wRefer/2), int(y_center-hRefer/2)), (int(x_center + wRefer/2), int(y_center + hRefer/2)), (0, 0, 255), 1)
            # cv2.circle(img, (int(x_center), int(y_center)), radius=2, color=(0, 0, 255), thickness=-1)
               
            h = radius
            w = radius
            area = 3.14*radius*radius
            x_range = [int(x_center-radius), int(x_center+radius)]
            y_range = [int(y_center-radius), int(y_center+radius)]
            markRange = [x_range, y_range]
            angle = 0
        
            hValue, vValue, hValue_min, hValue_max = self.mark_detect_calc_h_v(img_HSV, y_center, x_center, h, w, img.shape)
            color, mark_Value = self.mark_detect_calc_color(hValue, vValue, hValue_min, hValue_max)
            meangray_dis, rectangularity, area_rate, w_rate, h_rate = self.mark_detect_calc_others(img, markRange, area, w, h, n, n2)
            x_center, y_center = x_center + xmin, y_center + ymin

            result.append([x_center, y_center, angle, radius, markRange, color, mark_Value, detect_step, meangray_dis, rectangularity, area_rate, w_rate, h_rate])
        
        # cv2.imshow('match template', img)
        # cv2.waitKey()
        # cv2.destroyAllWindows() 
        return result
                
    
    def mark_detect_blob_markOk(self, area, w, h, width_limit_pixel, height_limit_pixel, rectangularity_limit):    
        """
            ## 2024.5.23 判断矩形度的函数          
        """        
        # 面积判断    
        area_ok = self.area_limit[0] < area < self.area_limit[1]

        # 边长判断      
        w1 = width_limit_pixel[0]
        w2 = width_limit_pixel[1]
        h1 = height_limit_pixel[0]
        h2 = height_limit_pixel[1] 
        size_ok = (w1 < w < w2 and h1 < h < h2) or (w1 < h < w2 and h1 < w < h2)
        
        # 矩形度判断  
        # 综合判断，同时确定是哪种类型的色标，面积长宽矩形度色标类型判断 根据矩形度判断是否一致  
        rect_ok = 0
        r1 = rectangularity_limit[0]
        r2 = rectangularity_limit[1]
        if w > 0 and h > 0:
            rectangularity = area / (w * h)
            if r1 < rectangularity <= r2:
                rect_ok = 1  # 计算出的矩形度和设置的mark_type一致

        mark_ok = area_ok and size_ok and rect_ok

        return mark_ok

    def mark_detect_calc_h_v(self, img_HSV, y_center, x_center, h, w, shape):
        # 计算色标中心点   在色标中点的一定范围内搜索，HSV的h平均值
        # print(x_center, y_center)
        hValue = img_HSV[int(y_center), int(x_center)][0]
        hValue_Center = img_HSV[int(y_center), int(x_center)][0]
        hValues = []
        hValue_min = img_HSV[int(y_center), int(x_center)][0]
        hValue_max = img_HSV[int(y_center), int(x_center)][0]
        
        # 搜索HSV的v平均值            
        vValue = img_HSV[int(y_center), int(x_center)][2]
        vValues = []
        # int(max((y_center - h/4),0) 排除图片顶部
        # int(min((y_center + h/4 + 1),img.shape[0])) 排除图片底部
        for y_mark in range(int(max((y_center - h / 4), 0)), int(min((y_center + h / 4 + 1), shape[0]))):
            for x_mark in range(int(max((x_center - w / 4), 0)), int(min((x_center + w / 4 + 1), shape[1]))):
                if img_HSV[int(y_mark), int(x_mark)][0] > hValue_max:
                    hValue_max = img_HSV[int(y_mark), int(x_mark)][0]
                elif img_HSV[int(y_mark), int(x_mark)][0] < hValue_min:
                    hValue_min = img_HSV[int(y_mark), int(x_mark)][0]
                    
                hValues.append(img_HSV[int(y_mark), int(x_mark)][0])
                vValues.append(img_HSV[int(y_mark), int(x_mark)][2])
        hValues = sorted(hValues)
        hValue = hValues[int(len(hValues) / 2)]  # hValues排序后，取中点值

        vValues = sorted(vValues)
        vValue = vValues[int(len(vValues) / 2)]
        
        return hValue, vValue, hValue_min, hValue_max
        
    def mark_detect_calc_color(self, hValue, vValue, hValue_min, hValue_max):        
        color = 3  # 0-青色， 1-红， 2-黄， 3-黑
        mark_Value = int(hValue)
        if 5 < mark_Value < 30:
            color = 0  # 青
        elif 75 <= mark_Value <= 120:
            color = 2  # 黄
        elif 140 < mark_Value < 169:
            color = 1  # 红
        # elif 173 < mark_Value <= 180:
        #     color = 3 #黑

        if (vValue < 90) and ((int(hValue_min) < 6) or (int(hValue_max) > 173)):  # 和照片亮度相关！光圈，光照，曝光时间
            color = 3
                
        return color, mark_Value    
    
    def mark_detect_calc_others(self, img, range, area, w, h, n, n2):            
        cal_other_output = 0  # area 矩形度，灰度对比
        if cal_other_output:
            # 计算实际尺寸和标准尺寸的比值
            area_rate = area * n2 / self.__mark_area_pixel
            w_rate = w * n / self.__mark_width_pixel
            h_rate = h * n / self.__mark_height_pixel

            # 计算矩形度
            rectangularity = area / (w * h)

            # 计算色标与背景灰度差别
            xmin1, xmax1 = range[0][0], range[0][1]
            ymin1, ymax1 = range[1][0], range[1][1]
            xmin2 = max(xmin1 - (xmax1 - xmin1), 0)
            xmax2 = min(xmax1 + (xmax1 - xmin1), img.shape[1])
            ymin2 = max(ymin1 - (ymax1 - ymin1), 0)
            ymax2 = min(ymax1 + (ymax1 - ymin1), img.shape[0])
            meangray1 = img[ymin1:ymax1, xmin1:xmax1].mean()
            meangray2 = img[ymin2:ymax2, xmin2:xmax2].mean()
            meangray_dis = abs(meangray1 - meangray2)
        else:
            meangray_dis, rectangularity, area_rate, w_rate, h_rate = 0, 0, 0, 0, 0
            
        return meangray_dis, rectangularity, area_rate, w_rate, h_rate
    
    
    def mark_detect_calc_head_tail_pixel(self, result_pos):        
        # 计算首尾两个色标的像素距离 
        self.__head_tail_pixel = float(self.head_tail_distance_mm / self.field_of_view_X_mm) * self.resolution_X # 2048
        # 根据首尾标校准距离 result_pos[:, :2] = result_pos[:, :2] * self.head_tail_distance_mm / (result_pos[-1,0] - result_pos[0, 0])  #更换参数

        # 计算首尾标间距
        breakflag = 0
        head_tail = 0
        # print(self.__head_tail_pixel)
        for i in range(0, result_pos.shape[0] - 1):
            for j in range(i + 1, result_pos.shape[0]):
                if result_pos[i, 3] == result_pos[j, 3]:
                    self.__head_tail_x_pixel = abs(result_pos[j, 0] - result_pos[i, 0])
                    self.__head_tail_y_pixel = abs(result_pos[j, 1] - result_pos[i, 1])
                    head_tail = math.sqrt(self.__head_tail_x_pixel ** 2 + self.__head_tail_y_pixel ** 2)
                    if 0.8 * self.__head_tail_pixel < head_tail < 1.2 * self.__head_tail_pixel:
                        # print(head_tail, self.__head_tail_pixel)
                        if abs(j - i) == 4:  # 首尾间距，防止识别错颜色
                            self.__head_tail_pixel = head_tail
                            breakflag = 1
                            break
            if breakflag == 1:
                break
  
    def mark_detect_calc_next_roi(self, origin_img_height, result_pos):
        # 将色标在图中的大致位置记录下来，下次检测时先默认检测该大致位置内的图片
        if abs(result_pos[:, 1].max() - result_pos[:, 1].min()) < self.__mark_height_pixel * 2:
            MarkDetect.global_ROI_ymin = int(max(result_pos[:, 1].min() - self.__mark_height_pixel * 2, 0))
            MarkDetect.global_ROI_ymax = int(min(result_pos[:, 1].max() + self.__mark_height_pixel * 2, origin_img_height))
        
                
    def mark_detect_show_boundary(self, img_HSV):
        # 打印包含轮廓的图片用于测试
        a1 = self.area_limit[0]
        a2 = self.area_limit[1]
        for i, contour in enumerate(self.contours):
            cnt_area = cv2.contourArea(contour)
            if a1 < cnt_area < a2:
                # 在图片显示面积符合要求的图形
                print("No: {}, area test OK, Area: {}".format(i, cnt_area))
                cv2.drawContours(img_HSV, self.contours, i, (0, 255, 0), 3)  # 用绿色线条绘制面积范围内的轮廓
            else:
                print("No: {}, area test NOT OK, Area: {}".format(i, cnt_area))
                cv2.drawContours(img_HSV, self.contours, i, (0, 0, 255), 3)  # 用红色线条绘制面积范围外的轮廓
        cv2.imshow("Diamond detection", img_HSV)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def mark_detect_show_result(self, img, binary_img, result):
        binary_img_plt = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
        img_plt = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_plt2 = img_plt.copy()
        plt.figure(figsize=(23, 12))
        for x_center, y_center, cnt, rect, *o in result:
            cv2.drawContours(img_plt, [cnt], -1, (0, 0, 255), 2)
            cv2.drawContours(binary_img_plt, [cnt], -1, (0, 0, 255), 2)
            box = cv2.boxPoints(rect)  # 外接矩形的定点坐标
            box = np.intp(box)  # 坐标整型化
            cv2.drawContours(img_plt2, [box], -1, (0, 0, 255), 2)
        plt.subplot(2, 2, 1), plt.imshow(img_plt, 'gray'), plt.title("img_color")
        plt.subplot(2, 2, 2), plt.imshow(binary_img_plt, 'gray'), plt.title('binary_img')
        plt.subplot(2, 2, 3), plt.imshow(img_plt2, 'gray'), plt.title('binary_img')
        plt.show()
        # 等待用户按下键盘上的任意键
        plt.waitforbuttonpress()        

if __name__ == "__main__":
    config_path = 'config.yaml'

    mark_detect = MarkDetect(config_path)

    # 选择需要处理的图片
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    pic_path = filedialog.askopenfilename(initialdir=os.getcwd()+"mypic")  # 打开文件选择对话框
    
    # 计算开始时间
    start_time = time.time()

    ### 使用本地图片覆盖掉获取的图片，用于调试
    # TODO: 2024.5.23，测试Image__2024-04-24__ExposTime100ms.jpg，的红标会检测为黑色，需要矫正
    img_default = cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), 1)

    detection_success, result_pos = mark_detect.mark_detect_main(img_default, mark_detect.mark_detect_template)

    # 计算结束时间
    end_time = time.time()

    print(result_pos)  
        
    execution_time = (end_time - start_time) * 1000
    print("程序运行时间：", execution_time, "毫秒")

    """
    2024.5.23 检测结果打印
    """
    if detection_success:
        print("检测成功")
    else:
        print("检测失败")

    marks_typ = ["菱形", "圆形"]
    colors_typ = ["青", "红", "黄", "黑"]

    for i in range(result_pos.shape[0]):
        """
        0, 1, 2, 3, 4
        x_center, y_center, rect[2], color, mark_Value

        0：菱形标，1：圆形标

        0-青色， 1-红， 2-黄， 3-黑
        """

        xpos = result_pos[i, 0]
        ypos = result_pos[i, 1]
        angle = result_pos[i, 2]
        color_iter = int(result_pos[i, 4])
        mark_iter = int(result_pos[i, 5])
        detect_step = result_pos[i, 6]

        print("======第 {} 个检测结果======".format(i + 1))
        print("x坐标: {:.2f}, y坐标: {:.2f}, 角度： {:.2f}, detect step: {}".format(xpos, ypos, angle, detect_step))
        print("颜色： {}".format(colors_typ[color_iter]))
        # print("颜色类型： {}, 图标类型 {}".format(colors_typ[color_iter], marks_typ[mark_iter]))

    """
    2024.5.23 将结果打印到图片上
    """
    for row in result_pos:
        # print("No: {}, Area: {}".format(i, cnt_area))
        point = (int(row[0]), int(row[1]))
        cv2.circle(img_default, point, radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(img_default, point, radius=int(row[3]), color=(255, 0, 0), thickness=1)
    cv2.imshow('binary_img', img_default)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
