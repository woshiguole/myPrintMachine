# -*- coding: utf-8 -*-
import yaml
import numpy as np
from typing import List


class GlobalData:
    def __init__(self, yaml_path: str):
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        self.modbus_setting = ModbusSetting()
        self.camera_setting = CameraSetting()
        self.blob_setting = BlobSetting()

        # set modbus para
        self.modbus_setting.ip_str = data['modbus_setting']['ip_str']
        self.modbus_setting.port = data['modbus_setting']['port']

        # set camera para
        self.camera_setting.camera_num = data['camera_setting']['camera_num']
        self.camera_setting.camera_run_type = data['camera_setting']['camera_run_type']
        self.camera_setting.exposure_time = data['camera_setting']['exposure_time']
        self.camera_setting.field_of_view_X_mm = data['camera_setting']['field_of_view_X_mm']
        self.camera_setting.head_tail_distance_mm = data['camera_setting']['head_tail_distance_mm']
        self.camera_setting.resolution_X = data['camera_setting']['resolution_X']
        self.camera_setting.resolution_Y = data['camera_setting']['resolution_Y']

        # set blob algo para
        self.blob_setting.mark_num = data['blob_setting']['mark_num']
        self.blob_setting.mark_type = data['blob_setting']['mark_type']
        self.blob_setting.mark_width = data['blob_setting']['mark_width']
        self.blob_setting.mark_height = data['blob_setting']['mark_height']
        self.blob_setting.rectangularity = data['blob_setting']['rectangularity']
        self.blob_setting.limit = data['blob_setting']['limit']
        self.blob_setting.scaling = data['blob_setting']['scaling']
        self.blob_setting.blur_kernel = data['blob_setting']['blur_kernel']
        self.blob_setting.adaptive_block = data['blob_setting']['adaptive_block']
        self.blob_setting.C = data['blob_setting']['C']
        self.blob_setting.C1 = data['blob_setting']['C1']
        self.blob_setting.C2 = data['blob_setting']['C2']

        self.stop_print_machine = False
        # 创建一个固定尺寸的全零图像
        height, width, channels = 800, 2048, 3
        self.img_list = [np.zeros((height, width, channels), dtype=np.uint8),
                         np.zeros((height, width, channels), dtype=np.uint8)]


class ModbusSetting:
    ip_str: str
    port: int


class CameraSetting:
    camera_num: int
    camera_run_type: int
    exposure_time: int
    field_of_view_X_mm: int  # 长度 落地
    head_tail_distance_mm: int
    resolution_X: int
    resolution_Y: int


class BlobSetting:
    mark_num: int
    mark_type: int  # 0：菱形标，1：圆形标
    mark_width: List  # 菱形对角线 圆形直径
    mark_height: List
    rectangularity: List  # PI/4 = 0.785
    limit: List  # [0.7, 1.2]，筛选面积和长宽的最大最小倍数范围

    scaling: int  # 3，先缩小3倍粗检测，时间花费和精度都比较合适
    blur_kernel: int  # 9，处理之前高斯滤波窗口大小，9比较合适
    adaptive_block: int  # 67，自适应二值化的窗口大小，对于目前测试的大小矩形标和三角标都较为合适
    C: int  # 12，自适应二值化的阈值偏差, 光亮直接相关！！！！！
    C1: int
    C2: int

