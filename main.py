# -*- coding: utf-8 -*-
"""
文件描述：主程序
作者：Wenjie Cui
创建日期：2024.5.28
最后修改日期：2024.6.4
"""

from printing_blob import MarkDetect
from printing_modbus import PaintMachineModbusServer
from global_data import GlobalData
from udp_img.udp_client import ImageSender
from pylon_camera import ImageAcquistionAndDetect
from utility import *

import threading
import time
import yaml
import cv2
import numpy as np
from array import array


def get_img(ix: int) -> None:
    # globalData.img_list[ix] = cv2.imdecode(np.fromfile("mypic/Image__2024-04-24__ExposTime50ms.jpg", dtype=np.uint8), 1)
    try:
        globalData.img_list[ix] = image_acquistion.get_RGB_img(ix=ix)
    except:
        pass
        # globalData.img_list[ix] = cv2.imdecode(
        #     np.fromfile("mypic/Image__2024-04-24__ExposTime50ms.jpg", dtype=np.uint8), 1)


def get_para(ix: int):
    # 从相应的modbus存储区，更新PrintMachine的参数
    data = print_modbus.readPrintMachinePara(camera_ix=ix)
    # [mark_shape, mark_size, size_limit_min, size_limit_max, exposure_time, head_tail_distance_mm,
    #  image_save, img_view, camera_run_type, reconnectCmd, C1, C2]
    para = array('i', [data[5], data[-2], data[-1]])  # head_tail_distance_mm, C1, C2
    return para


def pre_detect(ix: int):
    get_img(ix=ix)
    para = get_para(ix=ix)
    return para


def post_detect(result_pos, ix: int):
    # 将计算后的结果存入到相应的modbus存储区
    print_modbus.savePrintMachineData(result_pos=result_pos, ix=ix)


@run_in_thread
def run(ix: int):
    mark_detector = MarkDetect(config_path)
    interval = 0.5  # 设置循环时间

    while True:
        start_time = time.time()

        # 获取图片，从modbus的4X参数区读取参数
        para = pre_detect(ix=ix)
        # 将modbus获取的参数更新到mark_detector中
        mark_detector.update_para(para=para)
        # 将图片输入到mark_detector中，获取坐标
        detection_success, result_pos = mark_detector.mark_detect_main(img=globalData.img_list[ix], detect_func=mark_detector.mark_detect_blob)
        # 将mark_detector获取的数据存储到modbus的3X数据区
        post_detect(result_pos=result_pos, ix=ix)

        task_time = time.time() - start_time

        wait_time = interval - task_time
        print("ix={}, wait_time={:.2f}ms, detect_num={}, detection_success={}".format(ix, wait_time * 1000, result_pos.shape[0], detection_success))

        if globalData.stop_print_machine == 2:
            # print("try to break！")
            break

        if wait_time > 0:
            time.sleep(wait_time)


@run_in_thread
def run_udp_sender(ix: int):
    sender = ImageSender(server_port=9999)
    interval = 1  # 设置循环时间
    while True:
        start_time = time.time()

        response = sender.send_image(globalData.img_list[ix])
        print(response)
        # sender.close()

        task_time = time.time() - start_time
        wait_time = interval - task_time

        if globalData.stop_print_machine == 2:
            break

        if wait_time > 0:
            time.sleep(wait_time)
    sender.close()


@run_in_thread
def run_show_image(ix: int):
    scaling = 3
    while True:
        winname = 'Image IX:' + str(ix)
        img = globalData.img_list[ix][0:-1:scaling, 0:-1:scaling, :]
        cv2.imshow(winname=winname, mat=img)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if globalData.stop_print_machine == 2:
            break

        if key == 27:  # Check if ESC key is pressed
            globalData.stop_print_machine = 2
            break


if __name__ == "__main__":
    config_path = 'config.yaml'

    globalData = GlobalData(config_path)
    # modbus会启动一个线程作为server，处理相应请求
    print_modbus = PaintMachineModbusServer(globalData.modbus_setting)
    image_acquistion = ImageAcquistionAndDetect(globalData.camera_setting)

    # 2个相机，分2个线程获取相应数据
    camera_thread_list = []
    camera_event_list = []
    for num in range(globalData.camera_setting.camera_num):
        camera_thread_ix, camera_event_ix = run(ix=num)
        camera_thread_list.append(camera_thread_ix)
        camera_event_list.append(camera_event_ix)

    # udp_thread_ix0, udp_event_ix0 = run_udp_sender(ix=0)
    show_image_thread_ix0, show_image_event_ix0 = run_show_image(ix=0)

    # 读取modbus的reconnectCmd标志
    interval = 1  # 设置循环时间
    while True:
        data = print_modbus.server_databank.get_values(block_name='0', address=8, size=1)
        globalData.stop_print_machine = data[0]  # reconnectCmd

        time.sleep(interval)

        if globalData.stop_print_machine == 2:
            break

    if globalData.stop_print_machine:
        # print("start stop thread！")
        for num in range(globalData.camera_setting.camera_num):
            print("Stop camera {}!".format(num))
            stop_thread(camera_thread_list[num], camera_event_list[num])

        # stop_thread(udp_thread_ix0, udp_event_ix0)
        stop_thread(show_image_thread_ix0, show_image_event_ix0)
        print_modbus.tcp_server.stop()

    print("main thread over")
