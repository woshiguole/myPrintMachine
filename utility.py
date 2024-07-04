# -*- coding: utf-8 -*-
"""
文件描述：工具性程序
作者：Wenjie Cui
创建日期：2024.5.28
最后修改日期：2024.5.28
"""

import threading
import mmap
import base64
import cv2

def num2register(num):  # 一个数字转换为modbustcp中的两个寄存器[低16位，高16位]
    return [num & 65535, (num >> 16) & 65535]


def register2num(register0, register1):  # modbustcp中的两个寄存器转换为一个数字[低16位，高16位]
    return register0 + (register1 << 16)


def image_to_base64(image_np):
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]

    return image_code


def run_in_thread(func):
    def wrapper(*args, **kwargs):
        event = threading.Event()
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread, event

    return wrapper


def stop_thread(thread: threading.Thread, event: threading.Event):
    if thread.is_alive():
        event.clear()
        thread.join()


def save_image_to_shared_memory(image_data, shared_memory_size):
    # with open(image_path, 'rb') as image_file:
    #     image_data = image_file.read()

    encoded_image = base64.b64encode(image_data)

    with mmap.mmap(-1, shared_memory_size, tagname="shared_image") as shared_image:
        shared_image.write(encoded_image)
