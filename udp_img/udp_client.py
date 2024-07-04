import socket
import cv2
import numpy as np
import struct


class ImageSender:
    def __init__(self, server_ip='127.0.0.1', server_port=9999):
        self.server_address = (server_ip, server_port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 设置超时时间为3秒
        self.socket.settimeout(3)

    def send_image(self, img):
        img_encode = cv2.imencode('.jpg', img)[1]
        data_encode = np.array(img_encode)
        data = data_encode.tostring()

        fhead = struct.pack('l', len(data))
        self.socket.sendto(fhead, self.server_address)

        # 将图片字节流拆解为1024大小的片，逐个发送
        for i in range(len(data) // 1024 + 1):
            if 1024 * (i + 1) > len(data):
                self.socket.sendto(data[1024 * i:], self.server_address)
            else:
                self.socket.sendto(data[1024 * i:1024 * (i + 1)], self.server_address)

        try:
            response = self.socket.recv(1024).decode('utf-8')
        except socket.timeout:
            response = 'timeout'

        return response

    def close(self):
        self.socket.close()


# 使用示例
if __name__ == '__main__':
    sender = ImageSender()
    response = sender.send_image('./Image__2024-04-24__ExposTime50ms.jpg')
    print(response)
    sender.close()
