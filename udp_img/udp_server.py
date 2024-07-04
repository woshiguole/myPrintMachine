import socket
import cv2
import numpy as np
import struct


class ImageReceiver:
    def __init__(self, server_ip='127.0.0.1', server_port=9999):
        self.server_address = (server_ip, server_port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(self.server_address)
        print('Bind UDP on {}...'.format(server_port))

    def receive_image(self):
        fhead_size = struct.calcsize('l')

        while True:
            buf, addr = self.socket.recvfrom(1024)
            # 首先接收元组，元组包含接下来的数据大小
            if buf:
                # 这里结果是一个元组，所以把值取出来
                data_size = struct.unpack('l', buf)[0]
                break


        # 接收元组头后，依次接收后续的拆解小包
        recvd_size = 0
        data_total = b''
        while not recvd_size == data_size:
            if data_size - recvd_size > 1024:
                data, addr = self.socket.recvfrom(1024)
                recvd_size += len(data)
            else:
                data, addr = self.socket.recvfrom(1024)
                recvd_size = data_size
            data_total += data

        nparr = np.frombuffer(data_total, np.uint8)
        img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # cv2.imshow('Received Image', img_decode)
        # cv2.waitKey(0)

        reply = "Message received!"
        self.socket.sendto(reply.encode('utf-8'), addr)
        return data_total

    def close(self):
        self.socket.close()


# 使用示例
if __name__ == '__main__':
    receiver = ImageReceiver()
    receiver.receive_image()
    receiver.close()
