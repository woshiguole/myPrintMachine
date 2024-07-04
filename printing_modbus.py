# -*- coding: utf-8 -*-
"""
文件描述：用于色标检测的Modbus Slave通信
作者：Wenjie Cui，Dr. Zhu
创建日期：2024.5.27
最后修改日期：2024.5.28
"""

import modbus_tk.defines as cst
from modbus_tk import modbus_tcp
import time
import numpy as np

from global_data import ModbusSetting
from utility import *


class PaintMachineModbusServer:
    def __init__(self, setting: ModbusSetting):
        self.tcp_server = None
        self.setting = setting
        self.server_databank = self.modbusInit()

    def modbusInit(self):
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
        self.tcp_server = modbus_tcp.TcpServer(address=self.setting.ip_str, port=self.setting.port)  # address必须设置,port默认为502
        self.tcp_server.start()
        # 创建从站
        slave_1 = self.tcp_server.add_slave(1)  # slave_id = 1
        # 为从站添加存储
        # 保持寄存器（Holding Registers）40001 - 40017
        slave_1.add_block(block_name='0', block_type=cst.HOLDING_REGISTERS, starting_address=0, size=17)
        # 输入寄存器（Input Registers）30101 - 30173
        slave_1.add_block(block_name='1', block_type=cst.ANALOG_INPUTS, starting_address=100, size=74)
        # print("Modbus tcp running...")

        return slave_1

    def readPrintMachinePara(self, camera_ix):
        """
        主 -> 从
        读取保持寄存器（Holding Registers）存储设备的参数
        获取modbus的寄存器地址40001 - 40017的数据

        40001	色标形状	uint
        40002	色标尺寸	uint
        40003	曝光时间1	uint
        40004	曝光时间2	uint
        40005	色标首尾间距	uint
        40006	未应用 图片是否保存	uint
        40007	未应用 图片是否在线显示	uint
        40008	trigger模式	uint
        40009	未应用 重连命令	uint
        40010	未应用 色标颜色	uint
        40011	未应用	uint
        40012	未应用	uint
        40013	未应用	uint
        40014	未应用 镜头距离纸张距离	uint
        40015	未应用 镜头距离纸张距离	uint
        40016	滤波参数C1	uint
        40017	滤波参数C2	uint

        参数:
        camera_ix (int): 使用的camera id

        返回:
        mark_shape, mark_size, size_limit_min, size_limit_max, exposure_time, head_tail_distance_mm,
        image_save, img_view, camera_run_type, reconnectCmd, C1, C2
        """
        modbusdata = self.server_databank.get_values(block_name='0', address=0, size=17)  # 更换参数
        mark_shape = modbusdata[0]
        mark_size = modbusdata[1] * 0.001  # um->mm

        size_limit_min = 0.6
        size_limit_max = 1.2
        # 根据相机索引号选择对应的数据
        exposure_time = 0
        if camera_ix == 0:
            exposure_time = max(modbusdata[2], 30)  # standard 26以上限制
        elif camera_ix == 1:
            exposure_time = max(modbusdata[3], 30)
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

        exposure_time = max(10, exposure_time)
        exposure_time = min(exposure_time, 100)  # ultrashort 26以下
        head_tail_distance_mm = max(10, head_tail_distance_mm)
        head_tail_distance_mm = min(40, head_tail_distance_mm)
        if image_save not in [0, 1]:
            image_save = 0
        if img_view not in [0, 1]:
            img_view = 0
        if camera_run_type not in [0, 1]:
            camera_run_type = 0
        if reconnectCmd not in [0, 1, 2]:
            reconnectCmd = 0

        return [mark_shape, mark_size, size_limit_min, size_limit_max, exposure_time, head_tail_distance_mm,
                image_save, img_view, camera_run_type, reconnectCmd, C1, C2]

    def savePrintMachineData(self, result_pos, ix):
        """
        从 -> 主
        输入寄存器（Input Registers）存储设备的状态信息
        保存modbus的寄存器地址30101 - 30173的数据

        相机1	相机2
        30101	30102	心跳	uint
        30103/30104	30105/30106	相机状态位	udint
        30107	30108	trigger标志位	uint
        30109	30110	图片处理标志位	uint
        30111/30112	30113/30114	检测成功计数	udint
        30115	30116	检测的色标数目	uint
        30117/30118	30142/30143	色标1的x值	udint
        30119/30121	30144/30145	色标2的x值	udint
        30121/30122	30146/30147	色标3的x值	udint
        30123/30124	30148/30149	色标4的x值	udint
        301325/30126	30150/30151	色标5的x值	udint
        30127/30128	30152/30153	色标1的y值	udint
        30129/30130	30154/30155	色标2的y值	udint
        30131/30132	30156/30157	色标3的y值	udint
        30133/30134	30158/30159	色标4的y值	udint
        30135/30136	30160/30161	色标5的y值	udint
        30137	30162	色标1的颜色	uint
        30138	30163	色标2的颜色	uint
        30139	30164	色标3的颜色	uint
        30140	30165	色标4的颜色	uint
        30141	30166	色标5的颜色	uint
        30167	30169	计算出的图片横向距离	udint
        30171	30173	拍照成功计数	udint

        参数:
        result_pos  : 获取n个测试点的位置
        camera_ix (int): 使用的camera id

        返回:
        values :
        """
        values = []
        mark_num = min(result_pos.shape[0], 5)
        # 相机1，30115；相机2，30116。检测色标数目
        self.server_databank.set_values(block_name='1', address=100 + ix + 14, values=[mark_num])
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

        # 相机1，30117-30141；相机2，30142-30166。检测结果，x,y,颜色。
        self.server_databank.set_values(block_name='1', address=100 + ix * 25 + 16, values=values)
        # print(self.server_databank.get_values(block_name='1', address=100, size=74))
        return values


if __name__ == "__main__":
    from global_data import GlobalData

    config_path = 'config.yaml'
    global_settings = GlobalData(config_path)

    print_modbus = PaintMachineModbusServer(global_settings.modbus_setting)

    # 测试用数据
    result_pos = [
        [5.47750008e+02, 4.00249996e+02, 4.50000000e+01, 3.00000000e+00, 8.40000000e+01, 1.00000000e+00, 0.00000000e+00,
         0.00000000e+00],
        [7.04250008e+02, 3.89250004e+02, 4.50000000e+01, 3.00000000e+00, 1.40000000e+02, 1.00000000e+00, 0.00000000e+00,
         0.00000000e+00],
        [8.64500008e+02, 3.91500004e+02, 4.50000000e+01, 2.00000000e+00, 9.30000000e+01, 1.00000000e+00, 0.00000000e+00,
         0.00000000e+00],
        [1.03181693e+03, 3.88795391e+02, 4.81798325e+01, 0.00000000e+00, 1.60000000e+01, 1.00000000e+00, 0.00000000e+00,
         0.00000000e+00],
        [1.18750001e+03, 3.88500000e+02, 4.50000000e+01, 3.00000000e+00, 3.00000000e+01, 1.00000000e+00, 0.00000000e+00,
         0.00000000e+00]]

    result_pos = np.array(result_pos)

    print_modbus.savePrintMachineData(result_pos=result_pos, ix=0)
    i = 0
    while True:
        data = print_modbus.readPrintMachinePara(0)
        i = i + 1
        time.sleep(1)
        print("{}, 数据: {}".format(i, data))
