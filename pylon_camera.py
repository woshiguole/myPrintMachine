# -*- coding: utf-8 -*-

from pypylon import pylon
from pypylon import genicam

import cv2
import time

from global_data import CameraSetting


class ImageAcquistionAndDetect:
    def __init__(self, setting: CameraSetting):
        self.cameras = []
        self.converter = []
        self.IpList = []

        self.camera_num = setting.camera_num
        self.camera_run_type = setting.camera_run_type  # 0：硬触发运行，1：自由运
        self.exposure_time = setting.exposure_time
        self.resolution_X = setting.resolution_X
        self.resolution_Y = setting.resolution_Y
        self.camera_init()



    def camera_init(self):
        tl_factory = pylon.TlFactory.GetInstance()  #遍历相机
        print("Waiting for camera connecting...")
        # 2024.6.11 增加等待相机连接段
        devices = tl_factory.EnumerateDevices()
        need_devices_num = self.camera_num - len(devices)
        while True:
            if need_devices_num > 0:
                print("Still need {} camera!".format(need_devices_num))
            elif need_devices_num < 0:
                print("Still need {} camera!".format(-need_devices_num))
            else:
                print("All camera standby!")
                break
            # 每隔1s重新获取一次设备
            time.sleep(1)
            devices = tl_factory.EnumerateDevices()
            need_devices_num = self.camera_num - len(devices)

        for ix, dev_info in enumerate(devices):
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
                    self.cameras.append(camera)
                except:
                    continue
                else:
                    break

        # converting to opencv bgr format
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


    def camera_setting(self, camera):
        # 曝光参数设置
        camera.ExposureMode.SetValue("Timed")
        camera.ExposureTimeMode.SetValue("Standard")
        ### 设置曝光时间
        camera.ExposureTimeAbs.SetValue(self.exposure_time)

        # AOI设置
        if camera.Width.GetValue() > 1500:  # 大相机
            camera.Width.SetValue(self.resolution_X)
            camera.Height.SetValue(self.resolution_Y)
            camera.OffsetX.SetValue(0)
            camera.OffsetY.SetValue(300)

        # 设定MaxNumBuffer
        camera.MaxNumBuffer = 2

        if self.camera_run_type == 0:
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

    def get_RGB_img(self, ix: int):
        grabResult = self.cameras[ix].RetrieveResult(self.exposure_time, pylon.TimeoutHandling_ThrowException)
        image = self.converter.Convert(grabResult)
        img_decode = image.GetArray()
        return img_decode



if __name__ == "__main__":

    from global_data import GlobalData

    config_path = 'config.yaml'
    global_settings = GlobalData(config_path)

    imageAcquistion = ImageAcquistionAndDetect(global_settings.camera_setting)

    while True:
        img_decode = imageAcquistion.get_RGB_img(ix=0)

        cv2.imshow('Received Image', img_decode)
        key = cv2.waitKey(0)

        cv2.destroyAllWindows()
        if key == 27:  # Check if ESC key is pressed
            break

