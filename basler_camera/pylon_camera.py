# -*- coding: utf-8 -*-

from pypylon import pylon
from pypylon import genicam

import cv2

IpListOld = []
cameras = []
converter = []
camera_run_type = 1  # 0：硬触发运行，1：自由运



class ImageAcquistionAndDetect():
    def __init__(self):
        self.camera_init()

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
        camera.ExposureTimeAbs.SetValue(10000)

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


if __name__ == "__main__":
    imageAcquistion = ImageAcquistionAndDetect()

    while True:
        grabResult = cameras[0].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        image = converter.Convert(grabResult)
        img_decode = image.GetArray()

        cv2.imshow('Received Image', img_decode)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

