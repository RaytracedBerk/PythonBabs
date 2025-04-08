import depthai as dai
import cv2
import numpy as np

pipeline = dai.Pipeline()
colorCam = pipeline.createColorCamera()

colorCam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setPreviewSize(1920, 1080)
colorCam.setInterleaved(False)

# Create output
xout = pipeline.createXLinkOut()
xout.setStreamName("color")
# Link camera preview to output
colorCam.preview.link(xout.input)

with dai.Device(pipeline) as device:
    colorQueue = device.getOutputQueue(name = "color",maxSize=4,blocking=False)

    while True:
        frame = colorQueue.get().getCvFrame()  # ✅ Use getCvFrame()
        cv2.imshow("Color Preview", frame)

        if cv2.waitKey(1) == ord('q'):
            break
