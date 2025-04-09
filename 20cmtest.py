#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
import time

# Pipeline setup 
pipeline = dai.Pipeline()
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")

#Kamera Konfiguration
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Stereo konfiguration
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(False)

# Links 
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)

last_print_time = time.time()
COLORMAP_CHOICE = 'TURBO'  

with dai.Device(pipeline) as device:
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        inDepth = qDepth.get()
        depthFrame = inDepth.getFrame()
        depth_cm = depthFrame / 10.0 #millimmeter in centimeter

        h, w = depth_cm.shape
        cy, cx = h // 2, w // 2
        region_size = 80     
        lx = w // 4

        # Regionen festlegen
        center_region = depth_cm[cy-region_size:cy+region_size, cx-region_size:cx+region_size]
        left_region = depth_cm[cy-region_size:cy+region_size, lx-region_size:lx+region_size]

        
        total_center = center_region.size  # Alle pixel in der region
        total_left = left_region.size     

        #  werte kleiner 1 ignorieren = invalid
        center_valid = center_region[center_region > 1.0]
        left_valid = left_region[left_region > 1.0]

        # Nahesten pixel berechnen
        center_min = np.min(center_valid) if center_valid.size > 0 else 0.0
        left_min = np.min(left_valid) if left_valid.size > 0 else 0.0

        center_valid_ratio = center_valid.size / total_center if total_center > 0 else 0 #prozentsatz der validen pixel in der region
        left_valid_ratio = left_valid.size / total_left if total_left > 0 else 0

        # Abstand console ausgeben
        current_time = time.time()
        if current_time - last_print_time > 1.0:
            print(f"Center: {center_min:.2f}cm | Valid: {center_valid.size}/{total_center} ({center_valid_ratio:.1%})")
            print(f"Left: {left_min:.2f}cm | Valid: {left_valid.size}/{total_left} ({left_valid_ratio:.1%})")
            last_print_time = current_time
            
        # werte normalisieren, colormappen
        depth_norm = cv2.normalize(depthFrame, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = np.uint8(depth_norm)
        depth_color = cv2.applyColorMap(depth_norm, getattr(cv2, f'COLORMAP_{COLORMAP_CHOICE}'))

        # Regionen sichtbar machen
        cv2.rectangle(depth_color, (cx-region_size, cy-region_size), 
                     (cx+region_size, cy+region_size), (0, 0, 255), 2)
        cv2.rectangle(depth_color, (lx-region_size, cy-region_size), 
                     (lx+region_size, cy+region_size), (0, 0, 255), 2)

        # text overlays
        text_y = cy - region_size - 10
        cv2.putText(depth_color, 
                    f"Center Valid: {center_valid.size}/{total_center} ({center_valid_ratio:.1%})",
                    (cx - 180, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        cv2.putText(depth_color, 
                    f"Left Valid: {left_valid.size}/{total_left} ({left_valid_ratio:.1%})",
                    (lx - 160, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        #ausgabe fuer zu nahe objekte
        if center_min < 50:
            cv2.putText(depth_color, "Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Depth", depth_color)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()