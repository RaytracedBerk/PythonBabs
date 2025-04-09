#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
import time


# pipeline erstellen
pipeline = dai.Pipeline()

# nodes fuer graustufen kameras
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

# node fuer depth
stereo = pipeline.create(dai.node.StereoDepth)

# output fuer tiefe
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")

# Configure the mono cameras
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Configure the stereo depth node
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(False)

# Link the nodes
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)

last_print_time = time.time()

# Start the pipeline
with dai.Device(pipeline) as device:
    # Get the depth output queue
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        inDepth = qDepth.get()  # Blocking call: wait for a new frame
        # Get the depth frame in millimeters (DepthAI depth output returns float32 mm values)
        depthFrame = inDepth.getFrame()  # 2D numpy array of depth values in millimeters

        # Convert depth to centimeters:
        depth_cm = depthFrame / 10.0

  
        h, w = depth_cm.shape
        cy, cx = h // 2, w // 2
        region_size = 80
        # For center region
  
        lx = w //4

        center_region = depth_cm[cy - region_size:cy + region_size, cx - region_size:cx + region_size]
        left_region = depth_cm[cy - region_size:cy + region_size, (w // 4) - region_size:(w // 4) + region_size]

        # Mask out zero (invalid) values
        center_valid = center_region[center_region > 1.0]
        left_valid = left_region[left_region > 1.0]

        center_min = np.min(center_valid) if center_valid.size > 0 else 0.0
        left_min = np.min(left_valid) if left_valid.size > 0 else 0.0


        current_time = time.time()
        if current_time - last_print_time > 1.0:
            print(f"Center region depth: {center_min:.2f} cm")
            print(f"Left region depth: {left_min:.2f} cm")
            last_print_time = current_time
        # Normalize depth for display purposes
        depth_norm = cv2.normalize(depthFrame, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = np.uint8(depth_norm)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)

        cv2.rectangle(depth_color, (cx - region_size, cy - region_size), (cx + region_size, cy + region_size), (0, 0, 255), 2)
        cv2.rectangle(depth_color, (lx - region_size, cy - region_size), (lx + region_size, cy + region_size), (0, 0, 255), 2)
        cv2.setWindowTitle("Depth", f"Depth View - Region Size: {region_size}")

        cv2.putText(depth_color, f"Valid: {center_valid.size}", (cx - 90, cy - region_size - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # If the center pixel is closer than 20cm, overlay a message
        if center_min< 50:
            cv2.putText(depth_color, "Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
          

        cv2.imshow("Depth", depth_color)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
