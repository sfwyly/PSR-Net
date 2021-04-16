
"""
    @Author: Junjie Jin
    @Code: Junjie Jin

"""

import cv2
import numpy as np

def getHoles(image_shape ,num):

    imageHeight ,imageWidth = image_shape[0] ,image_shape[1]

    maxVertex = 20
    maxAngle = 30
    maxLength = 100
    maxBrushWidth = 20


    result = []

    for _ in range(num):

        mask = np.ones((imageHeight ,imageWidth) ,dtype = np.float32)
        numVertex =1 + np.random.randint(maxVertex)

        for i in range(numVertex):

            startX = np.random.randint(imageHeight // 4, imageHeight // 4 * 3)
            startY = np.random.randint(imageWidth // 4, imageWidth // 4 * 3)

            for j in range(1 + np.random.randint(5)):

                angle = 0.01 + np.random.randint(maxAngle)
                if (i % 2 == 0):
                    angle = 2 * np.pi - angle
                length = 5 + np.random.randint(maxLength)
                brushWidth = 5 + np.random.randint(maxBrushWidth)

                endX = (startX + length * np.sin(angle)).astype(np.int32)
                endY = (startY + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (startY, startX), (endY, endX), 0.0, brushWidth)

                startX, startY = endX, endY
        result.append(mask)

    return np.array(result)[...,np.newaxis]#batch img_size img_size 1