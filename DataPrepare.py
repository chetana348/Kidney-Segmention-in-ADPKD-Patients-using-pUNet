from __future__ import print_function, division
import os
import cv2
import numpy as np


def prepare2ddata(srcpath, maskpath, trainImage, trainMask, number, height, width):
    for i in range(0, number, 1):
        index = 0
        listsrc = []
        listmask = []
        for _ in os.listdir(srcpath + str(i)):
            image = cv2.imread(srcpath + str(i) + "/" + str(index) + ".nii", cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (width, height))
            label = cv2.imread(maskpath + str(i) + "/" + str(index) + ".nii", cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, (width, height))
            listsrc.append(image)
            listmask.append(label)
            index += 1

        imagearray = np.array(listsrc)
        imagearray = np.reshape(imagearray, (index, height, width))
        maskarray = np.array(listmask)
        maskarray = np.reshape(maskarray, (index, height, width))
        srcimg = np.clip(imagearray, 0, 255).astype('uint8')
        for j in range(index):
            if np.max(maskarray[j]) == 255:
                cv2.imwrite(trainImage + "\\" + str(i) + "_" + str(j) + ".nii", srcimg[j])
                cv2.imwrite(trainMask + "\\" + str(i) + "_" + str(j) + ".nii", maskarray[j])


def preparetumortrain2ddata():
    height = 512
    width = 512
    srcpath = r"E:\Chetana\data\Image\\"
    maskpath = r"E:\Chetana\data\Image\Mask\\"
    trainImage = r"E:\Chetana\data\Image\train\\"
    trainMask = r"E:\Chetana\data\Image\Mask\train\\"
    prepare2ddata(srcpath=srcpath, maskpath=maskpath, trainImage=trainImage, trainMask=trainMask, number=210,
                  height=height, width=width)


preparetumortrain2ddata()
