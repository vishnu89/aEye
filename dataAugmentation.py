'''
This is an example for data augmentation where
 input will be list of image paths (not image)
 output will be yield function with augmented image
'''

from glob import glob
import numpy as np
import cv2


def genAugmentation(img, imgScaleFactor, rotationFactor):
    # scale the image
    rows, cols = img.shape[:2]
    rows, cols = rows * imgScaleFactor, cols * imgScaleFactor
    rows, cols = int(rows), int(cols)
    resizedImg = cv2.resize(img, (cols, rows))
    # rotate image
    # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    # yield cv2.warpAffine(img, M, (cols, rows))
    resizedRotatedImg = cv2.rotate(resizedImg, rotationFactor)
    return resizedRotatedImg


def readImages(impaths, batchSize, nGenImages):
    imgScaleFactors = [.5, .25, 1, 1.5, 2]
    imgRotationFactors = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
    imgs = []
    while nGenImages:
        img = cv2.imread(np.random.choice(impaths))
        if img is None:
            # img is empty then go for next image
            pass
        else:
            imgs.append(genAugmentation(img, np.random.choice(imgScaleFactors), np.random.choice(imgRotationFactors)))
        if len(imgs) == batchSize:
            yield imgs
            nGenImages -= 1
            imgs = [] # empty all imgs


def main(impaths):
    for imgs in readImages(impaths, 3, nGenImages=50):
        for ix, img in enumerate(imgs):
            cv2.namedWindow('img_%s' % ix, 0)
            cv2.imshow('img_%s' % ix, img)
        cv2.waitKey(0)


if __name__ == '__main__':
    impaths = glob(r'C:\Users\Public\Pictures\Sample Pictures\*.*')
    main(impaths)

