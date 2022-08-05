#!/usr/bin/python
# -*- coding: utf-8 -*-
import albumentations as A
import imageio
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import random


class imageTransforms:

    def __init__(self, img, index):
        self.count_img = index
        self.initialize_image(img)
        self.create_dataset()

    def initialize_image(self, input_image):
        self.img = input_image
        self.grey_img = image = cv2.cvtColor(input_image,
                cv2.COLOR_BGR2RGB)
        (self.H, self.W, self.L) = input_image.shape

    # Flip Left

    def flip_left(self):
        img_clockwise_90 = cv2.flip(self.img, 0)
        cv2.imwrite('flip_left_' + str(self.count_img) + '.png',
                    img_clockwise_90)
        return img_clockwise_90

    # Flip top

    def flip_top(self):
        img_flip = cv2.flip(self.img, -1)
        cv2.imwrite('flip_top_' + str(self.count_img) + '.png',
                    img_flip)
        return img_flip

    # Flip Right

    def flip_right(self):
        img_anticlock_90 = cv2.flip(self.img, 1)
        cv2.imwrite('flip_right_' + str(self.count_img) + '.png',
                    img_anticlock_90)
        return img_anticlock_90

    # Centered Shear

    def shear_centered(self):
        M2 = np.float32([[1, 0, 0], [0.2, 1, 0]])
        M2[0, 2] = -M2[0, 1] * self.W / 2
        M2[1, 2] = -M2[1, 0] * self.H / 2
        centered_sheared = cv2.warpAffine(img, M2, (self.W, self.H))
        cv2.imwrite('shear_centered_' + str(self.count_img) + '.png',
                    centered_sheared)
        return centered_sheared

    def resize_image(self):
        resize_dimensions = (int(self.W / 2), int(self.H / 2))
        resize_img = cv2.resize(self.img, (500, 500),
                                interpolation=cv2.INTER_AREA)
        cv2.imwrite('resize_image_' + str(self.count_img) + '.png',
                    resize_img)
        return resize_img

    def rotate_img(
        self,
        angle,
        ID,
        rotating_points=None,
        ):
        if rotating_points is None:
            rotating_points = (self.W // 2, self.H // 2)
            dimensions = (self.W, self.H)
            rotateMatx = cv2.getRotationMatrix2D(rotating_points,
                    angle, 1.0)

        rotated_img = cv2.warpAffine(self.img, rotateMatx, dimensions)
        cv2.imwrite('rotate_img_' + str(self.count_img) + '_' + str(ID)
                    + '.png', rotated_img)
        return rotated_img

    def blur_image(self):
        kernelSizes = [(15, 15), (30, 30), (45, 45)]
        count = 0
        for (kX, kY) in kernelSizes:
            count += 1
            blurred = cv2.blur(self.img, (kX, kY))
            cv2.imwrite(str('blurred' + str(self.count_img)) + '_'
                        + str(count) + '.png', blurred)

        blurred_large = blurred = cv2.blur(self.img, (15, 15))
        return blurred_large

#    def gaussian_noise(self):
#        kernelSizes = [(15, 15), (30, 30), (45, 45)]
#        count = 0
#        for (kX, kY) in kernelSizes:
#            count += 1
#            blurred = cv2.GaussianBlur(self.img, (kX, kY), 0)
#            cv2.imwrite(str('gaussian_blurred' + str(self.count_img))
#                        + '_' + str(count) + '.png', blurred)

    def perspective_transforms(self):
        pts1 = np.float32([[100, 260], [640, 260], [0, 400], [640, 300]])
        pts2 = np.float32([[0, 0], [500, 0], [100, 640], [300, 640]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(self.img, matrix, (500, 600))
        cv2.imwrite('ptf__' + str(self.count_img) + '.png', result)

    def brightness(self):
        extremes = ((0.50, 3), (0.1, 4), (2, 6))
        count = 0
        for (low, high) in extremes:
            value = random.uniform(low, high)
            img_brightness = self.img
            hsv = cv2.cvtColor(img_brightness, cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 1] = hsv[:, :, 1] * value
            hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
            hsv[:, :, 2] = hsv[:, :, 2] * value
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
            hsv = np.array(hsv, dtype=np.uint8)
            img_brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            count += 1
            cv2.imwrite('brightness_' + str(self.count_img) + '_'
                        + str(count) + '.png', img_brightness)

    def distortions(self):
        transform = A.Compose([
            A.CLAHE(),
            A.RandomRotate90(),
            A.Transpose(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50,
                               rotate_limit=45, p=.75),
            A.Blur(blur_limit=3),
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.HueSaturationValue(),
            ])
        augmented_image = transform(image=self.img)
        cv2.imwrite('dist_' + str(self.count_img) + '.png',
                    augmented_image['image'])
        GridDistortion = A.GridDistortion()
        grid_dist = GridDistortion(image=self.img)
        cv2.imwrite('grid_dist_' + str(self.count_img) + '.png',
                    grid_dist['image'])

    def create_dataset(self):
        self.flip_top()
        self.flip_left()
        self.flip_right()
        self.blur_image()
        self.shear_centered()
        self.resize_image()
        count = 0
        for i in range(0, 360, 60):
            count += 1
            self.rotate_img(i, count)
#        self.gaussian_noise()
        self.perspective_transforms()
        self.brightness()
        self.distortions()


for i in range(1, 4):
    img_file = str(i) + '.JPG'
    img = cv2.imread(img_file)
    tf_obj = imageTransforms(img, i)
