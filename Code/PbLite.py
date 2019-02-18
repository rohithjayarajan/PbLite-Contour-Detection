#!/usr/bin/env python

"""
@file    PbLite.py
@author  rohithjayarajan
@date 1/28/2019
Template Credits: Nitin J Sanket and Chahatdeep Singh

Licensed under the
GNU General Public License v3.0
"""

import os
import math
import numpy as np
import cv2
from scipy.ndimage import convolve, rotate
import matplotlib.pyplot as plt
import imutils
from sklearn.cluster import KMeans

# debug = True
debug = False


def GenerateGaussian(Size, SigmaX, SigmaY):
    x, y = np.mgrid[-Size/2:Size/2+1, -Size/2:Size/2+1]
    GaussianFilter = np.exp(-(x**2/(2 * SigmaX**2) + y **
                              2/(2 * SigmaY**2)))/(2 * math.pi * SigmaX*SigmaY)

    return GaussianFilter


def GenerateDoGFilter(Size, SigmaX, SigmaY, Orientation):
    SobelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]])/8
    SobelY = np.transpose(SobelX)

    GaussianFilter = GenerateGaussian(Size, SigmaX, SigmaY)
    DoGFilter = convolve(GaussianFilter, SobelY)
    OrientedDoGFilter = rotate(DoGFilter, Orientation, reshape=False)

    if(debug):
        print("Sobel operator in x direction: ")
        print(SobelX)
        print("Sobel operator in y direction: ")
        print(SobelY)
        print("Gaussian: ")
        print(GaussianFilter)
        print("DoG: ")
        print(DoGFilter)
        print("Rotated DoG: ")
        print(OrientedDoGFilter)

        plt.imshow(OrientedDoGFilter, cmap=plt.get_cmap(
            'gray'), interpolation='nearest')
        plt.colorbar()
        plt.show()

    return OrientedDoGFilter


def GenerateSecondDoGFilter(Size, SigmaX, SigmaY, Orientation):
    SobelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]])/8
    SobelY = np.transpose(SobelX)

    DoGFilter = GenerateDoGFilter(Size, SigmaX, SigmaY, Orientation)
    SecondDoGFilter = convolve(DoGFilter, SobelY)
    OrientedSecondDoGFilter = rotate(
        SecondDoGFilter, Orientation, reshape=False)
    return OrientedSecondDoGFilter


def GenerateDoGFilterBank(Size, Scale, NumOrientations):
    DoGFilterBank = []

    for o in range(NumOrientations):
        DoGFilterBank.append(GenerateDoGFilter(
            Size, Scale, Scale, o*360/NumOrientations))

    return DoGFilterBank


def DisplayFilterBank(FilterBank, Rows, Columns, Cmap, Path, Dst):
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, Columns*Rows + 1):
        img = np.random.randint(10, size=(h, w))
        fig.add_subplot(Rows, Columns, i)
        plt.imshow(FilterBank[i-1], cmap=Cmap)
    fig.savefig(os.path.join(Path, Dst))
    plt.show()

# 48 filters
# first and second order derivatives of Gaussians
# at 6 orientations 3 scales
# = 36
# 8 LOG filters
# 4 Gaussians

# LMS
# Basic scales
# 1, math.sqrt(2), 2, 2.math.sqrt(2)
# first and second derivatives at first three scales
# sigmax = sigma; sigmay = 3sigma
# gaussians at four basic scales
# 8 LoG at sigma and 3sigma


def LaplacianOfGaussian(Size, SigmaX, SigmaY):
    LaplacianNegative = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]])

    GaussianFilter = GenerateGaussian(Size, SigmaX, SigmaY)
    LoGFilter = convolve(GaussianFilter, LaplacianNegative)
    return LoGFilter


def GenerateLMSFilterBank(Size):
    LMSFilterBank = []
    NumOrientations = 6
    LMSBasicScales = [1, math.sqrt(2), 2, 2*math.sqrt(2)]

    # first order derivative of Gaussians at 6 orientations and 3 scales
    for s in LMSBasicScales[:3]:
        for o in range(NumOrientations):
            LMSFilterBank.append(GenerateDoGFilter(
                Size, s, 3*s, o*360/NumOrientations))

    # second order derivative of Gaussians at 6 orientations and 3 scales
    for s in LMSBasicScales[:3]:
        for o in range(NumOrientations):
            LMSFilterBank.append(GenerateSecondDoGFilter(
                Size, s, 3*s, o*360/NumOrientations))

    # Gaussians at basic scales
    for s in LMSBasicScales:
        LMSFilterBank.append(GenerateGaussian(Size, s, s))

    # LoG at basic scales
    for s in LMSBasicScales:
        LMSFilterBank.append(LaplacianOfGaussian(Size, s, s))
        LMSFilterBank.append(LaplacianOfGaussian(Size, 3*s, 3*s))

    if debug:
        for i in LMSFilterBank:
            print(i.shape)

    return LMSFilterBank


def GenerateLMLFilterBank(Size):
    LMLFilterBank = []
    NumOrientations = 6
    LMLBasicScales = [math.sqrt(2), 2, 2*math.sqrt(2), 4]

    # first order derivative of Gaussians at 6 orientations and 3 scales
    for s in LMLBasicScales[:3]:
        for o in range(NumOrientations):
            LMLFilterBank.append(GenerateDoGFilter(
                Size, s, 3*s, o*360/NumOrientations))

    # second order derivative of Gaussians at 6 orientations and 3 scales
    for s in LMLBasicScales[:3]:
        for o in range(NumOrientations):
            LMLFilterBank.append(GenerateSecondDoGFilter(
                Size, s, 3*s, o*360/NumOrientations))

    # Gaussians at basic scales
    for s in LMLBasicScales:
        LMLFilterBank.append(GenerateGaussian(Size, s, s))

    # LoG at basic scales
    for s in LMLBasicScales:
        LMLFilterBank.append(LaplacianOfGaussian(Size, s, s))
        LMLFilterBank.append(LaplacianOfGaussian(Size, 3*s, 3*s))

    return LMLFilterBank


def GenerateGaborFilter(Size, Sigma, Theta, Lambda, Psi, Gamma):

    SigmaX = Sigma
    SigmaY = float(Sigma)/Gamma
    x, y = np.mgrid[-Size/2:Size/2+1, -Size/2:Size/2+1]
    xTheta = x*np.cos(Theta) + y*np.sin(Theta)
    yTheta = -x*np.sin(Theta) + y*np.cos(Theta)

    ModulatingPlane = np.cos((2*np.pi/Lambda)*xTheta + Psi)
    GaussianFilter = np.exp(-(xTheta**2/(2 * SigmaX**2) + yTheta **
                              2/(2 * SigmaY**2)))/(2 * math.pi * SigmaX*SigmaY)
    GaborFilter = GaussianFilter*ModulatingPlane
    return GaborFilter


def GenerateGaborFilterBank(Size, SigmaList, NumOrientations, LambdaList, PsiList, GammaList, Cmap, Path, Dst):
    GaborFilterBank = []

    for s in SigmaList:
        for o in range(NumOrientations):
            for l in LambdaList:
                for p in PsiList:
                    for g in GammaList:
                        GaborFilterBank.append(GenerateGaborFilter(
                            Size, s, o*360/NumOrientations, l, p, g))

    DisplayFilterBank(GaborFilterBank, int(
        len(SigmaList)*NumOrientations*len(LambdaList)*len(PsiList)*len(GammaList)/8), 8, Cmap, Path, Dst)
    return GaborFilterBank


def GenerateHalfDisc(Radius):
    HalfDisk = np.zeros((2*Radius+1, 2*Radius+1))

    for i in range(Radius):
        a = i - Radius
        for j in range(2*Radius + 1):
            b = j - Radius
            if a**2 + b**2 <= Radius**2:
                HalfDisk[i, j] = 1
    return HalfDisk


def GenerateHalfDiscBank(RadiusList, NumOrientations, Cmap, Path, Dst):

    HalfDisksBank = []
    for r in RadiusList:
        for o in range(int(NumOrientations/2)):
            RotatedDisc = rotate(GenerateHalfDisc(
                r), o*360/NumOrientations, reshape=False)
            # RotatedDisc = imutils.rotate(
            #     GenerateHalfDisc(r), o*360/NumOrientations)
            RotatedDisc[RotatedDisc >= 0.3] = 1
            RotatedDisc[RotatedDisc < 0.3] = 0
            HalfDisksBank.append(RotatedDisc)

            RotatedDisc = rotate(GenerateHalfDisc(
                r), o*360/NumOrientations + 180, reshape=False)
            # RotatedDisc = imutils.rotate(
            #     GenerateHalfDisc(r), o*360/NumOrientations + 180)
            RotatedDisc[RotatedDisc >= 0.3] = 1
            RotatedDisc[RotatedDisc < 0.3] = 0
            HalfDisksBank.append(RotatedDisc)
            # print(RotatedDisc)
    DisplayFilterBank(HalfDisksBank, int(
        len(RadiusList)*NumOrientations/8), 8, Cmap, Path, Dst)
    return HalfDisksBank


def ConvolveWithFilterBank(Image, FilterBank):

    NDImage = np.zeros((Image.shape[0], Image.shape[1], len(FilterBank)))
    i = 0
    for Filter in FilterBank:
        NDImage[:, :, i] = convolve(Image, Filter)
        i += 1

    return NDImage


def ChiSqauredMatrix(Image, LeftMask, RightMask, Bins):

    ChiSqauredMatrix = np.zeros((Image.shape))
    ChiSqauredMatrix = ChiSqauredMatrix.astype(float)
    E = np.full((Image.shape), np.spacing(np.single(1)), dtype=np.float64)
    for i in range(Bins):
        Temp = (Image == i)
        Temp = Temp.astype(float)
        Gi = convolve(Temp, LeftMask)
        Hi = convolve(Temp, RightMask)
        ChiSqauredMatrix += np.divide((Gi - Hi)**2, (Gi + Hi) + E)
        # ChiSqauredMatrix += np.divide((Gi - Hi)**2, (Gi + Hi),
        #                               out=np.zeros_like((Gi - Hi)**2), where=(Gi + Hi) != 0)
    if debug:
        print(ChiSqauredMatrix.shape)

    return ChiSqauredMatrix/2


def ComputeMapGradient(Image, HalfDiscBank, Bins):

    MapGradient = np.zeros(
        (Image.shape[0], Image.shape[1], int(len(HalfDiscBank)/2)))
    if debug:
        print(len(HalfDiscBank))
    count = 0
    for i in range(0, int(len(HalfDiscBank)/2)+1, 2):
        MapGradient[:, :, count] = ChiSqauredMatrix(
            Image, HalfDiscBank[i], HalfDiscBank[i+1], Bins)
        count += 1

    # print(MapGradient.shape)
    return(MapGradient)


def ComputeGradient(MapGradient):
    return np.mean(MapGradient, axis=2)


def PbLite(ImageName):
    ResultsPath = os.path.join(os.getcwd(), 'Code', 'Results')
    """
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png,
    use command "cv2.imwrite(...)"
    """
    DoGFilterBank1 = GenerateDoGFilterBank(12, 1.2, 16)
    DoGFilterBank2 = GenerateDoGFilterBank(12, 1.4, 16)
    DoGFilterBank = DoGFilterBank1 + DoGFilterBank2
    DisplayFilterBank(DoGFilterBank, 4, 8, 'gray', ResultsPath, 'DoG.png')

    """
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
    LMSFilterBank = GenerateLMSFilterBank(38)
    LMLFilterBank = GenerateLMLFilterBank(38)
    LMFilterBank = LMSFilterBank + LMLFilterBank
    DisplayFilterBank(LMFilterBank, 8, 12, 'gray', ResultsPath, 'LM.png')

    """
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""

    SigmaList = [4, 4*math.sqrt(2), 8]
    NumOrientations = 8
    LambdaList = [5, 7, 10]
    PsiList = [180]
    GammaList = [0.8]
    GaborFilterBank = GenerateGaborFilterBank(38, SigmaList, NumOrientations,
                                              LambdaList, PsiList, GammaList, 'gray', ResultsPath, 'Gabor.png')

    """
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""

    HalfDiscBank = GenerateHalfDiscBank(
        [5, 9, 15], 16, 'gray', ResultsPath, 'HDMasks.png')

    """
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""
    AbsolutePathImage = os.path.join(
        os.getcwd(), 'BSDS500', 'Images', ImageName + '.jpg')

    ColorImage = cv2.imread(AbsolutePathImage)

    ColorImageShape = ColorImage.shape
    BWImage = cv2.cvtColor(ColorImage, cv2.COLOR_BGR2GRAY)
    NDImage = ConvolveWithFilterBank(BWImage, DoGFilterBank)

    if debug:
        print("conv shape")
        print(NDImage.shape)

    DN1Image = NDImage.reshape((-1, 32))
    DN1Image = np.float32(DN1Image)

    criteria1 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K1 = 64
    ret1, TextonMap, center1 = cv2.kmeans(
        DN1Image, K1, None, criteria1, 10, cv2.KMEANS_RANDOM_CENTERS)

    """
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""
    TextonName = 'TextonMap_' + ImageName + '.png'
    TextonMap = TextonMap.reshape(ColorImageShape[0], ColorImageShape[1])
    # print(TextonMap.shape)
    plt.imsave(os.path.join(ResultsPath, TextonName), TextonMap, cmap='jet')
    plt.imshow(TextonMap, cmap='jet')
    plt.show()

    """
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
    TextonGradientName = 'Tg_' + ImageName + '.png'
    TextonMapGradient = ComputeMapGradient(TextonMap, HalfDiscBank, K1)
    TextonGradient = ComputeGradient(TextonMapGradient)

    plt.imsave(os.path.join(ResultsPath, TextonGradientName),
               TextonGradient, cmap='jet')
    plt.imshow(TextonGradient, cmap='jet')
    plt.show()

    """
	Generate Brightness Map
	Perform brightness binning
	"""

    D1BWImage = BWImage.reshape((-1, 1))
    D1BWImage = np.float32(D1BWImage)
    criteria2 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K2 = 16
    ret2, BrightnessMap, center2 = cv2.kmeans(
        D1BWImage, K2, None, criteria2, 10, cv2.KMEANS_RANDOM_CENTERS)

    BrightnessName = 'BrightnessMap_' + ImageName + '.png'
    BrightnessMap = BrightnessMap.reshape(
        ColorImageShape[0], ColorImageShape[1])
    # print(BrightnessMap.shape)
    plt.imsave(os.path.join(ResultsPath, BrightnessName),
               BrightnessMap, cmap='jet')
    plt.imshow(BrightnessMap, cmap='jet')
    plt.show()

    """
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
    BrightnessGradientName = 'Bg_' + ImageName + '.png'
    BrightnessMapGradient = ComputeMapGradient(BrightnessMap, HalfDiscBank, K2)
    BrightnessGradient = ComputeGradient(BrightnessMapGradient)

    plt.imsave(os.path.join(ResultsPath, BrightnessGradientName),
               BrightnessGradient, cmap='jet')
    plt.imshow(BrightnessGradient, cmap='jet')
    plt.show()

    """
	Generate Color Map
	Perform color binning or clustering
	"""

    DN3Image = ColorImage.reshape((-1, 3))
    DN3Image = np.float32(DN3Image)

    criteria3 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K3 = 16
    ret3, ColorMap, center3 = cv2.kmeans(
        DN3Image, K3, None, criteria3, 10, cv2.KMEANS_RANDOM_CENTERS)

    ColorName = 'ColorMap_' + ImageName + '.png'
    ColorMap = ColorMap.reshape(ColorImageShape[0], ColorImageShape[1])
    # print(ColorMap.shape)
    plt.imsave(os.path.join(ResultsPath, ColorName), ColorMap, cmap='jet')
    plt.imshow(ColorMap, cmap='jet')
    plt.show()

    """
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
    ColorGradientName = 'Cg_' + ImageName + '.png'
    ColorMapGradient = ComputeMapGradient(ColorMap, HalfDiscBank, K3)
    ColorGradient = ComputeGradient(ColorMapGradient)

    plt.imsave(os.path.join(ResultsPath, ColorGradientName),
               ColorGradient, cmap='jet')
    plt.imshow(ColorGradient, cmap='jet')
    plt.show()

    """
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""
    AbsolutePathSobelBaseline = os.path.join(
        os.getcwd(), 'BSDS500', 'SobelBaseline', ImageName + '.png')

    SobelBaselineImage = cv2.imread(AbsolutePathSobelBaseline, 0)
    # cv2.imshow('image', SobelBaselineImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    """
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""
    AbsolutePathCannyBaseline = os.path.join(
        os.getcwd(), 'BSDS500', 'CannyBaseline', ImageName + '.png')

    CannyBaselineImage = cv2.imread(AbsolutePathCannyBaseline, 0)
    # cv2.imshow('image', CannyBaselineImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    """
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
    PbLiteName = 'PbLite_' + ImageName + '.png'
    w1 = 0.5
    w2 = 0.5
    PbLite = np.multiply((TextonGradient + BrightnessGradient +
                          ColorGradient)/(3.0), w1*CannyBaselineImage + w2*SobelBaselineImage)

    plt.imsave(os.path.join(ResultsPath, PbLiteName), PbLite, cmap='gray')
    plt.imshow(PbLite, cmap='gray')
    plt.show()

    # cv2.imshow('PbLite', PbLite)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.imshow((TextonGradient + BrightnessGradient +
                ColorGradient)/(3.0), cmap='gray')
    plt.show()


def main():
    for i in range(1, 11):
        PbLite(str(i))


if __name__ == '__main__':
    main()
