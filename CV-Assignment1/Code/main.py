import numpy as np
import matplotlib.pyplot as plt
import cv2
from Kmeans import kMeans
from menu import KmeansMenu
from Gaussian import GuassianFilter
from SIFT import SIFT_Features
from sklearn.metrics import pairwise_distances


def convert_to_RGB(source):
    img = cv2.imread(source)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 360))
    img_shape = img.shape
    img = img.reshape((-1, 3))
    img = np.float32(img)
    return img, img_shape


def output_data(select, models, data, img_shape):

    # Choose model with lowest SSE
    pval = models[0].predict(data)

    # Construct image with clustering in RGB space
    labels = pval
    center = np.uint8(models[0].centroids)
    res = center[labels]

    # Output 2-D Plot for data
    if select == 1:
        plt.scatter(data[:, 0], data[:, 1],
                    c=pval, s=50, cmap='viridis')
        centers = models[0].centroids
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=300, alpha=0.8)
        plt.savefig("part2/K-means 2-D Plot.jpg")

    # write resulting images
    elif select == 2:
        result_image = res.reshape((img_shape))
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('part2/Kmean_img1_Result.jpg', result_image)

    elif select == 3:
        result_image = res.reshape((img_shape))
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('part2/Kmean_img2_Result.jpg', result_image)


def main():

    # data
    p1srcA = "part1/filter1_img.jpg"
    p1srcB = "part1/filter2_img.jpg"
    p2srcA = "part2/Kmean_img1.jpg"
    p2srcB = "part2/Kmean_img2.jpg"
    p3srcA = "part3/SIFT1_img.jpg"
    p3srcB = "part3/SIFT2_img.jpg"

    ### Part 1 ###

    # Kernels
    kernel_1 = np.array(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]]
    )
    kernel_2 = np.array(
        [[1, 4, 7, 4, 1],
         [4, 16, 26, 16, 4],
         [7, 26, 41, 26, 7],
         [4, 16, 26, 16, 4],
         [1, 4, 7, 4, 1]]
    )
    gxDoG = np.array(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]]
    )
    gyDoG = np.array(
        [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]]
    )

    guassian = GuassianFilter()

    # image1
    p1A = cv2.imread(p1srcA)
    p1A = cv2.cvtColor(src=p1A, code=cv2.COLOR_BGR2GRAY)

    # image2
    p1B = cv2.imread(p1srcB)
    p1B = cv2.cvtColor(src=p1B, code=cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    img1_k1 = guassian.convolution(p1A, kernel_1, padding=1)
    cv2.imwrite('part1/f1Gaussian3x3.jpg', img1_k1)
    img1_k2 = guassian.convolution(p1A, kernel_2, padding=2)
    cv2.imwrite('part1/f1Gaussian5x5.jpg', img1_k2)

    img2_k1 = guassian.convolution(p1B, kernel_1, padding=1)
    cv2.imwrite('part1/f2Gaussian3x3.jpg', img2_k1)
    img2_k2 = guassian.convolution(p1B, kernel_2, padding=2)
    cv2.imwrite('part1/f2Gaussian5x5.jpg', img2_k2)

    # Derivative of Gaussian
    img1_gxDoG = guassian.convolution(p1A, gxDoG, padding=1)
    cv2.imwrite('part1/f1_gxDoG.jpg', img1_gxDoG)
    img1_gyDoG = guassian.convolution(p1A, gyDoG, padding=1)
    cv2.imwrite('part1/f1_gyDoG.jpg', img1_gyDoG)

    img2_gxDoG = guassian.convolution(p1B, gxDoG, padding=1)
    cv2.imwrite('part1/f2_gxDoG.jpg', img2_gxDoG)
    img2_gyDoG = guassian.convolution(p1B, gyDoG, padding=1)
    cv2.imwrite('part1/f2_gyDoG.jpg', img2_gyDoG)

    # Sobel Filter
    img1_SF = guassian.sobelFilter(img1_gxDoG, img1_gyDoG)
    cv2.imwrite('part1/f1_SobelFilter.jpg', img1_SF)
    img2_SF = guassian.sobelFilter(img2_gxDoG, img2_gyDoG)
    cv2.imwrite('part1/f2_SobelFilter.jpg', img2_SF)

    ### Part 2 ###
    p2A, p2Ashape = convert_to_RGB(p2srcA)
    p2B, p2Bshape = convert_to_RGB(p2srcB)

    # Program menu for part 2 Kmeans algorithm
    choice = False
    print("Select the data for the kmeans algorithm")
    while choice == False:
        print("1. 510_cluster_dataset.txt")
        print("2. Kmean_img1.jpg")
        print("3. Kmean_img2.jpg")
        select = int(input())
        if(select < 1 or select > 3):
            print("Invalid selection. Try again")
        else:
            choice = True

    # Assign the data
    if select == 1:
        dataSrc = "part2/510_cluster_dataset.txt"
        data = np.loadtxt(dataSrc)

    elif select == 2:
        data = p2A
        img_shape = p2Ashape

    elif select == 3:
        data = p2B
        img_shape = p2Bshape

    # Alogrithm menu
    kMenu = KmeansMenu()
    kMenu.prompt()
    models = []

    # Run algorithm r number of times and seeds with r
    for i in range(kMenu.rvalue):
        km = kMeans(kMenu.kval, i)
        fitted = km.fit_kmeans(data)
        models.append(fitted)

    # Sort models by lowest SSE
    models.sort(key=lambda a: a.SSE)
    print("Sum of squares error(SSE) for selected model: ", models[0].SSE)

    output_data(select, models, data, img_shape)

    ### Part 3 ###
    sift = SIFT_Features()

    kp_img1, kp, des = sift.keypoints(p3srcA)
    kp_img2, kp2, des2 = sift.keypoints(p3srcB)
    cv2.imwrite('part3/SIFT_kp1.jpg', kp_img1)
    cv2.imwrite('part3/SIFT_kp2.jpg', kp_img2)

    sift.drawMatches(kp_img1, kp_img2, kp, kp2, des, des2)


if __name__ == "__main__":
    main()
