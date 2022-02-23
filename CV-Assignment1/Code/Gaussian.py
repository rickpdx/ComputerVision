import numpy as np


class GuassianFilter:

    # def __init__(self, kernel):
    #     self.kernel = kernel

    def convolution(self, image, kernel, padding=0, strides=1):
        # Cross Correlation
        kernel = np.flipud(np.fliplr(self.kernel))

        # Gather Shapes of Kernel + Image + Padding
        xKernShape = kernel.shape[0]
        yKernShape = kernel.shape[1]
        xImgShape = image.shape[0]
        yImgShape = image.shape[1]

        # Shape of Output
        xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
        yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
        output = np.zeros((xOutput, yOutput))

        # Apply Padding
        if padding != 0:
            imagePadded = np.zeros(
                (image.shape[0] + padding*2, image.shape[1] + padding*2))
            imagePadded[int(padding):int(-1 * padding),
                        int(padding):int(-1 * padding)] = image
        else:
            imagePadded = image

        # Iterate through image
        for y in range(image.shape[1]):

            # Stopping condition
            if y > image.shape[1] - yKernShape:
                break

            # Convolve if y has moved by the specified Strides
            if y % strides == 0:
                for x in range(image.shape[0]):
                    # Go to next row
                    if x > image.shape[0] - xKernShape:
                        break
                    try:
                        # Convolve if x has moved by the specified Strides
                        if x % strides == 0:
                            output[x, y] = np.sum(
                                kernel * imagePadded[x: x + xKernShape, y: y + yKernShape])
                            output[x, y] /= xKernShape * yKernShape
                    except:
                        break

        return output

    def sobelFilter(self, imageX, imageY):
        output = np.sqrt(np.square(imageX) + np.square(imageY))
        return output
