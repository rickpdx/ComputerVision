import cv2
import matplotlib.pyplot as plt


class SIFT_Features:

    def keypoints(self, source):
        sift = cv2.SIFT_create()
        img = cv2.imread(source)
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        kp_img = cv2.drawKeypoints(img, kp, gray)
        return kp_img, kp, des

    def drawMatches(self, kp_img1, kp_img2, kp, kp2, des, des2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des, des2, k=2)

        top = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                top.append([m])

        topImage = cv2.drawMatchesKnn(kp_img1, kp, kp_img2, kp2, top,
                                      None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite('part3/SIFT_draw.jpg', topImage)
