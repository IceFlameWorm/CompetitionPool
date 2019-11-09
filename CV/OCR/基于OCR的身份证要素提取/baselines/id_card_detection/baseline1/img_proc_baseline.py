#本文为基于OCR的身份证要素提取数据预处理的baseline
#该方法基于OpenCV，各位选手自行扩展
#By Alex-拉斐尔
#opencv环境 自行安装
import cv2
import numpy as np

#读取数据路径
image = cv2.imread('2.png',0)
#去掉边线白色的线框
image_1 = image[10:990,10:990]
#自适应阈值法过滤图片
img = cv2.adaptiveThreshold(image_1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,2)
#膨胀操作，用5*5的核，这个参数可以自行修改
#将目标区域连通起来
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel)
#OpenCV寻找轮廓函数
contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#处理轮廓
for i in range(0,len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    #保留轮廓的阈值
    if w>430 and h>250 and w<600 and h<600:
        print(x, y, w, h)
        img_t = image_1[y-10:y+h+10,x-10:x+w+10]
        #保存函数，注意保存路径
        #cv2.imwrite(‘.\00\‘+str(time.time())+’.png’,img_t)
        # cv2.drawContours(image_1, contours, i, (0,255,0), 3)
        cv2.rectangle(image_1, (x,y), (x+w,y+h), 255, 5)
        cv2.imshow('00',image_1)
        cv2.waitKey(0)