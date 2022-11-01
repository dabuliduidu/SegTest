import cv2
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import time


def translation(imgName,lblName,tx,ty,times):
    # 1. 读取图片
    img1 = cv2.imread(imgName)
    lbl = cv2.imread(lblName)
    img1 = cv2.resize(img1,(lbl.shape[1],lbl.shape[0]))
    list = os.listdir('data/segnet_deeptest_'+str(times)+'_image/')
    count = len(list)
    newimgName = 'data/segnet_deeptest_'+str(times)+'_image/'+ str(count) + '.png'
    newlblName = newimgName.replace('image', 'label')
    # 2. 图像平移
    rows, cols = img1.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])  # 平移矩阵
    dst = cv2.warpAffine(img1, M, (cols, rows))
    dstlbl = cv2.warpAffine(lbl, M, (cols, rows))
    cv2.imwrite(newimgName,dst)
    cv2.imwrite(newlblName,dstlbl)


    return 0

def scale(imgName,lblName,sx,sy,times):
    # 1. 读取图片
    img1 = cv2.imread(imgName)
    lbl = cv2.imread(lblName)
    img1 = cv2.resize(img1,(lbl.shape[1],lbl.shape[0]))
    list = os.listdir('data/segnet_deeptest_'+str(times)+'_image/')
    count = len(list)
    newimgName = 'data/segnet_deeptest_'+str(times)+'_image/'+ str(count) + '.png'
    newlblName = newimgName.replace('image', 'label')
    # 2.图像缩放
    # 2.1 绝对尺寸
    rows, cols = img1.shape[:2]
    res = cv2.resize(img1, (2 * cols, 2 * rows), interpolation=cv2.INTER_CUBIC)
    # 2.2 相对尺寸
    res1 = cv2.resize(img1, None, fx=sx, fy=sy)
    dstlbl = cv2.resize(lbl, None, fx=sx, fy=sy)
    cv2.imwrite(newimgName,res1)
    cv2.imwrite(newlblName,dstlbl)

    return 0

def shear(imgName,lblName,sx,sy,times):
    # 1. 读取图片
    img1 = cv2.imread(imgName)
    lbl = cv2.imread(lblName)
    img1 = cv2.resize(img1,(lbl.shape[1],lbl.shape[0]))
    list = os.listdir('data_deeptest_new/segnet_deeptest_'+str(times)+'_image/')
    count = len(list)
    newimgName = 'data_deeptest_new/segnet_deeptest_'+str(times)+'_image/'+ str(count) + '.png'
    newlblName = newimgName.replace('image', 'label')
    # 2 仿射变换
    rows, cols = img1.shape[:2]
    # 2.1 创建变换矩阵
    M = np.float32([[1, sx, 0], [sy, 1, 0]])  # 平移矩阵
    dst = cv2.warpAffine(img1, M, (cols, rows))
    dstlbl = cv2.warpAffine(lbl, M, (cols, rows))
    cv2.imwrite(newimgName,dst)
    cv2.imwrite(newlblName,dstlbl)

def rotation(imgName,lblName,degree,times):
    # 1. 读取图片
    img1 = cv2.imread(imgName)
    lbl = cv2.imread(lblName)
    img1 = cv2.resize(img1,(lbl.shape[1],lbl.shape[0]))
    list = os.listdir('data/segnet_deeptest_'+str(times)+'_image/')
    count = len(list)
    newimgName = 'data/segnet_deeptest_'+str(times)+'_image/'+ str(count) + '.png'
    newlblName = newimgName.replace('image', 'label')
    # 2 图像旋转
    rows, cols = img1.shape[:2]
    # 2.1 生成旋转矩阵
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
    # 2.2 进行旋转变换
    dst = cv2.warpAffine(img1, M, (cols, rows))
    dstlbl = cv2.warpAffine(lbl, M, (cols, rows))
    cv2.imwrite(newimgName,dst)
    cv2.imwrite(newlblName,dstlbl)
    return 0

def contrast(imgName,lblName,gain,times):
    img = cv2.imread(imgName)
    lbl = cv2.imread(lblName)
    img = cv2.resize(img,(lbl.shape[1],lbl.shape[0]))
    list = os.listdir('data/segnet_deeptest_'+str(times)+'_image/')
    count = len(list)
    newimgName = 'data/segnet_deeptest_'+str(times)+'_image/'+ str(count) + '.png'
    newlblName = newimgName.replace('image', 'label')
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, gain, blank, gain, 0)
    cv2.imwrite(newimgName,dst)
    cv2.imwrite(newlblName,lbl)
    return 0

def brightness(imgName,lblName,bias,times):
    img = cv2.imread(imgName)
    lbl = cv2.imread(lblName)
    img = cv2.resize(img,(lbl.shape[1],lbl.shape[0]))
    list = os.listdir('data/segnet_deeptest_'+str(times)+'_image/')
    count = len(list)
    newimgName = 'data/segnet_deeptest_'+str(times)+'_image/'+ str(count) + '.png'
    newlblName = newimgName.replace('image', 'label')
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, 1, blank, 0, bias)
    cv2.imwrite(newimgName,dst)
    cv2.imwrite(newlblName,lbl)
    return 0

def blur(imgName,lblName,strategy,times):
    img = cv2.imread(imgName)
    lbl = cv2.imread(lblName)
    img = cv2.resize(img,(lbl.shape[1],lbl.shape[0]))
    list = os.listdir('data/segnet_deeptest_'+str(times)+'_image/')
    count = len(list)
    newimgName = 'data/segnet_deeptest_'+str(times)+'_image/'+ str(count) + '.png'
    newlblName = newimgName.replace('image', 'label')
    if strategy == 'Averaging':
        dst = cv2.blur(img, (5,5))
    if strategy == 'Gaussian':
        dst = cv2.GaussianBlur(img, (5,5), 0, 0)
    if strategy == 'Median':
        dst = cv2.medianBlur(img, 3)
    if strategy == 'Bilateral Filter':
        dst = cv2.bilateralFilter(img,25,100,100)
    cv2.imwrite(newimgName,dst)
    cv2.imwrite(newlblName,lbl)
    return 0

if __name__ == '__main__':
    starttime = time.time() # 开始记录
    imgs = os.listdir('backgrounds/')
    imgs.remove('.DS_Store')
    imgs.sort()
    for times in [1,2,3,4,5]:
        for img in imgs:
            imgName = 'backgrounds/' + img
            lblName = imgName.replace('backgrounds', 'segnet_backseg')
            # a1 = random.randint(10, 100)
            # a2 = random.randint(10, 100)
            # b1 = random.randint(2, 6)
            # b2 = random.randint(2, 6)
            c1 = 0.1*random.randint(-5, 5)
            c2 = 0
            # d = random.randint(3, 30)
            # e = random.randint(2, 3)
            # f = random.randint(10, 100)
            # g = random.choice(['Averaging','Gaussian','Median','Bilateral Filter'])
            # translation(imgName,lblName,a1,a2,times)
            # scale(imgName,lblName,b1,b2,times)
            shear(imgName,lblName,c1,c2,times)
            # rotation(imgName,lblName,d,times)
            # contrast(imgName,lblName,e,times)
            # brightness(imgName,lblName,f,times)
            # blur(imgName,lblName,g,times)
            print(img)
    endtime = time.time() # 结束记录
    dtime = endtime - starttime

    print("程序运行时间：%.8s s" % dtime)  # 显示到微秒
