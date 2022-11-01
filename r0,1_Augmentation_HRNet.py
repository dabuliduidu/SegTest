import cv2
import numpy as np
import os
import random
import glob
import codecs
import math
import json
# labelPath = 'label/'
# objPath = ''
import time

starttime = time.time() # 开始记录

def augmentlist():
    with open('../obj_ifm.txt') as f:
        objList = eval(f.read())
    backgrounds = os.listdir('backgrounds')
    backgrounds.remove('.DS_Store')
    objpool = []
    augmentList = []
    for obj in objList:
        objpool.append(obj)
    for b in backgrounds:
        # bimg = cv2.imread('/Users/houzhonghao/academic/Datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test/whole/'+b)
        # shape = bimg.shape
        pointpool = []
        for i in range(256):
            for j in range(512):
                dist = min(i,j,255-i,511-j)
                weight = 1
                if dist < 50:
                    weight = weight*2
                for k in range(weight):
                    pointpool.append((i,j))
        selectedpoint = random.choice(pointpool)
        selectedobj = random.choice(objpool)
        augmentList.append([selectedpoint,selectedobj,b])
        print(len(augmentList))
    return augmentList
def findContours(fileName):
    img = cv2.imread(fileName)
    Contours = []
    contours = cv2.findContours(img)
    for a in contours:
        Contours.append(a)
    return Contours

def objectInsertion(labelName,imageName,obj,insertpoint,times):
    count = len(os.listdir('data/hrnet_r01_'+str(times)+'_image/'))
    newImagename = 'data/hrnet_r01_'+str(times)+'_image/'+str(count)+'.png'
    newLabelname = 'data/hrnet_r01_'+str(times)+'_label/'+str(count)+'.png'
    img = cv2.imread(imageName)
    label = cv2.imread(labelName)
    print(labelName)
    img = cv2.resize(img,(label.shape[1],label.shape[0]))
    # scale = img.shape[0]/1024


    objName = '../obj/'+str(obj[0])+'.png'
    # objlblName = 'obj_label/'+str(obj[0][0])+'.png'
    objimg = cv2.imread(objName)
    objimg = cv2.resize(objimg,(int(objimg.shape[1]/4),int(objimg.shape[0]/4)))
    # objimg = cv2.resize(objimg,(int(objimg.shape[1]*scale),int(objimg.shape[0]*scale)))
    centroid = (int(obj[6][0]/4),int(obj[6][1]/4))
    objsize = obj[1]
    realinsertpoint = (insertpoint[0]-centroid[0],insertpoint[1]-centroid[1])
    for i in range(min(objimg.shape[0],img.shape[0]-realinsertpoint[0])):
        for j in range(min(objimg.shape[1],img.shape[1]-realinsertpoint[1])):
            if (i+realinsertpoint[0] < 0 or j+realinsertpoint[1] < 0):
                continue
            if any(objimg[i][j] != [0,0,0]):
                img[i+realinsertpoint[0]][j+realinsertpoint[1]] = objimg[i][j]
                label[i+realinsertpoint[0]][j+realinsertpoint[1]] = [255,255,255]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if any(label[i][j] != [0,0,0]):
                label[i][j] = [255,255,255]
    cv2.imwrite(newImagename, img)
    cv2.imwrite(newLabelname, label)
    log = [str(count)+'.png',obj[1],obj[2],obj[3],obj[4],insertpoint,imageName]
    f = codecs.open('log_hrnet_r01_'+str(times)+'.txt','a','utf-8')
    f.write(str(log)+',')
    f.close()
    return newImagename,newLabelname




def ruleGenerate(augmentList,times):
    for augmentation in augmentList:
        image = 'backgrounds/'+augmentation[2]
        label = image.replace('backgrounds','hrnet_backseg')
        insertpoint = augmentation[0]
        obj = augmentation[1]
        objectInsertion(label,image,obj,insertpoint,times)
    return 0

for times in [6]:
    augmentList = augmentlist()
    ruleGenerate(augmentList,times)
endtime = time.time() # 结束记录
dtime = endtime - starttime

print("程序运行时间：%.8s s" % dtime)  # 显示到微秒


