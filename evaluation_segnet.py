import codecs

import cv2
import os

def evaluate(rst,lbl,insertpoint):
    print(rst)
    rstimg = cv2.imread(rst)
    lblimg = cv2.imread(lbl)
    imgray=cv2.cvtColor(lblimg,cv2.COLOR_BGR2GRAY)
    ret1,thresh1=cv2.threshold(imgray,1,255,0)
    contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    pt = (insertpoint[1],insertpoint[0])
    tgtcontour = []
    up = 10000
    down = 0
    left = 10000
    right = 0
    lostPixels = 0
    matchedPixels = 0
    extraPixels = 0
    print(len(contours))
    for contour in contours:
        if cv2.pointPolygonTest(contour, pt, False) ==1:
            print('in')
            tgtcontour = contour
            break
        elif cv2.pointPolygonTest(contour, pt, False) == 0:
            print('on')
            tgtcontour = contour
            break
    # print(tgtcontour)
    if len(tgtcontour) == 0:
        return 0
    for point in tgtcontour:
        if left > point[0][0]:
            left = point[0][0]

        if right < point[0][0]:
            right = point[0][0]

        if up > point[0][1]:
            up = point[0][1]

        if down < point[0][1]:
            down = point[0][1]
    print(down - up)
    print(right - left)
    for j in range(down-up):
        for k in range(right-left):
            if (all(lblimg[j+up][k+left] == [255,255,255]) and all(rstimg[j+up][k+left] == [255,255,255])):
                matchedPixels += 1
                # print('matched!')
            elif (all(rstimg[j+up][k+left] == [255,255,255]) and any(lblimg[j+up][k+left] != [255,255,255])):
                extraPixels += 1
                # print('extra!')
            elif (all(lblimg[j+up][k+left] == [255,255,255]) and any(rstimg[j+up][k+left] != [255,255,255])):
                lostPixels += 1
                # print('lost!')
    print(lostPixels)
    print(matchedPixels)
    if (lostPixels + matchedPixels) == 0:
        return 0
    USE = (extraPixels)/(matchedPixels+lostPixels)
    # if USE > 1 :
    #     USE = 1
    OSE = (lostPixels)/(matchedPixels+lostPixels)

    score = 1-(USE+OSE)/2
    print(score)
    return score

if __name__ == '__main__':
    for method in ['guided_1']:
        filePath = 'data/segnet_'+method+'_seg/'
        with open('log_segnet_'+method+'.txt') as f:
            filelist = eval(f.read())
        erroneouslist_1=[]
        erroneouslist_3=[]
        erroneouslist_5=[]
        erroneouslist_7=[]
        erroneouslist_9=[]
        for file in filelist:
            filename = file[0]
            predictfile = filePath + filename
            labelfile = 'data/segnet_'+method+'_label/' + filename
            score = evaluate(predictfile,labelfile,file[5])
            print(score)
            tag = int(filename.replace('.png',''))

            if score < 0.1:
                erroneouslist_1.append(tag)
            if score < 0.3:
                erroneouslist_3.append(tag)
            if score < 0.5:
                erroneouslist_5.append(tag)
            if score < 0.7:
                erroneouslist_7.append(tag)
            if score < 0.9:
                erroneouslist_9.append(tag)

        f = codecs.open('log_segnet_'+method+'_eval_0.1.txt','a','utf-8')
        f.write(str(erroneouslist_1))
        f.close()
        f = codecs.open('log_segnet_'+method+'_eval_0.3.txt','a','utf-8')
        f.write(str(erroneouslist_3))
        f.close()
        f = codecs.open('log_segnet_'+method+'_eval_0.5.txt','a','utf-8')
        f.write(str(erroneouslist_5))
        f.close()
        f = codecs.open('log_segnet_'+method+'_eval_0.7.txt','a','utf-8')
        f.write(str(erroneouslist_7))
        f.close()
        f = codecs.open('log_segnet_'+method+'_eval_0.9.txt','a','utf-8')
        f.write(str(erroneouslist_9))
        f.close()
        # f = codecs.open('log_guided_segnet_0.8.txt','a','utf-8')
        # f.write(str(erroneouslist_eight))
        # f.close()
        # f = codecs.open('log_guided_segnet_0.9.txt','a','utf-8')
        # f.write(str(erroneouslist_nine))
        # f.close()


