import codecs

import cv2
import os

def evaluate(rst,lbl):
    print(rst)
    rstimg = cv2.imread(rst)
    lblimg = cv2.imread(lbl)
    rstimg = cv2.resize(rstimg,(rstimg.shape[1]//16,rstimg.shape[0]//16))
    lblimg = cv2.resize(lblimg,(rstimg.shape[1],rstimg.shape[0]))
    matchedPixels = 0
    extraPixels = 0
    lostPixels = 0
    for j in range(rstimg.shape[0]):
        for k in range(rstimg.shape[1]):
            if (any(lblimg[j][k] != [0,0,0]) and any(rstimg[j][k] != [0,0,0])):
                matchedPixels += 1
                # print('matched!')
            elif (any(rstimg[j][k] != [0,0,0]) and all(lblimg[j][k] == [0,0,0])):
                extraPixels += 1
                # print('extra!')
            elif (any(lblimg[j][k] != [0,0,0]) and all(rstimg[j][k] == [0,0,0])):
                lostPixels += 1
                # print('lost!')
    print(lostPixels)
    print(matchedPixels)
    if matchedPixels == 0:
        return 1
    USE = (extraPixels)/(matchedPixels+lostPixels)
    if USE > 1 :
        USE = 1
    OSE = (lostPixels)/(matchedPixels+lostPixels)

    score = 1-(USE+OSE)/2
    print(score)
    return score

if __name__ == '__main__':
    for method in ['deeptest_1','deeptest_2','deeptest_3','deeptest_4','deeptest_5']:
        filePath = 'data/segnet_'+method+'_seg/'
        filelist = os.listdir(filePath)
        filelist.sort()
        erroneouslist_seven = []
        erroneouslist_eight = []
        erroneouslist_eive = []
        for file in filelist:
            filename = file
            predictfile = filePath + filename
            labelfile = 'data/segnet_'+method+'_label/' + filename
            score = evaluate(predictfile,labelfile)
            print(score)
            tag = int(filename.replace('.png',''))
            if score < 0.7:
                erroneouslist_seven.append(tag)
            if score < 0.8:
                erroneouslist_eight.append(tag)
            if score < 0.85:
                erroneouslist_eive.append(tag)
            print(len(erroneouslist_seven))
        f = codecs.open('log/0.7_new/log_segnet_'+method+'_eval_0.7.txt','a','utf-8')
        f.write(str(erroneouslist_seven))
        f.close()
        f = codecs.open('log/0.8_new/log_segnet_'+method+'_eval_0.8.txt','a','utf-8')
        f.write(str(erroneouslist_eight))
        f.close()
        f = codecs.open('log/0.85_new/log_segnet_'+method+'_eval_0.85.txt','a','utf-8')
        f.write(str(erroneouslist_eive))
        f.close()
        print(len(erroneouslist_seven))




