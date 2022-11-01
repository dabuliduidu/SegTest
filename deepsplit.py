import os

for SegSoftware in ['segnet','hrnet','IBM']:
    for i in ['1','2','3','4','5']:
        ST1 = []
        ST2 = []
        ST3 = []
        ST4 = []
        ST5 = []
        ST6 = []
        ST7 = []
        ErrorList = [ST1,ST2,ST3,ST4,ST5,ST6,ST7]
        with open('log/0.7_new/log_'+SegSoftware+'_deeptest_'+ i + '_eval_0.7.txt') as f:
            wholelist = eval(f.read())
        for error in wholelist:
            kind = error % 7
            ErrorList[kind].append(error)
        print(len(wholelist))
        print(SegSoftware + '_' + i + '_ST1:' +str(len(ErrorList[0])))
        print(SegSoftware + '_' + i + '_ST2:' +str(len(ErrorList[1])))
        # print(SegSoftware + '_' + i + '_ST3:' +str(len(ErrorList[2])))
        print(SegSoftware + '_' + i + '_ST4:' +str(len(ErrorList[3])))
        print(SegSoftware + '_' + i + '_ST5:' +str(len(ErrorList[4])))
        print(SegSoftware + '_' + i + '_ST6:' +str(len(ErrorList[5])))
        print(SegSoftware + '_' + i + '_ST7:' +str(len(ErrorList[6])))





