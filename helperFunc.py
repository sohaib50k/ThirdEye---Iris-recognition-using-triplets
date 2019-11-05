import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
#Some helper functions
def prepareFeatures(model,imList,imgDimension,FEATURES):
    model2 = model
    actList = []
    batchSize = 8
    for x in range(len(imList)//batchSize):
        imL = []
        imLH = []
        for y in range(batchSize):
            it = y
            y += x * batchSize
            im = cv2.imread(imList[y])
            im = cv2.resize(im,(imgDimension,imgDimension))
            imLH.append(np.flip(im,0))
            imL.append(im)
        pred = model2.predict([imL])
        predH = model2.predict([imLH])
        for y in range(pred.shape[0]):
            fConcat = np.zeros(shape = (FEATURES*2))

            featureIm = pred[y][:FEATURES]#/np.linalg.norm(pred[y][classes:])
            fConcat[:FEATURES] = featureIm
            featureImh = predH[y][:FEATURES]#/np.linalg.norm(predV[y][classes:])
            fConcat[FEATURES:FEATURES*2] = featureImh
            
            
            actList.append(fConcat)

    a = np.corrcoef(actList)
    print ("Correlation between feature vectors")
    plt.matshow(a)
    plt.show()
    return actList

"""
   Multicore implementation for feature histogram calculation Uses 8 cores for now
   Hacky implementation which manipulates file names.
"""
def featureSelect(lst):
    truePositive = 0
    actList = lst[0]
    imList = lst[1]
    count = lst[2]
    lstDiv = len(actList)//8
    #print ("LST DIV = " + str(lstDiv))
    #print ("ACT LIST  = " + str(len(actList)))

    blueList = []
    redList = []
#     print ((lstDiv)*(count-1),(lstDiv)*(count))
    for i in range((lstDiv)*(count-1),(lstDiv)*(count)):
        d = []
        splitI = re.split('[/.]',imList[i])
        irisImage = splitI[-2]
        irisImage = irisImage[:5]+irisImage[-1]
        for j in range(len(imList)):

            if (i == j):
                d.append(1000)
                continue
            else:
                split = re.split('[/.]',imList[j])
                testImage = split[-2]
                testImage = testImage[:5]+testImage[-1]
                """Cosine distance"""
                dis = np.dot(actList[i] , actList[j])/(np.linalg.norm(actList[i] )*np.linalg.norm(actList[j]))
                dis = 1-dis
#                 dis = distance.cosine(actList[i],actList[j])
                d.append(dis)
                if (irisImage == testImage):
                    blueList.append(dis)
                else:               
                    redList.append(dis)
        split = re.split('[/.]',imList[d.index(min(d))])
        testImage = split[-2]
        testImage = testImage[:5]+testImage[-1]
        if(irisImage == testImage):
            truePositive+=1
    return blueList,redList,truePositive



