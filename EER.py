# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from sklearn.metrics import auc
import glob
import os
import numpy as np
from multiprocessing import Pool
from itertools import product


def checkThresh(lst):
    thresh = lst[0]
    l = lst[1]

    counter = 0
    for x in range(l.shape[0]):
        if (l[x] < thresh):
            counter +=1
    return counter
def checkThreshT(lst):
    thresh = lst[0]
    l = lst[1]

    counter = 0
    for x in range(l.shape[0]):
        if (l[x] >thresh):
            counter +=1
    return counter
blueList = pd.read_csv("blue.txt",header = None)
redList = pd.read_csv("red.txt",header = None)

redList = redList.values
blueList = blueList.values

blueList = ((blueList))
redList = ((redList))
counter = 0
pool = Pool(processes=8)

thresh =0.5
far = []
frr = []
frrTrue = []
for x in range(35):
    lstDiv = redList.shape[0]//8

    retVal = pool.map(checkThresh,((thresh,redList[:lstDiv]),(thresh,redList[lstDiv:lstDiv*2]),(thresh,redList[lstDiv*2:lstDiv*3]),(thresh,redList[lstDiv*3:lstDiv*4]),(thresh,redList[lstDiv*4:lstDiv*5]),(thresh,redList[lstDiv*5:lstDiv*6]),(thresh,redList[lstDiv*6:lstDiv*7]),(thresh,redList[lstDiv*7:redList.shape[0]])))
    cumSum = 0
    print (blueList.shape)
    print (redList.shape)

    for x in range(len(retVal)):

        cumSum += retVal[x]
    print ("FAR " + str(cumSum/redList.shape[0]))
    far.append(cumSum/redList.shape[0])
    lstDiv = blueList.shape[0]//8
    retVal = pool.map(checkThreshT,((thresh,blueList[:lstDiv]),(thresh,blueList[lstDiv:lstDiv*2]),(thresh,blueList[lstDiv*2:lstDiv*3]),(thresh,blueList[lstDiv*3:lstDiv*4]),(thresh,blueList[lstDiv*4:lstDiv*5]),(thresh,blueList[lstDiv*5:lstDiv*6]),(thresh,blueList[lstDiv*6:lstDiv*7]),(thresh,blueList[lstDiv*7:blueList.shape[0]])))

    cumSum = 0
    for x in range(len(retVal)):
    #     print (retVal[x])
        cumSum += retVal[x]
    print ("FRR " + str(cumSum/blueList.shape[0]),thresh)
    frr.append( (cumSum/blueList.shape[0]))
    frrTrue.append(1 - cumSum/blueList.shape[0])
    thresh +=0.01
print (sorted(product(frrTrue, far), key=lambda t: abs(t[0]-t[1]))[0])
print (len(far))
print (far)
print (frrTrue)
plt.plot(far,color = 'r')
plt.plot(frr,color = 'b')
plt.xlabel("Threshold")
plt.ylabel("FAR/FRR")
print (auc(frr,far))
plt.show()
