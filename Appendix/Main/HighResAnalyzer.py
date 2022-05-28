import cv2 as cv
import numpy as np

from skimage import segmentation as segmLib
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import imutils
import pickle

import RegionsSelector_v6 as rs

import os
import sklearn


class HighResAnalyzer:
    def __init__(self, adaptorType='dist', mpp=0.25, nuclModel='nuclClassifier_v2', frmClassifier='framesInfiltClassifier_v1'):
        self.inputFrm = []
        self.nuclContours = []

        self.critValue = 0.7
        self.adaptorType = adaptorType # dist, thresh
        
        self.nuclModelPath = nuclModel
        self.nuclModel = self.loadModel(self.nuclModelPath)

        self.debugFrm = []
        self.debugMask = []

        self.refHColor = [131,42,22]

        self.frmSquare = 0
        self.meanRBg = 0
        self.stdRBg = 0

        self.propR = 0
        self.propE = 0
        self.REareaRatio = 0

        self.frameParams = []

        self.frameModel = self.loadModel(frmClassifier)

        self.adaptorFunc = None
        if self.adaptorType == 'dist':
            self.adaptorFunc = self.colorDistanceAdaptor
        if self.adaptorType == 'thresh':
            self.adaptorFunc = self.colorThresholdAdaptor

        self.emptyFieldThreshold = 225
        self.MIN_AREA_THRESHOLD_mkm = 16
        self.MAX_AREA_THRESHOLD_mkm = 250
        
        self.MPP = mpp
        

    def calcCntrParams(self,cntr):
        def getMinMajAxdummy(cnt):
            xMin = tuple(cnt[cnt[:,:,0].argmin()][0])[0]
            xMax = tuple(cnt[cnt[:,:,0].argmax()][0])[0]
            yMin = tuple(cnt[cnt[:,:,1].argmin()][0])[1]
            yMax = tuple(cnt[cnt[:,:,1].argmax()][0])[1]

            distX,distY = xMax-xMin,yMax-yMin
            if distX > distY:
                return (distY,distX)
            else:
                return (distY,distX)
                
        def func_1(c): # компактность
            area = cv.contourArea(c)
            perim = cv.arcLength(c, closed=True)
            return 4*np.pi*area/perim

        def func_2(c): # round-factor
            if len(c) > 4:
                majorAx,minorAx = cv.fitEllipse(c)[1]
            else:
                majorAx,minorAx = getMinMajAxdummy(c)
            
            return 2*np.sqrt(majorAx*minorAx)/(majorAx+minorAx)

        def func_3(c): # округлость
            area = cv.contourArea(c)
            if len(c) > 5:
                majorAx = cv.fitEllipse(c)[1][0]
            else:
                majorAx = getMinMajAxdummy(c)[1]

                if majorAx <= 0:
                    return 1
            return 4*area/(np.pi*majorAx*majorAx)

        p1 = func_1(cntr)
        p2 = func_2(cntr)
        p3 = func_3(cntr)

        return [p1,p2,p3]

    def filterContours(self,cntrs):
        filteredCntrs = []
        for c in cntrs:
            area = cv.contourArea(c)
            if (area < 2000) and (area > 5):
                filteredCntrs.append(c)
        return filteredCntrs

    def getChannelHist(self, im, ch):
        if ch=='b':
            chInd = [0]
        if ch=='r':
            chInd = [2]
        return cv.calcHist([im],chInd,None,[256],[0,256])

    def colorThresholdAdaptor(self,frm):
        def getNonEmptyIndices(colorH):
            indArr=[]
            for i,valH in enumerate(colorH):
                if valH != 0:
                    indArr.append(i)
            return indArr

        def intersectionPercent(colorX,colorYIndices):
            intersecNumber = 0
            for i,cX in enumerate(colorX):
                if (i in colorYIndices) and (cX != 0):
                    intersecNumber+=1
            return intersecNumber/len(colorX)

        def colorThr(im, rVal, returnColorImg=True):
            lowerRegion = np.array([0, 0, 0],np.uint8)
            upperRegion = np.array([255, 255, rVal],np.uint8)

            thresh = cv.inRange(im, lowerRegion, upperRegion)
            if returnColorImg:
                thresh = cv.bitwise_and(im,im,mask=thresh)
            return thresh

        # BEGIN

        #return colorThr(frm, 140, False) # DEBUG !!!
        
        blueH = self.getChannelHist(frm, 'b')
        redH = self.getChannelHist(frm, 'r')
        NEIndcsBlue = getNonEmptyIndices(blueH)
        intersectPercent = intersectionPercent(redH, NEIndcsBlue)

        for rVal in range(255,0,-5):
            thrImg = colorThr(frm, rVal)
            redH = self.getChannelHist(thrImg, 'r')
            IP = intersectionPercent(redH,NEIndcsBlue)
            if IP <= 0.25:
                return colorThr(frm, rVal, False)
        return colorThr(frm, rVal, False)
        # END


    def colorDistanceAdaptor(self, img):
        
        def getDistanceMap(im,color):
            b,g,r = im[:,:,0],im[:,:,1],im[:,:,2]
            bC,gC,rC = color

            b,g,r = b.astype(int),g.astype(int),r.astype(int)

            b = np.array(list(map(lambda x: (x-bC)**2, b)))
            g = np.array(list(map(lambda x: (x-gC)**2, g)))
            r = np.array(list(map(lambda x: (x-rC)**2, r)))

            res =  np.array(list(map(lambda x,y,z: (x+y+z)**.3, b,g,r)))
            res = res.astype(np.uint8)
            return cv.bitwise_not(res)
        

        dMap = getDistanceMap(img, self.refHColor)
        _, thr = cv.threshold(dMap,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        return thr

        
    def searchNuclei(self,frm):
        def convertLablesToContours(labels):
            cntrsResult = []
            for lbl in np.unique(labels):
                if lbl==0:
                    continue

                mask = np.zeros(labels.shape, dtype='uint8')
                mask[labels==lbl] = 255

                cntrs = cv.findContours(mask.copy(),
                                        cv.RETR_EXTERNAL,
                                        cv.CHAIN_APPROX_SIMPLE)
                cntrs = imutils.grab_contours(cntrs)
                c = max(cntrs, key=cv.contourArea)

                hull = cv.convexHull(c)
                cntrsResult.append(hull)
            return cntrsResult


        frm = cv.medianBlur(frm, 5)
        nuclMask = self.adaptorFunc(frm)

        distTransform = cv.distanceTransform(nuclMask, cv.DIST_L2,3)
        localMax = peak_local_max(distTransform,
                                  min_distance = 15,
                                  indices=False)

        markers, _ = ndi.label(localMax)

        segmented = segmLib.watershed(255 - distTransform,
                                      markers,
                                      mask = nuclMask)
        return convertLablesToContours(segmented)

    def loadModel(self,modelPath):
        return pickle.load(open(modelPath, 'rb'))

    def predict_(self, p):
        return self.nuclModel.predict( [p] )
        
    def predictNuclClass(self,params):
        return self.predict_(params)

    def predictFrameType(self, nuclsClassesList):
        values, counts = np.unique(nuclsClassesList,
                                   return_counts=True)

        #print('{} cell: {}, {} cell: {}'.format(values[0],counts[0],values[1],counts[1]))
        biggerCount,smallerCount = max(counts),min(counts)
        roundNuclCount,ellipsNuclCount = counts
        
        proportionRound = roundNuclCount/(roundNuclCount+ellipsNuclCount)
        proportionEll = ellipsNuclCount/(roundNuclCount+ellipsNuclCount)
        #print('Proportion round: ', proportionRound)
        #print('Proportion ell: ', proportionEll)

        self.propR = proportionRound
        self.propE = proportionEll
        
        if proportionRound > self.critValue or proportionEll > self.critValue:
            return 0
        else:
            return 1

    def createMask(self,origFrm,cntrs):
        mask = np.zeros((origFrm.shape[0],origFrm.shape[1]), dtype=np.uint8)
        for c in cntrs:
            mask = cv.drawContours(mask, [c],-1,
                                   255, -1)
        return mask

    def getRChBackground(self):
        backgroundMask = cv.bitwise_not(self.debugMask)
        backgroundFrm = cv.bitwise_and(self.inputFrm,
                                       self.inputFrm,
                                       mask = backgroundMask)

        b,g,r = cv.split(backgroundFrm)

        b,r = np.ravel(b), np.ravel(r)
        b,r = cv.findNonZero(b),cv.findNonZero(r)

        #RChannel = self.getChannelHist(backgroundFrm, 'r')
        #print('blue', np.mean(b), np.std(b))
        return np.mean(r), np.std(r)

    def predictFrameType_2(self, params):
        return self.frameModel.predict([params])

    def predictFrameType_3(self, params):
        countRratio,areaRratio,survClass = params

        if survClass == 0:
            if countRratio >= 2 and areaRratio > 2: # mucos
                # count was >= 3
                return 0

            if (countRratio < 1.6 and areaRratio < 1.6) or (countRratio > 0.7 and areaRratio > 0.7):
                if areaRratio < 0.9: # false survClass
                    return 0
                else:
                    return 1 # infilter

            return 0

        if survClass == 1:
            res = -1
            '''
            if (countRratio < 3 and areaRratio < 2) or (countRratio >= 3 and areaRratio > 1.5): # muscul
                return 0
            '''
            
            
            if countRratio < 3 and areaRratio < 2: # muscul
                #return 0
                res = 0
                
            if countRratio >= 3 and areaRratio > 1.5: # muscul(tangetal) or serose
                # areaRatio was > 2
                #return 0
                res = 0

            if countRratio < 0.6 and areaRratio < 0.6: # pure muscul
                return 0

            if areaRratio < 0.7:
                return 0

            res = 1

            #print( 'Count: ',countRratio, '; Area: ', areaRatio )
            return res


    def areaFilter(self,cnt):
        area = cv.contourArea(cnt)
        area_mkm = area*self.MPP

        if (area_mkm > self.MIN_AREA_THRESHOLD_mkm) and (area_mkm < self.MAX_AREA_THRESHOLD_mkm):
            return True
        return False
        

    def analyze(self, inpFrm, surveyClass):
        if np.mean(inpFrm) >= self.emptyFieldThreshold: # check for empty
            return -1
                
        self.debugFrm = inpFrm
        
        self.frmSquare = inpFrm.shape[0]*inpFrm.shape[1]
        self.inputFrm = inpFrm
        colors = [(0,0,255),
                  (0,255,0)]

        roundArea = 0
        ellipsArea = 0

        self.nuclContours = self.searchNuclei(inpFrm)
        self.nuclContours = self.filterContours(self.nuclContours)

        nuclsList = []
        for i,cnt in enumerate(self.nuclContours):
            if not self.areaFilter(cnt):
                continue
            
            paramsList = self.calcCntrParams(cnt) 
            nuclClass = self.predictNuclClass(paramsList)[0]
            nuclsList.append( nuclClass )

            if nuclClass == 1:
                roundArea += cv.contourArea(self.nuclContours[i])/self.frmSquare
            if nuclClass == 2:
                ellipsArea += cv.contourArea(self.nuclContours[i])/self.frmSquare

            cv.drawContours(self.debugFrm,
                            [self.nuclContours[i]], -1,
                            colors[nuclClass-1], -1)


        #self.debugMask = self.createMask(inpFrm,self.nuclContours)
            
        a,b = np.unique(nuclsList, return_counts=True )
        #print( 'Cells: ',a,b)

        try:
            self.predictFrameType( nuclsList )

            if ellipsArea != 0:
                self.REareaRatio = roundArea/ellipsArea
            else:
                self.REareaRatio = roundArea/(1/self.frmSquare)

            if self.propE != 0:
                countProp = self.propR/self.propE
            else:
                countProp = 1.0

            frmParams = [countProp,
                         self.REareaRatio, surveyClass]

            #print('Count proportion: ',countProp)
            #print('Area proportion: ',self.REareaRatio)

            self.frameParams = frmParams
            
            return self.predictFrameType_3( frmParams )
        except:
            return -1
        
        # 0 - uniform
        # 1 - disproportion (infeltrated)
        # -1 - error



##################################################################

'''
hrAnal = HighResAnalyzer('thresh')
img = cv.imread(r'C:\Python37\gallery\InfiltratedFrame_6.jpg')
res = hrAnal.analyze(img, 1)
print(res)

cv.imwrite(r'MASK.jpg', hrAnal.debugFrm)
'''










