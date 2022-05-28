import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



class MesenteryDetector:
    def __init__(self, preAnalMap, colorMtrx):
        self.map = preAnalMap

        self.emptyLabel = 0

        self.mucosLabel = 1
        self.nonMucosLabel = 2
        
        self.tissueLabels = [self.mucosLabel,
                             self.nonMucosLabel]
        
        self.nonEmptyMap = self.getNonEmptyMap()

        self.mucosMap = self.getMucosMap()
        self.nonMucosMap = self.getNonMucosMap()
        self.tissueMap = self.getTissueMap()

        self.appendixCnt = []
        self.mesenteryCnt = []
        
        self.drawnMap = []

        #self.calcMesenteryContour()

        self.mesenteriumAvailable = self.checkMesenteriumCondition()


    def checkMesenteriumCondition(self):
        return True
        # check 'white color' portion
        # check square


    def calcMesenteryContour(self):
        contArr, _ = cv.findContours(self.nonEmptyMap,
                                     cv.RETR_LIST,
                                     cv.CHAIN_APPROX_SIMPLE)
        contMain = max(contArr, key = cv.contourArea)

        hull = cv.convexHull(contMain, returnPoints = False)
        defects = cv.convexityDefects(contMain, hull)

        linesLengthArr = []
        pointsArr = []
        for i in range(defects.shape[0]):
           s,e,f,d = defects[i,0]
           start = tuple(contMain[s][0])
           end = tuple(contMain[e][0])
            
           far = tuple(contMain[f][0])

           linesLengthArr.append( self.calcDist(end,start) )
           pointsArr.append( far )

        twoBiggestIndices = self.findKLargest(linesLengthArr, 2, True)
        mesentBasePoint = np.array(pointsArr)[ twoBiggestIndices ]

        p1,p2 = mesentBasePoint
        cnt1,cnt2 = self.splitContour( contMain, p1,p2 )

        cnt1 = self.convertPoints2Contour(cnt1)
        cnt2 = self.convertPoints2Contour(cnt2)
    
        self.appendixCnt = cnt1
        self.mesenteryCnt = cnt2
        
        self.drawnMap = self.getDrawnMap()


    def getDrawnMap(self):
        splitedMap = self.nonEmptyMap.copy()
        splitedMap = cv.cvtColor(splitedMap, cv.COLOR_GRAY2RGB)

        cv.drawContours(splitedMap, [self.appendixCnt], 0, (0,255,0), 3)
        cv.drawContours(splitedMap, [self.mesenteryCnt], 0, (0,0,255), 3)

        return splitedMap
        
    def getLabelMap(self, M, lbls=[], needInvert=False):
        lbls = np.sort(lbls)
        M = cv.inRange(M, np.array(lbls[0]), np.array(lbls[-1]))
        if needInvert:
            M = cv.bitwise_not(M)
        return M

    def getNonEmptyMap(self):
        neMap = self.map.copy()
        return self.getLabelMap(neMap, [self.emptyLabel], True)

    def getMucosMap(self):
        mMap = self.map.copy()
        return self.getLabelMap(mMap, [self.mucosLabel])

    def getNonMucosMap(self):
        nmMap = self.map.copy()
        return self.getLabelMap(nmMap, [self.nonMucosLabel])

    def getTissueMap(self):
        tMap = self.map.copy()
        return self.getLabelMap(tMap, [self.mucosLabel,self.nonMucosLabel])

    def calcDist(self,p1,p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    def findKLargest(self,x,K, returnIndices=False):
        if not returnIndices: #return values
            return np.array(x)[ np.argsort(x)[-K:] ]
        else:
            return np.argsort(x)[-K:]

    def convertPoints2Contour(self,points):
        return np.array(points).reshape((-1,1,2)).astype(np.int32)

    def splitContour(self,cnt, p1,p2):

        def comparePoints(p1_,p2_):
            if (p1_[0]==p2_[0]) and (p1_[1]==p2_[1]):
                return True
            return False

        pointsCommonBuff = []
        firstCnt = []
        firstCnt_2 = []
        secondCnt = []

        firstFoundPoint = []
        secondPoint = []
        for cntPoint in cnt:
            cntPoint = cntPoint[0]

            if comparePoints(cntPoint,p1):
                firstFoundPoint = p1
                secondPoint = p2
                break
            if comparePoints(cntPoint,p2):
                firstFoundPoint = p2
                secondPoint = p1
                break

        fillingFirstCnt = True
        fillingSecondCnt = True
        
        for cntPoint in cnt:
            cntPoint = cntPoint[0]
            
            if fillingFirstCnt:
                if not comparePoints(cntPoint,firstFoundPoint):
                    firstCnt.append( cntPoint )
                else:
                    fillingFirstCnt = False
                    firstCnt.append( cntPoint )
            else:
                if fillingSecondCnt:
                    if not comparePoints(cntPoint,secondPoint):
                        secondCnt.append( cntPoint )
                    else:
                        fillingSecondCnt = False
                        secondCnt.append( cntPoint )
                else:
                    firstCnt_2.append( cntPoint )
        
        firstCnt = np.concatenate((firstCnt_2,firstCnt))
        firstCnt = np.concatenate((firstCnt,[secondCnt[0]]))

        return firstCnt, secondCnt



##################




'''

preAnalMap = cv.imread('testTiles/preAnlMap.bmp',0)

colorMatrix = {     (0,0,0)       : 0, # not in contour
                    (255,255,255) : 0,
                    (0,0,255)     : 1, # mucos tissue
                    (0,255,0)     : 2, # not mucos tissue
                    (255,0,0)     : 3} # empty field

mesentDetector = MesenteryDetector( preAnalMap, colorMatrix )

neMap = mesentDetector.nonEmptyMap
mMap = mesentDetector.mucosMap
nmMap = mesentDetector.nonMucosMap
tMap = mesentDetector.tissueMap


#plt.plot( range(len(linesLengthArr)), linesLengthArr, '.', color='r' )
#plt.show()
'''

'''
cv.imwrite('testTiles/{}.bmp'.format(str( 1 )) , neMap)
cv.imwrite('testTiles/{}.bmp'.format(str( 2 )) , mMap)
cv.imwrite('testTiles/{}.bmp'.format(str( 3 )) , nmMap)
cv.imwrite('testTiles/{}.bmp'.format(str( 4 )) , tMap)
'''





