import cv2 as cv
import numpy as np


class Ellipse:
    def __init__(self,cnt):
        el = cv.fitEllipse(cnt)
        
        self.cX = el[0][0]
        self.cY = el[0][1]
        self.w = el[1][0]
        self.h = el[1][1]
        self.angl = el[2]


class TrajectoryAnalyzer:
    def __init__(self, mucosMap,nonMucosMap,tissueMap,nonEmptyMap, directNum=4):
        self.directNum = directNum
        
        self.mucosMap = mucosMap
        self.nonMucosMap = nonMucosMap
        self.tissueMap = tissueMap
        self.nonEmptyMap = nonEmptyMap

        self.mucosMap = self.openClose(self.mucosMap)
        self.nonMucosMap = self.openClose(self.nonMucosMap)
        self.tissueMap = self.openClose(self.tissueMap)
        self.nonEmptyMap = self.openClose(self.nonEmptyMap)

        self.mucosContours = self.getContours(self.mucosMap)
        self.nonMucosContours = self.getContours(self.nonMucosMap)
        self.tissueContours = self.getContours(self.tissueMap)
        self.nonEmptyContours = self.getContours(self.nonEmptyMap)

        self.maxMucosCnt = self.getMaxCnt(self.mucosContours)
        self.maxNonMucosCnt = self.getMaxCnt(self.nonMucosContours)
        self.maxTissueCnt = self.getMaxCnt(self.tissueContours)
        self.maxNonEmptyCnt = self.getMaxCnt(self.nonEmptyContours)

        self.majorAxTissue,self.minorAxTissue = self.getEllipseAxes(self.maxTissueCnt)

        self.centerPointMucos = self.getCenterPoint(self.maxMucosCnt)
        self.outerPointsMucos = self.getOuterCoords(self.maxMucosCnt)
        self.outerPointsTissue = self.getOuterCoords(self.maxTissueCnt)
        self.outerPointsNonEmpty = self.getOuterCoords(self.maxNonEmptyCnt)

        
        self.analTrajects = self.formScanTrajects_2() # пока 4 штуки ортогональные

    def getEllipseAxes(self,cnt):
        el = cv.fitEllipse(cnt)
        return el[1]

    def multiPPT(self, cntArr, point, dist=False):
        for c in cntArr:
            if cv.pointPolygonTest(c,point,dist) >= 0:
                return True
        return False
    
    def checkClass(self,mapCoord):
        retClass = 0
        classesCntrs = [self.mucosContours,
                        self.nonMucosContours,
                        self.nonEmptyContours]
        for clsCntr in classesCntrs:
            if self.multiPPT(clsCntr,mapCoord,False):
                return retClass
            else:
                retClass+=1
        return retClass
            

    def openClose(self,map_):
        return map_
        '''
        kernel = np.ones((5,5),np.uint8)
        map_ = cv.morphologyEx(map_, cv.MORPH_OPEN, kernel)
        map_ = cv.morphologyEx(map_, cv.MORPH_CLOSE, kernel)
        return map_
        '''


    def formScanTrajects_2(self):
        trajectsArr = []

        startP = (self.centerPointMucos[0],self.centerPointMucos[1])
        anglsArr = np.linspace(0, 2*np.pi, self.directNum)
        for i,angl in enumerate(anglsArr):
            if i == len(anglsArr)-1: break;
            
            endPx = startP[0] + self.majorAxTissue*np.cos(angl)/2
            endPy = startP[1] + self.minorAxTissue*np.sin(angl)/2

            trajectsArr.append( (startP, ( int(endPx),int(endPy) )) )

        return trajectsArr
            

    def formScanTrajects(self):
        trajectsArr = []

        def getDirect(ind,centralP,outP):
            if ind==0: #left
                return (int(outP[ind]),int(centralP[1]))
            if ind==1: #top
                return (int(centralP[0]),int(outP[ind]))
            if ind==2: #right
                return (int(outP[ind]),int(centralP[0]))
            if ind==3: #bot
                return (int(centralP[1]),int(outP[ind]))


        def convertPoint(p):
            x,y = p
            if x<0: x=0;
            if y<0: y=0;
            return (x,y)

        startP = (self.centerPointMucos[0],self.centerPointMucos[1])
        for i in range(len(self.outerPointsTissue)):
            outCoord = getDirect(i,startP,self.outerPointsTissue)
            trajectsArr.append( (startP, convertPoint(outCoord)) )
        return trajectsArr
             
    def getContours(self,map_):
        return cv.findContours(map_,cv.RETR_LIST,
                                     cv.CHAIN_APPROX_SIMPLE)[0]

    def getMaxCnt(self,cntArr):
        return max(cntArr, key = cv.contourArea)

    def getOuterCoords(self,cnt):
        e = Ellipse(cnt)
        
        leftP = int(e.cX-e.w//2)
        topP = int(e.cY-e.h//2)
        rightP = int(e.cX+e.w//2)
        botP = int(e.cY+e.h//2)
        
        return (leftP,topP,rightP,botP)

    def getCenterPoint(self,cnt):
        e = Ellipse(cnt)
        return (int(e.cX),int(e.cY))

    def getCntCenter(self, cnt):
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return (cx,cy)


####################################################

'''
mMap = cv.imread('testTiles/Mucos_0033.bmp',0)
nmMap = cv.imread('testTiles/notMucos_0033.bmp',0)
tMap = cv.imread('testTiles/Tissue_0033.bmp',0)

ta = TrajectoryAnalyzer(mMap,nmMap,tMap)

imgShow = tMap.copy()
imgShow = cv.cvtColor(imgShow, cv.COLOR_GRAY2RGB)

ind=0
colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0)]
for l in ta.analTrajects:
    print(l)
    cv.line(imgShow, l[0], l[1], colors[ind], 1)
    ind+=1




cv.imwrite('testTiles/trAnl_0033.bmp' , imgShow)
'''



