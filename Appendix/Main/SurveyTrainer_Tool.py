import cv2 as cv
import numpy as np



class SurveyTrainer_Tool:
    def __init__(self, mapImg, numClasses, scaleSize=1, scaleShift=(0,0)):
        self.Map = mapImg
        if scaleSize != 1:
            self.Map = cv.resize(self.Map,
                                 (0,0), fx=scaleSize,fy=scaleSize)
        self.numClasses = numClasses
        self.classColors = [(0, 0, 255),
                            (0, 255, 0),
                            (255, 0, 0),
                            (255, 0, 115),
                            (115, 0, 255),
                            (115, 255, 0)]

        self.scaleSize = scaleSize
        self.scaleShift = scaleShift
        
        self.drawingMap = self.Map.copy()
        self.classCounter = 0
        
        self.regionPoints_noClass = []
        self.regionPoints_withClass = []
        self.previousPoints = []
        self.clickedPoints = []
        self.newClassDrawing = True

        cv.namedWindow("Trainer tool", cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback("Trainer tool", self.processClick)
        cv.imshow("Trainer tool" , self.Map)

        
    def processClick(self, event, x,y, flag,param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.regionPoints_noClass.append( [x,y] )
            self.clickedPoints.append([x,y])
            self.newClassDrawing = True
            self.updateCanvas()
        if event == cv.EVENT_RBUTTONDOWN:
            if self.regionPoints_withClass != []:
                self.regionPoints_noClass.append( self.regionPoints_noClass[0] )
                self.regionPoints_noClass = np.concatenate((self.regionPoints_noClass,
                                                            self.previousPoints))
            
            self.previousPoints = self.clickedPoints
            
            self.clickedPoints = []
            
            self.regionPoints_withClass.append( [self.classCounter,
                                                 self.regionPoints_noClass])
            self.classCounter += 1
            self.regionPoints_noClass = []
            self.newClassDrawing = False
            self.updateCanvas()

    def updateCanvas(self):
        self.drawingMap = self.Map.copy()
        if self.newClassDrawing:
            color = self.classColors[self.classCounter]
            for regPoint in self.regionPoints_noClass:
                cv.circle( self.drawingMap,
                                regPoint, 2 ,color,-1)
            if self.regionPoints_withClass != []:
               for regPoint_wc in self.regionPoints_withClass:
                    classInd = regPoint_wc[0]
                    points = regPoint_wc[1]
                    color = self.classColors[classInd]
                    points = np.array(points)
                    cv.fillPoly(self.drawingMap, [points], color) 
        else:
            for regPoint_wc in self.regionPoints_withClass:
                classInd = regPoint_wc[0]
                points = regPoint_wc[1]
                color = self.classColors[classInd]
                points = np.array(points)
                cv.fillPoly(self.drawingMap, [points], color)

                
        cv.imshow( "Trainer tool" , self.drawingMap )


    def getRealCoordinates(self):
        if (self.scaleShift[0]*self.scaleShift[1] !=0) or self.scaleSize !=1:
            shiftX,shiftY = self.scaleShift[0],self.scaleShift[1]
            scale = self.scaleSize
            newClassPoints = []
            for regPoint_wc in self.regionPoints_withClass:
                classInd = regPoint_wc[0]
                points = regPoint_wc[1]
                newPoints = []
                for p in points:
                    newPoint = [ int( (p[0]//scale) + shiftX),
                                 int( (p[1]//scale) + shiftY)]
                    newPoints.append(newPoint)
                newClassPoints.append( [classInd,newPoints] )
                
            return newClassPoints
        else:
            return self.regionPoints_withClass


#################################


















        
        
            
