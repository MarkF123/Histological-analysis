import openslide as wsi
import cv2 as cv
import numpy as  np
import os
os.environ["CUDA_VISIBLE_DEVICES"]='-1'

import SlicesDetector as sd

import SurveyTrainer as st
import SurveyTrainer_Tool as stt
import SurveyValidator as sv

import MesenteryDetector as md

import TrajectoryAnalyzer as ta
import HighResAnalyzer as hra


import matplotlib.pyplot as plt

import sklearn
import random as r

import time as t
from configobj import ConfigObj


class BRect:
    def __init__(self,cnt):
        self.x,self.y,self.w,self.h = cv.boundingRect(cnt)

class WsiSlicer:
    def __init__(self, wsiObjPath, mapZoom=8, surveyZoom=5, analZoom=0):
        self.DEBUG = False
        #####
        self.slidePath = wsiObjPath
        self.wsiObj = wsi.OpenSlide(wsiObjPath)
        self.levelsCount = self.wsiObj.level_count-1

        self.maxZoomWidth = self.wsiObj.level_dimensions[0][0]
        self.maxZoomHeight = self.wsiObj.level_dimensions[0][1]

        self.mapMpp = float(self.getMpp(mapZoom))
        self.surveyMpp = float(self.getMpp(surveyZoom))
        self.analMpp = float(self.getMpp(analZoom))

        self.mapZoom = mapZoom
        self.Map = []
        self.MapBW = []
        self.currentMapCoordinate = []
        
        self.createMaps()

        self.surveyZoom = surveyZoom
        self.surveyTileSize = (256,256)

        self.lineNumber = 1
        self.currentSurveyCoordinate = []
        
        self.analZoom = analZoom
        self.analTileSize = (256,256)

        self.objectContours = self.getObjectContours()
        self.objectRects = self.getObjectRects()
        self.objectsNumber = len(self.objectRects)

        for i in range(self.objectsNumber):
            print(i,":",self.objectRects[i].w*self.objectRects[i].h)

        if self.DEBUG:
            print('Objects number', self.objectsNumber)

        self.currentObjectInd = 0

        self.scales = self.calcScales(mapZoom,surveyZoom,analZoom)

    def getMpp(self, z):
        slideDirPath = os.path.dirname(self.slidePath)
        slideDirName = os.path.splitext(os.path.basename(self.slidePath))[0]
        print(os.path.join(slideDirPath,slideDirName,'Slidedat.ini'))
        #config.read( os.path.join(slideDirPath,slideDirName,'Slidedat.ini'),
        #             encoding="ansi" )

        config = ConfigObj(os.path.join(slideDirPath,slideDirName,'Slidedat.ini'))
        
        sectionName = 'LAYER_0_LEVEL_{}_SECTION'.format(z)
        return config[sectionName]['MICROMETER_PER_PIXEL_X']

    def calcScales(self,mapZ,survZ,anlZ):
        return {'sizeSurvey': pow(2,survZ),
                #'sizeAnal':   pow(2,mapZ-anlZ),
                'sizeAnal':   pow(2,anlZ),
                'location':   pow(2,mapZ)}

    def getMarkedMap(self):
        markedMap = self.Map.copy()
        for i,rect in enumerate(self.objectRects):
            area = rect.w*rect.h
            putText = "{}: {}".format(str(i),str(area))
            loc = (rect.x,rect.y)
            markedMap = cv.putText(markedMap,
                                   putText, loc, cv.FONT_HERSHEY_SIMPLEX,
                                   1, (0,255,0),
                                   2, cv.LINE_AA)

            markedMap = cv.circle(markedMap, loc, 5, (0,255,0), 2)
        return markedMap
            


    def filterObjects(self,rectsArr):
        filteredArr = []
        filteredCntArr = []
        #AREA_THR = (self.levelsCount-self.mapZoom+1)*1000
        if self.DEBUG:
            print('area thr',AREA_THR)
        areaArr = list([r.w*r.h for r in rectsArr])

        AREA_THR = np.sort(areaArr)[-2]//3
        for ind,rect in enumerate(rectsArr):
            if rect.w*rect.h >= AREA_THR:
              if self.DEBUG:
                  print(rect.w*rect.h)
              filteredArr.append( rect )
              filteredCntArr.append( self.objectContours[ind] )

        self.objectContours = filteredCntArr
        return filteredArr


    def getObjectContours(self):
        contours, hierarchy = cv.findContours(self.MapBW,
                                     cv.RETR_EXTERNAL,
                                     cv.CHAIN_APPROX_SIMPLE)
        return contours

    def getObjectRects(self):
        contours = self.objectContours
        
        objectsRects = []
        for cnt in contours:
            objectsRects.append( BRect(cnt) )

        return self.filterObjects(objectsRects)
        

    def createMaps(self):
        slicesDetector = sd.SlicesDetector(self.wsiObj, self.mapZoom , 'gaussAdapt')
        self.MapBW = slicesDetector.mapBinary
        self.Map = slicesDetector.mapImg
        self.Map = cv.cvtColor(self.Map, cv.COLOR_BGR2RGB)
        del slicesDetector

    def convertOpenSlideImg(self,im):
        im = np.array(im)
        return cv.cvtColor(im, cv.COLOR_BGRA2RGB)


    # public methods
    def iterOverSurvey(self, objInd):
        objRect = self.objectRects[objInd]
        tileWidth,tileHeight = self.surveyTileSize

        locScale = self.scales['location']
        xStep = tileWidth*self.scales['sizeSurvey']
        yStep = tileHeight*self.scales['sizeSurvey']

        yRange = range(objRect.y*locScale,
                           (objRect.y+objRect.h)*locScale,
                           yStep)
        xRange = range(objRect.x*locScale,
                           (objRect.x+objRect.w)*locScale,
                           xStep)
        ##
        if self.DEBUG:
            print('Locale scale',locScale)
            print('Size survey', self.scales['sizeSurvey'])

            print('X: from {} to {}, step = {}'.format(objRect.x*locScale,
                                                   (objRect.x+objRect.w)*locScale, xStep) )

            print('Y: from {} to {}, step = {}'.format(objRect.y*locScale,
                                                   (objRect.y+objRect.h)*locScale, yStep) )
        ##

        self.lineNumber = 1
        for yStart in yRange:
            for xStart in xRange:
                surveyTile = self.wsiObj.read_region( (xStart,yStart),
                                                      self.surveyZoom,
                                                      (tileWidth,tileHeight))
                surveyTile = self.convertOpenSlideImg(surveyTile)
                self.currentMapCoordinate = [int(xStart//locScale),
                                             int(yStart//locScale)]
                yield surveyTile
                
            self.lineNumber += 1


##################--- PyQt ---#######################

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget
# pyuic5 xyz.ui -o xyz.py

class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1350, 570)

        MainWindow.move(0,0)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(270, 20, 512, 512))
        self.label.setStyleSheet("background-color: rgb(168, 168, 168);")
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setEnabled(True)
        self.label_2.setGeometry(QtCore.QRect(810, 20, 512, 512))
        self.label_2.setStyleSheet("background-color: rgb(175, 175, 175);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(40, 50, 171, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setEnabled(False)
        
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(60, 460, 151, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setEnabled(False)
        
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setEnabled(False)
        self.checkBox.setGeometry(QtCore.QRect(40, 90, 161, 21))
        self.checkBox.setChecked(True)
        self.checkBox.setTristate(False)
        self.checkBox.setObjectName("checkBox")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1298, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(self.startFileDialog)

        self.WSIpath = ''

    def startFileDialog(self):
        print('start')
        pass

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Выбрать препарат"))
        self.pushButton_2.setText(_translate("MainWindow", "Галерея"))
        self.checkBox.setText(_translate("MainWindow", "Ручной выбор"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()


#######################################################
            

def generateUniqueID(size=8):
    s = ''
    for i in range(size):
        x = r.randint(0,9)
        s += str(x)
    return s


import pandas as pd


import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

initDir = r'C:\\'
file_path = filedialog.askopenfilename(initialdir=initDir)
# , filetypes=[('3DHistech virual slides','*.mrxs')]
if file_path == '':
    quit()
    
wsiPath = file_path


modelPath = r'ST_train_4.h5'

objectIndex = 0
mapWithMesentName = 'MapWithMesenterium_Demo2'

alphaMapName = 'aMap_Demo2'

#-Speed up-#
thinningRowOrder = 1 # 2

thinningColOrder = 1
counterCopy = 0
needCopy = False
############

mapZoom = 6
surveyZoom = 3 # 3 - orig
analyzeZoom = 0

slicer = WsiSlicer(wsiPath, mapZoom, surveyZoom, analyzeZoom)

surveyMatrix = (64,64)
analMatrix = (512,512)
#analMatrix = (surveyMatrix[0]*pow(2,surveyZoom-analyzeZoom) , surveyMatrix[1]*pow(2,surveyZoom-analyzeZoom))

slicer.surveyTileSize = surveyMatrix
slicer.analTileSize = analMatrix


MapBW = slicer.MapBW
Map = slicer.Map

mapToSurveyScale = slicer.scales['sizeSurvey']


mapForSelect = Map.copy()
alphaMap = Map.copy()
alphaMapAnl = Map.copy()
alphaMap_oneClass = Map.copy()

ind = 1
withClassNum = 0
uniqueID = generateUniqueID()


def processSelectionClick(event,x,y,flags,param):
    global objSelected
    global objectIndex
    if event == cv.EVENT_LBUTTONDOWN:
        x = int(x//selectScale)
        y = int(y//selectScale)
        for i,cnt in enumerate(slicer.objectContours):
            if cv.pointPolygonTest(cnt,(x,y),False)>=0:
                objectIndex = i
                objSelected = True
                break
        

objSelected = False       
selectScale = 0.25
selectWinName = 'Select object for analyze'

cv.drawContours(mapForSelect, slicer.objectContours, -1, (0,255,0), 2)
mapForSelect = cv.resize(mapForSelect, (0,0), fx=selectScale, fy=selectScale)
cv.namedWindow(selectWinName, cv.WINDOW_AUTOSIZE)
cv.moveWindow(selectWinName, 0,0)
cv.setMouseCallback(selectWinName, processSelectionClick)


while(1):
    cv.imshow(selectWinName, mapForSelect)
    cv.waitKey(5)
    if objSelected:
        cv.destroyWindow(selectWinName)
        break

totalArea_mkm = cv.contourArea(slicer.objectContours[objectIndex]) * slicer.mapMpp
print('Object with index {} was selected. Total area in mkm: {}'.format(objectIndex, totalArea_mkm))

needManualMarkUp = False

#######################
##-- MANUAL MARKUP --##
#######################

if needManualMarkUp:

    def getObjectFromMap(Map, objCoord):
        x,y = objCoord.x,objCoord.y
        w,h = objCoord.h,objCoord.h
        return x,y,Map[y:y+h,x:x+w]

    def getTileClass(tileCoord, classCoordinates):
        for singleClassCoord in classCoordinates:
            classID = singleClassCoord[0]
            coordinates = singleClassCoord[1]

            coordinates = np.array(coordinates).reshape((-1,1,2)).astype(np.int32)
            if cv.pointPolygonTest(coordinates, tileCoord, False) >= 0:
                return classID
        return -1
            
    
    numClasses = 6
    resizeCoeff = 1.5
    
    shX,shY,mapObject = getObjectFromMap(Map, slicer.objectRects[objectIndex] )
    
    markUpTool = stt.SurveyTrainer_Tool( mapObject, numClasses,
                                         scaleSize = resizeCoeff, scaleShift=(shX,shY) )
    cv.waitKey(0)
    
    realClassCoordinates = np.array( markUpTool.getRealCoordinates() )

    for i in range(markUpTool.classCounter):
        folderPath = 'testTiles/class_{}'.format(str(i))
        if not os.path.exists(folderPath):
            os.mkdir( folderPath )


if needManualMarkUp:
    for tile in slicer.iterOverSurvey( objectIndex ):
        classID = getTileClass(slicer.currentMapCoordinate,
                               realClassCoordinates)
        if classID != -1:
            withClassNum += 1

            fore = np.zeros((surveyMatrix[0], surveyMatrix[1], 3), dtype=np.uint8)
            fore = markUpTool.classColors[classID]

            alphaMap[slicer.currentMapCoordinate[1]:slicer.currentMapCoordinate[1]+surveyMatrix[1]//mapToSurveyScale,slicer.currentMapCoordinate[0]:slicer.currentMapCoordinate[0]+surveyMatrix[0]//mapToSurveyScale, :] = fore 
            cv.imwrite('testTiles/class_{}/{}_{}_{}.jpg'.format(str(classID),uniqueID,str(classID),str(ind)), tile)
        ind+=1
        if ind == -1:
            break

       
############################
##-- AUTOMATICAL MARKUP --##
############################


predictedNumber = 0
if not needManualMarkUp:

    cv.namedWindow('Drawing map...' , cv.WINDOW_GUI_NORMAL )
    
    def showDisplayDrawing(im,name='Drawing map...', delay=1):
        return None # no display
    
        def resizeTo(im, toW=512,toH=512):
            w,h = im.shape[:2]
            return toW/w, toH/h

        fy,fx = resizeTo(im)
        im = cv.resize(im, (0,0), fx=fx, fy=fy)

        cv.imshow(name, im)
        cv.moveWindow(name, 260, 30)
        cv.waitKey(delay)

        if cv.getWindowProperty(name, 0) == -1:
            cv.destroyWindow(name)
            quit()

    validator = sv.SurveyValidator(modelPath)

    classColors = [ [0, 0, 255],
                    [0, 255, 0],
                    [255, 0, 0], #color of empty tile
                    [255, 0, 115],
                    [115, 0, 255],
                    [115, 255, 0]]

    def isEmpty(im):
        b,g,r = cv.split(im)
        mB,mG,mR = np.mean(b),np.mean(g),np.mean(r)
        zHigh, zLow = 240, 20
        if mB>zHigh and mG>zHigh and mR>zHigh:
            return True
        if mB<zLow and mG<zLow and mR<zLow:
            return True
        return False


    def inObjectContour(point,cnt):
        if cv.pointPolygonTest(cnt,point,False) >= 0:
            return True
        return False

    def changeProbabilityColor(clr,prob):
        newClr = []
        for i in range(len(clr)):
            newClr.append(int(clr[i]*prob))
        return newClr


    startTime = t.time()

    objectContour = slicer.objectContours[ objectIndex ]
    objectRect = slicer.objectRects[ objectIndex ]
    
    firstLine = True # SURVEY ITERATION

    fore = np.zeros((surveyMatrix[0], surveyMatrix[1], 3), dtype=np.uint8)

    ############## - Display original image in the second label
    displayRect = alphaMap[objectRect.y:objectRect.y+objectRect.h,objectRect.x:objectRect.x+objectRect.w]
    
    def resizeTo(im, toW=512,toH=512):
        w,h = im.shape[:2]
        return toW/w, toH/h

    fy,fx = resizeTo(displayRect)
    displayRect = cv.resize(displayRect, (0,0), fx=fx, fy=fy)

    cv.namedWindow('Original image' , cv.WINDOW_GUI_NORMAL )
    cv.imshow('Original image', displayRect)
    cv.moveWindow('Original image', 800, 30)
    cv.waitKey(1)

    ##############

    tissueStartTime = t.time()
    
    for tile in slicer.iterOverSurvey( objectIndex ):
        #break # STOP
        
        if firstLine:
            slicer.lineNumber += thinningRowOrder-1
            firstLine = False
            
        mapCoord = slicer.currentMapCoordinate
        
        if slicer.lineNumber%thinningRowOrder != 0: # line duplication when thinningRowOrder > 1
            repeat = alphaMap[mapCoord[1]-surveyMatrix[1]//mapToSurveyScale:mapCoord[1] , mapCoord[0]:mapCoord[0]+surveyMatrix[0]//mapToSurveyScale, :]
            alphaMap[mapCoord[1]:mapCoord[1]+surveyMatrix[1]//mapToSurveyScale,mapCoord[0]:mapCoord[0]+surveyMatrix[0]//mapToSurveyScale, :] = repeat

            displayRect = alphaMap[objectRect.y:objectRect.y+objectRect.h,objectRect.x:objectRect.x+objectRect.w]
            showDisplayDrawing( displayRect )
            
            needCopy = False
            counterCopy = 0
            continue
        
        predictedNumber+=1

        insideCnt = inObjectContour(mapCoord,objectContour)

        if isEmpty(tile) or (not insideCnt):    # drawing empty fields
            fore = [255, 0, 0]
            
            if not insideCnt:
                fore = [0,0,0]

            alphaMap[mapCoord[1]:mapCoord[1]+surveyMatrix[1]//mapToSurveyScale,mapCoord[0]:mapCoord[0]+surveyMatrix[0]//mapToSurveyScale, :] = fore
            alphaMap_oneClass[mapCoord[1]:mapCoord[1]+surveyMatrix[1]//mapToSurveyScale,mapCoord[0]:mapCoord[0]+surveyMatrix[0]//mapToSurveyScale, :] = fore

            displayRect = alphaMap[objectRect.y:objectRect.y+objectRect.h,objectRect.x:objectRect.x+objectRect.w]
            showDisplayDrawing( displayRect )
            continue

        if needCopy:                            # col copying
            needCopy = False
            counterCopy = 0

            repeat = alphaMap[mapCoord[1]:mapCoord[1]+surveyMatrix[1]//mapToSurveyScale , mapCoord[0]-surveyMatrix[0]//mapToSurveyScale:mapCoord[0], :]
            alphaMap[mapCoord[1]:mapCoord[1]+surveyMatrix[1]//mapToSurveyScale,mapCoord[0]:mapCoord[0]+surveyMatrix[0]//mapToSurveyScale, :] = repeat

            displayRect = alphaMap[objectRect.y:objectRect.y+objectRect.h,objectRect.x:objectRect.x+objectRect.w]
            showDisplayDrawing( displayRect, delay=1 )
            
            continue

        if tile.shape[:2] != (64,64):
            tile = cv.resize(tile, (0,0), fx=64//tile.shape[1],fy=64//tile.shape[0])
        inputTile = tile/255.0
        mapClass, probability = validator.predict(inputTile) # probability - is experimental part
        
        withClassNum += 1
        
        fore = classColors[mapClass]
        #fore = changeProbabilityColor(fore, probability)

        if mapClass != 0:
            probability = 1-probability
            
        foreOC = changeProbabilityColor([0,0,255], probability)

        alphaMap[mapCoord[1]:mapCoord[1]+surveyMatrix[1]//mapToSurveyScale,mapCoord[0]:mapCoord[0]+surveyMatrix[0]//mapToSurveyScale, :] = fore
        # experimental
        alphaMap_oneClass[mapCoord[1]:mapCoord[1]+surveyMatrix[1]//mapToSurveyScale,mapCoord[0]:mapCoord[0]+surveyMatrix[0]//mapToSurveyScale, :] = foreOC

        displayRect = alphaMap[objectRect.y:objectRect.y+objectRect.h,objectRect.x:objectRect.x+objectRect.w]

        counterCopy += 1 # col copying
        if counterCopy == thinningColOrder -1:
            needCopy = True
        
        showDisplayDrawing( displayRect, delay=1 ) # delay for experiment

        #cv.imwrite('testTiles/allTiles/{}.jpg'.format(str(predictedNumber)), tile)

    #print('Total elapsed time: ', t.time()-startTime)
    
    #cv.imwrite('ORIG_MAP_smth1081755.png' , Map)
    #cv.imwrite('ALPHA_MAP_smth1081755.bmp' , alphaMap)
    #cv.imwrite('ALPA_MAP_smth1081755_OC.bmp' , alphaMap_oneClass)

    print( 'Tissue splitting time: ',t.time()-tissueStartTime )


def createPreAnalMap(frm, objRect, colorMtrx):
    obR = objRect
    mapTile = frm[obR.y:obR.y+obR.h, obR.x:obR.x+obR.w, :]
    
    preAnlMap = np.zeros((mapTile.shape[0],mapTile.shape[1]),dtype=np.uint8)
    
    for i in range(preAnlMap.shape[0]):
        for j in range(preAnlMap.shape[1]):
            preAnlMap[i,j] = colorMtrx[tuple(mapTile[i,j])]
    return preAnlMap


def addShift(point,x,y):
    return (point[0]+x,point[1]+y)

def calcDist(p1,p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def convertAnlToMap(anlCoord, backScale, shX,shY):
    X = anlCoord[0]//backScale
    Y = anlCoord[1]//backScale

    X -= shX
    Y -= shY
    
    return (int(X)+1,int(Y)+1)

def drawRect(im, coord, w,h, color=(0,255,0)):
    return cv.rectangle(im, (coord[0],coord[1]),
                            (coord[0]+h,coord[1]+w),
                            color,3)
    
##########################################################################


# UNCOMMENT IF SURVEY ITER WORKS


colorMatrix = {     (0,0,0)       : 0, # not in contour
                    (255,255,255) : 0,
                    (0,0,255)     : 1, # mucos tissue
                    (0,255,0)     : 2, # not mucos tissue
                    (255,0,0)     : 3} # empty field

preAnlMap = createPreAnalMap(alphaMap,
                             slicer.objectRects[objectIndex],
                             colorMatrix)


createMesent = True
if createMesent:
    mesentDetector = md.MesenteryDetector(preAnlMap, colorMatrix)


#cv.imwrite('testTiles/{}.bmp'.format('nonEmpty_1M01') , mesentDetector.nonEmptyMap)
#cv.imwrite('testTiles/{}.bmp'.format('Tissue_1M01') , mesentDetector.tissueMap)
#cv.imwrite('testTiles/{}.bmp'.format('Mucos_1M01') , mesentDetector.mucosMap)
#cv.imwrite('testTiles/{}.bmp'.format('notMucos_1M01') , mesentDetector.nonMucosMap)


if createMesent:
    if mesentDetector.mesenteriumAvailable:
        drawnMap = mesentDetector.drawnMap
        #cv.imwrite('testTiles/{}.bmp'.format(mapWithMesentName) , drawnMap)
        #cv.imshow('Mesentery', drawnMap)

        mucosMap = mesentDetector.mucosMap
        nonMucosMap = mesentDetector.nonMucosMap
        tissueMap = mesentDetector.tissueMap
        nonEmptyMap = mesentDetector.nonEmptyMap


# UNCOMMENT IF SURVEY ITER WORKS


# COMMENT IF SURVEY ITER WORKS
'''
mucosMap = cv.imread('testTiles/Mucos_1M01.bmp',0)
nonMucosMap = cv.imread('testTiles/notMucos_1M01.bmp',0)
tissueMap = cv.imread('testTiles/Tissue_1M01.bmp',0)
nonEmptyMap = cv.imread('testTiles/nonEmpty_1M01.bmp',0)
'''
# COMMENT IF SURVEY ITER WORKS


trajectAnl = ta.TrajectoryAnalyzer(mucosMap,
                                   nonMucosMap,
                                   tissueMap,
                                   nonEmptyMap,
                                   directNum = 8)



shX = slicer.objectRects[objectIndex].x
shY = slicer.objectRects[objectIndex].y


locScale = slicer.scales['location']
sizeScale = slicer.scales['sizeAnal']

HRAnl = hra.HighResAnalyzer('thresh' ,
                            slicer.analMpp ,
                            frmClassifier='framesInfiltClassifier_v2')

# Destroy older window
cv.destroyWindow('Drawing map...')


dfPath = 'trainDF.txt'
dfPath_debug = 'trainDF_debug.txt'

def addDFrow(path, params):
    with open(path, 'a', encoding='utf-8') as f:
        for i,p in enumerate(params):
            f.write(str(p))
            if i != len(params)-1:
                f.write(',')
        f.write('\n')
            

totalInd = 1
INFILT_COUNT_FRMS = 0

for fName in os.listdir(os.path.join(os.getcwd(),'gallery')):
    os.remove(os.path.join(os.getcwd(),'gallery',fName) )

for traj in trajectAnl.analTrajects:
    mapStartCoord = addShift(traj[0], shX, shY)
    mapEndCoord = addShift(traj[1], shX, shY)

    anlStartCoord = (mapStartCoord[0]*locScale,mapStartCoord[1]*locScale)
    anlEndCoord = (mapEndCoord[0]*locScale,mapEndCoord[1]*locScale)

    distance = calcDist(anlEndCoord,anlStartCoord)
    
    dirX = (anlEndCoord[0]-anlStartCoord[0])/distance
    dirY = (anlEndCoord[1]-anlStartCoord[1])/distance

    extraTrajPercent = 0.15 # was 0.15
    extraTraj = int(extraTrajPercent * distance//slicer.analTileSize[0])

    countRatioArr = []
    areaRatioArr = []

    #print('DIRECTION: ',dirX,dirY)
    for i in range(int(distance//slicer.analTileSize[0]) + extraTraj):
        currentAnlCoord = (int(anlStartCoord[0]+dirX*i*slicer.analTileSize[0]),
                           int(anlStartCoord[1]+dirY*i*slicer.analTileSize[1]))


        currentMapCoord = convertAnlToMap(currentAnlCoord,locScale,shX,shY)

        currentClass = trajectAnl.checkClass(currentMapCoord)
        
                           
        analTile = slicer.wsiObj.read_region( currentAnlCoord,
                                              slicer.analZoom,
                                              slicer.analTileSize)
        analTile = slicer.convertOpenSlideImg(analTile)

        displayAnalTile = analTile.copy()


        
        currentMapCoord_show = convertAnlToMap(currentAnlCoord,locScale,0,0)

        alphaMapAnl = drawRect(alphaMapAnl, currentMapCoord_show,
                               slicer.analTileSize[0]//(locScale),
                               slicer.analTileSize[1]//(locScale))

        
        displayRect = alphaMapAnl[objectRect.y:objectRect.y+objectRect.h,objectRect.x:objectRect.x+objectRect.w]
        showDisplayDrawing( displayRect, 'While analyzing', 5 )
        
        if currentClass < 3:
            #cv.imwrite('testTiles/HighRes/{}_{}/{}/{}.jpg'.format(dirX,dirY,currentClass,i+1), analTile)

            frameClass = HRAnl.analyze( analTile, currentClass )

            #countRatioArr.append( HRAnl.frameParams[0] )
            #sareaRatioArr.append( HRAnl.frameParams[1] )

            if frameClass == 1:
                INFILT_COUNT_FRMS += 1
                alphaMapAnl = drawRect(alphaMapAnl, currentMapCoord_show,
                               slicer.analTileSize[0]//(locScale),
                               slicer.analTileSize[1]//(locScale) , (0,0,255))
                
                displayRect = alphaMapAnl[objectRect.y:objectRect.y+objectRect.h,objectRect.x:objectRect.x+objectRect.w]
                showDisplayDrawing( displayRect, 'While analyzing', 5 )
                
                cv.imwrite('gallery/InfiltratedFrame_{}.jpg'.format(totalInd) , displayAnalTile)
                totalInd+=1
                
                #print('INFILTER ',HRAnl.frameParams)
                #print('*-----*')
            else:
                #print('--->',HRAnl.frameParams)
                pass

tk.messagebox.showinfo(title='Результат анализа', message='Обнаружено {} кадров с инфильтрацией. Число больше 20 указывает на высокую вероятность инфильтрации'.format(INFILT_COUNT_FRMS) )           
#filedialog.askdirectory(initialdir=os.path.join(os.getcwd(),'gallery'), title="Кадры с предполагаемой инфильтрацией")

folderToOpen = os.path.join(os.getcwd(),'gallery')
import subprocess
subprocess.Popen('explorer ' + folderToOpen)

    #print('Infiltrated frames number: ', INFILT_COUNT_FRMS)

    #plt.plot( range(len(countRatioArr)), countRatioArr, '.', color='r' )
    #plt.show()
    #plt.plot( range(len(areaRatioArr)), areaRatioArr, '.', color='b' )
    #plt.show()

    #break


