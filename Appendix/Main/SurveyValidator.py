import tensorflow as tf
import cv2 as cv
import numpy as np
from tensorflow import keras
import os
import time as t

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
print( '=== GPU ===' )
print( tf.__version__ )


class SurveyValidator:
    def __init__(self, modelPath):
        self.model = keras.models.load_model(modelPath)
        '''
        if modelPath[-3:]=='.h5':
            print("h5 model")
            self.model = keras.models.load_model(modelPath)
        else:
            self.model = tf.saved_model.load(modelPath)
        '''


    def img2tensor(self,img):
        return np.expand_dims(img, axis=0)

    def predict(self, img):
        singleTensor = self.img2tensor(img)
        probablityArr = self.model.predict(singleTensor)
        argMaximum = np.argmax(probablityArr)
        return argMaximum, probablityArr[0][argMaximum]


def pred(model,img):
    img = np.expand_dims(img, axis=0)
    return np.argmax( model.predict(img) )



createMatrix = False

if createMatrix:

    modelPath = r'testTiles/models/MODEL_AMD.h5'
    model = keras.models.load_model(modelPath)
    #model = tf.saved_model.load(modelPath)

    errorMatrix = []

    dirPath = r'C:\Python37\testTiles\TrainIters\TrainIter_2\class_0'
    ind = 0
    truePredict = 0
    trueClass = 0

    matrixRow = []
    class_0 = 0
    class_1 = 0
    class_2 = 0

    startTime = t.time()
    for imName in os.listdir(dirPath):
        imgPath = os.path.join(dirPath,imName)
        img = cv.imread(imgPath)
        img = img/255.0
        res = pred( model , img )
        ind += 1

        if trueClass == res:
            truePredict += 1

        if res == 0:
            class_0 += 1
        if res == 1:
            class_1 += 1
        if res == 2:
            class_2 += 1

    print('Elapsed time: ', t.time()-startTime)

    matrixRow.append( [class_0,class_1,class_2] )

    print( truePredict )
    print( ind )
    print( 'Accurace in 0 class: ', truePredict/ind )
    errorMatrix.append( matrixRow )


    ##################


    print('********')
    dirPath = r'C:\Python37\testTiles\TrainIters\TrainIter_2\class_1'
    ind = 0
    truePredict = 0
    trueClass = 1
    matrixRow = []
    class_0 = 0
    class_1 = 0
    class_2 = 0

    startTime = t.time()
    for imName in os.listdir(dirPath):
        imgPath = os.path.join(dirPath,imName)
        img = cv.imread(imgPath)
        img = img/255.0
        res = pred( model , img )
        ind += 1

        if trueClass == res:
            truePredict += 1

        if res == 0:
            class_0 += 1
        if res == 1:
            class_1 += 1
        if res == 2:
            class_2 += 1

    print('Elapsed time: ', t.time()-startTime)

    matrixRow.append( [class_0,class_1,class_2] )

    print( truePredict )
    print( ind )
    print( 'Accurace in 1 class: ', truePredict/ind )
    errorMatrix.append( matrixRow )


    ###############################

    print('********')
    dirPath = r'C:\Python37\testTiles\TrainIters\TrainIter_2\class_2'
    ind = 0
    truePredict = 0
    trueClass = 2
    matrixRow = []
    class_0 = 0
    class_1 = 0
    class_2 = 0

    startTime = t.time()
    for imName in os.listdir(dirPath):
        imgPath = os.path.join(dirPath,imName)
        img = cv.imread(imgPath)
        img = img/255.0
        res = pred( model , img )
        ind += 1

        if trueClass == res:
            truePredict += 1

        if res == 0:
            class_0 += 1
        if res == 1:
            class_1 += 1
        if res == 2:
            class_2 += 1

    print('Elapsed time: ', t.time()-startTime)

    matrixRow.append( [class_0,class_1,class_2] )
        
    print( truePredict )
    print( ind )
    print( 'Accurace in 2 class: ', truePredict/ind )
    errorMatrix.append( matrixRow )

    print( errorMatrix  )









