import cv2 as cv

import tensorflow as tf
#import tensorflow.compat.v1 as tf 

from tensorflow.keras import layers
from tensorflow.compat.v1.keras import layers

import pandas as pd
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D,BatchNormalization
#from tensorflow.compat.v1.keras.models import Sequential
#from tensorflow.compat.v1.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D,BatchNormalization




class SurveyTrainer:
    def __init__(self, inpShape, numClasses):
        self.inpShape = inpShape
        self.numClasses = numClasses
        self.trainEpochs = -1

        self.xTrain,self.yTrain = [],[]

        self.batchSize = 8

        self.model = None
        self.trainGenerator = None
        self.validGenerator = None
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def loadData_pandas(self, dataPath):
        colNames = ['frmPath', 'classID']
        data = pd.read_table(dataPath,
                             names = colNames,
                             sep = ',')
        for ind, row in data.iterrows():
            self.xTrain.append( cv.imread(row['frmPath']) )
            self.yTrain.append( int(row['classID']) )
        
        self.xTrain = np.array(self.xTrain)
        self.yTrain = np.array(self.yTrain)

    def loadData_folder(self, baseFolder):
        batch_size = self.batchSize
        
        trainDatagen = ImageDataGenerator(rescale= 1 / 255.0,
                                            rotation_range=20,
                                            zoom_range=0.05,
                                            width_shift_range=0.05,
                                            height_shift_range=0.05,
                                            shear_range=0.05,
                                            horizontal_flip=True,
                                            fill_mode="nearest",
                                            validation_split=0.20)

        testDatagen = ImageDataGenerator(rescale= 1 / 255.0)
        
        self.trainGenerator = trainDatagen.flow_from_directory(baseFolder,
                                                               target_size=(64,64),
                                                               color_mode="rgb",
                                                               class_mode="categorical",
                                                               subset='training',
                                                               batch_size=batch_size,
                                                               shuffle=True)

        self.validGenerator = testDatagen.flow_from_directory(baseFolder,
                                                              target_size=(64, 64),
                                                              color_mode="rgb",
                                                              batch_size=batch_size,
                                                              class_mode="categorical",
                                                              subset='validation',
                                                              shuffle=True)

        print('Class indices:', self.trainGenerator.class_indices )
        

    def createModel(self):
        '''
        model = tf.keras.models.Sequential([
          layers.InputLayer(input_shape = self.inpShape),
          #layers.Normalization(),
          layers.Conv2D(32, 3, padding='same', activation='relu'),
          layers.BatchNormalization(),
          layers.MaxPooling2D( (2,2) ),
          
          layers.Conv2D(64, 3, padding='same', activation='relu'),
          layers.BatchNormalization(),
          layers.MaxPooling2D( (2,2) ),
          
          layers.Conv2D(128, 3, padding='same', activation='relu'),
          layers.BatchNormalization(),
          layers.MaxPooling2D( (2,2) ),
          
          layers.Conv2D(128, 3, padding='same', activation='relu'),
          layers.BatchNormalization(),
          layers.MaxPooling2D( (2,2) ),

          layers.Conv2D(256, 3, padding='same', activation='relu'),
          layers.BatchNormalization(),
          layers.MaxPooling2D( (2,2) ),
          
          layers.Flatten(),
          layers.Dropout(0.5),
          layers.Dense(512, activation='relu'),
          layers.Dropout(0.5),
          layers.Dense(256, activation='relu'),
          
          layers.Dense(self.numClasses ,  activation='softmax')
        ])
        self.model = model
        '''

        AlexNet = Sequential()

        #1st Convolutional Layer
        AlexNet.add(Conv2D(filters=96, input_shape = self.inpShape, kernel_size=(11,11), strides=(4,4),
                           padding='same'))
        AlexNet.add(BatchNormalization())
        AlexNet.add(Activation('relu'))
        AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        #2nd Convolutional Layer
        AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
        AlexNet.add(BatchNormalization())
        AlexNet.add(Activation('relu'))
        AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        #3rd Convolutional Layer
        AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
        AlexNet.add(BatchNormalization())
        AlexNet.add(Activation('relu'))

        #4th Convolutional Layer
        AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
        AlexNet.add(BatchNormalization())
        AlexNet.add(Activation('relu'))

        #5th Convolutional Layer
        AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
        AlexNet.add(BatchNormalization())
        AlexNet.add(Activation('relu'))
        AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        #Passing it to a Fully Connected layer
        AlexNet.add(Flatten())
        # 1st Fully Connected Layer
        AlexNet.add(Dense(4096, input_shape=(32,32,3,)))
        AlexNet.add(BatchNormalization())
        AlexNet.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        AlexNet.add(Dropout(0.4))

        #2nd Fully Connected Layer
        AlexNet.add(Dense(4096))
        AlexNet.add(BatchNormalization())
        AlexNet.add(Activation('relu'))
        #Add Dropout
        AlexNet.add(Dropout(0.4))

        #3rd Fully Connected Layer
        AlexNet.add(Dense(1000))
        AlexNet.add(BatchNormalization())
        AlexNet.add(Activation('relu'))
        #Add Dropout
        AlexNet.add(Dropout(0.4))

        #Output Layer
        AlexNet.add(Dense(self.numClasses))
        AlexNet.add(BatchNormalization())
        AlexNet.add(Activation('softmax'))

        #Model Summary
        #AlexNet.summary()

        self.model = AlexNet

    def startTrain(self):
       self.model.compile(optimizer= tf.keras.optimizers.Adam(
                                                learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

       if self.trainEpochs > 1:

           if self.xTrain!=[] and self.yTrain!=[]:
               self.model.fit(self.xTrain,
                              self.yTrain,
                              epochs=self.trainEpochs)
           else:
               print('Train data size: ',self.trainGenerator.n )
               print('Test data size: ', self.validGenerator.n)

               print('Train Steps per epoch: ', self.trainGenerator.n//self.trainGenerator.batch_size)
               print('Validate Steps per epoch: ', self.validGenerator.n//self.validGenerator.batch_size)
               
               self.model.fit(self.trainGenerator,
                              validation_data = self.trainGenerator,
                              steps_per_epoch = self.trainGenerator.n//self.trainGenerator.batch_size,
                                        epochs = self.trainEpochs,
                                        verbose = 2 )

    def saveModel(self, modelPath):
        self.model.save(modelPath)


#################################################


needModelCreation = False


if needModelCreation:

    def pred(model,img):
        img = np.expand_dims(img, axis=0)
        return np.argmax( Model.predict(img) )


    numClasses = 2
    baseDir = r'testTiles/TRAIN4'

    trainer = SurveyTrainer( (64,64,3), numClasses )
    trainer.trainEpochs = 2

    trainer.createModel()

    trainer.loadData_folder( baseDir )
    trainer.startTrain()
    trainer.saveModel( 'testTiles/models/MODEL_AMD.h5' )



    Model = trainer.model



    predImg = cv.imread(r'C:\Python37\testTiles\TrainIter_1\class_0\38060999_0_377.jpg')
    res = pred(Model, predImg)
    print( 'Predict (0) class = ',res )

    predImg = cv.imread(r'C:\Python37\testTiles\TrainIter_1\class_1\38060999_1_983.jpg')
    res = pred(Model, predImg)
    print( 'Predict (1) class = ',res )

    predImg = cv.imread(r'C:\Python37\testTiles\TrainIter_1\class_2\38060999_2_365.jpg')
    res = pred(Model, predImg)
    print( 'Predict (2) class = ',res )



        

