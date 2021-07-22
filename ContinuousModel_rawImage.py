import numpy as np
import pandas as pd
from tensorflow import keras as kr
import matplotlib.pyplot as pl
import os
import cv2
import copy
from copy import deepcopy


def train_test_split(images, subParas, labels, trainRatio=0.99):
    amount = labels.shape[0]
    print(amount)
    array = np.arange(amount)
    trainIndices = np.random.permutation(array)[:int(amount*trainRatio)].tolist()
    testIndices = np.random.permutation(array)[-int(amount*trainRatio):].tolist()
    return images[trainIndices, :], subParas[trainIndices, :], labels[trainIndices, :], images[testIndices, :], subParas[testIndices, :], labels[testIndices, :]



class DeepModel():
    def __init__(self):
        # self.traceDataDirPath = '../data/trace1'
        self.traceDataDirPath = '../trace2_data'
        # load model
        self.modelPath = './continuousModel_rawImage.h5'
        self.model = self.__loadModel()


    # MARK: - Public Method

    def run(self):
        self.model.summary()
        # images, subParas, labels = self.__loadAllTrainingDataSet()
        # for images, subParas, labels in self.__loadOneTrainingDataSet():
        #     images_train, subParas_train, labels_train, images_test, subParas_test, labels_test = train_test_split(images, subParas, labels, trainRatio=0.9)
        #     # print(images_train.shape)
        #     # print(subParas_train.shape)
        #     # print(labels_train.shape)
        #     self.model.fit(
        #         [images_train, subParas_train],
        #         # images_train,
        #         labels_train,
        #         batch_size=16,
        #         epochs=2,
        #         validation_split=0.1
        #     )
        #     print("Evaluate on test data")
        #     results = self.model.evaluate([images_test, subParas_test], labels_test, batch_size=16)
        #     print("test loss, test acc:", results)
        #     self.model.save(self.modelPath)

        for _ in range(2000):
            self.model.fit(
                self.__loadOneTrainingDataSetForGenerator(batch_size=128),
                steps_per_epoch=500,
                epochs=10
                # validation_data=self.__loadOneTrainingDataSetForGenerator(batch_size=128),
                # validation_steps=10,
            )
            self.model.save(self.modelPath)


    # MARK: - Private Method

    def __loadModel(self):
        if os.path.exists(self.modelPath):
            return kr.models.load_model(self.modelPath)
        # if os.path.exists('continuousModel_succeed.h5'):
        #     return kr.models.load_model('continuousModel_succeed.h5')
        else:
            image_inputs = kr.layers.Input(shape=(160, 320, 3), dtype=np.float, name='image')
            image_layer = kr.layers.Conv2D(filters=128, kernel_size=4, strides=(2, 2), activation='relu', name='image_conv1')(image_inputs)
            image_layer = kr.layers.MaxPooling2D(pool_size=(4,4))(image_layer)
            image_layer = kr.layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1), activation='relu', name='image_conv2')(image_layer)
            image_layer = kr.layers.MaxPooling2D(pool_size=(4,4))(image_layer)
            image_layer = kr.layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1), activation='relu', name='image_conv3')(image_layer)
            image_layer = kr.layers.MaxPooling2D(pool_size=(2,2))(image_layer)
            # image_layer = kr.layers.Conv2D(filters=128, kernel_size=2, strides=(1, 1), activation='relu', name='image_conv4')(image_layer)
            # image_layer = kr.layers.MaxPooling2D(pool_size=(2,2))(image_layer)
            image_layer = kr.layers.Flatten(name='flattened')(image_layer)
            # image_layer = kr.layers.Dense(256, activation='relu', name='image_dense1')(image_layer)
            image_layer = kr.layers.Dense(256, activation='relu', name='image_dense2')(image_layer)
            image_layer = kr.layers.Dense(1, activation='tanh', name='image_dense3')(image_layer)

            # subPara_inputs = layers.Input(shape=env.observation_spec['subPara']['shape'], dtype=np.float, name='subPara')
            subPara_inputs = kr.layers.Input(shape=(1,), dtype=np.float, name='subPara')
            # subPara_dense = kr.layers.Dense(8, activation='tanh', name='subPara_dense')(subPara_inputs)

            common = kr.layers.concatenate([image_layer, subPara_inputs])

            num_actions = 1
            # common = kr.layers.Dense(8, activation="tanh", name='common_dense1')(common)
            # num_actions = env.action_spec['shape'][0]
            # action = kr.layers.Dense(num_actions, name='action_dense1')(common)
            common = kr.layers.Dense(num_actions, activation='tanh', name='common_output')(common)

            model = kr.Model(inputs=[image_inputs, subPara_inputs], outputs=common)
            # model = kr.Model(inputs=image_inputs, outputs=common)
            model.compile(
                optimizer=kr.optimizers.Adam(learning_rate=1e-3),
                loss='mse',
                metrics=['mse']
            )
            # model.compile(
            #     optimizer=kr.optimizers.Adam(learning_rate=5e-3),
            #     loss=kr.losses.CategoricalCrossentropy(),
            #     metrics=[kr.metrics.CategoricalAccuracy()],
            # )
            return model


    def __loadOneTrainingDataSetForGenerator(self, batch_size=32):
        dataSetDirNames = list(filter(lambda name: '2021' in name, os.listdir(f"{self.traceDataDirPath}/")))
        for dataSetDirName in dataSetDirNames:
            images = []
            subParas = []
            labels = []
            steering_angle_last = 0.0
            throttle_last = 0.0
            speed_last = 0.0
            logPath = f"{self.traceDataDirPath}/{dataSetDirName}/driving_log.csv"
            print(f"start training {logPath} ...")
            logData = pd.read_csv(logPath, names=[
                'Center Image',
                'Left Image',
                'Right Image',
                'Steering Angle',
                'Throttle',
                'Break',
                'Speed'
            ])
            # create label
            for index in logData.index[:-2]:
                logData.loc[index, 'Next Steering Angle'] = (logData.loc[index+1, 'Steering Angle'] + logData.loc[index+2, 'Steering Angle'])/2
                # print(logData.groupby('Next Steering Angle').count())
            logData = logData.dropna()
            logData.to_csv(f"{self.traceDataDirPath}/{dataSetDirName}/new_log.csv")

        images = []
        subParas = []
        labels = []
        count = 0
        while True:
            dataSetDirNames = np.random.permutation(dataSetDirNames).tolist()
            for dataSetDirName in dataSetDirNames:
                logPath = f"{self.traceDataDirPath}/{dataSetDirName}/new_log.csv"
                logData = pd.read_csv(logPath, names=[
                    'Center Image',
                    'Left Image',
                    'Right Image',
                    'Steering Angle',
                    'Throttle',
                    'Break',
                    'Speed',
                    'Next Steering Angle'
                ])
                for index in np.random.permutation(logData.index):
                    imageName = logData.loc[index, 'Center Image'].split('\\')[-1]
                    imagePath = f"{self.traceDataDirPath}/{dataSetDirName}/IMG/{imageName}"
                    image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    subPara = float(logData.loc[index, 'Steering Angle'])
                    label = float(logData.loc[index, 'Next Steering Angle'])
                    actionType = np.random.choice(6)
                    if actionType == 0: # normal
                        images.append(image.astype("float32") / 255)
                        subParas.append(subPara)
                        labels.append(label)
                    elif actionType == 1: # flipped
                        image_flipped = cv2.flip(deepcopy(image), 1)
                        images.append(image_flipped.astype("float32") / 255)
                        subParas.append(-subPara)
                        labels.append(-label)
                    elif actionType == 2: # dark
                        image_dark = cv2.cvtColor(deepcopy(image), cv2.COLOR_RGB2HSV)
                        newBrightnessDelta = np.random.normal(30, 15)
                        image_dark[:, :, 2] = np.clip(image_dark[:, :, 2] - newBrightnessDelta, 0, 255)
                        image_dark = cv2.cvtColor(image_dark, cv2.COLOR_HSV2RGB)
                        images.append(image_dark.astype("float32") / 255)
                        subParas.append(subPara)
                        labels.append(label)
                    elif actionType == 3: # dark flipped
                        image_dark_flipped = cv2.flip(deepcopy(image), 1)
                        image_dark = cv2.cvtColor(image_dark_flipped, cv2.COLOR_RGB2HSV)
                        newBrightnessDelta = np.random.normal(30, 15)
                        image_dark_flipped[:, :, 2] = np.clip(image_dark_flipped[:, :, 2] - newBrightnessDelta, 0, 255)
                        image_dark_flipped = cv2.cvtColor(image_dark_flipped, cv2.COLOR_HSV2RGB)
                        images.append(image_dark_flipped.astype("float32") / 255)
                        subParas.append(-subPara)
                        labels.append(-label)
                    count += 1
                    if count >= batch_size:
                        images_copy = deepcopy(np.array(images, dtype="float32"))
                        subParas_copy = deepcopy(np.array(subParas, dtype="float32").reshape(-1, 1))
                        labels_copy = deepcopy(np.array(labels, dtype="float32").reshape(-1, 1))
                        assert images_copy.shape[0] == subParas_copy.shape[0] == labels_copy.shape[0]
                        images = []
                        subParas = []
                        labels = []
                        count = 0
                        yield ([images_copy, subParas_copy], labels_copy)



    def __loadOneTrainingDataSet(self):
        while True:
            dataSetDirNames = list(filter(lambda name: '2021' in name, os.listdir(f"{self.traceDataDirPath}/")))
            dataSetDirNames = np.random.permutation(dataSetDirNames).tolist()
            for dataSetDirName in dataSetDirNames:
                images = []
                subParas = []
                labels = []
                steering_angle_last = 0.0
                throttle_last = 0.0
                speed_last = 0.0
                logPath = f"{self.traceDataDirPath}/{dataSetDirName}/driving_log.csv"
                print(f"start training {logPath} ...")
                logData = pd.read_csv(logPath, names=[
                    'Center Image',
                    'Left Image',
                    'Right Image',
                    'Steering Angle',
                    'Throttle',
                    'Break',
                    'Speed'
                ])
                # create label
                for index in logData.index[:-2]:
                    logData.loc[index, 'Next Steering Angle'] = (logData.loc[index+1, 'Steering Angle'] + logData.loc[index+2, 'Steering Angle'])/2
                    # print(logData.groupby('Next Steering Angle').count())
                logData = logData.dropna()
                logData.to_csv(f"{self.traceDataDirPath}/{dataSetDirName}/new_log.csv")
                # dump images, subparas, labels into array
                noneStraightDataCount = 0
                for index in np.random.permutation(logData.index)[:1000]:
                    # if abs(logData.loc[index, 'Next Steering Angle']) <= 1e-2:
                    #     continue
                    # load images
                    imageName = logData.loc[index, 'Center Image'].split('\\')[-1]
                    imagePath = f"{self.traceDataDirPath}/{dataSetDirName}/IMG/{imageName}"
                    image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # pl.title('origin image')
                    # pl.imshow(image)
                    # pl.show()
                    images.append(image.astype("float32") / 255)
                    labels.append(logData.loc[index, 'Next Steering Angle'])
                    # subPara = logData.loc[index, ['Steering Angle', 'Throttle', 'Speed']].values.ravel()
                    # subPara[2] /= 30.5
                    subPara = float(logData.loc[index, 'Steering Angle'])
                    subParas.append(subPara)

                    image_flipped = cv2.flip(deepcopy(image), 1)
                    # pl.title('flipped image')
                    # pl.imshow(image_flipped)
                    # pl.show()
                    images.append(image_flipped.astype("float32") / 255)
                    labels.append(logData.loc[index, 'Next Steering Angle'] * (-1.0))
                    subParas.append(-subPara)

                    image_dark = cv2.cvtColor(deepcopy(image), cv2.COLOR_RGB2HSV)
                    newBrightnessDelta = np.random.normal(30, 15)
                    image_dark[:, :, 2] = np.clip(image_dark[:, :, 2] - newBrightnessDelta, 0, 255)
                    image_dark = cv2.cvtColor(image_dark, cv2.COLOR_HSV2RGB)
                    # pl.title(f'dark image {newBrightnessDelta}')
                    # pl.imshow(image_dark)
                    # pl.show()
                    images.append(image_dark.astype("float32") / 255)
                    subParas.append(subPara)
                    labels.append(logData.loc[index, 'Next Steering Angle'])

                    image_dark_flipped = cv2.cvtColor(deepcopy(image_flipped), cv2.COLOR_RGB2HSV)
                    newBrightnessDelta = np.random.normal(30, 15)
                    image_dark_flipped[:, :, 2] = np.clip(image_dark_flipped[:, :, 2] - newBrightnessDelta, 0, 255)
                    image_dark_flipped = cv2.cvtColor(image_dark_flipped, cv2.COLOR_HSV2RGB)
                    # pl.title(f'dark image {newBrightnessDelta}')
                    # pl.imshow(image_dark_flipped)
                    # pl.show()
                    images.append(image_dark_flipped.astype("float32") / 255)
                    subParas.append(-subPara)
                    labels.append(logData.loc[index, 'Next Steering Angle'] * (-1.0))

                    image_bright = cv2.cvtColor(deepcopy(image), cv2.COLOR_RGB2HSV)
                    newBrightnessDelta = np.random.normal(30, 15)
                    image_bright[:, :, -1] = np.clip(image_bright[:, :, -1] + newBrightnessDelta, 0, 255)
                    image_bright = cv2.cvtColor(image_bright, cv2.COLOR_HSV2RGB)
                    # pl.title(f'dark image {newBrightnessDelta}')
                    # pl.imshow(image_bright)
                    # pl.show()
                    images.append(image_bright.astype("float32") / 255)
                    subParas.append(subPara)
                    labels.append(logData.loc[index, 'Next Steering Angle'])

                    image_bright_flipped = cv2.cvtColor(deepcopy(image_flipped), cv2.COLOR_RGB2HSV)
                    newBrightnessDelta = np.random.normal(30, 15)
                    image_bright_flipped[:, :, -1] = np.clip(image_bright_flipped[:, :, -1] + newBrightnessDelta, 0, 255)
                    image_bright_flipped = cv2.cvtColor(image_bright_flipped, cv2.COLOR_HSV2RGB)
                    # pl.title(f'dark image {newBrightnessDelta}')
                    # pl.imshow(image_bright_flipped)
                    # pl.show()
                    images.append(image_bright_flipped.astype("float32") / 255)
                    subParas.append(-subPara)
                    labels.append(logData.loc[index, 'Next Steering Angle'] * (-1.0))

                    noneStraightDataCount += 1

                # # add straight observations
                # neededStraightAmount = noneStraightDataCount
                # print(f"needed straight amount = {neededStraightAmount}")
                # shuffledIndices = np.random.permutation(logData.index)
                # straightCount = 0
                # for index in shuffledIndices:
                #     if abs(logData.loc[index, 'Next Steering Angle']) > 1e-2:
                #         continue
                #     labels.append(logData.loc[index, 'Next Steering Angle'])
                #     # load images
                #     imageName = logData.loc[index, 'Center Image'].split('\\')[-1]
                #     imagePath = f"{self.traceDataDirPath}/{dataSetDirName}/IMG/{imageName}"
                #     image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
                #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #     images.append(image.astype("float32") / 255)
                #
                #     # subPara = logData.loc[index, ['Steering Angle', 'Throttle', 'Speed']].values.ravel()
                #     # subPara[2] /= 30.5
                #     subPara = logData.loc[index, 'Steering Angle']
                #     subParas.append(subPara)
                #     if straightCount < neededStraightAmount:
                #         straightCount += 1
                #     else:
                #         break

                images = np.array(images, dtype="float32")
                subParas = np.array(subParas, dtype="float32").reshape(-1, 1)
                labels = np.array(labels, dtype="float32").reshape(-1, 1)
                assert images.shape[0] == subParas.shape[0] == labels.shape[0]

                yield images, subParas, labels


    def __loadAllTrainingDataSet(self):
        dataSetDirNames = list(filter(lambda name: '2021' in name, os.listdir(f"{self.traceDataDirPath}/")))
        dataSetDirNames = np.random.permutation(dataSetDirNames)
        images = []
        subParas = []
        labels = []
        for dataSetDirName in dataSetDirNames:
            logPath = f"{self.traceDataDirPath}/{dataSetDirName}/driving_log.csv"
            print(f"start training {logPath} ...")
            logData = pd.read_csv(logPath, names=[
                'Center Image',
                'Left Image',
                'Right Image',
                'Steering Angle',
                'Throttle',
                'Break',
                'Speed'
            ])

            noneStraightDataCount = 0
            for index in logData.index[:-1]:
                logData.loc[index, 'Next Steering Angle'] = logData.loc[index+1, 'Steering Angle']
                # if abs(logData.loc[index, 'Next Steering Angle']) <= 1e-2:
                #     continue
                labels.append(logData.loc[index, 'Next Steering Angle'])
                # load images
                imageName = logData.loc[index, 'Center Image'].split('\\')[-1]
                imagePath = f"{self.traceDataDirPath}/{dataSetDirName}/IMG/{imageName}"
                image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image = getCenterDeviationWithImage(image)
                images.append(image.astype("float32") / 255)

                # subPara = logData.loc[index, ['Steering Angle', 'Throttle', 'Speed']].values.ravel()
                # subPara[2] /= 30.5
                subPara = logData.loc[index, 'Steering Angle']
                subParas.append(subPara)
                noneStraightDataCount += 1

            # print(logData.groupby('Next Steering Angle').count())
            logData = logData.dropna()
            logData.to_csv(f"{self.traceDataDirPath}/{dataSetDirName}/new_log.csv")

            # # add straight observations
            # neededStraightAmount = noneStraightDataCount
            # print(f"needed straight amount = {neededStraightAmount}")
            # shuffledIndices = np.random.permutation(logData.index)
            # straightCount = 0
            # for index in shuffledIndices:
            #     if abs(logData.loc[index, 'Next Steering Angle']) > 1e-2:
            #         continue
            #     labels.append(logData.loc[index, 'Next Steering Angle'])
            #     # load images
            #     imageName = logData.loc[index, 'Center Image'].split('\\')[-1]
            #     imagePath = f"{self.traceDataDirPath}/{dataSetDirName}/IMG/{imageName}"
            #     image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
            #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #     images.append(image.astype("float32") / 255)
            #
            #     # subPara = logData.loc[index, ['Steering Angle', 'Throttle', 'Speed']].values.ravel()
            #     # subPara[2] /= 30.5
            #     subPara = logData.loc[index, 'Steering Angle']
            #     subParas.append(subPara)
            #     if straightCount < neededStraightAmount:
            #         straightCount += 1
            #     else:
            #         break

        images = np.array(images, dtype="float32")
        subParas = np.array(subParas, dtype="float32").reshape(-1, 1)
        labels = np.array(labels, dtype="float32").reshape(-1, 1)

        return images, subParas, labels


if __name__ == '__main__':
    deepModel = DeepModel()
    deepModel.run()
