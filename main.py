import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

import numpy as np
import pandas as pd
import json


class Settings:
    def __init__(self, max_len: int, answers: dict, file_name: str, epochs: int, with_init_model: bool):
        self.max_len = max_len
        self.answers = answers
        self.file_name = file_name
        self.epochs = epochs
        self.with_init_model = with_init_model

class Data:
    def __init__(self, settings_param: Settings) -> None:
        self.data = pd.read_csv(settings_param.file_name)
        self.max_len = settings_param.max_len
        self.answers = settings_param.answers

        self.Prepair_Data()


    def Prepair_Data(self) -> None:
        self.X = []
        self.Y = []
        for i in range(len(self.data["Name"])):
            self.X.append(self.Code_Name(self.data["Name"][i]))
            self.Y.append([])
            for j in range(len(self.answers)):
                self.Y[i].append(0)
            self.Y[i][self.answers[self.data["Gender"][i]]] = 1

    def Get_data(self) -> np.array:
        return np.array(self.X), np.array(self.Y)
    def Code_Name(self, name):
        decode_name = list(name)
        code_name = [] #20
        for i in range(self.max_len):
            if len(decode_name) - 1 >= i:
                code_name.append(ord(decode_name[i]) / 100)
            else:
                code_name.append(-1)
        return code_name

class Neural_Network:
    def __init__(self, settings_param: Settings, **kwargs):
        self.input_dim = settings_param.max_len
        self.output_dim = len(settings_param.answers)
        self.epochs = settings_param.epochs

        self.settings = settings_param

        if settings_param.with_init_model:
            self.Create_Model()
        else:
            self.Load_Model(kwargs["model_name"])
            pass

    def Create_Model(self):

        self.model = Sequential()
        self.model.add(Dense(64, input_dim=self.input_dim, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(self.output_dim, activation='relu'))

        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        self.model.summary()

    def Fit_Model(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=self.epochs)
    def Save_Model(self, name) -> int:
        try:
            self.model.save(name)
            save_data = {
                "max_len": self.settings.max_len,
                "file_name": self.settings.file_name,
                "answers": self.settings.answers
            }
            with open(name + ".json", "w") as json_file:
                json.dump(save_data, json_file, indent=4)
            return 0
        except:
            return 1

    def Load_Model(self, name_of_model: str) -> int:
        try:
            self.model = load_model(name_of_model, custom_objects={'mse': tf.keras.losses.MeanSquaredError})
            return 0
        except:
            return 1

    def Predict_Model(self, name) -> np.array:
        return self.model.predict(np.array([Data(self.settings).Code_Name(name)]))

class NN_Interface:
    def __init__(self, model_name: str):
        with open(model_name + ".json", "r") as json_file:
            js_info = json.load(json_file)
            self.settings = Settings(max_len=js_info["max_len"], file_name=js_info["file_name"], epochs=0, with_init_model=False, answers=js_info["answers"])
        self.model_name = model_name
    def Get_Model(self) -> Neural_Network:
        return Neural_Network(self.settings, model_name=self.model_name)


if __name__ == "__main__":
    Settings_Param = Settings(
         max_len=20,
         file_name="data.csv",
         epochs=100,
         with_init_model=True,
         answers={
             "M": 1,
             "F": 0
         })

    Data_Class = Data(settings_param=Settings_Param)
    Neural_Network_Class = Neural_Network(settings_param=Settings_Param)

    Data_Class.Prepair_Data()
    X, y = Data_Class.Get_data()

    Neural_Network_Class.Fit_Model(X, y)
    Neural_Network_Class.Save_Model('model.h5')
