# How understand person is man or woman from name
## It is so easy with me new neural network which can predict this info
## Contents
<!-- TOC -->
* [Settings](#Settings)
* [Code](#Import)
  * [About neural network](#Neural-network)
    * [Model](#Model)
  * [Data](#Data)
    * [Get Data](#Get-Data)
    * [Sort Data](#Get-Data)
<!-- TOC -->
___

# Settings:
```Python
class Settings:
    def __init__(self, max_len: int, answers: dict, file_name: str, epochs: int, with_init_model: bool):
        self.max_len = max_len
        self.answers = answers
        self.file_name = file_name
        self.epochs = epochs
        self.with_init_model = with_init_model
```
Max Len(int) - It is what is max length of name.\
answers(json) - Which answers do you need.\
file_name(str) - CSV file name.\
epochs(int) - How meny epochs do you want to train.\
with_init_model(bool) - initialize model. 



# Import
___

### install requirements:
```
pip install requirements.txt
```
### import modules
```
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

import numpy as np
import pandas as pd
import json
```


# Neural network
___
## Model
I use tensorflow for build default sequential model.
Also i use tactic \`less and less neurons`. It helps me make less loss and more accuracy.
### Create base sequential model:
```Python   
def Create_Model(self):

    self.model = Sequential()
    self.model.add(Dense(64, input_dim=self.input_dim, activation='relu'))
    self.model.add(Dense(32, activation='relu'))
    self.model.add(Dense(16, activation='relu'))
    self.model.add(Dense(self.output_dim, activation='relu'))

    self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    self.model.summary()
````

### Train Model:
```Python
def Fit_Model(self, X_train, y_train):
    self.model.fit(X_train, y_train, epochs=self.epochs)
```
### Predict Model:
```Python
def Predict_Model(self, name) -> np.array:
    return self.model.predict(np.array([Data(self.settings).Code_Name(name)]))
```
### Save Model:
```Python
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
```

### Load Model:
```Python
def Load_Model(self, name_of_model: str) -> int:
        try:
            self.model = load_model(name_of_model, custom_objects={'mse': tf.keras.losses.MeanSquaredError})
            return 0
        except:
            return 1
```

### Neural Network interface:
```Python
class NN_Interface:
    def __init__(self, model_name: str):
        with open(model_name + ".json", "r") as json_file:
            js_info = json.load(json_file)
            self.settings = Settings(max_len=js_info["max_len"], file_name=js_info["file_name"], epochs=0, with_init_model=False, answers=js_info["answers"])
        self.model_name = model_name
    def Get_Model(self) -> Neural_Network:
        return Neural_Network(self.settings, model_name=self.model_name)
```

# Data
___

## Get Data:
```Python
self.data = pd.read_csv(settings_param.file_name)
```

## Sort Data

### Name to numbers:
```Python
def Code_Name(self, name):
        decode_name = list(name)
        code_name = [] #20
        for i in range(self.max_len):
            if len(decode_name) - 1 >= i:
                code_name.append(ord(decode_name[i]) / 100)
            else:
                code_name.append(-1)
        return code_name
```

### Repair data
```Python
def Prepair_Data(self) -> None:
        self.X = []
        self.Y = []
        for i in range(len(self.data["Name"])):
            self.X.append(self.Code_Name(self.data["Name"][i]))
            self.Y.append([])
            for j in range(len(self.answers)):
                self.Y[i].append(0)
            self.Y[i][self.answers[self.data["Gender"][i]]] = 1
```