from main import Data, Neural_Network, NN_Interface

Inter = NN_Interface(model_name="model.h5")
nn = Inter.Get_Model()
print(nn.Predict_Model("Jonh"))