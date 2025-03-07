import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

#Apply simple linear regression on the data
class simple_linear_regression:
    #for all models it was better to take epochs = 1000 and learning rate = 0.0099999
    epochs=1000
    Learning_rate=0.0099999

    def __init__(self , x , y , x_label , y_label,i):
        self.x = x
        self.y = y
        self.m = 0
        self.c = 0
        self.n = float(len(self.x)) 
        self.x_label = x_label
        self.y_label = y_label
        self.loc=i
        self.y_pred = None
        self.fin = None
        self.gradient_descent_algorithm()

    #Apply gradient descent algorithm on the data    
    def gradient_descent_algorithm(self):
        for i in range(simple_linear_regression.epochs):
            self.y_pred = self.m * self.x + self.c  
            D_m = (-2/self.n) * sum((self.y - self.y_pred)* self.x) 
            D_c = (-2/self.n) * sum(self.y - self.y_pred)  
            self.m = self.m - simple_linear_regression.Learning_rate * D_m  
            self.c = self.c - simple_linear_regression.Learning_rate * D_c  
       
        #y predection calculation
        self.fin = self.m * self.x + self.c

        self.MSE()
        self.ploting()

    #Calculate mean squared error    
    def MSE(self):
        mse=metrics.mean_squared_error(self.y, self.y_pred)
        print("The mean squared error for ",self.x_label," feature is" , mse)

    def ploting(self):
        plt.xlabel(self.x_label, fontsize = 10)
        plt.ylabel(self.y_label, fontsize = 10)
        plt.scatter(self.x, self.y)
        plt.plot(self.x, self.fin, color='maroon', linewidth = 2)
        plt.show()

class file_handling:
    def __init__(self,data,col_name) :
        self.target_name = col_name[4]
        self.target = data[col_name[4]]
        self.feature = data[col_name[:4]]
        self.feature_name = col_name[:4]        


data = pd.read_csv('assignment1dataset.csv')
data['Free hours']=24-(data['Hours Studied']+data['Sleep Hours'])

scale = MinMaxScaler(feature_range=(0, 1))
res_data=scale.fit_transform(data)
col_name = list(data.columns)
sc_data=pd.DataFrame(res_data,columns=col_name)

obj1 = file_handling(sc_data,col_name)

feature1=simple_linear_regression(obj1.feature[col_name[0]],obj1.target,obj1.feature_name[0],obj1.target_name,1)

feature2=simple_linear_regression(obj1.feature[col_name[1]],obj1.target,obj1.feature_name[1],obj1.target_name,2)

feature3=simple_linear_regression(obj1.feature[col_name[2]],obj1.target,obj1.feature_name[2],obj1.target_name,3)

feature4=simple_linear_regression(obj1.feature[col_name[3]],obj1.target,obj1.feature_name[3],obj1.target_name,4)

feature5=simple_linear_regression(sc_data[col_name[5]],obj1.target,col_name[5],obj1.target_name,5)

