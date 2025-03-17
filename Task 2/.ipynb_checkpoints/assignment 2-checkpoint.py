#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
# For calculating MSE 
from sklearn.metrics import mean_squared_error as MSE
#for calculating r2-Score
from sklearn.metrics import r2_score  as score
#feature selection used libraries
from sklearn.feature_selection import mutual_info_classif as mic
# To calculate it for chi 2
from sklearn.feature_selection import chi2

def PolynomialFeatures(feat, deg):
    """
        as for deg = 1 to deg = n
        deg[x]= deg[x-1] concatenate with (features)^x
        so to calculate it  rows and columns are needed to 
        iterate over it and calculate the polynomial regression 
    """
    num_of_samples, num_of_features = feat.shape

    # as deg = 0 --> y = 1 so needed to make an array of the same number of rows 
    # so there won't be an error exists
    result = np.ones((num_of_samples, 1))

    # To iterate over the features 
    # put the added feature to the  previous result  
    for i in range(num_of_features):
        for j in range(1, deg + 1):
            # to add a feature put put it with power j 
            added_feature = feat[:, i] ** j

            #put the new column in the result 
            result = np.column_stack((result, added_feature))
    return result

def polynomial_regression(x_test, coefficients, degree):
    """Calculate y_prediction of the given coefficients , x part for test"""
    result = PolynomialFeatures(x_test, degree)
    y_pred = np.dot(result, coefficients)
    return y_pred


data = pd.read_csv("assignment2dataset.csv")
print(data.isnull().sum())
#checking data
data.dtypes

# make Extracurricular Activities column numerical
ordinal_mapping = {'Yes': 1, 'No': 0}
processed_data = data.replace({'Extracurricular Activities': ordinal_mapping})
print(processed_data)
processed_data.dtypes

#checking duplicates and remove it if exists
sum(processed_data.duplicated())
processed_data.drop_duplicates(inplace = True)
sum(processed_data.duplicated())

# split data into features and target
features = processed_data.iloc[:,:5]
feat_name = features.columns

#For making sure taking right columns
print(feat_name)
target = processed_data.iloc[:,-1]

# Apply feature selection by using chi_squared test
chi_Sc = chi2(features , target)
res_Sc = pd.Series(chi_Sc[0], index = feat_name)
res_Sc.sort_values(ascending = False, inplace = True)
res_Sc.plot(kind = 'bar')

# Take features has highly chi with target
selected_features = features[['Previous Scores', 'Hours Studied','Sample Question Papers Practiced']]
selected_features.info()

# Spliting data into train, test
X_train, X_test, y_train, y_test = train_test_split(selected_features, target, test_size = 0.30,shuffle=True,random_state=10)

#Converting data into array 
xt_arr = np.array(X_train)
yt_arr = np.array(y_train)
x_test= np.array(X_test)
y_test= np.array(y_test)

# To prevent overfitting the right degree was 2 
deg = 2


poly_reg_res = PolynomialFeatures(xt_arr, deg)

#make sure that there is a difference in the data
print(poly_reg_res.shape)

#calculate the coefficients of the result of applying polynomial feature function
coeff = np.linalg.inv(poly_reg_res.T.dot(poly_reg_res)).dot(poly_reg_res.T).dot(yt_arr)
print (coeff.shape)

# Used to calculate both accuracies to know the final accuracy 
# and if there is an overfitting or not 
y_pred = polynomial_regression(x_test, coeff,deg)
y_pred_t = polynomial_regression(xt_arr, coeff,deg)

#calculating the mean square error and accuracy for test
mse_test= MSE(y_test, y_pred)
test_accuracy = score(y_test, y_pred)
# Calculate  the mean square error and accuracy for train\
mse_train= MSE(yt_arr, y_pred_t)
train_accuracy = score(yt_arr, y_pred_t)

print("Mean square error for train: ", mse_train)
print("Mean square error for test: ",mse_test)

print("Accuracy for train: ",train_accuracy)
print("Accuracy for test: ",test_accuracy)