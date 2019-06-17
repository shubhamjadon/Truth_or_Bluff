# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Data not splitted as dataset given is very small.

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) #degree tells max degree of feature
X_poly = poly_reg.fit_transform(X) #it makes X_poly matrix containg powers of X matrix till degree 4
#Now we apply multilinear regression on X_poly. Also we don't need to insert a col of ones in starting as X_poly already did that for us
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#visualising polynomial regression modal
X_grid = np.arange(min(X), max(X), 0.1) #it has been introduced as given X contains 10 no so line plotted b/w points looks straight so to make it comlete courve we will predict value of salary at 0.1 distances from start to end 
X_grid = X_grid.reshape((len(X_grid),1)) #reshaping to make it a matrix
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)) ,color = 'blue')#Here using X_grid instead of X to get desired curve
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting salary at 6.5 to check if he told truth or not about his last salary
#Predicting using polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))

#Looking at result we know know that he told truth about his previous salary
