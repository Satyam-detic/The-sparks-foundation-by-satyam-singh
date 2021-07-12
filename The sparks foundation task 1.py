#!/usr/bin/env python
# coding: utf-8

# Data Science & Business Analytics - Task 1: Prediction using Supervised Machine Learning
# 
# Prepared by: Satyam singh
# 
# Aim: To predict the score of a student when he/she studies for 9.25 hours.

# In[ ]:


import pandas as pd  # for manipulating the dataset
import numpy as np   # for applying numerical operations on the observations
import matplotlib.pyplot as plt  # for plotting the graphs

from sklearn.model_selection import train_test_split     # for splitting the dataset into training and testing sets
from sklearn.linear_model import LinearRegression        # for building the linear regression model
from sklearn.metrics import mean_squared_error,mean_absolute_error  # for calculating mean squared error


# Reading the dataset
# Importing the raw dataset from GitHub:

# In[2]:


dataURL='https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'


# In[3]:


df=pd.read_csv(dataURL)


# In[4]:


df.head()


# 
# Plotting the relationship between hours and score:

# In[5]:


df.plot(x='Hours',y='Scores',style='x')
plt.title('Hours vs Percentage')
plt.xlabel('No. of hours')
plt.ylabel('Percentage')
plt.show()


# 
# By plotting the relationship between the no. of hours studied and the score obtained, we see that there is linear relationship between these two variables. We'll now split the dataset into two parts, to create training and testing sets to build the model. 

# In[6]:


x=df.iloc[:,0:1]
y=df.iloc[:,1:]


# 
# Splitting the dataset into training and test sets:

# In[7]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# Now we generate the Linear Regression Model by using the following commands:

# In[8]:


lr=LinearRegression()


# In[9]:


lr.fit(x_train,y_train)


# In[10]:


lr.score(x_train,y_train)


# In[11]:


lr.score(x_test,y_test)


# In[12]:


pred=lr.predict(x_test)


# 
# After the model is trained, we need to check how accurate the model is. For this we use the mean squared error metric (from Numpy):

# In[13]:


print(mean_squared_error(pred,y_test))


# In[14]:



print(np.sqrt(mean_squared_error(pred,y_test)))


# 
# As we can see, the value of MSE is 4.509. Lower the MSE, higher the accuracy of our model.
# Plotting the best fit line to ascertain the relationship betweeen the points in our scatter plot:

# In[15]:


line = lr.coef_*x + lr.intercept_

plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# 
# Predicting the values
# Now we can use this model to predict upcoming values against the test set. It'll help us in ascertaining the accuracy of the model.
# Comparing actual values with predicted values:

# In[16]:


df2=pd.DataFrame(y_test)
df2


# In[23]:


df2['Predicted values']=pred


# In[26]:


df2


# We know that the MSE is 4.509. The dataframe df2, shows this error, by comparing the predicted values against the actual values in the dataset.
# 
# We have created our Linear Regression Model, with the help of which we'll able to predict the score of a child when the number of studying hours is set to 9.25.
# 
# 
# The Linear Regression Model predicts a numerical variable, when a condition (in the form of numerical variable) is given. So, we'll set the number of hours to 9.25 and predict the score.

# In[19]:


hours= [[9.25]]


# In[27]:


prediction=lr.predict(hours)


# # The model is able to predict the score which is 93.89272. This means, if a student studies for 9.25 hours, his score will be 93.89272.

# In[ ]:




