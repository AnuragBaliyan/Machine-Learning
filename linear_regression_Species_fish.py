#!/usr/bin/env python
# coding: utf-8

# In[44]:


#importing all the packages, models and preprocessors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


# In[45]:


#loading the csv files

data=pd.read_csv("C:\\users\\anura\\Downloads\\fish.csv")
data


# In[46]:


data.head(10)


# In[47]:


#checking basic statistics of our data.

data.describe()


# In[48]:


#checking the datatypes

data.info()


# In[49]:


#checking if there are any null values

data.isnull().sum()


# In[50]:


type(data)


# In[51]:


#visualizing the correlation between our target value and features.

sns.heatmap(data.corr(),annot=True)


# In[52]:


sns.pairplot(data)


# In[ ]:





# In[53]:


#preprocessing categorical data to be converted into numerical form. 

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Species']=le.fit_transform(data['Species'])


# In[54]:


#splitting the dataset into feature variables and target variable

#feature variable

x=data.iloc[:,1:7]
print(x.head(5))


# In[55]:


#target variable

y=data.iloc[:,0]
print(y.head(5))


# In[56]:


#splitting data into training and testing data.

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=12)


# # LOGISTIC REGRESSION

# In[57]:


#applying LinearRegression algorithm to our training data.

regressor=LogisticRegression(solver='liblinear')
regressor.fit(x_train,y_train)


# In[58]:


#predicting our test data values to check the accuracy of our model.

y_predict_regressor=regressor.predict(x_test)
print("The Predicted Values by Logistic Regression model are :",y_predict_regressor)


# In[59]:


#test data accuracy 

test_score_lr=regressor.score(x_test,y_test)
print("The Testing Score of our model is :" ,test_score_lr*100,"%")


# In[60]:


#training data accuracy of our model.

train_score_lr=regressor.score(x_train,y_train)
print("The Training Score of our model is :" ,train_score_lr*100,"%")


# In[61]:


#creating confusion matrix for our logistic regression model

cm_lr=confusion_matrix(y_predict_regressor,y_test)
print(cm_lr)


# In[62]:


#creating classification report for our model/

cr_lr=classification_report(y_predict_regressor,y_test)
print(cr_lr)


# In[65]:


#cross validating our accuracies

cv_lr=cross_val_score(regressor,x,y,cv=5)
print(cv_lr*100)


# In[66]:


#mean of accuracies generated through cross validation

cv_lr_mean=np.mean(cv_lr)
print("The Mean percentage of Crossvalidation accuracies are:",cv_lr_mean*100)


# In[70]:


cv_dict={"cv_1":93.75,"cv_2":93.75,"cv_3":96.87,"cv_4":96.87,"cv_5":80.64,"cv_mean":92.37}


# In[72]:


#visualizing the accuracies of different numbers of cross validation 

colors = ["green", "red","blue","yellow","hotpink","orange"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("cross_validation_number")
sns.barplot(x=list(cv_dict.keys()), y=list(cv_dict.values()), palette=colors)
plt.show()


# In[ ]:




