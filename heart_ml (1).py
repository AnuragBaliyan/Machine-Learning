#!/usr/bin/env python
# coding: utf-8

# # HEART DISEASE PREDICTIVE MODEL
# 

# In[1]:


#This Model basically focuses on classifying whether a person is having a Heart Disease or not.


# # STEP 1 -  Importing the Libraries and the Dataset.

# In[2]:


#IMPORTING ALL THE NECESSARY LIBRARIES.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier


# In[3]:


#Importing the dataset

data=pd.read_csv("C:\\users\\anura\\Downloads\\heart.csv")
data.head(10) #for visualizing only the top 10 rows of the dataset.


# # STEP 2- Data Exploration

# In[4]:


#checking the size or shape of the dataset

data.shape


# In[5]:


type(data)


# In[6]:


#Checking Information of the Dataset

data.info()


# In[7]:


#we can verify that our dataset doesn't consists of any sort of categorical features, hence we're good to go.


# In[8]:


#Checking for any Null Values.

data.isnull().sum()


# In[9]:


#we can clearly see that there are no Null VAlues in our dataset


# In[10]:


#checking the distribution of feature "target" values in our dataset

data['target'].value_counts()


# In[11]:


#checking the distribution of feature "sex" values in our dataset

data['sex'].value_counts()


# In[12]:


#checking the distribution of feature "cp" values in our dataset

data['cp'].value_counts()


# In[13]:


#checking the distribution of feature "slope" values in our dataset

data['slope'].value_counts()


# In[14]:


#checking the Statistical Values of our dataset

data.describe()


# In[15]:


#with the help of Statistical Analysis we can clearly see that our dataset is fairly distributed


# # STEP4- Visualizing the Data

# In[16]:


#lets start visualizing with the heatmap

plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,cmap='Blues',linewidths=1,linecolor="red")


# In[17]:


#lets visualize the target column

sns.countplot(x="target", data=data,palette='Blues')


# In[18]:


#Scatterplot of Heartbeats over Age

plt.scatter(x=data.age[data.target==1], y=data.thalach[(data.target==1)], c="green")
plt.scatter(x=data.age[data.target==0], y=data.thalach[(data.target==0)],c="hotpink")
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


# In[19]:


#Heart Disease Frequency for Slope

pd.crosstab(data.slope,data.target).plot(kind="bar",figsize=(10,6),ec='red')
plt.title('Heart Disease Frequency for Slope')
plt.xlabel('The Slope')
plt.xticks(rotation = 0)
plt.ylabel('Frequency')
plt.grid()
plt.show()


# In[20]:


#Heart Disease Frequency for Ages

pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(16,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# In[21]:


pd.crosstab(data.age,data.slope).plot(kind="bar",figsize=(16,6))
plt.title('Slope type for Ages')
plt.xlabel('Age')
plt.ylabel('Slope')
plt.grid()
plt.show()


# # STEP5- Splitting the Data

# In[22]:


#splitting data into X and Y and then further into Training set and Test set

x=data.drop('target',axis=1)
y=data['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)


# In[23]:


#scaling down the independent features to same extent

ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)


# # STEP6- Evaluating Data over Different Models

# In[24]:


#We will Use different ML Algorithms to create models and then predict the accuracy of predictions by each Model.


# # LOGISTIC REGRESSION

# In[25]:


#Logistic Regression Model

lr=LogisticRegression(max_iter=1000,solver='liblinear')
lr.fit(x_train,y_train)


# In[26]:


#Training score 

Training_score_lr=lr.score(x_train,y_train)
print("Training score for Logistic Regression Model is :",Training_score_lr*100)


# In[27]:


#Testing score

Testing_score_lr=lr.score(x_test,y_test)
print("Testing score for Logistic Regression Model is :",Testing_score_lr*100)


# In[28]:


#Predicting the values of xtest for accuracy evaluation

y_pred_lr=lr.predict(x_test)
print(y_pred_lr)


# In[29]:


#creating a confusion matrix for predictions and actual outputs

cm_lr=confusion_matrix(y_pred_lr,y_test)
print(cm_lr)


# In[30]:


#classification report for our Model

cr_lr=classification_report(y_pred_lr,y_test)
print(cr_lr)


# In[31]:


Accuracy_LogisticRegression=Testing_score_lr*100

print("Accuracy for Logistic Regression Model is :", Accuracy_LogisticRegression)


# # KNEAREST NEIGHBORS CLASSIFIER

# In[32]:


#KNN Model

KNN=KNeighborsClassifier(n_neighbors=5)
KNN.fit(x_train,y_train)


# In[33]:


#evaluating training score for KNN model

Training_score_KNN=KNN.score(x_train,y_train)
print("Training score for KNN model is :",Training_score_KNN*100)


# In[34]:


#evaluating testing score for KNN model

Testing_score_KNN=KNN.score(x_test,y_test)
print("Testing score for KNN model is :",Testing_score_KNN*100)


# In[35]:


#predictions of our KNN model

y_pred_KNN=KNN.predict(x_test)
print(y_pred_KNN)


# In[36]:


#confusion matrix created by KNN

cm_KNN=confusion_matrix(y_pred_KNN,y_test)
print(cm_KNN)


# In[37]:


#classification report for KNN

cr_KNN=classification_report(y_pred_KNN,y_test)
print(cr_KNN)


# In[38]:


Accuracy_KNN=Testing_score_KNN*100
print("Accuracy for KNN Model is :", Accuracy_KNN)


# In[ ]:





# # NAIVE BAYES CLASSIFIER

# In[39]:


#NAIVE BAYES MODEL

naive_bayes=GaussianNB()
naive_bayes.fit(x,y)


# In[40]:


#predicting outcomes of x 

expected=y
predicted=naive_bayes.predict(x)


# In[41]:


#classification report of naive bayes

cr_naive_bayes=classification_report(expected,predicted)
print(cr_naive_bayes)


# In[42]:


cm_naive_bayes=confusion_matrix(expected,predicted)
print(cm_naive_bayes)


# In[43]:


Accuracy_naive_bayes=84.0
print("Accuracy for naive bayes model is :", Accuracy_naive_bayes)


# In[ ]:





# # RANDOM FOREST CLASSIFIER

# In[44]:


#random forest model

rf=RandomForestClassifier()


# In[45]:


#defining parameters for RandomForest model

n_estimators=[int(z) for z in np.linspace(start=10, stop=80, num=10)]
max_features=['auto','sqrt']
max_depth=[2,4,5]
min_samples_split=[2,5]
min_samples_leaf=[1,2]
bootstrap=[True,False]


param_grid={'n_estimators':n_estimators,
           'max_features':max_features,
            'max_depth':max_depth,
            'min_samples_split':min_samples_split,
            'min_samples_leaf':min_samples_leaf,
            'bootstrap':bootstrap}


# In[46]:


#implyig Grid Search cv for finding best parameters for our model

rf_grid=GridSearchCV(rf,param_grid=param_grid,verbose=2,cv=5,n_jobs=-1)
rf_grid.fit(x_train,y_train)


# In[47]:


#These are the best parameters for our model

rf_grid.best_params_


# In[48]:


#filling appropriate parameters for our model.

rf=RandomForestClassifier(bootstrap= False,max_depth=2,max_features= 'sqrt',min_samples_leaf= 2,min_samples_split= 5,n_estimators = 64)


# In[49]:


#Fitting training set in our model

rf.fit(x_train,y_train)


# In[50]:


#evaluating  training score for our model 

Training_score_rf=rf.score(x_train,y_train)
print("Training score for Random Forest Model is :", Training_score_rf*100)


# In[51]:


# evaluating Testing score for our model 

Testing_score_rf=rf.score(x_test,y_test)
print("Testing score for Random Forest Model is :", Testing_score_rf*100)


# In[52]:


#predictions made by our model

y_pred_rf=rf.predict(x_test)
print(y_pred_rf)


# In[53]:


#confusion matrix for Random Forest Model is

cm_rf=confusion_matrix(y_pred_rf,y_test)
print(cm_rf)


# In[54]:


#Classification report of our Random Forest model is

cr_rf=classification_report(y_pred_rf,y_test)
print(cr_rf)


# In[55]:


Accuracy_rf=Testing_score_rf*100


# In[56]:


print("Accuracy of our Random Forest Classifier Model is :", Accuracy_rf)


# # DEFINING BEST MODEL FOR OUR CASE

# In[57]:


accuracy={"Accuracy_rf" : 88.52, "Accuracy_LogisticRegression" : 90.16,"Accuracy_naive_bayes" : 84.0,"Accuracy_KNN" : 81.96}


# In[60]:


colors = ["green", "red","blue","yellow"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracy.keys()), y=list(accuracy.values()), palette=colors)
plt.show()


# In[61]:


Results=[print("Accuracy of LOGISTIC REGRESSION Model is :",Accuracy_LogisticRegression,"%"),
        print("Accuracy of RANDOM FOREST CLASSIFIER is :", Accuracy_rf,"%"),
        print("Accuracy of NAIVE BAYES MODEL is :", Accuracy_naive_bayes,"%"),
        print("Accuracy of KNEARESTNEIGHBORS CLASSIFIER is :",Accuracy_KNN,"%")]


# # HENCE WE CAN SAY THAT LOGISTIC REGRESSION MODEL IS BEST SUITED FOR OUR DATASET.

# In[ ]:




