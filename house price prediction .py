#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# ## analysing our dataset

# In[2]:



houses = pd.read_csv("data.csv")


# In[3]:


houses.head()


# In[4]:


houses.shape


# In[5]:



houses.describe()


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


houses.hist(bins = 50 , figsize=(50,30))


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## finding correlation

# In[9]:


corr_matrix = houses.corr()
corr_matrix


# In[10]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[11]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(houses[attributes], figsize = (12,8))


# ## tring to create new attributes from given

# In[12]:


houses["TAXRM"] = houses['TAX']/houses['RM']


# In[13]:


houses.head()


# In[14]:


corr_matrix = houses.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[15]:


## this has high negative correaltion so lets use it
houses.describe()


# ## train test split

# In[16]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(houses, houses['CHAS']):
    strat_train_set = houses.loc[train_index]
    strat_test_set = houses.loc[test_index]


# In[17]:


strat_train_set["CHAS"].value_counts()


# In[18]:


## seperating dependent and independent variables


# In[19]:


train_x = strat_train_set.drop("MEDV", axis = 1)
train_y = strat_train_set["MEDV"].copy()
test_x = strat_test_set.drop("MEDV", axis=1)
test_y = strat_test_set["MEDV"].copy()


#  ## creating pipeline for data preprocessing

# In[20]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])


# In[21]:


train_x_numpy = my_pipeline.fit_transform(train_x)


# In[22]:


## train_x_numpy is a numpy array not a dataframe


# ## choosing our model

# In[23]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(train_x_numpy, train_y )


# ## evaluating our model

# In[24]:


from sklearn.metrics import mean_squared_error
import numpy as np
train_y_pred = model.predict(train_x_numpy)
mse = mean_squared_error(train_y, train_y_pred)
rmse = np.sqrt(mse)


# In[25]:


rmse


# ## evaluation using cross validation

# In[26]:



from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, train_x_numpy, train_y, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[27]:


rmse_scores


# In[28]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[29]:


print_scores(rmse_scores)


# ## Saving the model - serialization

# In[30]:


from joblib import dump, load
dump(model, 'house price prediction.joblib') 


# ## using the saved model - deserialization

# In[31]:


trained_model = load("house price prediction.joblib")
test_y_pred = trained_model.predict(test_x)


# In[35]:


test_x_numpy = my_pipeline.fit_transform(test_x)


# In[36]:


test__y_pred_pipeline = trained_model.predict(test_x_numpy)


# In[37]:


## after passing through pipeline


# In[38]:


mse = mean_squared_error(test_y, test__y_pred_pipeline)
rmse = np.sqrt(mse)
rmse


# In[ ]:




