#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[6]:


df=pd.read_csv("Sales (1).csv")
df.head()


# In[8]:


data=df
# Data Preprocessing
data = data.fillna(method='ffill')
exog_vars = data[['TV', 'Radio', 'Newspaper']]
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]
exog_train, exog_test = exog_vars.iloc[:train_size], exog_vars.iloc[train_size:]

# Fit the SARIMAX Model
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
model = SARIMAX(train['Sales'], exog=exog_train, order=order, seasonal_order=seasonal_order)
model_fit = model.fit(disp=False)
print(model_fit.summary())

# Make Predictions
y_pred = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, exog=exog_test)

# Evaluate the Model
mse = mean_squared_error(test['Sales'], y_pred)
mae = mean_absolute_error(test['Sales'], y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Visualize Results
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['Sales'], label='Training Sales')
plt.plot(test.index, test['Sales'], label='Actual Sales')
plt.plot(test.index, y_pred, label='Predicted Sales', alpha=0.7)
plt.legend()
plt.title('Sales Forecasting using SARIMAX')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()


# In[ ]:




