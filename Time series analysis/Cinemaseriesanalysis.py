#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# # style to use

# In[2]:


color_pal = sns.color_palette('deep')
plt.style.use('dark_background')


# # IMPORT DATASET

# In[3]:


df = pd.read_csv('cinemaTicket_Ref.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info


# In[7]:


df.describe()


# # Sorting Missing values

# In[8]:


df.isnull().sum()


# In[9]:


missingV = df.isnull().sum().sort_values(ascending=False)
missingV = pd.DataFrame(data=df.isnull().sum().sort_values(ascending=False), columns=['MissingValNum'])


# In[10]:


missingV['Percent'] = missingV.MissingValNum.apply(lambda x : '{:.2f}'.format(float(x)/df.shape[0] * 100))
missingV = missingV[missingV.MissingValNum > 0]


# In[11]:


print(missingV)


# # plot the dataset

# ## convert 'date' column to pandas datetime format

# In[12]:


df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')


# # group the data by 'date' and sum the 'total_sales' values for each date

# In[13]:


sales_by_date = df.groupby('date')['total_sales'].sum()
plt.figure(figsize=(12,6)) # Set the size of the plot


# # plot the sales data

# In[14]:


plt.plot(sales_by_date.index, sales_by_date.values)


# # Plot the data

# In[15]:


plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.title('Total Sales by Date')
plt.show()


# ## using histplot

# In[16]:


plt.figure(figsize=(12,6))
plt.bar(sales_by_date.index, sales_by_date.values)


# In[17]:


# plot the total sales against the date
sns.histplot(data=df, x='date', y='total_sales')
plt.xlabel('Date')
plt.ylabel('Total sales')
plt.title('Distribution of Total Sales by Date')
plt.show()


# In[18]:


#using histplot
plt.figure(figsize=(12,6))
# Create a sample dataframe
df= pd.read_csv('cinemaTicket_Ref.csv')
# Plot the scatter plot
sns.scatterplot(data=df, x='tickets_sold', y='total_sales', hue='cinema_code')

# Set the axis labels and title
plt.xlabel('Tickets Sold')
plt.ylabel('Total Sales')
plt.title('Scatter plot of Total Sales vs Tickets Sold')

# Show the plot
plt.show()


# In[19]:


# Plot the scatter plot
sns.scatterplot(data=df, x='show_time', y='ticket_price', hue='cinema_code')

# Set the axis labels and title
plt.xlabel('Show Time')
plt.ylabel('Ticket Price')
plt.title('Scatter plot of Ticket Price vs Show Time')

# Show the plot
plt.show()


# # spliting the dataset

# In[20]:


# convert 'date' column to pandas datetime format
df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')


# In[21]:


# sort the dataframe by date
df = df.sort_values('date')


# In[22]:


# define the size of the test set
test_size = 0.2


# In[23]:


# calculate the index to split the data
split_index = int(len(df) * (1 - test_size))


# In[24]:


# split the data into training and test sets
train = df[:split_index]
test = df[split_index:]


# # Plot the Data

# In[25]:


fig, ax = plt.subplots(figsize=(15, 5))
train.plot(x='date', y='total_sales', ax=ax, label='Training Set', title='Data Train/Test Split')
test.plot(x='date', y='total_sales', ax=ax, label='Test Set')
ax.axvline(train['date'].max(), color='white', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()


# In[26]:


df = pd.read_csv('cinemaTicket_Ref.csv')


# In[27]:


# set the target and predictors
y = df.total_sales #target


# In[28]:


# use only those input features with numeric data type
df_temp = df.select_dtypes(include=["int64"])
X = df_temp.drop(["total_sales",],axis=1)  # predictors


# In[29]:


# split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 3)


# # Define the parameter grid

# In[30]:


params = {
    'learning_rate': [0.1, 0.01],
    'max_depth': [3, 6, 9],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}


# # Create an instance of the XGBRegressor class

# In[31]:


xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)


# # Create a GridSearchCV object

# In[32]:


grid_search = GridSearchCV(xgb_model, param_grid=params, cv=5, verbose=2)


# # Fit the GridSearchCV object on the training data

# In[33]:


grid_search.fit(X_train, y_train)


# In[34]:


# Get the best performing model from GridSearchCV
best_model = grid_search.best_estimator_


# In[35]:


# Make predictions on the test data using the best model
y_pred = best_model.predict(X_test)


# In[36]:


# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = best_model.score(X_test, y_test)


# In[37]:


print('MSE:', mse)


# In[38]:


print('R-squared:', r2)


# In[39]:


print('Best hyperparameters:', grid_search.best_params_)


# 
# # Make predictions for the next 3 months

# In[40]:


df = pd.read_csv('cinemaTicket_Ref.csv')


# In[41]:


import pandas as pd

# Create a datetime object from a string
date_str = '2022-01-01'
date = pd.to_datetime(date_str)

# Add a Timedelta object to the datetime object
new_date = date + pd.Timedelta(days=1)

print(new_date)


# In[42]:


future_dates = pd.date_range(start=pd.to_datetime(df['date']).max() + pd.Timedelta(days=1), periods=3, freq='M')


# In[43]:


future_sales = pd.DataFrame({'date': future_dates, 'film_code': 0, 'cinema_code': 0, 'tickets_sold': 0, 'tickets_out': 0, 'show_time': 0, 'ticket_use': 0})


# In[44]:


# Set the date column as the index
future_sales = future_sales.set_index('date')


# # Predict the total sales for the next 3 months using the best model

# In[45]:


future_sales['total_sales'] = best_model.predict(future_sales)
future_sales['total_sales'] = future_sales['total_sales'].astype(str)


# In[46]:


# Combine the test results and future sales dataframes
all_results = pd.concat([df[['date', 'total_sales']], future_sales], axis=0)


# In[47]:


# Set the date column as the index
all_results = all_results.set_index('date')


# In[48]:


print(all_results)


# In[49]:


all_results['total_sales'] = all_results['total_sales'].astype(float)


# In[50]:


print(all_results['total_sales'])


# In[51]:


print(all_results['total_sales'].dtype)


# # Plot the actual and predicted sales for the next 3 months

# In[52]:


all_results.index = all_results.index.astype(str)
all_results['total_sales'] = all_results['total_sales'].astype(float)

plt.figure(figsize=(12,6))
plt.plot(all_results.index, all_results['total_sales'], label='Total Sales')
plt.xlabel('Date')
plt.ylabel('Total Sales')



# In[52]:




