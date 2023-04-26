#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


# # Import Dataset

# In[2]:


df = pd.read_csv('brent.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[8]:


df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')


# In[9]:


sns.set_style('darkgrid')
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='Date', y='Price', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Graph of Brent Crude Oil Prices')
plt.show()


# In[10]:


df.dtypes


# In[19]:


df = pd.read_csv('brent.csv', index_col="Date")


# In[21]:


df.head()


# # Splitting dataset

# In[22]:


from sklearn.model_selection import train_test_split

# Split the data into training, validation, and test sets
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)
test_size = len(df) - train_size - val_size


# In[23]:


print('Train Size=', train_size)
print('Validation Size=', val_size)
print('Test Size=', test_size)


# In[24]:


train_data, val_data, test_data = np.split(df, [train_size, train_size + val_size])


# # Scale the Data

# In[25]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
val_data = scaler.transform(val_data)
test_data = scaler.transform(test_data)


# # Define the number of time steps and features

# In[26]:


time_steps = 50
features = 1


# # Creating the Sequence

# In[27]:


# Create training sequences
train_sequences = []
train_labels = []
for i in range(time_steps, len(train_data)):
    train_sequences.append(train_data[i - time_steps:i])
    train_labels.append(train_data[i])
train_sequences = np.array(train_sequences)
train_labels = np.array(train_labels)


# In[28]:


# Create validation sequences
val_sequences = []
val_labels = []
for i in range(time_steps, len(val_data)):
    val_sequences.append(val_data[i - time_steps:i])
    val_labels.append(val_data[i])
val_sequences = np.array(val_sequences)
val_labels = np.array(val_labels)


# In[29]:


#Create test sequences
test_sequences = []
test_labels = []
for i in range(time_steps, len(test_data)):
    test_sequences.append(test_data[i - time_steps:i])
    test_labels.append(test_data[i])
test_sequences = np.array(test_sequences)
test_labels = np.array(test_labels)


# # Define the LSTM model

# In[33]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping


# In[31]:


# Define the model architecture
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, features)))
model.add(Dropout(0.1))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(units=50))
model.add(Dropout(0.1))
model.add(Dense(units=1, activation='sigmoid'))


# # Compile the model

# In[39]:


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


# # Define early stopping

# In[40]:


early_stopping = EarlyStopping(monitor='val_loss', patience=10)


# # Train the model

# In[41]:


hist = model.fit(train_sequences, train_labels, epochs=100, batch_size=32, validation_data=(val_sequences, val_labels), callbacks=[early_stopping])


# # Evaluate the test set

# In[42]:


test_loss, test_mae = model.evaluate(test_sequences, test_labels)
print('Test Loss:', test_loss)
print('Test MAE:', test_mae)


# # Graph of the Predicted Test Price vs the Actual Price

# In[45]:


# Get the predicted values for the test set
y_pred = model.predict(test_sequences)

# Plot the predicted vs actual values
plt.figure(figsize=(12,6))
plt.plot(test_labels, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()


# # Save the Model

# In[46]:


model.save('model.h5')


# 
