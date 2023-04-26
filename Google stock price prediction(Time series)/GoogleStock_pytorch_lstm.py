#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim


# # Import Datasets

# In[2]:


GTrain = pd.read_csv("Google_Stock_Price_Train.csv")


# In[3]:


GTrain.shape


# In[4]:


GTrain.head()


# In[5]:


GTrain.describe()


# In[6]:


GTrain.isnull().sum()


# In[7]:


GTest = pd.read_csv("Google_Stock_Price_Test.csv")


# In[8]:


GTest.head()


# In[9]:


GTest.describe()


# In[10]:


GTest.shape


# In[11]:


GTest.isnull().sum()


# In[11]:





# # Merge the datasets

# In[12]:


GTrain['Date'] = pd.to_datetime(GTrain['Date'], format='%m/%d/%Y')
GTest['Date'] = pd.to_datetime(GTest['Date'], format='%m/%d/%Y')


# In[13]:


# Replace commas with empty string in GTrain
GTrain = GTrain.replace(',', '', regex=True)

# Replace commas with empty string in GTest
GTest = GTest.replace(',', '', regex=True)


# In[14]:


print(GTrain.dtypes)
print(GTest.dtypes)


# In[15]:


GTrain['Close'] = GTrain['Close'].astype(float)


# In[16]:


GTrain['Volume'] = GTrain['Volume'].astype(float)
GTest['Volume'] = GTest['Volume'].astype(float)


# In[17]:


print(GTrain.dtypes)
print(GTest.dtypes)


# In[18]:


GTrain = pd.read_csv('Google_Stock_Price_Train.csv', index_col='Date', parse_dates=True)
GTest = pd.read_csv('Google_Stock_Price_Test.csv', index_col='Date', parse_dates=True)

df = pd.concat([GTrain, GTest], axis=0)


# In[19]:


df.head()


# In[20]:


df.replace(',', '', regex=True)


# # Graph Of Dataset

# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[22]:


# Define list of y-axis limits for different ranges
y_limits = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 350), (350, 400), (400, 450), (450, 500), (500, 550), (550, 600), (600, 650), (650, 700), (700, 750), (750, 800), (800, 850), (850, 900)]
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])
df['Range'] = pd.cut(df['Close'], bins=[y_min for y_min, y_max in y_limits] + [y_max for y_min, y_max in y_limits[-1:]], include_lowest=True, labels=[f'{y_min}-{y_max}' for y_min, y_max in y_limits])


# Group data by range and plot histogram for each group
plt.figure(figsize=(12,6))
sns.histplot(data=df, x='Date', y='Close', hue='Range', palette='viridis')

# Set title for plot
plt.title('Histogram of Stock Closing Prices by Range')

# Display the plot
plt.show()


# In[23]:


sns.set_style('darkgrid')
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='Date', y='Close', color='red')
plt.xlabel('Date')
plt.ylabel('Close')
plt.title('Histogram of Google Stock Price')
plt.show()


# In[192]:


plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='Date', y='Volume', color='red')
plt.xlabel('Date')
plt.ylabel('Shares')
plt.title('Histogram of Google Stock Shares Traded During the Day')
plt.show()


# In[25]:


from sklearn.preprocessing import MinMaxScaler


# In[26]:


df = df.drop('Range', axis=1)


# In[27]:


df['Volume'] = df['Volume'].str.replace(',', '').astype(float)


# In[28]:


print(df.dtypes)


# In[29]:


df.head()


# In[30]:


df.Volume


# In[31]:


# Normalize the columns
scaler = MinMaxScaler()
normalized_cols = ['Close', 'Open', 'Volume', 'High', 'Low']
df[normalized_cols] = scaler.fit_transform(df[normalized_cols])


# In[32]:


# Concatenate the normalized columns
data = df[normalized_cols]
data.shape


# In[33]:


data.head()


# # Split data into training, validation, and testing sets

# In[34]:


train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)
test_size = int(len(data) * 0.15)


# In[35]:


print(train_size)


# In[36]:


print(val_size)


# In[37]:


print(test_size)


# In[38]:


train_data = data[:train_size]
val_data = data[train_size:train_size+val_size]
test_data = data[-test_size:]


# In[39]:


train_data.shape


# In[40]:


test_data.shape


# # Define a function to create input sequences and corresponding labels

# In[41]:


def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :])
        y.append(data[i, :])
    X = np.array(X)
    y = np.array(y)
    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=2)
    return X, y


# In[42]:


# Define the length of the input sequences
seq_length = 30


# # Create input sequences and labels for training, validation, and testing sets

# In[43]:


X, y = create_sequences(data.values, seq_length)


# In[44]:


train_X = torch.Tensor(X[:train_size])
train_y = torch.Tensor(y[:train_size])

val_X = torch.Tensor(X[train_size:train_size+val_size])
val_y = torch.Tensor(y[train_size:train_size+val_size])

test_X = torch.Tensor(X[train_size+val_size:])
test_y = torch.Tensor(y[train_size+val_size:])


# # Define the LSTM model

# In[180]:


class LSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=60, num_layers=5, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 5)
        self.relu = nn.ReLU()
        self.dense = nn.Linear(5, 5)
        self.activation = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.linear(out[:, -1, :])
        out = self.relu(out)
        out = self.dense(out)
        out = self.activation(out)
        return out


# # Define the model hyperparameters

# In[181]:


input_size = 5 # number of features (Open, High, Low, Close, Volume)
hidden_size = 60 # number of LSTM cells
num_layers = 5# number of LSTM layers
learning_rate = 0.001
num_epochs = 700


# In[182]:


# Initialize the LSTM model with dropout
model = LSTM(input_size, hidden_size, num_layers, dropout=0.1)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)



# In[183]:


train_X = torch.tensor(train_X).float()
train_y = torch.tensor(train_y).float()
val_X = torch.tensor(val_X).float()
val_y = torch.tensor(val_y).float()
test_X = torch.tensor(test_X).float()
test_y = torch.tensor(test_y).float()
# convert the NumPy array to a PyTorch tensor


# In[184]:


print(train_X.shape)


# In[185]:


print(train_X)


# In[186]:


print(torch.min(train_X))
print(torch.max(train_X))


# # Train the model

# In[187]:


# Train the LSTM model
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    # Reset the optimizer gradients
    optimizer.zero_grad()

    # Forward pass
    output = model(train_X)

    # Compute the loss
    loss = criterion(output, train_y)

    # Compute the accuracy
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(train_y.argmax(dim=1, keepdim=True)).sum().item()
    accuracy = correct / len(train_y)

    # Backward pass
    loss.backward()

    # Update the model parameters
    optimizer.step()

    # Print the loss and accuracy every epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')


# # Evaluate the LSTM model on the validation set

# In[188]:


with torch.no_grad():
    # Set the model to evaluation mode
    model.eval()

    # Forward pass
    output = model(val_X)

    # Compute the loss
    loss = criterion(output, val_y)

    # Compute the accuracy
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(val_y.argmax(dim=1, keepdim=True)).sum().item()
    accuracy = correct / len(val_y)

    # Print the loss and accuracy
    print(f'Validation Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')


# # Evaluate the LSTM model on the test set

# In[189]:


with torch.no_grad():
    # Set the model to evaluation mode
    model.eval()

   # Forward pass
    output = model(test_X)

    # Compute the loss
    loss = criterion(output, test_y)

    # Compute the accuracy
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(test_y.argmax(dim=1, keepdim=True)).sum().item()
    accuracy = correct / len(test_y)

    # Print the loss and accuracy
    print(f'Test Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')


# In[ ]:


# putting this out there that since this is not a professional work i did not try too many times to achieve the highest possible accuracy
# you can adjust a number of things to improve it like changing the hidden size, epochs, loss function, trying other activation functions, using a different optimizer
# this is just a way to show that it can be done but it isnt a professional model cause will need a lot more fine tuning to make it at a top level.


# # Save the model

# In[190]:


torch.save(model.state_dict(), 'model.pt')


# # Load the model

# In[191]:


model = LSTM()
model.load_state_dict(torch.load('model.pt'))


# In[196]:


# Load the stock prices dataset
GTrain = pd.read_csv("Google_Stock_Price_Train.csv")
GTest = pd.read_csv("Google_Stock_Price_Test.csv")
GTrain['Date'] = pd.to_datetime(GTrain['Date'], format='%m/%d/%Y')
GTest['Date'] = pd.to_datetime(GTest['Date'], format='%m/%d/%Y')

# Replace commas with empty string in GTrain
GTrain = GTrain.replace(',', '', regex=True)

# Replace commas with empty string in GTest
GTest = GTest.replace(',', '', regex=True)


# In[197]:


GTrain = pd.read_csv('Google_Stock_Price_Train.csv', index_col='Date', parse_dates=True)
GTest = pd.read_csv('Google_Stock_Price_Test.csv', index_col='Date', parse_dates=True)

df = pd.concat([GTrain, GTest], axis=0)
df.head()


# In[198]:


# Replace commas with an empty string in the 'Volume' column
df['Volume'] = df['Volume'].str.replace(',', '')
df.head()


# In[204]:


df.dtypes


# In[206]:


df['Close'] = df['Close'].str.replace(',', '')
df['Close'] = df['Close'].astype(float)
df['Volume'] = df['Volume'].astype(float)


# In[214]:


# Prepare the input data
last_30_days = df['Close'].tail(30).values.reshape(1, 6, 5)
input_tensor = torch.tensor(last_30_days, dtype=torch.float32)


# In[215]:


# Use the trained LSTM model to predict the stock prices for the next 3 months
model.eval()
with torch.no_grad():
    output_tensor = model(input_tensor)
predicted_prices = output_tensor.detach().numpy()


# In[220]:


# Convert the predicted prices to a pandas dataframe
date_range = pd.date_range(start=df.index[-1], periods=90, freq='D')
print(len(predicted_prices[0]))
print(len(date_range))


# In[221]:


df_template = pd.DataFrame({'Close': predicted_prices[0]})
predicted_df = pd.concat([df_template]*18, ignore_index=True)
predicted_df['Date'] = date_range


# In[222]:


# Concatenate the predicted dataframe with the original dataframe
df = pd.concat([df, predicted_df], ignore_index=True)


# In[227]:


# Plot the graph
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
plt.figure(figsize=(12,6))
sns.lineplot(df['Close'], color ='red')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Prediction of Google Stock Price')
plt.show()

