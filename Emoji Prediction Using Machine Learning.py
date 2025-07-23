#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


# load the train data:

train_data = pd.read_csv('train_emoji.csv', header=None)
train_data.head()


# In[4]:


# Load the test data:

test_data = pd.read_csv('test_emoji.csv', header= None)
test_data.head()


# In[5]:


# drop columns 2 and 3 from our train data:

train_data.drop(labels = [2,3], axis = 1, inplace = True)
train_data.head()


# In[6]:


import emoji

emoji.EMOJI_UNICODE['en']


# In[7]:


emoji.emojize(':Virgo')


# In[12]:


emoji_dict = {
    '0' : ':beating_heart:',
    '1' : ':baseball:',
    '2' : ':face_with_tears_of_joy:',
    '3' : ':confounded_face:',
    '4' : ':face_savoring_food:'
}


# In[13]:


for e in emoji_dict.values():
    print(emoji.emojize(e), end = ' ')


# In[16]:


# pre processing:

X_train = train_data[0].values
Y_train = train_data[1].values


# In[17]:


X_train[:10]


# In[18]:


Y_train[:10]


# In[19]:


X_train.shape , Y_train.shape


# In[22]:


# We are embedding the text as we are going to create RNN model:

f = open('glove.6B.50d.txt' , encoding = 'utf8', mode = 'r')


# In[23]:


embedding_matrix = {}


# In[24]:


for line in f:
    values = line.split()
    word = values[0]
    emb = np.array(values[1:], dtype = 'float')
    
    embedding_matrix[word] = emb


# In[25]:


embedding_matrix


# In[28]:


# We will create a function which will give embedding of our text data:

def get_embedding_matrix_for_data(data):
    max_len = 10
    embedding_data = np.zeros( (len(data), max_len, 50))
    
    for x in range(data.shape[0]):
        word_in_sen = data[x].split()
        
        for y in range(len(word_in_sen)):
            if embedding_matrix.get(word_in_sen[y].lower()) is not None:
                embedding_data[x][y] = embedding_matrix[word_in_sen[y].lower()]
                
    return embedding_data
                
                


# In[29]:


X_train = get_embedding_matrix_for_data(X_train)


# In[30]:


X_train.shape


# In[32]:


# covert the output to categorical:

import tensorflow
from keras.utils.np_utils import to_categorical


# In[33]:


Y_train = to_categorical(Y_train)


# In[34]:


Y_train


# In[38]:


# Create Model:

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, SimpleRNN, LSTM, Activation


# In[45]:


# build our model:

model = Sequential()
model.add(LSTM(64, input_shape = (10,50), return_sequences = True))

model.add(Dropout(0.5))

model.add(LSTM(128 , return_sequences=False))
model.add(Dropout(0.5))


model.add(Dense(5))
model.add(Activation('softmax'))


# In[46]:


model.summary()


# In[47]:


model.compile(optimizer = 'adam', loss= keras.losses.categorical_crossentropy, metrics= ['acc'])


# In[48]:


# model training:

history = model.fit(X_train, Y_train, validation_split = 0.2, batch_size=32, epochs = 50 )


# In[52]:


# plot acuuracy and loss graph:

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(8,6))
plt.title('ProjectGurukul Accuracy scores')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['accuracy', 'val_accuracy'])
plt.show()

plt.figure(figsize=(8,6))
plt.title('ProjectGurukul Loss value')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()


# In[53]:


model.evaluate(X_train, Y_train)[1]


# In[55]:


# preparing test data:

test_data[0] = test_data[0].apply(lambda x: x[:-1])

X_test = test_data[0].values
Y_test = test_data[1].values


# In[56]:


X_test = get_embedding_matrix_for_data(X_test)
Y_test = to_categorical(Y_test)


# In[57]:


model.evaluate(X_test, Y_test)[1]


# In[58]:


Y_pred = model.predict_classes(X_test)

for t in range(len(test_data)):
    print(test_data[0].iloc[t])
    print('predictions: ', emoji.emojize(emoji_dict[str(Y_pred[t])]))
    print('Actual: ',emoji.emojize(emoji_dict[str(test_data[1].iloc[t])]) )
    
    


# In[ ]:




