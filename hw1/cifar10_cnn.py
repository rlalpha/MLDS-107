
# coding: utf-8

# In[1]:


from keras.datasets import cifar10
from keras.utils import np_utils
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[2]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[3]:


import hw1_2 as cnn


# In[4]:


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# In[ ]:


cnn.train.counter = 0
cnn.train(x_train, y_train, x_test, y_test, 10, 'cifar')

