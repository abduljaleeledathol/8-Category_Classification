#!/usr/bin/env python
# coding: utf-8

# In[1]:
print("Started")

from keras.layers import *
from keras.models import *
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import keras
import numpy as np
import glob
#from glob import glob
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


# In[2]:
# Loading training data


class_dic={

'class1': glob.glob('train8/Meter/*.*'),
'class2': glob.glob('train8/Panel/*.*'),
'class3': glob.glob('train8/Bill/*.*'),
'class4': glob.glob('train8/Home/*.*'),
'class5': glob.glob('Augdataset/Breaker/*.*'),
'class6': glob.glob('Augdataset/Attic/*.*'),
'class7': glob.glob('Augdataset/Raffers/*.*'),
'class8': glob.glob('Augdataset/Rafferspc/*.*'),

}


# In[3]:


class_lab = {
    'class1' :0,
    'class2':1,
    'class3':2,
    'class4':3,
    'class5':4,
    'class6':5,
    'class7':6,
    'class8':7,

    }


# In[4]:


for class_name, images in class_dic.items():
  print(class_name)
  print(len(images))
  # 


# In[ ]:


x,y=[],[]

for class_name,images in class_dic.items():
  i=0
  for image in images:
    imag=cv2.imread(image)
    res_img=cv2.resize(imag,(224,224))
    x.append(res_img)
    y.append(class_lab[class_name])
    i=i+1
    if i==500:
      break
   


# In[ ]:


x=np.array(x)
y=np.array(y)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# In[ ]:


x_trainf=x_train/255
x_testf=x_test/255


# In[ ]:



num_class=8
# re-size all the images to this
IMAGE_SIZE = [224, 224]


# In[ ]:


# Define the Model
# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[ ]:


#don't train existing weights
for layer in vgg.layers:
  layer.trainable = False


# In[ ]:


# our layers - you can add more if you want
x = Flatten()(vgg.output)


# In[ ]:


# x = Dense(1000, activation='relu')(x)
prediction = Dense(num_class, activation='softmax')(x)


# In[ ]:


# create a model object
model = Model(inputs=vgg.input, outputs=prediction)


# In[ ]:


# view the structure of the model
model.summary()


# In[ ]:


model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])


# In[ ]:


#model.fit(x_trainf,y_train,epochs=30)
model.fit(x_trainf,y_train,epochs=30)


# In[ ]:


model.evaluate(x_testf,y_test)


# In[ ]:


get_ipython().system('pip install pyyaml h5py  # Required to save models in HDF5 format')
model.save('saved_model/my_model') 


# In[ ]:


tf.keras.models.save_model(model,'/content/drive/MyDrive/Savedmodelmc/my_model_c.hdf5')


# In[ ]:
# Printing confusion matrix for training data

predictionstr=model.predict(x_trainf)
pretr=np.argmax(predictionstr, axis=1)
print(confusion_matrix(y_train, pretr))


# In[ ]:

# Printing confusion matrix for validation data

predictionst=model.predict(x_testf)
pret=np.argmax(predictionst, axis=1)
print(confusion_matrix(y_test, pret))


# In[5]:
# load test data

class_dict={

'class1t': glob.glob('test8/Meter/*.*'),
'class2t': glob.glob('test8/Panel/*.*'),
'class3t': glob.glob('test8/Bill/*.*'),
'class4t': glob.glob('test8/Home/*.*'),
'class5t': glob.glob('test8/Breaker/*.*'),
'class6t': glob.glob('test8/Attic/*.*'),
'class7t': glob.glob('test8/Raffers/*.*'),
'class8t': glob.glob('test8/Rafferspc/*.*'),
}


# In[6]:


class_labt = {
   'class1t' :0,
    'class2t':1,
    'class3t':2,
    'class4t':3,
    'class5t':4,
    'class6t':5,
    'class7t':6,
    'class8t':7,
    }


# In[7]:


for class_namet, imagest in class_dict.items():
  print(class_namet)
  print(len(imagest))


# In[ ]:


xt,yt=[],[]

for class_namet,imagest in class_dict.items():
  for imaget in imagest:
    imagt=cv2.imread(imaget)
    res_imgt=cv2.resize(imagt,(224,224))
    xt.append(res_imgt)
    yt.append(class_labt[class_namet])
    


# In[ ]:


xt=np.array(xt)
yt=np.array(yt)


# In[ ]:


x_t=xt/255


# In[ ]:
# Printing condusion matrix for test data

predictionstt=model.predict(x_t)
prett=np.argmax(predictionstt, axis=1)
print(confusion_matrix(yt, prett))

