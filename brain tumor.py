#!/usr/bin/env python
# coding: utf-8

# In[13]:


import tensorflow as tf
print(tf.__version__)


# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# In[15]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# In[16]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.20
)


# In[17]:


valid_datagen=ImageDataGenerator( rescale=1./255,
    validation_split=0.20)

# In[18]:


train_generator = train_datagen.flow_from_directory( r"C:\Users\ACER\Downloads\dataset\data\Train", target_size=(224,224), batch_size=32, class_mode="categorical", subset="training",shuffle=True )
validation_generator =valid_datagen.flow_from_directory( r"C:\Users\ACER\Downloads\dataset\data\Train", target_size=(224,224), batch_size=32, class_mode="categorical", subset="validation",shuffle=True)
test_generator = train_datagen.flow_from_directory( r"C:\Users\ACER\Downloads\dataset\data\Test", target_size=(224,224), batch_size=32, class_mode="categorical",shuffle=False)

# In[19]:


train_generator.class_indices

# In[20]:


import numpy as np

# Labels of all images
y_train_classes = train_generator.classes

# Unique classes and their counts
classes, counts = np.unique(y_train_classes, return_counts=True)

# Combine into dictionary
class_counts = dict(zip(classes, counts))
print("Class counts (by label):", class_counts)


# In[21]:


!pip install scipy



# In[22]:


import sys
print(sys.executable)


# In[23]:


import scipy
print("Scipy version:", scipy.__version__)


# In[24]:


import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img

base_dir = r"C:\Users\ACER\Downloads\dataset\data\Train"

classes = os.listdir(base_dir)

plt.figure(figsize=(10, 8))

count = 1
for cls in classes:
    cls_path = os.path.join(base_dir, cls)
    images = os.listdir(cls_path)[:5]

    for img in images:
        img_path = os.path.join(cls_path, img)
        image = load_img(img_path, target_size=(128, 128))

        plt.subplot(len(classes), 5, count)
        plt.imshow(image)
        plt.title(cls)
        plt.axis("off")
        count += 1

plt.tight_layout()
plt.show()


# In[25]:


train_generator.samples

# In[26]:


x_batch,y_batch=train_generator[0]
image=x_batch[0]

# In[27]:


df=pd.DataFrame(image[:,:,0])


# In[28]:


pd.set_option("display.max_columns",None)

# In[ ]:




# In[29]:


df


# In[30]:


import seaborn as sns

# In[31]:


sns.heatmap(df,cmap="Greys")


# In[32]:


from tensorflow.keras.applications import MobileNetV2

# In[33]:


"""from sklearn.utils.class_weight import compute_class_weight
classes = list(train_generator.class_indices.keys()) 
y_train_classes = train_generator.classes 
class_weights = compute_class_weight( class_weight='balanced', classes=np.unique(y_train_classes), y=y_train_classes )
class_weights = dict(zip(np.unique(y_train_classes), class_weights))
print("Class weights:", class_weights)"""

# In[34]:


base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

# In[ ]:




# In[35]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# In[36]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,  MaxPooling2D,
    BatchNormalization, GlobalAveragePooling2D,
    Dense, Dropout
)

model=Sequential([
Input(shape=(224,224,3)),
base_model,
MaxPooling2D(2,2),
Conv2D(64,(3,3),activation='relu',padding='same'),
GlobalAveragePooling2D(),
Dense(128,activation='relu'),
Dropout(0.5),
Dense(4,activation='softmax')])


# In[37]:


base_model.trainable = True

for layer in base_model.layers[:-35]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[38]:


class_weights = {
    0: 0.80,  
    1: 1.10,  
    2: 1.24,  
    3: 0.97   
}

# In[39]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint(
    "models/final_image_classifier.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
Epochs=25
model_training=model.fit(train_generator,validation_data=validation_generator,class_weight=class_weights,epochs=Epochs, callbacks=[checkpoint,early_stop] )


# In[40]:



test_loss, test_acc = model.evaluate(test_generator)
print("Test accuracy:", test_acc)

# In[41]:


y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)

y_true = test_generator.classes

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))


# In[42]:


import os
os.makedirs("models", exist_ok=True)


# In[43]:


model.save("models/final_image_classifier.keras")


# In[44]:


from tensorflow.keras.models import load_model

model = load_model("models/final_image_classifier.keras")


# In[ ]:





# In[ ]:










