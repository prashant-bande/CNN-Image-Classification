#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout


columns = ["Prashant", "Nita", "Saanvi"]
#print(columns)

datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(r'C:\Users\saanvi\Desktop\IMDataset\Dataset\Training',
                                                 target_size = (100, 100),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
valid_generator = test_datagen.flow_from_directory(r'C:\Users\saanvi\Desktop\IMDataset\Dataset\Training',
                                            target_size = (100, 100),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (100, 100, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation = 'sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=250,
    epochs=1,
    validation_data=valid_generator,
    validation_steps=100)

  

model.save('image_rec_test.h5')
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r'C:\Users\saanvi\Desktop\IMDataset\Dataset\Testing\Nita\IMG_20180813_172618.jpg', target_size = (100, 100, 3))
test_image = image.img_to_array(test_image)
test_image = test_image/225
#test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image.reshape(1, 100, 100, 3))
print(result)
pred_bool = (result >0.5)
print(pred_bool)
predictions = pred_bool.astype(int)
print(predictions)
for i in predictions:
    for a,b in zip(i, columns):
        print(a,b)
