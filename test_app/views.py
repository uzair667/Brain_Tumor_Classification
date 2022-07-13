from cProfile import label
from calendar import firstweekday
from operator import mod
from pyexpat import model
from unittest import result
from cv2 import FileStorage, log
from django import http
from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader
# from test_app.models import Topic,webpage,Access_Record

import cv2
import os
import tensorflow as tf
from tensorflow import keras
import PIL
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import normalize
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from django.core.files.storage import FileSystemStorage
from django.conf import settings


def hello(request):
    return render(request,'test.html')

def test(request):

    image_directory="E:\\Django\\test_django\\test_project\\BTC_IMG_DTST\\Training"    
    
    
    glioma_tumor_images = os.listdir(image_directory+ "\\glioma_tumor")
    meningioma_tumor_images = os.listdir(image_directory+ "\\meningioma_tumor")
    no_tumor_images = os.listdir(image_directory+ "\\no_tumor")
    pituitary_tumor_images = os.listdir(image_directory+ "\\pituitary_tumor")
    dataset=[]
    label=[]

    INPUT_SIZE=64

    for i , image_name in enumerate(glioma_tumor_images):
        if(image_name.split('.')[1]=='jpg'):
            image=cv2.imread(image_directory+'/glioma_tumor/'+image_name)
            image=Image.fromarray(image , 'RGB')
            image=image.resize((INPUT_SIZE,INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(0)

    for i , image_name in enumerate(meningioma_tumor_images):
        if(image_name.split('.')[1]=='jpg'):
            image=cv2.imread(image_directory+'/meningioma_tumor/'+image_name)
            image=Image.fromarray(image, 'RGB')
            image=image.resize((INPUT_SIZE,INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(1)

    for i , image_name in enumerate(no_tumor_images):
        if(image_name.split('.')[1]=='jpg'):
            image=cv2.imread(image_directory+'/no_tumor/'+image_name)
            image=Image.fromarray(image,'RGB')
            image=image.resize((INPUT_SIZE,INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(2)

    for i , image_name in enumerate(pituitary_tumor_images):
        if(image_name.split('.')[1]=='jpg'):
            image=cv2.imread(image_directory+'/pituitary_tumor/'+image_name)
            image=Image.fromarray(image,'RGB')
            image=image.resize((INPUT_SIZE,INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(3)                

    dataset=np.array(dataset)
    label=np.array(label)


    x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)

    x_train=normalize(x_train, axis=1)
    x_test=normalize(x_test, axis=1)

    y_train=to_categorical(y_train , num_classes=4)
    y_test=to_categorical(y_test , num_classes=4)



    # Model Building
    # 64,64,3

    model=Sequential()

    model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))


    # Binary CrossEntropy= 1, sigmoid
    # Categorical Cross Entryopy= 2 , softmax

    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])


    model.fit(x_train, y_train, 
    batch_size=16, 
    verbose=1, epochs=32, 
    validation_data=(x_test, y_test),
    shuffle=False)


    model.save('BrainTumor32EpochsCategorical')
    return HttpResponse("hello trained and saved successfully!!")



def check(request):        

    if request.method == "POST":
        file_name = request.FILES['test_file']
        fss = FileSystemStorage()
        file = fss.save(file_name.name, file_name)
        file_url = fss.url(file)
        file_full_path = os.path.abspath(file_url)


        model = load_model("BrainTumor32EpochsCategorical")
        image=cv2.imread(fss.base_location + file_name.name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64,64))
        image=np.array(image)
        input_img = np.expand_dims(image, axis=0)
        result=model.predict(input_img)
        result = np.argmax(result,axis=1)[0]
        print(result)



        if result == 0:
            return HttpResponse("Model predicts that it is a Glioma Tumor!!")
        elif result == 1:
            return HttpResponse("Model predicts that it is a Meningioma Tumor!!")
        elif result == 2:
            return HttpResponse("Model predicts that there is NO tumor!!")
        elif result == 3 :
            return HttpResponse("Model predicts that it is a Pituitary Tumor!!")   
        else:
            return HttpResponse("no result found")


    template = loader.get_template('check/result.html')
    context = {
        'Results': result,
    }
    
    return HttpResponse(template.render(context,request))        