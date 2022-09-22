#Import 
import pathlib
import os
import sys
import json
import argparse
import pandas as pd
from sklearn.utils import shuffle
import json
import tensorflow as tf
from tensorflow import keras
from keras import utils, layers, metrics
from keras.models import Sequential
import pickle
import pandas as pd
import cv2
import numpy as np
import os
from sklearn import preprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load model libraries
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Import Support Vector Classifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix

#models
from sklearn.neighbors import KNeighborsClassifier


#DEFINE SOME CONSTANTS
BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128

class accuracy_calculator:
    def __init__(self, images_path, model_path, scale) -> None:
        self.path = images_path
        self.model_path = model_path #this is the path of the directory with all the model
        
        self.scale = scale #percentage of the data to get

    def load_data_for_NN(self):
        data_dir = pathlib.Path(self.path)
        self.val_ds = tf.keras.utils.image_dataset_from_directory(data_dir,seed=123,image_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH_SIZE)

        #set class number
        self.class_names = self.val_ds.class_names
        self.class_number = len(self.class_names)
    
    def load_data_for_other_analysis(self):
        images = []
        labels = []
        flag = cv2.IMREAD_GRAYSCALE
        for class_dir in os.listdir(self.path): #for each class name dir
            for filename in os.listdir(os.path.join(self.path, class_dir)): # for each images
                img = cv2.imread(os.path.join(self.path, class_dir,filename), flag) #load in greyscale
                new_array=cv2.resize(img,(IMG_HEIGHT,IMG_WIDTH))
                if img is not None:
                    images.append(new_array)
                    labels.append(class_dir)
            print(f"{class_dir} loaded")
        

        #get only a percentage of the data
        images = images[0:int(len(images) * self.scale)]
        labels = labels[0:int(len(labels) * self.scale)]
        

        #split train and validation set 
        self.X_val, self.y_val = shuffle(images, labels)

        #print the shapes
        print(f"\nLen X: {len(self.X_val)}")
        print(f"Len y: {len(self.y_val)}")

    def adaboost_accuracy(self, model_name):
        #define the parameter
        model_category = "ADABOOST"
        adaboost_model_path = f"{self.model_path}/{model_category}/{model_name}.sav"

        # convert dataset in order to be used for AdaBoost
        X_val_AB = []
        for i in range(len(self.X_val)):
            X_val_AB.append(self.X_val[i].flatten())

        #define the model
        svc=SVC(probability=True, kernel='linear')
        model = AdaBoostClassifier(n_estimators=20, base_estimator=svc,learning_rate=1, random_state=101)    
    
        #load the weight
        print(F"Loading {model_category} model --")
        exists = os.path.isfile(adaboost_model_path)
        if exists:
            with open(adaboost_model_path, "rb") as sav_file:
                model = pickle.load(sav_file)
        else:
            raise Exception("{model_category} model unavailable, check the path")
    
        print(F"{model_category} model loaded --\nStart Prediction")
        #predict the results
        y_pred = model.predict(X_val_AB)
        print(f"{model_category} model Accuracy:", "%.2f" % ( metrics.accuracy_score(self.y_val, y_pred) * 100))

    def svm_accuracy(self, model_name):
        #define the parameter
        model_category = "SVM"
        svm_model_path = f"{self.model_path}/{model_category}/{model_name}.sav"

        # convert dataset in order to be used for AdaBoost
        X_val_SVM = []
        for i in range(len(self.X_val)):
            X_val_SVM.append(self.X_val[i].flatten().astype("float32")/255)

        y_val_SVM = [self.class_names.index(i) for i in self.y_val]

        #convert the type
        y_val_SVM = np.array(y_val_SVM)


        #define the model
        model = SVC(kernel='linear',gamma='auto', random_state=101)
    
        #load the weight
        print(F"Loading {model_category} model --")
        exists = os.path.isfile(svm_model_path)
        if exists:
            with open(svm_model_path, "rb") as sav_file:
                model = pickle.load(sav_file)
        else:
            raise Exception("{model_category} model unavailable, check the path")
    
        print(F"{model_category} model loaded --\nStart Prediction")
        #predict the results
        y_pred = model.predict(X_val_SVM)
        print(f"{model_category} model Accuracy:", "%.2f" % ( metrics.accuracy_score(y_val_SVM, y_pred) * 100))

    def cnn_accuracy(self, model_name):
        #define the parameter
        model_category = "NN"
        cnn_model_path = f"{self.model_path}/{model_category}/{model_name}_model.h5"

        #define the model
        data_augmentation = keras.Sequential(
        [layers.RandomFlip("horizontal",input_shape=(IMG_HEIGHT,IMG_WIDTH,3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        ])

        model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(self.class_number)
        ])

        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
        #load the weig ht
        print(F"Loading {model_category} {model_name} --")
        exists = os.path.isfile(cnn_model_path)
        if exists:
            with open(cnn_model_path, "rb") as sav_file:
                model = keras.models.load_model(cnn_model_path)
        else:
            raise Exception("{model_category} model unavailable, check the path")
    
        print(F"{model_category} model loaded --\nStart Prediction")
        #predict the results
        test_loss, test_acc = model.evaluate(self.val_ds, verbose=0)
        print(f"{model_category} {model_name} Accuracy: % {test_acc * 100}")

    def de_accuracy(self, model_name):
        #define the parameter
        model_category = "DE"
        de_model_path = f"{self.model_path}/{model_category}/{model_name}.sav"

        # convert dataset in order to be used for AdaBoost
        X_val_DE = []
        for i in range(len(self.X_val)):
            X_val_DE.append(self.X_val[i].flatten().astype("float32"))

        
        y_val_DE = [self.class_names.index(i) for i in self.y_val]

        #convert the type
        y_val_DE = np.array(y_val_DE)

        
        #define the model
        model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    
        #load the weight
        print(F"Loading {model_category} model --")
        exists = os.path.isfile(de_model_path)
        if exists:
            with open(de_model_path, "rb") as sav_file:
                model = pickle.load(sav_file)
        else:
            raise Exception("{model_category} model unavailable, check the path")
    
        print(F"{model_category} model loaded --\nStart Prediction")
        #predict the results
        y_pred = model.predict(X_val_DE)
        print(f"{model_category} {model_name} Accuracy:", "%.2f" % ( metrics.accuracy_score(y_val_DE, y_pred) * 100))


def run(images_path, model_path, scale):
    ac = accuracy_calculator(images_path, model_path, scale)
    print(f"Load images---\n")
    ac.load_data_for_NN()
    ac.load_data_for_other_analysis()
    print("Images Loaded---\n")

    #define the name of the models
    MODEL_NAME_ADABOOST = "ADABOOST_SCV"
    MODEL_NAME_SVM = "SVM_linear_best"
    MODEL_NAME_CNN = "NN_CNN_2"
    MODEL_NAME_DE = "NearestNeighbors"


    #compute the accuracy
    ac.adaboost_accuracy(MODEL_NAME_ADABOOST)  
    ac.svm_accuracy(MODEL_NAME_SVM)  
    ac.cnn_accuracy(MODEL_NAME_CNN) 
    ac.de_accuracy(MODEL_NAME_DE)  


def parseArguments():
    #create argument Parser
    parser = argparse.ArgumentParser()

    #positional mandatory argument
    parser.add_argument("images_path", help="path of the directory with all the images", type=str)
    parser.add_argument("models_path", help="path of the directory with all the images", type=str)

    #optional argument
    parser.add_argument("-s", "--scale" , help="Percentage of the data to get, is a value between (0,1)", type=float, default=1)


    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')

    # Parse arguments
    args = parser.parse_args()

    return args

if __name__=="__main__":
    #MODEL_PATH = "/home/giuseppe/Scrivania/model_accuracy/models_images" #TODO MODIFICARE
    args = parseArguments()

    #Raw print arguments
    print("You are running the script with arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))

    #run function
    run(args.images_path, args.models_path, args.scale)


'''
to run:

 python accuracy_images_script.py --scale 0.05  "/home/giuseppe/Scrivania/universita/Magistrale/Machine and Deep Learning/Progetto ML/Dataset/immagini-3/immagini-3" "/home/giuseppe/Scrivania/model_accuracy/models_images"
'''