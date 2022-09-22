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

#Preprocessing
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.feature_extraction.text import TfidfTransformer

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class accuracy_calculator:
    def __init__(self, text_path, model_path, scale) -> None:
        self.text_path = text_path
        self.model_path = model_path #this is the path of the directory with all the model
        
        self.scale = scale #percentage of the data to get

    
    def load_data(self):
        data = pd.read_excel(self.text_path)
        X = data["cleaned_website_text"]
        y = data["Category"]

        self.N_CLASSES = data['Category'].nunique()
        self.CLASS_LABELS= data['Category'].unique()

        stemmer = WordNetLemmatizer()
        documents = []


        for sen in range(0, len(X)):
            #  Remove all the special characters
            document = re.sub(r'\W', ' ', str(X[sen]))
    
            #       remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
            #   Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
            #       Substituting multiple spaces with single space
            document = re.sub(r'\s+', ' ', document, flags=re.I)
    
            # Removing prefixed 'b'
            document = re.sub(r'^b\s+', '', document)
    
            # Converting to Lowercase
            document = document.lower()
    
            documents.append(document)
        
        count_vect = CountVectorizer(stop_words="english")
        X_train_counts = count_vect.fit_transform(documents)

        tfidf_transformer = TfidfTransformer()
        X = tfidf_transformer.fit_transform(X_train_counts)

        self.X_val ,self.y_val = shuffle(X, y)

    def adaboost_accuracy(self, model_name):
        #define the parameter
        model_category = "ADABOOST"
        adaboost_model_path = f"{self.model_path}/{model_category}/{model_name}.sav"

        #define the model
        model = RandomForestClassifier(random_state=101)

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
        y_pred = model.predict(self.X_val)
        print(f"{model_category} model Accuracy:", "%.2f" % ( metrics.accuracy_score(self.y_val, y_pred) * 100))

    def svm_accuracy(self, model_name):
        #define the parameter
        model_category = "SVM"
        svm_model_path = f"{self.model_path}/{model_category}/{model_name}.sav"

        #define the model
        model = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
    
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
        y_pred = model.predict(self.X_val)
        print(f"{model_category} model Accuracy:", "%.2f" % ( metrics.accuracy_score(self.y_val, y_pred) * 100))

    def nn_accuracy(self, model_name):
        #define the parameter
        model_category = "NN"
        nn_model_path = f"{self.model_path}/{model_category}/{model_name}_model.h5"

        #preprocessing
        le = preprocessing.LabelEncoder()
        y_val_ = le.fit(self.y_val)
        y_val_ = le.transform(self.y_val)

        #convert the sparse matrix to a dense matrix
        X_val_ = self.X_val.toarray().astype(float)
        

        model = keras.Sequential([
            keras.layers.Input(shape=X_val_.shape[1]),
            keras.layers.Dense(1280, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(640, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(320, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(self.N_CLASSES, activation="softmax"),
        ])

        model.compile(loss='sparse_categorical_crossentropy', optimizer="Adam", metrics=["accuracy"])
    
        #load the weig ht
        print(F"Loading {model_category} {model_name} --")
        exists = os.path.isfile(nn_model_path)
        if exists:
            with open(nn_model_path, "rb") as sav_file:
                model = keras.models.load_model(nn_model_path)
        else:
            raise Exception("{model_category} model unavailable, check the path")
    
        print(F"{model_category} model loaded --\nStart Prediction")
        #predict the results
        test_loss, test_acc = model.evaluate(X_val_, y_val_, verbose=0)
        print(f"{model_category} {model_name} Accuracy: %.2f {test_acc * 100}")

    def de_accuracy(self, model_name):
        #define the parameter
        model_category = "DE"
        de_model_path = f"{self.model_path}/{model_category}/{model_name}.sav"

        #preprocessing
        le = preprocessing.LabelEncoder()

        y_val_ = le.fit(self.y_val)

        y_val_ = le.transform(self.y_val)
        
        #define the model
        model = ComplementNB()
    
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
        y_pred = model.predict(self.X_val)
        print(f"{model_category} {model_name} Accuracy:", "%.2f" % ( metrics.accuracy_score(y_val_, y_pred) * 100))


def run(images_path, model_path, scale):
    ac = accuracy_calculator(images_path, model_path, scale)
    print(f"Load text---\n")
    ac.load_data()
    print("Text Loaded---\n")

    #define the name of the models
    MODEL_NAME_ADABOOST = "ADABOOST_random_forest"
    MODEL_NAME_SVM = "SVM_SGDClassifier"
    MODEL_NAME_NN = "NN_dense_3"
    MODEL_NAME_DE = "ComplementNB"


    #compute the accuracy
    ac.adaboost_accuracy(MODEL_NAME_ADABOOST) #OK
    ac.svm_accuracy(MODEL_NAME_SVM)  # OK
    ac.nn_accuracy(MODEL_NAME_NN) #
    ac.de_accuracy(MODEL_NAME_DE)  #ok


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

python accuracy_text_script.py --scale 0.05  "/home/giuseppe/Scrivania/universita/Magistrale/Machine and Deep Learning/Progetto ML/Dataset/testi-3/testi-3/testi-3.xlsx" "/home/giuseppe/Scrivania/model_accuracy/models_text"'''