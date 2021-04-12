from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from flask import Flask
import numpy as np
import pandas as pd
import os


app = Flask(__name__)


@app.route('/hello')
def home():
    return 'hello'


@app.route('/rf/<sym1>/<sym2>/<sym3>/<sym4>/<sym5>')
def predict(sym1, sym2, sym3, sym4, sym5):

# from gui_stuff import *

    l1=['back pain','constipation','abdominal pain','diarrhoea','mild fever','yellow urine',
        'acute liver failure','swelling of stomach','swelled lymph nodes','blurred and distorted vision',
        'phlegm','throat irritation','redness of eyes','runny nose','congestion',
        'chest pain','weakness in limbs','fast heart rate','bloody stool','irritation in anus','neck pain',
        'dizziness','cramps','bruising','obesity',
        'enlarged thyroid','excessive hunger','drying and tingling lips','knee pain','swelling joints','loss of balance','weakness of one body side',
        'loss of smell','bladder discomfort','foul smell of urine','internal itching',
        'depression','irritability','muscle pain','red spots over body',
        'abnormal menstruation','increased appetite','lack of concentration','visual disturbances','stomach bleeding','painful walking',
        'pus filled pimples','blackheads','skin peeling','red sore around nose']

    disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
        'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
        ' Migraine','Cervical spondylosis',
        'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Denggi','Typhoid','hepatitis A',
        'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
        'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
        'Heartattack','Varicose Veins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
        'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
        'Impetigo']

    l2=[]
    for x in range(0,len(l1)):
        l2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
    df=pd.read_csv("Training.csv")

    df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
        'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
        'Migraine':11,'Cervical spondylosis':12,
        'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Denggi':17,'Typhoid':18,'hepatitis A':19,
        'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
        'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose Veins':30,'Hypothyroidism':31,
        'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
        '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
        'Impetigo':40}},inplace=True)

# print(df.head())

    X= df[l1]

    y = df[["prognosis"]]
    np.ravel(y)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
    tr=pd.read_csv("Testing.csv")
    tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
        'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
        'Migraine':11,'Cervical spondylosis':12,
        'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Denggi':17,'Typhoid':18,'hepatitis A':19,
        'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
        'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose Veins':30,'Hypothyroidism':31,
        'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
        '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
        'Impetigo':40}},inplace=True)

    X_test= tr[l1]
    y_test = tr[["prognosis"]]
    np.ravel(y_test)
# ------------------------------------------------------------------------------------------------------

    #def DecisionTree():

    from sklearn.ensemble import RandomForestClassifier
    clf3 = RandomForestClassifier()
    clf3 = clf3.fit(X,np.ravel(y))
    # calculating accuracy-------------------------------------------------------------------
    #from sklearn.metrics import accuracy_score
    #y_pred=clf3.predict(X_test)
    #print(accuracy_score(y_test, y_pred))
    #print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------
    Symptom1 = '%s' % sym1
    
    Symptom2 = '%s' % sym2
    
    Symptom3 = '%s' % sym3
    
    Symptom4 = '%s' % sym4
    
    Symptom5 = '%s' % sym5
    
    
    psymptoms = [Symptom1,Symptom2,Symptom3,Symptom4,Symptom5]

    for k in range(0,len(l1)):
    # print (k,)
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]
    #print(predicted)
    #assigning a string value to "a"
    for a in range(0,len(disease)):
        if(predicted == a):
            break
    #Then comparing it to the disease list to get at the position of "a"
    #from the disease list    
    return disease[a]#returning the disease
    
if __name__ == '__main__':
    app.run(debug=True)
