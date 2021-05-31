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

    l1=['sakit belakang','sembelit','sakit perut','cirit-birit','demam ringan','air kencing kuning',
        'kegagalan hati akut','bengkak perut','nod limfa bengkak','penglihatan kabur',
        'kahak','kerengsaan tekak','kemerahan mata','hingus','kesesakan nafas',
        'sakit dada','anggota badan berasa lemah','kadar degupan jantung yang laju','najis berdarah','kerengsaan anus','sakit leher',
        'pening kepala','kekejangan','lebam','obesiti',
        'pembesaran tiriod','kelaparan berlebihan','bibir kering','sakit lutut','sendi bengkak','hilang keseimbangan','lemah satu sisi badan',
        'hilang deria bau','ketidakselesaan pundi kencing','air kencing busuk','gatal dalaman',
        'kemurungan','kerengsaan','kesakitan otot','tompok merah atas badan',
        'haid tidak normal','peningkatan selera makan','kurang tumpuan','gangguan visual','pendarahan perut','sakit semasa berjalan',
        'jerawat dipenuhi nanah','bintik hitam','kulit mengupas','ruam merah sekeliling hidung']

    disease=['Jangkitan kulat','Alergi','GERD','Kolestasis kronik','Tindak balas dadah',
        'Penyakit ulser peptik','AIDS','Diabetes','Radang perut dan usus','Asma bronkial','Tekanan darah tinggi',
        'Migrain','Spondylosis serviks',
        'Lumpuh (pendarahan otak)','Jaundis','Malaria','Cacar air','Denggi','Typhoid','hepatitis A',
        'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Hepatitis beralkohol','Tuberkulosis',
        'Demam selesema','Radang paru-paru','Buasir dimorphic',
        'Serangan jantung','Vena varikos','Hipotiroidisme','Hipertiroidisme','hipoglycemia','Osteoarthristis',
        'Artritis','(vertigo) Posisi Parosymsal Vertigo','Jerawat','Jangkitan saluran kencing','Psoriasis',
        'Impetigo']

    l2=[]
    for x in range(0,len(l1)):
        l2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
    df=pd.read_csv("Training.csv")

    df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
        'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
        'Migraine':11,'Cervical spondylosis':12,
        'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
        'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
        'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
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
        'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
        'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
        'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
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
