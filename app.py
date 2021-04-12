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
        'yellowing of eyes','acute liver failure','fluid overload','swelling of stomach',
        'swelled lymph nodes','malaise','blurred and distorted vision','phlegm','throat irritation',
        'redness of eyes','sinus pressure','runny nose','congestion','chest pain','weakness in limbs',
        'fast heart rate','pain during bowel movements','pain in anal region','bloody stool',
        'irritation in anus','neck pain','dizziness','cramps','bruising','obesity','swollen legs',
        'swollen blood vessels','puffy face and eyes','enlarged thyroid','brittle nails',
        'swollen extremeties','excessive hunger','extra marital contacts','drying and tingling lips',
        'slurred speech','knee pain','hip joint pain','muscle weakness','stiff neck','swelling joints',
        'movement stiffness','spinning movements','loss of balance','unsteadiness',
        'weakness of one body side','loss of smell','bladder discomfort','foul smell of urine',
        'continuous feel of urine','passage of gases','internal itching','toxic look (typhos)',
        'depression','irritability','muscle pain','altered sensorium','red spots over body','belly pain',
        'abnormal menstruation','dischromic  patches','watering from eyes','increased appetite','polyuria','family history','mucoid sputum',
        'rusty sputum','lack of concentration','visual disturbances','receiving blood transfusion',
        'receiving unsterile injections','coma','stomach bleeding','distention of abdomen',
        'history of alcohol consumption','fluid overload','blood in sputum','prominent veins on calf',
        'palpitations','painful walking','pus filled pimples','blackheads','scurring','skin peeling'
        'silver like dusting','small dents in nails','inflammatory nails','blister','red sore around nose',
        'yellow crust ooze']

    disease=['Jangkitan kulat','Alergi','GERD','Kolestasis kronik','Tindak balas dadah','Penyakit ulser peptik','AIDS','Diabetes','Radang perut dan usus','Asma bronkial','Tekanan darah tinggi',' Migrain','Spondylosis serviks','Lumpuh (pendarahan otak)','Jaundis','Malaria','Cacar air','Denggi','Typhoid','hepatitis A','hepatitis B','hepatitis C','hepatitis D','hepatitis E','Hepatitis beralkohol','Tuberkulosis','Demam selesema biasa','Radang paru-paru','Buasir dimorphic','Serangan jantung','Urat varikos','Hipotiroidisme','Hipertiroidisme','Hipoglycemia','Osteoarthristis','Artritis','(vertigo) Posisi Paroymsal Vertigo','Jerawat','Jangkitan saluran kencing','Psoriasis','Impetigo']

    l2=[]
    for x in range(0,len(l1)):
        l2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
    df=pd.read_csv("Training.csv")

    df.replace({'prognosis':{'Jangkitan kulat':0,'Alergi':1,'GERD':2,'Kolestasis kronik':3,'Tindak balas dadah':4,'Penyakit ulser peptik':5,'AIDS':6,'Diabetes':7,'Radang perut dan usus':8,'Asma bronkial':9,'Tekanan darah tinggi':10,'Migrain':11,'Spondylosis serviks':12,'Lumpuh (pendarahan otak)':13,'Jaundis':14,'Malaria':15,'Cacar air':16,'Denggi':17,'Typhoid':18,'hepatitis A':19,'hepatitis B':20,'hepatitis C':21,'hepatitis D':22,'hepatitis E':23,'Hepatitis beralkohol':24,'Tuberkulosis':25,'Demam selesema biasa':26,'Radang paru-paru':27,'Buasir dimorphic':28,'Serangan jantung':29,'Urat varikos':30,'Hipotiroidisme':31,'Hipertiroidisme':32,'Hipoglycemia':33,'Osteoarthristis':34,'Artritis':35,'(vertigo) Posisi Paroymsal Vertigo':36,'Jerawat':37,'Jangkitan saluran kencing':38,'Psoriasis':39,'Impetigo':40}},inplace=True)

# print(df.head())

    X= df[l1]

    y = df[["prognosis"]]
    np.ravel(y)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
    tr=pd.read_csv("Testing.csv")
    tr.replace({'prognosis':{'Jangkitan kulat':0,'Alergi':1,'GERD':2,'Kolestasis kronik':3,'Tindak balas dadah':4,'Penyakit ulser peptik':5,'AIDS':6,'Diabetes':7,'Radang perut dan usus':8,'Asma bronkial':9,'Tekanan darah tinggi':10,'Migrain':11,'Spondylosis serviks':12,'Lumpuh (pendarahan otak)':13,'Jaundis':14,'Malaria':15,'Cacar air':16,'Denggi':17,'Typhoid':18,'hepatitis A':19,'hepatitis B':20,'hepatitis C':21,'hepatitis D':22,'hepatitis E':23,'Hepatitis beralkohol':24,'Tuberkulosis':25,'Demam selesema biasa':26,'Radang paru-paru':27,'Buasir dimorphic':28,'Serangan jantung':29,'Urat varikos':30,'Hipotiroidisme':31,'Hipertiroidisme':32,'Hipoglycemia':33,'Osteoarthristis':34,'Artritis':35,'(vertigo) Posisi Paroymsal Vertigo':36,'Jerawat':37,'Jangkitan saluran kencing':38,'Psoriasis':39,'Impetigo':40}},inplace=True)

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
