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

    l1=['gatal badan', 'ruam', 'bersin berterusan','menggigil','sejuk','sakit sendi','sakit perut']

    disease=['Jangkitan kulat','Alergi','GERD','Kolestasis kronik','Tindak balas dadah',
        'Penyakit ulser peptik','AIDS','Diabetes','Radang perut dan usus','Asma bronkial','Tekanan darah tinggi',
        ' Migrain','Spondylosis serviks']

    l2=[]
    for x in range(0,len(l1)):
        l2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
    df=pd.read_csv("Training.csv")

    df.replace({'prognosis':{'Jangkitan kulat':0,'Alergi':1,'GERD':2,'Kolestasis kronik':3,'Tindak balas dadah':4,
        'Penyakit ulser peptik':5,'AIDS':6,'Diabetes':7,'Radang perut dan usus':8,'Asma bronkial':9,'Tekanan darah tinggi':10,
        'Migrain':11,'Spondylosis serviks':12}},inplace=True)

# print(df.head())

    X= df[l1]

    y = df[["prognosis"]]
    np.ravel(y)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
    tr=pd.read_csv("Testing.csv")
    tr.replace({'prognosis':{'Jangkitan kulat':0,'Alergi':1,'GERD':2,'Kolestasis kronik':3,'Tindak balas dadah':4,
        'Penyakit ulser peptik':5,'AIDS':6,'Diabetes ':7,'Radang perut dan usus':8,'Asma bronkial':9,'Tekanan darah tinggi':10,
        'Migrain':11,'Spondylosis serviks':12}},inplace=True)

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
