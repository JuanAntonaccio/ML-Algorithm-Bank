import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

#El Banco portugués está teniendo una baja en sus ingresos. 
#El Banco quiere predecir qué clientes tienen más probabilidades 
# de suscribir un depósito a plazo y tratar de concentrar sus
#  fuerzas en poder captar ese tipo de clientes.

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv', delimiter=';')
df_raw.to_csv('../data/raw/dataset_bank.csv')

#Removing duplicates

duplicated_data=df_raw[df_raw.duplicated(keep="last")]
print("Duplicated Data",duplicated_data.shape)
df_raw=df_raw.drop_duplicates()
df_raw.shape

objetivo = df_raw.y
df_raw_2 = df_raw.drop('y', axis=1)

def verValueCounts(df,col):
    print("Value counts de {}".format(col))
    print(df[col].value_counts())
    print("="*60)



# Voy a recorrer la lista de las columnas categoricas que continen unknow y los vamos a listar
lista=['job','marital','education','default','housing','loan']
for i in lista:
    verValueCounts(df_raw_2,i)

# Funcion para convertir los valores unknow por el valor mas frecuente de esa variable

def repl_with_freq(df,col):
    freq = df[col].value_counts().idxmax()
    print("El valor maximo de frecuencia es:", freq)
    df[col].replace('unknown', freq , inplace = True)
    print("Se remplazo el valor unknown por el que tiene mas valores:", freq)

for i in lista:
    repl_with_freq(df_raw_2,i)
    print("="*65)

lst=['basic.9y','basic.6y','basic.4y']
for i in lst:
    df_raw_2.loc[df_raw_2['education'] == i, 'education'] = "middle.school"

# Cambio el valor de la variable obejtivo a 0 y 1

objetivo = objetivo.replace('no',0)
objetivo = objetivo.replace('yes',1)    

# Cambio segun el punto pedido la edad en rangos

rangos=[10,20,30,40,50,60,70,80,90,100]
categorias=[0,1,2,3,4,5,6,7,8]
# son 9 rangos
df_raw_2['age']=pd.cut(df_raw_2['age'],bins=rangos,labels=categorias)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df_raw_2['education'] = encoder.fit_transform(df_raw_2['education'])

month_dict={'may':5,'jul':7,'aug':8,'jun':6,'nov':11,'apr':4,'oct':10,'sep':9,'mar':3,'dec':12}
df_raw_2['month']= df_raw_2['month'].map(month_dict) 

day_dict={'thu':5,'mon':2,'wed':4,'tue':3,'fri':6}
df_raw_2['day_of_week']= df_raw_2['day_of_week'].map(day_dict)

# utilizo dummies para las columnas con muchos valores
df_raw_2 = pd.get_dummies(df_raw_2, columns = ['job', 'marital', 'default','housing', 'loan', 'contact', 'poutcome'])

# Borro las columnas que estan yes y no, dejo solo las yes

df_raw_2 = df_raw_2.drop(['default_no','housing_no','loan_no'],axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_raw_2, objetivo, test_size=0.3, random_state=94)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Fijarse los hiperparametros que pueda mejorar la regresion logistica
pipe_lr = make_pipeline(StandardScaler(), LogisticRegression())
pipe_lr.fit(X_train, y_train)  # apply scaling on training data
pipe_lr.score(X_test, y_test)  # apply scaling on testing data, without leaking training data.

from sklearn.metrics import classification_report
target_names=['clase:no','clase:yes']
y_true=y_test
y_pred=pipe_lr.predict(X_test)
print(classification_report(y_true, y_pred, target_names=target_names))

# Empezamos a probar con los hiperparametros, utilizamos class_weight 

pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight={0:0.3,1:0.7}))
pipe_lr.fit(X_train, y_train)  # apply scaling on training data
pipe_lr.score(X_test, y_test)  # apply scaling on testing data, without leaking training data.

target_names=['clase:no','clase:yes']
y_true=y_test
y_pred=pipe_lr.predict(X_test)
print(classification_report(y_true, y_pred, target_names=target_names))

# Utilizamos Gridsearch

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"],"class_weight":[{0:0.1,1:0.9},{0:0.2,1:0.8}] }# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

target_names=['clase:no','clase:yes']
y_true=y_test
y_pred=logreg_cv.predict(X_test)
print(classification_report(y_true, y_pred, target_names=target_names))

# Cambiamos los parametros

print("="*60)
print("Cambiamos los parametros del GridSearch  --------")
print("="*60)

grid={"C":[0.001,0.01,0.1,0.2], "penalty":["l1","l2","elasticnet"],"class_weight":[{0:0.4,1:0.6},{0:0.3,1:0.7}] }# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

target_names=['clase:no','clase:yes']
y_true=y_test
y_pred=logreg_cv.predict(X_test)
print(classification_report(y_true, y_pred, target_names=target_names))

# Grabamos el modelo conseguido

import joblib

#save your model or results
joblib.dump(logreg_cv, '../models/modelo_optimizado_bank.pkl')

print()
print("="*70)
print("Hemos Grabado el modelo entrenado con exito, gracias")
print("="*70)

