# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.

## Importacion de librerias
import pandas as pd
import os
import json
import gzip
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".

def clean_data(path):
    df=pd.read_csv(path,index_col=False,compression='zip')
    df.rename(columns={'default payment next month':'default'}, inplace=True)
    df.drop(columns='ID', inplace=True)
    df=df.loc[df[(df['EDUCATION']!=0) & (df['MARRIAGE']!=0)].index] #.index para obtener los indices de los registros que cumplen la condicion
    df['EDUCATION']=df['EDUCATION'].apply(lambda x: 4 if x>4 else x)
    return df

train_path = 'files/input/train_data.csv.zip'
test_path = 'files/input/test_data.csv.zip'

train = clean_data(train_path)
test = clean_data(test_path)

#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.

x_train,y_train=train.drop(columns='default'),train['default']
x_test,y_test=test.drop(columns='default'),test['default']

#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
categorical_features =['EDUCATION','MARRIAGE','SEX']
numerical_features=[col for col in x_train.columns if col not in categorical_features]

steps=[('preprocessor',ColumnTransformer(transformers=[('cat',OneHotEncoder(),categorical_features),
                                                       ('num',StandardScaler(),numerical_features)],
                                                        remainder='passthrough')),
       ('pca',PCA()), # PCA con todas las componentes
       ('selector',SelectKBest(f_classif)), # Selección de las K mejores características
       ('classifier', SVC())    
       ]

pipeline=Pipeline(steps)

#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.

param_grid = {
    'pca__n_components': [20,x_train.shape[1]-2],
    'selector__k': [12],
    'classifier__gamma': [0.1],
    'classifier__kernel': ['rbf']
}

model = GridSearchCV(pipeline, param_grid, cv=10, scoring='balanced_accuracy',n_jobs=-1)
model.fit(x_train, y_train)

#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
model_filename='files/models'
os.makedirs(model_filename,exist_ok=True)

model_path=os.path.join(model_filename,'model.pkl.gz')
with gzip.open(model_path,'wb') as file:
    pickle.dump(model,file)

#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}

def metrics_calculate(model,x_train,x_test,y_train,y_test):
    
    y_train_pred=model.predict(x_train)
    y_test_pred=model.predict(x_test)
    
    metrics=[{'type':'metrics',
           'dataset':'train',
           'precision':precision_score(y_train,y_train_pred),
           'balanced_accuracy':balanced_accuracy_score(y_train,y_train_pred),
           'recall':recall_score(y_train,y_train_pred),
           'f1_score':f1_score(y_train,y_train_pred)},
             
            {'type':'metrics',
             'dataset':'test',
             'precision':precision_score(y_test,y_test_pred),
             'balanced_accuracy':balanced_accuracy_score(y_test,y_test_pred),
             'recall':recall_score(y_test,y_test_pred),
             'f1_score':f1_score(y_test,y_test_pred)}
            ]
            
    os.makedirs('files/output',exist_ok=True)
    with open('files/output/metrics.json','w') as file:
        for metric in metrics:
            file.write(json.dumps(metric)+'\n')

#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
def calculate_confusion_matrices(model, x_train, x_test, y_train, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    matrices = [
        {
            'type': 'cm_matrix',
            'dataset': 'train',
            'true_0': {'predicted_0': int(cm_train[0, 0]), 'predicted_1': int(cm_train[0, 1])},
            'true_1': {'predicted_0': int(cm_train[1, 0]), 'predicted_1': int(cm_train[1, 1])}
        },
        {
            'type': 'cm_matrix',
            'dataset': 'test',
            'true_0': {'predicted_0': int(cm_test[0, 0]), 'predicted_1': int(cm_test[0, 1])},
            'true_1': {'predicted_0': int(cm_test[1, 0]), 'predicted_1': int(cm_test[1, 1])}
        }
    ]

    with open("files/output/metrics.json", "a") as f:
        for matrix in matrices:
            f.write(json.dumps(matrix) + '\n')

#correr modelo y calcular metricas
metrics_calculate(model,x_train,x_test,y_train,y_test)
calculate_confusion_matrices(model,x_train,x_test,y_train,y_test)       