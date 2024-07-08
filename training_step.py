import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from joblib import dump
from sklearn.metrics import make_scorer, f1_score
from globals import *
import pandas as pd
from preprocessing_step import *

def train_models(X_train, y_train):
    # Define the models and their parameter grids
    models_param_grid = {
        'SVM': (SVC(), {
            'C': [0.1, 1, 10, 100, 200],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'linear']
        }),
        'RandomForest': (RandomForestClassifier(), {
            'n_estimators': [10, 50, 100, 200, 300],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [2, 4, 6, 8, 10, 12]
        }),
        'KNN': (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }),
        'LogisticRegression': (LogisticRegression(max_iter=200), {
            'C': [0.1, 1, 10, 100, 200],
            'solver': ['newton-cg', 'lbfgs', 'liblinear']
        })
    }

    # Loop through the models and perform GridSearchCV
    for model_name, (model, param_grid) in models_param_grid.items():
        print(f"Training {model_name}...")
        
        scorer = make_scorer(f1_score, average='macro')

        grid_search = GridSearchCV(model, param_grid, cv=5, verbose=1, n_jobs=-1, scoring='f1_macro')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best parameters for {model_name}: {best_params}")

        # Save the best model
        model_save_path = f"models/{model_name}_model.joblib"
        rename_to_log(model_save_path)
        dump(best_model, model_save_path)
        models_param_grid[model_name] = (best_model, param_grid)

def get_new_data():
    #get the data
    df_additional = load_gsheet('1MSw27792kMZvgDBW0uGw6OH21tPRwhSwIIX4-9GDDXo')
    df_additional = df_additional.drop([0, 1])
    df_additional = df_additional.drop(columns=[0, 1, 15, 16, 17, 18, 19])

    #rename column 
    map_column = {2: 'Jenis Kelamin', 3: 'Usia (thn)', 4: 'Tinggi (cm)', 5: 'Berat (Kg)', 6: 'BMI (Kg/m2)', 7: 'Lingkar Perut (cm)', 8: 'Lingkar Leher (cm)', 9: 'Suara Mengorok', 10: 'Terbangun (berapa kali): buang air kecil', 11: 'Terbangun (berapa kali): tersedak', 12: 'Durasi tidur (jam)', 13: 'Ngantuk saat beraktifitas', 14: 'Kondisi yang menyertai: Lainnya', 20: 'Label (OSA)'}
    df_additional = df_additional.rename(columns=map_column)

    # One Hot encoding process
    # Split the comma-separated values into lists
    df_additional['Kondisi yang menyertai: Lainnya'] = df_additional['Kondisi yang menyertai: Lainnya'].apply(lambda x: x.split(', '))
    
    # Expand the lists into separate rows
    df_expanded = df_additional.explode('Kondisi yang menyertai: Lainnya')

    #one hot encoding
    one_hot = one_hot_prep(df_expanded, ['Kondisi yang menyertai: Lainnya']).reset_index()

    # Aggregate back to original format
    df_agg = one_hot.groupby('index').sum().reset_index(drop=True)

    #combine with other column
    df_final = pd.concat([df_additional.reset_index(drop=True), df_agg], axis=1)

    # adjust column name with the original
    df_final = df_final.drop(columns=['Kondisi yang menyertai: Lainnya', 'Kondisi yang menyertai: Lainnya_'])
    df_final = df_final.rename(columns={'Kondisi yang menyertai: Lainnya_Diabetes': 'Kondisi yang menyertai: Diabetes', 'Kondisi yang menyertai: Lainnya_Hipertensi': 'Kondisi yang menyertai: Hipertensi', 'Kondisi yang menyertai: Lainnya_Hyperkolesterol': 'Kondisi yang menyertai: Hyperkolesterol', 'Kondisi yang menyertai: Lainnya_GERD': 'Kondisi yang menyertai: Lainnya_Gerd'})

    #categorical data preprocessing
    df_final = categorical_prep(df_final)

    #remove unlabeled data
    df_final = df_final[df_final['Label (OSA)'] != '']

    return df_final


def retrain_models():
    #get the original train data
    df_original = pd.read_csv(train_data_file_path)
    df_original = categorical_prep(df_original)
    df_original = one_hot_prep(df_original, ['Kondisi yang menyertai: Lainnya'])

    #get the new data
    df_additional = get_new_data()
    
    #combine the original and new data
    combined_df = pd.concat([df_original, df_additional], ignore_index=True, sort=False)
    combined_df = combined_df.fillna(0)

    #preprocess all the data
    numerical_column = get_numerical_column()
    combined_df[numerical_column] = combined_df[numerical_column].apply(pd.to_numeric)

    generate_stats_table(combined_df, numerical_column)
    prep_df = numerical_prep(combined_df, numerical_column)

    #retrain the model
    X = prep_df.drop(columns=['Label (OSA)'])
    y = prep_df['Label (OSA)']
    train_models(X, y)