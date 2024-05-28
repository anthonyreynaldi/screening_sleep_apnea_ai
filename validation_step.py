from IPython.display import display
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
from prediction_step import load_model

def calculate_metrics(conf_matrix):
    
    tp = conf_matrix[1, 1]
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)            #or recall
    ppv = tp / (tp + fp)                    #or precision
    npv = tn / (tn + fn)
    f1 = 2 * ((ppv * sensitivity) / (ppv + sensitivity))
    
    return accuracy, specificity, sensitivity, ppv, npv, f1

def calculate_metrics_per_class(y_true, y_pred, classes):
    multi_conf_matrix = multilabel_confusion_matrix(y_true, y_pred)

    df_result = {}

    for class_name, conf_matrix in zip(classes, multi_conf_matrix):
        accuracy, specificity, sensitivity, ppv, npv, f1 = calculate_metrics(conf_matrix)

        df_result[class_name] = {
            'Akurasi': accuracy,
            'Spesifisitas': specificity,
            'Sensitifitas': sensitivity,
            'Nilai Prediktif Positif': ppv,
            'Nilai Prediktif Negatif': npv,
            'F-1 Score': f1
        }

    return pd.DataFrame(df_result).transpose()
    
def calculate_model_metrics(model_name, X_test, y_test):
    
    model = load_model(model_name)
    
    y_pred = model.predict(X_test)
    
    classes = model.classes_

    result_per_class = calculate_metrics_per_class(y_test, y_pred, classes)

    result_all = result_per_class.mean()

    return result_per_class.round(2), result_all.round(2)

def validation_model(models_name, X_test, y_test):

    result_per_model = {}

    for model_name in models_name:
        result_per_class, result_all = calculate_model_metrics(model_name, X_test, y_test)

        result_per_model[model_name] = result_all
        display(result_per_class)

    return pd.DataFrame(result_per_model).transpose()