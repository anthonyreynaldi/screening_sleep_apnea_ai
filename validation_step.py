import os
from IPython.display import display
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, classification_report
from prediction_step import load_model
import seaborn as sns
import matplotlib.pyplot as plt

#old
def calculate_metrics(conf_matrix):
    
    #column - predicted, row - actual
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

#old
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

def create_heatmap_confussion_matrix(confussion_matrix, classes):
    plt.figure(figsize=(10, 7))
    sns.heatmap(confussion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Prediksi')
    plt.ylabel('Sebenarnya')
    plt.title('Confusion Matrix Heatmap')
    plt.show()

def calculate_metrics_all(y_true, y_pred, classes):

    # Convert labels to numerical format
    y_true_encoded = [classes.index(label) for label in y_true]
    y_pred_encoded = [classes.index(label) for label in y_pred]

    # Compute the confusion matrix with the labels parameter
    cm = confusion_matrix(y_true_encoded, y_pred_encoded, labels=range(len(classes)))

    create_heatmap_confussion_matrix(cm, classes)

    # Initialize dictionaries to hold the metrics for each class
    metrics = {cls: {} for cls in classes}

    # Compute metrics for each class
    # column - predicted, row - actual
    for i, class_label in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) != 0 else 0
        npv = tn / (tn + fn) if (tn + fn) != 0 else 0
        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) != 0 else 0
        
        metrics[class_label] = {
            'Akurasi': accuracy,
            'Spesifisitas': specificity,
            'Sensitifitas': sensitivity,
            'Nilai Prediktif Positif': ppv,
            'Nilai Prediktif Negatif': npv,
            'F1 Score': f1
        }

    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame(metrics).T

    return metrics_df
    
def calculate_model_metrics(model_name, X_test, y_test):
    
    model = load_model(model_name)
    
    y_pred = model.predict(X_test)
    
    classes = ['Normal', 'Ringan', 'Sedang', 'Berat']

    # result_per_class = calculate_metrics_per_class(y_test, y_pred, classes)
    result_per_class = calculate_metrics_all(y_test, y_pred, classes)

    result_all = result_per_class.mean()

    return result_per_class.round(2), result_all.round(2)

def validation_model(models_name, X_test, y_test, path_save=None):

    result_per_model = {}

    for model_name in models_name:
        result_per_class, result_all = calculate_model_metrics(model_name, X_test, y_test)

        result_per_model[model_name] = result_all

        if path_save:
            if not os.path.exists(path_save):
                os.makedirs(path_save)

            result_per_class.to_excel(f'{path_save}/{model_name}.xlsx')

        display(result_per_class)

    result = pd.DataFrame(result_per_model).transpose()

    if path_save:
        result.to_excel(f'{path_save}/overall.xlsx')

    return pd.DataFrame(result_per_model).transpose()