import os
import pandas as pd
from globals import *

def get_numerical_column():
    return ['Usia (thn)', 'Tinggi (cm)', 'Berat (Kg)', 'BMI (Kg/m2)', 'Lingkar Perut (cm)', 'Lingkar Leher (cm)', 'Terbangun (berapa kali): buang air kecil', 'Terbangun (berapa kali): tersedak', 'Durasi tidur (jam)']

def make_template_df(df):
    temp = df.copy()
    temp = temp.iloc[0:0]
    default_values = {col: 0 if pd.api.types.is_numeric_dtype(df[col]) else "" for col in df.columns}
    temp = pd.DataFrame([default_values])
    temp.to_excel(template_file_path)

def get_df_template():
    df = pd.read_excel(stats_file_path)
    df = df.drop(columns=['Unnamed: 0', 'Label (OSA )'])

    return df

def create_empty_stats_table(numerical_column):
  #result dataframe of stastical analysis
  stats = ['MIN', 'MAX', 'AVG', 'STD']

  # Create an empty DataFrame
  table_stats = pd.DataFrame(index=numerical_column, columns=stats, dtype=int)

  # Fill with default value 0
  table_stats.fillna(0, inplace=True)

  return table_stats


def generate_stats_table(df, numerical_column):
    table_stats = create_empty_stats_table(numerical_column)

    #iterate from every column
    for column in df:

        # print(column)
        # print(type(df[column][0]))

        sample_data = df[column][0]

        #check data type whether is it categorical or numerical
        if type(sample_data) is not str:
            # df.groupby('Label (OSA)')
            table_stats['MIN'][column] = df[column].min()
            table_stats['MAX'][column] = df[column].max()
            table_stats['AVG'][column] = df[column].mean()
            table_stats['STD'][column] = df[column].std()

    rename_to_log(stats_file_path)

    print(stats_file_path)
    table_stats.to_excel(stats_file_path)

    return table_stats


def minmax_norm(data, min, max):
    return (data - min) / (max - min)

def z_norm(data, mean, std):
    return (data - mean) / std

def numerical_prep(df_origin, numerical_column, norm_type='minmax'):

    path_stats_table = stats_file_path
    print(path_stats_table)
    if os.path.exists(path_stats_table):
        table_stats = pd.read_excel(path_stats_table)
        table_stats = table_stats.set_index("Unnamed: 0")
        table_stats.index.name = None
    else:
        table_stats = generate_stats_table(df=df_origin, numerical_column=numerical_column)
        
    df_result = df_origin.copy()

    for column in numerical_column:
        if norm_type == 'minmax':
            df_result[column] = df_origin[column].apply(minmax_norm, args=[table_stats['MIN'][column], table_stats['MAX'][column]])
        elif norm_type == 'z':
            df_result[column] = df_origin[column].apply(z_norm, args=[table_stats['AVG'][column], table_stats['STD'][column]])

    return df_result

def categorical_prep(df_origin: pd.DataFrame):
    df_result = df_origin.copy()

    # replace loudness of snoring
    category_snoring = {
        'Normal (tidak mengorok)': 0,
        'Pelan': 1,
        'Sedang': 2,
        'Keras': 3
    }

    df_result['Suara Mengorok'] = df_origin['Suara Mengorok'].replace(category_snoring)

    #replace sleepy during activities
    category_sleepy = {
        'Tidak Menggangu': 0,
        'Tidak Mengganggu': 0,
        'Sedikit Menggangu': 1,
        'Sedikit Mengganggu': 1,
        'Sangat Menganggu': 2,
        'Sangat Mengganggu': 2
    }

    df_result['Ngantuk saat beraktifitas'] = df_origin['Ngantuk saat beraktifitas'].replace(category_sleepy)

    #replace OSA label
    # category_OSA = {
    #     'Normal': 0,
    #     'Ringan': 1,
    #     'Sedang': 2,
    #     'Berat': 3
    # }

    # df_result['Label (OSA)'] = df_origin['Label (OSA)'].replace(category_OSA)

    #replace gender
    category_gender = {
        'Perempuan': 0,
        'Laki-laki': 1
    }

    df_result['Jenis Kelamin'] = df_origin['Jenis Kelamin'].replace(category_gender)

    #others
    category_other = {
        'TRUE': 1,
        'True': 1,
        'FALSE': 0,
        'False': 0,
        True: 1,
        False: 0
    }
    df_result = df_result.replace(category_other)

    #current codition
    current_condition = {
        '': None,

        'hidung tersumbat': 'Pilek',
        'sering pilek': 'Pilek',
        'sinusitis': 'Pilek',
        'pilek': 'Pilek',
        'pilek ': 'Pilek',
        'hidung buntu': 'Pilek',
        'Pilek': 'Pilek',

        'amandel besar': 'Amandel',
        'amandel': 'Amandel',
        'amandel gede': 'Amandel',

        'GERD': 'GERD',
        'gerd': 'GERD',
        'reflux': 'GERD',

        'alergi hidung': 'Rhintis Alergi',
        'rhinitis alergi': 'Rhintis Alergi',
        'alergi': 'Rhintis Alergi',

        'polip': 'Polip',
        'asthma, polip': 'Polip',

        'adenoid besar': 'Adenoid',
        'adenoid': 'Adenoid',
    }
    
    if 'Kondisi yang menyertai: Lainnya' in df_origin.columns:
        df_result['Kondisi yang menyertai: Lainnya'] = df_result['Kondisi yang menyertai: Lainnya'].replace(current_condition)
        df_result['Kondisi yang menyertai: Lainnya'] = df_result['Kondisi yang menyertai: Lainnya'].str.title()

    return df_result

def one_hot_prep(df_origin, columns):
    return pd.get_dummies(df_origin, columns = columns)