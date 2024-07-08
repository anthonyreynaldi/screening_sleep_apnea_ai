
from datetime import datetime
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# Get the current working directory
current_directory = os.getcwd()

# Construct the full path to the Excel file
stats_file_path = os.path.join(current_directory, 'data', 'stats_table.xlsx')
template_file_path = os.path.join(current_directory, 'data', 'stats_table.xlsx')
train_data_file_path = os.path.join(current_directory, 'data', 'train_data.csv')


def load_gsheet(sheet_key):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    json_keyfile_path = 'screening-sleep-apnea-463b102abcb8.json'

    # Authorize the credentials
    credentials = ServiceAccountCredentials.from_json_keyfile_name(json_keyfile_path, scope)
    gc = gspread.authorize(credentials)

    # Open the Google Sheet using the key
    docSheet = gc.open_by_key(sheet_key)

    # Get all values Convert to a DataFrame
    return pd.DataFrame.from_records(docSheet.sheet1.get_all_values(), )

def rename_to_log(original_file):
    # Get the current date and time and format it
    current_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

    # Split the original file path into directory, base name, and extension
    directory, original_file_name = os.path.split(original_file)
    file_name, file_extension = os.path.splitext(original_file_name)

    # Construct the new file name
    new_file_name = f"{current_time} {file_name}{file_extension}"
    new_file_path = os.path.join(directory, new_file_name)

    # Rename the file
    try:
        if os.path.exists(original_file):
            os.rename(original_file, new_file_path)
            print(f"File renamed from {original_file} to {new_file_path}")
        else:
            print(f"The file {original_file} does not exist.")
    except Exception as e:
        print(f"Error renaming file: {e}")

