import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from os.path import join

def build_excel_file():
    # Directory containing tab-delimited files
    input_directory = 'stats_files'

    # Output Excel file name
    output_excel_file = join('results', 'output.xlsx')

    # Initialize a new Excel workbook
    workbook = Workbook()

    # Loop through files in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.txt') or filename.endswith('.tsv'):
            file_path = os.path.join(input_directory, filename)

            # Read the tab-delimited file into a pandas DataFrame
            df = pd.read_csv(file_path, sep='\t')

            # Create a new worksheet with the file name (without extension) as the sheet name
            sheet_name = os.path.splitext(filename)[0]
            sheet = workbook.create_sheet(title=sheet_name)

            # Write the DataFrame to the worksheet
            for row in dataframe_to_rows(df, index=False, header=True):
                sheet.append(row)

    # Remove the default sheet created by openpyxl
    workbook.remove(workbook.active)

    # Save the Excel workbook
    workbook.save(output_excel_file)

    print(f"Excel file '{output_excel_file}' created with data from tab-delimited files.")

if __name__ == '__main__':
    build_excel_file()