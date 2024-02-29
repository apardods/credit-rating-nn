'''
Project: Credit Ratings Prediction Using Neural Networks
Author: Antonio Pardo de Santayana Navarro
Description: Take .xlsx files generated with Factset Filings Wizard, and organize information into a pandas DataFrame
'''

import openpyxl as xl
import pandas as pd
import json

NUM_BATCHES = 2
# metrics = ['Revenue', 'Cost of Revenues', 'Interest Expense', 'Net Income', 'EPS', 'Current Assets',
#            'PPE', 'Total Assets', 'Short-term Debt', 'Current Liabilities', 'Long-term Debt',
#            'Total Liabilities', 'Equity', 'Depreciation', 'Cash from Operations', 'Cash in Financing',
#            'Cash in Investing', 'Cash']

# Some companies report in millions, some in thousands. We ignore share count for now and get the multiplier for each company
def assign_multiplier(text):
    if 'millions' in text or 'million' in text:
        return 1e6
    if 'thousand' in text or 'thousands' in text:
        return 1e3
    return 1

def find_metric(ws, fields, multiplier):
    for row in ws.iter_rows(values_only=True):
        if row[0] and any(substring in row[0].lower() for substring in fields):
            value = row[1]
            return value * multiplier if value else None
    return None

def process_file(file_path, patterns):
    print(f'Processing File {file_path}')
    wb = xl.load_workbook(file_path, data_only=True)
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        company={'Name': ws['A2'].value, 'Ticker': ws['A3'].value}
        multiplier = assign_multiplier(ws['A9'].value.lower())
        for name, fields in patterns.items():
            if name == 'EPS':
                company[name] = find_metric(ws, fields, 1)
            else:
                company[name] = find_metric(ws, fields, multiplier)
        print(company, multiplier)

def main():
    df = pd.DataFrame()
    with open('patterns.json', 'r') as f:
        patterns = json.load(f)
    for i in range(NUM_BATCHES):
        file_path = f'data/batch_{i}.xlsx'
        process_file(file_path, patterns)
        

if __name__ == '__main__':
    main()