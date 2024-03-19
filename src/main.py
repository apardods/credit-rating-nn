import numpy as np

from data_load import load_data
from binary_baseline import make_binary_baseline
from binary_network import make_binary_classification

np.random.seed(123)

def classify_rating(rating):
    ordered_ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-']
    return 1 if ordered_ratings.index(rating) <= ordered_ratings.index('BBB+') else 0

def main():
    ratio_cols = ['Current Ratio', 'Cash Ratio', 'Debt to Equity',
            'Debt to Assets', 'TCI', 'Asset Turnover', 'CL Turnover', 'RoA',
            'CS PPE', 'CS Cash', 'CS CA', 'CS CL', 'Operating Margin',
            'Profit Margin', 'CS Interest', 'Gross Margin']
    target = 'Rating'
    df = load_data(ratio_cols + [target])
    df['Group'] = df['Rating'].apply(classify_rating)
    # binary_baseline_results = make_binary_baseline(df, ratio_cols, 'Group')
    # print(binary_baseline_results)
    network_results = make_binary_classification(df, ratio_cols, 'Group')
    print(network_results)

if __name__ == '__main__':
    main()