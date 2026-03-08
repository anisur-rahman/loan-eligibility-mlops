import pandas as pd

from src.preprocessing import LoanDataCleaner


def test_cleaner_creates_engineered_features():
    df = pd.DataFrame(
        {
            'Loan ID': [1],
            'Customer ID': [10],
            'Loan Status': ['Loan Given'],
            'Current Loan Amount': [99999999],
            'Term': ['Short Term'],
            'Credit Score': [7240],
            'Years in current job': ['10+ years'],
            'Home Ownership': ['HaveMortgage'],
            'Annual Income': [120000],
            'Purpose': ['Other'],
            'Monthly Debt': ['$1,000.00'],
            'Years of Credit History': [12.0],
            'Months since last delinquent': [None],
            'Number of Open Accounts': [6],
            'Number of Credit Problems': [0],
            'Current Credit Balance': [5000],
            'Maximum Open Credit': ['10000'],
            'Bankruptcies': [0],
            'Tax Liens': [0],
        }
    )

    out = LoanDataCleaner().fit_transform(df)
    assert 'Debt_to_Income_Ratio' in out.columns
    assert 'Credit_Utilization_Ratio' in out.columns
    assert out.loc[0, 'Credit Score'] == 724
    assert out.loc[0, 'Home Ownership'] == 'Home Mortgage'
    assert out.loc[0, 'Purpose'] == 'other'
