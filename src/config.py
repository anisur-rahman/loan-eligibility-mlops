from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PREDICTIONS_DIR = DATA_DIR / 'predictions'
MODELS_DIR = PROJECT_ROOT / 'models'

TRAIN_FILE = RAW_DATA_DIR / 'LoansTrainingSetV2.csv'
TEST_FILE = RAW_DATA_DIR / 'test_data.csv'
MODEL_FILE = MODELS_DIR / 'loan_eligibility_pipeline.joblib'
METRICS_FILE = MODELS_DIR / 'metrics.json'

TARGET_COLUMN = 'Loan Status'
DROP_COLUMNS = ['Loan ID']
ID_COLUMNS = ['Loan ID', 'Customer ID']

RANDOM_STATE = 42
TEST_SIZE = 0.2
