import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "synthetic_textile_water_dataset")
MODEL_V1_PATH = os.path.join(BASE_DIR, "classifier", "models", "v1")
MODEL_V2_PATH = os.path.join(BASE_DIR, "classifier", "models", "v2")
LEAKDB_RESULTS_PATH = os.path.join(BASE_DIR, "external_validation", "expected_flow_experiment")
REPORT_PATH = os.path.join(BASE_DIR, "classifier", "reports")
