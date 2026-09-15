from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RAW_FILE = RAW_DIR / "pima-indians-diabetes.csv"

COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Class"]

TARGET = "Class"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
OUTLIER_COLS = ["Insulin", "DiabetesPedigreeFunction", "BloodPressure", "Age"]


def load_data() -> pd.DataFrame:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_FILE}. "
            "It is committed to the repository (data/raw/pima-indians-diabetes.csv)."
        )
    df = pd.read_csv(RAW_FILE)
    if not set(COLUMNS).issubset(df.columns):        # file without a header
        df = pd.read_csv(RAW_FILE, names=COLUMNS)
    print(f"Loaded {len(df)} rows x {df.shape[1]} columns from {RAW_FILE.name}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    for col in ZERO_AS_MISSING:
        n_missing = int((df[col] == 0).sum())
        median = df.loc[df[col] != 0, col].median()
        df.loc[df[col] == 0, col] = median
        print(f"Imputed {n_missing} hidden zeros in '{col}' with median {median:.1f}")
    print(f"Missing values after imputation: {int(df.isna().sum().sum())}")

    for col in OUTLIER_COLS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        print(f"Removed {before - len(df)} outliers in '{col}' (bounds {lower:.2f}..{upper:.2f})")

    return df.reset_index(drop=True)


def split_and_save(df: pd.DataFrame) -> None:
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"Saved train ({len(train)} rows) and test ({len(test)} rows) to {PROCESSED_DIR}")


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    split_and_save(df)