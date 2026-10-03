
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# 1. Load the dataset
df = pd.read_csv("sleep_health.csv")
df.columns = df.columns.str.strip()

# Support the common dataset column names
df = df.rename(columns={
    "Sleep Duration": "SleepDuration",
    "Quality of Sleep": "QualityOfSleep",
    "Physical Activity Level": "PhysicalActivity",
    "Stress Level": "StressLevel",
    "Heart Rate": "HeartRate",
    "Daily Steps": "DailySteps",
    "Sleep Disorder": "SleepDisorder",
    "Blood Pressure": "BloodPressure",
    "Person ID": "PersonID",
})

target = "SleepDisorder"

# 2. Clean missing target labels and select usable columns
if target not in df.columns:
    raise ValueError(
        "Sleep Disorder column not found. Check your CSV headers."
    )

df = df.dropna(subset=[target]).copy()
df[target] = df[target].astype(str).str.strip()
df = df[df[target] != ""]

# Treat no recorded disorder as a category if the dataset uses it.
# We do not invent labels for missing target values.
if df.empty or df[target].nunique() < 2:
    raise ValueError("At least two target categories are needed.")

# 3. Select lifestyle and sleep features available in the CSV
possible_features = [
    "Age", "Gender", "Occupation", "SleepDuration",
    "QualityOfSleep", "PhysicalActivity", "StressLevel",
    "BMI Category", "BloodPressure", "HeartRate", "DailySteps"
]
features = [c for c in possible_features if c in df.columns]

if not features:
    raise ValueError("No expected input columns found in the CSV.")

# Drop columns that have no usable values
features = [c for c in features if df[c].notna().any()]
X = df[features].copy()
y = df[target].copy()

# Convert numeric-looking columns to numbers
for col in X.columns:
    if col not in ["Gender", "Occupation", "BMI Category",
                   "BloodPressure"]:
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().sum() > 0:
            X[col] = converted

# Remove columns that became entirely missing
X = X.dropna(axis=1, how="all")
features = list(X.columns)

numeric = X.select_dtypes(include="number").columns.tolist()
categorical = [c for c in features if c not in numeric]

# 4. Prepare missing values and category columns
transformers = []

if numeric:
    transformers.append((
        "numbers",
        SimpleImputer(strategy="median"),
        numeric
    ))

if categorical:
    transformers.append((
        "categories",
        Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical
    ))

preprocessor = ColumnTransformer(transformers)

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100, random_state=42
    ))
])

# 5. Train and evaluate the model
# Stratify only when each class has enough records
counts = y.value_counts()
stratify = y if counts.min() >= 2 else None

if len(df) >= 10 and y.nunique() >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=stratify
    )
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
else:
    model.fit(X, y)
    accuracy = None

# Fit on all available labelled records for the demo form
model.fit(X, y)

# 6. Calculate analytics
numeric_df = df.select_dtypes(include="number")
average_sleep = (
    round(numeric_df["SleepDuration"].mean(), 2)
    if "SleepDuration" in numeric_df.columns else None
)
average_stress = (
    round(numeric_df["StressLevel"].mean(), 2)
    if "StressLevel" in numeric_df.columns else None
)
average_steps = (
    round(numeric_df["DailySteps"].mean(), 0)
    if "DailySteps" in numeric_df.columns else None
)

disorder_counts = df[target].value_counts().to_dict()
table_rows = df[features + [target]].head(10).fillna("-").to_dict("records")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        row = {}

        for col in features:
            value = request.form.get(col, "").strip()

            if col in numeric:
                row[col] = pd.to_numeric(value, errors="coerce")
            else:
                row[col] = value if value else None

        input_df = pd.DataFrame([row], columns=features)
        prediction = str(model.predict(input_df)[0])

    return render_template(
        "index.html",
        total=len(df),
        sleep=average_sleep,
        stress=average_stress,
        steps=average_steps,
        counts=disorder_counts,
        rows=table_rows,
        features=features,
        numeric=numeric,
        prediction=prediction,
        accuracy=round(accuracy * 100, 1) if accuracy is not None else None
    )

if __name__ == "__main__":
    app.run(debug=True)