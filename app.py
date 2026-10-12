import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Aircraft Predictive Maintenance", layout="wide")
st.title("Aircraft Engine Predictive Maintenance")
st.caption("Portfolio prototype using sensor-like data. This is not an aircraft-certified maintenance system.")

DATA_PATH = Path("data/engine_sensor_data.csv")
FEATURES = ["temperature", "pressure", "vibration", "oil_quality", "operating_hours"]
TARGET = "remaining_useful_life"

def make_demo_data(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    hours = rng.uniform(50, 5000, n)
    temp = 65 + 0.006 * hours + rng.normal(0, 4, n)
    pressure = 120 - 0.004 * hours + rng.normal(0, 3, n)
    vibration = 0.3 + 0.00012 * hours + rng.normal(0, 0.08, n)
    oil = 100 - 0.012 * hours + rng.normal(0, 5, n)
    rul = np.maximum(0, 5000 - hours + rng.normal(0, 180, n))
    return pd.DataFrame({"temperature": temp, "pressure": pressure, "vibration": vibration,
                         "oil_quality": oil, "operating_hours": hours, TARGET: rul})

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        if set(FEATURES + [TARGET]).issubset(df.columns):
            return df, "CSV dataset"
    return make_demo_data(), "generated demonstration data"

df, data_source = load_data()
st.info(f"Data source: {data_source}. For a real experiment, replace the demo data with a properly documented dataset.")
st.subheader("1. Data quality")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
c3.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
df = df.drop_duplicates().copy()
for col in FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df[FEATURES + [TARGET]] = df[FEATURES + [TARGET]].fillna(df[FEATURES + [TARGET]].median())
st.dataframe(df.head(10), use_container_width=True)

st.subheader("2. Exploratory data analysis")
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots()
    ax.scatter(df["operating_hours"], df[TARGET], alpha=0.35)
    ax.set_xlabel("Operating hours"); ax.set_ylabel("Remaining useful life")
    ax.set_title("Operating hours vs. estimated RUL")
    st.pyplot(fig)
with right:
    corr = df[FEATURES + [TARGET]].corr(numeric_only=True)
    fig, ax = plt.subplots()
    im = ax.imshow(corr, aspect="auto")
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index)
    ax.set_title("Correlation matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)

st.subheader("3. Train and evaluate a baseline model")
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=150, random_state=42, min_samples_leaf=2)
model.fit(X_train, y_train)
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:.1f} hours")
m2.metric("RMSE", f"{rmse:.1f} hours")
m3.metric("R²", f"{r2:.3f}")
fig, ax = plt.subplots()
ax.scatter(y_test, pred, alpha=0.4)
lo, hi = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
ax.plot([lo, hi], [lo, hi])
ax.set_xlabel("Actual RUL"); ax.set_ylabel("Predicted RUL"); ax.set_title("Actual vs predicted")
st.pyplot(fig)
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
st.bar_chart(imp)
st.warning("Demo-data metrics only describe the generated data. Do not report them as real-world aircraft performance.")

st.subheader("4. Estimate RUL for a sensor reading")
col1, col2, col3 = st.columns(3)
with col1:
    temperature = st.number_input("Temperature", value=85.0)
    pressure = st.number_input("Pressure", value=100.0)
with col2:
    vibration = st.number_input("Vibration", value=0.65, min_value=0.0)
    oil_quality = st.number_input("Oil quality index", value=70.0)
with col3:
    operating_hours = st.number_input("Operating hours", value=2500.0, min_value=0.0)
if st.button("Estimate remaining useful life"):
    row = pd.DataFrame([[temperature, pressure, vibration, oil_quality, operating_hours]], columns=FEATURES)
    st.metric("Estimated RUL", f"{max(0, model.predict(row)[0]):.0f} hours")
    st.caption("Illustrative model output only. It must not be used to make actual aviation maintenance decisions.")
