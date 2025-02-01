import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json

decision_attrs = [
    "price",
    "log_return_q1",
    "log_return_q2",
    "log_return_q3",
    "log_return_q4",
    "diff_q1",
    "diff_q2",
    "diff_q3",
    "diff_q4",
    "class_1_q1",
    "class_1_q2",
    "class_1_q3",
    "class_1_q4",
    "class_2_q1",
    "class_2_q2",
    "class_2_q3",
    "class_2_q4",
]

base_attrs = [
    'roe',
    'operating_margin',
    'free_cash_flow_per_share',
    'operating_cash_flow_per_share',
    'gross_margin',
    'roa',
    'asset_turnover',
    'net_profit_margin',
    'revenue',
    'operating_income',
    'cash_flow_from_operating_activities',
    'ebitda',
    # 'inventory_turnover',
    'debt_equity_ratio',
    'net_income',
    'current_ratio',
    'long_term_debt_capital',
    # 'receiveable_turnover',
    'retained_earnings_accumulated_deficit',
    'roi',
    'ebit_margin',
    'book_value_per_share',
    # 'days_sales_in_receivables',
    'pre_tax_profit_margin',
    'total_current_assets',
    'cash_on_hand',
    'long_term_debt',
    'total_liabilities',
    'eps_earnings_per_share_diluted'
]

MODELPATH = 'models/RandomForestClassifier-class_1_q1-0674-2025-01-31-17-44-24/model.pkl'
PARAMETERS_PATH = 'models/RandomForestClassifier-class_1_q1-0674-2025-01-31-17-44-24/parameters.json'


def quarter_int_to_date(n):
    year = 2009 + n // 4
    quarter = (n % 4) + 1
    return f"{year}Q{quarter}"


# Page config
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.title("Stock Analysis Dashboard")

# Load data and model


@st.cache_data
def load_data():
    XY = pd.read_csv(
        "sp500/data_xy.csv").set_index(keys=["quarter", "ticker"], drop=True)
    CREDIT_DATA = pd.read_csv(
        'credit_data/credit_data.csv').set_index(keys='quarter', drop=True)
    XY = XY.join(CREDIT_DATA, on='quarter', how='left')
    XY = pd.get_dummies(XY, columns=["sector"])
    return XY


@st.cache_resource
def load_model():
    with open(MODELPATH, 'rb') as file:
        return pickle.load(file)


@st.cache_resource
def load_parameters():
    with open(PARAMETERS_PATH) as file:
        return json.load(file)["parameters"]


df = load_data()
model = load_model()
attrs = list(set(df.columns) - set(decision_attrs))


def prepare_features(company_data):
    if company_data.empty:
        return None
    parameters = load_parameters()
    latest_data = company_data.iloc[-1]
    features = latest_data[parameters].values.reshape(1, -1)
    return features


def make_prediction(company_data):
    features = prepare_features(company_data)
    if features is None:
        return None, None

    try:
        pred_class = model.predict(features)[0]
        confidence = model.predict_proba(features)[0][int(pred_class)]
        prediction = 'BUY' if pred_class == 0.0 else 'SELL'
        return prediction, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None


# Sidebar
st.sidebar.header("Controls")

# Company selector
selected_company = st.sidebar.selectbox(
    "Select Company",
    options=sorted(df.index.get_level_values('ticker').unique())
)

# Filter data for selected company
company_data = df.loc[df.index.get_level_values('ticker') == selected_company]

# Make prediction
prediction, confidence = make_prediction(company_data)

# Display prediction
col1, col2 = st.columns(2)
with col1:
    st.subheader("ML Model Prediction")
    if prediction:
        if prediction == 'BUY':
            st.success(f"Prediction: {prediction} (Confidence: {confidence:.2%})")
        else:
            st.error(f"Prediction: {prediction} (Confidence: {confidence:.2%})")

# Feature importance

with col2:
    st.subheader("Feature Importance")
    feature_cols = load_parameters()
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.named_steps['clf'].feature_importances_
    }).sort_values('Importance', ascending=False)
    st.dataframe(importance_df)

# Create main plot
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add price line
fig.add_trace(
    go.Scatter(
        x=company_data.reset_index()['quarter'].apply(quarter_int_to_date),
        y=company_data['price'],
        name="Price",
        line=dict(color="#1f77b4", width=2)
    ),
    secondary_y=False
)

# Add financial ratios on secondary y-axis
for ratio in base_attrs:
    if ratio in company_data.columns:
        fig.add_trace(
            go.Scatter(
                x=company_data.reset_index()['quarter'],
                y=company_data[ratio],
                name=ratio.replace('_', ' ').title(),
                line=dict(dash='dash'),
                visible='legendonly'
            ),
            secondary_y=True
        )

# Update layout
prediction_text = f"{prediction} (Confidence: {confidence:.2%})" if prediction else "No prediction"
fig.update_layout(
    title=f"{selected_company} Stock Analysis",
    xaxis_title="Quarter",
    yaxis_title="Price",
    yaxis2_title="Ratio Values",
    hovermode='x unified',
    showlegend=True,
    height=600
)

# Display plot
st.plotly_chart(fig, use_container_width=True)

# Display metrics
st.subheader("Latest Financial Metrics")

N = 5
if not company_data.empty:
    latest_data = company_data.iloc[-1]
    for metrics_col, attr_name in zip(st.columns(N), base_attrs[:N]):
        with metrics_col:
            st.metric(attr_name, f"{latest_data[attr_name]:.2f}")

# Model details
with st.expander("Model Details"):
    if not company_data.empty:
        for attr_name in base_attrs[N:]:
            st.write(attr_name, f"{latest_data[attr_name]:.2f}")

# Optional: Display raw data
if st.checkbox("Show Raw Data"):
    st.dataframe(company_data)
