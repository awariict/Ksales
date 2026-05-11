import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f6fa;
    }

    .metric-card {
        background: linear-gradient(135deg, #4e54c8, #8f94fb);
        padding: 20px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .small-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #4e54c8, #8f94fb);
    }

    h1, h2, h3 {
        color: #2f3640;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODELS
# ============================================================

rf_model = pickle.load(open("rf.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_excel("konga_sales_2015_2025.xlsx")

# ============================================================
# TITLE
# ============================================================

st.title("📊 SALES PERFORMANCE ANALYSIS DASHBOARD")
st.markdown("AI-Powered Sales Forecasting using Random Forest")

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("Dashboard Filters")

selected_year = st.sidebar.selectbox(
    "Select Year",
    sorted(df["Year"].unique())
)

selected_city = st.sidebar.multiselect(
    "Select City",
    df["City"].unique(),
    default=df["City"].unique()
)

selected_category = st.sidebar.multiselect(
    "Select Category",
    df["Category"].unique(),
    default=df["Category"].unique()
)

# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df[
    (df["Year"] == selected_year) &
    (df["City"].isin(selected_city)) &
    (df["Category"].isin(selected_category))
]

# ============================================================
# KPI METRICS
# ============================================================

revenue = filtered_df["Revenue_NGN"].sum()
orders = filtered_df["Order_ID"].count()
units = filtered_df["Units_Sold"].sum()
avg_price = filtered_df["Unit_Price_NGN"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class='metric-card'>
        <h3>Total Revenue</h3>
        <h1>₦{revenue:,.0f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class='small-card'>
        <h3>Total Orders</h3>
        <h1>{orders:,}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class='small-card'>
        <h3>Units Sold</h3>
        <h1>{units:,}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class='small-card'>
        <h3>Average Price</h3>
        <h1>₦{avg_price:,.0f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ============================================================
# MONTHLY REVENUE TREND
# ============================================================

monthly = filtered_df.groupby("Month")["Revenue_NGN"].sum().reset_index()

fig_month = px.line(
    monthly,
    x="Month",
    y="Revenue_NGN",
    markers=True,
    title="Monthly Revenue Trend"
)

fig_month.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# ============================================================
# CATEGORY SALES
# ============================================================

category_sales = filtered_df.groupby("Category")["Revenue_NGN"].sum().reset_index()

fig_category = px.bar(
    category_sales,
    x="Category",
    y="Revenue_NGN",
    title="Revenue by Category"
)

# ============================================================
# CITY SALES
# ============================================================

city_sales = filtered_df.groupby("City")["Revenue_NGN"].sum().reset_index()

fig_city = px.bar(
    city_sales,
    x="City",
    y="Revenue_NGN",
    title="Revenue by City"
)

# ============================================================
# PAYMENT METHODS
# ============================================================

payment_counts = filtered_df["Payment_Method"].value_counts().reset_index()
payment_counts.columns = ["Payment_Method", "Count"]

fig_payment = px.pie(
    payment_counts,
    names="Payment_Method",
    values="Count",
    title="Payment Method Distribution"
)

# ============================================================
# DISPLAY CHARTS
# ============================================================

c1, c2 = st.columns(2)

with c1:
    st.plotly_chart(fig_month, use_container_width=True)

with c2:
    st.plotly_chart(fig_category, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    st.plotly_chart(fig_city, use_container_width=True)

with c4:
    st.plotly_chart(fig_payment, use_container_width=True)

# ============================================================
# FORECASTING SECTION
# ============================================================

st.markdown("---")
st.subheader("📈 Future Revenue Forecast")

forecast_period = st.selectbox(
    "Select Forecast Period",
    [7, 30, 90]
)

# ============================================================
# PREPARE MODEL DATA
# ============================================================

model_df = df.copy()

categorical_cols = ["City", "Category", "Payment_Method"]

for col in categorical_cols:
    model_df[col] = encoders[col].transform(model_df[col])

# ============================================================
# FORECAST FUNCTION
# ============================================================

def future_forecast(periods):

    latest_data = model_df.iloc[-1:].copy()

    forecasts = []

    for i in range(periods):

        latest_data["Month"] += 1

        if latest_data["Month"].values[0] > 12:
            latest_data["Month"] = 1
            latest_data["Year"] += 1

        input_data = latest_data.drop(
            ["Revenue_NGN", "Order_ID"],
            axis=1
        )

        scaled_data = scaler.transform(input_data)

        prediction = rf_model.predict(scaled_data)[0]

        forecasts.append(prediction)

        latest_data["Revenue_NGN"] = prediction

    return forecasts

# ============================================================
# GENERATE FORECAST
# ============================================================

forecast_values = future_forecast(forecast_period)

forecast_df = pd.DataFrame({
    "Period": range(1, forecast_period + 1),
    "Forecast_Revenue": forecast_values
})

fig_forecast = px.line(
    forecast_df,
    x="Period",
    y="Forecast_Revenue",
    markers=True,
    title=f"{forecast_period}-Period Revenue Forecast"
)

st.plotly_chart(fig_forecast, use_container_width=True)

# ============================================================
# FORECAST SUMMARY
# ============================================================

forecast_total = np.sum(forecast_values)
forecast_avg = np.mean(forecast_values)
forecast_max = np.max(forecast_values)
forecast_min = np.min(forecast_values)

f1, f2, f3, f4 = st.columns(4)

f1.metric("Forecast Total", f"₦{forecast_total:,.0f}")
f2.metric("Forecast Average", f"₦{forecast_avg:,.0f}")
f3.metric("Highest Forecast", f"₦{forecast_max:,.0f}")
f4.metric("Lowest Forecast", f"₦{forecast_min:,.0f}")

# ============================================================
# RAW DATA TABLE
# ============================================================

st.markdown("---")
st.subheader("📁 Dataset Preview")

st.dataframe(filtered_df.head(50), use_container_width=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("Developed with Streamlit + Random Forest Machine Learning Model")