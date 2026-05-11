import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import os
import gdown

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
    .main { background-color: #f5f6fa; }
    .metric-card { background: linear-gradient(135deg, #4e54c8, #8f94fb); padding: 20px; border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);}
    .small-card { background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);}
    .sidebar .sidebar-content { background: linear-gradient(180deg, #4e54c8, #8f94fb);}
    h1, h2, h3 { color: #2f3640; }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# DOWNLOAD RF MODEL FROM GOOGLE DRIVE
# ============================================================

rf_url = "https://drive.google.com/uc?id=11s_IB6wKtNSCOwO3dXOD-meav_btwbIj"
rf_file = "rf.pkl"

if not os.path.exists(rf_file):
    st.info("Downloading Random Forest model (this may take a few minutes)...")
    gdown.download(rf_url, rf_file, quiet=False)

with open(rf_file, "rb") as f:
    rf_model = pickle.load(f)

# Load scaler and encoders (assumes small enough to include in repo)
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_excel("konga_sales_2015_2025.xlsx")

# ============================================================
# TITLE
# ============================================================

st.title("📊 CHIAMAKA LINDA OKOLI SALES PERFORMANCE ANALYSIS DASHBOARD")
st.markdown("Dublin Business school MSc.Business Analysis")
st.markdown("AI-Powered Sales Forecasting using Random Forest")

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.title("Name: Chiamaka Linda Okoli")
st.sidebar.title("RegNo: 20055892 ")
st.sidebar.title("Dublin Business school")
st.sidebar.title("MSc.Business Analysis")
st.sidebar.title("Dashboard Filters")
selected_year = st.sidebar.selectbox("Select Year", sorted(df["Year"].unique()))
selected_city = st.sidebar.multiselect("Select City", df["City"].unique(), default=df["City"].unique())
selected_category = st.sidebar.multiselect("Select Category", df["Category"].unique(), default=df["Category"].unique())

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
col1.markdown(f"<div class='metric-card'><h3>Total Revenue</h3><h1>₦{revenue:,.0f}</h1></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='small-card'><h3>Total Orders</h3><h1>{orders:,}</h1></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='small-card'><h3>Units Sold</h3><h1>{units:,}</h1></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='small-card'><h3>Average Price</h3><h1>₦{avg_price:,.0f}</h1></div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# MONTHLY REVENUE TABLE + CHART
# ============================================================

monthly = filtered_df.groupby("Month")["Revenue_NGN"].sum().reset_index()
st.subheader("📅 Monthly Revenue")
st.dataframe(monthly.style.format({"Revenue_NGN": "₦{0:,.0f}"}), use_container_width=True)

fig_month = px.line(monthly, x="Month", y="Revenue_NGN", markers=True, title="Monthly Revenue Trend")
fig_month.update_layout(plot_bgcolor='white', paper_bgcolor='white', yaxis_tickformat=",.0f")
st.plotly_chart(fig_month, use_container_width=True)

# ============================================================
# CATEGORY SALES TABLE + CHART
# ============================================================

category_sales = filtered_df.groupby("Category")["Revenue_NGN"].sum().reset_index()
st.subheader("📂 Revenue by Category")
st.dataframe(category_sales.style.format({"Revenue_NGN": "₦{0:,.0f}"}), use_container_width=True)

fig_category = px.bar(category_sales, x="Category", y="Revenue_NGN", title="Revenue by Category")
fig_category.update_layout(plot_bgcolor='white', paper_bgcolor='white', yaxis_tickformat=",.0f")
st.plotly_chart(fig_category, use_container_width=True)

# ============================================================
# CITY SALES TABLE + CHART
# ============================================================

city_sales = filtered_df.groupby("City")["Revenue_NGN"].sum().reset_index()
st.subheader("🏙 Revenue by City")
st.dataframe(city_sales.style.format({"Revenue_NGN": "₦{0:,.0f}"}), use_container_width=True)

fig_city = px.bar(city_sales, x="City", y="Revenue_NGN", title="Revenue by City")
fig_city.update_layout(plot_bgcolor='white', paper_bgcolor='white', yaxis_tickformat=",.0f")
st.plotly_chart(fig_city, use_container_width=True)

# ============================================================
# PAYMENT METHODS TABLE + PIE CHART
# ============================================================

payment_counts = filtered_df["Payment_Method"].value_counts().reset_index()
payment_counts.columns = ["Payment_Method", "Count"]
st.subheader("💳 Payment Method Distribution")
st.dataframe(payment_counts, use_container_width=True)

fig_payment = px.pie(payment_counts, names="Payment_Method", values="Count", title="Payment Method Distribution")
st.plotly_chart(fig_payment, use_container_width=True)

# ============================================================
# FORECASTING TABLE + CHART
# ============================================================

st.markdown("---")
st.subheader("📈 Future Revenue Forecast")
forecast_period = st.selectbox("Select Forecast Period", [7, 30, 90])

model_df = df.copy()
for col in ["City", "Category", "Payment_Method"]:
    model_df[col] = encoders[col].transform(model_df[col])

def future_forecast(periods):
    latest_data = model_df.iloc[-1:].copy()
    forecasts = []
    for _ in range(periods):
        month = latest_data["Month"].values[0] + 1
        year = latest_data["Year"].values[0]
        if month > 12:
            month = 1
            year += 1
        latest_data["Month"] = month
        latest_data["Year"] = year

        input_data = latest_data.drop(["Revenue_NGN", "Order_ID"], axis=1).apply(pd.to_numeric, errors='coerce')
        scaled_data = scaler.transform(input_data)
        prediction = rf_model.predict(scaled_data)[0]
        forecasts.append(max(0, prediction))
        latest_data["Revenue_NGN"] = prediction
    return forecasts

forecast_values = future_forecast(forecast_period)
forecast_df = pd.DataFrame({
    "Period": range(1, forecast_period+1),
    "Forecast_Revenue": forecast_values
})
st.dataframe(forecast_df.style.format({"Forecast_Revenue": "₦{0:,.0f}"}), use_container_width=True)

fig_forecast = px.line(forecast_df, x="Period", y="Forecast_Revenue", markers=True, title=f"{forecast_period}-Period Revenue Forecast")
fig_forecast.update_layout(plot_bgcolor='white', paper_bgcolor='white', yaxis_tickformat=",.0f")
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

st.markdown("---")
st.caption("Developed with Streamlit + Random Forest Machine Learning Model")
