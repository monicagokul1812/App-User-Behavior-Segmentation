import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="Customer Segmentation Dashboard")

# =========================
# LOAD FINAL CLUSTERED DATA
# =========================
df = pd.read_csv("final_clustered_data.csv")   # IMPORTANT: use clustered file

# =========================
# FEATURE ENGINEERING
# =========================

# Customer Lifetime Value
df["CLV"] = df["engagement_score"] * 50

# Revenue Potential
df["Revenue_Potential"] = df["engagement_score"] * df["daily_active_minutes"]

# User Status
df["User_Status"] = df["days_since_last_login"].apply(
    lambda x: "Active" if x <= 7 else "Inactive"
)

# =========================
# SMART CHURN STRATEGY (TOP 25%)
# =========================
threshold = df["churn_risk_score"].quantile(0.75)
df["Churn_Flag"] = (df["churn_risk_score"] >= threshold).astype(int)

# =========================
# DASHBOARD TITLE
# =========================
st.markdown(
    "<h1 style='text-align: center; color: #2E8B57;'>Customer Segmentation & Churn Intelligence Dashboard</h1>",
    unsafe_allow_html=True
)

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.header("Select Customer Segment")

clusters = st.sidebar.multiselect(
    "Choose Cluster",
    df['Cluster_Label'].unique(),
    default=df['Cluster_Label'].unique()
)

filtered_df = df[df['Cluster_Label'].isin(clusters)]

# =========================
# KPI SECTION
# =========================
st.subheader("📌 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

total_customers = len(filtered_df)
avg_clv = filtered_df["CLV"].mean()
avg_revenue = filtered_df["Revenue_Potential"].mean()
num_clusters = filtered_df['Cluster_Label'].nunique()
churn_rate = filtered_df["Churn_Flag"].mean() * 100

col1.metric("Total Customers", total_customers)
col2.metric("Avg CLV", f"{avg_clv:.2f}")
col3.metric("Revenue Potential", f"{avg_revenue:.2f}")
col4.metric("High Risk Users (%)", f"{churn_rate:.2f}%")
col5.metric("Segments", num_clusters)

st.divider()

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA",
    "🎯 Segments",
    "📈 Business Insights",
    "📄 Final Summary"
])

# =========================
# TAB 1 — EDA
# =========================
with tab1:

    numeric_cols = filtered_df.select_dtypes(include='number').columns

    feature = st.selectbox("Select Feature", numeric_cols)

    colA, colB = st.columns(2)

    with colA:
        fig = px.histogram(filtered_df, x=feature, color="Cluster_Label")
        st.plotly_chart(fig, use_container_width=False)

    with colB:
        fig2 = px.box(filtered_df, x="Cluster_Label", y=feature)
        st.plotly_chart(fig2, use_container_width=False)

    st.subheader("Correlation Heatmap")
    corr = filtered_df[numeric_cols].corr()
    fig3 = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig3, use_container_width=False)


# =========================
# TAB 2 — SEGMENT VISUALS
# =========================
with tab2:

    st.subheader("Cluster Scatter View")

    x_axis = st.selectbox("X Axis", numeric_cols, index=0)
    y_axis = st.selectbox("Y Axis", numeric_cols, index=1)

    fig4 = px.scatter(
        filtered_df,
        x=x_axis,
        y=y_axis,
        color="Cluster_Label",
        title="Cluster Distribution"
    )

    st.plotly_chart(fig4, use_container_width=False)

    st.subheader("Cluster Size Distribution")

    cluster_counts = filtered_df['Cluster_Label'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']

    fig5 = px.pie(cluster_counts, names='Cluster', values='Count')
    st.plotly_chart(fig5, use_container_width=False)


# =========================
# TAB 3 — BUSINESS INSIGHTS
# =========================
with tab3:

    st.subheader("High Risk Churn Rate by Cluster")

    cluster_churn = (
        filtered_df.groupby("Cluster_Label")["Churn_Flag"]
        .mean()
        .reset_index()
    )

    cluster_churn["Churn_Rate"] = cluster_churn["Churn_Flag"] * 100

    fig_churn = px.bar(
        cluster_churn,
        x="Cluster_Label",
        y="Churn_Rate",
        color="Cluster_Label",
        title="Top 25% High Risk Users by Segment"
    )

    st.plotly_chart(fig_churn, use_container_width=False)

    st.subheader("Cluster Performance Summary")

    summary_table = (
        filtered_df.groupby("Cluster_Label")
        .agg(
            Customers=("user_id", "count"),
            Avg_Engagement=("engagement_score", "mean"),
            Avg_Sessions=("sessions_per_week", "mean"),
            Avg_Churn_Risk=("churn_risk_score", "mean"),
            High_Risk_Rate=("Churn_Flag", "mean"),
            Avg_CLV=("CLV", "mean")
        )
        .reset_index()
    )

    summary_table["High_Risk_Rate"] = summary_table["High_Risk_Rate"] * 100

    st.dataframe(summary_table, use_container_width=False)


# =========================
# TAB 4 — FINAL SUMMARY
# =========================
with tab4:

    st.markdown("""
    ## 🎯 Project Outcome

    ✔ Successfully segmented 50,000 users using KMeans clustering  
    ✔ Identified 4 behavioral customer segments  
    ✔ Top 25% high-risk users identified using percentile-based churn strategy  
    ✔ Enabled customer-level targeting  

    ## 📈 Business Impact

    - Loyalty programs for high engagement users  
    - Retention campaigns for high-risk users  
    - Personalized engagement strategies  
    - Improved marketing ROI  

    ## 🚀 Technical Stack

    - Data Cleaning & Feature Engineering  
    - StandardScaler Normalization  
    - KMeans Clustering  
    - PCA Visualization  
    - Behavioral & Churn Profiling  
    """)