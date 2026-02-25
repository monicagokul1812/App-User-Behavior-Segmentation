import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="Customer Segmentation Dashboard")

# =========================
# Load Data
# =========================
clustered = pd.read_csv("clustered_data.csv")
business = pd.read_csv("final_business_table.csv")

st.title("📊 Customer Segmentation & Business Insights Dashboard")

# =========================
# Create Business Metrics
# =========================

df = clustered.copy()

# Customer Lifetime Value
if "engagement_score" in df.columns:
    df["CLV"] = df["engagement_score"] * 50
else:
    df["CLV"] = 0

# Revenue Potential
if "daily_active_minutes" in df.columns:
    df["Revenue_Potential"] = df["engagement_score"] * df["daily_active_minutes"]
else:
    df["Revenue_Potential"] = df["engagement_score"]

# User Status
if "days_since_last_login" in df.columns:
    df["User_Status"] = df["days_since_last_login"].apply(
        lambda x: "Active" if x <= 7 else "Inactive"
    )
else:
    df["User_Status"] = "Unknown"

# =========================
# Sidebar Filters
# =========================
st.sidebar.header("Filters")

if 'Cluster_Label' in df.columns:
    clusters = st.sidebar.multiselect(
        "Select Cluster",
        df['Cluster_Label'].unique(),
        default=df['Cluster_Label'].unique()
    )
    filtered_df = df[df['Cluster_Label'].isin(clusters)]
else:
    filtered_df = df.copy()

# =========================
# KPI SECTION
# =========================
st.subheader("📌 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

total_customers = len(filtered_df)
avg_clv = filtered_df["CLV"].mean()
avg_revenue = filtered_df["Revenue_Potential"].mean()
num_clusters = filtered_df['Cluster_Label'].nunique()

col1.metric("Total Customers", total_customers)
col2.metric("Avg Customer Lifetime Value", f"{avg_clv:.2f}")
col3.metric("Revenue Potential Score", f"{avg_revenue:.2f}")
col4.metric("No of Segments", num_clusters)

st.divider()

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA Analysis",
    "🎯 Customer Segments",
    "📈 Business Insights",
    "📄 Final Report"
])

# =========================
# TAB 1 — EDA
# =========================
with tab1:

    st.subheader("Data Distribution")

    numeric_cols = filtered_df.select_dtypes(include='number').columns

    colA, colB = st.columns(2)

    with colA:
        feature = st.selectbox("Select Feature", numeric_cols, key="eda_feature")
        fig = px.histogram(filtered_df, x=feature, color="Cluster_Label")
        st.plotly_chart(fig, width="stretch")

    with colB:
        fig2 = px.box(filtered_df, x="Cluster_Label", y=feature)
        st.plotly_chart(fig2, width="stretch")

    st.subheader("Correlation Heatmap")

    corr = filtered_df[numeric_cols].corr()
    fig3 = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig3, width="stretch")


# =========================
# TAB 2 — SEGMENTS
# =========================
with tab2:

    st.subheader("Customer Segment Visualization")

    if len(numeric_cols) >= 2:

        x_axis = st.selectbox("X Axis", numeric_cols, index=0, key="x_axis")
        y_axis = st.selectbox("Y Axis", numeric_cols, index=1, key="y_axis")

        fig4 = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color="Cluster_Label",
            title="Cluster Distribution"
        )

        st.plotly_chart(fig4, width="stretch")

    st.subheader("Cluster Size")

    cluster_counts = filtered_df['Cluster_Label'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']

    fig5 = px.pie(cluster_counts, names='Cluster', values='Count')
    st.plotly_chart(fig5, width="stretch")


# =========================
# TAB 3 — BUSINESS INSIGHTS
# =========================
with tab3:

    st.subheader("Active vs Inactive Users")

    status_counts = filtered_df["User_Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]

    fig6 = px.pie(status_counts, names="Status", values="Count")
    st.plotly_chart(fig6, width="stretch")

    st.subheader("Average CLV by Cluster")

    cluster_summary = filtered_df.groupby('Cluster_Label')["CLV"].mean().reset_index()

    fig7 = px.bar(
        cluster_summary,
        x="Cluster_Label",
        y="CLV",
        title="Customer Lifetime Value by Segment"
    )

    st.plotly_chart(fig7, width="stretch")

    st.subheader("Business Table Preview")

    st.dataframe(business.head(50), width="stretch")


# =========================
# TAB 4 — FINAL REPORT
# =========================
with tab4:

    st.subheader("Project Insights")

    st.markdown("""
    ### Key Findings

    ✅ Customers segmented using Machine Learning  
    ✅ High-value customer groups identified  
    ✅ Revenue optimization opportunities discovered  

    ### Business Impact

    📈 Improved marketing targeting  
    💰 Better customer lifetime value prediction  
    🎯 Personalized campaigns possible  

    ### Model Used

    - KMeans Clustering
    - Feature Engineering
    - Data Normalization
    """)

    st.subheader("Download Data")

    st.download_button(
        "Download Clustered File",
        clustered.to_csv(index=False),
        file_name="clustered_output.csv"
    )

    st.download_button(
        "Download Business Table",
        business.to_csv(index=False),
        file_name="final_business_table.csv"
    )