import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


st.set_page_config(
    page_title="Customer Segmentation System",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Customer Segmentation & Service Recommendation System")

st.markdown("""
### Veda Technology Customer Analytics

This application uses customer behavior, engagement,
spending and learning-interest data to identify customer
segments and recommend suitable services.
""")


@st.cache_data
def load_data():
    return pd.read_csv("data/customer_segmented_data.csv")


df = load_data()


# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.header("Dashboard Controls")

selected_segment = st.sidebar.selectbox(
    "Select Customer Segment",
    ["All"] + sorted(df["Segment"].unique())
)


# -------------------------
# FILTER
# -------------------------

if selected_segment == "All":
    filtered_df = df
else:
    filtered_df = df[
        df["Segment"] == selected_segment
    ]


# -------------------------
# KPI
# -------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    len(filtered_df)
)

col2.metric(
    "Average Spend",
    f"₹{filtered_df['Monthly_Spend'].mean():,.0f}"
)

col3.metric(
    "Avg Engagement",
    f"{filtered_df['Engagement_Score'].mean():.1f}"
)

col4.metric(
    "Avg Learning Interest",
    f"{filtered_df['Learning_Interest'].mean():.1f}"
)


st.divider()


# -------------------------
# SEGMENT DISTRIBUTION
# -------------------------

st.subheader("Customer Segment Distribution")

segment_counts = (
    df["Segment"]
    .value_counts()
)

st.bar_chart(segment_counts)


# -------------------------
# CLUSTER VISUALIZATION
# -------------------------

st.subheader("Customer Segmentation Map")

fig, ax = plt.subplots()

for segment in df["Segment"].unique():

    data = df[df["Segment"] == segment]

    ax.scatter(
        data["PCA1"],
        data["PCA2"],
        label=segment,
        alpha=0.7
    )

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("2D Customer Segmentation")

ax.legend()

st.pyplot(fig)


# -------------------------
# SEGMENT PROFILE
# -------------------------

st.subheader("Segment Profile")

profile = (
    filtered_df
    .groupby("Segment")
    [
        [
            "Monthly_Spend",
            "Purchase_Frequency",
            "Engagement_Score",
            "Service_Inquiries",
            "Learning_Interest"
        ]
    ]
    .mean()
    .round(2)
)

st.dataframe(
    profile,
    use_container_width=True
)
# -------------------------
# MODEL INSIGHTS
# -------------------------

st.subheader("Clustering Model Insights")

col1, col2 = st.columns(2)

col1.metric(
    "Number of Clusters",
    df["Cluster"].nunique()
)

col2.metric(
    "Customers Analyzed",
    len(df)
)

st.caption(
    "K-Means clustering was evaluated using the elbow method "
    "and silhouette score. PCA is used to visualize the "
    "high-dimensional customer segments in 2D."
)

# -------------------------
# RECOMMENDATIONS
# -------------------------

st.subheader("Recommended Services")
st.info(
    "Recommendations are generated using transparent "
    "business rules based on customer engagement, spending, "
    "learning interest and service inquiries."
)
recommendations = (
    filtered_df[
        [
            "Customer_ID",
            "Segment",
            "Recommended_Service"
        ]
    ]
    .head(50)
)

st.dataframe(
    recommendations,
    use_container_width=True
)


# -------------------------
# BUSINESS INSIGHTS
# -------------------------

st.subheader("Business Recommendations")

st.markdown("""
- **Premium Engaged Customers:** Offer premium memberships,
  advanced services and loyalty benefits.

- **Occasional Users:** Use targeted engagement campaigns,
  introductory offers and starter programs.

- **Learning-Focused Customers:** Recommend structured
  learning programs, certifications and advanced courses.

- **Service-Support Customers:** Prioritize customer support,
  service assistance and proactive issue resolution.
""")