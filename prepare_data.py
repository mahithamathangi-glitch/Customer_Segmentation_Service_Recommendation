import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ============================================================
# 1. LOAD CUSTOMER DATA
# ============================================================

input_file = "data/customer_segmented_data.csv"

df = pd.read_csv(input_file)

print("=" * 60)
print("CUSTOMER SEGMENTATION DATA PREPARATION")
print("=" * 60)

print("\nOriginal dataset shape:", df.shape)
print("\nOriginal columns:")
print(df.columns.tolist())


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

df["Spend_per_Purchase"] = (
    df["Monthly_Spend"] /
    df["Purchase_Frequency"].replace(0, 1)
)

df["Engagement_per_Session"] = (
    df["Engagement_Score"] /
    df["Avg_Session_Minutes"].replace(0, 1)
)

df["Customer_Value_Score"] = (
    0.4 * (df["Monthly_Spend"] / df["Monthly_Spend"].max())
    + 0.3 * (df["Purchase_Frequency"] / df["Purchase_Frequency"].max())
    + 0.3 * (df["Engagement_Score"] / df["Engagement_Score"].max())
) * 100


print("\nFeature engineering completed.")


# ============================================================
# 3. SELECT CLUSTERING FEATURES
# ============================================================

clustering_features = [
    "Age",
    "Tenure_Months",
    "Monthly_Spend",
    "Purchase_Frequency",
    "Engagement_Score",
    "Service_Inquiries",
    "Learning_Interest",
    "Avg_Session_Minutes",
    "Spend_per_Purchase",
    "Customer_Value_Score"
]

X = df[clustering_features]


# ============================================================
# 4. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Feature standardization completed.")


# ============================================================
# 5. ELBOW + SILHOUETTE EVALUATION
# ============================================================

inertia = []
silhouette_scores = []

K_range = range(2, 9)

for k in K_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertia.append(model.inertia_)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)


print("\nCluster Evaluation")
print("-" * 60)

for k, score in zip(K_range, silhouette_scores):
    print(f"K = {k} | Silhouette Score = {score:.4f}")


best_k = K_range[np.argmax(silhouette_scores)]

print("\nBest K based on silhouette score:", best_k)


# ============================================================
# 6. FINAL K-MEANS MODEL
# ============================================================

# Four segments provide good business interpretability
# while still being evaluated using clustering metrics.

final_k = 4

kmeans = KMeans(
    n_clusters=final_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

final_silhouette = silhouette_score(
    X_scaled,
    df["Cluster"]
)

print("\nFinal number of clusters:", final_k)
print(
    "Final silhouette score:",
    round(final_silhouette, 4)
)


# ============================================================
# 7. CLUSTER PROFILES
# ============================================================

cluster_profile = df.groupby("Cluster")[
    [
        "Monthly_Spend",
        "Purchase_Frequency",
        "Engagement_Score",
        "Service_Inquiries",
        "Learning_Interest",
        "Customer_Value_Score"
    ]
].mean()

print("\nCluster Profiles:")
print(cluster_profile.round(2))


# ============================================================
# 8. MEANINGFUL SEGMENT NAMES
# ============================================================

# Identify clusters according to their dominant business behavior.

remaining = set(cluster_profile.index)

# Highest customer value
premium_cluster = cluster_profile[
    "Customer_Value_Score"
].idxmax()

remaining.remove(premium_cluster)


# Highest learning interest among remaining clusters
learning_cluster = (
    cluster_profile.loc[list(remaining)]
    ["Learning_Interest"]
    .idxmax()
)

remaining.remove(learning_cluster)


# Highest service inquiries among remaining clusters
service_cluster = (
    cluster_profile.loc[list(remaining)]
    ["Service_Inquiries"]
    .idxmax()
)

remaining.remove(service_cluster)


# Remaining cluster = occasional/general customers
occasional_cluster = list(remaining)[0]


segment_names = {
    premium_cluster: "Premium Engaged Customers",
    learning_cluster: "Learning-Focused Customers",
    service_cluster: "Service-Support Customers",
    occasional_cluster: "Occasional Users"
}

df["Segment"] = df["Cluster"].map(segment_names)


print("\nSegment Mapping:")
for cluster, name in segment_names.items():
    print(f"Cluster {cluster} -> {name}")


# ============================================================
# 9. RECOMMENDATION ENGINE
# ============================================================

def recommend_service(row):

    if row["Learning_Interest"] >= 70:

        return "Advanced Data Science / AI Learning Program"

    elif (
        row["Monthly_Spend"] >= 7000
        and row["Engagement_Score"] >= 70
    ):

        return "Premium Membership & Advanced Services"

    elif row["Service_Inquiries"] >= 7:

        return "Priority Customer Support Program"

    elif row["Engagement_Score"] < 40:

        return "Engagement Booster Program"

    else:

        return "Starter Learning & Service Package"


df["Recommended_Service"] = df.apply(
    recommend_service,
    axis=1
)


print("\nRecommendation logic completed.")


# ============================================================
# 10. PCA FOR 2D VISUALIZATION
# ============================================================

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]


explained_variance = pca.explained_variance_ratio_

print("\nPCA completed.")

print(
    "Variance explained by PCA1:",
    round(explained_variance[0] * 100, 2),
    "%"
)

print(
    "Variance explained by PCA2:",
    round(explained_variance[1] * 100, 2),
    "%"
)


# ============================================================
# 11. SAVE FINAL DATASET
# ============================================================

output_file = "data/customer_segmented_data.csv"

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 12. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print("\nFinal shape:", df.shape)

print("\nFinal columns:")
print(df.columns.tolist())

print("\nSegment distribution:")
print(df["Segment"].value_counts())

print("\nSample records:")
print(
    df[
        [
            "Customer_ID",
            "Cluster",
            "Segment",
            "Recommended_Service"
        ]
    ].head(10)
)

print(
    f"\nSaved to: {output_file}"
)