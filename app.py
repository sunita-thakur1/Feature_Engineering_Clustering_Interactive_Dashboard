import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(layout="wide")
st.title("🤖 Clustering Analysis Dashboard")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    st.subheader("📊 Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("📈 Grouped Conversion Rates by Model")
    grouped_stats = df.groupby("Model_Used").agg({
        'sex': 'count',
        'user_age': 'mean',
        'user_cuisine': 'count',
        'taste': 'count',
        'Conversion_Rate (%)': 'mean',
        'Likes': 'count'
    })
    st.dataframe(grouped_stats)

    # Best performing model
    conversion_rate_summary = df.groupby("Model_Used")['Conversion_Rate (%)'].mean()
    max_model = conversion_rate_summary.idxmax()
    max_value = conversion_rate_summary.max()
    st.success(f"🏆 Model with highest conversion rate: **{max_model}** ({max_value:.2f}%)")

    # Encode categorical features
    categorical_cols = ["user_cuisine", "sex", "taste"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])

    # Create derived features
    df["Engagement_per_Like"] = df["Engagement_(min/session)"] / (df["Likes"] + 1)
    df["Donation_per_Minute"] = df["Donations ($)"] / (df["Time_Spent (min)"] + 1)

    # Feature selection
    features = [
        "Model_Used", "Likes", "Dislikes", "Donations ($)", "Time_Spent (min)",
        "Conversion_Rate (%)", "Recommendation_Accuracy (%)", "Engagement_(min/session)",
        "user_age", "user_cuisine", "sex", "taste", "Engagement_per_Like", "Donation_per_Minute"
    ]

    # Scale features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

    # Clustering
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df["KMeans_Cluster"] = kmeans.fit_predict(df_scaled)

    dbscan = DBSCAN(eps=1.5, min_samples=5)
    df["DBSCAN_Cluster"] = dbscan.fit_predict(df_scaled)

    hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
    df["Hierarchical_Cluster"] = hierarchical.fit_predict(df_scaled)

    # Evaluation Metrics
    scores = {}
    method_map = {
        "KMeans": "KMeans_Cluster",
        "DBSCAN": "DBSCAN_Cluster",
        "Hierarchical": "Hierarchical_Cluster"
    }

    for method, cluster_col in method_map.items():
        labels = df[cluster_col]
        if len(set(labels)) > 1:
            silhouette = silhouette_score(df_scaled, labels)
            bouldin = davies_bouldin_score(df_scaled, labels)
            scores[method] = {
                "Silhouette Score": round(silhouette, 3),
                "Davies-Bouldin Score": round(bouldin, 3)
            }
        else:
            scores[method] = "Not enough clusters for evaluation"

    # Sidebar
    st.sidebar.header("🔀 Cluster Selection")
    cluster_type = st.sidebar.selectbox("Choose Clustering Method", list(method_map.keys()))

    # Show cluster counts
    st.subheader("📦 Cluster Distribution")
    st.dataframe(df[method_map[cluster_type]].value_counts().rename_axis("Cluster").reset_index(name="Count"))

    # Show scores
    st.subheader("📐 Clustering Evaluation Metrics")
    if isinstance(scores[cluster_type], dict):
        st.json(scores[cluster_type])
    else:
        st.warning(scores[cluster_type])

    # Scatter Plot
    st.subheader(f"📍 {cluster_type} Cluster Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df_scaled.iloc[:, 0],
        y=df_scaled.iloc[:, 1],
        hue=df[method_map[cluster_type]],
        palette="Set2",
        ax=ax
    )
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title(f"{cluster_type} Clustering")
    st.pyplot(fig)

    # Optional: Show processed DataFrame
    st.subheader("🧾 Full Data (with cluster labels)")
    st.dataframe(df)

else:
    st.info("📁 Please upload a CSV file to begin.")
