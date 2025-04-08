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

    conversion_rate_summary = df.groupby("Model_Used")['Conversion_Rate (%)'].mean()
    max_model = conversion_rate_summary.idxmax()
    max_value = conversion_rate_summary.max()
    st.success(f"🏆 Model with highest conversion rate: **{max_model}** ({max_value:.2f}%)")

    # Encode categorical features
    categorical_cols = ["user_cuisine", "sex", "taste"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])

    # Add derived features
    df["Engagement_per_Like"] = df["Engagement_(min/session)"] / (df["Likes"] + 1)
    df["Donation_per_Minute"] = df["Donations ($)"] / (df["Time_Spent (min)"] + 1)

    # --- Feature Selection ---
    all_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_features = st.multiselect("🔧 Select features for clustering", all_features, default=all_features[:5])

    if len(selected_features) < 2:
        st.warning("Please select at least two features for clustering.")
        st.stop()

    # --- Standardize ---
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[selected_features]), columns=selected_features)

    # --- Sidebar Parameters ---
    st.sidebar.header("🔀 Clustering Options")
    n_clusters = st.sidebar.slider("Number of Clusters (KMeans & Hierarchical)", 2, 10, 4)
    dbscan_eps = st.sidebar.slider("DBSCAN: eps (neighborhood radius)", 0.1, 10.0, 1.5)
    dbscan_min_samples = st.sidebar.slider("DBSCAN: min_samples", 1, 10, 5)
    cluster_method = st.sidebar.selectbox("Clustering Method", ["KMeans", "DBSCAN", "Hierarchical"])

    # --- Clustering ---
    df["KMeans_Cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(df_scaled)
    df["DBSCAN_Cluster"] = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(df_scaled)
    df["Hierarchical_Cluster"] = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(df_scaled)

    # --- Evaluation Scores ---
    scores = {}
    method_map = {
        "KMeans": "KMeans_Cluster",
        "DBSCAN": "DBSCAN_Cluster",
        "Hierarchical": "Hierarchical_Cluster"
    }

    for method, col in method_map.items():
        labels = df[col]
        if len(set(labels)) > 1:
            scores[method] = {
                "Silhouette Score": round(silhouette_score(df_scaled, labels), 3),
                "Davies-Bouldin Score": round(davies_bouldin_score(df_scaled, labels), 3)
            }
        else:
            scores[method] = "Not enough clusters for evaluation"

    # --- Cluster Counts ---
    st.subheader("📦 Cluster Distribution")
    cluster_counts = df[method_map[cluster_method]].value_counts().sort_index()
    st.dataframe(cluster_counts.rename_axis("Cluster").reset_index(name="Count"))

    # --- Cluster Metrics ---
    st.subheader("📐 Clustering Evaluation Metrics")
    if isinstance(scores[cluster_method], dict):
        st.json(scores[cluster_method])
    else:
        st.warning(scores[cluster_method])

    # --- Visualization ---
    st.subheader("📍 Cluster Visualization (2D Projection)")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df_scaled[selected_features[0]],
        y=df_scaled[selected_features[1]],
        hue=df[method_map[cluster_method]],
        palette="tab10",
        ax=ax
    )
    ax.set_xlabel(selected_features[0])
    ax.set_ylabel(selected_features[1])
    ax.set_title(f"{cluster_method} Clustering")
    st.pyplot(fig)

    # --- Optional: Show processed data ---
    st.subheader("🧾 Data with Cluster Labels")
    st.dataframe(df.head())

else:
    st.info("📁 Please upload a CSV file to begin.")
