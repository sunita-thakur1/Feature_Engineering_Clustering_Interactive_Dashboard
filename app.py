import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Streamlit UI
st.title("Clustering Analysis Dashboard")

# Add file uploader to upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the uploaded data
    df = pd.read_csv(uploaded_file)
    #st.df(head)

    # Encode categorical variables
    categorical_cols = ["user_cuisine", "sex", "taste"]
    for col in categorical_cols:
        if col in df.columns:  # Ensure the column exists in the dataframe
            df[col] = LabelEncoder().fit_transform(df[col])

    # Create new features
    df["Engagement_per_Like"] = df["Engagement_(min/session)"] / (df["Likes"] + 1)
    df["Donation_per_Minute"] = df["Donations ($)"] / (df["Time_Spent (min)"] + 1)

    # Select features for clustering
    features = [
        "Model_Used", "Likes", "Dislikes", "Donations ($)", "Time_Spent (min)",
        "Conversion_Rate (%)", "Recommendation_Accuracy (%)", "Engagement_(min/session)",
        "user_age", "user_cuisine", "sex", "taste", "Engagement_per_Like", "Donation_per_Minute"
    ]

    # Standardize numerical features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

    # Apply clustering algorithms
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df["KMeans_Cluster"] = kmeans.fit_predict(df_scaled)

    dbscan = DBSCAN(eps=1.5, min_samples=5)
    df["DBSCAN_Cluster"] = dbscan.fit_predict(df_scaled)

    hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
    df["Hierarchical_Cluster"] = hierarchical.fit_predict(df_scaled)

    # Compute evaluation scores
    scores = {}
    method_map = {"KMeans": "KMeans_Cluster", "DBSCAN": "DBSCAN_Cluster", "Hierarchical": "Hierarchical_Cluster"}
    for method, cluster_col in method_map.items():
        labels = df[cluster_col]
        if len(set(labels)) > 1:  # Silhouette score requires at least 2 clusters
            silhouette = silhouette_score(df_scaled, labels)
            bouldin = davies_bouldin_score(df_scaled, labels)
            scores[method] = {"Silhouette Score": silhouette, "Davies-Bouldin Score": bouldin}
        else:
            scores[method] = "Not enough clusters for evaluation"

    # Streamlit UI - Cluster Selection
    st.sidebar.header("Cluster Selection")
    cluster_type = st.sidebar.selectbox("Select Clustering Method", list(method_map.keys()))

    st.write("### Cluster Distribution")
    st.write(df[method_map[cluster_type]].value_counts())

    # Display evaluation scores
    st.write("### Clustering Performance Metrics")
    if isinstance(scores[cluster_type], dict):
        st.json(scores[cluster_type])
    else:
        st.write(scores[cluster_type])

    # Scatter plot visualization
    st.write("### Cluster Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df_scaled.iloc[:, 0], y=df_scaled.iloc[:, 1], hue=df[method_map[cluster_type]], palette="viridis", ax=ax)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title(f"{cluster_type} Cluster Visualization")
    st.pyplot(fig)

else:
    st.write("Please upload a CSV file to proceed.")
