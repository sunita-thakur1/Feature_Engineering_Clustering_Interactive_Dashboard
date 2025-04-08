import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage


# Streamlit UI
st.title("Clustering Analysis Dashboard")

# Add file uploader to upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

st.set_page_config(page_title="Clustering Dashboard", layout="wide")

st.title("🔍 Interactive Clustering Dashboard")

# Upload CSV
#uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data", df.head())

    # Optional label encoding
    st.sidebar.header("Preprocessing Options")
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        df[col + '_encoded'] = LabelEncoder().fit_transform(df[col])
    
    # Feature selection
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.sidebar.multiselect(
        "Select features for clustering", 
        options=all_numeric, 
        default=all_numeric[:2]
    )

    if len(selected_features) < 2:
        st.warning("Please select at least 2 features.")
    else:
        # Standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df[selected_features])

        # Sidebar clustering settings
        st.sidebar.header("Clustering Parameters")
        k = st.sidebar.slider("Number of clusters (for KMeans & Hierarchical)", 2, 10, 3)
        eps = st.sidebar.slider("DBSCAN eps", 0.1, 5.0, 1.0)
        min_samples = st.sidebar.slider("DBSCAN min_samples", 1, 10, 2)

        # KMeans
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        df["KMeans_Cluster"] = kmeans.fit_predict(features_scaled)

        # DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df["DBSCAN_Cluster"] = dbscan.fit_predict(features_scaled)

        # Agglomerative
        agglom = AgglomerativeClustering(n_clusters=k)
        df["Hierarchical_Cluster"] = agglom.fit_predict(features_scaled)

        # Tabs for visualization
        tab1, tab2, tab3 = st.tabs(["📊 KMeans & DBSCAN", "🌿 Hierarchical Dendrogram", "📈 Cluster Scatter"])

        with tab1:
            st.write("### KMeans Clustering")
            fig1, ax1 = plt.subplots()
            sns.scatterplot(
                x=df[selected_features[0]], 
                y=df[selected_features[1]], 
                hue=df["KMeans_Cluster"], 
                palette="viridis", ax=ax1
            )
            st.pyplot(fig1)

            st.write("### DBSCAN Clustering")
            fig2, ax2 = plt.subplots()
            sns.scatterplot(
                x=df[selected_features[0]], 
                y=df[selected_features[1]], 
                hue=df["DBSCAN_Cluster"], 
                palette="plasma", ax=ax2
            )
            st.pyplot(fig2)

        with tab2:
            st.write("### Hierarchical Dendrogram")
            linkage_matrix = linkage(features_scaled, method='ward')
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            dendrogram(linkage_matrix, ax=ax3)
            st.pyplot(fig3)

        with tab3:
            st.write("### Combined Cluster Labels")
            st.dataframe(df[[*selected_features, "KMeans_Cluster", "DBSCAN_Cluster", "Hierarchical_Cluster"]])

else:
    st.info("👈 Upload a CSV file to get started.")



# Clustering after feature engineering
if uploaded_file is not None:
    # Load the uploaded data
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data:")
    st.write(df.head())
    st.write("### Summary Statistics")
    st.write(df.describe())
    st.write("### Group by Model_Used and find mean of Conversion_Rate ")
    st.write(df.groupby("Model_Used").agg({'sex': 'count',  'user_age': 'mean', 'user_cuisine': 'count', 'user_cuisine':'count', 'taste': 'count', 'Conversion_Rate (%)': 'mean', 'Likes': 'count' }))
    # Group by "Model_Used" and calculate the mean of "Conversion_Rate (%)"
    conversion_rate_summary = df.groupby("Model_Used")['Conversion_Rate (%)'].mean()

   # Find the model with the maximum conversion rate
    max_conversion_model = conversion_rate_summary.idxmax()
    max_conversion_value = conversion_rate_summary.max()

   # Output the result using Streamlit's st.write
    st.write(f"The model with the highest Conversion Rate is Model: {max_conversion_model}, with a Conversion Rate of: {max_conversion_value:.2f}%")

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
