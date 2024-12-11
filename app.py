import streamlit as st
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from io import BytesIO

# Helper function to create NetworkX graph
def create_graph(connections_df):
    G = nx.Graph()
    G.add_edges_from(connections_df.values)
    return G

# Helper function to visualize graph
def plot_graph(G):
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='lightblue', font_weight='bold', ax=ax)
    return fig

# Streamlit App
st.title("Network-Based Game Recommendation System")

# Sidebar for uploading data
st.sidebar.header("Upload Data Files")
connections_file = st.sidebar.file_uploader("Upload Connections CSV", type=["csv"])
player_game_file = st.sidebar.file_uploader("Upload Player-Game Data CSV", type=["csv"])
game_attributes_file = st.sidebar.file_uploader("Upload Game Attributes CSV", type=["csv"])

if connections_file and player_game_file and game_attributes_file:
    # Load datasets
    connections = pd.read_csv(connections_file)
    player_game_data = pd.read_csv(player_game_file)
    game_attributes = pd.read_csv(game_attributes_file)

    # Step 1: Network Analysis
    st.header("Network Analysis and Visualization")
    G = create_graph(connections)
    st.subheader("Player Network")
    st.pyplot(plot_graph(G))

    # Centrality Analysis
    st.subheader("Centrality Analysis")
    centrality_measure = st.selectbox("Select Centrality Measure", ["PageRank", "Closeness Centrality"])
    if centrality_measure == "PageRank":
        centrality_scores = nx.pagerank(G)
    elif centrality_measure == "Closeness Centrality":
        centrality_scores = nx.closeness_centrality(G)
    
    centrality_df = pd.DataFrame(centrality_scores.items(), columns=["Player", "Score"]).sort_values(by="Score", ascending=False)
    st.write("Centrality Scores:")
    st.dataframe(centrality_df)

    # Step 2: Player Similarity
    st.header("Player Similarity")
    player_game_matrix = player_game_data.drop("Player", axis=1).values
    similarity_matrix = cosine_similarity(player_game_matrix)
    similarity_df = pd.DataFrame(
        similarity_matrix, 
        index=player_game_data["Player"], 
        columns=player_game_data["Player"]
    )
    st.write("Player Similarity Matrix:")
    st.dataframe(similarity_df)

    # Step 3: Community Detection
    st.header("Community Detection")
    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G))
    community_mapping = {player: idx for idx, group in enumerate(communities) for player in group}
    st.write("Detected Communities:")
    st.write(community_mapping)

    # Step 4: Game Clustering
    st.header("Game Clustering")
    scaler = MinMaxScaler()
    normalized_attributes = scaler.fit_transform(game_attributes[["Popularity"]])
    kmeans = KMeans(n_clusters=2, random_state=42).fit(normalized_attributes)
    game_attributes["Cluster"] = kmeans.labels_
    st.write("Game Clusters:")
    st.dataframe(game_attributes)

    # Step 5: Recommendations
    st.header("Generate Recommendations")
    selected_player = st.selectbox("Select a Player for Recommendations", player_game_data["Player"])

    # Collaborative Filtering: Based on Similar Players
    def collaborative_filtering(player, top_n=2):
        similar_players = similarity_df.loc[player].sort_values(ascending=False).index[1:3]
        games_played = player_game_data[player_game_data["Player"].isin(similar_players)]
        top_games = games_played.drop("Player", axis=1).sum().sort_values(ascending=False).index[:top_n]
        return top_games

    # Content-Based Filtering: Based on Game Attributes
    def content_based_filtering(player, top_n=2):
        played_games = player_game_data[player_game_data["Player"] == player].drop("Player", axis=1).columns
        game_clusters = game_attributes[game_attributes["Game"].isin(played_games)]["Cluster"].unique()
        cluster_recommendations = game_attributes[game_attributes["Cluster"].isin(game_clusters)]["Game"]
        return cluster_recommendations[:top_n]

    # Hybrid Approach: Combine Collaborative and Content-Based
    def hybrid_approach(player, top_n=2):
        collaborative_games = set(collaborative_filtering(player, top_n))
        content_based_games = set(content_based_filtering(player, top_n))
        return collaborative_games.union(content_based_games)

    recommendation_method = st.selectbox("Select Recommendation Method", ["Collaborative Filtering", "Content-Based Filtering", "Hybrid Approach"])

    if recommendation_method == "Collaborative Filtering":
        recommendations = collaborative_filtering(selected_player)
    elif recommendation_method == "Content-Based Filtering":
        recommendations = content_based_filtering(selected_player)
    elif recommendation_method == "Hybrid Approach":
        recommendations = hybrid_approach(selected_player)

    if st.button("Get Recommendations"):
        st.write(f"Recommendations for {selected_player}: {recommendations}")

    # Export Similarity Matrix
    st.header("Download Results")
    buffer = BytesIO()
    similarity_df.to_csv(buffer, index=True)
    buffer.seek(0)
    st.download_button("Download Similarity Matrix CSV", buffer, "similarity_matrix.csv")
