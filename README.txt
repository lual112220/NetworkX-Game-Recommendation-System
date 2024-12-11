Requirements:
Python 3.x
Libraries: streamlit, networkx, pandas, scikit-learn, matplotlib

Install dependencies with:
	pip install streamlit networkx pandas scikit-learn matplotlib

Datasets
	Connections Dataset: Player relationships (CSV)
	Player-Game Data: Player game ratings (CSV)
	Game Attributes: Game genres, popularity (CSV)

How to Use:
	Place the datasets in the same folder as the script.

Run the program:
	steamlit run app.py


Make sure to upload the corrisponding files from the 'data' folder

The program will generate personalized game recommendations for each player based on their network and gaming preferences.
Output

# Network-Based Game Recommendation System

## Overview
This repository contains a game recommendation system that leverages player network analysis and machine learning techniques to provide personalized game suggestions. By analyzing the social connections between players, detecting communities, and considering game attributes, the system delivers more accurate and relevant recommendations than traditional methods.

## Key Features
- **Network Analysis (NetworkX):**  
  Constructs a player network from connection data, computes centrality measures, and detects communities to identify groups of players with similar interests.
  
- **Player Similarity (Collaborative Filtering):**  
  Uses cosine similarity to identify players with comparable gaming behaviors, recommending games that are popular among these similar players.
  
- **Game Clustering (Content-Based Filtering):**  
  Applies K-Means clustering on game attributes (e.g., genre, popularity) to group similar games, enabling suggestions of titles that share attributes with those a player already enjoys.
  
- **Hybrid Recommendation:**  
  Combines collaborative and content-based filtering for comprehensive recommendations, taking into account both social connections and game characteristics.

## Requirements
- Python 3.x
- Libraries:
  - `networkx`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `streamlit` (for the interactive web interface)

Install dependencies:
```bash
pip install networkx pandas numpy scikit-learn matplotlib streamlit
