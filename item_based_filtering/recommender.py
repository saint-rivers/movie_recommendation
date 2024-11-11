from surprise import KNNWithMeans

# To use item-based cosine similarity
sim_options = {
    "name": "pearson_baseline",
    "user_based": False,  # Compute similarities between items
}
model = KNNWithMeans(sim_options=sim_options)