import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
import numpy as np
import os
import json
import joblib
from flask import Flask, render_template_string, request, jsonify
import sys
import matplotlib 
matplotlib.use('Agg') # Use non-interactive backend for plotting
import matplotlib.pyplot as plt # Import matplotlib.pyplot for plotting
import io # Import io for image buffer
import base64 # Import base64 for embedding images
# --- Configuration ---
# Set the directory where your data files are located.
# IMPORTANT: Update this path to the actual location of your data files on your system.
DATA_PATH = r'C:\Users\rjaganmohanredddy\OneDrive\Desktop\rs_2' # Make sure this is correct

# Set the directory where trained model and assets will be saved and loaded from.
# This directory will be created if it doesn't exist.
ASSETS_SAVE_DIR = './recommender_assets_pytorch'

# Ensure the assets save directory exists
if not os.path.exists(ASSETS_SAVE_DIR):
    os.makedirs(ASSETS_SAVE_DIR)
    print(f"Created assets save directory: {ASSETS_SAVE_DIR}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# --- 1. Data Loading and Preprocessing ---
print("--- Step 1: Data Loading and Preprocessing ---")

data_paths_to_try = [DATA_PATH, './data/', './']
movies_df, ratings_df, movie_features_df, links_df, tags_df = None, None, None, None, None

for d_path in data_paths_to_try:
    try:
        print(f"Attempting to load data from: {d_path}")
        movies_df = pd.read_csv(os.path.join(d_path, 'movies.csv'))
        ratings_df = pd.read_csv(os.path.join(d_path, 'ratings.csv'))
        movie_features_df = pd.read_csv(os.path.join(d_path, 'movies_with_features.csv'))
        links_df = pd.read_csv(os.path.join(d_path, 'links.csv'))
        tags_df = pd.read_csv(os.path.join(d_path, 'tags.csv'))
        print("DataFrames loaded successfully.")
        break
    except FileNotFoundError as e:
        print(f"Error loading data from {d_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading data from {d_path}: {e}")

if movies_df is None or ratings_df is None or movie_features_df is None or links_df is None or tags_df is None:
    print("\nFailed to load data from all attempted paths. Please ensure data files are available.")
    sys.exit("Data loading failed. Exiting.")

# Handle missing values and prepare content features for CBF
movie_features_df['tmdbId'].fillna(-1, inplace=True)
movie_features_df['year'].fillna(-1, inplace=True)
movie_features_df['tmdb_content'].fillna('', inplace=True)
movie_features_df['content'] = movie_features_df['genres'].fillna('') + ' ' + movie_features_df['tmdb_content'].fillna('')

# Merge ratings with movie features to get content for rated movies
merged_ratings_df = pd.merge(ratings_df, movie_features_df[['movieId', 'content']], on='movieId', how='left')

# Encode user and movie IDs
user_enc = LabelEncoder()
merged_ratings_df['user_encoded'] = user_enc.fit_transform(merged_ratings_df['userId'])
movie_enc = LabelEncoder()
# Fit movie_enc on all unique movie IDs from movies_df to ensure all movies can be encoded
movie_enc.fit(movies_df['movieId'].unique())
merged_ratings_df['movie_encoded'] = movie_enc.transform(merged_ratings_df['movieId'])

# Get the number of unique users and movies
n_users = merged_ratings_df['user_encoded'].nunique()
n_movies = movie_enc.classes_.shape[0] # Use the size of movie_enc classes for total unique movies

print(f"Number of unique users: {n_users}")
print(f"Number of unique movies (from encoder): {n_movies}") # Use count from encoder

# Scale ratings
scaler = StandardScaler()
merged_ratings_df['rating_scaled'] = scaler.fit_transform(merged_ratings_df[['rating']])

# Split data into training and testing sets
train_df, test_df = train_test_split(merged_ratings_df, test_size=0.2, random_state=42)

# Define a custom PyTorch Dataset
class MovieRatingDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        user_id = row['user_encoded']
        movie_id = row['movie_encoded']
        rating = row['rating_scaled']

        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'movie_id': torch.tensor(movie_id, dtype=torch.long),
            'rating': torch.tensor(rating, dtype=torch.float32)
        }

# Create Datasets
train_dataset = MovieRatingDataset(train_df)
test_dataset = MovieRatingDataset(test_df)

# Create DataLoaders
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("\nData loading and preprocessing for PyTorch completed.")
print(f"Number of training batches: {len(train_dataloader)}")
print(f"Number of testing batches: {len(test_dataloader)}")

# Precompute TF-IDF features for all movies for CBF
all_movie_content = movie_features_df['content'].fillna('')
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
all_movie_tfidf_matrix = tfidf_vectorizer.fit_transform(all_movie_content)
print(f"\nTF-IDF matrix shape for all movies: {all_movie_tfidf_matrix.shape}")

# Create a mapping from *encoded* movie ID to its row index in the TF-IDF matrix (movie_features_df index)
# This requires aligning movie_enc classes with movie_features_df
encoded_movieid_to_feature_index = {}
# Iterate through the original movie IDs known by the encoder
for encoded_id, original_id in enumerate(movie_enc.classes_):
    # Find the index of the original movie ID in the movie_features_df
    feature_index = movie_features_df[movie_features_df['movieId'] == original_id].index
    if not feature_index.empty:
        encoded_movieid_to_feature_index[encoded_id] = feature_index[0]

print(f"Created mapping from encoded movie ID to TF-IDF feature index. Number of mapped movies: {len(encoded_movieid_to_feature_index)}")


# Store the dense TF-IDF tensor and the mapping for the model lookup
all_movie_tfidf_dense = all_movie_tfidf_matrix.todense()
all_movie_tfidf_tensor = torch.from_numpy(np.array(all_movie_tfidf_dense)).float().to(device) # Move to device

# --- 2. Define Hybrid Model (PyTorch) ---
print("\n--- Step 2: Defining Hybrid Model (PyTorch) ---")

class HybridRecommender(nn.Module):
    def __init__(self, n_users, n_movies, cf_embedding_size, cbf_feature_size, cbf_embedding_size):
        super().__init__()

        # CF components
        self.user_embedding = nn.Embedding(n_users, cf_embedding_size)
        self.movie_embedding_cf = nn.Embedding(n_movies, cf_embedding_size)

        # CBF components (layers to process the CBF features)
        self.cbf_fc1 = nn.Linear(cbf_feature_size, 256)
        self.cbf_dropout = nn.Dropout(0.3)
        self.cbf_embedding_output = nn.Linear(256, cbf_embedding_size)

        # Combined layers
        self.fc1 = nn.Linear(cf_embedding_size * 2 + cbf_embedding_size, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)

        # Initialize weights (optional, but often good practice)
        self._init_weights()

    def _init_weights(self):
        # Initialize embedding layers with normal distribution
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding_cf.weight, std=0.01)

        # Initialize linear layers with Kaiming/He initialization
        nn.init.kaiming_uniform_(self.cbf_fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.cbf_fc1.bias)
        nn.init.kaiming_uniform_(self.cbf_embedding_output.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.cbf_embedding_output.bias)
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)
        nn.init.kaiming_uniform_(self.output_layer.weight, mode='fan_in', nonlinearity='linear') # Linear output
        nn.init.zeros_(self.output_layer.bias)


    def forward(self, user_ids, movie_ids, all_movie_cbf_features, encoded_movieid_to_feature_index):
        # CF embeddings
        user_embedded = self.user_embedding(user_ids)
        movie_embedded_cf = self.movie_embedding_cf(movie_ids)

        # CBF processing
        # Need to gather the correct CBF features for the movies in the batch
        # Use the encoded movie IDs to look up the index in the all_movie_cbf_features tensor
        batch_cbf_features = []
        for encoded_movie_id in movie_ids.tolist(): # Convert movie_ids tensor to list of Python ints
             feature_index = encoded_movieid_to_feature_index.get(encoded_movie_id, -1) # Get index from mapping
             if feature_index != -1:
                 # Append the feature vector for this movie
                 batch_cbf_features.append(all_movie_cbf_features[feature_index])
             else:
                 # Handle case where movie has no CBF features (e.g., append zero vector or similar)
                 # For simplicity, let's append a zero vector of the correct size
                 batch_cbf_features.append(torch.zeros(all_movie_cbf_features.shape[1], device=self.cbf_fc1.weight.device))

        # Stack the collected features into a tensor
        batch_cbf_features = torch.stack(batch_cbf_features)


        cbf_processed = torch.relu(self.cbf_fc1(batch_cbf_features))
        cbf_processed = self.cbf_dropout(cbf_processed)
        cbf_embedded = torch.relu(self.cbf_embedding_output(cbf_processed))


        # Concatenate CF and CBF features
        combined_features = torch.cat([user_embedded, movie_embedded_cf, cbf_embedded], dim=1)

        # Pass through dense layers
        x = torch.relu(self.fc1(combined_features))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        output = self.output_layer(x)

        return output.squeeze(1) # Squeeze to remove the last dimension of size 1

# Define model parameters
cf_embedding_size = 75
cbf_feature_size = all_movie_tfidf_tensor.shape[1] # Dimension of TF-IDF features
cbf_embedding_size = 128 # Size of the CBF embedding output

# Instantiate the model
model = HybridRecommender(n_users, n_movies, cf_embedding_size, cbf_feature_size, cbf_embedding_size).to(device)
print("PyTorch Hybrid Model defined.")
print(model)

# --- 3. Training Loop (PyTorch) ---
print("\n--- Step 3: Training the Hybrid Model ---")

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10 # Define number of epochs

# Training loop
for epoch in range(num_epochs):
    model.train() # Set model to training mode
    running_loss = 0.0
    for batch in train_dataloader:
        user_ids = batch['user_id'].to(device)
        movie_ids = batch['movie_id'].to(device)
        ratings = batch['rating'].to(device)

        # Forward pass
        outputs = model(user_ids, movie_ids, all_movie_tfidf_tensor, encoded_movieid_to_feature_index)
        loss = criterion(outputs, ratings)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * user_ids.size(0) # Multiply by batch size

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

print("\nPyTorch Hybrid Model training completed.")

# --- 4. Performance Evaluation (PyTorch) ---
print("\n--- Step 4: Evaluating the Hybrid Model ---")

model.eval() # Set model to evaluation mode
test_loss = 0.0
all_predicted_ratings = []
all_actual_ratings = []

with torch.no_grad(): # Disable gradient calculation for evaluation
    for batch in test_dataloader:
        user_ids = batch['user_id'].to(device)
        movie_ids = batch['movie_id'].to(device)
        ratings = batch['rating'].to(device)

        # Forward pass
        outputs = model(user_ids, movie_ids, all_movie_tfidf_tensor, encoded_movieid_to_feature_index)
        loss = criterion(outputs, ratings)

        test_loss += loss.item() * user_ids.size(0)

        # Collect predictions and actual ratings
        all_predicted_ratings.extend(outputs.cpu().numpy()) # Move to CPU and convert to numpy
        all_actual_ratings.extend(ratings.cpu().numpy())

avg_test_loss = test_loss / len(test_dataset)
print(f"Average Test Loss (MSE on scaled ratings): {avg_test_loss:.4f}")

# Inverse transform scaled ratings to original scale for metrics
all_predicted_ratings_orig = scaler.inverse_transform(np.array(all_predicted_ratings).reshape(-1, 1)).flatten()
all_actual_ratings_orig = scaler.inverse_transform(np.array(all_actual_ratings).reshape(-1, 1)).flatten()

# Calculate regression metrics
mse = mean_squared_error(all_actual_ratings_orig, all_predicted_ratings_orig)
mae = mean_absolute_error(all_actual_ratings_orig, all_predicted_ratings_orig)
rmse = np.sqrt(mse)

print(f"\nRegression Metrics (on original rating scale):")
print(f"  Mean Squared Error (MSE): {mse:.4f}")
print(f"  Mean Absolute Error (MAE): {mae:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")

# Calculate classification metrics (e.g., for rating >= 4.0)
threshold = 4.0
actual_binary = (all_actual_ratings_orig >= threshold).astype(int)
predicted_binary = (all_predicted_ratings_orig >= threshold).astype(int)

if len(np.unique(actual_binary)) < 2:
    print("\nCannot calculate classification metrics: Actual ratings do not contain both positive and negative samples based on the threshold.")
else:
    # Handle cases where predicted_binary might lack one class if the model is biased
    precision = precision_score(actual_binary, predicted_binary, zero_division=0)
    recall = recall_score(actual_binary, predicted_binary, zero_division=0)
    f1 = f1_score(actual_binary, predicted_binary, zero_division=0)

    print(f"\nClassification Metrics (threshold >= {threshold}):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    f1 = f1_score(actual_binary, predicted_binary, zero_division=0)

# --- Add Actual vs. Predicted Plot ---
try:
    if len(all_actual_ratings_orig) > 0 and len(all_predicted_ratings_orig) > 0:
        print("\nGenerating Actual vs. Predicted Ratings Plot...")
        # Sample points for faster plotting
        sample_size = 5000 # Define sample size
        if len(all_actual_ratings_orig) > sample_size:
             sample_indices = np.random.choice(len(all_actual_ratings_orig), sample_size, replace=False)
             actual_sample = all_actual_ratings_orig[sample_indices]
             predicted_sample = all_predicted_ratings_orig[sample_indices]
             print(f"Plotting a random sample of {sample_size} points.")
        else:
             actual_sample = all_actual_ratings_orig
             predicted_sample = all_predicted_ratings_orig
             print("Plotting all available test points.")


        plt.figure(figsize=(8, 8))
        plt.scatter(actual_sample, predicted_sample, alpha=0.5, s=5) # Added s=5 for smaller points
        plt.title('Actual vs. Predicted Ratings on Test Set (Sample)') # Updated title to indicate sample
        plt.xlabel('Actual Rating')
        plt.ylabel('Predicted Rating')
        plt.grid(True)
        # Add a diagonal line for perfect prediction
        min_rating = min(min(actual_sample), min(predicted_sample))
        max_rating = max(max(actual_sample), max(predicted_sample))
        plt.plot([min_rating, max_rating], [min_rating, max_rating], 'k--', lw=2) # k-- is black dashed line, lw is line width
        # plt.show() # Removed plt.show() to prevent blocking
        # Save the plot to a file instead
        plot_save_path = os.path.join(ASSETS_SAVE_DIR, 'actual_vs_predicted_plot.png')
        plt.savefig(plot_save_path)
        plt.close() # Close the figure


    else:
        print("\nSkipping Actual vs. Predicted Ratings Plot: No test data available or prediction failed.")
except Exception as e:
    print(f"\nError generating Actual vs. Predicted Ratings Plot: {e}")


# --- 5. Save the Trained Model and Assets (PyTorch) ---
print("\n--- Step 5: Saving the Trained Model and Assets (PyTorch) ---")

# Save the model state dictionary
model_save_path = os.path.join(ASSETS_SAVE_DIR, 'hybrid_recommender_model.pth')
torch.save(model.state_dict(), model_save_path)
print(f"PyTorch model state dictionary saved to: {model_save_path}")

# Save the LabelEncoders and StandardScaler
user_encoder_save_path = os.path.join(ASSETS_SAVE_DIR, 'user_encoder_pytorch.joblib')
joblib.dump(user_enc, user_encoder_save_path)
print(f"User LabelEncoder saved to: {user_encoder_save_path}")

movie_encoder_save_path = os.path.join(ASSETS_SAVE_DIR, 'movie_encoder_pytorch.joblib')
joblib.dump(movie_enc, movie_encoder_save_path)
print(f"Movie LabelEncoder saved to: {movie_encoder_save_path}")

scaler_save_path = os.path.join(ASSETS_SAVE_DIR, 'scaler_pytorch.joblib')
joblib.dump(scaler, scaler_save_path)
print(f"StandardScaler saved to: {scaler_save_path}")

# Save the TF-IDF vectorizer
tfidf_vectorizer_save_path = os.path.join(ASSETS_SAVE_DIR, 'tfidf_vectorizer_pytorch.joblib')
joblib.dump(tfidf_vectorizer, tfidf_vectorizer_save_path)
print(f"TF-IDF Vectorizer saved to: {tfidf_vectorizer_save_path}")

# Save the encoded movie ID to feature index mapping
encoded_movieid_to_feature_index_save_path = os.path.join(ASSETS_SAVE_DIR, 'encoded_movieid_to_feature_index.json')
with open(encoded_movieid_to_feature_index_save_path, 'w') as f:
    # Convert numpy.int64 values to standard Python int before saving
    serializable_mapping = {k: int(v) for k, v in encoded_movieid_to_feature_index.items()}
    json.dump(serializable_mapping, f)
print(f"Encoded movie ID to feature index mapping saved to: {encoded_movieid_to_feature_index_save_path}")

# Save the all_movie_tfidf_tensor (as a numpy array or directly as a tensor file)
# Saving as numpy array first is often more portable
all_movie_tfidf_npy_save_path = os.path.join(ASSETS_SAVE_DIR, 'all_movie_tfidf_tensor.npy')
np.save(all_movie_tfidf_npy_save_path, all_movie_tfidf_tensor.cpu().numpy()) # Save CPU numpy version
print(f"All movie TF-IDF tensor saved to: {all_movie_tfidf_npy_save_path}")

# Save the movies DataFrame (for later lookup of movie titles, etc.)
movies_df_save_path = os.path.join(ASSETS_SAVE_DIR, 'movies_df_pytorch.csv')
movies_df.to_csv(movies_df_save_path, index=False)
print(f"Movies DataFrame saved to: {movies_df_save_path}")

# Save the movie_features_df (for later lookup of movie content for CBF recommendations)
movie_features_df_save_path = os.path.join(ASSETS_SAVE_DIR, 'movie_features_df_pytorch.csv')
movie_features_df.to_csv(movie_features_df_save_path, index=False)
print(f"Movie Features DataFrame saved to: {movie_features_df_save_path}")


# --- 6. Load the Trained Model and Assets (PyTorch) ---
print("\n--- Step 6: Loading the Trained Model and Assets (PyTorch) ---")

loaded_assets_successfully_pytorch = True

# Load the LabelEncoders and StandardScaler
try:
    loaded_user_enc_pytorch = joblib.load(user_encoder_save_path)
    print("User LabelEncoder loaded successfully.")
except Exception as e:
    print(f"Error loading user encoder: {e}")
    loaded_user_enc_pytorch = None
    loaded_assets_successfully_pytorch = False

try:
    loaded_movie_enc_pytorch = joblib.load(movie_encoder_save_path)
    print("Movie LabelEncoder loaded successfully.")
except Exception as e:
    print(f"Error loading movie encoder: {e}")
    loaded_movie_enc_pytorch = None
    loaded_assets_successfully_pytorch = False

try:
    loaded_scaler_pytorch = joblib.load(scaler_save_path)
    print("StandardScaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")
    loaded_scaler_pytorch = None
    loaded_assets_successfully_pytorch = False

# Load the TF-IDF vectorizer
try:
    loaded_tfidf_vectorizer_pytorch = joblib.load(tfidf_vectorizer_save_path)
    print("TF-IDF Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading TF-IDF vectorizer: {e}")
    loaded_tfidf_vectorizer_pytorch = None
    loaded_assets_successfully_pytorch = False

# Load the encoded movie ID to feature index mapping
try:
    with open(encoded_movieid_to_feature_index_save_path, 'r') as f:
        loaded_encoded_movieid_to_feature_index = json.load(f)
    print("Encoded movie ID to feature index mapping loaded successfully.")
    # Convert loaded string keys back to integers if necessary (json loads keys as strings)
    loaded_encoded_movieid_to_feature_index = {int(k): v for k, v in loaded_encoded_movieid_to_feature_index.items()}

except Exception as e:
    print(f"Error loading encoded movie ID to feature index mapping: {e}")
    loaded_encoded_movieid_to_feature_index = None
    loaded_assets_successfully_pytorch = False

# Load the all_movie_tfidf_tensor
try:
    loaded_all_movie_tfidf_tensor_npy = np.load(all_movie_tfidf_npy_save_path)
    loaded_all_movie_tfidf_tensor_pytorch = torch.from_numpy(loaded_all_movie_tfidf_tensor_npy).float().to(device)
    print("All movie TF-IDF tensor loaded successfully.")
except Exception as e:
    print(f"Error loading all movie TF-IDF tensor: {e}")
    loaded_all_movie_tfidf_tensor_pytorch = None
    loaded_assets_successfully_pytorch = False

# Load the movies DataFrame
try:
    loaded_movies_df_pytorch = pd.read_csv(movies_df_save_path)
    print("Movies DataFrame loaded successfully.")
except Exception as e:
    print(f"Error loading movies DataFrame: {e}")
    loaded_movies_df_pytorch = None
    loaded_assets_successfully_pytorch = False

# Load the movie_features_df
try:
    loaded_movie_features_df_pytorch = pd.read_csv(movie_features_df_save_path)
    print("Movie Features DataFrame loaded successfully.")
except Exception as e:
    print(f"Error loading movie features DataFrame: {e}")
    loaded_movie_features_df_pytorch = None
    loaded_assets_successfully_pytorch = False


# Load the model architecture and state dictionary
loaded_model_pytorch = None
if loaded_assets_successfully_pytorch and loaded_all_movie_tfidf_tensor_pytorch is not None and loaded_movie_enc_pytorch is not None and loaded_user_enc_pytorch is not None:
     try:
        # Need to re-instantiate the model with the correct dimensions
        loaded_n_users = len(loaded_user_enc_pytorch.classes_)
        # Use the size of the encoder classes for n_movies
        loaded_n_movies = len(loaded_movie_enc_pytorch.classes_)
        loaded_cbf_feature_size = loaded_all_movie_tfidf_tensor_pytorch.shape[1]

        loaded_model_pytorch = HybridRecommender(loaded_n_users, loaded_n_movies, cf_embedding_size, loaded_cbf_feature_size, cbf_embedding_size).to(device)
        loaded_model_pytorch.load_state_dict(torch.load(model_save_path, map_location=device))
        loaded_model_pytorch.eval() # Set to evaluation mode
        print("PyTorch model loaded successfully.")
     except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        loaded_model_pytorch = None
        loaded_assets_successfully_pytorch = False
else:
     print("Skipping PyTorch model loading due to previous asset loading errors.")


# --- 7. Recommendation Function (PyTorch) ---
print("\n--- Step 7: Defining Recommendation Function (PyTorch) ---")

if loaded_assets_successfully_pytorch and loaded_model_pytorch is not None:

    def recommend_movies_hybrid_pytorch(user_id, n_recommendations, loaded_merged_ratings_df):
        """
        Generates movie recommendations for a given user using the loaded PyTorch hybrid model.

        Args:
            user_id (int): The original user ID.
            n_recommendations (int): The number of movies to recommend.
            loaded_merged_ratings_df (pd.DataFrame): DataFrame containing merged ratings and movie info.

        Returns:
            list: A list of recommended movie titles.
            list: A list of predicted ratings for the recommended movies.
        """
        try:
            # Ensure all necessary loaded assets are available in the function's scope
            if loaded_user_enc_pytorch is None or loaded_movie_enc_pytorch is None or loaded_scaler_pytorch is None or \
               loaded_all_movie_tfidf_tensor_pytorch is None or loaded_encoded_movieid_to_feature_index is None or loaded_movies_df_pytorch is None:
                print("Error: Necessary loaded assets are not available for recommendation.")
                return [], []

            # Encode the user ID
            if user_id not in loaded_user_enc_pytorch.classes_:
                print(f"Error: User ID {user_id} not found in the dataset.")
                return [], []

            user_encoded = loaded_user_enc_pytorch.transform([user_id])[0]

            # Get all unique movie IDs that the user has NOT rated
            rated_movie_ids = loaded_merged_ratings_df[loaded_merged_ratings_df['userId'] == user_id]['movieId'].unique()
            all_movie_ids = loaded_movies_df_pytorch['movieId'].unique()
            unrated_movie_ids = np.setdiff1d(all_movie_ids, rated_movie_ids)

            # Filter unrated movies to only include those known by the movie encoder
            unrated_movie_ids_encoded = [
                 loaded_movie_enc_pytorch.transform([movie_id])[0] for movie_id in unrated_movie_ids
                 if movie_id in loaded_movie_enc_pytorch.classes_ # Ensure the original movie ID is known by the encoder
            ]

            if not unrated_movie_ids_encoded:
                print(f"No unrated movies known by encoder for user {user_id}.")
                return [], []

            # Prepare inputs for prediction
            user_ids_tensor = torch.tensor([user_encoded] * len(unrated_movie_ids_encoded), dtype=torch.long).to(device)
            movie_ids_tensor = torch.tensor(unrated_movie_ids_encoded, dtype=torch.long).to(device)

            # Predict ratings
            with torch.no_grad():
                # Pass necessary CBF data to the model's forward pass
                predicted_scaled_ratings_tensor = loaded_model_pytorch(
                    user_ids_tensor,
                    movie_ids_tensor,
                    loaded_all_movie_tfidf_tensor_pytorch,
                    loaded_encoded_movieid_to_feature_index
                )

            predicted_scaled_ratings = predicted_scaled_ratings_tensor.cpu().numpy()

            # Inverse transform the scaled ratings
            predicted_ratings = loaded_scaler_pytorch.inverse_transform(predicted_scaled_ratings.reshape(-1, 1)).flatten()

            # Create a DataFrame of unrated movies (using original movie IDs) and their predicted ratings
            # Need to map back from encoded movie ID to original movie ID
            encoded_to_original_movieid = {encoded_id: original_id for original_id, encoded_id in zip(loaded_movie_enc_pytorch.classes_, range(len(loaded_movie_enc_pytorch.classes_)))}

            predictions_df = pd.DataFrame({
                'movie_encoded': unrated_movie_ids_encoded,
                'predicted_rating': predicted_ratings
            })

            predictions_df['movieId'] = predictions_df['movie_encoded'].map(encoded_to_original_movieid)

            # Sort by predicted rating in descending order
            recommendations_df = predictions_df.sort_values(by='predicted_rating', ascending=False)

            # Get the top N recommendations
            top_n_recommendations = recommendations_df.head(n_recommendations)

            # Merge with loaded_movies_df_pytorch to get movie titles
            recommended_movies_info = pd.merge(top_n_recommendations, loaded_movies_df_pytorch[['movieId', 'title']], on='movieId', how='left')

            recommended_movie_titles = recommended_movies_info['title'].tolist()
            recommended_predicted_ratings = recommended_movies_info['predicted_rating'].tolist()

            return recommended_movie_titles, recommended_predicted_ratings

        except Exception as e:
            print(f"An error occurred during PyTorch recommendation generation: {e}")
            return [], []


    def recommend_similar_movies_cbf_pytorch(movie_title, n_recommendations=10):
        """
        Generates content-based movie recommendations based on similarity to a given movie title using PyTorch assets.

        Args:
            movie_title (str): The title of the movie to find similar movies for.
            n_recommendations (int): The number of similar movies to recommend.

        Returns:
            list: A list of recommended movie titles.
            list: A list of similarity scores for the recommended movies.
        """
        try:
            # Ensure all necessary loaded assets are available in the function's scope
            if loaded_tfidf_vectorizer_pytorch is None or loaded_movie_features_df_pytorch is None or loaded_movies_df_pytorch is None:
                 print("Error: Necessary loaded assets are not available for content-based recommendation.")
                 return [], []

            # Check if the movie title exists in the loaded movies DataFrame
            if movie_title not in loaded_movies_df_pytorch['title'].values:
                print(f"Movie '{movie_title}' not found in the dataset.")
                return [], []

            # Get the original movie ID for the given title
            movie_id = loaded_movies_df_pytorch[loaded_movies_df_pytorch['title'] == movie_title]['movieId'].iloc[0]

            # Find the index of this movie in the loaded_movie_features_df_pytorch
            movie_index_in_features = loaded_movie_features_df_pytorch[loaded_movie_features_df_pytorch['movieId'] == movie_id].index
            if movie_index_in_features.empty:
                print(f"Movie '{movie_title}' (ID: {movie_id}) not found in movie features data.")
                return [], []

            movie_index = movie_index_in_features[0]

            # Re-calculate TF-IDF matrix using the loaded vectorizer and movie features
            # This might be inefficient, ideally the full TF-IDF matrix or embeddings are loaded
            # Let's use the pre-calculated loaded_all_movie_tfidf_tensor_pytorch if available
            if loaded_all_movie_tfidf_tensor_pytorch is None:
                 print("Error: Pre-calculated TF-IDF tensor is not available for CBF.")
                 # Fallback: calculate TF-IDF for all movies (can be slow)
                 loaded_content_features = loaded_movie_features_df_pytorch['content'].fillna('')
                 loaded_tfidf_matrix = loaded_tfidf_vectorizer_pytorch.transform(loaded_content_features)
                 loaded_tfidf_tensor_for_sim = torch.from_numpy(loaded_tfidf_matrix.todense()).float().to(device)
                 print("Warning: Re-calculated TF-IDF matrix for CBF similarity (can be slow).")
            else:
                 loaded_tfidf_tensor_for_sim = loaded_all_movie_tfidf_tensor_pytorch
                 # Ensure the index aligns with this tensor - assumes loaded_movie_features_df_pytorch index is the same
                 # as the row index in loaded_all_movie_tfidf_tensor_pytorch

            # Get the TF-IDF vector for the target movie
            target_movie_vector = loaded_tfidf_tensor_for_sim[movie_index].unsqueeze(0) # Add batch dimension

            # Compute cosine similarity between the given movie vector and all other movie vectors
            # Using torch.nn.functional.cosine_similarity
            # Need to compute pairwise similarity efficiently
            # One way: matrix multiplication if using dense tensors
            # Or using sklearn's cosine_similarity on numpy arrays (might be easier)

            # Let's use sklearn's cosine_similarity on the loaded numpy matrix for simplicity
            loaded_all_movie_tfidf_matrix_np = loaded_tfidf_tensor_for_sim.cpu().numpy()
            cosine_sim = cosine_similarity(loaded_all_movie_tfidf_matrix_np[movie_index].reshape(1, -1), loaded_all_movie_tfidf_matrix_np).flatten()


            # Get the indices of movies sorted by similarity
            sim_scores = list(enumerate(cosine_sim))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get the top N most similar movies (excluding the input movie itself)
            sim_scores = sim_scores[1:n_recommendations+1]

            # Get the movie indices and similarity scores
            movie_indices = [i[0] for i in sim_scores]
            similarity_scores = [i[1] for i in sim_scores]

            # Get the movie titles for the recommended movies using the original movie_features_df index
            recommended_movie_titles = loaded_movie_features_df_pytorch['title'].iloc[movie_indices].tolist()

            return recommended_movie_titles, similarity_scores

        except Exception as e:
            print(f"An error occurred during content-based recommendation generation: {e}")
            return [], []


    print("PyTorch Recommendation functions defined.")

else:
    print("\nPyTorch Recommendation functions could not be defined because one or more necessary assets failed to load.")


# --- 8. Flask App Integration (PyTorch) ---
print("\n--- Step 8: Defining Basic Flask App (PyTorch) ---")

# Check if all necessary components for the Flask app are loaded
if loaded_assets_successfully_pytorch and loaded_model_pytorch is not None and merged_ratings_df is not None: # Ensure merged_ratings_df is available for hybrid recs

    app = Flask(__name__)
    port = 5002 # Using port 5002

    # Basic HTML template for the web interface
    HTML_TEMPLATE_PYTORCH = """
    <!doctype html>
    <html>
    <head><title>PyTorch Movie Recommender</title></head>
    <body>
        <h1>PyTorch Movie Recommender</h1>

        <h2>Hybrid Deep Learning Recommendation (PyTorch)</h2>
        <form action="/recommend_hybrid_pytorch" method="post">
            User ID: <input type="text" name="user_id"><br><br>
            Number of Recommendations: <input type="number" name="n_recommendations" value="10"><br><br>
            <input type="submit" value="Get Hybrid Recommendations">
        </form>

        <h2>Content-Based Recommendation (PyTorch)</h2>
        <form action="/recommend_cbf_pytorch" method="post">
            Movie Title: <input type="text" name="movie_title"><br><br>
            Number of Recommendations: <input type="number" name="n_recommendations" value="10"><br><br>
            <input type="submit" value="Get Content-Based Recommendations">
        </form>

        <div id="results"></div>

        <script>
            // Optional JavaScript to display results dynamically
            // For simplicity, the current implementation relies on server-side rendering or redirects
        </script>
    </body>
    </html>
    """

    @app.route('/')
    def index_pytorch():
        return render_template_string(HTML_TEMPLATE_PYTORCH)

    @app.route('/recommend_hybrid_pytorch', methods=['POST'])
    def recommend_hybrid_route_pytorch():
        try:
            user_id_str = request.form.get('user_id')
            n_recommendations_str = request.form.get('n_recommendations', '10')

            user_id = int(user_id_str)
            n_recommendations = int(n_recommendations_str)

            # Call the PyTorch recommendation function
            recommended_movies, predicted_ratings = recommend_movies_hybrid_pytorch(user_id, n_recommendations, merged_ratings_df) # Pass merged_ratings_df


            response_html = f"<h3>Hybrid Recommendations (PyTorch) for User {user_id}:</h3>"

            if recommended_movies:
                # --- Removed the bar chart generation and embedding ---
                # fig, ax = plt.subplots(figsize=(10, 6))
                # y_pos = np.arange(len(recommended_movies))
                #
                # # Sort recommendations for plotting from highest to lowest predicted rating
                # plot_data = sorted(zip(recommended_movies, predicted_ratings), key=lambda x: x[1], reverse=False) # Sort ascending for horizontal bar plot bottom-to-top
                # plot_movies = [item[0] for item in plot_data]
                # plot_ratings = [item[1] for item in plot_data]
                #
                #
                # ax.barh(y_pos, plot_ratings, align='center')
                # ax.set_yticks(y_pos)
                # ax.set_yticklabels(plot_movies)
                # ax.set_xlabel('Predicted Rating')
                # ax.set_title(f'Top {n_recommendations} Movie Recommendations for User {user_id}')
                # plt.tight_layout()
                #
                # # Save plot to a BytesIO buffer
                # buf = io.BytesIO()
                # plt.savefig(buf, format='png')
                # buf.seek(0)
                # img_base64 = base64.b64encode(buf.read()).decode('ascii')
                # plt.close(fig) # Close the plot figure to free memory
                #
                #
                # response_html += f'<img src="data:image/png;base64,{img_base64}" alt="Recommendations Bar Chart"><br><br>'
                # --- End of removed bar chart code ---


                response_html += "<h4>Predicted Ratings:</h4><ul>"
                # Print the recommendations in descending order of rating as they appear in the original list
                # Note: the recommended_movies and predicted_ratings lists are already sorted by predicted rating desc.
                for i, movie_title in enumerate(recommended_movies):
                    response_html += f"<li>{i+1}. {movie_title} (Predicted Rating: {predicted_ratings[i]:.2f})</li>"
                response_html += "</ul>"
            else:
                response_html += f"<p>Could not generate hybrid recommendations for user {user_id}. Please check the User ID.</p>"

            return render_template_string(HTML_TEMPLATE_PYTORCH + response_html)

        except ValueError:
            return render_template_string(HTML_TEMPLATE_PYTORCH + "<h3>Error: Invalid input. Please enter a valid User ID and number of recommendations.</h3>")
        except Exception as e:
             return render_template_string(HTML_TEMPLATE_PYTORCH + f"<h3>An error occurred: {e}</h3>")


    @app.route('/recommend_cbf_pytorch', methods=['POST'])
    def recommend_cbf_route_pytorch():
        try:
            movie_title = request.form.get('movie_title')
            n_recommendations_str = request.form.get('n_recommendations', '10')

            n_recommendations = int(n_recommendations_str)

            # Call the PyTorch content-based recommendation function
            recommended_movies, similarity_scores = recommend_similar_movies_cbf_pytorch(movie_title, n_recommendations)

            response_html = f"<h3>Content-Based Recommendations (PyTorch, similar to '{movie_title}'):</h3>"
            if recommended_movies:
                response_html += "<ul>"
                for i, movie_title in enumerate(recommended_movies):
                     response_html += f"<li>{i+1}. {movie_title} (Similarity Score: {similarity_scores[i]:.2f})</li>"
                response_html += "</ul>"
            else:
                response_html += f"<p>Could not generate content-based recommendations for '{movie_title}'. Please check the movie title.</p>"

            return render_template_string(HTML_TEMPLATE_PYTORCH + response_html)

        except ValueError:
             return render_template_string(HTML_TEMPLATE_PYTORCH + "<h3>Error: Invalid input. Please enter a valid number of recommendations.</h3>")
        except Exception as e:
             return render_template_string(HTML_TEMPLATE_PYTORCH + f"<h3>An error occurred: {e}</h3>")


    port = 5002 # Using port 5002
    print(f"Basic Flask app defined, running on port {port}.")

    # Simple local Flask run
    print(f"\n--- Running Flask App Locally (PyTorch) ---")
    print(f" * Running on http://127.0.0.1:{port}/ (Press CTRL+C to quit)")
    # Use debug=True for development, remove for production
    # Use a separate thread to allow the notebook cell to finish (optional, but cleaner in notebooks)
    # In a standard script, app.run() would be at the end and block.
    # For Colab/notebooks, running in a thread prevents the cell from being stuck.
    # Be cautious with threading and Flask in production environments.

    # To run in a thread for notebook execution:
    # Requires installing werkzeug.serving
    # try: # Removed threading try block for simpler execution
    #     from werkzeug.serving import run_simple
    #     flask_thread = threading.Thread(target=run_simple, args=('127.0.0.1', port, app))
    #     flask_thread.daemon = True  # Allow main thread to exit even if flask_thread is alive
    #     flask_thread.start()
    #     print(f"Flask app started in a separate thread. Access at http://127.0.0.1:{port}/")
        # Keep the current cell alive while the Flask app thread is running
        # A simple loop or input can do this, but let's just rely on the user
        # to interact or the notebook environment keeping the process alive.
        # Example: while flask_thread.is_alive(): time.sleep(1)

        # In a standalone script, you would just call:
    if __name__ == '__main__': # Added __name__ == '__main__' guard
        app.run(debug=True, port=port, use_reloader=False) # Run app directly in the main thread, disabled reloader

    #     # For notebook, maybe just a short sleep to let the thread start
    #     time.sleep(2)
    #     print("PyTorch Flask app should be running.")

    # except ImportError: # Removed threading except block
    #     print("Werkzeug is needed to run Flask in a separate thread. Install with `pip install Werkzeug`.")
    #     print("Attempting to run Flask directly (this will block).")
    #     app.run(debug=True, port=port)
    # except Exception as e: # Removed threading except block
    #     print(f"Error starting Flask app thread: {e}")
    #     print("Attempting to run Flask directly (this will block).")
    #     try:
    #         app.run(debug=True, port=port)
    #     except Exception as e2:
    #          print(f"Error running Flask directly: {e2}")
    #          print("Failed to start Flask application.")


else:
    print("\nFlask app could not be defined because one or more necessary assets failed to load.")

print("\n--- PyTorch Implementation Complete (in this cell) ---")






