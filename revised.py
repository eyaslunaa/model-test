import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import os
import re
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


# Load the data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


# Preprocess the data
def preprocess_data(df):
    # Fill NaN values
    df = df.fillna('')

    # Remove irrelevant columns that shouldn't influence recommendations
    columns_to_exclude = ['operating hours', 'contact information']
    for col in columns_to_exclude:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Create a combined text field for each destination
    df['combined_text'] = df['name'] + ' ' + df['description'] + ' ' + df['category'] + ' ' + df['metadata']

    # Encode the categories
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])

    # Save the label encoder mapping for future use
    category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    logger.info(f"Category mapping: {category_mapping}")

    return df, label_encoder


    # Function to extract city and category information from user query
def extract_query_info(query_text, available_cities, available_categories):
    """
    Extract city and category information from a user query.

    Args:
        query_text (str): The user's query
        available_cities (list): List of available cities in the dataset
        available_categories (list): List of available categories in the dataset

    Returns:
        tuple: (extracted_city, extracted_category, cleaned_query)
    """
    query_lower = query_text.lower()
    words = word_tokenize(query_lower)

    # Initialize variables
    extracted_city = None
    extracted_category = None

    # Check city mentions
    for city in available_cities:
        city_lower = city.lower()
        if city_lower in query_lower:
            extracted_city = city
            break

    # wala pang NER sa category
    category_mapping = {
        "cafe": ["cafe", "coffee", "restaurant", "dining"],
        "historical site": ["historical", "history", "heritage", "museum", "shrine"],
        "natural attraction": ["nature", "natural", "outdoors", "mountain", "lake", "volcano", "falls", "waterfall", "things to do"],
        "theme park": ["park", "amusement", "rides", "attraction", "entertainment"],
        "museum": ["museum", "collection", "exhibit", "puzzle", "art"],
        "resort": ["resort", "retreat", "relax", "spa", "wellness", "stay", "accommodation"]
    }

    for category in available_categories:
        category_lower = category.lower()
        # Direct match
        if category_lower in query_lower:
            extracted_category = category
            break

        # Check synonyms
        for cat, synonyms in category_mapping.items():
            if cat.lower() == category_lower:
                for synonym in synonyms:
                    if synonym in query_lower:
                        extracted_category = category
                        break
                if extracted_category:
                    break

    # Remove city and category mentions from query to get cleaner core query
    cleaned_query = query_text
    if extracted_city:
        cleaned_query = re.sub(r'\b' + re.escape(extracted_city) + r'\b', '', cleaned_query, flags=re.IGNORECASE)
    if extracted_category:
        cleaned_query = re.sub(r'\b' + re.escape(extracted_category) + r'\b', '', cleaned_query, flags=re.IGNORECASE)
        # Also remove synonyms
        for cat, synonyms in category_mapping.items():
            if cat.lower() == extracted_category.lower():
                for synonym in synonyms:
                    cleaned_query = re.sub(r'\b' + re.escape(synonym) + r'\b', '', cleaned_query, flags=re.IGNORECASE)

    # Clean up extra spaces
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()

    return extracted_city, extracted_category, cleaned_query


# Create dataset class
class DestinationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Model definition
class DestinationRecommender(torch.nn.Module):
    def __init__(self, num_labels, dropout=0.1):
        super(DestinationRecommender, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use CLS token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# Training function
def train_model(model, train_dataloader, val_dataloader, epochs=10, learning_rate=2e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Lists to store metrics
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for batch in train_progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")

        with torch.no_grad():
            for batch in val_progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                val_progress_bar.set_postfix({'loss': loss.item()})

        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return model, train_losses, val_losses, val_accuracies


# Function to create embeddings for destinations
def create_destination_embeddings(model, dataset, dataloader):
    model.eval()
    all_embeddings = []
    all_indices = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Creating embeddings")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Get the CLS token embedding
            outputs = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            all_embeddings.append(embeddings)
            # Calculate actual indices for this batch
            indices = list(range(batch_idx * dataloader.batch_size,
                                 min((batch_idx + 1) * dataloader.batch_size, len(dataset))))
            all_indices.extend(indices)

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)

    return all_embeddings, all_indices


# Recommendation function with intelligent filtering
def get_recommendations(query_text, tokenizer, model, embeddings, df, city=None, category=None, top_n=3):
    """
    Get destination recommendations based on a query text and optional filters.

    Args:
        query_text (str): The user's query for recommendations
        tokenizer: The tokenizer for encoding the query
        model: The trained RoBERTa model
        embeddings (np.array): Precalculated embeddings for all destinations
        df (pd.DataFrame): The destinations dataframe
        city (str, optional): City name to filter results by. If None, all cities are considered.
        category (str, optional): Category to filter results by. If None, all categories are considered.
        top_n (int): Number of recommendations to return

    Returns:
        tuple: (recommendations DataFrame, similarity scores)
    """
    # Tokenize the query
    query_encoding = tokenizer(
        query_text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    ).to(device)

    # Get the query embedding
    model.eval()
    with torch.no_grad():
        outputs = model.roberta(
            input_ids=query_encoding['input_ids'],
            attention_mask=query_encoding['attention_mask']
        )
        query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Create a series of similarities with df indices
    similarity_series = pd.Series(similarities, index=df.index)

    # Apply filters
    filter_applied = False
    filtered_df = df.copy()

    # Apply city filter if specified
    if city:
        city_mask = filtered_df['city'].str.lower() == city.lower()
        if not any(city_mask):
            logger.warning(f"No destinations found in city: {city}")
            return pd.DataFrame(), np.array([])
        filtered_df = filtered_df[city_mask]
        filter_applied = True

    # Apply category filter if specified
    if category:
        category_mask = filtered_df['category'].str.lower() == category.lower()
        if not any(category_mask):
            logger.warning(f"No destinations found with category: {category}")
            if filter_applied:  # If we already filtered by city
                return pd.DataFrame(), np.array([])
        else:
            filtered_df = filtered_df[category_mask]
            filter_applied = True

    if filter_applied:
        # Get similarities only for filtered destinations
        filtered_indices = filtered_df.index
        filtered_similarities = similarity_series[filtered_indices]

        # Get top recommendations
        if len(filtered_similarities) == 0:
            return pd.DataFrame(), np.array([])

        top_indices = filtered_similarities.nlargest(min(top_n, len(filtered_similarities))).index
        recommendations = df.loc[top_indices]
        scores = filtered_similarities[top_indices].values
    else:
        # Get top recommendations from all destinations
        top_indices = similarity_series.nlargest(top_n).index
        recommendations = df.loc[top_indices]
        scores = similarity_series[top_indices].values

    return recommendations, scores


def main():
    # csv
    file_path = "newdataset.csv"

    # Load df
    df = load_data(file_path)
    df, label_encoder = preprocess_data(df)

    # Data splitting
    texts = df['combined_text'].values
    labels = df['category_encoded'].values
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Initializer Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Datasets
    train_dataset = DestinationDataset(X_train, y_train, tokenizer)
    val_dataset = DestinationDataset(X_val, y_val, tokenizer)
    full_dataset = DestinationDataset(texts, labels, tokenizer)

    # Dataloaders
    batch_size = 4  # Small batch size due to limited data
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    full_dataloader = DataLoader(full_dataset, batch_size=batch_size)

    # Initializing
    num_labels = len(label_encoder.classes_)
    model = DestinationRecommender(num_labels=num_labels).to(device)

    # Training
    model, train_losses, val_losses, val_accuracies = train_model(
        model,
        train_dataloader,
        val_dataloader,
        epochs=5
    )

    output_dir = './model_output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(model.state_dict(), os.path.join(output_dir, 'roberta_destination_model.pt'))
    logger.info(f"Model saved to {output_dir}")

    embeddings, indices = create_destination_embeddings(model, full_dataset, full_dataloader)

    query = "I want a relaxing nature retreat with good views"
    recommendations, scores = get_recommendations(
        query,
        tokenizer,
        model,
        embeddings,
        df,
        city=None,  # No city filter
        top_n=3
    )

    print("\nRecommendations for query:", query)
    print("(No city filter)")
    for i, (idx, row) in enumerate(recommendations.iterrows()):
        print(f"{i + 1}. {row['name']} ({row['city']}) - {row['category']} - Similarity: {scores[i]:.4f}")
        print(f"   {row['description'][:100]}...")

    # Test with city filter
    city_query = ""
    recommendations, scores = get_recommendations(
        query,
        tokenizer,
        model,
        embeddings,
        df,
        city=city_query,  #
        top_n=3
    )

    print(f"\nRecommendations for query: {query}")
    print(f"(Filtered by city: {city_query})")
    if recommendations.empty:
        print(f"No destinations found in {city_query} matching your query.")
    else:
        for i, (idx, row) in enumerate(recommendations.iterrows()):
            print(f"{i + 1}. {row['name']} - {row['category']} - Similarity: {scores[i]:.4f}")
            print(f"   {row['description'][:100]}...")


if __name__ == "__main__":
    main()