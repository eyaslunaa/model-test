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
import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.download('punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
    def analyze_sentiment(self, text):
        """
        Analyze sentiment using both TextBlob and VADER for more accurate results
        """
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity  # -1 to 1
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        
        # Extract sentiment-bearing words
        sentiment_words = self._extract_sentiment_words(text)
        
        # Determine overall sentiment
        if textblob_sentiment > 0.1:
            overall = 'positive'
        elif textblob_sentiment < -0.1:
            overall = 'negative'
        else:
            overall = 'neutral'
            
        return {
            'textblob_sentiment': textblob_sentiment,
            'vader_scores': vader_scores,
            'sentiment_words': sentiment_words,
            'overall_sentiment': overall
        }
    
    def _extract_sentiment_words(self, text):
        """
        Extract words that contribute to sentiment
        """
        words = word_tokenize(text.lower())
        sentiment_words = []
        
        for word in words:
            # Check VADER sentiment
            if abs(self.vader.polarity_scores(word)['compound']) > 0.1:
                sentiment_words.append(word)
                
        return sentiment_words

# Download spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import sys

    logger.info("Downloading spaCy model for NER...")
    os.system(f"{sys.executable} -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

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


# Advanced function to extract query information using spaCy NER and pattern matching
def extract_query_info(query_text, available_cities, available_categories, available_budgets=None):
    """
    Extract city, category, and budget information from a user query using NER and pattern matching.
    Now includes sentiment analysis for better understanding of user preferences.
    """
    query_lower = query_text.lower()
    
    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer()
    
    # Get sentiment analysis
    sentiment_info = sentiment_analyzer.analyze_sentiment(query_text)
    
    # Initialize variables
    extracted_city = None
    extracted_category = None
    extracted_budget = None
    budget_amount = None
    
    # Process with spaCy for NER
    doc = nlp(query_text)
    
    # Extract cities using spaCy NER
    for ent in doc.ents:
        if ent.label_ == "GPE" and not extracted_city:  # GPE = Geopolitical Entity
            potential_city = ent.text
            # Verify against our available cities
            for city in available_cities:
                if potential_city.lower() in city.lower() or city.lower() in potential_city.lower():
                    extracted_city = city
                    break
    
    # Backup method: Check direct city mentions if spaCy NER didn't find any
    if not extracted_city:
        for city in available_cities:
            city_lower = city.lower()
            if city_lower in query_lower:
                extracted_city = city
                break
    
    # Category mapping with synonyms and related terms
    category_mapping = {
        "accommodation": ["hotel", "resort", "lodge", "inn", "hostel", "stay", "retreat", "spa resort", "bed and breakfast", "guesthouse", "motel", "lodging", "room", "suite", "villa", "apartment", "condo", "homestay"],
        "cafe": ["cafe", "coffee", "restaurant", "breakfast", "lunch", "dinner"],
        "restaurant":["eat", "hungry"],
        "historical site": ["historical", "history", "heritage", "museum", "shrine", "ancient", "old", "traditional"],
        "natural attraction": ["nature", "natural", "outdoors", "mountain", "lake", "volcano", "falls",
                               "waterfall", "beach", "ocean", "sea", "river", "hiking", "trek", "forest"],
        "theme park": ["park", "amusement", "rides", "attraction", "entertainment", "fun", "thrill"],
        "museum": ["museum", "collection", "exhibit", "gallery", "art", "cultural", "artifacts"],
        "beach resort": ["beach resort", "seaside resort", "coastal resort", "ocean resort", "beachfront resort"]
    }
    
    # Extract category using mapped terms
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
    
    # Extract budget using regex patterns
    budget_patterns = [
        r'(\d+\s*pesos|\d+\s*PHP)',  # Philippine Peso amounts
        r'(budget|cheap|affordable|expensive|luxury|high-end|low-cost)',
        r'(under|below|less than|more than|over)\s*\$*\s*(\d+)',
        r'(\d+)\s*\$',  # 100$
        r'\$\s*(\d+)'  # $ 100
    ]

    def classify_budget(query_lower, category='general'):
        extracted_budget = None

        for pattern in budget_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                # Process the budget information
                budget_text = str(matches[0])

                # Categorize budget based on keywords first
                if any(word in budget_text for word in ['cheap', 'affordable', 'low-cost', 'budget']):
                    extracted_budget = 'low'
                elif any(word in budget_text for word in ['moderate', 'mid', 'medium','mid-range']):
                    extracted_budget = 'medium'
                elif any(word in budget_text for word in ['expensive', 'luxury', 'high-end', 'premium']):
                    extracted_budget = 'high'
                else:
                    # Try to extract a numeric value
                    amount_match = re.search(r'(\d+)', budget_text)
                    if amount_match:
                        amount = int(amount_match.group(1))

                        # Budget classification based on category
                        if 'under' in budget_text or 'below' in budget_text or 'less than' in budget_text:
                            if category == 'restaurant':
                                if amount < 500:
                                    extracted_budget = 'low'
                                elif amount < 1000:
                                    extracted_budget = 'medium'
                                else:
                                    extracted_budget = 'high'
                            elif category == 'cafe':
                                if amount < 250:
                                    extracted_budget = 'low'
                                elif amount < 500:
                                    extracted_budget = 'medium'
                                else:
                                    extracted_budget = 'high'
                            elif category == 'hotel':
                                if amount < 2500:
                                    extracted_budget = 'low'
                                elif amount < 5000:
                                    extracted_budget = 'medium'
                                else:
                                    extracted_budget = 'high'
                            else:  # general category
                                if amount < 500:
                                    extracted_budget = 'low'
                                elif amount < 2000:
                                    extracted_budget = 'medium'
                                else:
                                    extracted_budget = 'high'
                        else:  # When there's no "under/below/less than" qualifier
                            if category == 'restaurant':
                                if amount < 500:
                                    extracted_budget = 'low'
                                elif amount < 1000:
                                    extracted_budget = 'medium'
                                else:
                                    extracted_budget = 'high'
                            elif category == 'cafe':
                                if amount < 250:
                                    extracted_budget = 'low'
                                elif amount < 500:
                                    extracted_budget = 'medium'
                                else:
                                    extracted_budget = 'high'
                            elif category == 'hotel':
                                if amount < 2500:
                                    extracted_budget = 'low'
                                elif amount < 5000:
                                    extracted_budget = 'medium'
                                else:
                                    extracted_budget = 'high'
                            else:  # general category
                                if amount < 500:
                                    extracted_budget = 'low'
                                elif amount < 2000:
                                    extracted_budget = 'medium'
                                else:
                                    extracted_budget = 'high'
                break

        return extracted_budget
    # Remove city, category, and budget mentions from query to get cleaner core query
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
    
    return extracted_city, extracted_category, extracted_budget, cleaned_query, sentiment_info, budget_amount


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


# Enhanced recommendation function with intelligent filtering including budget
def get_recommendations(query_text, tokenizer, model, embeddings, df, city=None, category=None, budget=None, budget_amount=None, top_n=5):
    """
    Get destination recommendations based on a query text and optional filters.
    Now includes numeric budget filtering.
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
            if filter_applied:
                return pd.DataFrame(), np.array([])
        else:
            filtered_df = filtered_df[category_mask]
            filter_applied = True

    # Apply budget filter if specified
    if budget_amount is not None:
        # Convert budget column to numeric, handling any non-numeric values
        filtered_df['budget'] = pd.to_numeric(filtered_df['budget'], errors='coerce')
        
        # Filter based on budget amount
        if 'under' in query_text.lower() or 'below' in query_text.lower() or 'less than' in query_text.lower():
            budget_mask = filtered_df['budget'] <= budget_amount
        else:
            budget_mask = filtered_df['budget'] <= budget_amount * 1.2  # Allow 20% flexibility
        
        if not any(budget_mask):
            logger.warning(f"No destinations found within budget: {budget_amount}")
            if filter_applied:
                return pd.DataFrame(), np.array([])
        else:
            filtered_df = filtered_df[budget_mask]
            filter_applied = True
    elif budget:  # Fallback to categorical budget if no amount specified
        budget_mask = filtered_df['budget'].str.lower() == budget.lower()
        if not any(budget_mask):
            logger.warning(f"No destinations found with budget: {budget}")
            if filter_applied:
                return pd.DataFrame(), np.array([])
        else:
            filtered_df = filtered_df[budget_mask]
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


def load_model(model_path, num_labels):
    """
    Load a saved model from disk.

    Args:
        model_path (str): Path to the saved model
        num_labels (int): Number of labels the model was trained with

    Returns:
        DestinationRecommender: The loaded model
    """
    model = DestinationRecommender(num_labels=num_labels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model


def format_recommendations(recommendations, scores):
    """
    Format recommendations in a user-friendly way.

    Args:
        recommendations (pd.DataFrame): Recommendations dataframe
        scores (np.array): Similarity scores

    Returns:
        str: Formatted recommendations
    """
    if recommendations.empty:
        return "No destinations found matching your criteria."

    result = "Here are your recommended destinations:\n\n"

    for i, (idx, row) in enumerate(recommendations.iterrows()):
        result += f"{i + 1}. {row['name']} ({row['city']})\n"
        result += f"   Category: {row['category']}\n"
        if 'budget' in row:
            result += f"   Budget: {row['budget']}\n"
        result += f"   Match score: {scores[i]:.2f}\n"
        result += f"   {row['description'][:150]}...\n\n"

    return result


def main():
    # Load data file
    file_path = "newdataset.csv"
    model_dir = './model_output/'
    model_path = os.path.join(model_dir, 'roberta_destination_model.pt')

    # Check if model exists, if not, train a new one
    if not os.path.exists(model_path):
        print("No trained model found. Training a new model...")
        # Load and preprocess data
        df = load_data(file_path)
        df, label_encoder = preprocess_data(df)

        # Add a budget column if it doesn't exist (for demo purposes)
        if 'budget' not in df.columns:
            # Assign random budgets for demonstration
            budgets = ['low', 'medium', 'high']
            df['budget'] = np.random.choice(budgets, size=len(df))

        # Data splitting
        texts = df['combined_text'].values
        labels = df['category_encoded'].values
        X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

        # Initialize tokenizer
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

        # Initialize model
        num_labels = len(label_encoder.classes_)
        model = DestinationRecommender(num_labels=num_labels).to(device)

        # Training
        model, train_losses, val_losses, val_accuracies = train_model(
            model,
            train_dataloader,
            val_dataloader,
            epochs=5
        )

        # Save model and embeddings
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        torch.save(model.state_dict(), model_path)

        # Save embeddings
        embeddings, indices = create_destination_embeddings(model, full_dataset, full_dataloader)
        np.save(os.path.join(model_dir, 'destination_embeddings.npy'), embeddings)

        # Save tokenizer
        tokenizer.save_pretrained(os.path.join(model_dir, 'tokenizer'))

        # Save label encoder and other metadata
        import pickle
        with open(os.path.join(model_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump({
                'label_encoder': label_encoder,
                'available_cities': df['city'].unique().tolist(),
                'available_categories': df['category'].unique().tolist(),
                'available_budgets': df['budget'].unique().tolist() if 'budget' in df.columns else None
            }, f)

        print(f"Model and resources saved to {model_dir}")
    else:
        print(f"Loading existing model from {model_path}")

    # Load all required resources
    df = load_data(file_path)
    df, label_encoder = preprocess_data(df)


    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(os.path.join(model_dir, 'tokenizer')
                                                 if os.path.exists(os.path.join(model_dir, 'tokenizer'))
                                                 else 'roberta-base')

    # Load metadata
    import pickle
    metadata_path = os.path.join(model_dir, 'metadata.pkl')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        available_cities = metadata['available_cities']
        available_categories = metadata['available_categories']
        available_budgets = metadata['available_budgets']
    else:
        available_cities = df['city'].unique().tolist()
        available_categories = df['category'].unique().tolist()
        available_budgets = df['budget'].unique().tolist() if 'budget' in df.columns else None

    # Load model
    num_labels = len(label_encoder.classes_)
    model = load_model(model_path, num_labels)

    # Load embeddings
    embeddings_path = os.path.join(model_dir, 'destination_embeddings.npy')
    if os.path.exists(embeddings_path):
        embeddings = np.load(embeddings_path)
    else:
        # Create dataset for embeddings
        texts = df['combined_text'].values
        labels = df['category_encoded'].values
        full_dataset = DestinationDataset(texts, labels, tokenizer)
        full_dataloader = DataLoader(full_dataset, batch_size=4)

        # Create embeddings
        embeddings, _ = create_destination_embeddings(model, full_dataset, full_dataloader)

    # User interaction loop
    print("\n=== Hello, I'm Wertigo, your travel assistant! ===")
    print("Tell me what kind of place you're looking for, and I'll recommend destinations!")
    print("You can specify a city, category (e.g., historical site, natural attraction), and budget.")

    while True:
        user_query = input("\nHow can I help you? ").strip()

        if user_query.lower() == 'exit':
            print("Thank you for using the Travel Destination Recommender. Goodbye!")
            break

        if not user_query:
            print("Please enter a query or type 'exit' to quit.")
            continue

        # Extract query information
        city, category, budget, clean_query, sentiment_info, budget_amount = extract_query_info(
            user_query,
            available_cities,
            available_categories,
            available_budgets
        )

        # Error handling for missing all information. NO CATEGORY, CITY, OR BUDGET EXTRACTED!
        if not city and not category and not budget and not budget_amount:
            print("\n❌ Error: Your query is missing all required information.")
            print("Please specify at least one of the following:")
            print("- A location (e.g., 'in Tagaytay', 'in Imus')")
            print("- A category (e.g., 'cafe', 'historical site', 'natural attraction', 'resort')")
            print("- A budget (e.g., 'under 2000 pesos', 'cheap', 'luxury')")
            print("\nExample queries:")
            print("- 'I want to visit Tagaytay and I only have a budget of 500 pesos, where should I go?'")
            print("- 'Find beach resorts in Ternate under 3000 pesos'")
            print("- 'Hidden gem cafes in Silang'")
            continue

        print("\nProcessing your request...")
        print(f"Detected filters - City: {city or 'Any'}, Category: {category or 'Any'}, Budget: {budget or 'Any'}")
        if budget_amount:
            print(f"Budget amount: {budget_amount} pesos")

        # Get recommendations
        recommendations, scores = get_recommendations(
            clean_query if clean_query else user_query,
            tokenizer,
            model,
            embeddings,
            df,
            city=city,
            category=category,
            budget=budget,
            budget_amount=budget_amount,
            top_n=5
        )

        # Handle the case when no recommendations are found
        if recommendations.empty:
            print("\n❌ Sorry, I couldn't find any destinations matching your criteria.")
            print("Try adjusting your search filters or use more general terms.")

            # Suggest alternatives
            suggestion = ""
            if city and category and budget:
                # Try without budget
                alt_recommendations, _ = get_recommendations(
                    clean_query if clean_query else user_query,
                    tokenizer, model, embeddings, df, city=city, category=category, budget=None, budget_amount=None, top_n=1
                )
                if not alt_recommendations.empty:
                    suggestion = f"Try searching for {category} in {city} without budget constraints."
                else:
                    # Try without category
                    alt_recommendations, _ = get_recommendations(
                        clean_query if clean_query else user_query,
                        tokenizer, model, embeddings, df, city=city, category=None, budget=None, budget_amount=None, top_n=1
                    )
                    if not alt_recommendations.empty:
                        suggestion = f"Try searching for any category in {city}."
                    else:
                        suggestion = "Try searching without any filters."
            elif city and category:
                # Try without category
                alt_recommendations, _ = get_recommendations(
                    clean_query if clean_query else user_query,
                    tokenizer, model, embeddings, df, city=city, category=None, budget=None, budget_amount=None, top_n=1
                )
                if not alt_recommendations.empty:
                    suggestion = f"Try searching for any category in {city}."
                else:
                    suggestion = "Try searching in a different city or without any filters."
            elif city:
                suggestion = "Try searching in a different city or use more general terms."
            elif category:
                suggestion = "Try searching for a different category or use more general terms."

            if suggestion:
                print(f"\nSuggestion: {suggestion}")
        else:
            # Display formatted recommendations
            print("\n✅ Here are your recommended destinations:\n")
            for i, (idx, row) in enumerate(recommendations.iterrows()):
                print(f"{i + 1}. {row['name']} ({row['city']})")
                print(f"   Category: {row['category']}")
                if 'budget' in row:
                    print(f"   Budget: {row['budget']}")
                print(f"   Match score: {scores[i]:.2f}")
                print(f"   {row['description'][:150]}...\n")


if __name__ == "__main__":
    main()