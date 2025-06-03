import requests
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch # Make sure torch is imported
import os

# --- Configuration ---
# # REPLACE THIS WITH THE ACTUAL URL FROM YOUR GOOGLE APPS SCRIPT DEPLOYMENT
# WEB_APP_URL = "https://script.google.com/macros/s/AKfycbyxyKvkq7pIv1V3Ok4AuXI8bYBGqSIqu2vjS_1OKeaymHRFV9HfSoJTk04p3gN094yQzg/exec"
# MODEL_NAME = 'all-MiniLM-L6-v2' # A good lightweight model for sentence embeddings

# Read WEB_APP_URL from environment variable, fallback to hardcoded if not set (e.g., for local testing)
WEB_APP_URL = os.environ.get("WEB_APP_URL", "YOUR_LOCAL_TEST_WEB_APP_URL_HERE")
MODEL_NAME = 'all-MiniLM-L6-v2' # A good lightweight model for sentence embeddings

# --- 1. Fetch Data from Google Sheet ---
def fetch_sheet_data(url):
    """Fetches data from the Google Apps Script web app URL."""
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()
        if not data:
            print("Warning: Google Sheet is empty or no data was returned.")
            return pd.DataFrame() # Return empty DataFrame
        # Assuming the first row is headers if data is not empty
        if len(data) < 1: # No headers or data
             print("Warning: Google Sheet has no headers or data.")
             return pd.DataFrame()
        
        headers = data[0]
        rows = data[1:]
        
        if not rows: # Only headers, no data
            print("Warning: Google Sheet has headers but no data rows.")
            return pd.DataFrame(columns=headers)

        df = pd.DataFrame(rows, columns=headers)
        print(f"Successfully fetched {len(df)} rows from Google Sheet.")
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Google Sheet: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

# --- 2. Load the Language Model ---
def load_model(model_name):
    """Loads a pre-trained Sentence Transformer model."""
    print(f"Loading sentence transformer model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# --- 3. Generate Answer (Intelligent Retrieval) ---
def answer_question_from_sheet(question, sheet_df, model):
    """
    Answers a question by:
    1. Finding the most relevant 'answer column' based on the question.
    2. Finding the most relevant 'lookup row' based on the question.
    3. Retrieving the specific answer from the intersection.
    """
    if sheet_df.empty:
        return "I couldn't retrieve any data from your Google Sheet. Please check the sheet and the web app URL."

    if model is None:
        return "The AI model could not be loaded. Cannot process question."

    columns = sheet_df.columns.tolist()
    if not columns:
        return "Your Google Sheet has no columns (headers). Cannot process question."

    # --- Step 1: Find the best 'answer' column ---
    print("Finding best answer column...")
    column_embeddings = model.encode(columns, convert_to_tensor=True)
    question_embedding = model.encode(question, convert_to_tensor=True)

    column_scores = util.cos_sim(question_embedding, column_embeddings)[0]
    best_column_score, best_column_idx = torch.max(column_scores, dim=0)
    best_answer_column_name = columns[best_column_idx.item()]

    # Define a threshold for how relevant the column needs to be
    COLUMN_THRESHOLD = 0.25 # Adjust as needed. Lower means more flexible.
    if best_column_score.item() < COLUMN_THRESHOLD:
        return f"I couldn't identify a clear column in your sheet to answer '{question}'. My best guess was '{best_answer_column_name}' with a score of {best_column_score.item():.2f}."

    print(f"Identified '{best_answer_column_name}' as the most relevant answer column (Score: {best_column_score.item():.2f}).")

    # --- Step 2: Find the best 'lookup' data in any other column ---
    # We want to find the entity (e.g., "France") that the question is about.
    # Exclude the identified answer column from the lookup columns.
    lookup_columns = [col for col in columns if col != best_answer_column_name]
    
    if not lookup_columns:
        return "Your sheet only has one column. Cannot cross-reference to find specific data."

    best_row_match_score = -1
    best_row_idx = -1
    matched_lookup_value = None
    
    # Iterate through all cells in lookup columns to find the most relevant one
    # This is a bit brute force but effective for small sheets
    print("Searching for relevant data in rows...")
    for col in lookup_columns:
        # Convert column to string type to avoid errors with numbers/booleans
        column_values = sheet_df[col].astype(str).tolist()
        if not column_values: continue # Skip empty columns

        value_embeddings = model.encode(column_values, convert_to_tensor=True)
        # Compare question embedding to each value in the lookup column
        value_scores = util.cos_sim(question_embedding, value_embeddings)[0]
        
        current_best_score, current_best_idx = torch.max(value_scores, dim=0)
        
        if current_best_score.item() > best_row_match_score:
            best_row_match_score = current_best_score.item()
            best_row_idx = current_best_idx.item()
            matched_lookup_value = sheet_df.iloc[best_row_idx][col] # Get the actual value that matched

    # Define a threshold for how relevant the row data needs to be
    ROW_THRESHOLD = 0.35 # Adjust as needed. Higher means more strict.
    if best_row_match_score < ROW_THRESHOLD or best_row_idx == -1:
        return f"I couldn't find specific data related to your question in any row. My best row match score was {best_row_match_score:.2f}."

    print(f"Found relevant row data '{matched_lookup_value}' (Score: {best_row_match_score:.2f}).")

    # --- Step 3: Retrieve the specific answer ---
    final_answer = sheet_df.iloc[best_row_idx][best_answer_column_name]

    # --- Step 4: Formulate a conversational response ---
    # You can enhance this with more sophisticated sentence construction
    # For now, a simple direct answer based on identified parts
    if final_answer is not None and pd.notna(final_answer):
        return f"Based on your sheet data: For '{matched_lookup_value}', the '{best_answer_column_name}' is '{final_answer}'."
    else:
        return f"I found the relevant row for '{matched_lookup_value}', but the value in the '{best_answer_column_name}' column was empty."

# --- Main Execution (keep this part the same) ---
if __name__ == "__main__":
    import torch # Import torch here to ensure it's available

    print("Starting Sheet QA system...")
    sheet_data_df = fetch_sheet_data(WEB_APP_URL)
    qa_model = load_model(MODEL_NAME)

    if not sheet_data_df.empty and qa_model is not None:
        while True:
            user_question = input("\nAsk a question about your Google Sheet data (type 'exit' to quit): \n> ")
            if user_question.lower() == 'exit':
                print("Exiting. Goodbye!")
                break
            
            response = answer_question_from_sheet(user_question, sheet_data_df, qa_model)
            print("\n--- AI Answer ---")
            print(response)
            print("-----------------")
    elif sheet_data_df.empty:
        print("Cannot run QA without sheet data. Please ensure your Google Apps Script is deployed and the URL is correct.")
    else: # qa_model is None
        print("Cannot run QA without a loaded AI model.")