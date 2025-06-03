# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sheet_qa import answer_question_from_sheet, fetch_sheet_data, load_model, WEB_APP_URL, MODEL_NAME
import os # Import os for path manipulation

app = Flask(__name__, static_folder='static', static_url_path='') # Set static folder
CORS(app)

# Load the model once when the server starts
qa_model = load_model(MODEL_NAME)

# --- New Route for Serving index.html ---
@app.route('/')
def serve_index():
    # This will serve the index.html file from the 'static' folder
    # Flask by default looks for 'index.html' if the URL is just '/'
    return send_from_directory(app.static_folder, 'index.html')

# --- Existing Route for AI Chatbot ---
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if qa_model is None:
        return jsonify({"error": "AI model not loaded"}), 500

    current_sheet_data_df = fetch_sheet_data(WEB_APP_URL)

    if current_sheet_data_df.empty:
         return jsonify({"answer": "Cannot retrieve data from Google Sheet. Please check the sheet and its permissions."})

    answer = answer_question_from_sheet(question, current_sheet_data_df, qa_model)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    # Ensure this runs on 0.0.0.0 and the port Cloud Run expects (8080)
    app.run(debug=False, host='0.0.0.0', port=os.environ.get('PORT', 8080))