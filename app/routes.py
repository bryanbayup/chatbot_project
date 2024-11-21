# app/routes.py

from flask import Blueprint, request, jsonify
from app.utils import chatbot_response

main = Blueprint('main', __name__)

@main.route('/health/', methods=['GET'])
def health():
    return jsonify({"message": "Chatbot API is running.", "status": "OK"}), 200

@main.route('/chat/', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'text' not in data or 'session_id' not in data:
            return jsonify({"error": "Invalid request data"}), 400

        text = data['text']
        session_id = data['session_id']

        response = chatbot_response(text)
        return jsonify({"response": response}), 200
    except Exception as e:
        # Log error details
        print(f"Error in /chat/ endpoint: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

