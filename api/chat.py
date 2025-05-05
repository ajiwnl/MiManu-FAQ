from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# Setup
app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")

FALLBACK_RESPONSE = "I'm not sure, but I can assist you with MiManuTMS Frequently asked Questions (FAQ)."

faq_questions = []
faq_answers = []
faq_embeddings = []

# Paths
faq_jsonl_path = os.path.join(os.path.dirname(__file__), "../dataset/faq.jsonl")
embeddings_path = os.path.join(os.path.dirname(__file__), "../dataset/faq_embeddings.npy")
answers_path = os.path.join(os.path.dirname(__file__), "../dataset/faq_answers.json")

# Load data
def load_data():
    global faq_questions, faq_answers, faq_embeddings
    with open(faq_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            user_msg = next((msg["content"] for msg in item["messages"] if msg["role"] == "user"), None)
            assistant_msg = next((msg["content"] for msg in item["messages"] if msg["role"] == "assistant"), None)
            if user_msg and assistant_msg:
                faq_questions.append(user_msg)
                faq_answers.append(assistant_msg)

    if os.path.exists(embeddings_path) and os.path.exists(answers_path):
        faq_embeddings = np.load(embeddings_path)
        with open(answers_path, "r", encoding="utf-8") as f:
            faq_answers = json.load(f)
    else:
        for question in faq_questions:
            response = openai.Embedding.create(
                model="text-embedding-3-small",
                input=question
            )
            embedding = response["data"][0]["embedding"]
            faq_embeddings.append(embedding)
            time.sleep(0.2)

        faq_embeddings_np = np.array(faq_embeddings)
        np.save(embeddings_path, faq_embeddings_np)
        with open(answers_path, "w", encoding="utf-8") as f:
            json.dump(faq_answers, f, ensure_ascii=False)

        faq_embeddings[:] = faq_embeddings_np

load_data()


def is_greeting_or_nonfaq(text):
    text = text.lower()
    keywords = [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "goodbye", "bye", "see you", "take care", "how are you", "how’s it going",
        "talk", "chat", "greetings", "start", "begin", "introduce", "mimanubot", "who are you",
        "help", "just saying", "nice to meet", "catch you", "have a nice", "hope you're well"
    ]
    return any(keyword in text for keyword in keywords)


@app.route("/api/chat", methods=["POST"])
def handler():
    try:
        data = request.get_json()
        message = data.get("message", "").strip()

        if not message:
            return jsonify({"error": "Message cannot be empty"}), 400

        embedding_response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=message
        )
        user_embedding = np.array(embedding_response["data"][0]["embedding"]).reshape(1, -1)

        similarities = cosine_similarity(user_embedding, faq_embeddings).flatten()
        max_index = similarities.argmax()
        max_score = similarities[max_index]

        if max_score > 0.82:
            response_message = faq_answers[max_index]
        elif is_greeting_or_nonfaq(message):
            completion = openai.ChatCompletion.create(
                model="ft:gpt-4.1-mini-2025-04-14:personal::BTh1TBmd",
                messages=[{"role": "user", "content": message}]
            )
            response_message = completion["choices"][0]["message"]["content"].strip()
            response_message = response_message.replace("ChatGPT", "MiManubot").replace("OpenAI", "MiManubot")

            if len(response_message.split()) < 3 or "I don't know" in response_message:
                response_message = FALLBACK_RESPONSE
        else:
            response_message = FALLBACK_RESPONSE

        return jsonify({"response": response_message})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
