from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
import json
import numpy as np

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")

FALLBACK_RESPONSE = "I'm not sure, but I can assist you with MiManuTMS Frequently asked Questions (FAQ)"

# Load and embed FAQ dataset once at startup
FAQS = []
FAQ_EMBEDDINGS = []


def load_faq_embeddings():
    global FAQS, FAQ_EMBEDDINGS
    with open("dataset/faq.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            user_msg = item["messages"][1]["content"]
            FAQS.append(item)
            embedding = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=user_msg
            )["data"][0]["embedding"]
            FAQ_EMBEDDINGS.append(np.array(embedding))


load_faq_embeddings()


def find_best_faq_match(user_input, threshold=0.80):
    user_embedding = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=user_input
    )["data"][0]["embedding"]
    user_embedding = np.array(user_embedding)

    similarities = [np.dot(user_embedding, faq_vec) / (np.linalg.norm(user_embedding) * np.linalg.norm(faq_vec))
                    for faq_vec in FAQ_EMBEDDINGS]

    best_idx = int(np.argmax(similarities))
    best_score = similarities[best_idx]

    if best_score >= threshold:
        return FAQS[best_idx]["messages"][2]["content"]
    return None


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        message = data.get("message", "").strip()

        if not message:
            return jsonify({"error": "Message cannot be empty"}), 400

        # First, try to answer from the FAQ
        faq_response = find_best_faq_match(message)
        if faq_response:
            return jsonify({"response": faq_response})

        # If not matched, fall back to fine-tuned OpenAI model
        completion = openai.ChatCompletion.create(
            model="ft:gpt-4.1-mini-2025-04-14:personal::BTh1TBmd",
            messages=[{"role": "user", "content": message}]
        )

        response_message = completion["choices"][0]["message"]["content"].strip()
        response_message = response_message.replace("ChatGPT", "MiManubot").replace("OpenAI", "MiManubot")

        # Final fallback if response is vague
        if len(response_message.split()) < 3 or "I don't know" in response_message.lower():
            response_message = FALLBACK_RESPONSE

        return jsonify({"response": response_message})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
