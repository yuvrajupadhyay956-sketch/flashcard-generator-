from flask import Flask, request, jsonify, render_template
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN"),
    provider="auto"
)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_flashcards():

    data = request.json
    text = data["text"]
    count = data.get("count", 5)

    prompt = f"""
Create {count} simple flashcards from the following study material.

Study material:
{text}

Give the output in this format:

Q1: Question
A1: Answer

Q2: Question
A2: Answer

Q3: Question
A3: Answer

Q4: Question
A4: Answer

Q5: Question
A5: Answer
"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500
    )

    result = response.choices[0].message.content

    return jsonify({"flashcards": result})


if __name__ == "__main__":
    app.run(debug=True)