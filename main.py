import os
from flask import Flask, request

app = Flask(__name__)

import pinecone
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("./all-MiniLM-L6-v2")


pinecone_api_key = os.environ.get('PINECONE_API_KEY')

pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
index = pinecone.Index("icon-generator")


@app.route('/ingest')
def hello_world():
    prompt = request.args.get("prompt")
    image_id = request.args.get("image_id")

    embedding = model.encode(prompt)

    resp = index.upsert(vectors=[{"id": image_id, "values": embedding}])

    return {"respone": resp}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
