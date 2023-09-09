import os
from flask import Flask, request, jsonify

app = Flask(__name__)

import pinecone
from sentence_transformers import SentenceTransformer

print('loading model')
model = SentenceTransformer("./all-MiniLM-L6-v2")


pinecone_api_key = os.environ.get('PINECONE_API_KEY')

pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
index = pinecone.Index("icon-generator")

@app.route('/')
def hello():
    return 'hello'

@app.route('/ingest')
def hello_world():
    prompt = request.args.get("prompt")
    image_id = request.args.get("image_id")

    print(prompt, image_id)

    embedding = model.encode(prompt)

    resp = index.upsert(vectors=[{"id": image_id, "values": embedding}])

    return jsonify({"respone": resp})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
