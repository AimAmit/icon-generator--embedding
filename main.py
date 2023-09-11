import os
from flask import Flask, request, jsonify

app = Flask(__name__)

import pinecone
from sentence_transformers import SentenceTransformer

print("loading model")
model = SentenceTransformer("all-MiniLM-L6-v2")


pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
index = pinecone.Index("icon-generator")


@app.route("/")
def hello():
    return "hello"


@app.route("/ingest")
def hello_world():
    prompt = request.args.get("prompt")
    image_id = request.args.get("image_id")

    print(prompt, image_id)

    embedding = model.encode(prompt)
    resp = index.upsert(
        vectors=[
            {
                "id": image_id,
                "values": embedding,
                "metadata": {"model": "Icon", "id_type": "thumbnail"},
            }
        ]
    )

    return jsonify({"respone": 'success'})

@app.route('/query')
def query():
    search = request.args.get("search")
    if not search:
        return []
    embedding = model.encode(search)

    query_response = index.query(
    top_k=50,
    # include_values=True,
    include_metadata=True,
    vector=embedding.tolist(),
    )
    res = query_response.to_dict()
    thumbnail_ids = list(map(lambda x: x['id'], res['matches']))
    return thumbnail_ids

@app.route('/remove')
def remove():
    image_id = request.args.get("image_id")
    if not image_id:
        return []
    index.delete(ids=[image_id])
    return jsonify({"respone": 'success'})



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
