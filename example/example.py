from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pprint import pprint
import logging
format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    format=format_str,
    level=logging.DEBUG
)
index_name = "image"
index_file = "index_example.json"
client = Elasticsearch()

# delete index
client.indices.delete(index=index_name, ignore=[404])

# create index
with open(index_file) as index_file:
    source = index_file.read().strip()
    client.indices.create(index=index_name, body=source)


def create_document(i):
    return {
        '_op_type': 'index',
        '_index': index_name,
        'text': str(i),
        'title': str(i),
        'text_vector': [-100.0, -200.0, 3.0*i]
    }


# insert data
docs = [create_document(i) for i in range(5)]
bulk(client, docs)

client.indices.refresh(index=index_name)

# search data
query_vector = [1, 2, 0]
script_query = {
    "script_score": {
        "query": {"match_all": {}},
        "script": {
            "source": "cosineSimilarity(params.query_vector, doc['text_vector']) + 1.0",
            "params": {"query_vector": query_vector}
        }
    }
}

response = client.search(index=index_name, body={"query": {"match_all": {}}})
pprint(response)

response = client.search(
    index=index_name,
    body={
        "size": 3,
        "query": script_query,
        "_source": {"includes": ["title", "text"]}
    }
)
pprint(response)
