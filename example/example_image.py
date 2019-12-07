from image_helper import ImageHelper
from image_helper import gram_matrix
from image_helper import get_upper_triangle_values
import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pprint import pprint
from sklearn.decomposition import PCA
import numpy as np
import logging
from random import sample
from tqdm import tqdm
format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    format=format_str,
    level=logging.INFO
)

index_name = "image"
index_file = "index.json"
client = Elasticsearch()

# delete index
client.indices.delete(index=index_name, ignore=[404])

# create index
with open(index_file) as index_file:
    source = index_file.read().strip()
    client.indices.create(index=index_name, body=source)


def create_document(title, content_vector, style_vector):
    return {
        '_op_type': 'index',
        '_index': index_name,
        'title': title,
        'content_vector': content_vector,
        'style_vector': style_vector
    }


image_helper = ImageHelper()
trained_model = image_helper.get_model(None)  # for content vector
trained_model_conv4 = image_helper.get_model('conv_4')  # for style vector

# calculate content and style vector
N = 200  # number of images
folder_path = "../data/resized/"
images = []
titles = []
content_vectors = []
style_vectors = []
for image_name in tqdm(sample(os.listdir(folder_path), N)):
    image_path = folder_path + image_name
    image = image_helper.image_loader(image_path)
    content_vector = trained_model(image).view(-1).detach().numpy()
    content_vectors.append(content_vector)
    style_vector = get_upper_triangle_values(
        gram_matrix(trained_model_conv4(image))).detach().numpy()
    style_vectors.append(style_vector)
    titles.append(image_name)
    images.append(image)


def reduce_dimension(vectors):
    pca = PCA(n_components=100)
    pca.fit(vectors)
    return pca.transform(vectors)


# reduce dimension by PCA
content_vectors = np.array(content_vectors)
style_vectors = np.array(style_vectors)
reduced_content_vectors = reduce_dimension(content_vectors)
reduced_style_vectors = reduce_dimension(style_vectors)

documents = []
for i, title in enumerate(titles):
    document = create_document(
        title, reduced_content_vectors[i, :].tolist(), reduced_style_vectors[i, :].tolist())
    documents.append(document)


# insert data
bulk(client, documents)
client.indices.refresh(index=index_name)


def search(vector_name, vector, topn=3):
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc['{}']) + 1.0".format(vector_name),
                "params": {"query_vector": vector}
            }
        }
    }
    response = client.search(
        index=index_name,
        body={
            "size": topn,
            "query": script_query,
            "_source": {"includes": ["title"]}
        }
    )
    return response


i = 0
topn = 5
query_vector = reduced_style_vectors[i, :].tolist()
pprint(search("style_vector", query_vector, topn))
query_vector = reduced_content_vectors[i, :].tolist()
pprint(search("content_vector", query_vector, topn))
