Style Search using Pytorch and Elastisearch
====

Find similar style of paintings by using vector search in Elasticsearch.
The style vecor of a painting is calculated from gram matrix of feature map in deep learning using Pytorch.

## Code

* example.py is the simple python code to create an index and insert dummy data and search data in elasticsearch. This file is for understanding the elastisearch in python.

* example_image.py is the python code to create an index and insert images' content and style vector and search data in elasticsearch. 

## Art data
Please download the data from [kaggle](https://www.kaggle.com/ikarus777/best-artworks-of-all-time/data)

After downloading please copy "resized" folder into data/resized

## Getting Started

```
cd example
pip install -r requirements.txt
python exmple_index.json
```

## Check the data via Kibana

```
docker-compose up
```
Go to http://localhost:5601/


