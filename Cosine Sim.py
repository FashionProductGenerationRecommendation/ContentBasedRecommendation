"""
Team Name: KRAB
Team Members: Annuj Jain, Bharat Goel, Keshav Aditya Rajupet Premkumar, Rutvij Mehta

General Description of Code:
The code takes the input as an item vector. It compares its similarities with all the
other item vectors and returns the top 10 most similar items.

Data Frameworks: HDFS, Apache Spark
Concept: Cosine Similarity

Type of System: Ubuntu 16.04 / Google Cloud DataProc running Debian 1.4

"""

import json
import re
import numpy as np
import math
from pyspark import SparkContext
from collections import Counter
from datetime import datetime

sc = SparkContext()
# filename = "dummy_data_v2.json"
filename = "data_ready_v1.json"

# convert item row to (item_id, (features))
def convert(item_vector):

    item_id = item_vector["ItemID"]
    features = []
    
    for feature, value in item_vector.items():
        if feature == "ItemID":
            continue
        features.append((feature, value))
        
    return (item_id, features)

# normalize the vector
def normalize_vector(v):
    item = v[0]
    users = v[1]    
    mean_rating = sum(map(lambda x: x[1], users)) / len(users)
    normalized_vector = list(map(lambda x: (x[0], x[1] - mean_rating), users))
    return (item, normalized_vector)

# calculate mod of the vector
def get_mod(x):
    item = x[0]
    users = x[1]
    mod = math.sqrt(sum(map(lambda x: x[1]**2, users)))
    normalized_vector = list(map(lambda x: (x[0], x[1], mod), users))
    return (item, normalized_vector)

# calculating similarity mapper
def create_sim_tuples(x):
    item1_vec = x[1][0]
    item2_vec = x[1][1]
    
    item1 = item1_vec[0]
    item2 = item2_vec[0]
    
    item1_val = item1_vec[1]
    item2_val = item2_vec[1]
    
    item1_mod = item1_vec[2]
    item2_mod = item2_vec[2]
    
    product = item1_val * item2_val
    
    return ((item1, item2), (product, item1_mod, item2_mod, 1, 1))

# calculating similarity reducer
def reduce_sim(x,y):
    sim_productx, mod1, mod2, countx, simx = x
    sim_producty, mod1, mod2, county, simy = y

    sim_product = sim_productx + sim_producty
    count = countx + county
    sim = 1
    
    return (sim_product, mod1, mod2, count, sim)

# final map after map-reduce to calculate similarity
def calculate_sim(x):
    item1, item2 = x[0]
    sim_product, mod1, mod2, count, sim = x[1]
    
    sim = sim_product / (mod1 * mod2) 
    return (item1, (item2, sim))

start = datetime.now()
def cosine_similarity(recommend_id):

    # read the data and convert it to format -> (item, (features))    
    rdd = sc.textFile(filename).map(lambda x: json.loads(x)).map(lambda x: convert(x))

    # get normalized vectors
    normalized_rdd = rdd.map(lambda x: normalize_vector(x))

    # get vectors along with mod
    mod_rdd = normalized_rdd.map(lambda x: get_mod(x)).flatMapValues(lambda x: x).filter(lambda x: x[1][2] != 0)

    # storing to broadcast variable
    recommend_id = sc.broadcast(recommend_id)

    # recommend vector and item vectors
    rec_vec = mod_rdd.filter(lambda x: x[0] == recommend_id.value).map(lambda x: (x[1][0], (x[0], x[1][1], x[1][2])))
    item_vecs = mod_rdd.filter(lambda x: x != recommend_id.value).map(lambda x: (x[1][0], (x[0], x[1][1], x[1][2])))    

    # joined vectors from rec_vec and item_vecs for calculating similarity
    joined_vectors = rec_vec.join(item_vecs).filter(lambda x: x[1][0][0] != x[1][1][0]).map(lambda x: create_sim_tuples(x))

    # top 10 recommendations after calculating similarity
    top10rec = joined_vectors.reduceByKey(lambda x,y: reduce_sim(x,y)).map(lambda x: calculate_sim(x)).top(10, lambda x: x[1][1])
    top10rec = list(map(lambda x: x[1][0], top10rec))

    return top10rec

# print(cosine_similarity("11"))
# print(cosine_similarity(23832))

# print("Time taken: ", datetime.now() - start)
