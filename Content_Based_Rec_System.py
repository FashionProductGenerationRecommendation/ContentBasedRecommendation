"""
Team Name: KRAB
Team Members: Annuj Jain, Bharat Goel, Keshav Aditya Rajupet Premkumar, Rutvij Mehta

General Description of Code:
The code takes input as user id, item id and rating. The then uses the concept of 
Content Based Recommendation system to give the top 10 recommended products 
based on the user's previous rating. The user vectors dataset is generated randomly 
and saved to a file which can be used.

Data Frameworks: HDFS, Apache Spark
Concept: Content Based Recommendation System

Type of System: Ubuntu 16.04 / Google Cloud DataProc running Debian 1.4

"""

import json
import re
import numpy as np
import math
from pyspark import SparkContext
from collections import Counter
import pickle
from datetime import datetime

start = datetime.now()

sc = SparkContext()
data_filename = "data_ready_v1.json"
item_user_filename = "item_user_rating_v2.json"

def convert(item_vector):

    item_id = item_vector["ItemID"]
    features = []
    
    for feature, value in item_vector.items():
        if feature == "ItemID":
            continue
        features.append((feature, value))     
    return (item_id, features)

# normalize item vectors (need to change according to the value of features)
def normalize_vector(v):
    item = v[0]
    features = v[1]    
    mean_rating = sum(map(lambda x: x[1], features)) / len(features)
    normalized_vector = list(map(lambda x: (x[0], x[1] - mean_rating), features))
    
    return (item, normalized_vector)

def content_based_recommendation(user_id, item_id, rating):

    input_rdd = sc.parallelize([((user_id, item_id), rating)])

    # reading user vectors
    with open(item_user_filename, 'rb') as fp:
        r = pickle.load(fp)
    user_item_rating = sc.parallelize(r).filter(lambda x: x[0][0] == user_id)
    user_item_rating = user_item_rating.union(input_rdd)

    # storing items of user_id in broadcast variable
    user_id_items = sc.broadcast(set(user_item_rating.map(lambda x: x[0][1]).collect()))

    item_user_rating = user_item_rating.map(lambda x: (x[0][1], (x[0][0], x[1])))
    # o/p -> (item, (user, rating))

    # reading item vectos
    rdd = sc.textFile(data_filename).map(lambda x: json.loads(x)).map(lambda x: convert(x))

    normalized_rdd = rdd.map(lambda x: normalize_vector(x)).flatMapValues(list)
    # o/p -> (item, (feature, rating))

    joined_rdd = normalized_rdd.join(item_user_rating)
    # o/p -> (item, ((feature, value), (user, rating)))

    joined_rdd = joined_rdd.map(lambda x: (x[1][0][0], (x[0], x[1][1][0], x[1][0][1] * x[1][1][1])))\
                            .reduceByKey(lambda x, y: (x[0], x[1], x[2] * y[2]))\
                            .map(lambda x: (x[0], (x[1][1], x[1][2])))
    # o/p -> (feature, (user_id, value))

    item_ratings = normalized_rdd.filter(lambda x: x[0] not in user_id_items.value)\
                                    .map(lambda x: (x[1][0], (x[0], x[1][1])))
    # o/p -> (feature, (item, value))
                        
    top10rec = item_ratings.join(joined_rdd)\
                            .map(lambda x: ((x[1][1][0], x[1][0][0]), (x[1][0][1] * x[1][1][1])))\
                            .reduceByKey(lambda x,y: x + y)\
                            .top(10, lambda x: x[1])
    # o/p -> ((user_id, item), value)

    top10rec = list(map(lambda x: x[0][1], top10rec))

    # print(top10rec)
    return top10rec
    
print(content_based_recommendation(50, 23832, 4))
print("Time taken: ", datetime.now() - start)
