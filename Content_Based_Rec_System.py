import json
import re
import numpy as np
import math
from pyspark import SparkContext
from collections import Counter

#trial
from datetime import datetime
start = datetime.now()
sc = SparkContext()

user_id = 50

# reading user vectors
import pickle
filename = "item_user_rating.json"
with open(filename, 'rb') as fp:
    r = pickle.load(fp)
user_item_rating = sc.parallelize(r).filter(lambda x: x[0][0] == user_id)

# storing items of user_id in broadcast variable
user_id_items = sc.broadcast(set(user_item_rating.map(lambda x: x[0][1]).collect()))

# reading item vectos
filename = "dummy_data_v2.json"
def convert(item_vector):

    item_id = item_vector["ItemID"]
    features = []
    
    for feature, value in item_vector.items():
        if feature == "ItemID":
            continue
        features.append((feature, value))     
    return (item_id, features)

rdd = sc.textFile(filename).map(lambda x: json.loads(x)).map(lambda x: convert(x))

# normalize item vectors (need to change according to the value of features)
def normalize_vector(v):
#     item = v[0]
#     features = v[1]    
#     mean_rating = sum(map(lambda x: x[1], features)) / len(features)
#     normalized_vector = list(map(lambda x: (x[0], x[1] - mean_rating), features))
    
    item = v[0]
    features = v[1]    
    mean_rating = sum(map(lambda x: x[1], features)) / len(features)
    normalized_vector = list(map(lambda x: (x[0], x[1] / mean_rating), features))
    
    return (item, normalized_vector)
normalized_rdd = rdd.map(lambda x: normalize_vector(x)).flatMapValues(list)
# o/p -> (item, (feature, rating))

item_user_rating = user_item_rating.map(lambda x: (x[0][1], (x[0][0], x[1])))
# o/p -> (item, (user, rating))

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

print(top10rec)
print("Time taken: ", datetime.now() - start)



