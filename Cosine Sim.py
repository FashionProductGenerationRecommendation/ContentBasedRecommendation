#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import findspark
# findspark.init()

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

filename = "dummy_data_v2.json"
rdd = sc.textFile(filename).map(lambda x: json.loads(x))

# item_id, [(feature, value)]
def convert(item_vector):

    item_id = str(item_vector["ItemID"])
    features = []
    
    for feature, value in item_vector.items():
        if feature == "ItemID":
            continue
        features.append((feature, value))
        
    return (item_id, features)


rdd = rdd.map(lambda x: convert(x))

def normalize_vector(v):
    item = v[0]
    users = v[1]    
    mean_rating = sum(map(lambda x: x[1], users)) / len(users)
    normalized_vector = list(map(lambda x: (x[0], x[1] - mean_rating), users))
    return (item, normalized_vector)
normalized_rdd = rdd.map(lambda x: normalize_vector(x))

def get_mod(x):
    item = x[0]
    users = x[1]
    mod = math.sqrt(sum(map(lambda x: x[1]**2, users)))
    normalized_vector = list(map(lambda x: (x[0], x[1], mod), users))
    return (item, normalized_vector)

mod_rdd = rdd.map(lambda x: get_mod(x)).flatMapValues(lambda x: x).filter(lambda x: x[1][2] != 0)

recommend_id = "11"
recommend_id = sc.broadcast(recommend_id)
a = mod_rdd.filter(lambda x: x[0] == recommend_id.value).map(lambda x: (x[1][0], (x[0], x[1][1], x[1][2])))
b = mod_rdd.filter(lambda x: x != recommend_id.value).map(lambda x: (x[1][0], (x[0], x[1][1], x[1][2])))

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
    
c = a.join(b).filter(lambda x: x[1][0][0] != x[1][1][0]).map(lambda x: create_sim_tuples(x))

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
    # if count >= 2 and sim > 0:
    #     return (item1, (item2, sim))
    # else:
    #     return None

sim_matrix = c.reduceByKey(lambda x,y: reduce_sim(x,y)).map(lambda x: calculate_sim(x)).top(5, lambda x: x[1][1])

# sim_matrix.take(2)
print(sim_matrix)
print("Time taken: ", datetime.now() - start)
