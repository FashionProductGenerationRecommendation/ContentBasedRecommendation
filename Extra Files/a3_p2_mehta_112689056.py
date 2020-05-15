import json
import re
import numpy as np
import math
from pyspark import SparkContext
from collections import Counter
import sys

# filtering the JSON
def filter_json(line):
    keys = ["overall", "reviewerID", "asin", "reviewText", "summary", "verified", "unixReviewTime"]
    for key in keys:
        if key not in line:
            return False
    return True

# map for latest review
def map_utility(x):
    rating = x["overall"]
    user = x["reviewerID"]
    item = x["asin"]
    time = x["unixReviewTime"]

    return ((item, user), (rating, time))

# reduce for latest review
def reduce_utility(x, y):
    rating1, time1 = x
    rating2, time2 = y

    if time1 > time2:
        return (rating1, time1)

    return (rating2, time2)    

# normalize the input vector
def normalize_vector(v):
    item = v[0]
    users = v[1]    
    mean_rating = sum(map(lambda x: x[1], users)) / len(users)
    normalized_vector = list(map(lambda x: (x[0], x[1] - mean_rating), users))
    return (item, normalized_vector)

# mod of normalized vector
def get_mod(x):
    item = x[0]
    users = x[1]
    mod = math.sqrt(sum(map(lambda x: x[1]**2, users)))
    normalized_vector = list(map(lambda x: (x[0], x[1], mod), users))
    return (item, normalized_vector)

# generate recommend rows
def get_recommended_rows(user_vector):
    user = user_vector[0]
    items = user_vector[1]
    item_dict = {item[0] : item[1] for item in items}
    
    result = []
    for vec in products_to_recommend.value:
        vec_item = vec

        if vec_item in item_dict:
            result.append(((vec_item, user), item_dict[vec_item]))
            continue

        sum_sim_uti = 0
        sum_sim = 0
        count = 0
        for user_item in items:
            user_item_id = user_item[0]
            rating = user_item[1]

            if (vec_item, user_item_id) in sim_matrix.value:
                sum_sim_uti += sim_matrix.value[(vec_item, user_item_id)] * rating
                sum_sim +=  sim_matrix.value[(vec_item, user_item_id)]
                count += 1 
        
        if count >= 2:
            uti = sum_sim_uti / sum_sim
            result.append(((vec_item, user), uti))

    return result

# map for creating similarity tuples
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

# reduce for creating similarity tuples
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
    
    if count >= 2 and sim > 0:
        return (item1, (item2, sim))
    else:
        return None

# start = datetime.now()
sc = SparkContext()

filename = sys.argv[1]

rdd = sc.textFile(filename).map(lambda x: json.loads(x)).filter(lambda x: filter_json(x))

word = sys.argv[2]
word = word.strip("[]").split(',')

word = list(map(lambda x: x.strip("\"\' "), word))
products_to_recommend = sc.broadcast(word)

# Output -> (item , [(user, rating),   ,   ,])
utility_matrix = rdd.map(lambda x: map_utility(x)).reduceByKey(lambda x,y: reduce_utility(x, y))\
                    .map(lambda x: (x[0][0], (x[0][1], x[1][0]))).groupByKey().filter(lambda x: len(x[1]) >= 25)\
                    .flatMapValues(lambda x: x)\
                    .map(lambda x: (x[1][0], (x[0], x[1][1]))).groupByKey().filter(lambda x: len(x[1]) >= 5)

# persist the user vectors since I have already done a groupBy(expensive operation) once which is needed ahead
persisted_users = utility_matrix
persisted_users.persist()

# item vectors
utility_matrix = utility_matrix.flatMapValues(lambda x: x)\
                                .map(lambda x: (x[1][0], (x[0], x[1][1]))).groupByKey()

# normalizing item vectors, calculating the mod and skipping the ones with mod 0 
normalized_utility_matrix_flat = utility_matrix.map(lambda x: normalize_vector(x)).map(lambda x: get_mod(x))\
                                                .flatMapValues(lambda x: x).filter(lambda x: x[1][2] != 0)

# vectors of to_recommend products
recommend_vectors_rdd = normalized_utility_matrix_flat.filter(lambda x: x[0] in products_to_recommend.value)



user_vectors_recommend = recommend_vectors_rdd.map(lambda x: (x[1][0], (x[0], x[1][1], x[1][2])))
user_vectors_all = normalized_utility_matrix_flat.map(lambda x: (x[1][0], (x[0], x[1][1], x[1][2])))
joined_user_vectors = user_vectors_recommend.join(user_vectors_all).filter(lambda x: x[1][0][0] != x[1][1][0]).map(lambda x: create_sim_tuples(x))

    
# storing the similarity matrix globally
sim_matrix = joined_user_vectors.reduceByKey(lambda x,y: reduce_sim(x,y)).map(lambda x: calculate_sim(x))\
                .filter(lambda x: x != None).map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x + y)\
                .map(lambda x: (x[0], sorted(list(x[1]), key = lambda x: x[1], reverse=True)[:50]))\
                .flatMapValues(lambda x: x).map(lambda x: ((x[0],x[1][0]), x[1][1])).collectAsMap()

sim_matrix = sc.broadcast(sim_matrix)

# from the user vectors, find the rows of to_recommend products
recommended_rows = persisted_users.flatMap(lambda x: get_recommended_rows(x))\
                                    .map(lambda x: (x[0][0], (x[0][1], round(x[1], 3))))\
                                    .groupByKey().collect()
print()
for item, vector in recommended_rows:
    print("Product Name: ", item)
    print("Product Vector", list(vector))
    print()
