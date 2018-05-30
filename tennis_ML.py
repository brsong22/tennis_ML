# -*- coding: utf-8 -*-
"""
Created on Tue May 29 12:35:48 2018

@author: 22Ninjas
"""
import requests
import numpy
import io
url = "https://raw.githubusercontent.com/brsong22/tennis_atp/master/atp_matches_1991.csv"

#get the data from csv file at url
atp_data = numpy.genfromtxt(io.BytesIO(requests.get(url).content), delimiter=',', dtype=None, encoding=None) #shape (3728, 49)

#create a dict of categories (columns) in the data
stat_cats = {category: c_index for c_index, category in enumerate(atp_data[0])} #(49 columns)
atp_data = atp_data[1:]
# =============================================================================
# inputs: best_of, loser_age, winner_age, loser_rank, winner_rank,
# best_of: number of sets played will effect how long the match takes
# loser_age/winner_age: younger players faster?
# loser_rank/winner_rank: rank disparity causes quicker matches?
# =============================================================================

feature_cols = numpy.array([stat_cats['best_of'], stat_cats['loser_age'], stat_cats['winner_age'], stat_cats['loser_rank'], stat_cats['winner_rank']])
#clean up atp_data; remove rows where minutes or any feature is unpopulated
atp_data = numpy.delete(atp_data, numpy.where(atp_data[:, stat_cats['minutes']] == '')[0], axis=0)
atp_data = numpy.delete(atp_data, numpy.where(atp_data[:, feature_cols] == '')[0], axis=0)
features = atp_data[:, feature_cols].astype(numpy.float64) #shape (3062, 5)
print(features.shape)
targets = atp_data[:, stat_cats['minutes']].astype(numpy.float64) #shape (3237, 1)
print(targets.shape)
weights = numpy.random.uniform(-.1, .1, features.shape[1]) #shape (5, 1)
print(weights.shape)
biases = numpy.random.uniform(-.1, .1, targets.shape[0]) #shape (3237, 1)
print(biases.shape)

y = numpy.dot(features, weights) + biases
print(y[0])
print(targets[0])
#learning rate alpha
alpha = 0.05


