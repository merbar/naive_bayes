import numpy as np
import math

class GNB(object):

	def __init__(self):
		self.possible_labels = ['left', 'keep', 'right']
		# holds mean and std_dev for each class
		self.class_summaries = {}

	def calculateProbability(self, x, mean, stdev):
		exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
		return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

	def train(self, data, labels):
		"""
		Trains the classifier with N data points and labels.

		INPUTS
		data - array of N observations
		  - Each observation is a tuple with x values

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
		"""
		data_cols_i = data.shape[-1]
		for i in range(len(self.possible_labels)):
			self.class_summaries[i] = []
			for col in range(data_cols_i):
				col_data = data[:,col]
				label = self.possible_labels[i]
				indeces = [x for x in range(len(labels)) if labels[x]==label]
				mean = np.mean(col_data[indeces])
				std_dev = np.std(col_data[indeces])
				self.class_summaries[i].append((mean, std_dev))

	def predict(self, observation):
		"""
		Once trained, this method is called and expected to return 
		a predicted behavior for the given observation.

		INPUTS

		observation - a tuple of size x, same as used for training

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
		obs_cols_i = len(observation)
		# get probability for each label
		pred = [0.0 for i in range(len(self.possible_labels))]
		for obs_i in range(obs_cols_i):
			for label_i in range(len(self.possible_labels)):
				# sum up probabilities for each data point
				cur_observation = observation[obs_i]
				mean = self.class_summaries[label_i][obs_i][0]
				std_dev = self.class_summaries[label_i][obs_i][1]
				prob = self.calculateProbability(cur_observation, mean, std_dev)
				pred[label_i] += prob

		# get max probability
		max_v = max(pred)
		max_i = pred.index(max_v)
		return self.possible_labels[max_i]