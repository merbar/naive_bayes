from classifier import GNB
import numpy as np
import json

def main():
	# TRAINING
	gnb = GNB()
	with open('train.json', 'r') as f:
   		j = json.load(f)
	#print(j.keys())
	'''
   	X - array of N observations
		  - Each observation is a tuple with 4 values: s, d, 
		    s_dot and d_dot.
	'''
	X = j['states']
	X = np.array(X)
	Y = j['labels']
	Y = np.array(Y)
	# d-values seem biased where 0 is the center of the left lane.
	X[:,1] = X[:,1] + 2

	# just feed delta d to classifier
	#X = np.array([X[:,3]]).T
	# feed d and delta_d into classifier
	X = np.vstack((X[:,2], X[:,3])).T
	gnb.train(X, Y)

	# CLASSIFICATION
	with open('test.json', 'r') as f:
		j = json.load(f)
	X = j['states']
	X = np.array(X)
	Y = j['labels']
	Y = np.array(Y)
	X[:,1] = X[:,1] + 2
	#X = np.array([X[:,3]]).T
	X = np.vstack((X[:,2], X[:,3])).T
	score = 0
	for coords, label in zip(X,Y):
		predicted = gnb.predict(coords)
		if predicted == label:
			score += 1
	fraction_correct = float(score) / len(X)
	print("You got {} percent correct".format(100 * fraction_correct))


if __name__ == "__main__":
	main()