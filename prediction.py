#!/usr/bin/env python

from classifier import GNB
import json

def main():
	gnb = GNB()
	with open('train.json', 'r') as f:
   		j = json.load(f)
	print(j.keys())
	X = j['states']
	X = np.array(X)
	Y = j['labels']
	Y = np.array(Y)
	# d-values seem biased where 0 is the center of the left lane.
	X[:,1] = X[:,1] + 2

	gnb.train(X, Y)

	with open('test.json', 'rb') as f:
		j = json.load(f)

	X = j['states']
	X = np.array(X)
	Y = j['labels']
	Y = np.array(Y)
	X[:,1] = X[:,1] + 2
	score = 0
	for coords, label in zip(X,Y):
		predicted = gnb.predict(coords)
		if predicted == label:
			score += 1
	fraction_correct = float(score) / len(X)
	print("You got {} percent correct".format(100 * fraction_correct))


if __name__ == "__main__":
	main()