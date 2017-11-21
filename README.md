# Naive Bayesian Classifiers

Naive Bayesian classifiers are one of the most commonly used forms of computational intelligence.
Bayesian inference can be used to build a continuously improving expert system that has the power
to classify inputs into a variety of categories. In the general sense, Bayesian classifiers (and probabilistic
classifiers in general) can be used to determine the likelihood that any one event causes another event, under the assumption the the two events are somewhat dependent. Naive Bayesian Classifiers are special in that they assume 
that no other variables affect you likelihood that your input is related to your categorical output, hence the 
"naive".

## Pre-requisities
1.	If you haven't already install scikit-learn, pandas, and all of their depedencies.
2.	Clone this repo (and star it, and follow me on Git if you aren't already)
3.  Download a set of Amazon reviews that have been preformatted into JSON from [here](http://jmcauley.ucsd.edu/data/amazon/). One of the 5-cores should do just fine.

## Running Instructions
To run the script in most environments, just type `python3 nbc.py [path/to/5.json.gz]` where that last bit is the location of the amazon review dataset on your computer. By default, the classifier is set up to split the set into two pieces. One piece is used to train the classifier, and the other piece is use to test the classifier. It then prints out the overall accuracy of it's predictions for the test set as a value from [0,1]. 
