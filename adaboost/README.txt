Adaboost
Copyright Nathan Briese 2020

- Summary of what this program does:
The goal of this project is to write python scripts that can classify data using the Adaboost boosting algorithm.

- Describe Adaboost:
Adaboost is THE boosting learning algorithm. It was invented purely through theory, but works really well in practice.
Adaboost is a process which iteratively adds weak learners to an ensemble.
To make a prediction, each learner makes a vote for the label and the label with the majority is choosen.
Each learner is weighted with a distribution; the distribution is modified for each iteration based on previous results.
To me this is a really interesting algorithm because it can take very poor predictors (i.e. not much better than random guessing) and combine them to make a highly accurate ensemble.
Plus it's fast!

- How do I plan to expand this program in the future?
Significantly more documentation for this project is in the works. However, I wanted to get a preliminary draft published in the meatime.
I would like to find a more interesting data set to use for this problem.
I would like to learn how to save the model to a file

- How to run the program:
In addition to python3, you will need numpy, pandas, matplotlib, and scikit-learn libraries installed.
Run using "python3 adaboost.py"