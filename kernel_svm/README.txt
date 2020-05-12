Support Vector Machine
Copyright Nathan Briese 2020

- Summary of what this program does:
The goal of this project is to write python scripts that can classify data using support vector machines.
Almost all the relevant parts of the code (with the exception of the convex optimization) has been written from scratch.

- SVM:
SVM is one of the classic examples of classification algorithms.
The linear decision boundary is decided by simultaneously maximizing the distance between the boundary and the nearest points and minimizing the misclassification error.
This is a dual valued, convex optimization problem. 
Let w be the vector that makes the decision boundary.
To maximize the margin, the objective function should maximize 1/||w|| the inverse of the norm of w. This is actually the same as minimizing (1/2)||w||^2 under the condition that for all i y_i * w^T * x_i >= 1.
The minimizer w* is unique if it exists.
When the data is not perfectly linearly separable, introduce the slack variable, k, to allow for weighted errors.
The new objective function is the same plus C, the misclassification weight, times the sum of k_i for all i under the condition that y_i * w^T * x_i >= 1 - k_i for all i and k_i >= 0 for all i.
The provided python script is made to test multiple different values for the misclassification weight for hyperparameter tuning.

- Kernel SVM:
The key to kernel SVM is that the decision boundary is no longer linear.
The boundary can be parameterized by any feature expansion, but for this example I have chosen to use the Gaussian kernel.
Apply a kernel transformation to the data and then find the "linear" decision boundary in the new kernel space.
The new objective function is similar to the linear one, but now the w that defines the decision boundary has infinite dimension.
Instead the value of the slack variables, lambda_i, are used along with the kernel function of each training point to make predictions.
The script is made to test multiple values for both the misclassification weight and the variance of the Gaussian kernel.

- Choice of Dataset:
I chose to use a set of points distributed around the origin; the key is that they are labeled based on which quadrant they are in. 
This means the data is very much not linearly separable.
I made this choice so that there would be a large performance difference between linear and kernel svm.
We see roughly 50% test accuracy for the linear svm, but accuracy in the 90s for kernel svm.

- How do I plan to expand this program in the future?
I would like to be able to save and load models for reuse.
I would like to find a better / more interesting data set to use for this problem. Although it is interesting to see the difference in performance between linear and kernel SVM on non-linearly separable data.

- How to compile and run the program:
In addition to python 3, you will need numpy, pandas, matplotlib, and cxvopt libraries installed.
Run using "python3 kernel_svm.py" or "python3 linear_svm.py"
