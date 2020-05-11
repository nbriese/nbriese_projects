Support Vector Machine
Copyright Nathan Briese 2020

- Summary of what this program does:
The goal of this project is to write python scripts that can classify data using support vector machines.
Almost all the relavant parts of the code (with the exception of the convex optimization) has been written from scratch (meaning, I did not just use a library for the learning).

- SVM:
SVM is one of the classic examples of classification algorithms.
The descision boundary is decided by simaltaniously maximizing the distance between the boundary and the nearest points and minimizing the misclassification error. This is a dual valued, convex optimization problem. This will be discussed in further detail in later versions of this documentation.

- Kernel SVM:
The key to kernel SVM is that the descision boundary is no longer linear. The boundry can be parameterized by any feature expansion, but for this example I ave choosen to use the Gaussian kernel. Surprisingly, the underlying optimization problem remains the same! One needs to apply a kernel transformation to the data and then find the "linear" descision boundary in the new kernel space. More detail (and actual maths) to come.

- How do I plan to expand this program in the future?
Significantly more documentation for this project is in the works. However, I wanted to get a preliminary draft published in the meatime.
I would like to find a better / more interesting data set to use for this problem. Although it is interesting to see the difference in performance between linear and kernel SVM on non-linearly separable data.

- How to compile and run the program:
In addition to python3, you will need numpy, Pandas, matplotlib, and cxvopt libraries installed.
Run "python3 kernel_svm.py" or "python3 linear_svm.py"
