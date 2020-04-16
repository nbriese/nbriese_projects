Multi Threaded Map Reduce
Copyright Nathan Briese 2020

- Summary of what this program does:
This program will read through the words in a txt file and perform word count statistics on it. This means, for each letter, it will count the number of words in the text file which start with that letter. The result is put in the file "result.txt".
The task is completed using the map reduce algorithm.
The mapping is parellelized through the use of threads. There is a producer (reduce phase) thread which reads the text file and feeds packets of words to the consumers (map phase) which read the packets and count the starting letters and send their partial results to the parent thread.

- Describe the Map-Reduce algorithm:
Map-Reduce is a classic data aggregation algorithm. It consists of two aptly named phases: mapping and reducing.
Mapping is the first step in which a subsection of the data is processed. This step is easily parrallelizable, and in this program it is parrallelized through the use of threads. Reduce takes the many outputs of the mappers and aggregates it into a single result.

- Describe what threads are and how I used them:
Threads are essentially 'light weight processes'. They are meant to complete small tasks that are mostly independent. The issue that can arrise when using threads is collisions. When more than one thread tries to access (or worse modify) the same data then they will collide. The results are unpredictable and not wanted. To fix this I used semaphores. Named after devices used to control railroad traffic, semaphores limit the number of threads that can continue running at a time.

- How do I plan to expand this program in the future?
I think this program should have more descriptive or useful error messages. I should use system flags perhaps.
Additionally, I could change it to be more general. Instead of just doing word counts directly, the mappers could be organized to call a given function on each line of text. The function could be changed for a different task. This reduces coupleing between the consumers and the action they are meant to complete.
Also, I should add more explination about semaphores and atomized actions in this readme.

- How to compile and run the program:
Use the included makefile with the command "make" to compile.
Then the program can be run using the syntax: $ ./mtmr [filename] [num_consumers]
Where:
	filename (optional, default "./testing/test2.txt") is the relative directory of the .txt file to open
	num_consumers (optional, default 4) is the number of consumer threads
