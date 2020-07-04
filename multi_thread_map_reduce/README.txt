Multi Threaded Map Reduce
Copyright Nathan Briese 2020

- Summary of what this program does:
This program will read through the words in a txt file and perform word count statistics on it. This means, for each letter, it will count the number of words in the text file which start with that letter. The result is put in the file "result.txt".
The task is completed using the map reduce paradigm.
The mapping is parallelized through the use of threads. There is a thread which reads the text file and feeds packets of words to the map phase threads which read the packets and count the starting letters and send their partial results to the reduce thread (the parent thread).

- Describe the Map-Reduce algorithm:
Map-Reduce is a classic data aggregation algorithm. It consists of two aptly named phases: mapping and reducing.
Mapping is the first step in which a subsection of the data is processed. This step is easily parallelizable, and in this program it is parallelized through the use of threads. Reduce takes the many outputs of the mappers and aggregates it into a single result.

- Describe what threads are and how I used them:
Threads are essentially 'light weight processes'. They are meant to complete small, mostly independent tasks in parallel.
The issue that can arise when using threads is collisions. When multiple threads try to access (or worse modify) the same data then they will collide. The results are unpredictable.
To prevent this I used semaphores. Named after devices used to control railroad traffic, semaphores limit the number of threads that can continue running at a time.
A 'critical section' of code is typically where a thread is accessing or modifying a shared resource. When a thread arrives at a critical section, it will check if the semaphore is open or closed.
If the semaphore is open, then it will enter the critical section and close the semaphore. If the semaphore is closed, the thread will enter a queue and wait for its turn for the semaphore to be open.
When the thread leaves the critical section, it will open the semaphore so the other threads can have a turn with the shared resource.
The actions of checking, opening, and closing the semaphore are atomized, meaning the thread cannot be interrupted or become inactive while doing so (unless the thread is waiting inactive in the queue).

- How do I plan to expand this program in the future?
I think this program should have more descriptive or useful error messages. I could use system flags perhaps.
Additionally, I could change it to be more general. Instead of just doing word counts directly, the mappers could be organized to call a given function on each line of text. The function could be changed for a different task.
This reduces coupling between the consumers and the action they are meant to complete.

- How to compile and run the program:
Use the included makefile with the command "make" to compile.
Then the program can be run using: ./mtmr [filename] [num_consumers]
Where:
	filename (optional, default "./testing/test2.txt") is the relative directory of the .txt file to open
	num_consumers (optional, default 4) is the number of consumer threads
