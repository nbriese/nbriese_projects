Client Server Map Reduce
Nathan Briese

- Summary of what this program does:
This program will read through the words in a txt file and perform word count statistics on it. This means, for each letter, it will count the number of words in the text file which start with that letter.
The result is put in the file "result.txt".
The task is completed using the map reduce paradigm. The mapping is parallelized through the use of threads.

- How do I plan to expand this program in the future?
Make the folder name argument optional and have a default value
Additionally, I could change it to be more general. Instead of just doing word counts directly, the mappers could be organized to call a given function on each line of text. The function could be changed for a different task.
This reduces coupling between the consumers and the action they are meant to complete.

- How to compile and run the program:
compile using "make clean; make -B update"

run the server program first using: ./server <Server Port>

then run the client program using: ./client <Folder Name> <# of Mappers> <Server IP> <Server Port>
● <Folder Name> the name of the root folder to be traversed
● <# of Mappers> the number of mapper processes
● <Server IP> IP address of the server to connect to
● <Server Port> port number of the server to connect to
