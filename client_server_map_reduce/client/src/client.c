#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <zconf.h>
#include <arpa/inet.h>
#include <ctype.h>
#include "../include/protocol.h"
#define _BSD_SOURCE

// define the global file handle so that threads can write to the log
FILE *logfp;

int main(int argc, char *argv[]) {
  int mappers;
  char folderName[100] = {'\0'};
  char *server_ip;
  int server_port;

  // interpret arguments
  if (argc == 5) {
    strcpy(folderName, argv[1]);
    mappers = atoi(argv[2]);
    server_ip = argv[3];
    server_port = atoi(argv[4]);
    if (mappers > MAX_MAPPER_PER_MASTER) {
      printf("Maximum number of mappers is %d.\n", MAX_MAPPER_PER_MASTER);
      printf("./client <Folder Name> <# of mappers> <server IP> <server Port>\n");
      exit(1);
    }
  } else {
    printf("Usage: ./client <Folder Name> <# of mappers> <server IP> <server Port>\n");
    exit(1);
  }

  // create log file
  pid_t p = fork();
  if (p==0) execl("/bin/rm", "rm", "-rf", "./client/log", NULL);
  wait(NULL);
  mkdir("./client/log", ACCESSPERMS);
  logfp = fopen("./client/log/log_client.txt", "w");

  // partition the files to the mappers
  traverseFS(mappers, folderName);

	int mapperID = 0;
	// Master process assigns a unique mapperID to each mapper processes
	for(int i=1; i<=mappers; i++) {
		pid_t child_pid = fork();
		if(child_pid == 0){
			mapperID = i;
			break;
		}
		else if(child_pid > 0) continue;
		else return -1;
	}
	if(mapperID != 0) {
		// Mapper clients can access their own mapper file in MapperIntput folder.
		char input_path[64];
		sprintf(input_path, "./client/MapperInput/Mapper_%d.txt", mapperID);
		FILE *files = fopen(input_path,"r");

		// Each mapper client sets up a TCP connection to the server
		int sockfd = socket(AF_INET , SOCK_STREAM , 0);

		struct sockaddr_in address;
		address.sin_family = AF_INET;
		address.sin_port = htons(server_port);
		address.sin_addr.s_addr = inet_addr(server_ip);

		if (connect(sockfd, (struct sockaddr *) &address, sizeof(address)) != 0) {
			printf("Connection error for mapper %d\n",mapperID);
			return -1;
		}
		fprintf(logfp,"[%d] open connection\n", mapperID);

		int request[REQUEST_MSG_SIZE];
		int response[RESPONSE_MSG_SIZE];

// Mapper Clients Request Handling
// 1. CHECKIN
		// Mapper clients should send this request before any other requests
		request[0] = CHECKIN;
		request[1] = mapperID;
		send(sockfd, &request, REQUEST_MSG_SIZE*sizeof(int),0);
		recv(sockfd, &response, RESPONSE_MSG_SIZE*sizeof(int),MSG_WAITALL);
		fprintf(logfp,"[%d] CHECKING: %d %d\n", mapperID,response[1],response[2]);

// 2. UPDATE_AZLIST
    // Mapper clients sends to the server with PER-FILE word count results
    // If there is no files in the mapper file, mapper clients SHOULD NOT send this message
		char any_files = -1;
		while(!feof(files)) {
			any_files = 1;
			char line_buf[50] = "";
			fgets(line_buf,50,files);
			char *p = strtok(line_buf, " \n\t");
			FILE *fd = fopen(p,"r");
			if(fd == NULL) continue;
			request[0] = UPDATE_AZLIST;
			request[1] = mapperID;
			for(int i=2; i<28; i++)
				request[i] = 0;
			while(!feof(fd)) {
				char line_buf2[1024] = "";
				fgets(line_buf2,1024,fd);
				char letter = tolower(line_buf2[0]);
				if(97 <= letter && letter <= 122)
					request[letter-97+2] += 1;
				else if (letter == 0)
					continue;
				else
					printf("Bad character! %c %d %s This is BAD!\n",letter,letter,p);
			}
			fclose(fd);
			send(sockfd, &request, REQUEST_MSG_SIZE*sizeof(int),0);
			recv(sockfd, &response, RESPONSE_MSG_SIZE*sizeof(int),MSG_WAITALL);
		}
		fclose(files);
// 3. GET_MAPPER_UPDATES
		//find total number of messages sent to server
		request[0] = GET_MAPPER_UPDATES;
		request[1] = mapperID;
		send(sockfd, &request, REQUEST_MSG_SIZE*sizeof(int),0);
		recv(sockfd, &response, RESPONSE_MSG_SIZE*sizeof(int),MSG_WAITALL);
		if(any_files > 0) fprintf(logfp,"[%d] UPDATE_AZLIST: %d\n", mapperID, response[2]);
    fprintf(logfp,"[%d] GET_MAPPER_UPDATES: %d %d\n",mapperID,response[1],response[2]);


// 4. GET_AZLIST
		request[0] = GET_AZLIST;
		request[1] = mapperID;
		int long_response[LONG_RESPONSE_MSG_SIZE] = {0};
		send(sockfd, &request, REQUEST_MSG_SIZE*sizeof(int),0);
		recv(sockfd, &long_response, LONG_RESPONSE_MSG_SIZE*sizeof(int),MSG_WAITALL);
		fprintf(logfp,"[%d] GET_AZLIST:",mapperID);
		for(int j=2; j<28; j++)
			fprintf(logfp," %d",long_response[j]);
		fprintf(logfp,"\n");

// 5. GET_ALL_UPDATES
		request[0] = GET_ALL_UPDATES;
		request[1] = mapperID;
		send(sockfd, &request, REQUEST_MSG_SIZE*sizeof(int),0);
		recv(sockfd, &response, RESPONSE_MSG_SIZE*sizeof(int),MSG_WAITALL);
		fprintf(logfp,"[%d] GET_ALL_UPDATES: %d %d\n",mapperID,response[1],response[2]);

// 6. CHECKOUT
		//This request is the last request sent from a mapper client
		request[0] = CHECKOUT;
		request[1] = mapperID;
		send(sockfd, &request, REQUEST_MSG_SIZE*sizeof(int),0);
		recv(sockfd, &response, RESPONSE_MSG_SIZE*sizeof(int),MSG_WAITALL);
		fprintf(logfp,"[%d] CHECKOUT: %d %d\n",mapperID,response[1],response[2]);

		//close connection
		close(sockfd);
		fprintf(logfp,"[%d] close connection\n", mapperID);
		return -1;
	}

	//Master waits until all mapper processes are terminated
	for (int i = 0; i < mappers; i++)
		wait(NULL);

  // master opens a connection to the server to get the final results
  FILE *results_file = fopen("./results.txt","w");
  int sockfd = socket(AF_INET , SOCK_STREAM , 0);

  struct sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_port = htons(server_port);
  address.sin_addr.s_addr = inet_addr(server_ip);

  if (connect(sockfd, (struct sockaddr *) &address, sizeof(address)) != 0) {
    printf("Connection error for master\n");
    return -1;
  }
  fprintf(logfp,"[m] open connection\n");

  int request[REQUEST_MSG_SIZE];
  int long_response[LONG_RESPONSE_MSG_SIZE] = {0};
  request[0] = GET_AZLIST;
  request[1] = 1;
  send(sockfd, &request, REQUEST_MSG_SIZE*sizeof(int),0);
  recv(sockfd, &long_response, LONG_RESPONSE_MSG_SIZE*sizeof(int),MSG_WAITALL);
  fprintf(logfp,"[m] GET_AZLIST:\n");
  for(int j=2; j<28; j++)
    fprintf(results_file,"%c %d\n", j+63, long_response[j]);

  close(sockfd);
  fclose(results_file);
  fclose(logfp);
  return 0;

}
