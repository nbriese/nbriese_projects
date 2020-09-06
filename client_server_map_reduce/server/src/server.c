#include <stdio.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <zconf.h>
#include <pthread.h>
#include <signal.h>
#include <arpa/inet.h>
#include "../include/protocol.h"
#include <semaphore.h>
#include <pthread.h>
int *azList;
int *updateStatus;
int *updateStatus_index;
sem_t sem_check;
void *connection(void *arg);

int main(int argc, char *argv[]) {
  int server_port;

  // interpret arguments
  if (argc == 2) { // 1 arguments
    server_port = atoi(argv[1]);
  } else {
    printf("Invalid or less number of arguments provided\n");
    printf("./server <server Port>\n");
    exit(0);
  }

  // Create a TCP socket.
	int sock = socket(AF_INET , SOCK_STREAM , 0);

	// Bind it to a local address.
	struct sockaddr_in servAddress;
	servAddress.sin_family = AF_INET;
	servAddress.sin_port = htons(server_port);
	servAddress.sin_addr.s_addr = htonl(INADDR_ANY);
	bind(sock, (struct sockaddr *) &servAddress, sizeof(servAddress));

	// Initialize semaphore
	sem_init(&sem_check,0,1);

	// azList saves the sum of all the word count results
	azList = (int *) malloc(26*sizeof(int));
	for(int i=0; i<26; i++)
		azList[i] = 0;
	updateStatus = (int *) malloc(MAX_CONCURRENT_CLIENTS*3*sizeof(int));
	for(int i=0; i<MAX_CONCURRENT_CLIENTS*3; i++)
		updateStatus[i] = 0;
	// *(updateStatus + r*3 + c) = is how you access info in this setup
    updateStatus_index = (int*) malloc(sizeof(int));
    *updateStatus_index = 0;
	pthread_t tids[MAX_CONCURRENT_CLIENTS];

	// listen on this port
	printf("server is listening\n");
	listen(sock, MAX_CONCURRENT_CLIENTS);
	int con_num = 0;
	while (con_num <= MAX_CONCURRENT_CLIENTS) {
		// Establish a TCP connection between a client and server.
		struct sockaddr_in clientAddress;
		socklen_t size = sizeof(struct sockaddr_in);

		int *clientfd = (int *) malloc(sizeof(int));
		*clientfd = accept(sock, (struct sockaddr*) &clientAddress, &size);

		// branch a new thread and send it clientfd
		pthread_create(&tids[con_num], NULL, connection, clientfd);
		con_num++;
	}

	// wait for all threads to join back
	for (int i = 0; i<con_num; i++)
		pthread_join(tids[i], NULL);

	// Close the socket.
	close(sock);
	free(azList);
	free(updateStatus);
	sem_destroy(&sem_check);
	return 0;
}

void *connection(void *arg) {
	// threads are responsible for handling requests from a client
  // read client’s messages from a socket, do necessary computation, and send response
	int *temp = (int *) arg;
	int clientfd = (int) *temp;
	while(1) {
		int request[REQUEST_MSG_SIZE] = {0};
		int response[RESPONSE_MSG_SIZE] = {0};
		sem_wait(&sem_check);
		recv(clientfd, &request, REQUEST_MSG_SIZE*sizeof(int),0);
		sem_post(&sem_check);

		response[0] = request[0];
		response[1] = 0;

		if(request[0] == CHECKIN) {
// 1. CHECKIN
			/* Server creates a new entry in updateStatus table for a new mapper client if
			corresponding entry does not exist in the table. If there is an existing entry in the table, the
			server simply changes the check in/out field to checked-in (1).*/
			sem_wait(&sem_check);
			int contains = -1;
			for(int i=0; i<*updateStatus_index; i++) {
				if(*(updateStatus + i*3) == request[1]) {
					contains = i;
					break;
				}
			}
			if(contains<0) {
				(*updateStatus_index)++;
				int i = *updateStatus_index -1;
				updateStatus[i*3] = request[1];
				updateStatus[(i*3)+1] = 0;
				updateStatus[(i*3)+2] = 1;
			}
			else {
				updateStatus[(contains*3)+2] = 1;
			}
			sem_post(&sem_check);

			response[2] = request[1];
			send(clientfd, &response, RESPONSE_MSG_SIZE*sizeof(int),0);
			printf("[%d] CHECKIN\n", request[1]);
		}

		else if(request[0] == UPDATE_AZLIST) {
// 2. UPDATE_AZLIST
			/* Server sums the word count results in the azList, and increases the number of
			update field of updateStatus table by 1 for the corresponding mapper client.*/
			sem_wait(&sem_check);
			for(int i=0; i<26; i++)
				azList[i] += request[i+2];

			for(int i=0; i<(*updateStatus_index); i++) {
				if(updateStatus[i*3] == request[1]) {
					updateStatus[i*3 + 1]++;
					break;
				}
			}
			sem_post(&sem_check);
			response[2] = request[1];
			send(clientfd, &response, RESPONSE_MSG_SIZE*sizeof(int),0);
		}

		else if(request[0] == GET_AZLIST) {
// 3. GET_AZLIST
			// Server returns the current values of the azList
			int long_response[LONG_RESPONSE_MSG_SIZE] = {0};
			long_response[0] = request[0];
			long_response[1] = 0;
			sem_wait(&sem_check);
			for(int i=0; i<26; i++)
				long_response[i+2] = azList[i];
			sem_post(&sem_check);
			send(clientfd, &long_response, LONG_RESPONSE_MSG_SIZE*sizeof(int),0);
			printf("[%d] GET_AZLIST\n",request[1]);
		}

		else if(request[0] == GET_MAPPER_UPDATES) {
// 4. GET_MAPPER_UPDATES
			// returns value of “number of updates” field of updateStatus table for the 	corresponding mapperID
			sem_wait(&sem_check);
			for(int i=0; i<*updateStatus_index; i++) {
				if(*(updateStatus+i*3) == request[1]) {
					response[2] = *(updateStatus+i*3 + 1);
					break;
				}
			}
			sem_post(&sem_check);
			send(clientfd, &response, RESPONSE_MSG_SIZE*sizeof(int),0);
			printf("[%d] GET_MAPPER_UPDATES\n",request[1]);
		}

		else if(request[0] == GET_ALL_UPDATES) {
// 5. GET_ALL_UPDATES
			// Server returns the sum of all values of “number of updates” field in the updateStatus table.
			sem_wait(&sem_check);
			response[2] = 0;
			for(int i=0; i<*updateStatus_index; i++)
					response[2] += *(updateStatus+i*3 + 1);
			sem_post(&sem_check);
			send(clientfd, &response, RESPONSE_MSG_SIZE*sizeof(int),0);
			printf("[%d] GET_ALL_UPDATES\n",request[1]);
		}

		else if(request[0] == CHECKOUT) {
// 6. CHECKOUT
			// Server updates check in/out field of theupdateStatus table to checked-out (0)
			sem_wait(&sem_check);
			for(int i=0; i<*updateStatus_index; i++) {
				if(*(updateStatus+i*3) == request[1]) {
					*(updateStatus+i*3 + 2) = 0;
					break;
				}
			}
			sem_post(&sem_check);
			response[2] = request[1];
			send(clientfd, &response, RESPONSE_MSG_SIZE*sizeof(int),0);
			printf("[%d] CHECKOUT\n",request[1]);

			//Close the TCP connection between a client and server.
			close(clientfd);
			free(temp);
		    return NULL;
		}
		else if(request[0] == 0) {
			// printf("recieved 0 so connection is done.\n");
			return NULL;
		}
	}
}
