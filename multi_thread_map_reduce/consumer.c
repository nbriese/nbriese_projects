// @copyright Nathan Briese 2020

#include "header.h"
#include <ctype.h>

void *consumption(void *arg) {
	long con_num = (long) arg;
	fprintf(logfile,"Consumer %ld: check in\n",con_num);

	while(1) {
		struct package *pack;
		sem_wait(&queue);
		if(head == NULL) {
			sem_post(&queue);
			continue;
		}
		// When consumer reads the eof package it will leave it at the head of the queue
		// so the rest of the consumers also get the message
		// Main will free the eof package at the end once all consumers are joined
		else if(head->line_num == -1) {
			sem_post(&queue);
			break;
		}
		else {
			// Extract the package from the head of the list
			pack = head;
			head = head->next;
		}
		sem_post(&queue);

		fprintf(logfile,"Consumer %ld: line %d\n",con_num,pack->line_num);

		// Word count
		int temp_result_list[26] = {0};
		for(int i=0; i < strlen(pack->str); i++) {
			if(97 <= tolower(pack->str[i]) <= 122)
				temp_result_list[tolower(pack->str[i])-97]++;
			else
				// TODO Make this error message more formal or actually do something
				printf("Bad character! %c This is BAD!\n",tolower(pack->str[i]));
		}

		// Update the result list
		sem_wait(&update);
		for(int i=0; i<26; i++)
			result_list[i] += temp_result_list[i];
		sem_post(&update);

		free(pack);
	}

	fprintf(logfile,"Consumer %ld: EOF\n",con_num);

	return NULL;
}
