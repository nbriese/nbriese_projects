// @copyright Nathan Briese 2020

#include "header.h"

// This is most ascii characters that aren't letters
// Used as a list of deliniators for spliting words
char  *DELIM_LIST = " !\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`{|}~\t\n";

void *production(void *arg) {
	fprintf(logfile,"Producer  : check in\n");

	// Read the input file one line at a time
	int line_num = -1;
	while(!feof(dataf)) {
		line_num++;
		fprintf(logfile,"Producer  : line %d\n", line_num);

		char str[1024] = "";
		char str2[1024] = "";
		fgets(str, MAX_LINE, dataf);
		char* token = strtok(str,DELIM_LIST);

		// Keep only the first letter of each word from the line
		while (token != NULL) {
			strncat(str2,token,1);
  		token = strtok(NULL,DELIM_LIST);
		}

		// Make a package
		struct package *pack = (struct package *) malloc(sizeof(struct package));
		pack->line_num = line_num;
		pack->next = NULL;
		pack->str[0] = '\0';
		strcpy(pack->str,str2);

		// Pass the pakage into the shared queue
		sem_wait(&queue);
		if(head == NULL) {
			head = pack;
		} else {
			struct package *temp = head;
			while(temp->next != NULL) {
				temp = temp->next;
			}
			temp->next = pack;
		}
		sem_post(&queue);
	}

	// When EOF is reached, send notifications to consumers specifying there will be no more data
	// Send a package with -1 for line number
	fprintf(logfile,"Producer  : EOF\n");
	struct package *eof_pack = (struct package *) malloc(sizeof(struct package));
	eof_pack->line_num = -1;
	eof_pack->next = NULL;
	sem_wait(&queue);
	//insert at tail of the &queue
	if(head == NULL)
		head = eof_pack;
	else{
		struct package *temp = head;
		while(temp->next != NULL) {
			temp = temp->next;
		}
		temp->next = eof_pack;
	}
	sem_post(&queue);
	sem_post(&queue);

	return NULL;
}
