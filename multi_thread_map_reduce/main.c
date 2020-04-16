// @copyright Nathan Briese 2020

#include "header.h"

int main(int argc, char *argv[]){
	// con_num is number of consumer threads to use
	// default is 4
	int con_num = 4;
	char *file_name = "testing/test2.txt";
	printf("%d",argc);
	if(argc > 1) {
		file_name = argv[1];
	}
	else if(argc > 2) {
		con_num = atoi(argv[2]);
	}
	else if(argc != 1){
		printf("Too many arguments.\n");
		return -1;
	}

	dataf = fopen(file_name, "r");
	if(dataf == NULL) {
		printf("Error opening sepcified file.\n");
		return -1;
	}

	logfile = fopen("log.txt","w");

	// Initialize the semaphores
	if(sem_init(&queue,0,1) == -1) return -1;
	if(sem_init(&update,0,1) == -1) return -1;

	// Launch producer thread
	pthread_t tids[con_num+1];
	pthread_create(&tids[con_num], NULL, production, NULL);

	// Launch consumer threads
	// TODO Why long instead of int? Not the worst problem but definitly weird
	for (long i = 0; i<con_num; i++)
		pthread_create(&tids[i], NULL, consumption, (void *) i);

	// Wait for all threads to join back
	for (int i = 0; i < con_num+1; i++)
		pthread_join(tids[i], NULL);

	// Write the final result to result.txt
	FILE *result = fopen("result.txt","w");
	if(result == NULL) return -1;
	for(int i=0; i<26; i++)
		fprintf(result, "%c: %d\n", i+97, result_list[i]);
	fclose(result);

	// Clean up (deallocate)
	sem_destroy(&queue);
	sem_destroy(&update);
	fclose(dataf);
	fclose(logfile);
	if(head->next != NULL) free(head->next);
	free(head);

	return 0;
}
