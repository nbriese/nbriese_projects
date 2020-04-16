// @copyright Nathan Briese 2020

#ifndef _HEADER_H_
#define _HEADER_H_
#include <semaphore.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// package definition
struct package{
  int line_num;
  char str[1024];
  struct package* next;
};

// Global variables:
static int MAX_LINE = 1024;
FILE *logfile;
FILE *dataf;
int result_list[26];
sem_t queue;
sem_t update;
struct package *head;

// Function declarations:
void *production(void *arg);
void *consumption(void *arg);

#endif
