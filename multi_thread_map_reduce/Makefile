# Makefile for Multi-Thread Map-Reduce
# copyright Nathan Briese 2020
CC = gcc
CFLAGS = -g -std=c99

mtmr: main.o producer.o consumer.o
	gcc -o mtmr $^ $(CFLAGS) -lpthread

main.o: main.c header.h
	gcc -o $@ -c main.c $(CFLAGS)

producer.o: producer.c header.h
	gcc -o $@ -c producer.c $(CFLAGS)

consumer.o: consumer.c header.h
	gcc -o $@ -c consumer.c $(CFLAGS)

clean: 
	rm mtmr main.o producer.o consumer.o utils.o




