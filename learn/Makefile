# Makefile for cs224u rank svm code

CC=gcc
CFLAGS= -g -fnested-functions 
LD=gcc
LDFLAGS= -g

all: rank_svm

clean: 
	rm -f *.o
	rm rank_svm

rank_svm: rank_svm.o
	$(LD) $(LDFLAGS) rank_svm.o -o rank_svm

rank_svm.o: rank_svm.c
	$(CC) -c $(CFLAGS) rank_svm.c -o rank_svm.o