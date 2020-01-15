# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = nvcc

all: test_v1 test_v2 test_v3

test_v1:
	cd ising; make lib; cd ..
	cd ising; cp lib/*.a inc/ising.h ../; cd ..
	$(CC) tester.cu ising_sequential.a ising_v1.a -o $@
	#"*********************************"
	#"*******v1_implementation*********"
	#"*********************************"
	./test_v1

test_v2:
	cd ising; make lib; cd ..
	cd ising; cp lib/*.a inc/ising.h ../; cd ..
	$(CC) tester.cu ising_sequential.a ising_v2.a -o $@
	#"*********************************"
	#"*******v2_implementation*********"
	#"*********************************"
	./test_v2

test_v3:
	cd ising; make lib; cd ..
	cd ising; cp lib/*.a inc/ising.h ../; cd ..
	$(CC) tester.cu ising_sequential.a ising_v3.a -o $@
	#"*********************************"
	#"*******v3_implementation*********"
	#"*********************************"
	./test_v3


