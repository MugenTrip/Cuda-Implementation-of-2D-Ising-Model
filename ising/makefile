SHELL := /bin/bash

IDIR =./inc

ODIR=./lib

SDIR=./src

_OBJ = ising_sequential.o ising_v1.o ising_v2.o ising_v3.o 
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

_SRC = ising_sequential.c ising_v1.c ising_v2.c ising_v3.c 
SRC = $(patsubst %,$(SDIR)/%,$(_SRC))

_DEPS = ising.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_FINAL = ising_sequential.a ising_v1.a ising_v2.a ising_v3.a 
FINAL = $(patsubst %,$(ODIR)/%,$(_FINAL))

# define the C/C++ compiler to use,default here is clang 
#-lm -L$(NETLIB_LAPACK_ROOT)/lib64 -I$(NETLIB_LAPACK_ROOT)/include 
CC = nvcc
CFLAGS = -I$(IDIR) 

all: 
	$(CC) -c src/ising_sequential.cu -o lib/ising_sequential.o $(CFLAGS)
	$(CC) -c src/ising_v1.cu -o lib/ising_v1.o $(CFLAGS)
	$(CC) -c src/ising_v2.cu -o lib/ising_v2.o $(CFLAGS)
	$(CC) -c src/ising_v3.cu -o lib/ising_v3.o $(CFLAGS)
	ar rcs lib/ising_sequential.a lib/ising_sequential.o 
	ar rcs lib/ising_v1.a lib/ising_v1.o
	ar rcs lib/ising_v2.a lib/ising_v2.o 
	ar rcs lib/ising_v3.a lib/ising_v3.o 