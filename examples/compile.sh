#!/bin/bash 

g++ -std=c++11 $1 `pkg-config --libs --cflags opencv` -o $2
