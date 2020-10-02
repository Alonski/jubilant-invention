# Parallel Programming Histogram

Created by Alon Bukai and Oleg Danilyuk


## Instructions to Build

To build this project a *Makefile* is included.

Just run `make` in your terminal.

## Instructions to Run

This project must be run using `make run` and supplied the number(N) of processes to create in parallel.

### An example to run the program with a supplied file(VALUES):
```sh
make run N=2
```
The above should receive the amount of numbers from stdin and then request all number inputs from stdin.

### An example to run the program without a supplied file(VALUES):
```sh
make run N=2
4 1 1 2 3
```
The above should print the following histogram:
```sh
====== Histogram ======
1: 2
2: 1
3: 1
```
