# Parallel Programming Histogram

Created by Alon Bukai and Oleg Danilyuk


## Instructions to Build

To build this project a *Makefile* is included.

Just run `make` in your terminal.

## Instructions to Run

This project must be run using `make run` and supplied the number(N) of processes to create in parallel.

### An example to run the program expecting input from stdin:
```sh
make run N=2
```
The above should receive the amount of numbers from stdin and then request all number inputs from stdin.

### An example to run the program directly in one command:
```sh
make run N=2 < <(echo '28 1 1 1 1 2 2 2 2 3 3 3 3 3 3 4 4 4 4 5 5 6 6 6 6 7 8 8 9')
```
The above should print the following histogram:
```sh
====== Histogram ======
1: 4
2: 4
3: 6
4: 4
5: 2
6: 4
7: 1
8: 2
9: 1
```
