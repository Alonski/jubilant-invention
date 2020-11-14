# Parallel Programming Histogram

Created by Alon Bukai and Oleg Danilyuk


## Instructions to Build

To build this project a *Makefile* is included.

Just run `make` in your terminal.

## Instructions to Run

This project must be run using `make run`.

### An example to run the program expecting input from stdin:
```sh
make run
```
The above should receive the amount of numbers from stdin and then request all number inputs from stdin.

### An example to run the program directly in one command:
```sh
make run < <(echo '12 1 1 1 2 2 2 2 3 4 4 5 6 6')
```
The above should print the following histogram:
```sh
====== Histogram ======
1: 3
2: 4
3: 1
4: 2
5: 1
6: 1
```
