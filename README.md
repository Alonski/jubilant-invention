# Parallel Programming Histogram

Created by Alon Bukai and Oleg Danilyuk

## Instructions to Build

To build this project a _Makefile_ is included.

Just run `make` in your terminal.

Make sure that your CUDA runtime is here: `/usr/local/cuda-9.1/lib64/libcudart_static.a`

Edit the `makefile` if your CUDA runtime is in a different location.

## Instructions to Run

This project must be run using `make run`.

### An example to run the program expecting input from stdin:

```sh
make run
```

The above should receive the amount of numbers from stdin and then request all number inputs from stdin.

### An example to run the program directly in one command:

```sh
make run < <(echo '28 1 2 2 3 3 3 4 4 4 4 5 5 5 5 5 6 6 6 6 6 6 7 7 7 7 7 7 7')
```

The above should print the following histogram:

```sh
====== Histogram ======
1: 1
2: 2
3: 3
4: 4
5: 5
6: 6
7: 7
```
