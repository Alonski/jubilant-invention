#pragma once

#define HISTOGRAM_SIZE 256

void test(int *data, int n);
int histogramOnGPU(int *data, int n, int *out_histogram);
