#pragma once

#define HISTOGRAM_SIZE 256

int calculateHistogramCUDA(int *histogram, int *data, int numElements);
