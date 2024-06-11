#pragma once

#include <stdint.h>
#include "ap_fixed.h"

#define MAX_CHANNELS 256

//Number of FXP_t that can fit in the row buffer. This row buffer needs to fit num_channels * inputWidth FXP_t elements.
#define MAX_ROW_BUFFER_SIZE 4192

#define CONV_FILTER_HEIGHT 3
#define CONV_FILTER_WIDTH 3

const uint32_t FXP_NUM_DECIMALS = 20;

// Based on Xilinx documentation, the integer bit includes the signed bit.
using FXP_t = ap_fixed<32, 32-FXP_NUM_DECIMALS>;
using FXP_MULT_t = ap_fixed<64, 64-2*FXP_NUM_DECIMALS>;

void Conv(FXP_t* input, FXP_t* output, FXP_t* filters, FXP_t* biases, uint32_t numFilters,  uint32_t numChannels, uint32_t inputWidth, uint32_t inputHeight, bool performReLu, bool& resultOk);
