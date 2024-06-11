#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>

#include "conv.h"

#define MAX_WIDTH 128
#define MAX_HEIGHT 128
#define MAX_FILTERS 256

using SW_FXP_t = int32_t;
using SW_FXP_MULT_t = int64_t;

SW_FXP_t input[MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS];
SW_FXP_t coeffs[MAX_CHANNELS * MAX_FILTERS * 9];
SW_FXP_t biases[MAX_FILTERS];

SW_FXP_t outputSW[MAX_WIDTH * MAX_HEIGHT * MAX_FILTERS];
SW_FXP_t outputHW[MAX_WIDTH * MAX_HEIGHT * MAX_FILTERS];

///////////////////////////////////////////////////////////////////////////////

inline SW_FXP_t FXP_Mult(SW_FXP_t a, SW_FXP_t b)
{
  //return a*b;
  SW_FXP_MULT_t res = (SW_FXP_MULT_t)a * (SW_FXP_MULT_t)b;
  res = res >> FXP_NUM_DECIMALS;
  return res;
}

///////////////////////////////////////////////////////////////////////////////

inline float Fxp2Float(SW_FXP_t value)
{
  //return value;
  return ((value) / (float)(1 << (FXP_NUM_DECIMALS)));
}

void ReLU_SW(SW_FXP_t * input, uint32_t channels, uint32_t width, uint32_t height)
{
  for (uint32_t ii = 0; ii < channels*width*height; ++ ii) {
    if ( Fxp2Float(input[ii]) < 0.0 )
      input[ii] = 0;
  }
}

void AddBiases_SW(SW_FXP_t * input, SW_FXP_t * biases, uint32_t channels, uint32_t width, uint32_t height)
{
  for (uint32_t iChannel = 0; iChannel < channels; ++ iChannel) {
    for (uint32_t iPixel = 0; iPixel < width * height; ++ iPixel) {
      *input = *input + *biases;
      ++ input;
    }
    ++ biases;
  }
}

void Conv2D_SW(SW_FXP_t *input, SW_FXP_t * output, SW_FXP_t * filters,
      uint32_t numFilters, uint32_t numChannels,
      uint32_t inputWidth, uint32_t inputHeight)
{
  for (uint32_t iFilter = 0; iFilter < numFilters; ++ iFilter) {
    for (uint32_t y = 0; y < (inputHeight-2); ++y) {
      for (uint32_t x = 0; x < (inputWidth-2); ++ x) {
        SW_FXP_t acc;
        acc = 0;
        for (uint32_t iChannel = 0; iChannel < numChannels; ++ iChannel) {
          for (uint32_t cy = 0; cy < CONV_FILTER_HEIGHT; ++ cy) {
            for (uint32_t cx = 0; cx < CONV_FILTER_WIDTH; ++cx) {
              //acc += filters[iFilter][iChannel][cy][cx] * input[iChannel][y+cy][x+cx];
              SW_FXP_t v, f;
              f = *(filters + iFilter*numChannels*CONV_FILTER_HEIGHT*CONV_FILTER_WIDTH + iChannel*CONV_FILTER_HEIGHT*CONV_FILTER_WIDTH + cy*CONV_FILTER_WIDTH + cx);
              v = *(input + iChannel*inputWidth*inputHeight + (y+cy)*inputWidth + (x+cx));
              acc += FXP_Mult(f, v);
            }
          }
        }
        //output[iFilter][y][x] = acc;
        *(output + iFilter * (inputHeight-2)*(inputWidth-2) + y*(inputWidth-2) + x) = acc;
      }
    }
  }
}

void Conv_SW(SW_FXP_t* input, SW_FXP_t* output, SW_FXP_t* filters, SW_FXP_t* biases,
      uint32_t numFilters, uint32_t numChannels,
      uint32_t inputWidth, uint32_t inputHeight, bool useReLu)
{
	const int outputHeight = inputHeight - CONV_FILTER_HEIGHT + 1;
	const int outputWidth = inputWidth - CONV_FILTER_WIDTH + 1;

	Conv2D_SW(input, output, filters, numFilters, numChannels, inputWidth, inputHeight);
	AddBiases_SW(output, biases, numFilters, outputWidth, outputHeight);

	if(useReLu) {
		ReLU_SW(output, numFilters, outputWidth, outputHeight);
	}
}

///////////////////////////////////////////////////////////////////////////////
void InitVectors(SW_FXP_t * input, uint32_t sizeInput, SW_FXP_t * coeffs, uint32_t sizeCoeffs, SW_FXP_t* biases, uint32_t sizeBiases)
{
  for (uint32_t ii = 0; ii < sizeInput; ++ ii)
    input[ii] = rand();
  for (uint32_t ii = 0; ii < sizeCoeffs; ++ ii)
    coeffs[ii] = rand();
  for (uint32_t ii = 0; ii < sizeBiases; ++ ii)
	biases[ii] = rand();
}

///////////////////////////////////////////////////////////////////////////////
bool CompareVectors(SW_FXP_t * input1, SW_FXP_t * input2, uint32_t size)
{
  bool res = true;

  for (uint32_t ii = 0; res && ii < size; ++ ii)
    res = (input1[ii] == input2[ii]);

  return res;
}

///////////////////////////////////////////////////////////////////////////////
void PrintVector(uint16_t* v, uint32_t size) {
	const uint32_t MAX_SIZE = 64;

	if (size > MAX_SIZE) {
		printf("[Not printed due to exceeding max size %d > %d]\n", size, MAX_SIZE);
		return;
	}

	printf("[ ");

	for(uint32_t i = 0; i < size; ++i) {
		printf("%#06X ", v[i]);
	}

	printf("]\n");
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char ** argv)
{
  uint32_t width = MAX_WIDTH, height = MAX_HEIGHT;
//  uint32_t width = 4, height = 4;
  uint32_t channels, filters;
  uint32_t currentOutputSize;
//  uint32_t sizes[][2] = { {3, 32}, {16, 16}, {32, 32}, {64, 64}, {128, 128}, {256, 256} };
  uint32_t sizes[][2] = { {3, 32}, {16, 16}, {32, 32}};
//  uint32_t sizes[][2] = { {2, 1} };

  uint32_t numSizes = sizeof(sizes) / (sizeof(uint32_t) * 2);

  srand(time(NULL));
  InitVectors(input, MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS, coeffs, MAX_CHANNELS * MAX_FILTERS * 9, biases, MAX_FILTERS);
  for (uint32_t iRelu = 0; iRelu <= 1; ++iRelu) {
	  bool useRelu = iRelu;
	  for (uint32_t iTest = 0; iTest < numSizes; ++ iTest) {
		channels = sizes[iTest][0]; filters = sizes[iTest][1];
		currentOutputSize = (width-2) * (height-2) * filters;
		printf("Evaluating execution for %" PRIu32 " --> %" PRIu32 ", ReLu: %d\n", channels, filters, useRelu);
		memset(outputSW, 0, currentOutputSize * sizeof(FXP_t));
		memset(outputHW, 0, currentOutputSize * sizeof(FXP_t));
		printf("  SW\n");

		Conv_SW(input, outputSW, coeffs, biases, filters, channels, width, height, useRelu);

		printf("  HW\n");
		bool resOK = false;
		Conv(reinterpret_cast<FXP_t*>(input), reinterpret_cast<FXP_t*>(outputHW), reinterpret_cast<FXP_t*>(coeffs), reinterpret_cast<FXP_t*>(biases), filters, channels, width, height, useRelu, resOK);
		if (!resOK) {
			  printf("\n\n====== ERROR: CONV FUNCTION RETURNED NOT OK (resOK=false) ======\n\n");
			  return 1;
		}

		printf("SW output: ");
		PrintVector(reinterpret_cast<uint16_t*>(outputSW), currentOutputSize);
		printf("HW output: ");
		PrintVector(reinterpret_cast<uint16_t*>(outputHW), currentOutputSize);

		if (!CompareVectors(outputSW, outputHW, currentOutputSize)) {
			  printf("\n\n====== ERROR COMPARING RESULTS WITH REFERENCE!!! ======\n\n");
			  return 1;
		} else {
			  printf("  --> OK!\n");
		}
	  }
  }
  printf("\n");

  return 0;
}


