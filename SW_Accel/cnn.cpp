#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "model.h"
#include "cnn.h"

void MaxPool(TFXP * input, TFXP * output, uint32_t channels, uint32_t width, uint32_t height)
{
  // The input width is the argument received. The output width can be one less if the dimensions are not even numbers.
  uint32_t outWidth, outHeight;
  outWidth = ( (width % 2) == 0) ? width : width - 1;
  outHeight = ( (height % 2) == 0) ? height : height - 1;

  TFXP * p = input;
  for (uint32_t iChannel = 0; iChannel < channels; ++ iChannel) {
    for (uint32_t iRow = 0; iRow < outHeight; iRow += 2) {
      for (uint32_t iCol = 0; iCol < outWidth; iCol += 2) {
        TFXP val;
        val = * p;
        if (*(p+1) > val) val = *(p+1);
        if (*(p+width) > val) val = *(p+width);
        if (*(p+width+1) > val) val = *(p+width+1);
        *output = val;
        ++ output;
        p += 2;
      }
      p += width; // Skip one row that has already been processed
      if (width != outWidth)
        ++p; // Skip also the last column of the previous one
    }
    if (width != outWidth)
      p += width; // When crossing channels, also skip the last row if odd number.
  }
}

void ReLU(TFXP * input, uint32_t channels, uint32_t width, uint32_t height)
{
  for (uint32_t ii = 0; ii < channels*width*height; ++ ii) {
    if ( Fxp2Float(input[ii], DECIMALS) < 0.0 )
      input[ii] = 0;
  }
}

void AddBiases(TFXP * input, TFXP * biases, uint32_t channels, uint32_t width, uint32_t height)
{
  for (uint32_t iChannel = 0; iChannel < channels; ++ iChannel) {
    for (uint32_t iPixel = 0; iPixel < width * height; ++ iPixel) {
      *input = *input + *biases;
      ++ input;
    }
    ++ biases;
  }
}

void Conv2D(TFXP *input, TFXP * output, TFXP * filters,
      uint32_t numFilters, uint32_t numChannels,
      uint32_t inputWidth, uint32_t inputHeight,
      uint32_t convWidth, uint32_t convHeight)
{
  for (uint32_t iFilter = 0; iFilter < numFilters; ++ iFilter) {
    for (uint32_t y = 0; y < (inputHeight-2); ++y) {
      for (uint32_t x = 0; x < (inputWidth-2); ++ x) {
        TFXP acc;
        acc = 0;
        for (uint32_t iChannel = 0; iChannel < numChannels; ++ iChannel) {
          for (uint32_t cy = 0; cy < convHeight; ++ cy) {
            for (uint32_t cx = 0; cx < convWidth; ++cx) {
              //acc += filters[iFilter][iChannel][cy][cx] * input[iChannel][y+cy][x+cx];
              TFXP v, f;
              f = *(filters + iFilter*numChannels*convHeight*convWidth + iChannel*convHeight*convWidth + cy*convWidth + cx);
              v = *(input + iChannel*inputWidth*inputHeight + (y+cy)*inputWidth + (x+cx));
              acc += FXP_Mult(f, v, DECIMALS);
            }
          }
        }
        //output[iFilter][y][x] = acc;
        *(output + iFilter * (inputHeight-2)*(inputWidth-2) + y*(inputWidth-2) + x) = acc;
      }
    }
  }
}


void Dense(TFXP * input, TFXP * output, uint32_t inputSize, uint32_t outputSize,
      TFXP * weights, TFXP * biases)
{
  TFXP tmp;

  for (uint32_t ii = 0; ii < outputSize; ++ ii) {
    tmp = 0;
    for (uint32_t jj = 0; jj < inputSize; ++ jj) {
      tmp += FXP_Mult(input[jj], (*weights));
      ++ weights;
    }
    output[ii] = tmp + biases[ii];
  }
}

void Sigmoid(TFXP * input, uint32_t numParams)
{
  for (uint32_t ii = 0; ii < numParams; ++ ii) {
    float tmp;
    tmp = Fxp2Float(input[ii], DECIMALS);
    tmp = 1.0 / ( 1 + expf(-tmp) );
    input[ii] = Float2Fxp(tmp, DECIMALS);
  }
}

// The Keras/TF Flatten layer tranposes the array.
void Flatten(TFXP * input, TFXP * output, uint32_t numFilters, uint32_t width, uint32_t height)
{
  TFXP * p = output;

  for (uint32_t iRow = 0; iRow < height; ++ iRow) {
    for (uint32_t iCol = 0; iCol < width; ++ iCol) {
      for (uint32_t iFilter = 0; iFilter < numFilters; ++ iFilter) {
        *(p++) = *(input + iFilter*width*height + iRow*width + iCol);  
      }
    }
  }
}




