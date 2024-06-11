#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>

#include "model.h"
#include "cnn.h"

bool ConvertWeightsToFxP(CConvDriver& convolver, const uint32_t numLayers, float ** floatWeights, TFXP ** fxpWeights)
{
  float * pFloat;
  TFXP * pFxp;

  for (uint32_t iLayer = 0; iLayer < numLayers; ++ iLayer) {
    pFloat = floatWeights[iLayer];
    uint32_t layerSize = LayerTypes[iLayer] == CONV ? LayerShapes[iLayer][0] * LayerShapes[iLayer][1] * 3*3 : LayerShapes[iLayer][0] * LayerShapes[iLayer][1];
    
    if ( (fxpWeights[iLayer] = (TFXP*)convolver.AllocDMACompatible(layerSize * sizeof(TFXP))) == NULL ) {
      printf("Error allocating %" PRIu32 " bytes for FxP weights in layer %u\n", (uint32_t)(layerSize*sizeof(TFXP)), iLayer);
      return false;
    }

    pFxp = fxpWeights[iLayer];
    for (uint32_t iFilter = 0; iFilter < LayerShapes[iLayer][1]; ++ iFilter) {
      for (uint32_t iChannel = 0; iChannel < LayerShapes[iLayer][0]; ++ iChannel) {
        if (LayerTypes[iLayer] == CONV) {
          for (uint32_t iWeight = 0; iWeight < 9; ++ iWeight) {
            *pFxp = Float2Fxp(*pFloat, DECIMALS);
            ++ pFxp;
            ++ pFloat;
          }
        } else {
          *pFxp = Float2Fxp(*pFloat, DECIMALS);
          ++ pFxp;
          ++ pFloat;
        }
      }
    }
  }
  return true;
}

void FreeParams(const uint32_t numLayers, void ** params)
{
  for (uint32_t ii = 0; ii < numLayers; ++ ii) {
    if (params[ii]) {
      free(params[ii]);
      params[ii] = NULL;
    }
  }
}

bool LoadFloatWeights(const uint32_t numLayers, float ** weights)
{
  FILE * input = NULL;

  for (uint32_t iLayer = 0; iLayer < numLayers; ++ iLayer) {
    char title[256];
    snprintf(title, sizeof(title), "model/weights_%u.bin", iLayer);
    input = fopen(title, "rb");
    if (input == NULL) {
      printf("Error opening file [%s]\n", title);
      return false;
    }
    uint32_t layerSize = LayerTypes[iLayer] == CONV ? LayerShapes[iLayer][0] * LayerShapes[iLayer][1] * 3*3 : LayerShapes[iLayer][0] * LayerShapes[iLayer][1];
    if ( (weights[iLayer] = (float*)malloc(layerSize * sizeof(float))) == NULL ) {
      printf("Error allocating %" PRIu32 " bytes to read file [%s]\n", (uint32_t)(layerSize*sizeof(float)), title);
      fclose(input);
      return false;
    }
    if ( fread(weights[iLayer], sizeof(float), layerSize, input) != layerSize ) {
      printf("Error reading %u values from file [%s]\n", layerSize, title);
      fclose(input);
      return false;
    }
  }

  return true;
}

bool ConvertBiasesToFxP(CConvDriver& convolver, const uint32_t numLayers, float ** floatBiases, TFXP ** fxpBiases)
{
  float * pFloat;
  TFXP * pFxp;

  for (uint32_t iLayer = 0; iLayer < numLayers; ++ iLayer) {
    pFloat = floatBiases[iLayer];
    uint32_t layerSize = LayerShapes[iLayer][1];
    if ( (fxpBiases[iLayer] = (TFXP*)convolver.AllocDMACompatible(layerSize * sizeof(TFXP))) == NULL ) {
      printf("Error allocating %" PRIu32 " bytes for FxP biases in layer %u\n", (uint32_t)(layerSize*sizeof(TFXP)), iLayer);
      return false;
    }
    pFxp = fxpBiases[iLayer];
    for (uint32_t iFilter = 0; iFilter < LayerShapes[iLayer][1]; ++ iFilter) {
          *pFxp = Float2Fxp(*pFloat, DECIMALS);
          ++ pFxp;
          ++ pFloat;
    }
  }
  return true;
}

bool LoadFloatBiases(const uint32_t numLayers, float ** biases)
{
  FILE * input = NULL;

  for (uint32_t iLayer = 0; iLayer < numLayers; ++ iLayer) {
    char title[256];
    snprintf(title, sizeof(title), "model/bias_%u.bin", iLayer);
    input = fopen(title, "rb");
    if (input == NULL) {
      printf("Error opening file [%s]\n", title);
      return false;
    }
    uint32_t layerSize = LayerShapes[iLayer][1]; // One bias per output filter.
    if ( (biases[iLayer] = (float*)malloc(layerSize * sizeof(float))) == NULL ) {
      printf("Error allocating %" PRIu32 " bytes to read file [%s]\n", (uint32_t)(layerSize*sizeof(float)), title);
      fclose(input);
      return false;
    }
    if ( fread(biases[iLayer], sizeof(float), layerSize, input) != layerSize ) {
      printf("Error reading %u values from file [%s]\n", layerSize, title);
      fclose(input);
      return false;
    }
  }

  return true;
}

bool LoadModelInFxP(CConvDriver& convolver, TFXP ** fxpWeights, TFXP ** fxpBiases)
{
  float * floatWeights[NUM_LAYERS];
  float * floatBiases[NUM_LAYERS];

  for (uint32_t ii = 0; ii < NUM_LAYERS; ++ ii) {
    fxpWeights[ii] = NULL;
    floatWeights[ii] = NULL;
    fxpBiases[ii] = NULL;
    floatBiases[ii] = NULL;
  }

  if (!LoadFloatWeights(NUM_LAYERS, floatWeights)) {
    printf("Error reading the float weights.\n");
    FreeParams(NUM_LAYERS, (void**)floatWeights);
    return false;
  }
  ConvertWeightsToFxP(convolver, NUM_LAYERS, floatWeights, fxpWeights);
  FreeParams(NUM_LAYERS, (void**)floatWeights);

  if (!LoadFloatBiases(NUM_LAYERS, floatBiases)) {
    printf("Error reading the float biases.\n");
    FreeParams(NUM_LAYERS, (void**)floatBiases);
    return false;
  }
  ConvertBiasesToFxP(convolver, NUM_LAYERS, floatBiases, fxpBiases);
  FreeParams(NUM_LAYERS, (void**)floatBiases);

  return true;
}

bool LoadImageInFxp(const char * fileName, TFXP * inputImageFxp, uint8_t * inputImageRGB, uint32_t inputSize)
{
  FILE * inputImageFile;

  // Load input image and convert to FxP
  inputImageFile = fopen(fileName, "rb");
  if (inputImageFile == NULL) {
    printf("Error opening image [%s]\n", fileName);
    return false;
  }  

  if ( fread(inputImageRGB, 1, inputSize, inputImageFile) != inputSize ) {
    printf("Error reading %u bytes from [%s]\n", inputSize, fileName);
    fclose(inputImageFile);
    return false;
  }
  fclose(inputImageFile);

  // Convert image from RGB 8-8-8 pixels to fxp, normalized to [0.0-1.0)
  for (uint32_t ii = 0; ii < inputSize; ++ ii)
    inputImageFxp[ii] = Float2Fxp((inputImageRGB[ii]/255.0), DECIMALS);

  return true;
}

TFXP Inference(CConvDriver& convolver, TFXP * inputImageFxp, TFXP * buffer0, TFXP * buffer1, TFXP ** fxpWeights, TFXP ** fxpBiases, TTimes & times)
{
  uint32_t iLayer, size;
  struct timespec start, end;

  // TODO: Call the accelerator Conv_HW() function instead of calling the Conv2D function. Also remove the AddBiases and ReLu calls.
  iLayer = 0, size = 256;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  convolver.Conv(inputImageFxp, buffer0, fxpWeights[iLayer], fxpBiases[iLayer], LayerShapes[iLayer][1], LayerShapes[iLayer][0], size, size, true);
  size -= 2;
  // AddBiases(buffer0, fxpBiases[iLayer], LayerShapes[iLayer][1], size, size);
  // ReLU(buffer0, LayerShapes[iLayer][1], size, size);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  times.timeConv[iLayer] = CalcTimeDiff(end, start);
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  MaxPool(buffer0, buffer1, LayerShapes[iLayer][1], size, size);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  times.timeMaxPool[iLayer] = CalcTimeDiff(end, start);
  ++ iLayer;

  size = 127;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  convolver.Conv(buffer1, buffer0, fxpWeights[iLayer], fxpBiases[iLayer], LayerShapes[iLayer][1], LayerShapes[iLayer][0], size, size, true);
  size -= 2;
  // AddBiases(buffer0, fxpBiases[iLayer], LayerShapes[iLayer][1], size, size);
  // ReLU(buffer0, LayerShapes[iLayer][1], size, size);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  times.timeConv[iLayer] = CalcTimeDiff(end, start);
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  MaxPool(buffer0, buffer1, LayerShapes[iLayer][1], size, size);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  times.timeMaxPool[iLayer] = CalcTimeDiff(end, start);
  ++ iLayer;

  size = 62;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  convolver.Conv(buffer1, buffer0, fxpWeights[iLayer], fxpBiases[iLayer], LayerShapes[iLayer][1], LayerShapes[iLayer][0], size, size, true);
  size -= 2;
  // AddBiases(buffer0, fxpBiases[iLayer], LayerShapes[iLayer][1], size, size);
  // ReLU(buffer0, LayerShapes[iLayer][1], size, size);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  times.timeConv[iLayer] = CalcTimeDiff(end, start);
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  MaxPool(buffer0, buffer1, LayerShapes[iLayer][1], size, size);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  times.timeMaxPool[iLayer] = CalcTimeDiff(end, start);
  ++ iLayer;

  size = 30;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  convolver.Conv(buffer1, buffer0, fxpWeights[iLayer], fxpBiases[iLayer], LayerShapes[iLayer][1], LayerShapes[iLayer][0], size, size, true);
  size -= 2;
  // AddBiases(buffer0, fxpBiases[iLayer], LayerShapes[iLayer][1], size, size);
  // ReLU(buffer0, LayerShapes[iLayer][1], size, size);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  times.timeConv[iLayer] = CalcTimeDiff(end, start);
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  MaxPool(buffer0, buffer1, LayerShapes[iLayer][1], size, size);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  times.timeMaxPool[iLayer] = CalcTimeDiff(end, start);
  ++ iLayer;

  size = 14;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  convolver.Conv(buffer1, buffer0, fxpWeights[iLayer], fxpBiases[iLayer], LayerShapes[iLayer][1], LayerShapes[iLayer][0], size, size, true);
  size -= 2;
  // AddBiases(buffer0, fxpBiases[iLayer], LayerShapes[iLayer][1], size, size);
  // ReLU(buffer0, LayerShapes[iLayer][1], size, size);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  times.timeConv[iLayer] = CalcTimeDiff(end, start);
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  MaxPool(buffer0, buffer1, LayerShapes[iLayer][1], size, size);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  times.timeMaxPool[iLayer] = CalcTimeDiff(end, start);

  size = 6;
  // Flatten the output for the next dense layer: [row, col, filter]
  // From [64, 6, 6] to [2304]
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  Flatten(buffer1, buffer0, LayerShapes[iLayer][1], size, size);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  times.timeFlatten = CalcTimeDiff(end, start);
  ++ iLayer;

  // Output is now 6x6x64 --> 2304. Goes to a fully-connected layer.
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  Dense(buffer0, buffer1, LayerShapes[iLayer][0], LayerShapes[iLayer][1], fxpWeights[iLayer], fxpBiases[iLayer]);
  ReLU(buffer1, 1, LayerShapes[iLayer][1], 1);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  times.timeDense[iLayer] = CalcTimeDiff(end, start);
  ++ iLayer;

  // Output is now an array of 512 values. Goes to the final fully-connected layer.
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  Dense(buffer1, buffer0, LayerShapes[iLayer][0], LayerShapes[iLayer][1], fxpWeights[iLayer], fxpBiases[iLayer]);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  times.timeDense[iLayer] = CalcTimeDiff(end, start);

  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  Sigmoid(buffer0, 1);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  times.timeSigmoid = CalcTimeDiff(end, start);

  return buffer0[0];
}

uint64_t CalcTimeDiff(const struct timespec & time2, const struct timespec & time1)
{
  return time2.tv_sec == time1.tv_sec ?
    time2.tv_nsec - time1.tv_nsec :
    (time2.tv_sec - time1.tv_sec - 1) * 1e9 + (1e9 - time1.tv_nsec) + time2.tv_nsec;
}




