#ifndef CNN_H
#define CNN_H

void MaxPool(TFXP * input, TFXP * output, uint32_t channels, uint32_t width, uint32_t height);
void ReLU(TFXP * input, uint32_t channels, uint32_t width, uint32_t height);
void Sigmoid(TFXP * input, uint32_t numParams);
void AddBiases(TFXP * input, TFXP * biases, uint32_t channels, uint32_t width, uint32_t height);
void Conv2D(TFXP *input, TFXP * output, TFXP * filters,
      uint32_t numFilters, uint32_t numChannels,
      uint32_t inputWidth, uint32_t inputHeight,
      uint32_t convWidth = 3, uint32_t convHeight = 3);
void Dense(TFXP * input, TFXP * output, uint32_t inputSize, uint32_t outputSize,
      TFXP * weights, TFXP * biases);
void Flatten(TFXP * input, TFXP * output, uint32_t numFilters, uint32_t width, uint32_t height);

#endif 


