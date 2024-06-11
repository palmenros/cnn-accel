#include <stdio.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <map>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include "CConvDriver.hpp"

uint32_t CConvDriver::Conv(void* input, void* output, void* filters, void* biases, uint32_t numFilters, uint32_t numChannels, uint32_t inputWidth, uint32_t inputHeight, bool performReLu)
{
  uint32_t phyInput, phyOutput, phyFilters, phyBiases;

  if (logging) {
    printf("CConvDriver::Add(Input=0x%08X, Output=0x%08X, Filters=0x%08X, NumFilters=%u, NumChannels=%u, InputWidth=%u, InputHeight=%u)\n", 
          (uint32_t)input, (uint32_t)output, (uint32_t)filters, numFilters, numChannels, inputWidth, inputHeight);
  }

  if (driver == 0) {
    if (logging)
      printf("Error: Calling Conv() on a non-initialized accelerator.\n");
    return DEVICE_NOT_INITIALIZED;
  }

  // We need to obtain the physical addresses corresponding to each of the virtual addresses passed by the application.
  // The accelerator uses only the physical addresses (and only contiguous memory).
  phyInput = GetDMAPhysicalAddr(input);
  if (phyInput == 0) {
    printf("Error: No physical address found for virtual address 0x%08X\n", (uint32_t)input);
    return VIRT_ADDR_NOT_FOUND;
  }
  phyOutput = GetDMAPhysicalAddr(output);
  if (phyOutput == 0) {
    printf("Error: No physical address found for virtual address 0x%08X\n", (uint32_t)output);
    return VIRT_ADDR_NOT_FOUND;
  }

  phyFilters = GetDMAPhysicalAddr(filters);
  if (phyFilters == 0) {
    printf("Error: No physical address found for virtual address 0x%08X\n", (uint32_t)phyFilters);
    return VIRT_ADDR_NOT_FOUND;
  }

  phyBiases = GetDMAPhysicalAddr(biases);
  if (phyFilters == 0) {
    printf("Error: No physical address found for virtual address 0x%08X\n", (uint32_t)phyFilters);
    return VIRT_ADDR_NOT_FOUND;
  }

  uint32_t resultOK;

  // struct user_message {  
  //   uint32_t input;
  //   uint32_t output;
  //   uint32_t filters;
  //   uint32_t biases;
  //   uint32_t numFilters;
  //   uint32_t numChannels;
  //   uint32_t inputWidth;
  //   uint32_t inputHeight;
  //   uint32_t performReLu;
  //   uint32_t resultOkPtr;
  // };
  struct user_message message = {
    (uint32_t)phyInput, 
    (uint32_t)phyOutput,
    (uint32_t)phyFilters,
    (uint32_t)phyBiases,
    numFilters,
    numChannels,
    inputWidth,
    inputHeight,
    performReLu,
    (uint32_t)(&resultOK)
    };

  if (logging)
    printf("\nStarting accel...\n");

  int32_t readBytes = read(driver, (void *)&message, sizeof(message));
  if (readBytes != 0)
    printf("Warning! Read %d bytes instead than %d\n", readBytes, 0);

  if(!resultOK) {
    printf("ERROR: Accelerator returned resultOK=false!\n");
    return DEVICE_CALL_ERROR;
  }

  return OK;
}