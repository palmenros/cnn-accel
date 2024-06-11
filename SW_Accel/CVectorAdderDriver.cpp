#include <stdio.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <map>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include "util.hpp"
#include "CAccelDriver.hpp"
#include "CVectorAdderDriver.hpp"

uint32_t CVectorAdderDriver::Add(void * input1, void * input2, void * output,
          uint32_t length, uint32_t accum, uint64_t & elapsed)
{
  uint32_t phyInput1, phyInput2, phyOutput;
  struct timespec start, end;

  if (logging)
    printf("CVectorAdderDriver::Add(Input1=0x%08X, Input2=0x%08X, Output=0x%08X, Length=%u, Accum=%u)\n", 
          (uint32_t)input1, (uint32_t)input2, (uint32_t)output, length, accum);

  if (driver == 0) {
    if (logging)
      printf("Error: Calling Add() on a non-initialized accelerator.\n");
    return DEVICE_NOT_INITIALIZED;
  }

  // We need to obtain the physical addresses corresponding to each of the virtual addresses passed by the application.
  // The accelerator uses only the physical addresses (and only contiguous memory).
  phyInput1 = GetDMAPhysicalAddr(input1);
  if (phyInput1 == 0) {
    if (logging)
      printf("Error: No physical address found for virtual address 0x%08X\n", (uint32_t)input1);
    return VIRT_ADDR_NOT_FOUND;
  }
  phyInput2 = GetDMAPhysicalAddr(input2);
  if (phyInput2 == 0) {
    if (logging)
      printf("Error: No physical address found for virtual address 0x%08X\n", (uint32_t)input2);
    return VIRT_ADDR_NOT_FOUND;
  }
  phyOutput = GetDMAPhysicalAddr(output);
  if (phyOutput == 0) {
    if (logging)
      printf("Error: No physical address found for virtual address 0x%08X\n", (uint32_t)output);
    return VIRT_ADDR_NOT_FOUND;
  }

  struct user_message message = {(uint32_t)phyInput1, (uint32_t)phyInput2, (uint32_t)phyOutput, length, accum};

  if (logging)
    printf("\nStarting accel...\n");
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  int32_t readBytes = read(driver, (void *)&message, sizeof(message));
  if (readBytes != 0)
    printf("Warning! Read %d bytes instead than %d\n", readBytes, 0);

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  elapsed = CalcTimeDiff(end, start);

  return OK;
}

