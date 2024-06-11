#ifndef CVECTORADDER_HPP
#define CVECTORADDER_HPP

#include "CAccelDriver.hpp"

#define CONV_FILTER_HEIGHT 3
#define CONV_FILTER_WIDTH 3

class CConvDriver : public CAccelDriver {
  protected:
    // Structure used to pass commands between user-space and kernel-space.
    struct user_message {  
      uint32_t input;
      uint32_t output;
      uint32_t filters;
      uint32_t biases;
      uint32_t numFilters;
      uint32_t numChannels;
      uint32_t inputWidth;
      uint32_t inputHeight;
      uint32_t performReLu;
      uint32_t resultOkPtr;
    };

  public:
    CConvDriver(bool Logging = false)
      : CAccelDriver(Logging) {}

    ~CConvDriver() {}


    // The data must be organized as follows:
    // uint16_t input[NUM_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH]
    // uint16_t output[NUM_FILTERS][OUTPUT_HEIGHT][OUTPUT_WIDTH]
    // uint16_t filters[NUM_FILTERS][NUM_CHANNELS][CONV_HEIGHT][CONV_WIDTH]
    uint32_t Conv(void* input, void* output, void* filters, void* biases, uint32_t numFilters, uint32_t numChannels, uint32_t inputWidth, uint32_t inputHeight, bool performReLu);
};

// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read/COR)
//        bit 7  - auto_restart (Read/Write)
//        bit 9  - interrupt (Read)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0 - enable ap_done interrupt (Read/Write)
//        bit 1 - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0 - ap_done (Read/TOW)
//        bit 1 - ap_ready (Read/TOW)
//        others - reserved
// 0x10 : Data signal of input_r
//        bit 31~0 - input_r[31:0] (Read/Write)
// 0x14 : Data signal of input_r
//        bit 31~0 - input_r[63:32] (Read/Write)
// 0x18 : reserved
// 0x1c : Data signal of output_r
//        bit 31~0 - output_r[31:0] (Read/Write)
// 0x20 : Data signal of output_r
//        bit 31~0 - output_r[63:32] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of filters_offset
//        bit 31~0 - filters_offset[31:0] (Read/Write)
// 0x2c : Data signal of filters_offset
//        bit 31~0 - filters_offset[63:32] (Read/Write)
// 0x30 : reserved
// 0x34 : Data signal of biases_offset
//        bit 31~0 - biases_offset[31:0] (Read/Write)
// 0x38 : Data signal of biases_offset
//        bit 31~0 - biases_offset[63:32] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of numFilters
//        bit 31~0 - numFilters[31:0] (Read/Write)
// 0x44 : reserved
// 0x48 : Data signal of numChannels
//        bit 31~0 - numChannels[31:0] (Read/Write)
// 0x4c : reserved
// 0x50 : Data signal of inputWidth
//        bit 31~0 - inputWidth[31:0] (Read/Write)
// 0x54 : reserved
// 0x58 : Data signal of inputHeight
//        bit 31~0 - inputHeight[31:0] (Read/Write)
// 0x5c : reserved
// 0x60 : Data signal of performReLu
//        bit 0  - performReLu[0] (Read/Write)
//        others - reserved
// 0x64 : reserved
// 0x68 : Data signal of resultOk
//        bit 0  - resultOk[0] (Read)
//        others - reserved
// 0x6c : Control signal of resultOk
//        bit 0  - resultOk_ap_vld (Read/COR)
//        others - reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)


#endif