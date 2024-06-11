#include "conv.h"
#include <stdint.h>

// NOTE: *Must* be a power of 2
// NOTE: If changed, must also update NUM_PARALLEL_CHANNELS_SHIFT.
#define NUM_PARALLEL_CHANNELS 2

// log2(NUM_PARALLEL_CHANNELS)
#define NUM_PARALLEL_CHANNELS_SHIFT 1

#define PAR_CHAN_MASK (NUM_PARALLEL_CHANNELS - 1)

// Define some loop tripcounts to get performance estimates inside Vitis HLS.
#define LOOP_TRIPCOUNT_INPUT_WIDTH 256
#define LOOP_TRIPCOUNT_INPUT_HEIGHT 256
#define LOOP_TRIPCOUNT_OUTPUT_WIDTH 254
#define LOOP_TRIPCOUNT_OUTPUT_HEIGHT 254
#define LOOP_TRIPCOUNT_CHANNELS 32
#define LOOP_TRIPCOUNT_FILTERS 32

// input[iChannel][y][x]
#define IN_IDX(iChannel, y, x) ((x) + (y) * inputWidth + (iChannel) * inputWidth * inputHeight)

// filters[iFilter][iChannel][cy][cx]
#define FILT_IDX(iFilter, iChannel, cy, cx) ((cx) + (cy) * CONV_FILTER_WIDTH + (iChannel) * CONV_FILTER_WIDTH * CONV_FILTER_HEIGHT + (iFilter) * numChannels * CONV_FILTER_WIDTH * CONV_FILTER_HEIGHT)

// output[iFilter][y][x]
#define OUT_IDX(iFilter, y, x) ((x) + (y) * outputWidth + (iFilter) * outputHeight * outputWidth)

FXP_t ReLu(FXP_t x) {
	if (x < 0) {
		return 0;
	} else {
		return x;
	}
}

void Conv(FXP_t* input, FXP_t* output, FXP_t* filters, FXP_t* biases, uint32_t numFilters,  uint32_t numChannels, uint32_t inputWidth, uint32_t inputHeight, bool performReLu, bool& resultOk)
{
#pragma HLS INTERFACE s_axilite port=numFilters
#pragma HLS INTERFACE s_axilite port=numChannels
#pragma HLS INTERFACE s_axilite port=inputWidth
#pragma HLS INTERFACE s_axilite port=inputHeight
#pragma HLS INTERFACE s_axilite port=performReLu
#pragma HLS INTERFACE s_axilite port=resultOk
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS INTERFACE m_axi depth=1024 port=input offset=slave latency=30 bundle=inout
#pragma HLS INTERFACE m_axi depth=1024 port=filters offset=slave latency=30 bundle=inout
#pragma HLS INTERFACE m_axi depth=1024 port=output offset=slave latency=30 bundle=inout
#pragma HLS INTERFACE m_axi depth=1024 port=biases offset=slave latency=30 bundle=inout

	if (numChannels > MAX_CHANNELS) {
		resultOk = false;
		return;
	}

	if (numChannels * inputWidth > MAX_ROW_BUFFER_SIZE) {
		// This parameter combination won't fit on our row buffer.
		resultOk = false;
		return;
	}

	const int outputHeight = inputHeight - CONV_FILTER_HEIGHT + 1;
	const int outputWidth = inputWidth - CONV_FILTER_WIDTH + 1;

	filter_loop: for (uint32_t iFilter = 0; iFilter < numFilters; ++ iFilter) {
#pragma HLS LOOP_TRIPCOUNT min=LOOP_TRIPCOUNT_FILTERS max=LOOP_TRIPCOUNT_FILTERS

		// Cache the filter coefficients here. Only read the coefficients once and use them in all the output pixel computation.
		FXP_t filter_coeffs[MAX_CHANNELS][CONV_FILTER_WIDTH][CONV_FILTER_HEIGHT];
#pragma HLS ARRAY_PARTITION variable=filter_coeffs type=complete dim=3
#pragma HLS ARRAY_PARTITION variable=filter_coeffs type=complete dim=2

		filt_cache_iChannel_loop: for(uint32_t iChannel = 0; iChannel < numChannels; ++iChannel) {
#pragma HLS LOOP_TRIPCOUNT min=LOOP_TRIPCOUNT_CHANNELS max=LOOP_TRIPCOUNT_CHANNELS
			filt_cache_cy_loop: for (uint32_t cy = 0; cy < CONV_FILTER_HEIGHT; ++ cy) {
				filt_cache_cx_loop: for (uint32_t cx = 0; cx < CONV_FILTER_WIDTH; ++cx) {
					filter_coeffs[iChannel][cy][cx] = filters[FILT_IDX(iFilter, iChannel, cy, cx)];
				}
			}
		}

		FXP_t bias = biases[iFilter];

		// Cache the three rows that we are going to read for computing a single filter row
		FXP_t row_buffers[CONV_FILTER_HEIGHT][NUM_PARALLEL_CHANNELS][MAX_ROW_BUFFER_SIZE / NUM_PARALLEL_CHANNELS];

		// Power of two cyclic factor (4 instead of 3) to avoid expensive modulo 3 to choose the correct array,
		//	and adds a noticeable iteration latency to the ichannel_loop (which has a relatively small number of
		//	iterations, so the effect is quite noticeable).
		#pragma HLS ARRAY_PARTITION variable=row_buffers type=cyclic factor=4 dim=3
		#pragma HLS ARRAY_PARTITION variable=row_buffers type=complete dim=2
		#pragma HLS ARRAY_PARTITION variable=row_buffers type=complete dim=1

		// Row the first two rows from the input image to kick-off the caching
		for(uint32_t iChannel = 0; iChannel < numChannels; ++iChannel)  {
#pragma HLS LOOP_TRIPCOUNT min=LOOP_TRIPCOUNT_CHANNELS max=LOOP_TRIPCOUNT_CHANNELS
			for(uint8_t iRow = 0; iRow < CONV_FILTER_HEIGHT-1; ++iRow) {
				for (uint32_t x = 0; x < inputWidth; ++x) {
#pragma HLS LOOP_TRIPCOUNT min=LOOP_TRIPCOUNT_INPUT_WIDTH max=LOOP_TRIPCOUNT_INPUT_WIDTH
					row_buffers[iRow][iChannel & PAR_CHAN_MASK][(iChannel >> NUM_PARALLEL_CHANNELS_SHIFT) * inputWidth + x] = input[IN_IDX(iChannel, iRow, x)];
				}
			}
		}

		uint8_t firstRowBufferIndex = 0;
		uint8_t nextRowBufferToFill = CONV_FILTER_HEIGHT - 1;

		// For each filter, compute the output[x][y] for that filter
		output_y_loop: for (uint32_t y = 0; y < (inputHeight-CONV_FILTER_HEIGHT+1); ++y) {
#pragma HLS LOOP_TRIPCOUNT min=LOOP_TRIPCOUNT_OUTPUT_HEIGHT max=LOOP_TRIPCOUNT_OUTPUT_HEIGHT

			// Fill in the appropriate row in the next row buffer to fill
			for(uint32_t iChannel = 0; iChannel < numChannels; ++iChannel)  {
#pragma HLS LOOP_TRIPCOUNT min=LOOP_TRIPCOUNT_CHANNELS max=LOOP_TRIPCOUNT_CHANNELS
					for (uint32_t x = 0; x < inputWidth; ++x) {
#pragma HLS LOOP_TRIPCOUNT min=LOOP_TRIPCOUNT_INPUT_WIDTH max=LOOP_TRIPCOUNT_INPUT_WIDTH
						row_buffers[nextRowBufferToFill][iChannel & PAR_CHAN_MASK][(iChannel >> NUM_PARALLEL_CHANNELS_SHIFT) * inputWidth + x] = input[IN_IDX(iChannel, y+2, x)];
					}
			}

			nextRowBufferToFill++;
			if(nextRowBufferToFill == CONV_FILTER_HEIGHT) {
				nextRowBufferToFill = 0;
			}

			output_x_loop: for (uint32_t x = 0; x < (inputWidth-CONV_FILTER_WIDTH+1); ++x) {
#pragma HLS LOOP_TRIPCOUNT min=LOOP_TRIPCOUNT_OUTPUT_WIDTH max=LOOP_TRIPCOUNT_OUTPUT_WIDTH

				// Generate the pixel output[y][x]

				FXP_t accs[NUM_PARALLEL_CHANNELS];
				for(uint8_t iParChannel = 0; iParChannel < NUM_PARALLEL_CHANNELS; ++iParChannel) {
#pragma HLS UNROLL
					accs[iParChannel] = 0;
				}

				ichannel_loop: for (uint16_t iChannel = 0; iChannel < numChannels; ++iChannel) {
#pragma HLS LOOP_TRIPCOUNT min=LOOP_TRIPCOUNT_CHANNELS max=LOOP_TRIPCOUNT_CHANNELS
#pragma HLS UNROLL factor=NUM_PARALLEL_CHANNELS

					uint8_t rowBufferIndex = firstRowBufferIndex;
					cy_loop: for (uint8_t cy = 0; cy < CONV_FILTER_HEIGHT; ++ cy) {
#pragma HLS UNROLL
						cx_loop: for (uint8_t cx = 0; cx < CONV_FILTER_WIDTH; ++cx) {
#pragma HLS UNROLL
							// acc += filters[iFilter][iChannel][cy][cx] * input[iChannel][y+cy][x+cx]
							FXP_t filtVal = filter_coeffs[iChannel][cy][cx];
//							FXP_t inVal = input[IN_IDX(iChannel, y+cy, x+cx)];

							FXP_t inVal = row_buffers[rowBufferIndex][iChannel & PAR_CHAN_MASK][(iChannel >> NUM_PARALLEL_CHANNELS_SHIFT)*inputWidth + (x+cx)];

							FXP_t mult = FXP_MULT_t(inVal) * FXP_MULT_t(filtVal);
							accs[iChannel & PAR_CHAN_MASK] += mult;
						}

						rowBufferIndex++;
						if (rowBufferIndex == CONV_FILTER_HEIGHT) {
							rowBufferIndex = 0;
						}
					}
				}

				FXP_t acc = 0;
				for(uint8_t iParChannel = 0; iParChannel < NUM_PARALLEL_CHANNELS; ++iParChannel) {
#pragma HLS UNROLL
					acc += accs[iParChannel];
				}

				FXP_t out = acc + bias;

				if(performReLu) {
					out = ReLu(out);
				}

				// output[iFilter][y][x] = out
				output[OUT_IDX(iFilter, y, x)] = out;
			}

			firstRowBufferIndex++;
			if (firstRowBufferIndex == CONV_FILTER_HEIGHT) {
				firstRowBufferIndex = 0;
			}
		}
	}
	resultOk = true;
	return;
}
