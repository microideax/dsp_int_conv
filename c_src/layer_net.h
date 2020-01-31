#include <iostream>
#include <stdint.h>
#include "dsp_conv_int.h"
#include <ap_int.h>

using namespace std;

dsp_conv_int<int16_t, int16_t, int16_t, int16_t, int16_t, 16, 4, 8, 8, 1, 3, 10, 16, 16> conv_layer_acc;

void conv_layer_proc(
    int *param_port,
    int16_t *bias_port,
#if IN_PORT_WIDTH == 64
    ap_uint<64> *weight_in,
    ap_uint<64> *data_in,
	ap_uint<64> *data_out,
#else
    int16_t *weight_in,
    int16_t *data_in,
	int16_t *data_out,
#endif
    bool dsp_clk)
{

#pragma HLS inline off
  // #pragma HLS dataflow

  int param_conv_local[16];
  // int param_pool_local[16];

  cout << "LAYER ACC: CONV Loading layer number for current accelerator ..." << endl;
  for (unsigned int i = 0; i < 16; i++)
  {
#pragma HLS pipeline
    param_conv_local[i] = param_port[i];
  }
  conv_layer_acc.conv_core_1i1o(param_conv_local[0],  // N
                                param_conv_local[1],  // K
                                param_conv_local[2],  // M
                                param_conv_local[3],  // Rin
                                param_conv_local[4],  // C
                                param_conv_local[5],  // R
                                param_conv_local[6],  // C
                                param_conv_local[7],  // S
                                param_conv_local[8],  // P
                                param_conv_local[9],  // act
                                param_conv_local[10], //inport
                                param_conv_local[11], // w_offset
                                param_conv_local[12], // b_offset
                                param_conv_local[13], // in_offset
                                param_conv_local[14], // out_offset
                                bias_port,
                                weight_in,
                                data_in,
                                data_out,
                                dsp_clk);
};

void sub_net_proc(
    int param_port[32],
    int16_t bias_port[256],
#if IN_PORT_WIDTH == 64
    ap_uint<64> weight_in[64],
    ap_uint<64> data_in[1024],
	ap_uint<64> data_out[1024],
#else
    int16_t weight_in[256],
    int16_t data_in[1024],
	int16_t data_out[1024],
#endif
    bool dsp_clk)
{

#pragma HLS INTERFACE s_axilite port = return bundle = CRTL_BUS

#pragma HLS INTERFACE BRAM port = param_port
#pragma HLS INTERFACE s_axilite port = weight_in bundle = CRTL_BUS
#pragma HLS INTERFACE m_axi port = weight_in offset = slave depth = 20300 bundle = W_IN
#pragma HLS INTERFACE m_axi port = bias_port offset = slave depth = 256 bundle = B_IN

#pragma HLS INTERFACE s_axilite port = data_in bundle = CRTL_BUS
#pragma HLS INTERFACE m_axi port = data_in offset = slave depth = 1024 bundle = D_IN
#pragma HLS INTERFACE m_axi port = data_out offset = slave depth = 1024 bundle = F_OUT

#pragma HLS INTERFACE ap_stable port = dsp_clk

	cout << "Accelerator configuration: Tm=" <<conv_layer_acc.accTm << " ,Tn="<< conv_layer_acc.accTn << " ,Tr="<< conv_layer_acc.accTr << " ,Tc="<< conv_layer_acc.accTc << endl;

  conv_layer_proc(param_port,
                  bias_port,
                  weight_in,
                  data_in,
                  data_out,
                  dsp_clk);
};
