#include <iostream>
#include <stdint.h>
#include "dsp_conv_int.h"
#include <ap_int.h>

using namespace std;

//#define IN_PORT_WIDTH 64

dsp_conv_int<int16_t, int16_t, int16_t, int16_t, int16_t, 4, 4, 8, 8, 1, 3, 10, 16, 16> conv_layer_acc;

void conv_layer_proc(
    int *param_port,
    int16_t *bias_port,
#if IN_PORT_WIDTH == 64
    ap_uint<64> *weight_in,
    ap_uint<64> *data_in,
#else
    int16_t *weight_in,
    int16_t *data_in,
#endif
    ap_uint<64> *data_out,
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
#else
    int16_t weight_in[256],
    int16_t data_in[1024],
#endif
    ap_uint<64> data_out[1024],
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

  conv_layer_proc(param_port,
                  bias_port,
                  weight_in,
                  data_in,
                  data_out,
                  dsp_clk);
};

int main()
{

  int16_t in_buf[4][10][10];
#if IN_PORT_WIDTH == 64
  ap_uint<64> in_port[4096];
  ap_uint<64> w_port[256];
#else
  int16_t in_port[1024];
  int16_t w_port[256];
#endif
  ap_uint<64> i_tmp_buf = ap_uint<64>(0);
  ap_uint<64> w_tmp_buf = ap_uint<64>(0);
  int16_t w_buf[4][4][3][3];
  // int16_t b_buf[4];
  int16_t b_port[256];
  bool dsp_clk;
  int32_t out_buf[4][8][8];
  ap_uint<64> out_port[1024];

  int param[16] = {
      4,
      3,
      8,
      10,
      10,
      8,
      8,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0};

  //init in_buf and b_buf
  for (int j = 0; j < param[3]; j++)
  {
    for (int k = 0; k < param[4]; k++)
    {
      for (int n_dim = 0; n_dim < param[0]; n_dim += 4)
      {
        if (n_dim < 4)
        {
          for (int i = 0; i < 4; i++)
          {
#if IN_PORT_WIDTH == 64
            i_tmp_buf.range(16 * i + 15, 16 * i) = int16_t(k);
            // cout << "in_port data fill: " << n_dim << " " << int16_t(k) << endl;
            //    	in_buf[i][j][k] = int16_t(k);
          }
        }
        else
        {
          i_tmp_buf = 0;
        }
        *(in_port + n_dim / 4 * param[3] * param[4] + j * param[3] + k) = i_tmp_buf;
#else
            *(in_port + i * 100 + j * 10 + k) = int16_t(k);
#endif
      }
    }
  }

  // cout << "inport data: " << endl;
  // for (int idx = 0; idx < 400; idx++)
  // {
  //   cout << *(in_port + idx) << endl;
  // }

  for (int i = 0; i < 256; i++)
  {
    b_port[i] = i;
  }

  // init out_buf to 0s
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 8; j++)
    {
      for (int k = 0; k < 8; k++)
      {
        *(out_port + i * 64 + j * 8 + k) = int16_t(0);
      }
    }
  }

  // init w_buf with int16_t values

  for (int j = 0; j < 4; j++)
  {
    for (int k = 0; k < 3; k++)
    {
      for (int l = 0; l < 3; l++)
      {
#if IN_PORT_WIDTH == 64
        for (int i = 0; i < 4; i++)
        {
          w_tmp_buf.range(16 * i + 15, 16 * i) = int16_t(l);
        }
        w_port[j * 9 + k * 3 + l] = w_tmp_buf;
#else
            for (int i = 0; i < 4; i++)
            {
              w_buf[i][j][k][l] = int16_t(l);
              w_port[i * 36 + j * 9 + k * 3 + l] = int16_t(l);
            }
#endif
      }
    }
  }

  sub_net_proc(param, b_port, w_port, in_port, out_port, dsp_clk);

  // display b_buf
  cout << endl;
  cout << endl;
  cout << "main: The bias buffer:" << endl;
  for (int i = 0; i < param[2]; i++)
  {
    cout << b_port[i] << "  ";
  }
  cout << endl;

  // display output buffer
  cout << endl;
  cout << endl;
  cout << "main: The output buffer:" << endl;
  for (int i = 0; i < param[2]; i++)
  {
    for (int j = 0; j < param[6]; j++)
    {
      for (int k = 0; k < param[6]; k++)
      {
        cout << *(out_port + i * param[6] * param[6] + j * param[6] + k) << "  ";
      }
      cout << endl;
    }
    cout << endl;
  }

  return 0;
};
