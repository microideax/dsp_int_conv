#ifndef _DSP_CONV_INT_H_
#define _DSP_CONV_INT_H_

#include <iostream>
#include <dsp_builtins.h>
#include "ap_int.h"
#include <stdint.h>

using namespace std;
//#define __HLS_SYN__ 1
#define IN_PORT_WIDTH 64

template <typename Itf, typename Tparam, typename T, typename W, typename G, int Tm, int Tn, int Tr, int Tc, int S_max, int K_max, int IBUF_t, int WBUF_t, int OBUF_t>
class dsp_conv_int
{

public:
  dsp_conv_int() { ; }

  ////-----------------------------Accelerator Functions---------------------------------------////
  // dsp computation cores
  void mm_dsp_mm(int16_t A1, int16_t A2, int16_t B1, int16_t B2, int32_t *C, bool clear, bool ap_clk_div2)
  {
#pragma HLS INLINE OFF
    ap_int<42> accum_tmp = 0;
    accum_tmp.range(31, 0) = *C;
    // cout << "Pass in initial value: " << *C << endl;

    //#ifdef __HLS_SYN__
    // cout << "HLS C execution ..." << endl;
    accum_tmp = __builtin_mac16x2(A1, A2, B1, B2, accum_tmp, clear, ap_clk_div2);
    //#else
    // cout << "Regular C execution ..." << endl;
    //    accum_tmp = A1 * B1 + A2 * B2 + (clear ? 0 : accum_tmp);
    //#endif
    *C = accum_tmp.range(31, 0);
    // cout << "Calculated value: " << *C << endl;
  };

  // HLS register initialize
  void reg_to_reg(int16_t *reg_in, int16_t *reg_out)
  {
#pragma HLS inline off
#pragma HLS interface register port = reg_in

    *reg_out = *reg_in;
  }

  // Load bias data
  void b_buf_load(int16_t buf[], int16_t *layer_bias, int bias_offset, int m)
  {
#pragma HLS inline off
    for (ap_uint<8> i = 0; i < Tm; i++)
    {
      buf[i] = *(layer_bias + bias_offset + m + i);
      //cout << "Read bias location: " << bias_offset + i + m << "  Read bias data: " << buf[i].range(15,0)<< endl;
    }
  };

  void in_buf_load(
      int16_t buf_0[Tn][IBUF_t][IBUF_t],
#if IN_PORT_WIDTH == 64
      ap_uint<64> *i_data,
#else
      int16_t *i_data,
#endif
      ap_uint<8> in_offset,
      int n,
      int r,
      int c,
      int S,
      int K,
      int P,
      int R_IN,
      int C_IN,
      ap_uint<8> N)
  {
#pragma HLS inline off
    ap_uint<8> idx_n = 0;
    ap_uint<8> idx_r = 0;
    ap_uint<8> idx_k = 0;
    ap_uint<8> idx_f = 0;
    ap_uint<8> i;
    ap_uint<8> j;
    ap_uint<4> k;
    ap_uint<8> d_idx;
    ap_uint<8> idx_tn;
    ap_uint<8> idx_tnn;
    // int idx_z = 0;

    idx_tn = n >> 2;
    idx_tnn = (n + Tn) >> 2;
    ap_uint<64> local_i_buf = 0;
    for (i = idx_tn; i < idx_tnn; i++)
    {
      // cout << endl;
      // cout << "in buf load n, r, c, R_IN, C_IN:  " << i << "  " << r << "  " << c << "  " << R_IN << "  " << C_IN << endl;
      // cout << "in buf load feature number count: " << i << endl;
      idx_n = i * R_IN * C_IN;
      for (j = 0; j < IBUF_t; j++)
      {
        idx_r = (r + j) * C_IN;
        // cout << "in buf load R dim counter: " << (r + j) << " and " << idx_r << endl;
        for (k = 0; k < IBUF_t; k++)
        {
          idx_k = c + k;
          // cout << "in buf load C dim counter: " << idx_k << endl;
#pragma HLS PIPELINE
          // for (idx_f = 0; idx_f < Tn / 4; idx_f++)
          // {
          // idx_f = idx_n + idx_f * R_IN * C_IN;
          local_i_buf.range(63, 0) = *(i_data + idx_n + idx_r + idx_k + in_offset);
          for (d_idx = 0; d_idx < 4; d_idx++)
          {
#pragma HLS UNROLL
            buf_0[d_idx][j][k] = local_i_buf.range(16 * d_idx + 15, 16 * d_idx);
            // cout << "Fill local buffer: " << idx_n << " " << idx_r << " " << idx_k << " " << buf_0[d_idx][j][k] << "  " << local_i_buf << endl;
          }
          // }
        }
      }
    }
  };

  // Load weight
  void w_buf_load(int16_t buf[][Tm][K_max][K_max],
#if IN_PORT_WIDTH == 64
                  ap_uint<64> *layer_weights,
#else
                  int16_t *layer_weights,
#endif
                  int weight_offset,
                  int n, int m,
                  ap_uint<4> K,
                  ap_uint<8> N,
                  ap_uint<8> M)
  {
#pragma HLS inline off
    ap_uint<8> idx_k = 0;
    ap_uint<8> idx_m = 0;
    ap_uint<4> k1;
    ap_uint<4> k2;
    ap_uint<8> j;
    ap_uint<8> i;
    ap_uint<3> i_idx;

    ap_uint<64> w_tmp_buf = 0;

    for (k1 = 0; k1 < K; k1++)
    {
      for (k2 = 0; k2 < K; k2++)
      {
        for (j = 0; j < Tm; j++)
        {
#pragma HLS PIPELINE
#if IN_PORT_WIDTH == 64
          idx_k = j * K * K;
          for (i = 0; i < Tn; i += 4)
          {
            idx_m = i * j * K * K;
            w_tmp_buf = *(layer_weights + weight_offset + idx_k + idx_m + k1 * K + k2);
            for (i_idx = 0; i_idx < 4; i_idx++)
            {
#pragma HLS UNROLL
              buf[i_idx][j][k1][k2] = w_tmp_buf.range(16 * i_idx + 15, 16 * i_idx);
            }
#else
          for (i = 0; i < Tn; i++)
          {
            buf[i][j][k1][k2] = *(layer_weights + weight_offset + j * Tm * K * K + i * K * K + k1 * K + k2);
          }
#endif
          }
        }
      }
    }
  };

  //   void output_init(ap_fixed<16, 10> out_buf[][Tr][Tc], ap_fixed<16, 10> b_buf[])
  //   {
  //     for (int tm = 0; tm < Tm; tm++)
  //     {
  // #pragma HLS unroll
  //       for (int i = 0; i < Tr; i++)
  //       {
  //         for (int j = 0; j < Tc; j++)
  //         {
  // #pragma HLS pipeline
  //           out_buf[tm][i][j] = b_buf[tm];
  //         }
  //       }
  //     }
  //   }

  // Convolution computation kernel Tm, Tn based
  void conv_engine(
      int16_t in_buf[Tn][IBUF_t][IBUF_t],
      int16_t w_buf[Tn][Tm][K_max][K_max],
      int16_t b_buf[Tm],
      int16_t out_buf[Tm][Tr][Tc],
      int S,
      int n,
      int N,
      int r,
      int c,
      int K,
      int R_OUT,
      int C_OUT,
      int w_offset,
      int i_offset,
      bool dsp_clk)
  {
#pragma HLS inline off

    ap_uint<12> i_tmp;
    ap_uint<12> f_h_tmp;
    ap_uint<12> f_w_tmp;
    ap_uint<4> ker_0;
    ap_uint<4> ker_1;
    ap_uint<8> tr;
    ap_uint<8> tc;
    ap_uint<8> tm;
    ap_uint<4> tn;
    ap_uint<4> i_index;
    ap_uint<8> j_index;

    int16_t a1in_i;
    int16_t a2in_i;
    int16_t b1in_i;
    int16_t b2in_i;
    int16_t mac_i[4];
#pragma HLS ARRAY_PARTITION variable = mac_i complete dim = 1

    int16_t a1in_o;
    int16_t a2in_o;
    int16_t b1in_o;
    int16_t b2in_o;

    int16_t accum_out = 0;
    int32_t mac_tmp = 0;
    bool clear = 0;

    // int16_t i_tmp_0[Tn][(Tc - 1) * S_max + K_max];
    // int16_t i_tmp_1[Tn][(Tc - 1) * S_max + K_max];
    int16_t i_tmp_0[Tn][IBUF_t];
    int16_t i_tmp_1[Tn][IBUF_t];
#pragma HLS ARRAY_PARTITION variable = i_tmp_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = i_tmp_1 complete dim = 1
    // #pragma HLS ARRAY_PARTITION variable = i_tmp_0 complete dim = 2
    // #pragma HLS ARRAY_PARTITION variable = i_tmp_1 complete dim = 2

    if (n >= 0 && n - Tn < N)
    {
      cout << "Executing conv engine at n = " << n << endl;
      for (ker_0 = 0; ker_0 < K; ker_0++)
      {
        i_tmp = ker_0 + w_offset;
        for (ker_1 = 0; ker_1 < K; ker_1++)
        {
          for (tr = 0; tr < Tr; tr++)
          {
            // #pragma HLS PIPELINE
            f_h_tmp = S * tr + ker_0 + i_offset;
            // #pragma HLS PIPELINE
            // for (int j_index = 0; j_index < (Tc - 1) * S_max + K_max; j_index++)

            for (i_index = 0; i_index < Tn / 2; i_index++)
            {
#pragma HLS UNROLL
              for (j_index = 0; j_index < IBUF_t; j_index++)
              {
                i_tmp_0[i_index][j_index] = in_buf[2 * i_index][f_h_tmp][j_index];
                i_tmp_1[i_index][j_index] = in_buf[2 * i_index + 1][f_h_tmp][j_index];
              }
            }
#pragma HLS PIPELINE
            for (tc = 0; tc < Tc; tc++)
            {
              f_w_tmp = S * (tc) + ker_1;
              for (tm = 0; tm < Tm; tm++)
              {
#pragma HLS UNROLL
                for (tn = 0; tn < Tn; tn = tn + 2)
                {
#pragma HLS UNROLL
                  a1in_i = w_buf[tn][tm][i_tmp][ker_1];
                  a2in_i = w_buf[tn + 1][tm][i_tmp][ker_1];
                  b1in_i = i_tmp_0[tn >> 1][f_w_tmp];
                  b2in_i = i_tmp_1[tn >> 1][f_w_tmp];
                  /*
                  mac_i[0] = w_buf[tn][tm][i_tmp][ker_1];
                  mac_i[1] = w_buf[tn + 1][tm][i_tmp][ker_1];
                  mac_i[2] = i_tmp_0[tn >> 1][f_w_tmp];
                  mac_i[3] = i_tmp_1[tn >> 1][f_w_tmp];
                  */
                  if (ker_0 == 0 && ker_1 == 0 && tn == 0 && n == 0)
                  {
                    clear = 0;
                    mac_tmp = b_buf[tm];
                    // cout << "Initial output buffer as bias value: " << accum_tmp << endl;
                  }
                  else
                  {
                    clear = 0;
                    mac_tmp = out_buf[tm][tr][tc];
                  }
                  // mm_dsp_mm(mac_i[0], mac_i[1], mac_i[2], mac_i[3], &mac_tmp, clear, dsp_clk);
                  mm_dsp_mm(a1in_i, b1in_i, a2in_i, b2in_i, &mac_tmp, clear, dsp_clk);
                  if (mac_tmp >= 16384)
                  {
                    accum_out = 16384;
                  }
                  else
                  {
                    accum_out = mac_tmp;
                  }
                  out_buf[tm][tr][tc] = accum_out;
                }
              }
            }
          }
        }
      }
    }
    else
    {
      cout << "Jump current conv_engine at n =" << n << endl;
    }
  };

  // Ouput out_buf data to output interface
  /*  void output_res(int16_t out_buf[][Tr][Tc],
                  int16_t *out_data,
                  int out_offset,
                  int n,
                  int m,
                  int r,
                  int c,
                  int N,
                  int M,
                  int R_OUT,
                  int C_OUT,
                  bool act)
  {
#pragma HLS inline off
    // if (n >= N - Tn && m > 0)
    if (n >= N - Tn && n < N && m >= 0)
    {
      cout << "output buffer data: " << n << " " << m << endl;
      for (int tm = 0; tm < Tm; tm++)
      {
        for (int tr = 0; tr < Tr; tr++)
        {
          for (int tc = 0; tc < Tc; tc++)
          {
#pragma HLS PIPELINE
            *(out_data + out_offset + tm * Tr * Tc + tr * Tc + tc) = out_buf[tm][tr][tc];
          }
        }
      }
    }
    else
    {
      cout << "Skipping output buffer due to initial empty " << endl;
    }
  };*/

  void output_res(int16_t out_buf[][Tr][Tc],
                  //int16_t *out_data,
                  ap_uint<64> *out_data,
                  int out_offset,
                  int n,
                  int m,
                  int r,
                  int c,
                  int N,
                  int M,
                  int R_OUT,
                  int C_OUT,
                  bool act)
  {

#pragma HLS inline off

    ap_uint<64> local_o_buf = 0;
    ap_uint<8> idx_tm;
    ap_uint<5> tr;
    ap_uint<5> tc;
    ap_uint<8> d_idx;

    idx_tm = m >> 2;
    // if (n >= N - Tn && m > 0)
    if (n >= N - Tn && n < N && m >= 0)
    {
      cout << "output buffer data: " << n << " " << m << endl;
      for (tr = 0; tr < Tr; tr++)
      {
        for (tc = 0; tc < Tc; tc++)
        {
#pragma HLS PIPELINE
          //*(out_data + out_offset + tm * Tr * Tc + tr * Tc + tc) = out_buf[tm][tr][tc];

          for (d_idx = 0; d_idx < 4; d_idx++)
          {
#pragma HLS UNROLL
            local_o_buf.range(16 * d_idx + 15, 16 * d_idx) = out_buf[d_idx][tr][tc];
            // cout << "Fill local buffer: " << idx_n << " " << idx_r << " " << idx_k << " " << buf_0[d_idx][j][k] << "  " << local_i_buf << endl;
          }
          *(out_data + out_offset + idx_tm * Tr * Tc + tr * Tc + tc) = local_o_buf.range(63, 0);
        }
      }
    }
    else
    {
      cout << "Skipping output buffer due to initial empty " << endl;
    }
  };

  void print_inputbuf(
      int16_t array_name[][IBUF_t][IBUF_t],
      int dim_n,
      int dim_r,
      int dim_c)
  {
    // display in_buf
    cout << endl;
    cout << endl;
    cout << "Printing input buffer" << endl;
    for (int i = 0; i < dim_n; i++)
    {
      for (int j = 0; j < dim_r; j++)
      {
        for (int k = 0; k < dim_c; k++)
        {
          cout << array_name[i][j][k] << "  ";
        }
        cout << endl;
      }
      cout << endl;
    }
  };

  void conv_core_1i1o(
      int N,     //input feature number
      int K,     //input kernel size
      int M,     // output feature number
      int R_IN,  // input Row
      int C_IN,  // input column
      int R_OUT, // output Row
      int C_OUT, // output column
      int S,     // stride size
      int P,     // padding size
      int act,   // activation function bit (1-- with act, 0--without act)
      int inport,
      int weight_offset,
      int bias_offset,
      int in_offset,
      int out_offset,
      int16_t *i_bias,
#if IN_PORT_WIDTH == 64
      ap_uint<64> *i_weight,
      ap_uint<64> *i_data,
#else
    int16_t *i_weight,
    int16_t *i_data,
#endif
      int16_t *out_data,
      bool clk2)
  {

    //#pragma HLS inline off
    //local data buffer groups
    T in_buf_0[Tn][(Tr - 1) * S_max + K_max][(Tc - 1) * S_max + K_max];
    W w_buf_0[Tn][Tm][K_max][K_max];
    W b_buf_0[Tm];

    G out_buf_0[Tm][Tr][Tc];

    //    bool clk2;
#pragma HLS ARRAY_PARTITION variable = in_buf_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = w_buf_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = w_buf_0 complete dim = 2
#pragma HLS ARRAY_PARTITION variable = b_buf_0 complete

#pragma HLS ARRAY_PARTITION variable = out_buf_0 complete dim = 1

    //--------------------------Initial data load ---------------------------------------------//
    for (int r = 0; r < R_OUT; r += Tr)
    {
      for (int c = 0; c < C_OUT; c += Tc)
      {
        for (int m = 0; m < M; m += Tm)
        {
          for (int n = 0; n < N; n += Tn)
          {
            cout << "load buffer set 0" << endl;
            b_buf_load(b_buf_0, i_bias, bias_offset, m);
            w_buf_load(w_buf_0, i_weight, weight_offset, n, m, K, N, M);
            cout << "Loading location: " << n << " " << r << " " << c << " " << N << " " << endl;
            in_buf_load(in_buf_0, i_data, in_offset, n, r, c, S, K, P, R_IN, C_IN, N);
            print_inputbuf(in_buf_0, Tn, IBUF_t, IBUF_t);
            cout << "Process buffer set 0" << endl;
            conv_engine(in_buf_0, w_buf_0, b_buf_0, out_buf_0, S, n, N, r, c, K, R_OUT, C_OUT, 0, 0, clk2);
          }
          output_res(out_buf_0, out_data, out_offset, N, m, r, c, N, M, R_OUT, C_OUT, act);
        }
      }
    }
  };

  void conv_core_2i1o(
      int N,     //input feature number
      int K,     //input kernel size
      int M,     // output feature number
      int R_IN,  // input Row
      int C_IN,  // input column
      int R_OUT, // output Row
      int C_OUT, // output column
      int S,     // stride size
      int P,     // padding size
      int act,   // activation function bit (1-- with act, 0--without act)
      int inport,
      int weight_offset,
      int bias_offset,
      int in_offset,
      int out_offset,
      int16_t *i_bias,
#if IN_PORT_WIDTH == 64
      ap_uint<64> *i_weight,
      ap_uint<64> *i_data,
#else
    int16_t *i_weight,
    int16_t *i_data,
#endif
      int16_t *out_data,
      bool clk2)
  {

#pragma HLS inline off
    //local data buffer groups
    T in_buf_0[Tn][(Tr - 1) * S_max + K_max][(Tc - 1) * S_max + K_max];
    W w_buf_0[Tn][Tm][K_max][K_max];
    W b_buf_0[Tm];

    T in_buf_1[Tn][(Tr - 1) * S_max + K_max][(Tc - 1) * S_max + K_max];
    W w_buf_1[Tn][Tm][K_max][K_max];
    W b_buf_1[Tm];

    G out_buf_0[Tm][Tr][Tc];
    G out_buf_1[Tm][Tr][Tc];

    // buffer pointer group
    T in_ptr[Tn][(Tr - 1) * S_max + K_max][(Tc - 1) * S_max + K_max];
    W w_ptr[Tn][Tm][K_max][K_max];
    W b_ptr[Tm];
    G o_ptr[Tm][Tr][Tc];

    bool com_ptr = 0;
    ap_uint<8> loop_counter_n;
    ap_uint<8> loop_counter_m;
    //    bool clk2;
#pragma HLS allocation instances = conv_engine limit = 1 function

#pragma HLS ARRAY_PARTITION variable = in_buf_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = w_buf_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = w_buf_0 complete dim = 2
#pragma HLS ARRAY_PARTITION variable = b_buf_0 complete
#pragma HLS ARRAY_PARTITION variable = in_buf_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = w_buf_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = w_buf_1 complete dim = 2
#pragma HLS ARRAY_PARTITION variable = b_buf_1 complete

#pragma HLS ARRAY_PARTITION variable = out_buf_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = out_buf_1 complete dim = 1

    //--------------------------Initial data load ---------------------------------------------//
    for (int r = 0; r < R_OUT; r += Tr)
    {
      for (int c = 0; c < C_OUT; c += Tc)
      {
        loop_counter_m = 0;
        for (int m = 0; m < M; m += Tm)
        {
          loop_counter_n = 0;
          for (int n = 0; n < N + Tn; n += Tn)
          {
            cout << "loop counter number:" << loop_counter_n << endl;
            if (loop_counter_n[0] == 0)
            {
              cout << "load buffer set 0" << endl;
              b_buf_load(b_buf_0, i_bias, bias_offset, m);
              w_buf_load(w_buf_0, i_weight, weight_offset, n, m, K, N, M);
              in_buf_load(in_buf_0, i_data, in_offset, n, r, c, S, K, P, R_IN, C_IN, N);
              // print_inputbuf(in_buf_0, 4, 10, 10);
              cout << "Process buffer set 1" << endl;
              conv_engine(in_buf_1, w_buf_1, b_buf_1, out_buf_0, S, n - Tn, N, r, c, K, R_OUT, C_OUT, 0, 0, clk2);
            }
            else
            {
              cout << "load buffer set 1" << endl;
              b_buf_load(b_buf_1, i_bias, bias_offset, m);
              w_buf_load(w_buf_1, i_weight, weight_offset, n, m, K, N, M);
              in_buf_load(in_buf_1, i_data, in_offset, n, r, c, S, K, P, R_IN, C_IN, N);
              // print_inputbuf(in_buf_0, 4, 10, 10);
              cout << "Process buffer set 0" << endl;
              conv_engine(in_buf_0, w_buf_0, b_buf_0, out_buf_0, S, n - Tn, N, r, c, K, R_OUT, C_OUT, 0, 0, clk2);
            }
            loop_counter_n++;
          }
          for (int index = 0; index < 8; index++)
          {
            cout << out_buf_0[0][0][index] << " ";
          }
          cout << endl;
          output_res(out_buf_0, out_data, out_offset, N, m, r, c, N, M, R_OUT, C_OUT, act);
          loop_counter_m++;
        }
      }
    }
  };

  ///------------------conv accelerator----------------///
  void conv_core_2i2o(
      int N,     //input feature number
      int K,     //input kernel size
      int M,     // output feature number
      int R_IN,  // input Row
      int C_IN,  // input column
      int R_OUT, // output Row
      int C_OUT, // output column
      int S,     // stride size
      int P,     // padding size
      int act,   // activation function bit (1-- with act, 0--without act)
      int inport,
      int weight_offset,
      int bias_offset,
      int in_offset,
      int out_offset,
      int16_t *i_bias,
#if IN_PORT_WIDTH == 64
      ap_uint<64> *i_weight,
      ap_uint<64> *i_data,
#else
    int16_t *i_weight,
    int16_t *i_data,
#endif
      ap_uint<64> *out_data,
      bool clk2)
  {

    //#pragma HLS inline off
    //local data buffer groups
    // T in_buf_0[Tn][(Tr - 1) * S_max + K_max][(Tc - 1) * S_max + K_max];
    T in_buf_0[Tn][IBUF_t][IBUF_t];
    W w_buf_0[Tn][Tm][K_max][K_max];
    W b_buf_0[Tm];

    // T in_buf_1[Tn][(Tr - 1) * S_max + K_max][(Tc - 1) * S_max + K_max];
    T in_buf_1[Tn][IBUF_t][IBUF_t];
    W w_buf_1[Tn][Tm][K_max][K_max];
    W b_buf_1[Tm];

    G out_buf_0[Tm][Tr][Tc];
    G out_buf_1[Tm][Tr][Tc];

    // T in_ptr[Tn][(Tr - 1) * S_max + K_max][(Tc - 1) * S_max + K_max];
    // W ****w_ptr;
    // W *b_ptr;
    // G ***o_ptr;

    bool com_ptr = 0;
    ap_uint<8> loop_counter_n;
    ap_uint<8> loop_counter_m;
    //    bool clk2;

#pragma HLS resource variable = in_buf_0 core = RAM_2P
#pragma HLS ARRAY_PARTITION variable = in_buf_0 complete dim = 1
// #pragma HLS ARRAY_PARTITION variable = in_buf_0 block factor = 2 dim = 2
#pragma HLS ARRAY_PARTITION variable = w_buf_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = w_buf_0 complete dim = 2
#pragma HLS ARRAY_PARTITION variable = b_buf_0 complete

#pragma HLS resource variable = in_buf_1 core = RAM_2P
#pragma HLS ARRAY_PARTITION variable = in_buf_1 complete dim = 1
// #pragma HLS ARRAY_PARTITION variable = in_buf_1 block factor = 2 dim = 2
#pragma HLS ARRAY_PARTITION variable = w_buf_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = w_buf_1 complete dim = 2
#pragma HLS ARRAY_PARTITION variable = b_buf_1 complete

#pragma HLS ARRAY_PARTITION variable = out_buf_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = out_buf_1 complete dim = 1

    //--------------------------Initial data load ---------------------------------------------//
    for (int r = 0; r < R_OUT; r += Tr)
    {
      for (int c = 0; c < C_OUT; c += Tc)
      {
        loop_counter_m = 0;
        for (int m = 0; m < M; m += Tm)
        {
          cout << "LOOP_M: M loop counter: " << loop_counter_m << endl;
          if (loop_counter_m % 2 == 0)
          {
            cout << "LOOP_M: Set out buf 1" << endl;
            loop_counter_n = 0;
            // cout << "LOOP_M: Offloading out buf 1" << endl;
            // output_res(out_buf_1, out_data, out_offset, N, m - Tm, r, c, N, M, R_OUT, C_OUT, act);
            for (int n = 0; n < N + Tn; n += Tn)
            {
              cout << "----LOOP_N: loop counter number:" << loop_counter_n << endl;
              if (loop_counter_n % 2 == 0)
              {
                // if (n < N)
                // {
                cout << "----LOOP_N: load input buffer set 0" << endl;
                b_buf_load(b_buf_0, i_bias, bias_offset, m);
                w_buf_load(w_buf_0, i_weight, weight_offset, n, m, K, N, M);
                in_buf_load(in_buf_0, i_data, in_offset, n, r, c, S, K, P, R_IN, C_IN, N);
                // }
                // else
                // {
                // cout << "Skip load input buffer set 0 " << endl;
                // }
                // print_inputbuf(in_buf_0, 4, 10, 10);
                cout << "----LOOP_N: Process input buffer set 1:   ";
                conv_engine(in_buf_1, w_buf_1, b_buf_1, out_buf_0, S, n - Tn, N, r, c, K, R_OUT, C_OUT, 0, 0, clk2);
                cout << "----LOOP_N: Output out buf 1" << endl;
                output_res(out_buf_1, out_data, out_offset, n, m - Tm, r, c, N, M, R_OUT, C_OUT, act);
              }
              else
              {
                // if (n < N)
                // {
                cout << "----LOOP_N: load input buffer set 1" << endl;
                b_buf_load(b_buf_1, i_bias, bias_offset, m);
                w_buf_load(w_buf_1, i_weight, weight_offset, n, m, K, N, M);
                in_buf_load(in_buf_1, i_data, in_offset, n, r, c, S, K, P, R_IN, C_IN, N);
                // }
                // else
                // {
                // cout << "Skip load input buffer set 1 " << endl;
                // }
                // print_inputbuf(in_buf_0, 4, 10, 10);
                cout << "----LOOP_N: Process buffer set 0:   ";
                conv_engine(in_buf_0, w_buf_0, b_buf_0, out_buf_0, S, n - Tn, N, r, c, K, R_OUT, C_OUT, 0, 0, clk2);
                cout << "----LOOP_N: Output out buf 1" << endl;
                output_res(out_buf_1, out_data, out_offset, n, m - Tm, r, c, N, M, R_OUT, C_OUT, act);
              }
              loop_counter_n++;
            }
          }
          else
          {
            cout << "LOOP_M: Set out buf 0" << endl;
            loop_counter_n = 0;
            // cout << "LOOP_M: Offloading out buf 0" << endl;
            // output_res(out_buf_0, out_data, out_offset, N, m - Tm, r, c, N, M, R_OUT, C_OUT, act);
            for (int n = 0; n < N + Tn; n += Tn)
            {
              cout << "----LOOP_N: loop counter number:" << loop_counter_n << endl;
              if (loop_counter_n % 2 == 0)
              {
                // if (n < N)
                // {
                cout << "----LOOP_N: load input buffer set 0" << endl;
                b_buf_load(b_buf_0, i_bias, bias_offset, m);
                w_buf_load(w_buf_0, i_weight, weight_offset, n, m, K, N, M);
                in_buf_load(in_buf_0, i_data, in_offset, n, r, c, S, K, P, R_IN, C_IN, N);
                // }
                // else
                // {
                // cout << "Skip load input buffer set 0" << endl;
                // }
                // print_inputbuf(in_buf_0, 4, 10, 10);
                cout << "----LOOP_N: Process input buffer set 1:    ";
                conv_engine(in_buf_1, w_buf_1, b_buf_1, out_buf_1, S, n - Tn, N, r, c, K, R_OUT, C_OUT, 0, 0, clk2);
                cout << "----LOOP_N: Output out buf 0" << endl;
                output_res(out_buf_0, out_data, out_offset, n, m - Tm, r, c, N, M, R_OUT, C_OUT, act);
              }
              else
              {
                cout << "----LOOP_N: load input buffer set 1" << endl;
                b_buf_load(b_buf_1, i_bias, bias_offset, m);
                w_buf_load(w_buf_1, i_weight, weight_offset, n, m, K, N, M);
                in_buf_load(in_buf_1, i_data, in_offset, n, r, c, S, K, P, R_IN, C_IN, N);
                // print_inputbuf(in_buf_0, 4, 10, 10);
                cout << "----LOOP_N: Process input buffer set 0:   ";
                conv_engine(in_buf_0, w_buf_0, b_buf_0, out_buf_1, S, n - Tn, N, r, c, K, R_OUT, C_OUT, 0, 0, clk2);
                cout << "----LOOP_N: Output out buf 0" << endl;
                output_res(out_buf_0, out_data, out_offset, n, m - Tm, r, c, N, M, R_OUT, C_OUT, act);
              }
              loop_counter_n++;
            }
          }
          loop_counter_m++;
        }
        // TODO: add a check statement based on the loop_counter_m.
        cout << "LOOP_M: Last output buffer check" << endl;
        if ((loop_counter_m - 1) % 2 == 1)
        {
          output_res(out_buf_1, out_data, out_offset, N - Tn, M, r, c, N, M, R_OUT, C_OUT, act);
        }
      }
    }
  };
};

#endif
