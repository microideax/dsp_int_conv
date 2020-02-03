#ifndef CONV_VALIDATE_H
#define CONV_VALIDATE_H
// #include "/opt/Xilinx/Vivado/2018.1/include/gmp.h"
// #include "/opt/Xilinx/Vivado/2018.1/include/mpfr.h"
#include "ap_fixed.h"
#include "ap_int.h"
#include "dsp_conv_int.h"

class conv_validate
{
public:
  //  int layer_num;
  int *param_list;

#if IN_PORT_WIDTH == 64
  ap_uint<64> w_port[1024];
  ap_uint<64> i_port[4096];
  ap_uint<64> o_port[4096];
  int16_t b_port[1024];
#else
#define IN_PORT_WIDTH == 512
  ap_int<512> w_port[16384];
  ap_int<512> i_port[6400];
  ap_int<512> o_port[4096];
  ap_int<16> b_port[1024];
#endif

  int N, K, M, Ri, Ci, R, C, S, P, act;

  int16_t w_buf_software[8][16][5][5];
  int16_t b_buf_software[16];
  int16_t i_buf_software[8][10][10];
  int16_t o_buf_software[16][8][8];

  conv_validate(int *param_list); //(int layer_num, int num_input,int num_output,int kernel_size,int stride,int padding, int inputfeature_size, int inport);

  void test_initialize(void);
  void prepare_weight(void);
  void prepare_bias(void);
  void prepare_feature_in(void);

  // hardware execution related verifications
  void clear_outbuf(void);
  void compare_s_h_weight(void);
  int compare_s_h_output(void);
  void print_weight(void);
  void print_feature_in(void);
  void print_bias(void);
  void print_feature_out(void);

  // software execution related verifications
  void print_weight_software(void);
  void print_feature_in_software(void);
  void print_bias_software(void);
  void software_conv_process(void);
  void print_feature_out_software(void);
  void clear_outbuf_software(void);

  //	void test_fun(void);
  void layer_result_verification(void);
};

#endif
