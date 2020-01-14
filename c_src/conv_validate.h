#ifndef CONV_VALIDATE_H
#define CONV_VALIDATE_H
// #include "/opt/Xilinx/Vivado/2018.1/include/gmp.h"
// #include "/opt/Xilinx/Vivado/2018.1/include/mpfr.h"
#include "ap_fixed.h"
#include "dsp_conv_int.h"

class conv_validate
{
public:
  int layer_num;
#if IN_PORT_WIDTH == 64
  ap_int<64> weight[16384];
  ap_int<64> input_feature[6400];
  ap_int<64> output_feature[4096];
  ap_int<64> output_feature_software[4096];
  ap_fixed<16> bias[1024];
  ap_fixed<16> *param_list;
#else if IN_PORT_WIDTH == 512
  ap_int<512> weight[16384];
  ap_int<512> input_feature[6400];
  ap_int<512> output_feature[4096];
  ap_int<512> output_feature_software[4096];
  ap_fixed<32, 26> bias[1024];
  ap_uint<32> *param_list;
#else

#endif

  conv_validate(ap_uint<32> *param_list); //(int layer_num, int num_input,int num_output,int kernel_size,int stride,int padding, int inputfeature_size, int inport);
  void print_weight(void);
  void print_feature_in(void);
  void print_bias(void);
  //	void print_feature_out(void);
  //
  //	void software_conv_process(void);
  //	void print_feature_out_softeare(void);
  //	void test_fun(void);
};

#endif
