#include <iostream>
#include <stdint.h>
#include "dsp_conv_int.h"
#include "layer_net.h"
#include <ap_int.h>

#include "conv_validate.h"

using namespace std;

//#define IN_PORT_WIDTH 64

int main()
{

  int16_t in_buf[4][10][10];
  bool dsp_clk;
  /*
#if IN_PORT_WIDTH == 64
  ap_uint<64> in_port[4096];
  ap_uint<64> w_port[256];
  ap_uint<64> out_port[1024];
#else
  int16_t in_port[1024];
  int16_t w_port[256];
  int16_t out_port[1024];
#endif
  ap_uint<64> i_tmp_buf = ap_uint<64>(0);
  ap_uint<64> w_tmp_buf = ap_uint<64>(0);
  int16_t w_buf[4][4][3][3];
  int16_t b_port[256];
  int32_t out_buf[4][8][8];
*/
  int param[16] = {
      8,
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

  conv_validate layer_test(param);

  cout << "initialized layer: N=" << layer_test.N << " M=" << layer_test.M << " Ri=" << layer_test.Ri << " K=" << layer_test.K << endl;
  // layer_test.print_weight_software();
  //  layer_test.print_weight();
  //  layer_test.compare_s_h_weight();
  layer_test.test_initialize();

  sub_net_proc(layer_test.param_list, layer_test.b_port, layer_test.w_port, layer_test.i_port, layer_test.o_port, dsp_clk);

  layer_test.clear_outbuf_software();
  // layer_test.software_conv_process();
  // layer_test.print_feature_out_software();
  // layer_test.print_feature_out();
  layer_test.layer_result_verification();

  /*
  // display output buffer
  cout << endl;
  cout << endl;
  cout << "main: The output buffer:" << endl;
  ap_uint<64> o_tmp = 0;
  ap_uint<16> o_tmp_3d[8][8][8];
  ap_uint<16> data_reg;
  for (int i = 0; i < param[2]; i+=4)
  {
    for (int j = 0; j < param[5]; j++)
    {
      for (int k = 0; k < param[6]; k++)
      {
        o_tmp = *(out_port + (i/4) * param[5] * param[6] + j * param[6] + k);
        for (int idx = 0; idx < 4; idx++){
        	data_reg.range(15,0) = o_tmp.range(idx*16 + 15, idx * 16);
        	o_tmp_3d[i + idx][j][k] = data_reg;
        }
      }
//      cout << endl;
    }
//    cout << endl;
  }
  for (int i = 0; i < param[2]; i++){
	  for(int j = 0; j < param[5]; j++){
		  for(int k = 0; k < param[6]; k++){
			  cout<< o_tmp_3d[i][j][k] << " ";
		  }
		  cout << endl;
	  }
	  cout << endl;
  }
  cout << endl;
*/
  return 0;
};
