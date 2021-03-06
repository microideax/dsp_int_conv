#include <iostream>
#include <math.h>
#include <stdlib.h>
#include "conv_validate.h"

using namespace std;

conv_validate::conv_validate(int *param_list)
{
  //	int i,j,k;
  //	int input_num;
  //	int input_feature_size;
  //	int output_num;

  this->param_list = param_list;
  if (param_list != NULL)
  {
    cout << "Initialized layer parameters!" << endl;
  }
  else
  {
    cout << "Please specify param values!" << endl;
  }
  // software validation data packages
  N = param_list[0];
  K = param_list[1];
  M = param_list[2];
  Ri = param_list[3];
  Ci = param_list[4];
  R = param_list[5];
  C = param_list[6];
  S = param_list[7];
  P = param_list[8];
  act = param_list[9];
  //  param_list[10];
  //  param_list[11];
  //  param_list[12];
  //  param_list[13];
  //  param_list[14];
  //  int param_list[15];
  //	this->layer_num = param_list[16];
  //	input_num = param_list[16+0];
  //	input_feature_size = param_list[16+3];
};

void conv_validate::prepare_weight(void)
{
  // initialize weight data for layer test
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < M; j++)
    {
      for (int k1 = 0; k1 < K; k1++)
      {
        for (int k2 = 0; k2 < K; k2++)
        {
          w_buf_software[i][j][k1][k2] = k2 % K;
        }
      }
    }
  }
  for (int i = 0; i < N; i++)
  {
    for (int k1 = 0; k1 < K; k1++)
    {
      for (int k2 = 0; k2 < K; k2++)
      {
        for (int j = 0; j < M; j++)
        {
          //					for(int idx = 0; idx < 4; idx++){
          w_port[i * (M / 4) * K * K + (j / 4) * K * K + k1 * K + k2].range(16 * (j % 4) + 15, 16 * (j % 4)) = w_buf_software[i][j][k1][k2];
          //					cout << "location: i=" << i << " j=" << j << " k1=" << k1 << " k2=" << k2 << " value_i="<< w_port[i*(M/4)*K*K + (j/4)*K*K + k1*K + k2].range(16*(j%4)+15, 16*(j%4))
          //							<< " value_o="<< w_buf_software[i][j][k1][k2]<< endl;
          //					}
        }
      }
    }
  }
};

void conv_validate::prepare_feature_in(void)
{

  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < Ri; j++)
    {
      for (int k = 0; k < Ci; k++)
      {
        i_buf_software[i][j][k] = k % 10;
        i_port[(i / 4) * Ri * Ci + j * Ci + k].range(16 * (i % 4) + 15, 16 * (i % 4)) = i_buf_software[i][j][k];
      }
    }
  }
};

void conv_validate::prepare_bias(void)
{
  for (int i = 0; i < M; i++)
  {
    b_buf_software[i] = i % 10;
    b_port[i] = b_buf_software[i];
  }
}

void conv_validate::test_initialize(void)
{
  prepare_bias();
  prepare_weight();
  prepare_feature_in();
};

void conv_validate::software_conv_process(void)
{
  int i, x, y, j, k, z;
  for (i = 0; i < M; i++)
  {
    for (x = 0; x < R; x++)
    {
      for (y = 0; y < C; y++)
      {
        o_buf_software[i][x][y] = b_buf_software[i];
      }
    }
  }
  for (i = 0; i < M; i++)
  {
    for (x = 0; x < R; x += 1)
    {
      for (y = 0; y < C; y += 1)
      {
        for (j = 0; j < N; j++)
        {
          for (k = 0; k < K; k++)
          {
            for (z = 0; z < K; z++)
            {
              o_buf_software[i][x][y] += i_buf_software[j][S * x + k][S * y + z] * w_buf_software[j][i][k][z];
            }
          }
        }
        // o_buf_software[i][x][y] = (o_buf_software[i][x][y] < 0) ? 0 : o_buf_software[i][x][y];
      }
    }
  }
};

void conv_validate::print_weight_software(void)
{
  int i, j, k1, k2;
  cout << "Print software verification weights:" << endl;
  for (j = 0; j < M; j++)
  {
    for (i = 0; i < N; i++)
    {
      for (k1 = 0; k1 < K; k1++)
      {
        for (k2 = 0; k2 < K; k2++)
        {
          cout << w_buf_software[i][j][k1][k2] << " ";
        }
        cout << endl;
      }
      cout << endl;
    }
    cout << "-----------------" << endl;
  }
  cout << "-----------------" << endl;
};

void conv_validate::print_weight(void)
{
  int i, j, k1, k2;

  cout << "Print squeezed testing weights:" << endl;
  for (i = 0; i < N; i++)
  {
    for (k1 = 0; k1 < K; k1++)
    {
      for (k2 = 0; k2 < K; k2++)
      {
        for (j = 0; j < M; j++)
        {
          w_buf_software[i][j][k1][k2] = w_port[i * (M / 4) * K * K + (j / 4) * K * K + k1 * K + k2].range(16 * (j % 4) + 15, 16 * (j % 4));
          //					cout << "location: i=" << i << " j=" << j << " k1=" << k1 << " k2=" << k2 << " value="<< w_port[i*(M/4)*K*K + (j/4)*K*K + k1*K + k2].range(16*(j%4)+15, 16*(j%4)) << endl;
        }
      }
    }
  }
  print_weight_software();
};

void conv_validate::print_feature_out_software(void)
{
  int i, j, k;
  cout << "Printing software buffer data !!!" << endl;
  for (i = 0; i < M; i++)
  {
    for (j = 0; j < R; j++)
    {
      for (k = 0; k < C; k++)
      {
        cout << o_buf_software[i][j][k] << "  ";
      }
      cout << endl;
    }
    cout << endl;
  }
  cout << endl;
};

void conv_validate::print_feature_out(void)
{
  int i, j, k;
  ap_uint<64> o_tmp;
  cout << "Printing output port data !!!" << endl;
  for (i = 0; i < M; i++)
  {
    for (j = 0; j < R; j++)
    {
      for (k = 0; k < C; k++)
      {
        o_tmp = o_port[(i / 4) * R * C + j * C + k];
        cout << o_tmp.range(16 * (i % 4) + 15, 16 * (i % 4)) << "  ";
      }
      cout << endl;
    }
    cout << endl;
  }
  cout << endl;
};

void conv_validate::compare_s_h_weight(void)
{
  int i, j, k1, k2;

  cout << "Print squeezed testing weights:" << endl;
  for (j = 0; j < M; j++)
  {
    for (i = 0; i < N; i++)
    {
      for (k1 = 0; k1 < K; k1++)
      {
        for (k2 = 0; k2 < K; k2++)
        {
          if (w_buf_software[i][j][k1][k2] != w_port[i * (M / 4) * K * K + (j / 4) * K * K + k1 * K + k2].range(16 * (j % 4) + 15, 16 * (j % 4)))
          {
            cout << "Incompatible software and squeezed weights !!! " << w_buf_software[i][j][k1][k2] << " and " << w_port[i * (M / 4) * K * K + (j / 4) * K * K + k1 * K + k2].range(16 * (j % 4) + 15, 16 * (j % 4)) << endl;
          }
          else
          {
            ;
          }
        }
      }
    }
  }
  //	print_weight_software();
};

int conv_validate::compare_s_h_output(void)
{
  int i, j, k, verification_flag = 0;
  for (i = 0; i < M; i++)
  {
    for (j = 0; j < R; j++)
    {
      for (k = 0; k < C; k++)
      {
        if (o_buf_software[i][j][k] != o_port[i / 4 * R * C + j * C + k].range((i % 4) * 16 + 15, (i % 4) * 16))
        {
          cout << "Wrong output!!! at M = " << i << ",  R=" << j << ",  C=" << k << endl;
          verification_flag = 1;
        }
        else
        {
          ;
        }
      }
    }
  }
  return verification_flag;
};

void conv_validate::layer_result_verification(void)
{
  int ver_flag = 0;
  software_conv_process();
  ver_flag = compare_s_h_output();
  if (ver_flag == 0)
  {
    cout << "Verification done successfully!" << endl;
  }
  else
  {
    cout << "Wrong output, please check the implementation !" << endl;
  }
};

/*
void conv_validate :: print_feature_in(void)
{
	int i,j;
	cout << "input feature:" << endl;
	for(i = 0 ; i <12*12; i++)
	{
		cout <<i<<":";
		for(j = 0 ; j < 32; j++)
			cout <<(short int)(input_feature[i].range(15+16*j,16*j)) <<" ";
		cout << endl;
	}
}
//
//

//
//void conv_validate :: print_bias(void)
//{
//	int i;
//	cout <<"bias:"<<endl;
//	for(i = 0 ; i < num_output; i++)
//		cout << i << ":" << bias[i] <<endl;
//}
//
*/

/*
void conv_validate :: software_conv_process(void)       //assume padding = 2
{
	int i,j,k,x,y,z;
	int16_t temp;
	short int* temp_array = new short int[(Ri + 2 * P) * (Ci + 2 * P) * N];

//	for(k = 0 ; k < num_input; k++)
//		for(i = 0 ; i < inputfeature_size+2*padding; i++)
//			for(j = 0 ; j < inputfeature_size+2*padding ; j++)
//				temp_array[ k * (inputfeature_size+2*padding ) * (inputfeature_size+2*padding ) + i * (inputfeature_size+2*padding ) + j] = 0;
//
//	for(j = 0 ; j < num_input; j++)
//		for(x = padding; x < inputfeature_size + padding; x++)
//			for(y = padding; y < inputfeature_size + padding; y++)
//				temp_array[ j * (inputfeature_size+2*padding ) * (inputfeature_size+2*padding ) + x * (inputfeature_size+2*padding ) + y]
//							= input_feature[(j/32) * inputfeature_size * inputfeature_size +(x-padding)*inputfeature_size + (y-padding)].range(16*(j%32)+15,16*(j%32));

	cout <<"software processing..." << endl;
	for(i = 0 ; i < M; i++)
	{
		for(x = 0 ; x < R; x += 1)
		{
			for(y = 0; y < C; y += 1)
			{
				o_buf_software[i][x][y] = b_buf_software[i];
				for(j = 0 ; j < N; j++)
				{
					for(k = 0 ; k < K; k++)
					{
						for(z = 0 ; z < K; z++)
						{
							o_buf_software[i][x][y] += i_buf_software[j][S*x + k][S*y + z] * w_buf_software[j][i][k][z];
						}
					}
				}
				o_buf_software[i][x][y] = (o_buf_software[i][x][y] < 0) ? 0 : o_buf_software[i][x][y];
			}
		}
	}
};
*/
/*
void conv_validate :: print_feature_out(void)
{

	ap_uint<64> o_tmp = 0;
	ap_uint<16> o_tmp_3d[8][8][8];
	ap_uint<16> data_reg;
	for (int i = 0; i < param_list[2]; i+=4)
	{
	  for (int j = 0; j < param_list[5]; j++)
	  {
	    for (int k = 0; k < param_list[6]; k++)
	    {
	      o_tmp = *(o_port + (i/4) * param_list[5] * param_list[6] + j * param_list[6] + k);
	      for (int idx = 0; idx < 4; idx++){
	      data_reg.range(15,0) = o_tmp.range(idx*16 + 15, idx * 16);
	      o_tmp_3d[i + idx][j][k] = data_reg;
	     }
	   }
	   cout << endl;
	  }
	 cout << endl;
	  }
	  for (int i = 0; i < param_list[2]; i++){
		  for(int j = 0; j < param_list[5]; j++){
			  for(int k = 0; k < param_list[6]; k++){
				  cout<< o_tmp_3d[i][j][k] << " ";
			  }
			  cout << endl;
		  }
		  cout << endl;
	  }
	  cout << endl;
};
*/

/*
//
//void conv_validate :: print_feature_out_softeare(void)
//{
//	int i,j;
//	cout <<"software feature out:"<<endl;
//	for(i = 0 ; i < outputfeature_size * outputfeature_size * (int)(ceil(double(num_output)/32)); i++)
//	{
//		cout <<i<<":";
//		for(j = 0 ; j < 32; j++)
//			cout <<output_feature_software[i].range(15+16*j,16*j) <<" ";
//		cout << endl;
//	}
//}
//
//void conv_validate :: test_fun(void)
//{
//	cout << "class test : "<< outputfeature_size << endl;
//	cout << (int)(ceil((double)num_input/32)) << endl;

//}

 */
void conv_validate::clear_outbuf(void)
{
  cout << "Output buffer is cleared!" << endl;
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 8; j++)
    {
      for (int k = 0; k < 8; k++)
      {
        *(o_port + i * 64 + j * 8 + k) = int16_t(0);
      }
    }
  }
};

void conv_validate::clear_outbuf_software(void)
{
  cout << "Software output buffer is cleared!" << endl;
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < R; j++)
    {
      for (int k = 0; k < C; k++)
      {
        o_buf_software[i][j][k] = int16_t(0);
      }
    }
  }
};
