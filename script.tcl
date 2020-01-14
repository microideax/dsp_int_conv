############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
############################################################
open_project core_gen
set_top sub_net_proc
add_files ./c_src/dsp_conv_int.h
add_files ./c_src/top_test.cpp
add_files -tb ./c_src/top_test.cpp -cflags "-Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas"
open_solution "engine_stable"
set_part {xc7z020clg484-1} -tool vivado
create_clock -period 4 -name default
remove_core Mul
config_core -latency 2 DSP48
config_schedule -disable_reduceDpCEfanout
#source "./core_gen/engine_stable/directives.tcl"
csim_design
#csynth_design
#cosim_design -trace_level all
#export_design -rtl verilog -format ip_catalog
