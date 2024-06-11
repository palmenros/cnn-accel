open_project CNN_HLS
set_top Conv
add_files HLS/conv.cpp
add_files HLS/conv.h
add_files -tb HLS/main.cpp
open_solution "solution1" -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
config_export -display_name Conv -format ip_catalog -output ./IP-catalog/HLS_Conv.zip -rtl vhdl -vendor EPFL -vivado_clock 10
quit
