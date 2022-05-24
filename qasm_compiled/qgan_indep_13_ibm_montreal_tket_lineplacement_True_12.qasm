OPENQASM 2.0;
include "qelib1.inc";

qreg node[26];
creg meas[13];
rz(0.5*pi) node[4];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[15];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[21];
sx node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
sx node[4];
sx node[6];
sx node[7];
sx node[10];
sx node[12];
sx node[13];
sx node[15];
sx node[17];
sx node[18];
sx node[21];
rz(1.513499863884849*pi) node[23];
sx node[24];
sx node[25];
rz(1.5*pi) node[4];
rz(1.5*pi) node[6];
rz(1.5*pi) node[7];
rz(1.5*pi) node[10];
rz(1.5*pi) node[12];
rz(1.5*pi) node[13];
rz(1.5*pi) node[15];
rz(1.5*pi) node[17];
rz(3.5*pi) node[18];
rz(3.5*pi) node[21];
sx node[23];
rz(1.5*pi) node[24];
rz(1.5*pi) node[25];
sx node[4];
sx node[6];
sx node[7];
sx node[10];
sx node[12];
sx node[13];
sx node[15];
sx node[17];
sx node[18];
sx node[21];
sx node[24];
sx node[25];
rz(3.639606028395803*pi) node[4];
rz(0.38522532582539426*pi) node[6];
rz(3.928554183072297*pi) node[7];
rz(0.5270555212489987*pi) node[10];
rz(0.004465872930330739*pi) node[12];
rz(3.8101623541459726*pi) node[13];
rz(0.3841876674274236*pi) node[15];
rz(3.9713966565693632*pi) node[17];
rz(1.0014179144119497*pi) node[18];
rz(1.047371959515485*pi) node[21];
rz(0.3986133759763074*pi) node[24];
rz(0.001764727067163907*pi) node[25];
cx node[23],node[24];
sx node[24];
rz(2.5*pi) node[24];
sx node[24];
rz(1.5*pi) node[24];
cx node[25],node[24];
cx node[24],node[25];
cx node[25],node[24];
cx node[23],node[24];
cx node[23],node[21];
cx node[25],node[24];
cx node[23],node[21];
sx node[24];
cx node[21],node[23];
rz(2.5*pi) node[24];
cx node[23],node[21];
sx node[24];
cx node[21],node[18];
rz(1.5*pi) node[24];
cx node[21],node[18];
cx node[25],node[24];
cx node[18],node[21];
cx node[24],node[25];
cx node[21],node[18];
cx node[25],node[24];
cx node[18],node[15];
cx node[24],node[23];
cx node[18],node[17];
cx node[24],node[23];
cx node[18],node[15];
cx node[23],node[24];
cx node[15],node[18];
cx node[24],node[23];
cx node[18],node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[15],node[12];
cx node[23],node[21];
sx node[24];
cx node[15],node[12];
cx node[21],node[23];
rz(2.5*pi) node[24];
cx node[12],node[15];
cx node[23],node[21];
sx node[24];
cx node[15],node[12];
cx node[21],node[18];
rz(1.5*pi) node[24];
cx node[12],node[10];
cx node[21],node[18];
cx node[25],node[24];
cx node[12],node[13];
cx node[18],node[21];
cx node[24],node[25];
cx node[12],node[10];
cx node[21],node[18];
cx node[25],node[24];
cx node[10],node[12];
cx node[18],node[17];
cx node[24],node[23];
cx node[12],node[10];
cx node[18],node[15];
cx node[24],node[23];
cx node[10],node[7];
cx node[18],node[15];
cx node[23],node[24];
cx node[10],node[7];
cx node[15],node[18];
cx node[24],node[23];
cx node[7],node[10];
cx node[18],node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[10],node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
cx node[7],node[4];
cx node[15],node[12];
cx node[18],node[17];
cx node[21],node[23];
rz(2.5*pi) node[24];
cx node[7],node[6];
cx node[12],node[15];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
sx node[7];
cx node[15],node[12];
cx node[21],node[18];
rz(1.5*pi) node[24];
rz(3.4447301577156164*pi) node[7];
cx node[12],node[13];
cx node[21],node[18];
cx node[25],node[24];
sx node[7];
cx node[12],node[10];
cx node[18],node[21];
cx node[24],node[25];
rz(1.0*pi) node[7];
cx node[12],node[10];
cx node[21],node[18];
cx node[25],node[24];
cx node[4],node[7];
cx node[10],node[12];
cx node[18],node[17];
cx node[24],node[23];
cx node[7],node[4];
cx node[12],node[10];
cx node[18],node[15];
cx node[23],node[24];
cx node[4],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[24],node[23];
cx node[10],node[7];
cx node[12],node[13];
cx node[15],node[18];
cx node[23],node[24];
cx node[10],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[7],node[10];
cx node[15],node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
cx node[10],node[7];
cx node[12],node[15];
cx node[18],node[17];
cx node[21],node[23];
rz(2.5*pi) node[24];
cx node[7],node[6];
cx node[15],node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
sx node[7];
cx node[12],node[15];
cx node[21],node[18];
rz(1.5*pi) node[24];
rz(3.805227524588507*pi) node[7];
cx node[12],node[13];
cx node[21],node[18];
cx node[25],node[24];
sx node[7];
cx node[12],node[10];
cx node[18],node[21];
cx node[24],node[25];
rz(1.0*pi) node[7];
cx node[12],node[10];
cx node[21],node[18];
cx node[25],node[24];
cx node[6],node[7];
cx node[10],node[12];
cx node[18],node[17];
cx node[24],node[23];
cx node[7],node[6];
cx node[12],node[10];
cx node[18],node[15];
cx node[24],node[23];
cx node[6],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[23],node[24];
cx node[10],node[7];
cx node[12],node[13];
cx node[15],node[18];
cx node[24],node[23];
sx node[10];
cx node[13],node[12];
cx node[18],node[15];
cx node[23],node[21];
cx node[25],node[24];
rz(0.5030537118800071*pi) node[10];
cx node[15],node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
sx node[10];
cx node[12],node[15];
cx node[18],node[17];
cx node[21],node[23];
rz(2.5*pi) node[24];
rz(1.0*pi) node[10];
cx node[15],node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
cx node[7],node[10];
cx node[12],node[15];
cx node[21],node[18];
rz(1.5*pi) node[24];
cx node[10],node[7];
cx node[12],node[13];
cx node[21],node[18];
cx node[25],node[24];
cx node[7],node[10];
cx node[18],node[21];
cx node[24],node[25];
cx node[12],node[10];
cx node[21],node[18];
cx node[25],node[24];
sx node[12];
cx node[18],node[17];
cx node[24],node[23];
rz(3.129645470513012*pi) node[12];
cx node[18],node[15];
cx node[23],node[24];
sx node[12];
cx node[18],node[15];
cx node[24],node[23];
rz(1.0*pi) node[12];
cx node[15],node[18];
cx node[23],node[24];
cx node[13],node[12];
cx node[18],node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[12],node[13];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
cx node[13],node[12];
cx node[18],node[17];
cx node[21],node[23];
rz(2.5*pi) node[24];
cx node[15],node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
cx node[15],node[12];
cx node[21],node[18];
rz(1.5*pi) node[24];
cx node[12],node[15];
cx node[21],node[18];
cx node[25],node[24];
cx node[15],node[12];
cx node[18],node[21];
cx node[24],node[25];
cx node[12],node[10];
cx node[21],node[18];
cx node[25],node[24];
sx node[12];
cx node[18],node[17];
cx node[24],node[23];
rz(0.5062327382330409*pi) node[12];
cx node[18],node[15];
cx node[23],node[24];
sx node[12];
cx node[18],node[15];
cx node[24],node[23];
rz(1.0*pi) node[12];
cx node[15],node[18];
cx node[23],node[24];
cx node[10],node[12];
cx node[18],node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[12],node[10];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
cx node[10],node[12];
cx node[18],node[17];
cx node[21],node[23];
rz(2.5*pi) node[24];
cx node[15],node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
sx node[15];
cx node[21],node[18];
rz(1.5*pi) node[24];
rz(3.0628444756163615*pi) node[15];
cx node[21],node[18];
cx node[25],node[24];
sx node[15];
cx node[18],node[21];
cx node[24],node[25];
rz(1.0*pi) node[15];
cx node[21],node[18];
cx node[25],node[24];
cx node[12],node[15];
cx node[18],node[17];
cx node[24],node[23];
cx node[15],node[12];
cx node[23],node[24];
cx node[12],node[15];
cx node[24],node[23];
cx node[18],node[15];
cx node[23],node[24];
sx node[18];
cx node[23],node[21];
cx node[25],node[24];
rz(0.5291425562188703*pi) node[18];
cx node[23],node[21];
sx node[24];
sx node[18];
cx node[21],node[23];
rz(2.5*pi) node[24];
rz(1.0*pi) node[18];
cx node[23],node[21];
sx node[24];
cx node[17],node[18];
rz(1.5*pi) node[24];
cx node[18],node[17];
cx node[25],node[24];
cx node[17],node[18];
cx node[24],node[25];
cx node[21],node[18];
cx node[25],node[24];
cx node[21],node[18];
cx node[24],node[23];
cx node[18],node[21];
cx node[23],node[24];
cx node[21],node[18];
cx node[24],node[23];
cx node[18],node[15];
cx node[23],node[24];
sx node[18];
cx node[23],node[21];
cx node[25],node[24];
rz(0.025833150106902236*pi) node[18];
cx node[23],node[21];
sx node[24];
sx node[18];
cx node[21],node[23];
rz(2.5*pi) node[24];
rz(1.0*pi) node[18];
cx node[23],node[21];
sx node[24];
cx node[15],node[18];
rz(1.5*pi) node[24];
cx node[18],node[15];
cx node[25],node[24];
cx node[15],node[18];
cx node[24],node[25];
cx node[21],node[18];
cx node[25],node[24];
sx node[21];
cx node[24],node[23];
rz(0.9312783952554322*pi) node[21];
cx node[25],node[24];
sx node[21];
cx node[24],node[23];
rz(1.0*pi) node[21];
cx node[25],node[24];
cx node[18],node[21];
cx node[24],node[23];
cx node[21],node[18];
sx node[23];
cx node[18],node[21];
rz(2.5*pi) node[23];
sx node[23];
rz(1.5*pi) node[23];
cx node[21],node[23];
cx node[23],node[21];
cx node[21],node[23];
cx node[24],node[23];
sx node[24];
rz(0.8132471586800021*pi) node[24];
sx node[24];
rz(1.0*pi) node[24];
cx node[25],node[24];
cx node[24],node[25];
cx node[25],node[24];
cx node[24],node[23];
cx node[21],node[23];
sx node[24];
sx node[21];
rz(0.21965921759954288*pi) node[23];
rz(3.569028213435318*pi) node[24];
rz(0.6869465530300345*pi) node[21];
sx node[23];
sx node[24];
sx node[21];
rz(0.5*pi) node[23];
rz(1.0*pi) node[24];
rz(1.0*pi) node[21];
sx node[23];
rz(1.5*pi) node[23];
barrier node[4],node[6],node[7],node[13],node[10],node[12],node[17],node[15],node[18],node[25],node[24],node[21],node[23];
measure node[4] -> meas[0];
measure node[6] -> meas[1];
measure node[7] -> meas[2];
measure node[13] -> meas[3];
measure node[10] -> meas[4];
measure node[12] -> meas[5];
measure node[17] -> meas[6];
measure node[15] -> meas[7];
measure node[18] -> meas[8];
measure node[25] -> meas[9];
measure node[24] -> meas[10];
measure node[21] -> meas[11];
measure node[23] -> meas[12];
