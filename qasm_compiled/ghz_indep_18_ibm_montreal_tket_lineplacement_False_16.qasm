OPENQASM 2.0;
include "qelib1.inc";

qreg node[20];
creg meas[18];
sx node[8];
rz(0.5*pi) node[8];
sx node[8];
cx node[8],node[5];
cx node[5],node[3];
cx node[3],node[2];
cx node[2],node[1];
cx node[1],node[0];
cx node[4],node[1];
cx node[1],node[4];
cx node[4],node[1];
cx node[0],node[1];
cx node[7],node[4];
cx node[4],node[7];
cx node[7],node[4];
cx node[1],node[4];
cx node[10],node[7];
cx node[7],node[10];
cx node[10],node[7];
cx node[4],node[7];
cx node[12],node[10];
cx node[7],node[6];
cx node[10],node[12];
cx node[12],node[10];
cx node[10],node[7];
cx node[13],node[12];
cx node[7],node[10];
cx node[12],node[13];
cx node[10],node[7];
cx node[13],node[12];
cx node[6],node[7];
cx node[12],node[10];
cx node[14],node[13];
cx node[10],node[12];
cx node[13],node[14];
cx node[12],node[10];
cx node[14],node[13];
cx node[7],node[10];
cx node[15],node[12];
cx node[12],node[15];
cx node[15],node[12];
cx node[10],node[12];
cx node[12],node[13];
cx node[13],node[14];
cx node[14],node[11];
cx node[13],node[14];
cx node[14],node[11];
cx node[11],node[8];
cx node[16],node[14];
cx node[8],node[11];
cx node[14],node[16];
cx node[11],node[8];
cx node[16],node[14];
cx node[8],node[9];
cx node[14],node[11];
cx node[19],node[16];
cx node[11],node[14];
cx node[16],node[19];
cx node[14],node[11];
cx node[19],node[16];
cx node[11],node[8];
cx node[16],node[14];
cx node[8],node[11];
cx node[14],node[16];
cx node[11],node[8];
cx node[16],node[14];
cx node[9],node[8];
cx node[14],node[11];
cx node[11],node[14];
cx node[14],node[11];
cx node[8],node[11];
barrier node[11],node[8],node[9],node[14],node[13],node[12],node[10],node[7],node[6],node[15],node[4],node[1],node[0],node[19],node[2],node[3],node[5],node[16];
measure node[11] -> meas[0];
measure node[8] -> meas[1];
measure node[9] -> meas[2];
measure node[14] -> meas[3];
measure node[13] -> meas[4];
measure node[12] -> meas[5];
measure node[10] -> meas[6];
measure node[7] -> meas[7];
measure node[6] -> meas[8];
measure node[15] -> meas[9];
measure node[4] -> meas[10];
measure node[1] -> meas[11];
measure node[0] -> meas[12];
measure node[19] -> meas[13];
measure node[2] -> meas[14];
measure node[3] -> meas[15];
measure node[5] -> meas[16];
measure node[16] -> meas[17];
