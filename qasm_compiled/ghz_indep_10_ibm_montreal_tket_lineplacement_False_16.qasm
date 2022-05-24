OPENQASM 2.0;
include "qelib1.inc";

qreg node[11];
creg meas[10];
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
cx node[7],node[6];
barrier node[6],node[7],node[4],node[1],node[0],node[10],node[2],node[3],node[5],node[8];
measure node[6] -> meas[0];
measure node[7] -> meas[1];
measure node[4] -> meas[2];
measure node[1] -> meas[3];
measure node[0] -> meas[4];
measure node[10] -> meas[5];
measure node[2] -> meas[6];
measure node[3] -> meas[7];
measure node[5] -> meas[8];
measure node[8] -> meas[9];
