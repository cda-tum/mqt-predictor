OPENQASM 2.0;
include "qelib1.inc";

qreg node[26];
creg meas[4];
sx node[23];
rz(0.5*pi) node[23];
sx node[23];
cx node[23],node[24];
cx node[24],node[25];
cx node[25],node[22];
barrier node[22],node[25],node[24],node[23];
measure node[22] -> meas[0];
measure node[25] -> meas[1];
measure node[24] -> meas[2];
measure node[23] -> meas[3];
