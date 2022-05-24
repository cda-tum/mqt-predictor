OPENQASM 2.0;
include "qelib1.inc";

qreg node[26];
creg meas[3];
sx node[25];
rz(0.5*pi) node[25];
sx node[25];
cx node[25],node[24];
cx node[24],node[23];
barrier node[23],node[24],node[25];
measure node[23] -> meas[0];
measure node[24] -> meas[1];
measure node[25] -> meas[2];
