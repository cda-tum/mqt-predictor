OPENQASM 2.0;
include "qelib1.inc";

qreg node[26];
creg meas[2];
sx node[24];
x node[25];
rz(0.5*pi) node[24];
sx node[24];
barrier node[24],node[25];
measure node[24] -> meas[0];
measure node[25] -> meas[1];
