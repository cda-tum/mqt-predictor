OPENQASM 2.0;
include "qelib1.inc";

qreg node[2];
creg meas[2];
sx node[1];
rz(0.5*pi) node[1];
sx node[1];
cx node[1],node[0];
barrier node[0],node[1];
measure node[0] -> meas[0];
measure node[1] -> meas[1];
