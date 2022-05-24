OPENQASM 2.0;
include "qelib1.inc";

qreg node[2];
creg meas[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(1.0*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
barrier node[1],node[0];
measure node[1] -> meas[0];
measure node[0] -> meas[1];
