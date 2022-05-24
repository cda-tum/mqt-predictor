OPENQASM 2.0;
include "qelib1.inc";

qreg node[126];
creg meas[2];
sx node[125];
rz(0.5*pi) node[125];
sx node[125];
cx node[125],node[124];
barrier node[124],node[125];
measure node[124] -> meas[0];
measure node[125] -> meas[1];
