OPENQASM 2.0;
include "qelib1.inc";

qreg node[126];
creg meas[2];
x node[124];
sx node[125];
rz(0.5*pi) node[125];
sx node[125];
barrier node[125],node[124];
measure node[125] -> meas[0];
measure node[124] -> meas[1];
