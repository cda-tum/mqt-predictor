OPENQASM 2.0;
include "qelib1.inc";

qreg node[127];
creg meas[4];
sx node[123];
rz(0.5*pi) node[123];
sx node[123];
cx node[123],node[124];
cx node[124],node[125];
cx node[125],node[126];
barrier node[126],node[125],node[124],node[123];
measure node[126] -> meas[0];
measure node[125] -> meas[1];
measure node[124] -> meas[2];
measure node[123] -> meas[3];
