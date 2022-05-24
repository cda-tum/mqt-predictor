OPENQASM 2.0;
include "qelib1.inc";

qreg node[127];
creg meas[5];
sx node[126];
rz(0.5*pi) node[126];
sx node[126];
cx node[126],node[125];
cx node[125],node[124];
cx node[124],node[123];
cx node[123],node[122];
barrier node[122],node[123],node[124],node[125],node[126];
measure node[122] -> meas[0];
measure node[123] -> meas[1];
measure node[124] -> meas[2];
measure node[125] -> meas[3];
measure node[126] -> meas[4];
