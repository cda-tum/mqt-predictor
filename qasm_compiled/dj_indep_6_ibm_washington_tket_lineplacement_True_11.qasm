OPENQASM 2.0;
include "qelib1.inc";

qreg node[126];
creg c[5];
sx node[111];
sx node[121];
sx node[122];
sx node[123];
sx node[124];
sx node[125];
rz(0.5*pi) node[111];
rz(0.5*pi) node[121];
rz(0.5*pi) node[122];
rz(0.5*pi) node[123];
rz(0.5*pi) node[124];
rz(0.5*pi) node[125];
sx node[111];
sx node[121];
sx node[122];
sx node[123];
sx node[124];
sx node[125];
rz(0.5*pi) node[111];
rz(0.5*pi) node[121];
rz(0.5*pi) node[122];
rz(0.5*pi) node[123];
rz(0.5*pi) node[124];
rz(1.0*pi) node[125];
cx node[125],node[124];
cx node[124],node[125];
cx node[125],node[124];
cx node[123],node[124];
rz(0.5*pi) node[123];
cx node[125],node[124];
sx node[123];
rz(0.5*pi) node[125];
rz(3.5*pi) node[123];
sx node[125];
sx node[123];
rz(3.5*pi) node[125];
rz(1.0*pi) node[123];
sx node[125];
cx node[124],node[123];
rz(1.0*pi) node[125];
cx node[123],node[124];
cx node[124],node[123];
cx node[122],node[123];
rz(0.5*pi) node[122];
sx node[122];
rz(3.5*pi) node[122];
sx node[122];
rz(1.0*pi) node[122];
cx node[123],node[122];
cx node[122],node[123];
cx node[123],node[122];
cx node[111],node[122];
rz(0.5*pi) node[111];
cx node[121],node[122];
sx node[111];
rz(0.5*pi) node[121];
rz(3.5*pi) node[111];
sx node[121];
sx node[111];
rz(3.5*pi) node[121];
rz(1.0*pi) node[111];
sx node[121];
rz(1.0*pi) node[121];
barrier node[124],node[125],node[123],node[111],node[121],node[122];
measure node[124] -> c[0];
measure node[125] -> c[1];
measure node[123] -> c[2];
measure node[111] -> c[3];
measure node[121] -> c[4];
