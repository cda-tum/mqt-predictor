OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg meas[10];
rxx(pi/2) q[0],q[1];
rxx(pi/2) q[2],q[3];
rxx(pi/2) q[2],q[5];
ry(-pi/2) q[2];
rx(-pi/2) q[5];
rxx(pi/2) q[4],q[5];
ry(-pi/2) q[5];
rz(pi/2) q[5];
rxx(pi/2) q[1],q[6];
ry(-pi/2) q[1];
rxx(pi/2) q[0],q[7];
ry(-pi/2) q[0];
rx(-pi/2) q[7];
rxx(pi/2) q[6],q[7];
ry(-pi/2) q[6];
ry(-pi/2) q[7];
rz(pi/2) q[7];
rxx(pi/2) q[3],q[8];
ry(-pi/2) q[3];
rxx(pi/2) q[4],q[9];
ry(-pi/2) q[4];
rx(-pi/2) q[9];
rxx(pi/2) q[8],q[9];
ry(-pi/2) q[8];
ry(-pi/2) q[9];
rz(pi/2) q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
