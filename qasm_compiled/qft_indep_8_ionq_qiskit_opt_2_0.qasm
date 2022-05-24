OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
creg meas[8];
rx(-23.574217) q[7];
rxx(pi/2) q[7],q[6];
rx(-pi/2) q[6];
rz(-pi/4) q[6];
rxx(pi/2) q[7],q[6];
ry(-pi/4) q[6];
rx(-15.732507) q[6];
rxx(pi/2) q[7],q[5];
rx(-pi/2) q[5];
rz(-pi/8) q[5];
rxx(pi/2) q[7],q[5];
rx(-pi/2) q[5];
rz(pi/8) q[5];
rxx(pi/2) q[6],q[5];
rx(-pi/2) q[5];
rz(-pi/4) q[5];
rxx(pi/2) q[6],q[5];
ry(-pi/4) q[5];
rx(-12.615458) q[5];
rxx(pi/2) q[7],q[4];
rx(-pi/2) q[4];
rz(-pi/16) q[4];
rxx(pi/2) q[7],q[4];
rx(-pi/2) q[4];
rz(pi/16) q[4];
rxx(pi/2) q[6],q[4];
rx(-pi/2) q[4];
rz(-pi/8) q[4];
rxx(pi/2) q[6],q[4];
rx(-pi/2) q[4];
rz(pi/8) q[4];
rxx(pi/2) q[5],q[4];
rx(-pi/2) q[4];
rz(-pi/4) q[4];
rxx(pi/2) q[5],q[4];
ry(-pi/4) q[4];
rx(-9.5229527) q[4];
rxx(pi/2) q[7],q[3];
rx(-pi/2) q[3];
rz(-pi/32) q[3];
rxx(pi/2) q[7],q[3];
rx(-pi/2) q[3];
rz(pi/32) q[3];
rxx(pi/2) q[6],q[3];
rx(-pi/2) q[3];
rz(-pi/16) q[3];
rxx(pi/2) q[6],q[3];
rx(-pi/2) q[3];
rz(pi/16) q[3];
rxx(pi/2) q[5],q[3];
rx(-pi/2) q[3];
rz(-pi/8) q[3];
rxx(pi/2) q[5],q[3];
rx(-pi/2) q[3];
rz(pi/8) q[3];
rxx(pi/2) q[4],q[3];
rx(-pi/2) q[3];
rz(-pi/4) q[3];
rxx(pi/2) q[4],q[3];
ry(-pi/4) q[3];
rx(-8.0503312) q[3];
rxx(pi/2) q[7],q[2];
rx(-pi/2) q[2];
rz(-pi/64) q[2];
rxx(pi/2) q[7],q[2];
rx(-pi/2) q[2];
rz(pi/64) q[2];
rxx(pi/2) q[6],q[2];
rx(-pi/2) q[2];
rz(-pi/32) q[2];
rxx(pi/2) q[6],q[2];
rx(-pi/2) q[2];
rz(pi/32) q[2];
rxx(pi/2) q[5],q[2];
rx(-pi/2) q[2];
rz(-pi/16) q[2];
rxx(pi/2) q[5],q[2];
rx(-pi/2) q[2];
rz(pi/16) q[2];
rxx(pi/2) q[4],q[2];
rx(-pi/2) q[2];
rz(-pi/8) q[2];
rxx(pi/2) q[4],q[2];
rx(-pi/2) q[2];
rz(pi/8) q[2];
rxx(pi/2) q[3],q[2];
rx(-pi/2) q[2];
rz(-pi/4) q[2];
rxx(pi/2) q[3],q[2];
ry(-pi/4) q[2];
rx(-13*pi/8) q[2];
rxx(pi/2) q[7],q[1];
rx(-pi/2) q[1];
rz(-pi/128) q[1];
rxx(pi/2) q[7],q[1];
rx(-pi/2) q[1];
rz(pi/128) q[1];
rxx(pi/2) q[6],q[1];
rx(-pi/2) q[1];
rz(-pi/64) q[1];
rxx(pi/2) q[6],q[1];
rx(-pi/2) q[1];
rz(pi/64) q[1];
rxx(pi/2) q[5],q[1];
rx(-pi/2) q[1];
rz(-pi/32) q[1];
rxx(pi/2) q[5],q[1];
rx(-pi/2) q[1];
rz(pi/32) q[1];
rxx(pi/2) q[4],q[1];
rx(-pi/2) q[1];
rz(-pi/16) q[1];
rxx(pi/2) q[4],q[1];
rx(-pi/2) q[1];
rz(pi/16) q[1];
rxx(pi/2) q[3],q[1];
rx(-pi/2) q[1];
rz(-pi/8) q[1];
rxx(pi/2) q[3],q[1];
rx(-pi/2) q[1];
rz(pi/8) q[1];
rxx(pi/2) q[2],q[1];
rx(-pi/2) q[1];
rz(-pi/4) q[1];
rxx(pi/2) q[2],q[1];
ry(-pi/4) q[1];
rx(-3*pi/4) q[1];
rxx(pi/2) q[7],q[0];
rx(-pi/2) q[0];
rz(-pi/256) q[0];
rxx(pi/2) q[7],q[0];
rx(-pi/2) q[0];
rz(pi/256) q[0];
rxx(pi/2) q[6],q[0];
rx(-pi/2) q[0];
rz(-pi/128) q[0];
rxx(pi/2) q[6],q[0];
rx(-pi/2) q[0];
rz(pi/128) q[0];
rxx(pi/2) q[5],q[0];
rx(-pi/2) q[0];
rz(-pi/64) q[0];
rxx(pi/2) q[5],q[0];
rx(-pi/2) q[0];
rz(pi/64) q[0];
rxx(pi/2) q[4],q[0];
rx(-pi/2) q[0];
rz(-pi/32) q[0];
rxx(pi/2) q[4],q[0];
rx(-pi/2) q[0];
rz(pi/32) q[0];
rxx(pi/2) q[3],q[0];
rx(-pi/2) q[0];
rz(-pi/16) q[0];
rxx(pi/2) q[3],q[0];
rx(-pi/2) q[0];
rz(pi/16) q[0];
rxx(pi/2) q[2],q[0];
rx(-pi/2) q[0];
rz(-pi/8) q[0];
rxx(pi/2) q[2],q[0];
rx(-pi/2) q[0];
rz(pi/8) q[0];
rxx(pi/2) q[1],q[0];
rx(-pi/2) q[0];
rz(-pi/4) q[0];
rxx(pi/2) q[1],q[0];
rx(pi/2) q[0];
rz(-pi/4) q[0];
ry(-pi/2) q[4];
rxx(pi/2) q[3],q[4];
ry(-pi/2) q[3];
rx(-pi/2) q[4];
ry(pi/2) q[4];
rxx(pi/2) q[4],q[3];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rx(-pi/2) q[4];
ry(-pi/2) q[4];
rxx(pi/2) q[3],q[4];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
rx(-pi/2) q[4];
ry(-pi/2) q[5];
rxx(pi/2) q[2],q[5];
ry(-pi/2) q[2];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rxx(pi/2) q[5],q[2];
rx(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[5];
ry(-pi/2) q[5];
rxx(pi/2) q[2],q[5];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
rx(-pi/2) q[5];
ry(-pi/2) q[6];
rxx(pi/2) q[1],q[6];
ry(-pi/2) q[1];
rx(-pi/2) q[6];
ry(pi/2) q[6];
rxx(pi/2) q[6],q[1];
rx(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[6];
ry(-pi/2) q[6];
rxx(pi/2) q[1],q[6];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
rx(-pi/2) q[6];
ry(-pi/2) q[7];
rxx(pi/2) q[0],q[7];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
rx(-pi/2) q[7];
ry(pi/2) q[7];
rxx(pi/2) q[7],q[0];
rx(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[7];
ry(-pi/2) q[7];
rxx(pi/2) q[0],q[7];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
rx(-pi/2) q[7];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
