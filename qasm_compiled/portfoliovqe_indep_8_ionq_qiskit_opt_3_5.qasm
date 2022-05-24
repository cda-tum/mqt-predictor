OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg meas[8];
ry(2.8597772) q[0];
rz(-pi) q[1];
ry(-1.7851197) q[1];
rxx(pi/2) q[0],q[1];
rx(-3*pi) q[0];
ry(-1.323403) q[2];
rz(-pi) q[2];
rxx(pi/2) q[0],q[2];
rxx(pi/2) q[1],q[2];
rz(-pi) q[3];
ry(-0.073072061) q[3];
rxx(pi/2) q[0],q[3];
rxx(pi/2) q[1],q[3];
rxx(pi/2) q[2],q[3];
rz(-pi) q[4];
ry(-0.13307866) q[4];
rxx(pi/2) q[0],q[4];
rxx(pi/2) q[1],q[4];
rxx(pi/2) q[2],q[4];
rxx(pi/2) q[3],q[4];
ry(-0.16308715) q[5];
rz(-pi) q[5];
rxx(pi/2) q[0],q[5];
rxx(pi/2) q[1],q[5];
rxx(pi/2) q[2],q[5];
rxx(pi/2) q[3],q[5];
rxx(pi/2) q[4],q[5];
ry(-0.80896504) q[6];
rz(-pi) q[6];
rxx(pi/2) q[0],q[6];
rxx(pi/2) q[1],q[6];
rxx(pi/2) q[2],q[6];
rxx(pi/2) q[3],q[6];
rxx(pi/2) q[4],q[6];
rxx(pi/2) q[5],q[6];
ry(-1.4396286) q[7];
rz(-pi) q[7];
rxx(pi/2) q[0],q[7];
rz(2.4220539) q[0];
rx(-7*pi/2) q[0];
rx(-3*pi) q[7];
rxx(pi/2) q[1],q[7];
rz(1.4578201) q[1];
rx(-3*pi/2) q[1];
rxx(pi/2) q[0],q[1];
rxx(pi/2) q[2],q[7];
rz(0.53797297) q[2];
rx(-3*pi/2) q[2];
rxx(pi/2) q[0],q[2];
rxx(pi/2) q[1],q[2];
rxx(pi/2) q[3],q[7];
rz(-1.8367592) q[3];
rx(-3*pi/2) q[3];
rxx(pi/2) q[0],q[3];
rxx(pi/2) q[1],q[3];
rxx(pi/2) q[2],q[3];
rxx(pi/2) q[4],q[7];
rz(1.0184867) q[4];
rx(-3*pi/2) q[4];
rxx(pi/2) q[0],q[4];
rxx(pi/2) q[1],q[4];
rxx(pi/2) q[2],q[4];
rxx(pi/2) q[3],q[4];
rxx(pi/2) q[5],q[7];
rz(-1.5159837) q[5];
rx(-3*pi/2) q[5];
rxx(pi/2) q[0],q[5];
rxx(pi/2) q[1],q[5];
rxx(pi/2) q[2],q[5];
rxx(pi/2) q[3],q[5];
rxx(pi/2) q[4],q[5];
rxx(pi/2) q[6],q[7];
rz(-1.5994918) q[6];
rx(-3*pi/2) q[6];
rxx(pi/2) q[0],q[6];
rxx(pi/2) q[1],q[6];
rxx(pi/2) q[2],q[6];
rxx(pi/2) q[3],q[6];
rxx(pi/2) q[4],q[6];
rxx(pi/2) q[5],q[6];
rz(1.2185984) q[7];
rx(-7*pi/2) q[7];
rxx(pi/2) q[0],q[7];
rz(1.8527783) q[0];
rx(-7*pi/2) q[0];
rxx(pi/2) q[1],q[7];
rz(-1.7164068) q[1];
rx(-3*pi/2) q[1];
rxx(pi/2) q[0],q[1];
rxx(pi/2) q[2],q[7];
rz(-2.3329815) q[2];
rx(-3*pi/2) q[2];
rxx(pi/2) q[0],q[2];
rxx(pi/2) q[1],q[2];
rxx(pi/2) q[3],q[7];
rz(1.8823493) q[3];
rx(-3*pi/2) q[3];
rxx(pi/2) q[0],q[3];
rxx(pi/2) q[1],q[3];
rxx(pi/2) q[2],q[3];
rxx(pi/2) q[4],q[7];
rz(-0.6147805) q[4];
rx(-3*pi/2) q[4];
rxx(pi/2) q[0],q[4];
rxx(pi/2) q[1],q[4];
rxx(pi/2) q[2],q[4];
rxx(pi/2) q[3],q[4];
rxx(pi/2) q[5],q[7];
rz(1.7798283) q[5];
rx(-3*pi/2) q[5];
rxx(pi/2) q[0],q[5];
rxx(pi/2) q[1],q[5];
rxx(pi/2) q[2],q[5];
rxx(pi/2) q[3],q[5];
rxx(pi/2) q[4],q[5];
rxx(pi/2) q[6],q[7];
rz(-1.263099) q[6];
rx(-3*pi/2) q[6];
rxx(pi/2) q[0],q[6];
rxx(pi/2) q[1],q[6];
rxx(pi/2) q[2],q[6];
rxx(pi/2) q[3],q[6];
rxx(pi/2) q[4],q[6];
rxx(pi/2) q[5],q[6];
rz(3.0480363) q[7];
rx(-7*pi/2) q[7];
rxx(pi/2) q[0],q[7];
rz(0.90470315) q[0];
rx(-pi/2) q[0];
rxx(pi/2) q[1],q[7];
rz(2.4904291) q[1];
rx(-pi/2) q[1];
rxx(pi/2) q[2],q[7];
rz(-0.04411489) q[2];
rx(-pi/2) q[2];
rxx(pi/2) q[3],q[7];
rz(-1.4990089) q[3];
rx(-pi/2) q[3];
rxx(pi/2) q[4],q[7];
rz(1.2605606) q[4];
rx(-pi/2) q[4];
rxx(pi/2) q[5],q[7];
rz(0.97609865) q[5];
rx(-pi/2) q[5];
rxx(pi/2) q[6],q[7];
rz(0.46585838) q[6];
rx(-pi/2) q[6];
rz(-2.3356993) q[7];
rx(pi/2) q[7];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
