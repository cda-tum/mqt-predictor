OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
creg meas[6];
rz(-3*pi/4) q[4];
ry(-pi/2) q[4];
rz(-0.70927268) q[5];
ry(1.1994265) q[5];
rz(2.7415693) q[5];
rxx(pi/2) q[4],q[5];
rx(pi/2) q[4];
rz(pi) q[4];
rz(-2.8566685) q[5];
ry(2.5935642) q[5];
rz(-0.28492413) q[5];
rxx(pi/2) q[4],q[5];
ry(pi/4) q[4];
rz(pi/2) q[4];
rz(-2.657698) q[5];
ry(2.0062433) q[5];
rz(-0.67631533) q[5];
rxx(pi/2) q[5],q[3];
rx(-pi/2) q[3];
rz(-pi/8) q[3];
rx(-12.222759) q[5];
rxx(pi/2) q[5],q[3];
rx(-pi/2) q[3];
rz(pi/8) q[3];
rxx(pi/2) q[4],q[3];
rx(-pi/2) q[3];
rz(-pi/4) q[3];
rx(-11.879147) q[4];
rxx(pi/2) q[4],q[3];
ry(-pi/4) q[3];
rx(-6.4795348) q[3];
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
ry(-pi/2) q[3];
rxx(pi/2) q[2],q[3];
ry(-pi/2) q[2];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rxx(pi/2) q[3],q[2];
rx(-pi/2) q[2];
ry(pi/2) q[2];
rx(-pi/2) q[3];
ry(-pi/2) q[3];
rxx(pi/2) q[2],q[3];
rx(-pi/2) q[2];
ry(-pi/2) q[2];
rx(-pi/2) q[3];
ry(-pi/2) q[4];
rxx(pi/2) q[1],q[4];
ry(-pi/2) q[1];
rx(-pi/2) q[4];
ry(pi/2) q[4];
rxx(pi/2) q[4],q[1];
rx(-pi/2) q[1];
ry(pi/2) q[1];
rx(-pi/2) q[4];
ry(-pi/2) q[4];
rxx(pi/2) q[1],q[4];
rx(-pi/2) q[1];
ry(-pi/2) q[1];
rx(-pi/2) q[4];
ry(-pi/2) q[5];
rxx(pi/2) q[0],q[5];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rxx(pi/2) q[5],q[0];
rx(-pi/2) q[0];
ry(pi/2) q[0];
rx(-pi/2) q[5];
ry(-pi/2) q[5];
rxx(pi/2) q[0],q[5];
rx(-pi/2) q[0];
ry(-pi/2) q[0];
rx(-pi/2) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
