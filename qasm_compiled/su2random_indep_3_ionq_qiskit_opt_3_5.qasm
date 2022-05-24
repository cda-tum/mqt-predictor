OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg meas[3];
ry(1.9196584) q[0];
rx(-0.94519516) q[0];
ry(0.50169489) q[1];
rz(0.8129628) q[1];
rxx(pi/2) q[0],q[1];
rx(-pi/2) q[1];
ry(pi/2) q[1];
ry(0.88741297) q[2];
rz(0.41316044) q[2];
rxx(pi/2) q[0],q[2];
rz(0.74072738) q[0];
rx(-2.6313171) q[0];
rx(-pi/2) q[2];
rxx(pi/2) q[1],q[2];
rz(-2.8444099) q[1];
ry(pi/2) q[1];
rz(2.2024108) q[1];
rxx(pi/2) q[0],q[1];
rx(-pi/2) q[1];
ry(pi/2) q[1];
rz(-1.4887452) q[2];
ry(pi/2) q[2];
rz(2.2648144) q[2];
rxx(pi/2) q[0],q[2];
rz(0.77075308) q[0];
rx(-2.2799337) q[0];
rx(-pi/2) q[2];
rxx(pi/2) q[1],q[2];
rz(-2.3953714) q[1];
ry(pi/2) q[1];
rz(1.7476975) q[1];
rxx(pi/2) q[0],q[1];
rx(-pi/2) q[1];
ry(pi/2) q[1];
rz(-1.2041146) q[2];
ry(pi/2) q[2];
rz(1.6944101) q[2];
rxx(pi/2) q[0],q[2];
rz(-2.416412) q[0];
ry(pi/2) q[0];
rz(1.7846271) q[0];
rx(-pi/2) q[2];
rxx(pi/2) q[1],q[2];
rz(-3.0085465) q[1];
ry(pi/2) q[1];
rz(2.3502798) q[1];
rz(-1.2752873) q[2];
ry(pi/2) q[2];
rz(1.7906562) q[2];
barrier q[0],q[1],q[2];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
