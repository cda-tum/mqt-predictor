OPENQASM 2.0;
include "qelib1.inc";
qreg eval[3];
qreg q[1];
creg meas[4];
rx(1.7149692) eval[0];
rx(0.41837352) eval[1];
rz(pi) eval[1];
rx(2.4035937) eval[2];
rz(3*pi/4) q[0];
ry(pi/2) q[0];
rxx(pi/2) eval[0],q[0];
rz(-2.8017557) q[0];
ry(2.4980915) q[0];
rz(-0.33983691) q[0];
rz(pi) eval[0];
rxx(pi/2) eval[0],q[0];
rz(2.3856232) q[0];
ry(1.9390642) q[0];
rz(0.75596941) q[0];
rx(-1.0339243) eval[0];
rz(pi) eval[0];
rxx(pi/2) eval[1],q[0];
rz(-2.6539764) q[0];
ry(2.2652946) q[0];
rz(-0.48761626) q[0];
rx(pi/2) eval[1];
rz(pi) eval[1];
rxx(pi/2) eval[1],q[0];
rz(0.48761626) q[0];
ry(2.2652946) q[0];
rz(2.6539764) q[0];
rx(-1.2037717) eval[1];
rxx(pi/2) eval[2],q[0];
rz(-2.9382048) q[0];
ry(2.742979) q[0];
rz(-0.20338781) q[0];
rz(pi) eval[2];
rxx(pi/2) eval[2],q[0];
rx(pi/4) q[0];
ry(-1.0032081) q[0];
rx(-0.73799894) eval[2];
rz(pi) eval[2];
rxx(pi/2) eval[1],eval[2];
rx(-pi/2) eval[2];
rz(pi/4) eval[2];
rxx(pi/2) eval[1],eval[2];
rx(-pi/2) eval[2];
rz(-pi/4) eval[2];
rxx(pi/2) eval[0],eval[2];
rx(-5*pi/4) eval[0];
rx(-pi/2) eval[2];
rz(pi/8) eval[2];
rxx(pi/2) eval[0],eval[2];
rxx(pi/2) eval[0],eval[1];
rz(pi/4) eval[1];
rxx(pi/2) eval[0],eval[1];
rx(-pi/2) eval[1];
rz(-pi/4) eval[1];
rx(-pi/2) eval[2];
rz(-pi/8) eval[2];
barrier eval[0],eval[1],eval[2],q[0];
measure eval[0] -> meas[0];
measure eval[1] -> meas[1];
measure eval[2] -> meas[2];
measure q[0] -> meas[3];
