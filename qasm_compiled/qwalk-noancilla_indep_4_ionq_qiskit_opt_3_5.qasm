OPENQASM 2.0;
include "qelib1.inc";
qreg node[3];
qreg coin[1];
creg meas[4];
ry(-pi/2) node[0];
rz(-7*pi/8) node[0];
rz(pi/8) node[1];
rz(pi/8) node[2];
rx(-13.351769) coin[0];
rxx(pi/2) coin[0],node[1];
rx(-pi/2) node[1];
rz(-pi/8) node[1];
rxx(pi/2) coin[0],node[1];
rx(-pi/2) node[1];
ry(pi/2) node[1];
rxx(pi/2) node[1],node[2];
rx(-3*pi/2) node[1];
rx(-pi/2) node[2];
rz(-pi/8) node[2];
rxx(pi/2) coin[0],node[2];
rx(-pi/2) node[2];
rz(pi/8) node[2];
rxx(pi/2) node[1],node[2];
rx(-pi/2) node[2];
rz(-pi/8) node[2];
rxx(pi/2) coin[0],node[2];
rx(-pi/2) node[2];
ry(pi/2) node[2];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) node[1],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rx(-5*pi/2) node[2];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) coin[0],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) node[1],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) coin[0],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rxx(pi/2) node[2],node[1];
rz(-pi/4) node[1];
rxx(pi/2) coin[0],node[1];
rx(-pi/2) node[1];
rz(pi/4) node[1];
rxx(pi/2) node[2],node[1];
rx(-pi/2) node[1];
rz(-pi/4) node[1];
rxx(pi/2) coin[0],node[1];
ry(pi/4) node[1];
rz(5*pi/8) node[1];
rz(-2.9650852) node[2];
ry(1.7446167) node[2];
rz(-0.76997938) node[2];
rxx(pi/2) node[2],coin[0];
rz(pi) coin[0];
rxx(pi/2) coin[0],node[1];
rx(-10.787848) coin[0];
rx(-pi/2) node[1];
rz(-pi/8) node[1];
rxx(pi/2) coin[0],node[1];
rx(-pi/2) node[1];
ry(pi/2) node[1];
rx(1.3237066) node[2];
rz(-5*pi/8) node[2];
rxx(pi/2) node[1],node[2];
rx(-3*pi/2) node[1];
rx(-pi/2) node[2];
rz(-pi/8) node[2];
rxx(pi/2) coin[0],node[2];
rx(-pi/2) node[2];
rz(pi/8) node[2];
rxx(pi/2) node[1],node[2];
rx(-pi/2) node[2];
rz(-pi/8) node[2];
rxx(pi/2) coin[0],node[2];
rx(-pi/2) node[2];
ry(pi/2) node[2];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) node[1],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rx(-5*pi/2) node[2];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) coin[0],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) node[1],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) coin[0],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rxx(pi/2) node[2],node[1];
rz(-pi/4) node[1];
rxx(pi/2) coin[0],node[1];
rx(-pi/2) node[1];
rz(pi/4) node[1];
rxx(pi/2) node[2],node[1];
rx(-pi/2) node[1];
rz(-pi/4) node[1];
rxx(pi/2) coin[0],node[1];
ry(pi/4) node[1];
rz(5*pi/8) node[1];
rz(-1.5951473) node[2];
ry(2.3560462) node[2];
rz(-0.017217906) node[2];
rxx(pi/2) node[2],coin[0];
rz(2.8075913) coin[0];
ry(1.7813218) coin[0];
rz(-2.5995374) coin[0];
rxx(pi/2) coin[0],node[1];
rx(-10.602875) coin[0];
rx(-pi/2) node[1];
rz(-pi/8) node[1];
rxx(pi/2) coin[0],node[1];
rx(-pi/2) node[1];
ry(pi/2) node[1];
rx(0.017220459) node[2];
rz(-5*pi/8) node[2];
rxx(pi/2) node[1],node[2];
rx(-3*pi/2) node[1];
rx(-pi/2) node[2];
rz(-pi/8) node[2];
rxx(pi/2) coin[0],node[2];
rx(-pi/2) node[2];
rz(pi/8) node[2];
rxx(pi/2) node[1],node[2];
rx(-pi/2) node[2];
rz(-pi/8) node[2];
rxx(pi/2) coin[0],node[2];
rx(-pi/2) node[2];
ry(pi/2) node[2];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) node[1],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rx(-5*pi/2) node[2];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) coin[0],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) node[1],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) coin[0],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rxx(pi/2) node[2],node[1];
rz(-pi/4) node[1];
rxx(pi/2) coin[0],node[1];
rx(-pi/2) node[1];
rz(pi/4) node[1];
rxx(pi/2) node[2],node[1];
rx(-pi/2) node[1];
rz(-pi/4) node[1];
rxx(pi/2) coin[0],node[1];
ry(pi/4) node[1];
rz(5*pi/8) node[1];
rz(-2.9650852) node[2];
ry(1.7446167) node[2];
rz(-0.76997938) node[2];
rxx(pi/2) node[2],coin[0];
rz(pi) coin[0];
rxx(pi/2) coin[0],node[1];
rx(-10.787848) coin[0];
rx(-pi/2) node[1];
rz(-pi/8) node[1];
rxx(pi/2) coin[0],node[1];
rx(-pi/2) node[1];
ry(pi/2) node[1];
rx(1.3237066) node[2];
rz(-5*pi/8) node[2];
rxx(pi/2) node[1],node[2];
rx(-3*pi/2) node[1];
rx(-pi/2) node[2];
rz(-pi/8) node[2];
rxx(pi/2) coin[0],node[2];
rx(-pi/2) node[2];
rz(pi/8) node[2];
rxx(pi/2) node[1],node[2];
rx(-pi/2) node[2];
rz(-pi/8) node[2];
rxx(pi/2) coin[0],node[2];
rx(-pi/2) node[2];
ry(pi/2) node[2];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) node[1],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rx(-5*pi/2) node[2];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) coin[0],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) node[1],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) coin[0],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rxx(pi/2) node[2],node[1];
rz(-pi/4) node[1];
rxx(pi/2) coin[0],node[1];
rx(-pi/2) node[1];
rz(pi/4) node[1];
rxx(pi/2) node[2],node[1];
rx(-pi/2) node[1];
rz(-pi/4) node[1];
rxx(pi/2) coin[0],node[1];
ry(pi/4) node[1];
rz(5*pi/8) node[1];
rz(-1.5951473) node[2];
ry(2.3560462) node[2];
rz(-0.017217906) node[2];
rxx(pi/2) node[2],coin[0];
rz(2.8075913) coin[0];
ry(1.7813218) coin[0];
rz(-2.5995374) coin[0];
rxx(pi/2) coin[0],node[1];
rx(-10.602875) coin[0];
rx(-pi/2) node[1];
rz(-pi/8) node[1];
rxx(pi/2) coin[0],node[1];
rx(-pi/2) node[1];
ry(pi/2) node[1];
rx(0.017220459) node[2];
rz(-5*pi/8) node[2];
rxx(pi/2) node[1],node[2];
rx(-3*pi/2) node[1];
rx(-pi/2) node[2];
rz(-pi/8) node[2];
rxx(pi/2) coin[0],node[2];
rx(-pi/2) node[2];
rz(pi/8) node[2];
rxx(pi/2) node[1],node[2];
rx(-pi/2) node[2];
rz(-pi/8) node[2];
rxx(pi/2) coin[0],node[2];
rx(-pi/2) node[2];
ry(pi/2) node[2];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) node[1],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rx(-5*pi/2) node[2];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) coin[0],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) node[1],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) coin[0],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rxx(pi/2) node[2],node[1];
rz(-pi/4) node[1];
rxx(pi/2) coin[0],node[1];
rx(-pi/2) node[1];
rz(pi/4) node[1];
rxx(pi/2) node[2],node[1];
rx(-pi/2) node[1];
rz(-pi/4) node[1];
rxx(pi/2) coin[0],node[1];
ry(pi/4) node[1];
rz(5*pi/8) node[1];
rz(-2.9650852) node[2];
ry(1.7446167) node[2];
rz(-0.76997938) node[2];
rxx(pi/2) node[2],coin[0];
rz(pi) coin[0];
rxx(pi/2) coin[0],node[1];
rx(-13.417701) coin[0];
rx(-pi/2) node[1];
rz(-pi/8) node[1];
rxx(pi/2) coin[0],node[1];
rx(-pi/2) node[1];
ry(pi/2) node[1];
rx(1.3237066) node[2];
rz(-5*pi/8) node[2];
rxx(pi/2) node[1],node[2];
rx(-3*pi/2) node[1];
rx(-pi/2) node[2];
rz(-pi/8) node[2];
rxx(pi/2) coin[0],node[2];
rx(-pi/2) node[2];
rz(pi/8) node[2];
rxx(pi/2) node[1],node[2];
rx(-pi/2) node[2];
rz(-pi/8) node[2];
rxx(pi/2) coin[0],node[2];
rx(-pi/2) node[2];
ry(pi/2) node[2];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) node[1],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rx(-5*pi/2) node[2];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) coin[0],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) node[1],node[0];
rx(-pi/2) node[0];
rz(pi/8) node[0];
rxx(pi/2) node[2],node[0];
rx(-pi/2) node[0];
rz(-pi/8) node[0];
rxx(pi/2) coin[0],node[0];
ry(-pi/2) node[0];
rz(pi/2) node[0];
rxx(pi/2) node[2],node[1];
rz(-pi/4) node[1];
rxx(pi/2) coin[0],node[1];
rx(-pi/2) node[1];
rz(pi/4) node[1];
rxx(pi/2) node[2],node[1];
rx(-pi/2) node[1];
rz(-pi/4) node[1];
rxx(pi/2) coin[0],node[1];
ry(pi/4) node[1];
rz(pi/2) node[1];
rz(-2.2529699) node[2];
ry(2.2308583) node[2];
rz(-0.4620877) node[2];
rxx(pi/2) node[2],coin[0];
ry(pi/2) coin[0];
rz(-0.065932203) coin[0];
rx(0.52135045) node[2];
rz(-3*pi/4) node[2];
barrier node[0],node[1],node[2],coin[0];
measure node[0] -> meas[0];
measure node[1] -> meas[1];
measure node[2] -> meas[2];
measure coin[0] -> meas[3];
