OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg meas[2];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
ry(0.5*pi) q[1];
rxx(0.5*pi) q[1],q[0];
rx(3.5*pi) q[0];
ry(3.5*pi) q[1];
rz(3.5*pi) q[1];
barrier q[0],q[1];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
