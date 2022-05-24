OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[1];
rz(3.5*pi) q[0];
rz(0.5*pi) q[1];
rx(1.5*pi) q[0];
rx(1.5*pi) q[1];
ry(0.5*pi) q[0];
rz(0.5*pi) q[1];
rxx(0.5*pi) q[0],q[1];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
rz(3.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
barrier q[0],q[1];
measure q[0] -> c[0];
