OPENQASM 2.0;
include "qelib1.inc";

qreg flag[1];
qreg q[1];
creg meas[2];
rz(0.5*pi) flag[0];
rz(0.5*pi) q[0];
rx(1.0*pi) flag[0];
rx(0.5*pi) q[0];
rz(0.5*pi) flag[0];
rz(0.5*pi) q[0];
barrier q[0],flag[0];
measure flag[0] -> meas[1];
measure q[0] -> meas[0];
