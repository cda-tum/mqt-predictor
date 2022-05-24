OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[2];
rz(pi/2) q[118];
sx q[118];
rz(pi/2) q[118];
cx q[118],q[110];
barrier q[110],q[118];
measure q[110] -> meas[0];
measure q[118] -> meas[1];
