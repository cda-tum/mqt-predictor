OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[2];
sx q[110];
rz(pi/4) q[110];
sx q[110];
x q[118];
cx q[118],q[110];
sx q[110];
rz(pi/4) q[110];
sx q[110];
cx q[110],q[118];
barrier q[110],q[118];
measure q[110] -> meas[0];
measure q[118] -> meas[1];
