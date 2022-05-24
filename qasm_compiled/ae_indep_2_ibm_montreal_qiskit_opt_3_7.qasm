OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[2];
rz(pi/2) q[19];
sx q[19];
rz(2.0032313) q[19];
rz(-pi) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[19],q[22];
x q[19];
rz(0.92729522) q[22];
cx q[19],q[22];
rz(2.0032313) q[19];
sx q[19];
rz(-pi/2) q[19];
rz(-2.8577985) q[22];
sx q[22];
rz(-pi) q[22];
barrier q[19],q[22];
measure q[19] -> meas[0];
measure q[22] -> meas[1];
