OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[1];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
x q[22];
rz(pi/2) q[22];
cx q[22],q[19];
rz(-pi/2) q[19];
cx q[22],q[19];
rz(-pi) q[19];
sx q[19];
rz(pi/2) q[19];
barrier q[19],q[22];
measure q[19] -> c[0];
