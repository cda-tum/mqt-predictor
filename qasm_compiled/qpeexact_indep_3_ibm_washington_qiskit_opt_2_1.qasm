OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg c[2];
x q[91];
rz(pi/2) q[91];
rz(pi/2) q[97];
sx q[97];
rz(pi/2) q[97];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
cx q[91],q[98];
rz(-pi/2) q[98];
cx q[91],q[98];
rz(pi/2) q[98];
cx q[98],q[97];
cx q[97],q[98];
cx q[98],q[97];
rz(pi/4) q[97];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
cx q[97],q[98];
rz(pi/4) q[98];
cx q[97],q[98];
sx q[97];
rz(pi/2) q[97];
rz(-pi/4) q[98];
barrier q[98],q[97],q[91];
measure q[98] -> c[0];
measure q[97] -> c[1];
