OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[2];
sx q[19];
rz(-2.1632116) q[19];
sx q[19];
rz(-2.1998708) q[19];
sx q[22];
rz(-2.5296223) q[22];
sx q[22];
rz(-2.5619687) q[22];
cx q[19],q[22];
sx q[19];
rz(-2.2159169) q[19];
sx q[19];
rz(-2.2680618) q[19];
sx q[22];
rz(-3.0542003) q[22];
sx q[22];
rz(-2.9562523) q[22];
cx q[19],q[22];
sx q[19];
rz(-3.0573236) q[19];
sx q[19];
rz(-2.9404197) q[19];
sx q[22];
rz(-3.1025201) q[22];
sx q[22];
rz(-2.715443) q[22];
cx q[19],q[22];
sx q[19];
rz(-2.8801165) q[19];
sx q[19];
rz(-2.1869475) q[19];
sx q[22];
rz(-3.062073) q[22];
sx q[22];
rz(-2.6331289) q[22];
barrier q[19],q[22];
measure q[19] -> meas[0];
measure q[22] -> meas[1];
