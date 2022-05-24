OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[2];
rz(-pi) q[110];
sx q[110];
rz(0.27644941) q[110];
sx q[110];
sx q[118];
rz(1.3049033) q[118];
sx q[118];
rz(-pi) q[118];
cx q[110],q[118];
rz(-pi) q[110];
sx q[110];
rz(1.0779126) q[110];
sx q[110];
sx q[118];
rz(0.93796724) q[118];
sx q[118];
rz(-pi) q[118];
cx q[110],q[118];
rz(-pi) q[110];
sx q[110];
rz(3.0611389) q[110];
sx q[110];
sx q[118];
rz(0.85400806) q[118];
sx q[118];
rz(-pi) q[118];
cx q[110],q[118];
rz(-pi) q[110];
sx q[110];
rz(1.9538414) q[110];
sx q[110];
rz(-pi) q[118];
sx q[118];
rz(0.99432837) q[118];
sx q[118];
barrier q[110],q[118];
measure q[110] -> meas[0];
measure q[118] -> meas[1];
