OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
qreg q[8];
creg meas[2];
rz(2.5933277) q[5];
sx q[5];
rz(-1.4101434) q[5];
sx q[5];
rz(3.0442091) q[5];
sx q[6];
rz(1.3035608) q[6];
sx q[6];
ecr q[6],q[5];
rz(-2.3221129) q[5];
sx q[5];
rz(-1.9840244) q[5];
sx q[5];
rz(-1.2119486) q[5];
rz(pi/2) q[6];
sx q[6];
rz(1.8414876) q[6];
sx q[6];
barrier q[5],q[6];
measure q[5] -> meas[0];
measure q[6] -> meas[1];
