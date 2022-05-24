OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
qreg q[8];
creg c[1];
rz(pi/2) q[5];
sx q[5];
rz(-1.1746127) q[5];
sx q[5];
rz(-pi/2) q[5];
ecr q[6],q[5];
rz(-pi/2) q[5];
sx q[5];
rz(-0.39618364) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi) q[6];
sx q[6];
rz(pi/2) q[6];
barrier q[5],q[6];
measure q[5] -> c[0];
