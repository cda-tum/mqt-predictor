OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
qreg q[8];
creg c[2];
rz(-pi) q[3];
sx q[3];
rz(-pi/2) q[4];
sx q[5];
ecr q[4],q[5];
x q[4];
rz(-pi/2) q[4];
ecr q[4],q[3];
rz(-pi) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
barrier q[5],q[3],q[4];
measure q[5] -> c[0];
measure q[3] -> c[1];
