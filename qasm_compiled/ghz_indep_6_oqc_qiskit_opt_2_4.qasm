OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
qreg q[8];
creg meas[6];
rz(-pi/2) q[1];
sx q[1];
sx q[2];
ecr q[1],q[2];
x q[2];
rz(-pi/2) q[2];
sx q[3];
ecr q[2],q[3];
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[4];
sx q[4];
ecr q[4],q[3];
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[4];
sx q[4];
ecr q[4],q[3];
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[4];
sx q[4];
ecr q[4],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[4];
sx q[4];
sx q[5];
ecr q[4],q[5];
rz(-pi) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[6];
sx q[6];
ecr q[6],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
sx q[6];
rz(-pi/2) q[7];
sx q[7];
ecr q[7],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
barrier q[1],q[7],q[3],q[0],q[4],q[6],q[5],q[2];
measure q[7] -> meas[0];
measure q[6] -> meas[1];
measure q[5] -> meas[2];
measure q[4] -> meas[3];
measure q[2] -> meas[4];
measure q[1] -> meas[5];
