OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
qreg q[8];
creg meas[5];
rz(-pi/2) q[2];
sx q[2];
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[4];
sx q[4];
ecr q[4],q[3];
rz(2.5155615) q[3];
sx q[3];
x q[4];
rz(-pi/2) q[4];
ecr q[4],q[3];
sx q[3];
ecr q[2],q[3];
x q[2];
rz(-pi/2) q[2];
rz(2.5155615) q[3];
sx q[3];
ecr q[2],q[3];
x q[2];
rz(-pi/2) q[2];
rz(pi/2) q[3];
sx q[3];
rz(-2.8084099) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[4];
sx q[4];
sx q[5];
ecr q[4],q[5];
rz(pi/2) q[4];
sx q[4];
rz(2.5155615) q[4];
sx q[4];
sx q[5];
ecr q[4],q[5];
rz(-2.8084099) q[4];
sx q[4];
ecr q[4],q[3];
rz(2.246459) q[3];
sx q[3];
x q[4];
rz(-pi/2) q[4];
ecr q[4],q[3];
sx q[3];
ecr q[2],q[3];
rz(-pi/2) q[2];
sx q[2];
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
ecr q[2],q[3];
rz(-pi/2) q[2];
sx q[2];
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
ecr q[2],q[3];
rz(-pi/2) q[2];
sx q[2];
sx q[3];
x q[4];
rz(-pi/2) q[4];
sx q[5];
x q[6];
rz(-pi/2) q[6];
ecr q[6],q[5];
rz(-pi) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[6];
sx q[6];
ecr q[6],q[5];
rz(-pi) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[6];
sx q[6];
ecr q[6],q[5];
rz(-pi) q[5];
sx q[5];
rz(pi/2) q[5];
ecr q[4],q[5];
rz(-pi/2) q[4];
sx q[4];
rz(-pi) q[5];
sx q[5];
rz(pi/2) q[5];
ecr q[4],q[5];
rz(-pi/2) q[4];
sx q[4];
rz(-pi) q[5];
sx q[5];
rz(pi/2) q[5];
ecr q[4],q[5];
x q[4];
rz(-pi/2) q[4];
ecr q[4],q[3];
rz(2.5155615) q[3];
sx q[3];
x q[4];
rz(-pi/2) q[4];
ecr q[4],q[3];
rz(pi/2) q[3];
sx q[3];
rz(0.33318277) q[3];
sx q[3];
ecr q[2],q[3];
rz(pi/2) q[2];
sx q[2];
rz(2.246459) q[2];
sx q[2];
sx q[3];
ecr q[2],q[3];
rz(-2.4921964) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
x q[4];
rz(-pi/2) q[4];
rz(-pi) q[5];
sx q[5];
rz(pi/2) q[5];
x q[6];
rz(-pi/2) q[6];
ecr q[6],q[5];
rz(-pi) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[6];
sx q[6];
ecr q[6],q[5];
rz(-pi) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[6];
sx q[6];
ecr q[6],q[5];
rz(-pi) q[5];
sx q[5];
rz(pi/2) q[5];
ecr q[4],q[5];
x q[4];
rz(-pi/2) q[4];
rz(2.5155615) q[5];
sx q[5];
ecr q[4],q[5];
rz(pi/2) q[4];
sx q[4];
rz(-1.2376136) q[4];
sx q[4];
ecr q[4],q[3];
rz(2.246459) q[3];
sx q[3];
x q[4];
rz(-pi/2) q[4];
ecr q[4],q[3];
rz(pi/2) q[3];
sx q[3];
rz(-7.20458535178101) q[3];
sx q[3];
rz(5*pi/2) q[3];
x q[4];
rz(-pi/2) q[4];
rz(pi/2) q[5];
sx q[5];
rz(0.33318277) q[5];
sx q[5];
x q[6];
rz(-pi/2) q[6];
ecr q[6],q[5];
sx q[5];
rz(pi/2) q[6];
sx q[6];
rz(2.246459) q[6];
sx q[6];
ecr q[6],q[5];
rz(-pi) q[5];
sx q[5];
rz(pi/2) q[5];
ecr q[4],q[5];
x q[4];
rz(-pi/2) q[4];
rz(2.246459) q[5];
sx q[5];
ecr q[4],q[5];
rz(pi/2) q[4];
sx q[4];
rz(-7.20458535178101) q[4];
sx q[4];
rz(5*pi/2) q[4];
rz(pi/2) q[5];
sx q[5];
rz(-7.20458535178101) q[5];
sx q[5];
rz(5*pi/2) q[5];
rz(-2.4921964) q[6];
sx q[6];
rz(pi/2) q[6];
barrier q[5],q[3],q[1],q[6],q[7],q[2],q[0],q[4];
measure q[4] -> meas[0];
measure q[5] -> meas[1];
measure q[3] -> meas[2];
measure q[6] -> meas[3];
measure q[2] -> meas[4];
