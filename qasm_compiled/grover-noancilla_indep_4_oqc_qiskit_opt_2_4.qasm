OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
qreg q[8];
creg meas[4];
rz(-pi/2) q[1];
sx q[1];
rz(-pi/8) q[1];
rz(-pi) q[2];
sx q[2];
rz(-pi) q[2];
ecr q[1],q[2];
x q[1];
rz(-pi/2) q[1];
rz(-pi/8) q[2];
sx q[2];
ecr q[1],q[2];
x q[1];
rz(-pi/2) q[1];
x q[2];
rz(-5*pi/8) q[2];
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
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
ecr q[2],q[3];
sx q[2];
ecr q[1],q[2];
x q[1];
rz(-pi/2) q[1];
x q[2];
rz(-3*pi/8) q[2];
sx q[3];
ecr q[2],q[3];
x q[2];
rz(-pi/2) q[2];
rz(pi/8) q[3];
sx q[3];
ecr q[2],q[3];
sx q[2];
ecr q[1],q[2];
x q[1];
rz(-pi/2) q[1];
x q[2];
rz(-5*pi/8) q[2];
rz(-pi/8) q[3];
sx q[3];
ecr q[2],q[3];
x q[2];
rz(-pi/2) q[2];
rz(-pi/8) q[3];
sx q[3];
ecr q[2],q[3];
x q[2];
rz(-pi/2) q[2];
rz(-7*pi/8) q[3];
sx q[3];
rz(pi/2) q[3];
x q[4];
rz(-pi/2) q[4];
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
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
ecr q[2],q[3];
sx q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
x q[2];
rz(-pi/2) q[2];
rz(7*pi/8) q[3];
sx q[3];
rz(pi/2) q[3];
x q[4];
rz(-pi/2) q[4];
ecr q[4],q[3];
sx q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/8) q[4];
sx q[4];
ecr q[4],q[3];
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
ecr q[2],q[3];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-7*pi/8) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(-pi/8) q[4];
sx q[4];
ecr q[4],q[3];
sx q[3];
rz(pi/2) q[4];
sx q[4];
rz(-pi/8) q[4];
sx q[4];
ecr q[4],q[3];
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
ecr q[2],q[3];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
x q[1];
rz(-pi/2) q[1];
x q[2];
rz(-pi/2) q[2];
rz(7*pi/8) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/8) q[4];
sx q[4];
ecr q[4],q[3];
sx q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/8) q[4];
sx q[4];
ecr q[4],q[3];
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
ecr q[2],q[3];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
x q[1];
rz(-pi/2) q[1];
rz(-3*pi/4) q[2];
sx q[2];
rz(-7*pi/8) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(-pi/8) q[4];
sx q[4];
ecr q[4],q[3];
sx q[3];
rz(pi/2) q[4];
sx q[4];
rz(-pi/8) q[4];
sx q[4];
ecr q[4],q[3];
sx q[3];
rz(-pi/2) q[3];
ecr q[2],q[3];
rz(pi/2) q[2];
sx q[2];
rz(3*pi/4) q[2];
sx q[2];
ecr q[1],q[2];
rz(-pi/4) q[1];
sx q[1];
rz(-3*pi/4) q[2];
sx q[2];
sx q[3];
ecr q[2],q[3];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/4) q[2];
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
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
ecr q[2],q[3];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(-pi/4) q[1];
sx q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi/4) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[2];
sx q[2];
rz(-7*pi/8) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/8) q[4];
sx q[4];
ecr q[4],q[3];
sx q[3];
rz(pi/2) q[4];
sx q[4];
rz(-pi/8) q[4];
sx q[4];
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
sx q[2];
ecr q[1],q[2];
x q[1];
rz(-pi/2) q[1];
sx q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
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
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
ecr q[2],q[3];
rz(-pi/2) q[2];
sx q[2];
rz(7*pi/8) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/8) q[4];
sx q[4];
ecr q[4],q[3];
sx q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/8) q[4];
sx q[4];
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
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
x q[1];
rz(-pi/2) q[1];
x q[2];
rz(-pi/2) q[2];
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
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
ecr q[2],q[3];
rz(-pi/2) q[2];
sx q[2];
rz(-7*pi/8) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(-pi/8) q[4];
sx q[4];
ecr q[4],q[3];
sx q[3];
rz(pi/2) q[4];
sx q[4];
rz(-pi/8) q[4];
sx q[4];
ecr q[4],q[3];
sx q[3];
ecr q[2],q[3];
x q[2];
rz(-pi/2) q[2];
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
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
x q[2];
rz(-pi/2) q[2];
rz(pi/2) q[3];
sx q[3];
rz(-5*pi/8) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/8) q[4];
sx q[4];
ecr q[4],q[3];
sx q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/8) q[4];
sx q[4];
ecr q[4],q[3];
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
ecr q[2],q[3];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-7*pi/8) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(-pi/8) q[4];
sx q[4];
ecr q[4],q[3];
sx q[3];
rz(pi/2) q[4];
sx q[4];
rz(-pi/8) q[4];
sx q[4];
ecr q[4],q[3];
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
ecr q[2],q[3];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
x q[1];
rz(-pi/2) q[1];
x q[2];
rz(-pi/2) q[2];
rz(7*pi/8) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/8) q[4];
sx q[4];
ecr q[4],q[3];
sx q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/8) q[4];
sx q[4];
ecr q[4],q[3];
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
ecr q[2],q[3];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
x q[1];
rz(-pi/2) q[1];
rz(-3*pi/4) q[2];
sx q[2];
rz(-7*pi/8) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(-pi/8) q[4];
sx q[4];
ecr q[4],q[3];
sx q[3];
rz(pi/2) q[4];
sx q[4];
rz(-pi/8) q[4];
sx q[4];
ecr q[4],q[3];
sx q[3];
rz(-pi/2) q[3];
ecr q[2],q[3];
rz(pi/2) q[2];
sx q[2];
rz(3*pi/4) q[2];
sx q[2];
ecr q[1],q[2];
rz(-pi/4) q[1];
sx q[1];
rz(-3*pi/4) q[2];
sx q[2];
sx q[3];
ecr q[2],q[3];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/4) q[2];
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
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
ecr q[2],q[3];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(-pi/4) q[1];
sx q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi/4) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
rz(pi/2) q[4];
sx q[4];
rz(5*pi/8) q[4];
barrier q[4],q[5],q[3],q[2],q[7],q[1],q[0],q[6];
measure q[2] -> meas[0];
measure q[1] -> meas[1];
measure q[3] -> meas[2];
measure q[4] -> meas[3];
