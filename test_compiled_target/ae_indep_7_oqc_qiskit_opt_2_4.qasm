OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
qreg q[8];
creg meas[7];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[1];
sx q[1];
sx q[2];
rz(-0.64350111) q[2];
sx q[2];
rz(pi/2) q[2];
sx q[3];
ecr q[2],q[3];
sx q[2];
rz(0.92729522) q[2];
sx q[2];
rz(-pi/2) q[2];
sx q[3];
ecr q[2],q[3];
rz(-pi) q[2];
sx q[2];
rz(2.4980915) q[2];
ecr q[1],q[2];
x q[1];
rz(-pi/2) q[1];
sx q[2];
rz(-1.8545904) q[2];
ecr q[1],q[2];
x q[1];
rz(-pi/2) q[1];
sx q[2];
rz(1.8545904) q[2];
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
sx q[1];
ecr q[0],q[1];
x q[0];
rz(-pi/2) q[0];
sx q[1];
rz(2.5740044) q[1];
ecr q[0],q[1];
x q[0];
rz(-pi/2) q[0];
sx q[1];
rz(-2.5740044) q[1];
ecr q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
ecr q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
ecr q[0],q[1];
x q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
sx q[3];
sx q[5];
sx q[6];
rz(-pi) q[7];
sx q[7];
rz(pi/2) q[7];
ecr q[0],q[7];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[7];
sx q[7];
rz(pi/2) q[7];
ecr q[0],q[7];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[7];
sx q[7];
rz(pi/2) q[7];
ecr q[0],q[7];
x q[0];
rz(-pi/2) q[0];
rz(-pi/2) q[7];
sx q[7];
ecr q[7],q[6];
sx q[6];
sx q[7];
rz(1.1351764) q[7];
sx q[7];
rz(-pi/2) q[7];
ecr q[7],q[6];
rz(-pi) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi) q[7];
sx q[7];
rz(2.7059727) q[7];
ecr q[0],q[7];
x q[0];
rz(-pi/2) q[0];
sx q[7];
rz(-2.2703524) q[7];
ecr q[0],q[7];
rz(-pi/2) q[0];
sx q[0];
sx q[7];
rz(2.2703529) q[7];
sx q[7];
rz(-pi/2) q[7];
ecr q[7],q[6];
rz(-pi) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[7];
sx q[7];
ecr q[7],q[6];
rz(-pi) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[7];
sx q[7];
ecr q[7],q[6];
rz(-pi/2) q[6];
sx q[6];
ecr q[6],q[5];
sx q[5];
sx q[6];
rz(-1.7424795) q[6];
sx q[6];
rz(-pi/2) q[6];
ecr q[6],q[5];
sx q[5];
sx q[6];
rz(0.17168321) q[6];
sx q[6];
rz(pi/2) q[6];
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
sx q[6];
rz(-pi) q[7];
sx q[7];
rz(pi/2) q[7];
ecr q[0],q[7];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[7];
sx q[7];
rz(pi/2) q[7];
ecr q[0],q[7];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[7];
sx q[7];
rz(pi/2) q[7];
ecr q[0],q[7];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[7];
sx q[7];
rz(pi/4) q[7];
ecr q[7],q[6];
rz(pi/4) q[6];
sx q[6];
x q[7];
rz(-pi/2) q[7];
ecr q[7],q[6];
rz(-pi/4) q[6];
sx q[6];
rz(-pi) q[7];
sx q[7];
rz(pi/2) q[7];
ecr q[0],q[7];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[7];
sx q[7];
rz(pi/2) q[7];
ecr q[0],q[7];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[7];
sx q[7];
rz(pi/2) q[7];
ecr q[0],q[7];
rz(-pi/2) q[0];
sx q[0];
x q[7];
rz(-3*pi/8) q[7];
ecr q[7],q[6];
rz(pi/8) q[6];
sx q[6];
x q[7];
rz(-pi/2) q[7];
ecr q[7],q[6];
rz(7*pi/8) q[6];
sx q[6];
rz(pi/2) q[6];
rz(3*pi/4) q[7];
sx q[7];
rz(pi/2) q[7];
ecr q[0],q[7];
rz(pi/2) q[0];
sx q[0];
rz(pi/4) q[0];
sx q[0];
sx q[7];
ecr q[0],q[7];
rz(pi/2) q[0];
sx q[0];
rz(-pi/4) q[0];
sx q[0];
ecr q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
ecr q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
ecr q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(pi/16) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[7];
sx q[7];
ecr q[7],q[6];
rz(-pi) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[7];
sx q[7];
ecr q[7],q[6];
rz(-pi) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[7];
sx q[7];
ecr q[7],q[6];
rz(-pi) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi) q[7];
sx q[7];
rz(pi/2) q[7];
ecr q[0],q[7];
x q[0];
rz(-pi/2) q[0];
rz(pi/16) q[7];
sx q[7];
ecr q[0],q[7];
x q[0];
rz(-3*pi/8) q[0];
ecr q[0],q[1];
x q[0];
rz(-pi/2) q[0];
rz(pi/8) q[1];
sx q[1];
ecr q[0],q[1];
x q[0];
rz(-pi/2) q[0];
rz(-5*pi/8) q[1];
sx q[1];
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
rz(pi/2) q[1];
sx q[1];
rz(-1.6689711) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/16) q[7];
sx q[7];
ecr q[0],q[7];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[7];
sx q[7];
rz(pi/2) q[7];
ecr q[0],q[7];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[7];
sx q[7];
rz(pi/2) q[7];
ecr q[0],q[7];
rz(-pi/2) q[0];
sx q[0];
ecr q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(pi/32) q[0];
sx q[0];
sx q[1];
ecr q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-pi/32) q[0];
sx q[0];
rz(-pi/2) q[1];
sx q[1];
rz(pi/16) q[1];
ecr q[1],q[2];
x q[1];
rz(-pi/2) q[1];
rz(pi/16) q[2];
sx q[2];
ecr q[1],q[2];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
ecr q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
ecr q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
ecr q[0],q[1];
x q[0];
rz(-pi/2) q[0];
x q[1];
rz(-pi/2) q[1];
rz(-9*pi/16) q[2];
sx q[2];
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
rz(pi/2) q[2];
sx q[2];
rz(-1.6198837) q[2];
sx q[2];
rz(pi/2) q[2];
ecr q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(pi/64) q[1];
sx q[1];
sx q[2];
ecr q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(-pi/64) q[1];
sx q[1];
rz(-pi/2) q[2];
sx q[2];
rz(pi/32) q[2];
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
ecr q[2],q[3];
x q[2];
rz(-pi/2) q[2];
rz(pi/32) q[3];
sx q[3];
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
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/32) q[3];
x q[7];
rz(-pi/4) q[7];
ecr q[7],q[6];
rz(pi/4) q[6];
sx q[6];
x q[7];
rz(-pi/2) q[7];
ecr q[7],q[6];
rz(-pi/4) q[6];
sx q[6];
sx q[7];
ecr q[0],q[7];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[7];
sx q[7];
rz(pi/2) q[7];
ecr q[0],q[7];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[7];
sx q[7];
rz(pi/2) q[7];
ecr q[0],q[7];
x q[0];
rz(-pi/2) q[0];
rz(-pi/2) q[7];
sx q[7];
rz(pi/8) q[7];
ecr q[7],q[6];
rz(pi/8) q[6];
sx q[6];
x q[7];
rz(-pi/2) q[7];
ecr q[7],q[6];
rz(-pi/8) q[6];
sx q[6];
rz(3*pi/4) q[7];
sx q[7];
rz(pi/2) q[7];
ecr q[0],q[7];
rz(pi/2) q[0];
sx q[0];
rz(pi/4) q[0];
sx q[0];
sx q[7];
ecr q[0],q[7];
rz(-pi/2) q[0];
sx q[0];
rz(pi/4) q[0];
ecr q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
ecr q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
ecr q[0],q[1];
x q[0];
rz(-7*pi/16) q[0];
sx q[1];
x q[7];
rz(-pi/2) q[7];
ecr q[7],q[6];
rz(-pi) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[7];
sx q[7];
ecr q[7],q[6];
rz(-pi) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[7];
sx q[7];
ecr q[7],q[6];
rz(-pi) q[6];
sx q[6];
rz(pi/2) q[6];
sx q[7];
ecr q[0],q[7];
x q[0];
rz(-pi/2) q[0];
rz(pi/16) q[7];
sx q[7];
ecr q[0],q[7];
x q[0];
rz(-3*pi/8) q[0];
ecr q[0],q[1];
x q[0];
rz(-pi/2) q[0];
rz(pi/8) q[1];
sx q[1];
ecr q[0],q[1];
x q[0];
rz(-pi/4) q[0];
rz(-pi/8) q[1];
rz(-9*pi/16) q[7];
sx q[7];
ecr q[7],q[6];
rz(-pi) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[7];
sx q[7];
ecr q[7],q[6];
rz(-pi) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[7];
sx q[7];
ecr q[7],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi) q[7];
sx q[7];
rz(pi/2) q[7];
ecr q[0],q[7];
x q[0];
rz(-pi/2) q[0];
rz(pi/4) q[7];
sx q[7];
ecr q[0],q[7];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/4) q[7];
barrier q[1],q[6],q[0],q[5],q[2],q[4],q[7],q[3];
measure q[0] -> meas[0];
measure q[7] -> meas[1];
measure q[1] -> meas[2];
measure q[6] -> meas[3];
measure q[3] -> meas[4];
measure q[2] -> meas[5];
measure q[5] -> meas[6];
