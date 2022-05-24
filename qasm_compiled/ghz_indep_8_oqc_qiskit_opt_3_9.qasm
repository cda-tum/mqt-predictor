OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
qreg q[8];
creg meas[8];
rz(-pi/2) q[0];
sx q[0];
rz(2.6587688) q[0];
sx q[1];
sx q[2];
sx q[3];
rz(-pi/2) q[4];
sx q[4];
rz(-0.5555773) q[4];
rz(-pi/2) q[5];
sx q[5];
rz(-2.4949847) q[5];
sx q[5];
rz(pi/2) q[5];
ecr q[4],q[5];
rz(2.5860154) q[4];
rz(-0.047564056) q[5];
sx q[5];
rz(-1.6336994) q[5];
sx q[5];
rz(-0.92269162) q[5];
rz(pi/2) q[6];
sx q[6];
rz(0.51426249) q[6];
ecr q[6],q[5];
rz(-pi/2) q[5];
sx q[5];
rz(-3.0627501) q[5];
rz(-1.0565338) q[6];
sx q[6];
rz(-0.078842588) q[6];
sx q[6];
rz(-pi/2) q[6];
rz(pi/2) q[7];
sx q[7];
rz(0.51426249) q[7];
ecr q[7],q[6];
rz(-pi/2) q[6];
sx q[6];
rz(-3.0627501) q[6];
rz(2.0850588) q[7];
sx q[7];
rz(-0.4403808) q[7];
sx q[7];
rz(-pi/2) q[7];
ecr q[0],q[7];
rz(1.0879725) q[0];
sx q[0];
ecr q[0],q[1];
x q[1];
rz(-pi/2) q[1];
ecr q[1],q[2];
x q[2];
rz(-pi/2) q[2];
ecr q[2],q[3];
rz(pi/2) q[7];
sx q[7];
rz(2.7012119) q[7];
barrier q[3],q[2],q[1],q[0],q[7],q[6],q[5],q[4];
measure q[3] -> meas[0];
measure q[2] -> meas[1];
measure q[1] -> meas[2];
measure q[0] -> meas[3];
measure q[7] -> meas[4];
measure q[6] -> meas[5];
measure q[5] -> meas[6];
measure q[4] -> meas[7];
