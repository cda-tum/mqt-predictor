OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
qreg q[8];
creg meas[3];
rz(-pi/2) q[3];
sx q[3];
rz(-2.0517709) q[3];
sx q[3];
rz(-pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(-2.7200073) q[4];
ecr q[4],q[3];
rz(pi/2) q[3];
sx q[3];
rz(1.0898217) q[3];
rz(1.9923817) q[4];
sx q[4];
rz(-pi) q[4];
sx q[5];
ecr q[4],q[5];
barrier q[5],q[4],q[3];
measure q[5] -> meas[0];
measure q[4] -> meas[1];
measure q[3] -> meas[2];
