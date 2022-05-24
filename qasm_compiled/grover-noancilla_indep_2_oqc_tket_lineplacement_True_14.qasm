OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }

qreg node[6];
creg meas[2];
sx node[4];
x node[5];
rz(0.5*pi) node[4];
sx node[4];
barrier node[4],node[5];
measure node[4] -> meas[0];
measure node[5] -> meas[1];
