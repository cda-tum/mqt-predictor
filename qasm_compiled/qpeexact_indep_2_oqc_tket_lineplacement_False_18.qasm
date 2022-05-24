OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }

qreg node[2];
creg c[1];
x node[0];
sx node[1];
x node[0];
rz(3.5*pi) node[0];
ecr node[0],node[1];
barrier node[1],node[0];
measure node[1] -> c[0];
