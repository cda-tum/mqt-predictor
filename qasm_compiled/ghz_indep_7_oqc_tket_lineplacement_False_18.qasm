OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }

qreg node[8];
creg meas[7];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[7];
rz(0.5*pi) node[5];
sx node[5];
x node[5];
rz(3.5*pi) node[5];
ecr node[5],node[4];
x node[4];
rz(3.5*pi) node[4];
ecr node[4],node[3];
x node[3];
rz(3.5*pi) node[3];
ecr node[3],node[2];
x node[2];
rz(3.5*pi) node[2];
ecr node[2],node[1];
x node[1];
rz(3.5*pi) node[1];
ecr node[1],node[0];
x node[0];
rz(3.5*pi) node[0];
ecr node[0],node[7];
barrier node[7],node[0],node[1],node[2],node[3],node[4],node[5];
measure node[7] -> meas[0];
measure node[0] -> meas[1];
measure node[1] -> meas[2];
measure node[2] -> meas[3];
measure node[3] -> meas[4];
measure node[4] -> meas[5];
measure node[5] -> meas[6];
