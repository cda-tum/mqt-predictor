OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }

qreg node[7];
creg c[3];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
rz(0.5*pi) node[3];
rz(1.0*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
x node[3];
sx node[4];
sx node[5];
x node[6];
rz(3.5*pi) node[3];
rz(3.5*pi) node[6];
ecr node[6],node[5];
x node[5];
x node[6];
rz(3.5*pi) node[5];
rz(3.5*pi) node[6];
ecr node[5],node[4];
sx node[4];
sx node[5];
ecr node[6],node[5];
x node[5];
rz(0.5*pi) node[6];
rz(3.5*pi) node[5];
sx node[6];
ecr node[5],node[4];
rz(3.5*pi) node[6];
sx node[4];
x node[5];
sx node[6];
rz(3.5*pi) node[5];
rz(1.0*pi) node[6];
ecr node[5],node[4];
sx node[4];
rz(0.5*pi) node[5];
ecr node[3],node[4];
sx node[5];
rz(0.5*pi) node[3];
rz(3.5*pi) node[5];
sx node[3];
sx node[5];
rz(3.5*pi) node[3];
rz(1.0*pi) node[5];
sx node[3];
rz(1.0*pi) node[3];
barrier node[6],node[5],node[3],node[4];
measure node[6] -> c[0];
measure node[5] -> c[1];
measure node[3] -> c[2];
