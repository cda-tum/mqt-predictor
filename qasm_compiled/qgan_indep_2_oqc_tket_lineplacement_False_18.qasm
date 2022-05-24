OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }

qreg node[2];
creg meas[2];
sx node[0];
rz(0.5*pi) node[1];
rz(0.9402703339703881*pi) node[0];
sx node[1];
sx node[0];
rz(1.5*pi) node[1];
x node[0];
sx node[1];
rz(3.5*pi) node[0];
rz(0.41493627510882347*pi) node[1];
sx node[1];
ecr node[0],node[1];
sx node[0];
rz(0.5861637087812577*pi) node[1];
rz(3.285039247456618*pi) node[0];
sx node[1];
sx node[0];
rz(0.5*pi) node[1];
rz(1.0*pi) node[0];
sx node[1];
rz(1.5*pi) node[1];
barrier node[0],node[1];
measure node[0] -> meas[0];
measure node[1] -> meas[1];
