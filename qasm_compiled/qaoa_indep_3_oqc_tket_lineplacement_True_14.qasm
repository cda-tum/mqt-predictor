OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }

qreg node[6];
creg meas[3];
sx node[3];
sx node[4];
sx node[5];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
sx node[3];
sx node[4];
sx node[5];
rz(0.5*pi) node[3];
sx node[4];
x node[5];
x node[3];
rz(3.5*pi) node[5];
rz(3.5*pi) node[3];
ecr node[5],node[4];
rz(0.15947236868659476*pi) node[4];
x node[5];
sx node[4];
rz(3.5*pi) node[5];
ecr node[5],node[4];
sx node[4];
x node[5];
ecr node[3],node[4];
rz(3.5*pi) node[5];
x node[3];
rz(0.15947236868659476*pi) node[4];
rz(3.5*pi) node[3];
sx node[4];
ecr node[3],node[4];
x node[3];
rz(0.5*pi) node[4];
rz(3.5*pi) node[3];
sx node[4];
rz(1.5444445738791162*pi) node[4];
sx node[4];
rz(0.5*pi) node[4];
sx node[4];
ecr node[5],node[4];
x node[4];
sx node[5];
rz(3.5*pi) node[4];
ecr node[4],node[5];
sx node[4];
x node[5];
rz(3.5*pi) node[5];
ecr node[5],node[4];
sx node[4];
sx node[5];
ecr node[3],node[4];
x node[3];
rz(0.15947236868659476*pi) node[4];
rz(3.5*pi) node[3];
sx node[4];
ecr node[3],node[4];
sx node[3];
rz(0.5*pi) node[4];
rz(3.544444573879116*pi) node[3];
sx node[4];
sx node[3];
rz(3.544444573879116*pi) node[4];
rz(1.0*pi) node[3];
sx node[4];
x node[3];
rz(2.5*pi) node[4];
rz(3.5*pi) node[3];
x node[4];
rz(3.5*pi) node[4];
ecr node[4],node[5];
x node[4];
rz(3.7401116731560076*pi) node[5];
rz(3.5*pi) node[4];
sx node[5];
ecr node[4],node[5];
sx node[4];
sx node[5];
ecr node[3],node[4];
sx node[3];
x node[4];
rz(3.5*pi) node[4];
ecr node[4],node[3];
x node[3];
sx node[4];
rz(3.5*pi) node[3];
ecr node[3],node[4];
sx node[3];
x node[4];
rz(3.5*pi) node[4];
ecr node[4],node[5];
x node[4];
rz(3.7401116731560076*pi) node[5];
rz(3.5*pi) node[4];
sx node[5];
ecr node[4],node[5];
x node[4];
rz(0.5*pi) node[5];
rz(3.5*pi) node[4];
sx node[5];
ecr node[4],node[3];
rz(3.728397350920363*pi) node[5];
rz(3.7401116731560076*pi) node[3];
x node[4];
sx node[5];
sx node[3];
rz(3.5*pi) node[4];
rz(0.5*pi) node[5];
ecr node[4],node[3];
rz(0.5*pi) node[3];
sx node[4];
sx node[3];
rz(3.728397350920363*pi) node[4];
rz(3.728397350920363*pi) node[3];
sx node[4];
sx node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[3];
barrier node[4],node[3],node[5];
measure node[4] -> meas[0];
measure node[3] -> meas[1];
measure node[5] -> meas[2];
