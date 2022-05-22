OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }

qreg node[8];
creg meas[6];
sx node[0];
rz(0.5*pi) node[1];
sx node[2];
sx node[3];
sx node[6];
sx node[7];
rz(0.5*pi) node[0];
sx node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
sx node[0];
rz(3.5*pi) node[1];
sx node[2];
sx node[3];
sx node[6];
sx node[7];
x node[0];
sx node[1];
x node[2];
x node[3];
rz(0.5*pi) node[6];
x node[7];
rz(3.5*pi) node[0];
rz(0.7951672359369731*pi) node[1];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
x node[6];
rz(3.5*pi) node[7];
sx node[1];
rz(3.5*pi) node[6];
ecr node[0],node[1];
rz(3.96875*pi) node[0];
rz(3.7048327640630268*pi) node[1];
x node[0];
sx node[1];
rz(3.5*pi) node[0];
ecr node[0],node[1];
sx node[0];
rz(0.29516723593697314*pi) node[1];
ecr node[7],node[0];
sx node[1];
x node[0];
ecr node[2],node[1];
sx node[7];
rz(3.5*pi) node[0];
rz(3.409665540858449*pi) node[1];
rz(3.9375*pi) node[2];
ecr node[0],node[7];
sx node[1];
x node[2];
sx node[0];
rz(3.5*pi) node[2];
x node[7];
ecr node[2],node[1];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(0.5903344591415509*pi) node[1];
sx node[2];
x node[0];
sx node[1];
ecr node[3],node[2];
sx node[7];
rz(3.5*pi) node[0];
x node[2];
sx node[3];
ecr node[6],node[7];
ecr node[0],node[1];
rz(3.5*pi) node[2];
sx node[6];
x node[7];
rz(3.875*pi) node[0];
rz(2.3193310498859097*pi) node[1];
ecr node[2],node[3];
rz(3.5*pi) node[7];
x node[0];
sx node[1];
sx node[2];
x node[3];
ecr node[7],node[6];
rz(3.5*pi) node[0];
rz(3.5*pi) node[1];
rz(3.5*pi) node[3];
x node[6];
sx node[7];
sx node[1];
ecr node[3],node[2];
rz(3.5*pi) node[6];
rz(1.5*pi) node[1];
x node[2];
x node[3];
ecr node[6],node[7];
sx node[1];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
x node[6];
x node[7];
ecr node[0],node[1];
rz(3.5*pi) node[6];
rz(3.5*pi) node[7];
sx node[0];
rz(3.5*pi) node[1];
ecr node[7],node[0];
sx node[1];
x node[0];
rz(0.5*pi) node[1];
sx node[7];
rz(3.5*pi) node[0];
sx node[1];
ecr node[0],node[7];
rz(0.6806689523994236*pi) node[1];
sx node[0];
sx node[1];
x node[7];
ecr node[2],node[1];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(1.6386621316028078*pi) node[1];
rz(0.25*pi) node[2];
x node[0];
sx node[1];
x node[2];
x node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[2];
rz(3.5*pi) node[7];
ecr node[2],node[1];
rz(0.3613378706825252*pi) node[1];
x node[2];
sx node[1];
rz(3.5*pi) node[2];
ecr node[0],node[1];
x node[0];
rz(3.2773243905295706*pi) node[1];
rz(3.5*pi) node[0];
sx node[1];
ecr node[0],node[1];
rz(0.5*pi) node[0];
rz(0.22267560947042941*pi) node[1];
sx node[0];
sx node[1];
rz(3.5*pi) node[0];
rz(3.5*pi) node[1];
sx node[0];
sx node[1];
rz(1.0*pi) node[0];
rz(1.5*pi) node[1];
sx node[0];
sx node[1];
ecr node[2],node[1];
x node[1];
sx node[2];
rz(3.5*pi) node[1];
ecr node[1],node[2];
sx node[1];
x node[2];
rz(3.5*pi) node[2];
ecr node[2],node[1];
x node[1];
sx node[2];
rz(3.5*pi) node[1];
ecr node[3],node[2];
ecr node[1],node[0];
x node[2];
sx node[3];
rz(0.25*pi) node[0];
x node[1];
rz(3.5*pi) node[2];
sx node[0];
rz(3.5*pi) node[1];
ecr node[2],node[3];
ecr node[1],node[0];
sx node[2];
x node[3];
rz(3.75*pi) node[0];
rz(0.5*pi) node[1];
rz(3.5*pi) node[3];
sx node[0];
sx node[1];
ecr node[3],node[2];
ecr node[7],node[0];
rz(3.5*pi) node[1];
x node[2];
rz(0.125*pi) node[0];
sx node[1];
rz(3.5*pi) node[2];
rz(0.25*pi) node[7];
sx node[0];
rz(1.0*pi) node[1];
x node[7];
sx node[1];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.875*pi) node[0];
x node[7];
x node[0];
rz(3.5*pi) node[7];
rz(3.5*pi) node[0];
ecr node[0],node[1];
sx node[0];
x node[1];
rz(3.5*pi) node[1];
ecr node[1],node[0];
x node[0];
sx node[1];
rz(3.5*pi) node[0];
ecr node[0],node[1];
sx node[0];
sx node[1];
ecr node[7],node[0];
ecr node[2],node[1];
rz(0.25*pi) node[0];
rz(0.0625*pi) node[1];
rz(3.875*pi) node[2];
x node[7];
sx node[0];
sx node[1];
x node[2];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[2];
rz(3.75*pi) node[0];
ecr node[2],node[1];
rz(0.5*pi) node[7];
x node[0];
rz(3.9375*pi) node[1];
x node[2];
sx node[7];
rz(3.5*pi) node[0];
sx node[1];
rz(3.5*pi) node[2];
rz(3.5*pi) node[7];
ecr node[0],node[1];
sx node[7];
sx node[0];
x node[1];
rz(1.0*pi) node[7];
rz(3.5*pi) node[1];
sx node[7];
ecr node[1],node[0];
x node[0];
sx node[1];
rz(3.5*pi) node[0];
ecr node[0],node[1];
x node[0];
sx node[1];
rz(3.5*pi) node[0];
ecr node[2],node[1];
ecr node[0],node[7];
rz(0.125*pi) node[1];
rz(0.25*pi) node[2];
sx node[0];
sx node[1];
x node[2];
x node[7];
rz(3.5*pi) node[2];
rz(3.5*pi) node[7];
ecr node[7],node[0];
ecr node[2],node[1];
x node[0];
rz(3.875*pi) node[1];
x node[2];
sx node[7];
rz(3.5*pi) node[0];
sx node[1];
rz(3.5*pi) node[2];
ecr node[0],node[7];
x node[0];
sx node[7];
rz(3.5*pi) node[0];
ecr node[6],node[7];
ecr node[0],node[1];
rz(3.9375*pi) node[6];
rz(0.03125*pi) node[7];
sx node[0];
x node[1];
x node[6];
sx node[7];
rz(3.5*pi) node[1];
rz(3.5*pi) node[6];
ecr node[1],node[0];
ecr node[6],node[7];
x node[0];
sx node[1];
x node[6];
rz(3.96875*pi) node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[6];
sx node[7];
ecr node[0],node[1];
ecr node[6],node[7];
sx node[0];
sx node[1];
sx node[6];
x node[7];
ecr node[2],node[1];
rz(3.5*pi) node[7];
rz(0.25*pi) node[1];
x node[2];
ecr node[7],node[6];
sx node[1];
rz(3.5*pi) node[2];
x node[6];
sx node[7];
ecr node[2],node[1];
rz(3.5*pi) node[6];
rz(3.75*pi) node[1];
rz(0.5*pi) node[2];
ecr node[6],node[7];
sx node[1];
sx node[2];
x node[7];
rz(3.5*pi) node[2];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[2];
rz(0.0625*pi) node[0];
rz(1.0*pi) node[2];
rz(3.875*pi) node[7];
sx node[0];
x node[2];
x node[7];
rz(3.5*pi) node[2];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.9375*pi) node[0];
x node[7];
sx node[0];
rz(3.5*pi) node[7];
ecr node[7],node[0];
x node[0];
sx node[7];
rz(3.5*pi) node[0];
ecr node[0],node[7];
sx node[0];
x node[7];
rz(3.5*pi) node[7];
ecr node[7],node[0];
x node[0];
rz(3.5*pi) node[0];
ecr node[0],node[1];
rz(0.25*pi) node[0];
rz(0.125*pi) node[1];
x node[0];
sx node[1];
rz(3.5*pi) node[0];
ecr node[0],node[1];
x node[0];
rz(3.875*pi) node[1];
rz(3.5*pi) node[0];
sx node[1];
ecr node[2],node[1];
x node[1];
sx node[2];
rz(3.5*pi) node[1];
ecr node[1],node[2];
sx node[1];
x node[2];
rz(3.5*pi) node[2];
ecr node[2],node[1];
sx node[1];
ecr node[0],node[1];
x node[0];
rz(0.25*pi) node[1];
rz(3.5*pi) node[0];
sx node[1];
ecr node[0],node[1];
rz(0.5*pi) node[0];
rz(3.75*pi) node[1];
sx node[0];
rz(3.5*pi) node[0];
sx node[0];
rz(1.0*pi) node[0];
barrier node[0],node[1],node[2],node[7],node[6],node[3];
measure node[0] -> meas[0];
measure node[1] -> meas[1];
measure node[2] -> meas[2];
measure node[7] -> meas[3];
measure node[6] -> meas[4];
measure node[3] -> meas[5];
