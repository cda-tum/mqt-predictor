OPENQASM 2.0;
include "qelib1.inc";

qreg node[19];
creg meas[7];
sx node[4];
rz(0.5*pi) node[4];
sx node[4];
cx node[4],node[3];
cx node[3],node[2];
sx node[4];
cx node[2],node[1];
rz(0.5*pi) node[4];
cx node[1],node[0];
sx node[4];
cx node[0],node[14];
rz(0.49218750000000044*pi) node[4];
cx node[4],node[3];
cx node[14],node[18];
rz(3.75*pi) node[3];
cx node[4],node[3];
rz(0.25*pi) node[3];
sx node[3];
rz(0.5*pi) node[3];
sx node[3];
rz(0.48437500000000044*pi) node[3];
cx node[4],node[3];
cx node[3],node[4];
cx node[4],node[3];
cx node[3],node[2];
rz(3.875*pi) node[2];
cx node[3],node[2];
rz(0.125*pi) node[2];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[1];
cx node[4],node[3];
rz(3.9375*pi) node[1];
rz(3.75*pi) node[3];
cx node[2],node[1];
cx node[4],node[3];
rz(0.0625*pi) node[1];
rz(0.25*pi) node[3];
cx node[2],node[1];
sx node[3];
cx node[1],node[2];
rz(0.5*pi) node[3];
cx node[2],node[1];
sx node[3];
cx node[1],node[0];
rz(0.46875000000000044*pi) node[3];
rz(3.96875*pi) node[0];
cx node[4],node[3];
cx node[1],node[0];
cx node[3],node[4];
rz(0.03125*pi) node[0];
cx node[4],node[3];
cx node[1],node[0];
cx node[3],node[2];
cx node[0],node[1];
rz(3.875*pi) node[2];
cx node[1],node[0];
cx node[3],node[2];
cx node[0],node[14];
rz(0.125*pi) node[2];
cx node[3],node[2];
rz(3.984375*pi) node[14];
cx node[0],node[14];
cx node[2],node[3];
cx node[3],node[2];
rz(0.015625*pi) node[14];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[3];
cx node[14],node[0];
rz(3.9375*pi) node[1];
rz(3.75*pi) node[3];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[3];
rz(0.0625*pi) node[1];
rz(0.25*pi) node[3];
cx node[14],node[18];
cx node[1],node[2];
sx node[3];
rz(3.9921875*pi) node[18];
cx node[2],node[1];
rz(0.5*pi) node[3];
cx node[14],node[18];
cx node[1],node[2];
sx node[3];
rz(0.0078125*pi) node[18];
cx node[1],node[0];
rz(0.43750000000000044*pi) node[3];
cx node[18],node[14];
rz(3.96875*pi) node[0];
cx node[4],node[3];
cx node[14],node[18];
cx node[1],node[0];
cx node[3],node[4];
cx node[18],node[14];
rz(0.03125*pi) node[0];
cx node[4],node[3];
cx node[1],node[0];
cx node[3],node[2];
cx node[0],node[1];
rz(3.875*pi) node[2];
cx node[1],node[0];
cx node[3],node[2];
cx node[0],node[14];
rz(0.125*pi) node[2];
cx node[3],node[2];
rz(3.984375*pi) node[14];
cx node[0],node[14];
cx node[2],node[3];
cx node[3],node[2];
rz(0.015625*pi) node[14];
cx node[14],node[0];
cx node[2],node[1];
cx node[4],node[3];
cx node[0],node[14];
rz(3.9375*pi) node[1];
rz(3.75*pi) node[3];
cx node[14],node[0];
cx node[2],node[1];
cx node[4],node[3];
rz(0.0625*pi) node[1];
rz(0.25*pi) node[3];
cx node[1],node[2];
sx node[3];
cx node[2],node[1];
rz(0.5*pi) node[3];
cx node[1],node[2];
sx node[3];
cx node[1],node[0];
rz(0.37500000000000044*pi) node[3];
rz(3.96875*pi) node[0];
cx node[4],node[3];
cx node[1],node[0];
cx node[3],node[4];
rz(0.03125*pi) node[0];
cx node[4],node[3];
cx node[0],node[1];
cx node[3],node[2];
cx node[1],node[0];
rz(3.875*pi) node[2];
cx node[0],node[1];
cx node[3],node[2];
rz(0.125*pi) node[2];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[1];
cx node[4],node[3];
rz(3.9375*pi) node[1];
rz(3.75*pi) node[3];
cx node[2],node[1];
cx node[4],node[3];
rz(0.0625*pi) node[1];
rz(0.25*pi) node[3];
cx node[1],node[2];
sx node[3];
cx node[2],node[1];
rz(0.5*pi) node[3];
cx node[1],node[2];
sx node[3];
rz(0.25*pi) node[3];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[3];
cx node[4],node[3];
rz(3.875*pi) node[3];
cx node[4],node[3];
rz(0.125*pi) node[3];
cx node[2],node[3];
rz(3.75*pi) node[3];
cx node[2],node[3];
rz(3.25*pi) node[3];
sx node[3];
rz(1.5*pi) node[3];
sx node[3];
rz(1.0*pi) node[3];
barrier node[18],node[14],node[0],node[1],node[4],node[2],node[3];
measure node[18] -> meas[0];
measure node[14] -> meas[1];
measure node[0] -> meas[2];
measure node[1] -> meas[3];
measure node[4] -> meas[4];
measure node[2] -> meas[5];
measure node[3] -> meas[6];
