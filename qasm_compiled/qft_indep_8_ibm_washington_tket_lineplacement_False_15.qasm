OPENQASM 2.0;
include "qelib1.inc";

qreg node[22];
creg c[8];
creg meas[8];
sx node[1];
rz(0.5*pi) node[1];
sx node[1];
rz(0.49609375000000044*pi) node[1];
cx node[1],node[2];
rz(3.75*pi) node[2];
cx node[1],node[2];
cx node[1],node[0];
rz(0.25*pi) node[2];
rz(3.875*pi) node[0];
sx node[2];
cx node[1],node[0];
rz(0.5*pi) node[2];
rz(0.125*pi) node[0];
sx node[2];
cx node[0],node[1];
rz(0.49218750000000044*pi) node[2];
cx node[1],node[0];
cx node[0],node[1];
cx node[0],node[14];
cx node[2],node[1];
rz(3.75*pi) node[1];
rz(3.9375*pi) node[14];
cx node[0],node[14];
cx node[2],node[1];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[14];
cx node[0],node[14];
sx node[1];
cx node[14],node[0];
rz(0.5*pi) node[1];
cx node[0],node[14];
sx node[1];
rz(0.48437500000000044*pi) node[1];
cx node[14],node[18];
cx node[2],node[1];
rz(3.96875*pi) node[18];
cx node[1],node[2];
cx node[14],node[18];
cx node[2],node[1];
rz(0.03125*pi) node[18];
cx node[1],node[0];
cx node[14],node[18];
rz(3.875*pi) node[0];
cx node[18],node[14];
cx node[1],node[0];
cx node[14],node[18];
rz(0.125*pi) node[0];
cx node[18],node[19];
cx node[0],node[1];
rz(3.984375*pi) node[19];
cx node[1],node[0];
cx node[18],node[19];
cx node[0],node[1];
rz(0.015625*pi) node[19];
cx node[0],node[14];
cx node[2],node[1];
cx node[19],node[18];
rz(3.75*pi) node[1];
rz(3.9375*pi) node[14];
cx node[18],node[19];
cx node[0],node[14];
cx node[2],node[1];
cx node[19],node[18];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[14];
cx node[19],node[20];
cx node[0],node[14];
sx node[1];
rz(3.9921875*pi) node[20];
cx node[14],node[0];
rz(0.5*pi) node[1];
cx node[19],node[20];
cx node[0],node[14];
sx node[1];
rz(0.0078125*pi) node[20];
rz(0.46875000000000044*pi) node[1];
cx node[14],node[18];
cx node[20],node[19];
cx node[2],node[1];
rz(3.96875*pi) node[18];
cx node[19],node[20];
cx node[1],node[2];
cx node[14],node[18];
cx node[20],node[19];
cx node[2],node[1];
rz(0.03125*pi) node[18];
cx node[20],node[21];
cx node[1],node[0];
cx node[14],node[18];
rz(3.99609375*pi) node[21];
rz(3.875*pi) node[0];
cx node[18],node[14];
cx node[20],node[21];
cx node[1],node[0];
cx node[14],node[18];
rz(0.00390625*pi) node[21];
rz(0.125*pi) node[0];
cx node[18],node[19];
cx node[21],node[20];
cx node[0],node[1];
rz(3.984375*pi) node[19];
cx node[20],node[21];
cx node[1],node[0];
cx node[18],node[19];
cx node[21],node[20];
cx node[0],node[1];
rz(0.015625*pi) node[19];
cx node[0],node[14];
cx node[2],node[1];
cx node[18],node[19];
rz(3.75*pi) node[1];
rz(3.9375*pi) node[14];
cx node[19],node[18];
cx node[0],node[14];
cx node[2],node[1];
cx node[18],node[19];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[14];
cx node[19],node[20];
cx node[0],node[14];
sx node[1];
rz(3.9921875*pi) node[20];
cx node[14],node[0];
rz(0.5*pi) node[1];
cx node[19],node[20];
cx node[0],node[14];
sx node[1];
rz(0.0078125*pi) node[20];
rz(0.43750000000000044*pi) node[1];
cx node[14],node[18];
cx node[20],node[19];
cx node[2],node[1];
rz(3.96875*pi) node[18];
cx node[19],node[20];
cx node[1],node[2];
cx node[14],node[18];
cx node[20],node[19];
cx node[2],node[1];
rz(0.03125*pi) node[18];
cx node[1],node[0];
cx node[14],node[18];
rz(3.875*pi) node[0];
cx node[18],node[14];
cx node[1],node[0];
cx node[14],node[18];
rz(0.125*pi) node[0];
cx node[18],node[19];
cx node[0],node[1];
rz(3.984375*pi) node[19];
cx node[1],node[0];
cx node[18],node[19];
cx node[0],node[1];
rz(0.015625*pi) node[19];
cx node[0],node[14];
cx node[2],node[1];
cx node[19],node[18];
rz(3.75*pi) node[1];
rz(3.9375*pi) node[14];
cx node[18],node[19];
cx node[0],node[14];
cx node[2],node[1];
cx node[19],node[18];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[14];
cx node[14],node[0];
sx node[1];
cx node[0],node[14];
rz(0.5*pi) node[1];
cx node[14],node[0];
sx node[1];
rz(0.37500000000000044*pi) node[1];
cx node[14],node[18];
cx node[2],node[1];
rz(3.96875*pi) node[18];
cx node[1],node[2];
cx node[14],node[18];
cx node[2],node[1];
rz(0.03125*pi) node[18];
cx node[1],node[0];
cx node[18],node[14];
rz(3.875*pi) node[0];
cx node[14],node[18];
cx node[1],node[0];
cx node[18],node[14];
rz(0.125*pi) node[0];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[0],node[14];
cx node[2],node[1];
rz(3.75*pi) node[1];
rz(3.9375*pi) node[14];
cx node[0],node[14];
cx node[2],node[1];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[14];
cx node[14],node[0];
sx node[1];
cx node[0],node[14];
rz(0.5*pi) node[1];
cx node[14],node[0];
sx node[1];
rz(0.25*pi) node[1];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[2],node[1];
rz(3.875*pi) node[1];
cx node[2],node[1];
rz(0.125*pi) node[1];
cx node[0],node[1];
rz(3.75*pi) node[1];
cx node[0],node[1];
rz(3.25*pi) node[1];
sx node[1];
rz(1.5*pi) node[1];
sx node[1];
rz(1.0*pi) node[1];
barrier node[21],node[20],node[19],node[18],node[14],node[2],node[0],node[1];
measure node[21] -> meas[0];
measure node[20] -> meas[1];
measure node[19] -> meas[2];
measure node[18] -> meas[3];
measure node[14] -> meas[4];
measure node[2] -> meas[5];
measure node[0] -> meas[6];
measure node[1] -> meas[7];
