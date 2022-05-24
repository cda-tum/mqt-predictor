OPENQASM 2.0;
include "qelib1.inc";

qreg node[26];
creg c[2];
sx node[23];
sx node[24];
sx node[25];
rz(0.5*pi) node[23];
rz(0.5*pi) node[24];
rz(3.5*pi) node[25];
sx node[23];
sx node[24];
sx node[25];
rz(0.5*pi) node[23];
rz(0.5*pi) node[24];
cx node[25],node[24];
cx node[24],node[25];
cx node[25],node[24];
cx node[23],node[24];
rz(0.5*pi) node[23];
cx node[25],node[24];
sx node[23];
rz(0.5*pi) node[25];
rz(3.5*pi) node[23];
sx node[25];
sx node[23];
rz(3.5*pi) node[25];
rz(1.0*pi) node[23];
sx node[25];
rz(1.0*pi) node[25];
barrier node[23],node[25],node[24];
measure node[23] -> c[0];
measure node[25] -> c[1];
