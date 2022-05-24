OPENQASM 2.0;
include "qelib1.inc";

qreg node[26];
creg c[5];
sx node[15];
sx node[18];
sx node[21];
sx node[23];
sx node[24];
sx node[25];
rz(0.5*pi) node[15];
rz(0.5*pi) node[18];
rz(0.5*pi) node[21];
rz(0.5*pi) node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
sx node[15];
sx node[18];
sx node[21];
sx node[23];
sx node[24];
sx node[25];
rz(0.5*pi) node[15];
rz(0.5*pi) node[18];
rz(0.5*pi) node[21];
rz(0.5*pi) node[23];
rz(0.5*pi) node[24];
rz(1.0*pi) node[25];
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
cx node[24],node[23];
rz(1.0*pi) node[25];
cx node[23],node[24];
cx node[24],node[23];
cx node[21],node[23];
rz(0.5*pi) node[21];
sx node[21];
rz(3.5*pi) node[21];
sx node[21];
rz(1.0*pi) node[21];
cx node[23],node[21];
cx node[21],node[23];
cx node[23],node[21];
cx node[18],node[21];
rz(0.5*pi) node[18];
sx node[18];
rz(3.5*pi) node[18];
sx node[18];
rz(1.0*pi) node[18];
cx node[21],node[18];
cx node[18],node[21];
cx node[21],node[18];
cx node[15],node[18];
rz(0.5*pi) node[15];
sx node[15];
rz(3.5*pi) node[15];
sx node[15];
rz(1.0*pi) node[15];
barrier node[24],node[25],node[23],node[21],node[15],node[18];
measure node[24] -> c[0];
measure node[25] -> c[1];
measure node[23] -> c[2];
measure node[21] -> c[3];
measure node[15] -> c[4];
