OPENQASM 2.0;
include "qelib1.inc";

qreg node[26];
creg c[8];
sx node[10];
sx node[12];
sx node[15];
sx node[17];
sx node[18];
sx node[21];
sx node[23];
sx node[24];
sx node[25];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
rz(0.5*pi) node[15];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[21];
rz(0.5*pi) node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
sx node[10];
sx node[12];
sx node[15];
sx node[17];
sx node[18];
sx node[21];
sx node[23];
sx node[24];
sx node[25];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
rz(0.5*pi) node[15];
rz(0.5*pi) node[17];
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
cx node[17],node[18];
sx node[15];
rz(0.5*pi) node[17];
rz(3.5*pi) node[15];
sx node[17];
sx node[15];
rz(3.5*pi) node[17];
rz(1.0*pi) node[15];
sx node[17];
cx node[18],node[15];
rz(1.0*pi) node[17];
cx node[15],node[18];
cx node[18],node[15];
cx node[12],node[15];
rz(0.5*pi) node[12];
sx node[12];
rz(3.5*pi) node[12];
sx node[12];
rz(1.0*pi) node[12];
cx node[15],node[12];
cx node[12],node[15];
cx node[15],node[12];
cx node[10],node[12];
rz(0.5*pi) node[10];
sx node[10];
rz(3.5*pi) node[10];
sx node[10];
rz(1.0*pi) node[10];
barrier node[24],node[25],node[23],node[21],node[18],node[17],node[15],node[10],node[12];
measure node[24] -> c[0];
measure node[25] -> c[1];
measure node[23] -> c[2];
measure node[21] -> c[3];
measure node[18] -> c[4];
measure node[17] -> c[5];
measure node[15] -> c[6];
measure node[10] -> c[7];
