OPENQASM 2.0;
include "qelib1.inc";

qreg node[26];
creg meas[7];
sx node[18];
sx node[19];
sx node[21];
sx node[22];
sx node[23];
sx node[24];
sx node[25];
rz(0.5*pi) node[18];
rz(0.5*pi) node[22];
sx node[18];
sx node[22];
cx node[18],node[21];
cx node[22],node[25];
cx node[19],node[22];
sx node[21];
sx node[25];
cx node[22],node[19];
rz(2.5*pi) node[21];
rz(2.5*pi) node[25];
cx node[19],node[22];
sx node[21];
sx node[25];
rz(1.5*pi) node[21];
rz(1.5*pi) node[25];
cx node[21],node[23];
cx node[25],node[24];
cx node[18],node[21];
sx node[24];
cx node[21],node[18];
rz(2.5*pi) node[24];
cx node[18],node[21];
sx node[24];
rz(1.5*pi) node[24];
cx node[24],node[23];
sx node[23];
rz(2.5*pi) node[23];
sx node[23];
rz(1.5*pi) node[23];
cx node[21],node[23];
cx node[23],node[21];
cx node[21],node[23];
cx node[23],node[24];
cx node[24],node[23];
cx node[23],node[24];
cx node[24],node[25];
cx node[25],node[22];
cx node[24],node[25];
cx node[25],node[22];
cx node[19],node[22];
sx node[22];
rz(2.5*pi) node[22];
sx node[22];
rz(1.5*pi) node[22];
barrier node[24],node[18],node[19],node[25],node[22],node[23],node[21];
measure node[24] -> meas[0];
measure node[18] -> meas[1];
measure node[19] -> meas[2];
measure node[25] -> meas[3];
measure node[22] -> meas[4];
measure node[23] -> meas[5];
measure node[21] -> meas[6];
