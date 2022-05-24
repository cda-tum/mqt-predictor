OPENQASM 2.0;
include "qelib1.inc";

qreg node[27];
creg meas[16];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
sx node[7];
sx node[8];
sx node[14];
sx node[16];
sx node[18];
sx node[19];
sx node[20];
sx node[21];
sx node[26];
rz(0.5*pi) node[0];
rz(0.5*pi) node[4];
rz(0.5*pi) node[7];
rz(0.5*pi) node[18];
rz(0.5*pi) node[26];
sx node[0];
sx node[4];
sx node[7];
sx node[18];
sx node[26];
cx node[0],node[1];
cx node[4],node[3];
cx node[7],node[8];
cx node[18],node[19];
cx node[0],node[14];
sx node[1];
sx node[3];
cx node[4],node[5];
cx node[7],node[6];
sx node[8];
sx node[19];
rz(2.5*pi) node[1];
rz(2.5*pi) node[3];
sx node[5];
rz(2.5*pi) node[8];
cx node[18],node[14];
rz(2.5*pi) node[19];
sx node[1];
sx node[3];
rz(2.5*pi) node[5];
sx node[8];
sx node[14];
sx node[19];
rz(1.5*pi) node[1];
rz(1.5*pi) node[3];
sx node[5];
rz(1.5*pi) node[8];
rz(2.5*pi) node[14];
rz(1.5*pi) node[19];
cx node[1],node[2];
rz(1.5*pi) node[5];
cx node[8],node[16];
sx node[14];
cx node[19],node[20];
cx node[3],node[2];
cx node[5],node[6];
rz(1.5*pi) node[14];
cx node[26],node[16];
sx node[20];
sx node[2];
sx node[6];
sx node[16];
rz(2.5*pi) node[20];
cx node[26],node[25];
rz(2.5*pi) node[2];
rz(2.5*pi) node[6];
rz(2.5*pi) node[16];
sx node[20];
cx node[25],node[26];
sx node[2];
sx node[6];
sx node[16];
rz(1.5*pi) node[20];
cx node[26],node[25];
rz(1.5*pi) node[2];
rz(1.5*pi) node[6];
rz(1.5*pi) node[16];
cx node[20],node[21];
cx node[25],node[24];
cx node[24],node[25];
cx node[25],node[24];
cx node[24],node[23];
cx node[23],node[24];
cx node[24],node[23];
cx node[23],node[22];
cx node[22],node[23];
cx node[23],node[22];
cx node[22],node[21];
sx node[21];
rz(2.5*pi) node[21];
sx node[21];
rz(1.5*pi) node[21];
barrier node[0],node[1],node[18],node[19],node[20],node[4],node[3],node[14],node[7],node[8],node[22],node[16],node[5],node[2],node[21],node[6];
measure node[0] -> meas[0];
measure node[1] -> meas[1];
measure node[18] -> meas[2];
measure node[19] -> meas[3];
measure node[20] -> meas[4];
measure node[4] -> meas[5];
measure node[3] -> meas[6];
measure node[14] -> meas[7];
measure node[7] -> meas[8];
measure node[8] -> meas[9];
measure node[22] -> meas[10];
measure node[16] -> meas[11];
measure node[5] -> meas[12];
measure node[2] -> meas[13];
measure node[21] -> meas[14];
measure node[6] -> meas[15];
