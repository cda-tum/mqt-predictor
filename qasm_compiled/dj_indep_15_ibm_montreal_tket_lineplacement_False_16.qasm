OPENQASM 2.0;
include "qelib1.inc";

qreg node[16];
creg c[14];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
sx node[7];
sx node[8];
sx node[9];
sx node[10];
sx node[11];
sx node[12];
sx node[13];
sx node[15];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[15];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
sx node[7];
sx node[8];
sx node[9];
sx node[10];
sx node[11];
sx node[12];
sx node[13];
sx node[15];
rz(0.5*pi) node[0];
rz(1.0*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[15];
cx node[0],node[1];
rz(0.5*pi) node[0];
cx node[2],node[1];
sx node[0];
cx node[4],node[1];
rz(0.5*pi) node[2];
rz(3.5*pi) node[0];
sx node[2];
rz(0.5*pi) node[4];
sx node[0];
rz(3.5*pi) node[2];
sx node[4];
rz(1.0*pi) node[0];
sx node[2];
rz(3.5*pi) node[4];
rz(1.0*pi) node[2];
sx node[4];
cx node[3],node[2];
rz(1.0*pi) node[4];
cx node[2],node[3];
cx node[7],node[4];
cx node[3],node[2];
cx node[4],node[7];
cx node[2],node[1];
cx node[5],node[3];
cx node[7],node[4];
cx node[4],node[1];
rz(0.5*pi) node[2];
cx node[3],node[5];
cx node[10],node[7];
sx node[2];
cx node[5],node[3];
rz(0.5*pi) node[4];
cx node[7],node[10];
rz(3.5*pi) node[2];
sx node[4];
cx node[8],node[5];
cx node[10],node[7];
sx node[2];
rz(3.5*pi) node[4];
cx node[5],node[8];
cx node[12],node[10];
rz(1.0*pi) node[2];
sx node[4];
cx node[8],node[5];
cx node[10],node[12];
cx node[3],node[2];
rz(1.0*pi) node[4];
cx node[11],node[8];
cx node[12],node[10];
cx node[2],node[3];
cx node[7],node[4];
cx node[8],node[11];
cx node[13],node[12];
cx node[3],node[2];
cx node[4],node[7];
cx node[11],node[8];
cx node[12],node[13];
cx node[2],node[1];
cx node[5],node[3];
cx node[7],node[4];
cx node[13],node[12];
cx node[4],node[1];
rz(0.5*pi) node[2];
cx node[3],node[5];
cx node[6],node[7];
sx node[2];
cx node[5],node[3];
rz(0.5*pi) node[4];
cx node[7],node[6];
rz(3.5*pi) node[2];
sx node[4];
cx node[8],node[5];
cx node[6],node[7];
sx node[2];
rz(3.5*pi) node[4];
cx node[5],node[8];
rz(1.0*pi) node[2];
sx node[4];
cx node[8],node[5];
cx node[3],node[2];
rz(1.0*pi) node[4];
cx node[9],node[8];
cx node[2],node[3];
cx node[7],node[4];
cx node[8],node[9];
cx node[3],node[2];
cx node[4],node[7];
cx node[9],node[8];
cx node[5],node[3];
cx node[7],node[4];
cx node[4],node[1];
cx node[3],node[5];
cx node[10],node[7];
cx node[2],node[1];
cx node[5],node[3];
rz(0.5*pi) node[4];
cx node[7],node[10];
rz(0.5*pi) node[2];
sx node[4];
cx node[8],node[5];
cx node[10],node[7];
sx node[2];
rz(3.5*pi) node[4];
cx node[5],node[8];
cx node[12],node[10];
rz(3.5*pi) node[2];
sx node[4];
cx node[8],node[5];
cx node[10],node[12];
sx node[2];
rz(1.0*pi) node[4];
cx node[12],node[10];
rz(1.0*pi) node[2];
cx node[7],node[4];
cx node[15],node[12];
cx node[3],node[2];
cx node[4],node[7];
cx node[12],node[15];
cx node[2],node[3];
cx node[7],node[4];
cx node[15],node[12];
cx node[4],node[1];
cx node[3],node[2];
cx node[10],node[7];
cx node[2],node[1];
cx node[5],node[3];
rz(0.5*pi) node[4];
cx node[7],node[10];
rz(0.5*pi) node[2];
cx node[3],node[5];
sx node[4];
cx node[10],node[7];
sx node[2];
cx node[5],node[3];
rz(3.5*pi) node[4];
cx node[12],node[10];
rz(3.5*pi) node[2];
sx node[4];
cx node[10],node[12];
sx node[2];
rz(1.0*pi) node[4];
cx node[12],node[10];
rz(1.0*pi) node[2];
cx node[7],node[4];
cx node[3],node[2];
cx node[4],node[7];
cx node[2],node[3];
cx node[7],node[4];
cx node[3],node[2];
cx node[10],node[7];
cx node[2],node[1];
cx node[7],node[10];
cx node[4],node[1];
rz(0.5*pi) node[2];
cx node[10],node[7];
sx node[2];
rz(0.5*pi) node[4];
rz(3.5*pi) node[2];
sx node[4];
sx node[2];
rz(3.5*pi) node[4];
rz(1.0*pi) node[2];
sx node[4];
rz(1.0*pi) node[4];
cx node[7],node[4];
cx node[4],node[7];
cx node[7],node[4];
cx node[4],node[1];
rz(0.5*pi) node[4];
sx node[4];
rz(3.5*pi) node[4];
sx node[4];
rz(1.0*pi) node[4];
barrier node[0],node[11],node[13],node[9],node[6],node[8],node[15],node[12],node[5],node[10],node[3],node[2],node[7],node[4],node[1];
measure node[0] -> c[0];
measure node[11] -> c[1];
measure node[13] -> c[2];
measure node[9] -> c[3];
measure node[6] -> c[4];
measure node[8] -> c[5];
measure node[15] -> c[6];
measure node[12] -> c[7];
measure node[5] -> c[8];
measure node[10] -> c[9];
measure node[3] -> c[10];
measure node[2] -> c[11];
measure node[7] -> c[12];
measure node[4] -> c[13];
