OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[22];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
rz(-3*pi/2) q[4];
sx q[4];
rz(-pi/2) q[4];
cx q[1],q[4];
sx q[1];
rz(-pi/2) q[1];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi) q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi) q[7];
cx q[7],q[4];
sx q[7];
rz(pi/2) q[7];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[6],q[7];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[7],q[10];
cx q[10],q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[11];
sx q[11];
rz(pi) q[11];
rz(-pi/2) q[12];
sx q[12];
rz(-pi) q[12];
cx q[12],q[10];
sx q[12];
rz(-pi/2) q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-pi/2) q[13];
sx q[13];
rz(-pi) q[13];
rz(-pi/2) q[14];
sx q[14];
rz(-pi) q[14];
rz(-pi/2) q[15];
sx q[15];
rz(-pi) q[15];
cx q[15],q[12];
sx q[15];
rz(-pi/2) q[15];
rz(-pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[19];
sx q[19];
rz(pi) q[19];
rz(-pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
cx q[13],q[12];
sx q[13];
rz(-pi/2) q[13];
rz(-3*pi/2) q[15];
sx q[15];
rz(-pi/2) q[15];
cx q[17],q[18];
cx q[18],q[17];
cx q[17],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
sx q[14];
rz(-pi/2) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[11],q[14];
sx q[11];
rz(pi/2) q[11];
rz(-3*pi/2) q[15];
sx q[15];
rz(-pi/2) q[15];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[8],q[9];
cx q[9],q[8];
cx q[8],q[9];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[11],q[14];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[19],q[16];
sx q[19];
rz(pi/2) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(-pi/2) q[22];
sx q[22];
rz(-pi) q[22];
cx q[22],q[19];
cx q[16],q[19];
rz(-3*pi/2) q[16];
sx q[16];
rz(-pi/2) q[16];
sx q[22];
rz(-pi/2) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(-pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(-pi/2) q[24];
sx q[24];
rz(-pi) q[24];
rz(pi/2) q[25];
sx q[25];
rz(pi) q[25];
cx q[25],q[22];
sx q[25];
rz(pi/2) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[24],q[25];
sx q[24];
rz(-pi/2) q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[25];
rz(-3*pi/2) q[24];
sx q[24];
rz(-pi/2) q[24];
rz(-pi/2) q[26];
sx q[26];
rz(-pi) q[26];
cx q[26],q[25];
sx q[26];
rz(-pi/2) q[26];
barrier q[8],q[2],q[9],q[5],q[15],q[13],q[20],q[26],q[24],q[1],q[4],q[25],q[7],q[12],q[14],q[16],q[19],q[22],q[0],q[3],q[6],q[10],q[11],q[21],q[17],q[23],q[18];
measure q[1] -> c[0];
measure q[4] -> c[1];
measure q[6] -> c[2];
measure q[10] -> c[3];
measure q[7] -> c[4];
measure q[21] -> c[5];
measure q[17] -> c[6];
measure q[18] -> c[7];
measure q[12] -> c[8];
measure q[15] -> c[9];
measure q[13] -> c[10];
measure q[5] -> c[11];
measure q[9] -> c[12];
measure q[8] -> c[13];
measure q[11] -> c[14];
measure q[14] -> c[15];
measure q[19] -> c[16];
measure q[16] -> c[17];
measure q[22] -> c[18];
measure q[23] -> c[19];
measure q[24] -> c[20];
measure q[26] -> c[21];
