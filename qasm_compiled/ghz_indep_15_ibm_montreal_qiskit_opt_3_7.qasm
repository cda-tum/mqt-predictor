OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[15];
rz(pi/2) q[19];
rz(pi/2) q[26];
sx q[26];
rz(pi/2) q[26];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[21];
cx q[21],q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[16];
rz(-pi) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[19];
sx q[16];
rz(-pi/2) q[16];
sx q[16];
rz(pi/2) q[19];
cx q[16],q[19];
rz(-pi) q[16];
sx q[16];
rz(-pi) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[14],q[11];
cx q[11],q[8];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi) q[19];
sx q[19];
rz(-pi/2) q[19];
cx q[8],q[5];
cx q[8],q[9];
cx q[9],q[8];
cx q[8],q[9];
cx q[5],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
barrier q[25],q[0],q[3],q[11],q[6],q[12],q[18],q[21],q[15],q[24],q[2],q[5],q[9],q[14],q[8],q[17],q[20],q[26],q[23],q[1],q[7],q[4],q[10],q[16],q[13],q[19],q[22];
measure q[14] -> meas[0];
measure q[11] -> meas[1];
measure q[5] -> meas[2];
measure q[9] -> meas[3];
measure q[8] -> meas[4];
measure q[13] -> meas[5];
measure q[19] -> meas[6];
measure q[16] -> meas[7];
measure q[12] -> meas[8];
measure q[15] -> meas[9];
measure q[21] -> meas[10];
measure q[23] -> meas[11];
measure q[24] -> meas[12];
measure q[25] -> meas[13];
measure q[26] -> meas[14];
