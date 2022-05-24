OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[13];
cx q[12],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[19];
cx q[16],q[14];
cx q[13],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[19],q[22];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[20];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[11],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
barrier q[20],q[25],q[3],q[0],q[6],q[9],q[15],q[10],q[18],q[24],q[21],q[2],q[5],q[11],q[13],q[17],q[8],q[14],q[23],q[26],q[1],q[4],q[12],q[7],q[16],q[19],q[22];
measure q[19] -> meas[0];
measure q[20] -> meas[1];
measure q[10] -> meas[2];
measure q[16] -> meas[3];
measure q[13] -> meas[4];
measure q[5] -> meas[5];
measure q[22] -> meas[6];
measure q[12] -> meas[7];
measure q[8] -> meas[8];
measure q[11] -> meas[9];
measure q[14] -> meas[10];
