OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[13];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[12],q[15];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[13];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi) q[16];
x q[16];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
cx q[12],q[10];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[7],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[19];
sx q[19];
rz(2.9276986) q[19];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[15],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[19],q[22];
sx q[19];
rz(-pi) q[19];
cx q[16],q[19];
sx q[16];
rz(-pi/2) q[16];
sx q[16];
rz(pi/2) q[19];
cx q[16],q[19];
rz(pi/2) q[16];
sx q[16];
rz(-2.9276986) q[16];
sx q[19];
rz(-pi) q[22];
sx q[22];
cx q[21],q[23];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[21],q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(-pi/2) q[25];
cx q[22],q[25];
sx q[22];
rz(-pi/2) q[22];
sx q[22];
rz(pi/2) q[25];
cx q[22],q[25];
sx q[22];
rz(-pi) q[22];
cx q[19],q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
sx q[25];
rz(pi/2) q[25];
barrier q[16],q[22],q[3],q[0],q[6],q[9],q[18],q[7],q[12],q[24],q[21],q[2],q[8],q[5],q[11],q[17],q[14],q[20],q[23],q[26],q[1],q[4],q[13],q[10],q[15],q[19],q[25];
measure q[16] -> meas[0];
measure q[25] -> meas[1];
measure q[19] -> meas[2];
measure q[14] -> meas[3];
measure q[15] -> meas[4];
measure q[12] -> meas[5];
measure q[13] -> meas[6];
measure q[7] -> meas[7];
measure q[18] -> meas[8];
measure q[10] -> meas[9];
measure q[21] -> meas[10];
measure q[23] -> meas[11];
measure q[22] -> meas[12];
