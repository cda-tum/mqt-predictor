OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[8];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[19],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[18],q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[14],q[11];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[8],q[5];
cx q[3],q[5];
cx q[5],q[3];
cx q[3],q[5];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[1],q[4];
cx q[4],q[7];
barrier q[22],q[3],q[1],q[11],q[13],q[8],q[17],q[20],q[26],q[0],q[24],q[5],q[9],q[6],q[15],q[14],q[23],q[21],q[18],q[2],q[7],q[4],q[10],q[12],q[19],q[16],q[25];
measure q[7] -> meas[0];
measure q[4] -> meas[1];
measure q[1] -> meas[2];
measure q[8] -> meas[3];
measure q[14] -> meas[4];
measure q[18] -> meas[5];
measure q[25] -> meas[6];
measure q[19] -> meas[7];
