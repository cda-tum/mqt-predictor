OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[13];
sx q[10];
rz(2*pi/3) q[10];
sx q[10];
rz(-pi) q[10];
sx q[11];
rz(3*pi/4) q[11];
sx q[11];
rz(-pi) q[11];
sx q[12];
rz(0.33983693) q[12];
sx q[12];
sx q[13];
rz(0.32175053) q[13];
sx q[13];
sx q[14];
rz(0.30627733) q[14];
sx q[14];
sx q[15];
rz(2.186276) q[15];
sx q[15];
rz(-pi) q[15];
sx q[16];
rz(0.29284273) q[16];
sx q[16];
sx q[18];
rz(0.36136713) q[18];
sx q[18];
sx q[19];
rz(0.28103493) q[19];
sx q[19];
sx q[21];
rz(2.034444) q[21];
sx q[21];
rz(-pi) q[21];
x q[22];
cx q[22],q[19];
sx q[19];
rz(0.28103493) q[19];
sx q[19];
cx q[19],q[16];
sx q[16];
rz(0.29284273) q[16];
sx q[16];
cx q[16],q[14];
sx q[14];
rz(0.30627733) q[14];
sx q[14];
cx q[14],q[13];
sx q[13];
rz(0.32175053) q[13];
sx q[13];
cx q[13],q[12];
sx q[12];
rz(0.33983693) q[12];
sx q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[18];
sx q[18];
rz(0.36136713) q[18];
sx q[18];
cx q[18],q[21];
cx q[19],q[22];
cx q[16],q[19];
cx q[14],q[16];
cx q[13],q[14];
cx q[11],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[11];
cx q[11],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[15],q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[21],q[18];
cx q[18],q[21];
sx q[23];
rz(1.9913307) q[23];
sx q[23];
rz(-pi) q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
sx q[24];
rz(0.38759673) q[24];
sx q[24];
cx q[23],q[24];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[15];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
sx q[24];
rz(0.38759673) q[24];
sx q[24];
cx q[24],q[23];
sx q[23];
rz(0.42053433) q[23];
sx q[23];
cx q[23],q[21];
sx q[21];
rz(0.46364763) q[21];
sx q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
sx q[12];
rz(pi/6) q[12];
sx q[12];
cx q[12],q[13];
sx q[13];
rz(0.61547971) q[13];
sx q[13];
cx q[13],q[14];
sx q[14];
rz(pi/4) q[14];
sx q[14];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
x q[23];
rz(pi/2) q[24];
sx q[24];
rz(-pi) q[24];
cx q[23],q[24];
sx q[23];
rz(-pi/2) q[23];
sx q[23];
rz(pi/2) q[24];
cx q[23],q[24];
rz(pi/2) q[23];
sx q[23];
cx q[21],q[23];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
cx q[12],q[15];
cx q[13],q[12];
cx q[14],q[13];
rz(pi/2) q[24];
sx q[24];
barrier q[0],q[6],q[3],q[9],q[21],q[24],q[13],q[15],q[23],q[2],q[5],q[14],q[8],q[11],q[17],q[18],q[20],q[26],q[4],q[1],q[7],q[10],q[12],q[16],q[19],q[25],q[22];
measure q[14] -> meas[0];
measure q[13] -> meas[1];
measure q[12] -> meas[2];
measure q[15] -> meas[3];
measure q[18] -> meas[4];
measure q[23] -> meas[5];
measure q[24] -> meas[6];
measure q[21] -> meas[7];
measure q[10] -> meas[8];
measure q[11] -> meas[9];
measure q[16] -> meas[10];
measure q[19] -> meas[11];
measure q[22] -> meas[12];
