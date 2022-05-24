OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[14];
sx q[3];
rz(0.38759673) q[3];
sx q[3];
sx q[7];
rz(0.42053433) q[7];
sx q[7];
sx q[11];
rz(0.36136713) q[11];
sx q[11];
x q[13];
sx q[14];
rz(0.27054973) q[14];
sx q[14];
cx q[13],q[14];
sx q[14];
rz(0.27054973) q[14];
sx q[14];
sx q[16];
rz(0.28103493) q[16];
sx q[16];
cx q[14],q[16];
cx q[14],q[13];
sx q[16];
rz(0.28103493) q[16];
sx q[16];
sx q[17];
rz(pi/4) q[17];
sx q[17];
sx q[18];
rz(0.46364763) q[18];
sx q[18];
sx q[19];
rz(0.29284273) q[19];
sx q[19];
cx q[16],q[19];
cx q[16],q[14];
sx q[19];
rz(0.29284273) q[19];
sx q[19];
sx q[20];
rz(0.30627733) q[20];
sx q[20];
cx q[19],q[20];
cx q[19],q[16];
sx q[20];
rz(0.30627733) q[20];
sx q[20];
cx q[19],q[20];
cx q[20],q[19];
sx q[21];
rz(pi/6) q[21];
sx q[21];
sx q[22];
rz(0.32175053) q[22];
sx q[22];
cx q[19],q[22];
sx q[22];
rz(0.32175053) q[22];
sx q[22];
sx q[23];
rz(-2.5261129) q[23];
sx q[23];
rz(-pi/2) q[23];
sx q[25];
rz(0.33983693) q[25];
sx q[25];
cx q[22],q[25];
cx q[22],q[19];
sx q[25];
rz(0.33983693) q[25];
sx q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[14],q[11];
sx q[11];
rz(0.36136713) q[11];
sx q[11];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[5],q[3];
sx q[3];
rz(0.38759673) q[3];
sx q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[1],q[4];
cx q[4],q[1];
cx q[1],q[4];
cx q[4],q[7];
sx q[7];
rz(0.42053433) q[7];
sx q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[18];
sx q[18];
rz(0.46364763) q[18];
sx q[18];
cx q[18],q[21];
rz(0.19530623) q[21];
sx q[21];
rz(-1.055469) q[21];
sx q[21];
rz(-3.0444063) q[21];
cx q[21],q[23];
sx q[21];
rz(-pi/2) q[21];
sx q[21];
rz(pi/2) q[23];
cx q[21],q[23];
rz(0.61547971) q[21];
sx q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[18],q[17];
sx q[17];
rz(pi/4) q[17];
sx q[17];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
sx q[23];
rz(1.7404731) q[23];
sx q[23];
rz(-pi) q[23];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[5],q[8];
cx q[3],q[5];
cx q[5],q[3];
cx q[3],q[5];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[4],q[1];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[15],q[12];
cx q[18],q[15];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
x q[21];
cx q[21],q[23];
sx q[21];
rz(-pi/2) q[21];
sx q[21];
rz(pi/2) q[23];
cx q[21],q[23];
rz(pi/2) q[21];
sx q[21];
cx q[18],q[21];
cx q[17],q[18];
rz(pi/2) q[23];
sx q[23];
barrier q[0],q[6],q[12],q[9],q[7],q[23],q[10],q[21],q[24],q[5],q[11],q[1],q[14],q[19],q[17],q[18],q[25],q[26],q[2],q[3],q[15],q[13],q[4],q[22],q[20],q[8],q[16];
measure q[17] -> meas[0];
measure q[18] -> meas[1];
measure q[21] -> meas[2];
measure q[23] -> meas[3];
measure q[15] -> meas[4];
measure q[12] -> meas[5];
measure q[1] -> meas[6];
measure q[8] -> meas[7];
measure q[16] -> meas[8];
measure q[25] -> meas[9];
measure q[20] -> meas[10];
measure q[22] -> meas[11];
measure q[19] -> meas[12];
measure q[13] -> meas[13];
