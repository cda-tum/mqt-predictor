OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[7];
x q[7];
sx q[10];
rz(0.38759673) q[10];
sx q[10];
cx q[7],q[10];
sx q[10];
rz(0.38759673) q[10];
sx q[10];
sx q[12];
rz(0.42053433) q[12];
sx q[12];
cx q[10],q[12];
cx q[10],q[7];
sx q[12];
rz(0.42053433) q[12];
sx q[12];
sx q[13];
rz(0.46364763) q[13];
sx q[13];
cx q[12],q[13];
cx q[12],q[10];
sx q[13];
rz(0.46364763) q[13];
sx q[13];
sx q[14];
rz(pi/6) q[14];
sx q[14];
cx q[13],q[14];
cx q[13],q[12];
sx q[14];
rz(pi/6) q[14];
sx q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
sx q[19];
rz(0.61547971) q[19];
sx q[19];
cx q[16],q[19];
cx q[16],q[14];
sx q[19];
rz(0.61547971) q[19];
sx q[19];
sx q[22];
rz(pi/4) q[22];
sx q[22];
cx q[19],q[22];
cx q[19],q[16];
sx q[22];
rz(pi/4) q[22];
sx q[22];
cx q[22],q[19];
barrier q[23],q[20],q[26],q[0],q[6],q[3],q[9],q[12],q[18],q[15],q[21],q[24],q[4],q[1],q[7],q[14],q[10],q[13],q[19],q[25],q[22],q[2],q[5],q[11],q[8],q[16],q[17];
measure q[22] -> meas[0];
measure q[19] -> meas[1];
measure q[16] -> meas[2];
measure q[14] -> meas[3];
measure q[12] -> meas[4];
measure q[10] -> meas[5];
measure q[7] -> meas[6];
