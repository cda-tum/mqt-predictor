OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[13];
sx q[10];
rz(2.7777575) q[10];
sx q[10];
rz(-pi) q[10];
sx q[11];
rz(1.4809363) q[11];
sx q[11];
rz(-pi) q[11];
sx q[12];
rz(-1.5283853) q[12];
sx q[12];
sx q[13];
rz(1.5763404) q[13];
sx q[13];
rz(-pi) q[13];
rz(-pi) q[14];
sx q[14];
rz(1.5663418) q[14];
sx q[14];
sx q[15];
rz(2.8230772) q[15];
sx q[15];
rz(-pi) q[15];
cx q[12],q[15];
cx q[12],q[13];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi) q[18];
sx q[18];
rz(1.4219729) q[18];
sx q[18];
cx q[15],q[18];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[12],q[10];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[14],q[11];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[14],q[11];
cx q[15],q[18];
cx q[15],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[10];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[11];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[10];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[14],q[11];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[11];
sx q[19];
rz(1.2102211) q[19];
sx q[19];
sx q[21];
rz(1.5848263) q[21];
sx q[21];
rz(-pi) q[21];
cx q[18],q[21];
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
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[15],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[12],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
sx q[22];
rz(-0.22445365) q[22];
sx q[22];
sx q[23];
rz(-1.1322111) q[23];
sx q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
sx q[24];
rz(-0.59639255) q[24];
sx q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(-pi) q[25];
sx q[25];
rz(3.0565952) q[25];
sx q[25];
cx q[24],q[25];
cx q[24],q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[13];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[16],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[18],q[21];
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
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[12],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[12],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[16],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[11],q[14];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[12],q[13];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[11];
cx q[11],q[14];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[11];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[15];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[15],q[12];
cx q[12],q[15];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[24],q[25];
cx q[24],q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[24],q[25];
rz(-pi) q[24];
sx q[24];
rz(1.7444317) q[24];
sx q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[22];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[24],q[25];
rz(-pi) q[24];
sx q[24];
rz(0.61189578) q[24];
sx q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[12],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[11],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[10],q[12];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[10],q[12];
cx q[15],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[24],q[25];
sx q[24];
rz(1.5803898) q[24];
sx q[24];
rz(-pi) q[24];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[16],q[19];
rz(-pi) q[16];
sx q[16];
rz(2.7342994) q[16];
sx q[16];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[14],q[16];
sx q[14];
rz(1.5903771) q[14];
sx q[14];
rz(-pi) q[14];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[10],q[12];
rz(-pi) q[10];
sx q[10];
rz(2.9441609) q[10];
sx q[10];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[15],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
sx q[15];
rz(1.6623504) q[15];
sx q[15];
rz(-pi) q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[16],q[14];
cx q[11],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[15],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[16],q[14];
cx q[11],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[15],q[12];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[16],q[14];
cx q[11],q[14];
sx q[11];
rz(2.9256974) q[11];
sx q[11];
rz(-pi) q[11];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[12],q[13];
sx q[12];
rz(2.5548913) q[12];
sx q[12];
rz(-pi) q[12];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi) q[14];
sx q[14];
rz(1.3539378) q[14];
sx q[14];
cx q[15],q[12];
sx q[12];
rz(2.2608761) q[12];
sx q[12];
sx q[15];
rz(2.1581062) q[15];
sx q[15];
rz(-pi) q[15];
sx q[16];
rz(0.081157235) q[16];
sx q[16];
rz(-pi) q[16];
barrier q[15],q[26],q[1],q[4],q[10],q[7],q[24],q[12],q[25],q[14],q[11],q[3],q[0],q[6],q[21],q[9],q[23],q[22],q[13],q[16],q[5],q[2],q[8],q[18],q[17],q[19],q[20];
measure q[21] -> meas[0];
measure q[23] -> meas[1];
measure q[24] -> meas[2];
measure q[22] -> meas[3];
measure q[19] -> meas[4];
measure q[10] -> meas[5];
measure q[18] -> meas[6];
measure q[16] -> meas[7];
measure q[11] -> meas[8];
measure q[13] -> meas[9];
measure q[14] -> meas[10];
measure q[15] -> meas[11];
measure q[12] -> meas[12];
