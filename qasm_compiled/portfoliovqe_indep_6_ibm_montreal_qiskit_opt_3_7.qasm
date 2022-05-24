OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[6];
sx q[12];
rz(0.42061033) q[12];
sx q[12];
sx q[13];
rz(1.9124681) q[13];
sx q[13];
rz(-pi) q[13];
sx q[14];
rz(0.070991671) q[14];
sx q[14];
rz(-pi) q[14];
sx q[15];
rz(2.1244446) q[15];
sx q[15];
rz(-pi) q[15];
sx q[16];
rz(0.5274324) q[16];
sx q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[12],q[15];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[15],q[12];
cx q[12],q[15];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
sx q[18];
rz(0.17483319) q[18];
sx q[18];
cx q[15],q[18];
cx q[15],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi) q[15];
sx q[15];
rz(2.5780201) q[15];
sx q[15];
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
cx q[12],q[13];
rz(-pi) q[12];
sx q[12];
rz(1.7085457) q[12];
sx q[12];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
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
cx q[13],q[14];
rz(-pi) q[13];
sx q[13];
rz(2.3002244) q[13];
sx q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/2) q[12];
sx q[12];
rz(-pi/2) q[12];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[14],q[16];
sx q[14];
rz(1.5927066) q[14];
sx q[14];
rz(-pi) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[13];
rz(-pi) q[13];
cx q[14],q[16];
sx q[14];
rz(-2.791379) q[14];
sx q[14];
rz(pi/2) q[14];
sx q[16];
rz(2.8800814) q[16];
sx q[16];
rz(pi/2) q[16];
sx q[18];
rz(-pi) q[18];
cx q[15],q[18];
sx q[15];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[18];
cx q[15],q[18];
rz(-pi/2) q[15];
sx q[15];
rz(-0.010479255) q[15];
sx q[15];
cx q[12],q[15];
sx q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[15];
cx q[12],q[15];
rz(-pi/2) q[12];
sx q[12];
rz(-1.5603171) q[12];
sx q[12];
rz(-pi/2) q[12];
cx q[12],q[13];
sx q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[13];
cx q[12],q[13];
rz(pi/2) q[12];
sx q[12];
rz(-pi/2) q[13];
sx q[13];
rz(-pi) q[13];
cx q[13],q[14];
sx q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[14];
cx q[13],q[14];
rz(pi/2) q[13];
sx q[13];
sx q[14];
rz(-1.6185212) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[16];
sx q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[15];
sx q[15];
rz(-pi) q[15];
rz(pi/2) q[16];
cx q[14],q[16];
rz(-pi/2) q[14];
sx q[14];
rz(2.3751215) q[16];
sx q[16];
rz(-1.5364046) q[16];
sx q[16];
rz(1.537701) q[16];
rz(-pi) q[18];
sx q[18];
rz(-pi) q[18];
cx q[18],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[18],q[15];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[18],q[15];
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
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[15],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi) q[13];
sx q[13];
rz(1.5190695) q[13];
sx q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[16];
sx q[15];
rz(0.33344972) q[15];
sx q[15];
rz(-pi) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[16],q[14];
cx q[14],q[16];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[14],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[18],q[15];
cx q[12],q[15];
sx q[12];
rz(1.1179081) q[12];
sx q[12];
rz(-pi) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
sx q[15];
rz(0.15598684) q[15];
sx q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
sx q[18];
rz(1.719355) q[18];
sx q[18];
cx q[15],q[18];
cx q[15],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[15],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi) q[15];
sx q[15];
rz(1.7418483) q[15];
sx q[15];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[12],q[15];
cx q[12],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[12],q[13];
rz(-pi) q[12];
sx q[12];
rz(2.534237) q[12];
sx q[12];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
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
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-1.5502309) q[13];
sx q[13];
rz(-pi) q[13];
cx q[12],q[13];
sx q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[13];
cx q[12],q[13];
rz(0.29799211) q[12];
sx q[12];
rz(-1.5768351) q[12];
sx q[12];
rz(-1.5511374) q[12];
rz(pi/2) q[13];
sx q[13];
cx q[14],q[16];
cx q[14],q[13];
sx q[14];
rz(2.2675479) q[14];
sx q[14];
rz(-pi) q[14];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[14],q[13];
sx q[13];
rz(0.31968257) q[13];
sx q[13];
sx q[14];
rz(0.19053015) q[14];
sx q[14];
rz(-pi) q[14];
barrier q[7],q[16],q[10],q[15],q[19],q[25],q[22],q[2],q[5],q[11],q[8],q[18],q[20],q[24],q[17],q[23],q[26],q[0],q[6],q[3],q[9],q[12],q[14],q[13],q[21],q[1],q[4];
measure q[18] -> meas[0];
measure q[15] -> meas[1];
measure q[12] -> meas[2];
measure q[16] -> meas[3];
measure q[14] -> meas[4];
measure q[13] -> meas[5];
