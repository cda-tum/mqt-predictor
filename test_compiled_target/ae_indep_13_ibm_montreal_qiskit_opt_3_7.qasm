OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[13];
rz(-3*pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-3*pi/2) q[11];
sx q[11];
rz(1.6689711) q[11];
rz(-3*pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi) q[13];
sx q[13];
rz(-3*pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-3*pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-3*pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-3*pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-3*pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(-pi) q[21];
sx q[21];
rz(2.2142974) q[21];
sx q[21];
cx q[18],q[21];
sx q[21];
rz(2.2142974) q[21];
sx q[21];
rz(-pi) q[21];
cx q[18],q[21];
rz(-pi) q[21];
sx q[21];
rz(2.2142974) q[21];
sx q[21];
rz(-3*pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(-3*pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[23],q[21];
sx q[21];
rz(1.2870023) q[21];
sx q[21];
rz(-pi) q[21];
cx q[23],q[21];
rz(-pi) q[21];
sx q[21];
rz(1.2870023) q[21];
sx q[21];
rz(-3*pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[23],q[21];
rz(-pi) q[21];
sx q[21];
rz(0.56758825) q[21];
sx q[21];
cx q[23],q[21];
sx q[21];
rz(0.56758825) q[21];
sx q[21];
rz(-pi) q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(-pi/2) q[12];
sx q[12];
rz(-pi/2) q[12];
cx q[13],q[12];
rz(-pi/2) q[12];
sx q[13];
rz(-pi) q[13];
cx q[13],q[12];
rz(0.43561993) q[12];
sx q[13];
cx q[13],q[12];
rz(-pi) q[12];
x q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-0.092038847) q[10];
rz(-1.1351764) q[13];
sx q[13];
cx q[14],q[13];
sx q[13];
rz(0.87124027) q[13];
sx q[13];
rz(-pi) q[13];
cx q[14],q[13];
rz(-pi) q[13];
sx q[13];
rz(0.87123975) q[13];
sx q[13];
cx q[12],q[13];
rz(-pi) q[13];
sx q[13];
rz(1.3991131) q[13];
sx q[13];
cx q[12],q[13];
cx q[12],q[15];
sx q[13];
rz(1.3991131) q[13];
sx q[13];
rz(-pi) q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[15],q[18];
cx q[16],q[14];
sx q[14];
rz(0.34336642) q[14];
sx q[14];
rz(-pi) q[14];
cx q[16],q[14];
rz(-pi) q[14];
sx q[14];
rz(0.34336645) q[14];
sx q[14];
cx q[11],q[14];
rz(-pi) q[14];
sx q[14];
rz(2.4548618) q[14];
sx q[14];
cx q[11],q[14];
sx q[14];
rz(2.4548597) q[14];
sx q[14];
rz(-pi) q[14];
cx q[13],q[14];
rz(-pi) q[14];
sx q[14];
rz(1.768131) q[14];
sx q[14];
cx q[13],q[14];
sx q[14];
rz(1.7681268) q[14];
sx q[14];
rz(-pi) q[14];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-3*pi/16) q[12];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-0.073631078) q[18];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[22],q[19];
rz(-pi) q[19];
sx q[19];
rz(0.39465931) q[19];
sx q[19];
cx q[22],q[19];
sx q[19];
rz(0.39466095) q[19];
sx q[19];
rz(-pi) q[19];
cx q[16],q[19];
sx q[19];
rz(2.352274) q[19];
sx q[19];
rz(-pi) q[19];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(pi/4) q[14];
rz(-pi) q[19];
sx q[19];
rz(2.3522708) q[19];
sx q[19];
cx q[16],q[19];
sx q[19];
rz(1.5629554) q[19];
sx q[19];
rz(-pi) q[19];
cx q[16],q[19];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[14],q[16];
rz(pi/4) q[16];
cx q[14],q[16];
sx q[14];
rz(pi/2) q[14];
rz(-pi/4) q[16];
rz(-pi) q[19];
sx q[19];
rz(1.5629554) q[19];
sx q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(pi/8) q[19];
cx q[19],q[16];
rz(pi/8) q[16];
cx q[19],q[16];
rz(-pi/8) q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
rz(pi/16) q[13];
cx q[12],q[13];
rz(-pi/16) q[13];
cx q[19],q[16];
rz(pi/4) q[16];
cx q[19],q[16];
rz(-pi/4) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[11],q[14];
cx q[12],q[13];
rz(pi/8) q[13];
cx q[12],q[13];
rz(-pi/8) q[13];
rz(pi/32) q[14];
cx q[11],q[14];
rz(-pi/32) q[14];
rz(-0.14726216) q[16];
cx q[16],q[14];
rz(pi/64) q[14];
cx q[16],q[14];
rz(-pi/64) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[11],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/16) q[14];
cx q[11],q[14];
rz(-pi/16) q[14];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(-pi/256) q[12];
cx q[16],q[14];
rz(pi/32) q[14];
cx q[16],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(1.3230994) q[14];
sx q[14];
cx q[18],q[15];
rz(pi/128) q[15];
cx q[18],q[15];
rz(-pi/128) q[15];
cx q[12],q[15];
rz(pi/256) q[15];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/128) q[13];
rz(-pi/256) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[10],q[12];
rz(pi/512) q[12];
cx q[10],q[12];
rz(-pi/512) q[12];
cx q[18],q[15];
rz(pi/64) q[15];
cx q[18],q[15];
rz(-pi/64) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
rz(pi/128) q[12];
cx q[13],q[12];
rz(-pi/128) q[12];
cx q[10],q[12];
rz(pi/256) q[12];
cx q[10],q[12];
rz(-pi/256) q[12];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(-pi/32) q[12];
rz(3*pi/4) q[15];
sx q[15];
cx q[18],q[21];
sx q[19];
rz(pi/2) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[14];
rz(pi/2) q[14];
sx q[16];
rz(-pi) q[16];
cx q[16],q[14];
rz(pi/4) q[14];
sx q[16];
cx q[16],q[14];
x q[14];
rz(3*pi/4) q[14];
cx q[11],q[14];
rz(pi/8) q[14];
cx q[11],q[14];
rz(-pi/8) q[14];
rz(-2.1084975) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[11],q[14];
rz(pi/4) q[14];
cx q[11],q[14];
sx q[11];
rz(pi/2) q[11];
rz(-pi/4) q[14];
rz(pi/16) q[19];
cx q[19],q[16];
rz(pi/16) q[16];
cx q[19],q[16];
rz(-pi/16) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[12],q[13];
rz(-pi/32) q[13];
rz(-pi/64) q[14];
cx q[14],q[13];
rz(pi/64) q[13];
cx q[14],q[13];
cx q[11],q[14];
rz(-pi/64) q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[10],q[12];
rz(pi/128) q[12];
cx q[10],q[12];
rz(2.3316508) q[12];
sx q[12];
rz(-pi/16) q[13];
cx q[14],q[11];
cx q[11],q[14];
rz(-0.29452431) q[11];
cx q[19],q[16];
rz(pi/8) q[16];
cx q[19],q[16];
rz(-pi/8) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/16) q[14];
cx q[13],q[14];
rz(-pi/16) q[14];
cx q[11],q[14];
rz(pi/32) q[14];
cx q[11],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(7*pi/8) q[14];
sx q[14];
cx q[19],q[16];
rz(pi/4) q[16];
cx q[19],q[16];
rz(-pi/4) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[14];
rz(pi/2) q[14];
sx q[16];
rz(-pi) q[16];
cx q[16],q[14];
rz(3*pi/8) q[14];
sx q[16];
cx q[16],q[14];
x q[14];
rz(7*pi/8) q[14];
cx q[11],q[14];
rz(pi/16) q[14];
cx q[11],q[14];
rz(-pi/16) q[14];
rz(3*pi/4) q[16];
sx q[19];
rz(pi/2) q[19];
cx q[16],q[19];
rz(pi/4) q[19];
cx q[16],q[19];
sx q[16];
rz(pi/2) q[16];
rz(-pi/4) q[19];
cx q[21],q[18];
cx q[18],q[21];
rz(-pi) q[21];
sx q[21];
rz(-1.5738643) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[23],q[21];
rz(-pi/2) q[21];
sx q[23];
rz(-pi) q[23];
cx q[23],q[21];
rz(1.5677284) q[21];
sx q[23];
cx q[23],q[21];
rz(pi/2) q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(0.77926224) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[15];
rz(pi/2) q[15];
sx q[18];
rz(-pi) q[18];
cx q[18],q[15];
rz(1.5646604) q[15];
sx q[18];
cx q[18],q[15];
rz(1.5830682) q[15];
sx q[15];
rz(-pi/2) q[15];
cx q[15],q[12];
rz(-pi/2) q[12];
sx q[15];
rz(-pi) q[15];
cx q[15],q[12];
rz(1.5585245) q[12];
sx q[15];
cx q[15],q[12];
rz(pi/4) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[10],q[12];
rz(pi/64) q[12];
cx q[10],q[12];
rz(-pi/64) q[12];
rz(-pi/128) q[13];
cx q[13],q[12];
rz(pi/128) q[12];
cx q[13],q[12];
rz(-pi/128) q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
rz(-pi/32) q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[12],q[13];
rz(-pi/32) q[13];
rz(-pi/64) q[14];
cx q[14],q[13];
rz(pi/64) q[13];
cx q[14],q[13];
cx q[11],q[14];
rz(-pi/64) q[13];
cx q[14],q[11];
cx q[11],q[14];
rz(-0.29452431) q[11];
x q[15];
rz(2.3439226) q[15];
cx q[16],q[14];
cx q[14],q[16];
rz(pi/4) q[14];
rz(pi/8) q[16];
cx q[16],q[19];
rz(2.3500586) q[18];
rz(pi/8) q[19];
cx q[16],q[19];
cx q[16],q[14];
rz(-pi/4) q[14];
sx q[16];
rz(pi/2) q[16];
rz(-pi/8) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-pi/4096) q[21];
x q[23];
rz(1.5677284) q[23];
rz(-pi/2048) q[24];
cx q[24],q[23];
rz(pi/2048) q[23];
cx q[24],q[23];
rz(-pi/2048) q[23];
cx q[21],q[23];
rz(pi/4096) q[23];
cx q[21],q[23];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
rz(-pi/2048) q[18];
rz(-pi/4096) q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(-pi/1024) q[23];
cx q[23],q[21];
rz(pi/1024) q[21];
cx q[23],q[21];
rz(-pi/1024) q[21];
cx q[18],q[21];
rz(pi/2048) q[21];
cx q[18],q[21];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
rz(-pi/1024) q[15];
rz(-pi/2048) q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(-pi/512) q[21];
cx q[21],q[18];
rz(pi/512) q[18];
cx q[21],q[18];
rz(-pi/512) q[18];
cx q[15],q[18];
rz(pi/1024) q[18];
cx q[15],q[18];
rz(-pi/1024) q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(-0.036815539) q[12];
cx q[12],q[10];
rz(pi/256) q[10];
cx q[12],q[10];
rz(-pi/256) q[10];
cx q[12],q[13];
rz(pi/128) q[13];
cx q[12],q[13];
rz(-pi/128) q[13];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(-0.018407769) q[12];
cx q[12],q[10];
rz(pi/512) q[10];
cx q[12],q[10];
rz(-pi/512) q[10];
cx q[12],q[13];
rz(pi/256) q[13];
cx q[12],q[13];
rz(-pi/256) q[13];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/8) q[13];
rz(-3*pi/16) q[14];
cx q[14],q[16];
rz(pi/16) q[16];
cx q[14],q[16];
cx q[14],q[13];
rz(-pi/8) q[13];
rz(-pi/16) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[11],q[14];
rz(pi/32) q[14];
cx q[11],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[11],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[12],q[15];
rz(pi/16) q[14];
cx q[11],q[14];
rz(-pi/16) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[15],q[12];
cx q[12],q[15];
rz(-pi/128) q[12];
rz(pi/4) q[16];
cx q[16],q[19];
rz(-pi/64) q[18];
cx q[18],q[15];
rz(pi/64) q[15];
cx q[18],q[15];
rz(-pi/64) q[15];
cx q[12],q[15];
rz(pi/128) q[15];
cx q[12],q[15];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/64) q[13];
rz(-pi/128) q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(-pi/32) q[15];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[15],q[12];
rz(-pi/32) q[12];
cx q[13],q[12];
rz(pi/64) q[12];
cx q[13],q[12];
rz(-pi/64) q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/4) q[19];
cx q[16],q[19];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(pi/4) q[14];
rz(pi/8) q[16];
rz(-pi/4) q[19];
cx q[16],q[19];
rz(pi/8) q[19];
cx q[16],q[19];
cx q[16],q[14];
rz(-pi/4) q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/8) q[13];
rz(-3*pi/16) q[14];
sx q[16];
rz(pi/2) q[16];
rz(-pi/8) q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[14],q[16];
rz(pi/16) q[16];
cx q[14],q[16];
cx q[14],q[13];
rz(-pi/8) q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/16) q[12];
rz(-0.29452431) q[13];
rz(-pi/16) q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
cx q[13],q[12];
rz(-pi/16) q[12];
rz(-pi/32) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/4) q[16];
cx q[16],q[19];
rz(pi/4) q[19];
cx q[16],q[19];
sx q[16];
rz(pi/2) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(pi/8) q[16];
rz(-pi/4) q[19];
cx q[16],q[19];
rz(pi/8) q[19];
cx q[16],q[19];
cx q[16],q[14];
rz(pi/4) q[14];
cx q[16],q[14];
rz(-pi/4) q[14];
sx q[16];
rz(pi/2) q[16];
rz(-pi/8) q[19];
barrier q[2],q[5],q[11],q[8],q[13],q[20],q[17],q[14],q[26],q[1],q[4],q[7],q[12],q[15],q[18],q[23],q[25],q[21],q[0],q[6],q[3],q[9],q[10],q[16],q[24],q[22],q[19];
measure q[16] -> meas[0];
measure q[14] -> meas[1];
measure q[19] -> meas[2];
measure q[12] -> meas[3];
measure q[13] -> meas[4];
measure q[15] -> meas[5];
measure q[18] -> meas[6];
measure q[11] -> meas[7];
measure q[10] -> meas[8];
measure q[21] -> meas[9];
measure q[23] -> meas[10];
measure q[24] -> meas[11];
measure q[22] -> meas[12];
