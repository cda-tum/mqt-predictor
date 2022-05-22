OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[12];
rz(-3*pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi) q[14];
sx q[14];
rz(-pi/2) q[14];
rz(-3*pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-3*pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi) q[18];
sx q[18];
rz(2.2142974) q[18];
sx q[18];
cx q[15],q[18];
sx q[18];
rz(2.2142974) q[18];
sx q[18];
rz(-pi) q[18];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(0.78079622) q[12];
rz(-pi) q[18];
sx q[18];
rz(2.2142974) q[18];
sx q[18];
cx q[15],q[18];
sx q[18];
rz(1.2870023) q[18];
sx q[18];
rz(-pi) q[18];
cx q[15],q[18];
rz(0.7823302) q[15];
sx q[15];
rz(-pi) q[15];
rz(-pi) q[18];
sx q[18];
rz(2.8577986) q[18];
sx q[18];
rz(-3*pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
sx q[21];
cx q[21],q[18];
rz(-pi/2) q[18];
sx q[21];
rz(-pi) q[21];
cx q[21],q[18];
rz(1.0032081) q[18];
sx q[21];
cx q[21],q[18];
rz(2.3500586) q[18];
sx q[18];
rz(2.1383846) q[21];
sx q[21];
rz(-3*pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(-pi) q[23];
sx q[23];
cx q[23],q[21];
rz(pi/2) q[21];
sx q[23];
rz(-pi) q[23];
cx q[23],q[21];
rz(0.43561993) q[21];
sx q[23];
cx q[23],q[21];
rz(2.3439226) q[21];
sx q[21];
rz(2.7059727) q[23];
sx q[23];
rz(-pi) q[23];
rz(-pi) q[24];
sx q[24];
cx q[24],q[23];
rz(-pi/2) q[23];
sx q[24];
rz(-pi) q[24];
cx q[24],q[23];
rz(0.69955606) q[23];
sx q[24];
cx q[24],q[23];
rz(-3.117049) q[23];
sx q[23];
rz(0.69955657) q[24];
sx q[24];
rz(-pi) q[25];
sx q[25];
cx q[25],q[24];
rz(-pi/2) q[24];
sx q[25];
rz(-pi) q[25];
cx q[25],q[24];
rz(0.17168321) q[24];
sx q[25];
cx q[25],q[24];
x q[24];
rz(-1.3991131) q[25];
sx q[25];
cx q[22],q[25];
sx q[25];
rz(0.34336642) q[25];
sx q[25];
rz(-pi) q[25];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-pi) q[25];
sx q[25];
rz(0.34336645) q[25];
sx q[25];
cx q[22],q[25];
rz(-pi) q[25];
sx q[25];
rz(2.4548618) q[25];
sx q[25];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
sx q[25];
rz(2.4548597) q[25];
sx q[25];
rz(-pi) q[25];
cx q[22],q[25];
rz(-pi) q[25];
sx q[25];
rz(1.768131) q[25];
sx q[25];
cx q[22],q[25];
sx q[25];
rz(1.7681268) q[25];
sx q[25];
rz(-pi) q[25];
rz(-3*pi/2) q[26];
sx q[26];
rz(pi/2) q[26];
cx q[26],q[25];
rz(-pi) q[25];
sx q[25];
rz(0.39465931) q[25];
sx q[25];
cx q[26],q[25];
sx q[25];
rz(0.39466095) q[25];
sx q[25];
rz(-pi) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(-pi/2) q[16];
sx q[16];
cx q[16],q[14];
rz(-pi/2) q[14];
sx q[16];
rz(-pi) q[16];
cx q[16],q[14];
rz(0.78147771) q[14];
sx q[16];
cx q[16],q[14];
rz(-0.78147771) q[14];
sx q[14];
rz(-pi) q[14];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-0.29452431) q[16];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(3*pi/16) q[19];
sx q[19];
rz(-pi) q[19];
rz(-pi/2) q[22];
sx q[22];
rz(-pi) q[22];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
rz(pi/4) q[25];
sx q[25];
rz(-pi/2) q[25];
cx q[25],q[22];
rz(pi/2) q[22];
sx q[25];
rz(-pi) q[25];
cx q[25],q[22];
rz(pi/4) q[22];
sx q[25];
cx q[25],q[22];
sx q[22];
rz(pi/2) q[22];
x q[25];
rz(-pi/4) q[25];
rz(pi/8) q[26];
cx q[26],q[25];
rz(pi/8) q[25];
cx q[26],q[25];
rz(-pi/8) q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
rz(-pi/4) q[22];
sx q[22];
rz(-pi/2) q[22];
cx q[22],q[19];
rz(-pi/2) q[19];
sx q[22];
rz(-pi) q[22];
cx q[22],q[19];
rz(7*pi/16) q[19];
sx q[22];
cx q[22],q[19];
x q[19];
rz(-13*pi/16) q[19];
cx q[16],q[19];
rz(pi/32) q[19];
cx q[16],q[19];
rz(-pi/32) q[19];
rz(pi/8) q[22];
sx q[22];
cx q[26],q[25];
rz(pi/4) q[25];
cx q[26],q[25];
rz(-pi/4) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[22];
rz(pi/2) q[22];
sx q[25];
rz(-pi) q[25];
cx q[25],q[22];
rz(3*pi/8) q[22];
sx q[25];
cx q[25],q[22];
x q[22];
rz(7*pi/8) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
rz(pi/16) q[19];
cx q[16],q[19];
rz(-pi/16) q[19];
sx q[22];
rz(-pi) q[22];
rz(3*pi/4) q[25];
sx q[26];
rz(pi/2) q[26];
cx q[25],q[26];
rz(pi/4) q[26];
cx q[25],q[26];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
rz(-pi/64) q[25];
sx q[25];
rz(-pi/2) q[25];
cx q[25],q[22];
rz(-pi/2) q[22];
sx q[25];
rz(-pi) q[25];
cx q[25],q[22];
rz(1.5217089) q[22];
sx q[25];
cx q[25],q[22];
x q[22];
rz(3.0434179) q[22];
cx q[22],q[19];
rz(pi/32) q[19];
cx q[22],q[19];
rz(-pi/32) q[19];
sx q[19];
rz(-pi) q[19];
rz(-1.6198837) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
sx q[24];
rz(-pi/2) q[24];
cx q[24],q[23];
rz(-pi/2) q[23];
sx q[24];
rz(-pi) q[24];
cx q[24],q[23];
rz(1.5462526) q[23];
sx q[24];
cx q[24],q[23];
rz(0.80994186) q[23];
sx q[23];
rz(-pi/2) q[23];
cx q[23],q[21];
rz(-pi/2) q[21];
sx q[23];
rz(-pi) q[23];
cx q[23],q[21];
rz(1.5585245) q[21];
sx q[23];
cx q[23],q[21];
rz(1.5585245) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[18];
rz(pi/2) q[18];
sx q[21];
rz(-pi) q[21];
cx q[21],q[18];
rz(1.5646604) q[18];
sx q[21];
cx q[21],q[18];
rz(1.5769322) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[15];
rz(-pi/2) q[15];
sx q[18];
rz(-pi) q[18];
cx q[18],q[15];
rz(1.5677284) q[15];
sx q[18];
cx q[18],q[15];
x q[15];
rz(-2.3592625) q[15];
cx q[12],q[15];
rz(pi/2048) q[15];
cx q[12],q[15];
sx q[12];
rz(-pi) q[12];
rz(-pi/2048) q[15];
rz(-pi/512) q[18];
sx q[18];
rz(2.3439226) q[21];
sx q[21];
rz(-pi) q[21];
x q[23];
rz(3*pi/4) q[23];
rz(-pi/2) q[24];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(-pi/128) q[24];
sx q[24];
rz(-pi) q[24];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
rz(-pi/64) q[22];
sx q[22];
rz(-pi/2) q[22];
cx q[22],q[19];
rz(-pi/2) q[19];
sx q[22];
rz(-pi) q[22];
cx q[22],q[19];
rz(1.5217089) q[19];
sx q[22];
cx q[22],q[19];
rz(-pi) q[19];
x q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-pi/32) q[16];
rz(-pi/8) q[19];
rz(-1.6198837) q[22];
rz(-pi/4) q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[19],q[22];
rz(pi/8) q[22];
cx q[19],q[22];
rz(-pi/8) q[22];
sx q[25];
rz(-pi/2) q[25];
cx q[25],q[24];
rz(-pi/2) q[24];
sx q[25];
rz(-pi) q[25];
cx q[25],q[24];
rz(1.5462526) q[24];
sx q[25];
cx q[25],q[24];
x q[24];
rz(3.117049) q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
sx q[23];
rz(-pi/2) q[23];
cx q[23],q[21];
rz(-pi/2) q[21];
sx q[23];
rz(-pi) q[23];
cx q[23],q[21];
rz(1.5585245) q[21];
sx q[23];
cx q[23],q[21];
rz(-0.77312632) q[21];
sx q[21];
rz(-pi/2) q[21];
cx q[21],q[18];
rz(pi/2) q[18];
sx q[21];
rz(-pi) q[21];
cx q[21],q[18];
rz(1.5646604) q[18];
sx q[21];
cx q[21],q[18];
x q[18];
rz(2.3500586) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(-pi/4) q[15];
sx q[15];
rz(-pi/2) q[15];
cx q[15],q[12];
rz(-pi/2) q[12];
sx q[15];
rz(-pi) q[15];
cx q[15],q[12];
rz(1.5677284) q[12];
sx q[15];
cx q[15],q[12];
x q[12];
rz(-2.3592625) q[12];
rz(-2.3623304) q[15];
sx q[15];
rz(-pi) q[15];
rz(2.3439226) q[21];
sx q[21];
rz(-pi) q[21];
rz(-1.59534) q[23];
sx q[23];
rz(-pi) q[23];
rz(-pi/2) q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
rz(-pi/16) q[25];
cx q[25],q[22];
rz(pi/16) q[22];
cx q[25],q[22];
rz(-pi/16) q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[16],q[19];
rz(pi/32) q[19];
cx q[16],q[19];
rz(-pi/32) q[19];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(pi/4) q[25];
cx q[25],q[24];
rz(pi/4) q[24];
cx q[25],q[24];
rz(-pi/4) q[24];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(pi/4) q[22];
rz(pi/8) q[25];
cx q[25],q[24];
rz(pi/8) q[24];
cx q[25],q[24];
rz(-pi/8) q[24];
cx q[25],q[22];
rz(-pi/4) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/8) q[16];
rz(-3*pi/16) q[19];
sx q[22];
rz(-pi) q[22];
sx q[25];
rz(pi/2) q[25];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
rz(-pi/64) q[25];
sx q[25];
rz(-pi/2) q[25];
cx q[25],q[22];
rz(-pi/2) q[22];
sx q[25];
rz(-pi) q[25];
cx q[25],q[22];
rz(1.5217089) q[22];
sx q[25];
cx q[25],q[22];
rz(-pi) q[22];
x q[22];
rz(-1.6198837) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
sx q[24];
rz(-pi/2) q[24];
cx q[24],q[23];
rz(-pi/2) q[23];
sx q[24];
rz(-pi) q[24];
cx q[24],q[23];
rz(1.5462526) q[23];
sx q[24];
cx q[24],q[23];
rz(pi/128) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[23],q[21];
rz(-pi/2) q[21];
sx q[23];
rz(-pi) q[23];
cx q[23],q[21];
rz(1.5585245) q[21];
sx q[23];
cx q[23],q[21];
x q[21];
rz(3.1293208) q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
sx q[18];
rz(-pi/2) q[18];
cx q[18],q[15];
rz(-pi/2) q[15];
sx q[18];
rz(-pi) q[18];
cx q[18],q[15];
rz(1.5646604) q[15];
sx q[18];
cx q[18],q[15];
x q[15];
rz(3.1354567) q[15];
rz(-pi/2) q[18];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
rz(2.3439226) q[21];
sx q[21];
rz(-1.59534) q[23];
sx q[23];
rz(-pi) q[23];
rz(-1.6198837) q[24];
sx q[24];
rz(-pi) q[24];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[19],q[22];
rz(pi/16) q[22];
cx q[19],q[22];
cx q[19],q[16];
rz(-pi/8) q[16];
rz(3*pi/16) q[22];
sx q[22];
rz(-pi) q[22];
rz(-0.88357293) q[25];
sx q[25];
rz(-pi/2) q[25];
cx q[25],q[22];
rz(-pi/2) q[22];
sx q[25];
rz(-pi) q[25];
cx q[25],q[22];
rz(1.4726216) q[22];
sx q[25];
cx q[25],q[22];
x q[22];
rz(-3*pi/4) q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(-pi/16) q[19];
cx q[19],q[16];
rz(pi/16) q[16];
cx q[19],q[16];
rz(-pi/16) q[16];
rz(-2.4543693) q[25];
sx q[25];
rz(-pi/2) q[25];
cx q[25],q[24];
rz(-pi/2) q[24];
sx q[25];
rz(-pi) q[25];
cx q[25],q[24];
rz(1.5217089) q[24];
sx q[25];
cx q[25],q[24];
rz(pi/64) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[24],q[23];
rz(-pi/2) q[23];
sx q[24];
rz(-pi) q[24];
cx q[24],q[23];
rz(1.5462526) q[23];
sx q[24];
cx q[24],q[23];
rz(0.80994186) q[23];
sx q[23];
rz(-pi/2) q[23];
cx q[23],q[21];
rz(-pi/2) q[21];
sx q[23];
rz(-pi) q[23];
cx q[23],q[21];
rz(1.5585245) q[21];
sx q[23];
cx q[23],q[21];
rz(0.77312632) q[21];
x q[23];
rz(3*pi/4) q[23];
rz(-pi/2) q[24];
rz(-pi/2) q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(-pi/32) q[19];
cx q[19],q[16];
rz(pi/32) q[16];
cx q[19],q[16];
rz(-pi/32) q[16];
rz(pi/4) q[25];
cx q[25],q[26];
rz(pi/4) q[26];
cx q[25],q[26];
sx q[25];
rz(pi/2) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(-pi/64) q[19];
cx q[19],q[16];
rz(pi/64) q[16];
cx q[19],q[16];
rz(-pi/64) q[16];
rz(pi/8) q[25];
rz(-pi/4) q[26];
cx q[25],q[26];
rz(pi/8) q[26];
cx q[25],q[26];
cx q[25],q[24];
rz(pi/4) q[24];
cx q[25],q[24];
rz(-pi/4) q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
sx q[25];
rz(pi/2) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(-pi/16) q[25];
rz(-pi/8) q[26];
cx q[25],q[26];
rz(pi/16) q[26];
cx q[25],q[26];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(pi/8) q[24];
cx q[24],q[23];
rz(pi/8) q[23];
cx q[24],q[23];
rz(-pi/8) q[23];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(-pi/128) q[19];
cx q[19],q[16];
rz(pi/128) q[16];
cx q[19],q[16];
rz(-pi/128) q[16];
cx q[24],q[25];
rz(pi/4) q[25];
cx q[24],q[25];
sx q[24];
rz(pi/2) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(-pi/4) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(-0.29452431) q[25];
rz(-pi/16) q[26];
cx q[25],q[26];
rz(pi/32) q[26];
cx q[25],q[26];
cx q[25],q[24];
rz(pi/16) q[24];
cx q[25],q[24];
rz(-pi/16) q[24];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
rz(-pi/8) q[22];
cx q[22],q[19];
rz(pi/8) q[19];
cx q[22],q[19];
rz(-pi/8) q[19];
rz(-0.14726216) q[25];
rz(-pi/32) q[26];
cx q[25],q[26];
rz(pi/64) q[26];
cx q[25],q[26];
cx q[25],q[24];
rz(pi/32) q[24];
cx q[25],q[24];
cx q[22],q[25];
rz(-pi/32) q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[25],q[22];
cx q[22],q[25];
rz(-pi/16) q[22];
cx q[22],q[19];
rz(pi/16) q[19];
cx q[22],q[19];
rz(-pi/16) q[19];
rz(pi/4) q[25];
cx q[25],q[24];
rz(pi/4) q[24];
cx q[25],q[24];
rz(-pi/4) q[24];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(pi/4) q[22];
rz(pi/8) q[25];
cx q[25],q[24];
rz(pi/8) q[24];
cx q[25],q[24];
rz(-pi/8) q[24];
cx q[25],q[22];
rz(-pi/4) q[22];
sx q[25];
rz(pi/2) q[25];
rz(-pi/64) q[26];
barrier q[2],q[5],q[11],q[8],q[18],q[20],q[17],q[19],q[12],q[1],q[4],q[7],q[13],q[10],q[15],q[21],q[26],q[16],q[0],q[6],q[3],q[9],q[22],q[14],q[25],q[24],q[23];
measure q[25] -> meas[0];
measure q[22] -> meas[1];
measure q[24] -> meas[2];
measure q[19] -> meas[3];
measure q[23] -> meas[4];
measure q[26] -> meas[5];
measure q[16] -> meas[6];
measure q[21] -> meas[7];
measure q[15] -> meas[8];
measure q[12] -> meas[9];
measure q[18] -> meas[10];
measure q[14] -> meas[11];
