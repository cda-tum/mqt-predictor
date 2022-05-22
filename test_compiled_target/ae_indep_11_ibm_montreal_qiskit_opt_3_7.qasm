OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[11];
rz(-pi/2) q[10];
sx q[10];
rz(-0.10820809) q[10];
rz(-3*pi/2) q[11];
sx q[11];
rz(1.2026409) q[11];
sx q[12];
rz(-pi/2) q[12];
cx q[10],q[12];
x q[10];
rz(0.92729522) q[12];
cx q[10],q[12];
rz(0.66798619) q[10];
rz(2.8577985) q[12];
sx q[12];
sx q[13];
rz(-3*pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-3*pi/2) q[15];
sx q[15];
rz(1.5646604) q[15];
cx q[15],q[12];
sx q[12];
rz(1.2870023) q[12];
sx q[12];
rz(-pi) q[12];
cx q[15],q[12];
rz(-pi) q[12];
sx q[12];
rz(2.8577986) q[12];
sx q[12];
cx q[13],q[12];
rz(-pi/2) q[12];
sx q[13];
rz(-pi) q[13];
cx q[13],q[12];
rz(1.0032081) q[12];
sx q[13];
cx q[13],q[12];
rz(-pi/256) q[12];
sx q[12];
rz(-pi) q[12];
rz(0.56758825) q[13];
sx q[13];
rz(-pi) q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[11],q[14];
sx q[14];
rz(2.0064163) q[14];
sx q[14];
rz(-pi) q[14];
cx q[11],q[14];
rz(-pi) q[14];
sx q[14];
rz(2.0064163) q[14];
sx q[14];
cx q[13],q[14];
sx q[14];
rz(0.87124027) q[14];
sx q[14];
rz(-pi) q[14];
cx q[13],q[14];
rz(-pi/64) q[13];
sx q[14];
rz(0.69955657) q[14];
sx q[14];
rz(-pi) q[16];
sx q[16];
cx q[16],q[14];
rz(-pi/2) q[14];
sx q[16];
rz(-pi) q[16];
cx q[16],q[14];
rz(0.17168321) q[14];
sx q[16];
cx q[16],q[14];
rz(2.4543693) q[14];
sx q[14];
rz(-2.9699094) q[16];
sx q[16];
rz(-pi) q[16];
sx q[19];
cx q[19],q[16];
rz(-pi/2) q[16];
sx q[19];
rz(-pi) q[19];
cx q[19],q[16];
rz(1.2274299) q[16];
sx q[19];
cx q[19],q[16];
rz(3*pi/16) q[16];
sx q[16];
rz(-pi) q[16];
rz(0.34336645) q[19];
sx q[19];
rz(-3*pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(-3*pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[22];
rz(-pi) q[22];
sx q[22];
rz(2.4548618) q[22];
sx q[22];
cx q[25],q[22];
sx q[22];
rz(2.4548597) q[22];
sx q[22];
rz(-pi) q[22];
cx q[19],q[22];
rz(-pi) q[22];
sx q[22];
rz(1.768131) q[22];
sx q[22];
cx q[19],q[22];
rz(pi/4) q[19];
sx q[22];
rz(1.7681268) q[22];
sx q[22];
rz(-pi) q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(pi/2) q[25];
sx q[25];
rz(-pi/2) q[25];
sx q[26];
cx q[26],q[25];
rz(pi/2) q[25];
sx q[26];
rz(-pi) q[26];
cx q[26],q[25];
rz(1.176137) q[25];
sx q[26];
cx q[26],q[25];
rz(-pi/2) q[25];
sx q[25];
rz(-pi/2) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
rz(pi/4) q[22];
cx q[19],q[22];
sx q[19];
rz(pi/2) q[19];
rz(-pi/4) q[22];
rz(pi/8) q[25];
cx q[25],q[22];
rz(pi/8) q[22];
cx q[25],q[22];
rz(-pi/8) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(-pi/4) q[19];
sx q[19];
rz(-pi/2) q[19];
cx q[19],q[16];
rz(-pi/2) q[16];
sx q[19];
rz(-pi) q[19];
cx q[19],q[16];
rz(7*pi/16) q[16];
sx q[19];
cx q[19],q[16];
rz(pi/16) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[14];
rz(-pi/2) q[14];
sx q[16];
rz(-pi) q[16];
cx q[16],q[14];
rz(1.4726216) q[14];
sx q[16];
cx q[16],q[14];
x q[14];
rz(-2.4543693) q[14];
cx q[13],q[14];
rz(pi/64) q[14];
cx q[13],q[14];
rz(-pi/64) q[14];
cx q[11],q[14];
rz(pi/128) q[14];
cx q[11],q[14];
rz(-pi/128) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
sx q[13];
rz(-pi/2) q[13];
cx q[13],q[12];
rz(-pi/2) q[12];
sx q[13];
rz(-pi) q[13];
cx q[13],q[12];
rz(1.5585245) q[12];
sx q[13];
cx q[13],q[12];
x q[12];
rz(3.1293208) q[12];
rz(-1.59534) q[13];
sx q[13];
rz(-pi) q[13];
rz(0.68722339) q[14];
sx q[14];
rz(-pi) q[14];
cx q[15],q[12];
rz(pi/512) q[12];
cx q[15],q[12];
rz(-pi/512) q[12];
cx q[10],q[12];
rz(pi/1024) q[12];
cx q[10],q[12];
sx q[10];
rz(-pi/1024) q[12];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(-pi/256) q[12];
sx q[12];
rz(-pi) q[12];
rz(3.007153) q[16];
sx q[16];
rz(-pi) q[16];
rz(pi/8) q[19];
sx q[19];
cx q[25],q[22];
rz(pi/4) q[22];
cx q[25],q[22];
rz(-pi/4) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[19];
rz(pi/2) q[19];
sx q[22];
rz(-pi) q[22];
cx q[22],q[19];
rz(3*pi/8) q[19];
sx q[22];
cx q[22],q[19];
rz(-0.33078919) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[19],q[16];
rz(-pi/2) q[16];
sx q[19];
rz(-pi) q[19];
cx q[19],q[16];
rz(7*pi/16) q[16];
sx q[19];
cx q[19],q[16];
rz(1.705236) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[14];
rz(-pi/2) q[14];
sx q[16];
rz(-pi) q[16];
cx q[16],q[14];
rz(1.4726216) q[14];
sx q[16];
cx q[16],q[14];
x q[14];
rz(-2.4543693) q[14];
cx q[11],q[14];
rz(pi/64) q[14];
cx q[11],q[14];
rz(-pi/64) q[14];
sx q[14];
rz(-pi/2) q[14];
cx q[14],q[13];
rz(-pi/2) q[13];
sx q[14];
rz(-pi) q[14];
cx q[14],q[13];
rz(1.5462526) q[13];
sx q[14];
cx q[14],q[13];
rz(pi/128) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[12];
rz(-pi/2) q[12];
sx q[13];
rz(-pi) q[13];
cx q[13],q[12];
rz(1.5585245) q[12];
sx q[13];
cx q[13],q[12];
rz(-0.77312632) q[12];
sx q[12];
rz(-pi/2) q[12];
cx q[12],q[10];
rz(pi/2) q[10];
sx q[12];
rz(-pi) q[12];
cx q[12],q[10];
rz(1.5646604) q[10];
sx q[12];
cx q[12],q[10];
x q[10];
rz(2.3500586) q[10];
rz(2.3439226) q[12];
sx q[12];
rz(-pi) q[12];
rz(-1.59534) q[13];
sx q[13];
rz(-pi) q[13];
rz(-pi/2) q[14];
rz(-3*pi/4) q[16];
rz(-0.84730806) q[19];
rz(3*pi/4) q[22];
sx q[25];
rz(pi/2) q[25];
cx q[22],q[25];
rz(pi/4) q[25];
cx q[22],q[25];
sx q[22];
rz(pi/2) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(pi/8) q[22];
rz(-pi/4) q[25];
cx q[22],q[25];
rz(pi/8) q[25];
cx q[22],q[25];
cx q[22],q[19];
rz(pi/4) q[19];
cx q[22],q[19];
rz(-pi/4) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-3*pi/16) q[19];
sx q[22];
rz(pi/2) q[22];
rz(-pi/8) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
rz(pi/16) q[22];
cx q[19],q[22];
cx q[19],q[16];
rz(pi/8) q[16];
cx q[19],q[16];
rz(-pi/8) q[16];
rz(-pi/16) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[11],q[14];
rz(pi/32) q[14];
cx q[11],q[14];
rz(-pi/32) q[14];
rz(-pi/64) q[16];
cx q[16],q[14];
rz(pi/64) q[14];
cx q[16],q[14];
rz(-pi/64) q[14];
sx q[14];
rz(-pi/2) q[14];
cx q[14],q[13];
rz(-pi/2) q[13];
sx q[14];
rz(-pi) q[14];
cx q[14],q[13];
rz(1.5462526) q[13];
sx q[14];
cx q[14],q[13];
rz(pi/128) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[12];
rz(-pi/2) q[12];
sx q[13];
rz(-pi) q[13];
cx q[13],q[12];
rz(1.5585245) q[12];
sx q[13];
cx q[13],q[12];
x q[12];
rz(3.1293208) q[12];
rz(-1.6444274) q[13];
rz(-pi/2) q[14];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[11],q[14];
rz(pi/16) q[14];
cx q[11],q[14];
rz(-pi/16) q[14];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(-pi/32) q[16];
cx q[16],q[14];
rz(pi/32) q[14];
cx q[16],q[14];
rz(-pi/32) q[14];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-pi/64) q[16];
cx q[16],q[14];
rz(pi/64) q[14];
cx q[16],q[14];
rz(-pi/64) q[14];
cx q[13],q[14];
rz(pi/128) q[14];
cx q[13],q[14];
sx q[13];
rz(-pi) q[13];
rz(-pi/128) q[14];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(-pi/8) q[14];
rz(1.3230994) q[22];
sx q[22];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[22];
rz(pi/2) q[22];
sx q[25];
rz(-pi) q[25];
cx q[25],q[22];
rz(pi/4) q[22];
sx q[25];
cx q[25],q[22];
x q[22];
rz(3*pi/4) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
rz(pi/8) q[16];
cx q[14],q[16];
rz(-pi/8) q[16];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(-pi/16) q[19];
cx q[19],q[16];
rz(pi/16) q[16];
cx q[19],q[16];
rz(-pi/16) q[16];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(-pi/32) q[19];
cx q[19],q[16];
rz(pi/32) q[16];
cx q[19],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
sx q[14];
rz(-pi/2) q[14];
cx q[14],q[13];
rz(-pi/2) q[13];
sx q[14];
rz(-pi) q[14];
cx q[14],q[13];
rz(1.5217089) q[13];
sx q[14];
cx q[14],q[13];
x q[13];
rz(3.0925053) q[13];
rz(-0.88357293) q[14];
sx q[14];
rz(-pi) q[14];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(3*pi/16) q[16];
sx q[16];
rz(-pi) q[16];
rz(pi/4) q[19];
rz(-2.1084975) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
rz(pi/4) q[22];
cx q[19],q[22];
sx q[19];
rz(pi/2) q[19];
rz(-pi/4) q[22];
rz(pi/8) q[25];
cx q[25],q[22];
rz(pi/8) q[22];
cx q[25],q[22];
rz(-pi/8) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(-pi/4) q[19];
sx q[19];
rz(-pi/2) q[19];
cx q[19],q[16];
rz(-pi/2) q[16];
sx q[19];
rz(-pi) q[19];
cx q[19],q[16];
rz(7*pi/16) q[16];
sx q[19];
cx q[19],q[16];
rz(pi/16) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[14];
rz(-pi/2) q[14];
sx q[16];
rz(-pi) q[16];
cx q[16],q[14];
rz(1.4726216) q[14];
sx q[16];
cx q[16],q[14];
x q[14];
rz(-2.4543693) q[14];
rz(3.007153) q[16];
sx q[16];
rz(-pi) q[16];
rz(pi/8) q[19];
sx q[19];
cx q[25],q[22];
rz(pi/4) q[22];
cx q[25],q[22];
rz(-pi/4) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[19];
rz(pi/2) q[19];
sx q[22];
rz(-pi) q[22];
cx q[22],q[19];
rz(3*pi/8) q[19];
sx q[22];
cx q[22],q[19];
rz(-0.33078919) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[19],q[16];
rz(-pi/2) q[16];
sx q[19];
rz(-pi) q[19];
cx q[19],q[16];
rz(7*pi/16) q[16];
sx q[19];
cx q[19],q[16];
x q[16];
rz(2.2217548) q[16];
rz(-0.84730806) q[19];
rz(3*pi/4) q[22];
sx q[25];
rz(pi/2) q[25];
cx q[22],q[25];
rz(pi/4) q[25];
cx q[22],q[25];
sx q[22];
rz(pi/2) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(pi/8) q[22];
rz(-pi/4) q[25];
cx q[22],q[25];
rz(pi/8) q[25];
cx q[22],q[25];
cx q[22],q[19];
rz(pi/4) q[19];
cx q[22],q[19];
rz(-pi/4) q[19];
sx q[22];
rz(pi/2) q[22];
rz(-pi/8) q[25];
rz(2.7469333) q[26];
sx q[26];
barrier q[2],q[5],q[16],q[8],q[14],q[20],q[17],q[23],q[15],q[1],q[4],q[7],q[25],q[22],q[13],q[11],q[12],q[10],q[0],q[6],q[3],q[9],q[26],q[18],q[19],q[21],q[24];
measure q[22] -> meas[0];
measure q[19] -> meas[1];
measure q[25] -> meas[2];
measure q[16] -> meas[3];
measure q[14] -> meas[4];
measure q[13] -> meas[5];
measure q[11] -> meas[6];
measure q[12] -> meas[7];
measure q[10] -> meas[8];
measure q[15] -> meas[9];
measure q[26] -> meas[10];
