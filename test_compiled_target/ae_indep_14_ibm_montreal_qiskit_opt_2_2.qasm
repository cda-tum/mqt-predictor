OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[14];
rz(-3*pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-3*pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-3*pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-3*pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-3*pi/2) q[17];
sx q[17];
rz(1.5704128) q[17];
rz(-pi) q[18];
sx q[18];
rz(2.2142974) q[18];
sx q[18];
cx q[17],q[18];
sx q[18];
rz(2.2142974) q[18];
sx q[18];
rz(-pi) q[18];
cx q[17],q[18];
rz(-pi) q[18];
sx q[18];
rz(2.2142974) q[18];
sx q[18];
rz(-3*pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(-3*pi/2) q[20];
sx q[20];
rz(1.5462526) q[20];
rz(-3*pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[18];
sx q[18];
rz(1.2870023) q[18];
sx q[18];
rz(-pi) q[18];
cx q[21],q[18];
rz(-pi) q[18];
sx q[18];
rz(1.2870023) q[18];
sx q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(-pi/4096) q[18];
rz(-3*pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(-3*pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
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
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(-pi/2048) q[21];
rz(-3*pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[24],q[23];
sx q[23];
rz(2.0064163) q[23];
sx q[23];
rz(-pi) q[23];
cx q[24],q[23];
rz(-pi) q[23];
sx q[23];
rz(2.0064163) q[23];
sx q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(-pi/1024) q[23];
rz(-3*pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(-3*pi/2) q[26];
sx q[26];
rz(1.4787575) q[26];
cx q[26],q[25];
sx q[25];
rz(0.87124027) q[25];
sx q[25];
rz(-pi) q[25];
cx q[26],q[25];
rz(-pi) q[25];
sx q[25];
rz(0.87123975) q[25];
sx q[25];
cx q[24],q[25];
rz(-pi) q[25];
sx q[25];
rz(1.3991131) q[25];
sx q[25];
cx q[24],q[25];
rz(-pi/256) q[24];
sx q[25];
rz(1.3991131) q[25];
sx q[25];
rz(-pi) q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[20],q[19];
sx q[19];
rz(0.34336642) q[19];
sx q[19];
rz(-pi) q[19];
cx q[20],q[19];
rz(-pi) q[19];
sx q[19];
rz(0.34336645) q[19];
sx q[19];
cx q[16],q[19];
rz(-pi) q[19];
sx q[19];
rz(2.4548618) q[19];
sx q[19];
cx q[16],q[19];
sx q[19];
rz(2.4548597) q[19];
sx q[19];
rz(-pi) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
rz(-pi) q[16];
sx q[16];
rz(1.768131) q[16];
sx q[16];
cx q[14],q[16];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
sx q[16];
rz(1.7681268) q[16];
sx q[16];
rz(-pi) q[16];
cx q[14],q[16];
rz(-pi) q[16];
sx q[16];
rz(0.39465931) q[16];
sx q[16];
cx q[14],q[16];
sx q[16];
rz(0.39466095) q[16];
sx q[16];
rz(-pi) q[16];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[16];
sx q[16];
rz(2.352274) q[16];
sx q[16];
rz(-pi) q[16];
cx q[14],q[16];
rz(-pi) q[16];
sx q[16];
rz(2.3522708) q[16];
sx q[16];
cx q[19],q[16];
sx q[16];
rz(1.5629554) q[16];
sx q[16];
rz(-pi) q[16];
cx q[19],q[16];
rz(-pi) q[16];
sx q[16];
rz(1.562949) q[16];
sx q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/4) q[16];
cx q[22],q[19];
rz(-pi) q[19];
sx q[19];
rz(0.01568181) q[19];
sx q[19];
cx q[22],q[19];
sx q[19];
rz(0.01568181) q[19];
sx q[19];
rz(-pi) q[19];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
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
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(-pi/16) q[16];
rz(-pi/8) q[19];
cx q[16],q[19];
rz(pi/16) q[19];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
rz(pi/8) q[14];
cx q[14],q[11];
rz(pi/8) q[11];
cx q[14],q[11];
rz(-pi/8) q[11];
rz(pi/4) q[16];
cx q[14],q[16];
sx q[14];
rz(pi/2) q[14];
rz(-pi/4) q[16];
rz(-pi/16) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(-pi/64) q[19];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[8];
rz(-0.29452431) q[14];
cx q[14],q[16];
rz(pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[19],q[16];
rz(pi/64) q[16];
cx q[19],q[16];
rz(-pi/64) q[16];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[20],q[19];
rz(pi/128) q[19];
cx q[20],q[19];
rz(-pi/128) q[19];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[24],q[25];
rz(pi/256) q[25];
cx q[24],q[25];
rz(-pi/256) q[25];
cx q[26],q[25];
rz(pi/512) q[25];
cx q[26],q[25];
rz(-pi/512) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[22],q[25];
cx q[23],q[24];
rz(pi/1024) q[24];
cx q[23],q[24];
rz(-pi/1024) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[21],q[23];
rz(pi/2048) q[23];
cx q[21],q[23];
rz(-pi/2048) q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[18],q[21];
rz(pi/4096) q[21];
cx q[18],q[21];
rz(-pi/4096) q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[17],q[18];
rz(pi/8192) q[18];
cx q[17],q[18];
rz(-pi/8192) q[18];
cx q[17],q[18];
cx q[18],q[17];
cx q[17],q[18];
rz(-pi/2048) q[21];
rz(-pi/1024) q[23];
rz(-pi/512) q[24];
cx q[25],q[22];
cx q[22],q[25];
rz(-pi/128) q[22];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
rz(pi/16) q[11];
cx q[14],q[11];
rz(-pi/16) q[11];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(-pi/32) q[14];
cx q[14],q[11];
rz(pi/32) q[11];
cx q[14],q[11];
rz(-pi/32) q[11];
rz(-pi/8) q[16];
cx q[16],q[19];
rz(pi/8) q[19];
cx q[16],q[19];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(pi/4) q[11];
cx q[11],q[8];
rz(-pi/16) q[16];
rz(-pi/8) q[19];
cx q[16],q[19];
rz(pi/16) q[19];
cx q[16],q[19];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(-pi/16) q[19];
cx q[19],q[20];
cx q[20],q[19];
rz(-0.14726216) q[19];
cx q[19],q[16];
rz(pi/64) q[16];
cx q[19],q[16];
rz(-pi/64) q[16];
rz(pi/32) q[20];
cx q[19],q[20];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-pi/32) q[20];
cx q[22],q[19];
rz(pi/128) q[19];
cx q[22],q[19];
rz(-pi/128) q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(-pi/64) q[19];
cx q[19],q[20];
rz(pi/64) q[20];
cx q[19],q[20];
rz(-pi/64) q[20];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[26],q[25];
rz(pi/256) q[25];
cx q[26],q[25];
rz(-pi/256) q[25];
cx q[24],q[25];
rz(pi/512) q[25];
cx q[24],q[25];
rz(-pi/512) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[23],q[24];
rz(pi/1024) q[24];
cx q[23],q[24];
rz(-pi/1024) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[21],q[23];
rz(pi/2048) q[23];
cx q[21],q[23];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(-pi/1024) q[18];
rz(-pi/4096) q[21];
rz(-pi/2048) q[23];
cx q[21],q[23];
rz(pi/4096) q[23];
cx q[21],q[23];
rz(-pi/4096) q[23];
rz(-pi/512) q[24];
rz(pi/4) q[8];
cx q[11],q[8];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(pi/8) q[11];
rz(pi/4) q[14];
rz(-pi/4) q[8];
cx q[11],q[8];
rz(pi/8) q[8];
cx q[11],q[8];
cx q[11],q[14];
sx q[11];
rz(pi/2) q[11];
rz(-pi/4) q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(-3*pi/16) q[14];
rz(pi/8) q[16];
rz(-pi/8) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
rz(pi/16) q[11];
cx q[14],q[11];
rz(-pi/16) q[11];
cx q[14],q[16];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(pi/4) q[11];
cx q[11],q[8];
rz(-pi/8) q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(-0.29452431) q[16];
cx q[16],q[14];
rz(pi/32) q[14];
cx q[16],q[14];
rz(-pi/32) q[14];
rz(pi/16) q[19];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(-pi/16) q[19];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(-pi/256) q[22];
cx q[26],q[25];
rz(pi/128) q[25];
cx q[26],q[25];
rz(-pi/128) q[25];
cx q[22],q[25];
rz(pi/256) q[25];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(-pi/256) q[25];
cx q[24],q[25];
rz(pi/512) q[25];
cx q[24],q[25];
rz(-pi/512) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[22],q[25];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[18],q[21];
rz(pi/1024) q[21];
cx q[18],q[21];
rz(-pi/1024) q[21];
rz(-pi/2048) q[23];
cx q[23],q[21];
rz(pi/2048) q[21];
cx q[23],q[21];
rz(-pi/2048) q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
rz(-pi/512) q[21];
cx q[25],q[22];
cx q[22],q[25];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(-pi/128) q[22];
cx q[26],q[25];
rz(pi/64) q[25];
cx q[26],q[25];
rz(-pi/64) q[25];
cx q[22],q[25];
rz(pi/128) q[25];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[19];
rz(-pi/64) q[20];
rz(-pi/256) q[22];
rz(-pi/128) q[25];
cx q[22],q[25];
rz(pi/256) q[25];
cx q[22],q[25];
rz(-pi/256) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[21],q[23];
rz(pi/512) q[23];
cx q[21],q[23];
rz(-pi/512) q[23];
rz(-pi/1024) q[24];
cx q[24],q[23];
rz(pi/1024) q[23];
cx q[24],q[23];
rz(-pi/1024) q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
rz(-pi/256) q[23];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(-pi/32) q[22];
cx q[22],q[19];
rz(pi/32) q[19];
cx q[22],q[19];
rz(-pi/32) q[19];
cx q[20],q[19];
rz(pi/64) q[19];
cx q[20],q[19];
rz(-pi/64) q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[20];
rz(-pi/128) q[25];
cx q[25],q[22];
rz(pi/128) q[22];
cx q[25],q[22];
rz(-pi/128) q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[23],q[24];
rz(pi/256) q[24];
cx q[23],q[24];
rz(-pi/256) q[24];
rz(-pi/512) q[25];
cx q[25],q[24];
rz(pi/512) q[24];
cx q[25],q[24];
rz(-pi/512) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(-pi/128) q[24];
rz(pi/4) q[8];
cx q[11],q[8];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(pi/8) q[11];
rz(pi/4) q[14];
rz(-pi/4) q[8];
cx q[11],q[8];
rz(pi/8) q[8];
cx q[11],q[8];
cx q[11],q[14];
sx q[11];
rz(pi/2) q[11];
rz(-pi/4) q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(-3*pi/16) q[14];
rz(pi/8) q[16];
rz(-pi/8) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
rz(pi/16) q[11];
cx q[14],q[11];
rz(-pi/16) q[11];
cx q[14],q[16];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(pi/4) q[11];
cx q[11],q[8];
rz(-pi/8) q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(-0.29452431) q[16];
cx q[16],q[14];
rz(pi/32) q[14];
cx q[16],q[14];
rz(-pi/32) q[14];
rz(pi/16) q[19];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(-pi/16) q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(-0.14726216) q[19];
cx q[19],q[16];
rz(pi/64) q[16];
cx q[19],q[16];
rz(-pi/64) q[16];
rz(pi/32) q[22];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-pi/32) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(-pi/256) q[22];
cx q[24],q[25];
rz(pi/128) q[25];
cx q[24],q[25];
rz(-pi/128) q[25];
cx q[22],q[25];
rz(pi/256) q[25];
cx q[22],q[25];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(-pi/128) q[19];
rz(-pi/256) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
rz(-pi/64) q[25];
cx q[25],q[22];
rz(pi/64) q[22];
cx q[25],q[22];
rz(-pi/64) q[22];
cx q[19],q[22];
rz(pi/128) q[22];
cx q[19],q[22];
rz(-pi/128) q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(pi/4) q[8];
cx q[11],q[8];
sx q[11];
rz(pi/2) q[11];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(pi/8) q[11];
rz(-pi/4) q[8];
cx q[11],q[8];
rz(pi/8) q[8];
cx q[11],q[8];
cx q[11],q[14];
rz(pi/4) q[14];
cx q[11],q[14];
sx q[11];
rz(pi/2) q[11];
rz(-pi/4) q[14];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(pi/16) q[14];
rz(-pi/8) q[8];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
rz(pi/16) q[11];
cx q[14],q[11];
rz(-pi/16) q[11];
cx q[14],q[16];
rz(pi/8) q[16];
cx q[14],q[16];
rz(-pi/8) q[16];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
rz(pi/4) q[11];
cx q[14],q[11];
rz(-pi/4) q[11];
cx q[11],q[8];
sx q[14];
rz(pi/2) q[14];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(-pi/32) q[14];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
rz(pi/32) q[11];
cx q[14],q[11];
rz(-pi/32) q[11];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(-pi/16) q[16];
cx q[16],q[19];
rz(pi/16) q[19];
cx q[16],q[19];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(pi/8) q[11];
cx q[11],q[8];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(-pi/16) q[19];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(-0.14726216) q[19];
cx q[19],q[16];
rz(pi/64) q[16];
cx q[19],q[16];
rz(-pi/64) q[16];
cx q[19],q[22];
rz(pi/32) q[22];
cx q[19],q[22];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(-pi/32) q[22];
rz(pi/8) q[8];
cx q[11],q[8];
cx q[11],q[14];
rz(pi/4) q[14];
cx q[11],q[14];
sx q[11];
rz(pi/2) q[11];
rz(-pi/4) q[14];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(pi/16) q[14];
rz(-pi/8) q[8];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
rz(pi/16) q[11];
cx q[14],q[11];
rz(-pi/16) q[11];
cx q[14],q[16];
rz(pi/8) q[16];
cx q[14],q[16];
rz(-pi/8) q[16];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
rz(pi/4) q[11];
cx q[14],q[11];
rz(-pi/4) q[11];
sx q[14];
rz(pi/2) q[14];
barrier q[2],q[5],q[21],q[18],q[23],q[25],q[14],q[16],q[22],q[1],q[4],q[7],q[13],q[10],q[24],q[26],q[19],q[17],q[0],q[6],q[3],q[9],q[12],q[20],q[15],q[11],q[8];
measure q[14] -> meas[0];
measure q[11] -> meas[1];
measure q[16] -> meas[2];
measure q[8] -> meas[3];
measure q[22] -> meas[4];
measure q[19] -> meas[5];
measure q[25] -> meas[6];
measure q[24] -> meas[7];
measure q[23] -> meas[8];
measure q[21] -> meas[9];
measure q[18] -> meas[10];
measure q[26] -> meas[11];
measure q[17] -> meas[12];
measure q[20] -> meas[13];
