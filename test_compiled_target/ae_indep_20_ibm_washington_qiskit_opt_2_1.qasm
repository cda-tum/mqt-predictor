OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[20];
rz(-3*pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
rz(-3*pi/2) q[61];
sx q[61];
rz(1.5706525) q[61];
rz(-3*pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(-pi) q[63];
sx q[63];
rz(2.2142974) q[63];
sx q[63];
rz(-3*pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[64],q[63];
sx q[63];
rz(2.2142974) q[63];
sx q[63];
rz(-pi) q[63];
cx q[64],q[63];
cx q[54],q[64];
rz(-pi) q[63];
sx q[63];
rz(2.2142974) q[63];
sx q[63];
cx q[62],q[63];
sx q[63];
rz(1.2870023) q[63];
sx q[63];
rz(-pi) q[63];
cx q[62],q[63];
rz(-pi) q[63];
sx q[63];
rz(1.2870023) q[63];
sx q[63];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-pi/262144) q[63];
cx q[64],q[54];
cx q[54],q[64];
rz(-3*pi/2) q[65];
sx q[65];
rz(pi/2) q[65];
rz(-3*pi/2) q[66];
sx q[66];
rz(pi/2) q[66];
rz(-3*pi/2) q[67];
sx q[67];
rz(pi/2) q[67];
rz(-3*pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[72],q[62];
rz(-pi) q[62];
sx q[62];
rz(0.56758825) q[62];
sx q[62];
cx q[72],q[62];
sx q[62];
rz(0.56758825) q[62];
sx q[62];
rz(-pi) q[62];
cx q[61],q[62];
sx q[62];
rz(2.0064163) q[62];
sx q[62];
rz(-pi) q[62];
cx q[61],q[62];
rz(-pi) q[62];
sx q[62];
rz(2.0064163) q[62];
sx q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(-3*pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
rz(-3*pi/2) q[80];
sx q[80];
rz(1.5702211) q[80];
rz(-3*pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[81],q[72];
sx q[72];
rz(0.87124027) q[72];
sx q[72];
rz(-pi) q[72];
cx q[81],q[72];
rz(-pi) q[72];
sx q[72];
rz(0.87123975) q[72];
sx q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(-pi/32768) q[72];
cx q[80],q[81];
rz(-pi) q[81];
sx q[81];
rz(1.3991131) q[81];
sx q[81];
cx q[80],q[81];
sx q[81];
rz(1.3991131) q[81];
sx q[81];
rz(-pi) q[81];
rz(-3*pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[82],q[81];
sx q[81];
rz(0.34336642) q[81];
sx q[81];
rz(-pi) q[81];
cx q[82],q[81];
rz(-pi) q[81];
sx q[81];
rz(0.34336645) q[81];
sx q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(-pi/8192) q[81];
rz(-3*pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(-3*pi/2) q[84];
sx q[84];
rz(pi/2) q[84];
rz(-3*pi/2) q[85];
sx q[85];
rz(pi/2) q[85];
rz(-3*pi/2) q[86];
sx q[86];
rz(pi/2) q[86];
rz(-3*pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[92],q[83];
rz(-pi) q[83];
sx q[83];
rz(2.4548618) q[83];
sx q[83];
cx q[92],q[83];
sx q[83];
rz(2.4548597) q[83];
sx q[83];
rz(-pi) q[83];
rz(-3*pi/2) q[102];
sx q[102];
rz(pi/2) q[102];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[92],q[83];
rz(-pi) q[83];
sx q[83];
rz(1.768131) q[83];
sx q[83];
cx q[92],q[83];
sx q[83];
rz(1.7681268) q[83];
sx q[83];
rz(-pi) q[83];
cx q[84],q[83];
rz(-pi) q[83];
sx q[83];
rz(0.39465931) q[83];
sx q[83];
cx q[84],q[83];
sx q[83];
rz(0.39466095) q[83];
sx q[83];
rz(-pi) q[83];
cx q[82],q[83];
sx q[83];
rz(2.352274) q[83];
sx q[83];
rz(-pi) q[83];
cx q[82],q[83];
rz(-pi/512) q[82];
rz(-pi) q[83];
sx q[83];
rz(2.3522708) q[83];
sx q[83];
rz(-3*pi/2) q[103];
sx q[103];
rz(pi/2) q[103];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
rz(-pi/2048) q[102];
cx q[92],q[83];
sx q[83];
rz(1.5629554) q[83];
sx q[83];
rz(-pi) q[83];
cx q[92],q[83];
rz(-pi) q[83];
sx q[83];
rz(1.562949) q[83];
sx q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
rz(-pi) q[85];
sx q[85];
rz(0.01568181) q[85];
sx q[85];
cx q[73],q[85];
sx q[85];
rz(0.015694754) q[85];
sx q[85];
rz(-pi) q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
sx q[73];
rz(3.110229) q[73];
sx q[73];
rz(-pi) q[73];
cx q[66],q[73];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
rz(-0.14726216) q[67];
rz(-pi) q[73];
sx q[73];
rz(3.1102032) q[73];
sx q[73];
cx q[66],q[73];
sx q[73];
rz(3.0786654) q[73];
sx q[73];
rz(-pi) q[73];
cx q[66],q[73];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
rz(-pi) q[73];
sx q[73];
rz(3.0788137) q[73];
sx q[73];
cx q[66],q[73];
sx q[73];
rz(3.0157382) q[73];
sx q[73];
rz(-pi) q[73];
cx q[66],q[73];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
rz(-pi) q[73];
sx q[73];
rz(3.0160347) q[73];
sx q[73];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[85],q[73];
sx q[73];
rz(2.8908837) q[73];
sx q[73];
rz(-pi) q[73];
cx q[85],q[73];
rz(-pi) q[73];
sx q[73];
rz(2.8904767) q[73];
sx q[73];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[73];
sx q[73];
rz(2.6381747) q[73];
sx q[73];
rz(-pi) q[73];
cx q[85],q[73];
rz(-pi) q[73];
sx q[73];
rz(2.6393608) q[73];
sx q[73];
cx q[66],q[73];
sx q[73];
rz(2.1347568) q[73];
sx q[73];
rz(-pi) q[73];
cx q[66],q[73];
rz(pi/2) q[66];
sx q[66];
rz(pi/2) q[66];
rz(-pi) q[73];
sx q[73];
rz(2.1347568) q[73];
sx q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
rz(-pi/524288) q[64];
rz(-pi/32) q[65];
rz(-pi/16) q[66];
rz(pi/4) q[85];
cx q[85],q[73];
rz(pi/4) q[73];
cx q[85],q[73];
rz(-pi/4) q[73];
sx q[85];
rz(pi/2) q[85];
cx q[85],q[84];
cx q[84],q[85];
rz(pi/4) q[84];
rz(pi/8) q[85];
cx q[85],q[73];
rz(pi/8) q[73];
cx q[85],q[73];
rz(-pi/8) q[73];
cx q[66],q[73];
rz(pi/16) q[73];
cx q[66],q[73];
rz(-pi/16) q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[65],q[66];
rz(pi/32) q[66];
cx q[65],q[66];
rz(-pi/32) q[66];
cx q[67],q[66];
rz(pi/64) q[66];
cx q[67],q[66];
rz(-pi/64) q[66];
cx q[85],q[84];
rz(-pi/4) q[84];
sx q[85];
rz(pi/2) q[85];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
rz(-pi/128) q[73];
cx q[73],q[66];
rz(pi/128) q[66];
cx q[73],q[66];
rz(-pi/128) q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
rz(-pi/64) q[65];
rz(pi/8) q[85];
cx q[85],q[84];
rz(pi/8) q[84];
cx q[85],q[84];
rz(-pi/8) q[84];
cx q[85],q[86];
rz(pi/4) q[86];
cx q[85],q[86];
sx q[85];
rz(pi/2) q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
rz(-pi/16) q[73];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[73],q[85];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
rz(-pi/1024) q[84];
rz(pi/16) q[85];
cx q[73],q[85];
rz(-pi/16) q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[67],q[66];
rz(pi/32) q[66];
cx q[67],q[66];
rz(-pi/32) q[66];
cx q[65],q[66];
rz(pi/64) q[66];
cx q[65],q[66];
rz(-pi/64) q[66];
rz(pi/8) q[85];
rz(-pi/4) q[86];
cx q[85],q[86];
rz(pi/8) q[86];
cx q[85],q[86];
cx q[85],q[73];
rz(pi/4) q[73];
cx q[85],q[73];
rz(-pi/4) q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
sx q[85];
rz(pi/2) q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
rz(-pi/16) q[73];
rz(-pi/8) q[86];
rz(-pi/256) q[92];
cx q[92],q[83];
rz(pi/256) q[83];
cx q[92],q[83];
rz(-pi/256) q[83];
cx q[82],q[83];
rz(pi/512) q[83];
cx q[82],q[83];
rz(-pi/512) q[83];
cx q[84],q[83];
rz(pi/1024) q[83];
cx q[84],q[83];
rz(-pi/1024) q[83];
cx q[83],q[92];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[92],q[83];
cx q[83],q[92];
cx q[102],q[92];
rz(pi/2048) q[92];
cx q[102],q[92];
cx q[103],q[102];
cx q[102],q[103];
cx q[103],q[102];
rz(-pi/4096) q[102];
rz(-pi/2048) q[92];
cx q[102],q[92];
rz(pi/4096) q[92];
cx q[102],q[92];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
rz(-pi/1024) q[102];
rz(-pi/4096) q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[81],q[82];
rz(pi/8192) q[82];
cx q[81],q[82];
rz(-pi/8192) q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[80],q[81];
rz(pi/16384) q[81];
cx q[80],q[81];
rz(-pi/16384) q[81];
cx q[72],q[81];
rz(pi/32768) q[81];
cx q[72],q[81];
rz(-pi/32768) q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[61],q[62];
rz(pi/65536) q[62];
cx q[61],q[62];
rz(-pi/65536) q[62];
rz(-pi/131072) q[72];
cx q[72],q[62];
rz(pi/131072) q[62];
cx q[72],q[62];
rz(-pi/131072) q[62];
cx q[63],q[62];
rz(pi/262144) q[62];
cx q[63],q[62];
rz(-pi/262144) q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(-3.5952675e-05) q[63];
rz(-pi/4096) q[82];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi/256) q[84];
rz(-pi/128) q[92];
cx q[92],q[83];
rz(pi/128) q[83];
cx q[92],q[83];
rz(-pi/128) q[83];
cx q[84],q[83];
rz(pi/256) q[83];
cx q[84],q[83];
rz(-pi/256) q[83];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
rz(-pi/512) q[84];
cx q[84],q[83];
rz(pi/512) q[83];
cx q[84],q[83];
rz(-pi/512) q[83];
cx q[83],q[92];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[73],q[85];
rz(pi/16) q[85];
cx q[73],q[85];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
rz(pi/8) q[66];
cx q[66],q[67];
rz(pi/8) q[67];
cx q[66],q[67];
cx q[66],q[73];
rz(-pi/8) q[67];
rz(pi/4) q[73];
cx q[66],q[73];
sx q[66];
rz(pi/2) q[66];
cx q[66],q[65];
cx q[65],q[66];
rz(pi/4) q[65];
rz(pi/32) q[66];
rz(-pi/4) q[73];
rz(-pi/16) q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
rz(pi/32) q[73];
cx q[66],q[73];
cx q[66],q[67];
rz(pi/16) q[67];
cx q[66],q[67];
rz(-pi/16) q[67];
rz(-pi/32) q[73];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
rz(pi/8) q[73];
cx q[66],q[73];
cx q[66],q[65];
rz(-pi/4) q[65];
sx q[66];
rz(pi/2) q[66];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
rz(-pi/8) q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
rz(-pi/128) q[85];
rz(-0.085902924) q[86];
cx q[92],q[83];
cx q[83],q[92];
cx q[102],q[92];
rz(pi/1024) q[92];
cx q[102],q[92];
cx q[103],q[102];
cx q[102],q[103];
cx q[103],q[102];
rz(-pi/2048) q[102];
rz(-pi/1024) q[92];
cx q[102],q[92];
rz(pi/2048) q[92];
cx q[102],q[92];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
rz(-pi/512) q[102];
rz(-0.0092038847) q[103];
rz(-pi/2048) q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[82],q[83];
rz(pi/4096) q[83];
cx q[82],q[83];
rz(-pi/4096) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[80],q[81];
rz(pi/8192) q[81];
cx q[80],q[81];
rz(-pi/8192) q[81];
rz(-pi/16384) q[82];
cx q[82],q[81];
rz(pi/16384) q[81];
cx q[82],q[81];
rz(-pi/16384) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[61],q[62];
rz(pi/32768) q[62];
cx q[61],q[62];
rz(-pi/32768) q[62];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
rz(-pi/131072) q[62];
rz(-pi/65536) q[81];
cx q[81],q[72];
rz(pi/65536) q[72];
cx q[81],q[72];
rz(-pi/65536) q[72];
cx q[62],q[72];
rz(pi/131072) q[72];
cx q[62],q[72];
rz(-pi/131072) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
rz(pi/262144) q[62];
cx q[63],q[62];
rz(-pi/262144) q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(-pi/65536) q[62];
rz(-pi/16384) q[72];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
rz(-0.00067111659) q[80];
rz(-pi/4096) q[81];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
rz(-0.14726216) q[83];
cx q[83],q[84];
rz(pi/64) q[84];
cx q[83],q[84];
rz(-pi/64) q[84];
cx q[85],q[84];
rz(pi/128) q[84];
cx q[85],q[84];
cx q[73],q[85];
rz(-pi/128) q[84];
cx q[85],q[73];
cx q[73],q[85];
rz(-pi/64) q[73];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[83],q[84];
rz(pi/32) q[84];
cx q[83],q[84];
rz(-pi/32) q[84];
cx q[86],q[85];
rz(pi/256) q[85];
cx q[86],q[85];
rz(-pi/256) q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
rz(pi/64) q[85];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
rz(-pi/32) q[66];
rz(-pi/64) q[85];
cx q[86],q[85];
rz(pi/128) q[85];
cx q[86],q[85];
rz(-pi/128) q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
rz(-pi/16) q[85];
cx q[85],q[73];
rz(pi/16) q[73];
cx q[85],q[73];
rz(-pi/16) q[73];
cx q[66],q[73];
rz(pi/32) q[73];
cx q[66],q[73];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
rz(pi/16) q[65];
rz(-pi/32) q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
rz(-pi/8) q[73];
cx q[73],q[66];
rz(pi/8) q[66];
cx q[73],q[66];
rz(-pi/8) q[66];
cx q[65],q[66];
rz(pi/16) q[66];
cx q[65],q[66];
rz(-pi/16) q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
rz(pi/4) q[66];
cx q[66],q[67];
rz(pi/4) q[67];
cx q[66],q[67];
sx q[66];
rz(pi/2) q[66];
rz(-pi/4) q[67];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[65],q[66];
rz(pi/8) q[66];
cx q[65],q[66];
rz(-pi/8) q[66];
cx q[86],q[85];
rz(pi/64) q[85];
cx q[86],q[85];
rz(-pi/64) q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
rz(-pi/32) q[85];
cx q[85],q[73];
rz(pi/32) q[73];
cx q[85],q[73];
rz(-pi/32) q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
rz(-pi/16) q[73];
cx q[73],q[66];
rz(pi/16) q[66];
cx q[73],q[66];
rz(-pi/16) q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
cx q[65],q[66];
rz(pi/4) q[66];
cx q[65],q[66];
sx q[65];
rz(pi/2) q[65];
rz(-pi/4) q[66];
rz(-pi/8) q[67];
cx q[67],q[66];
rz(pi/8) q[66];
cx q[67],q[66];
rz(-pi/8) q[66];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[102],q[92];
rz(pi/512) q[92];
cx q[102],q[92];
rz(-pi/512) q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[103],q[102];
rz(pi/1024) q[102];
cx q[103],q[102];
rz(-pi/1024) q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
rz(-pi/256) q[83];
cx q[83],q[84];
rz(pi/256) q[84];
cx q[83],q[84];
rz(-pi/256) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[85],q[84];
cx q[84],q[85];
rz(pi/64) q[84];
rz(-0.17180585) q[85];
cx q[85],q[86];
rz(pi/128) q[86];
cx q[85],q[86];
cx q[85],q[84];
rz(-pi/64) q[84];
cx q[85],q[73];
rz(pi/32) q[73];
cx q[85],q[73];
rz(-pi/32) q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
rz(-pi/16) q[73];
cx q[73],q[66];
rz(pi/16) q[66];
cx q[73],q[66];
rz(-pi/16) q[66];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
rz(pi/4) q[66];
cx q[66],q[65];
rz(pi/4) q[65];
cx q[66],q[65];
rz(-pi/4) q[65];
sx q[66];
rz(pi/2) q[66];
cx q[66],q[73];
cx q[73],q[66];
rz(pi/8) q[66];
cx q[66],q[65];
rz(pi/8) q[65];
cx q[66],q[65];
rz(-pi/8) q[65];
rz(pi/4) q[73];
cx q[66],q[73];
sx q[66];
rz(pi/2) q[66];
rz(-pi/4) q[73];
rz(-pi/128) q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
rz(-pi/2048) q[92];
cx q[92],q[102];
rz(pi/2048) q[102];
cx q[92],q[102];
rz(-pi/2048) q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[103],q[102];
rz(pi/512) q[102];
cx q[103],q[102];
rz(-pi/512) q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
rz(pi/4096) q[82];
cx q[81],q[82];
rz(-pi/4096) q[82];
rz(-pi/8192) q[83];
cx q[83],q[82];
rz(pi/8192) q[82];
cx q[83],q[82];
rz(-pi/8192) q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[72],q[81];
rz(pi/16384) q[81];
cx q[72],q[81];
rz(-pi/16384) q[81];
cx q[80],q[81];
rz(pi/32768) q[81];
cx q[80],q[81];
rz(-pi/32768) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
rz(pi/65536) q[72];
cx q[62],q[72];
rz(-pi/65536) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
rz(pi/131072) q[62];
cx q[63],q[62];
rz(-pi/131072) q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-pi/65536) q[62];
rz(-pi/32768) q[72];
rz(-pi/8192) q[81];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(-pi/4096) q[82];
rz(-pi/1024) q[92];
cx q[92],q[102];
rz(pi/1024) q[102];
cx q[92],q[102];
rz(-pi/1024) q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
rz(-pi/512) q[84];
rz(-pi/2048) q[92];
cx q[92],q[102];
rz(pi/2048) q[102];
cx q[92],q[102];
rz(-pi/2048) q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
rz(-pi/256) q[102];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[102],q[92];
cx q[82],q[83];
rz(pi/4096) q[83];
cx q[82],q[83];
rz(-pi/4096) q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[81],q[82];
rz(pi/8192) q[82];
cx q[81],q[82];
rz(-pi/8192) q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[80],q[81];
rz(pi/16384) q[81];
cx q[80],q[81];
rz(-pi/16384) q[81];
cx q[72],q[81];
rz(pi/32768) q[81];
cx q[72],q[81];
rz(-pi/32768) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
rz(pi/65536) q[72];
cx q[62],q[72];
rz(-pi/65536) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(-0.0002876214) q[72];
rz(-pi/4096) q[82];
rz(pi/256) q[92];
cx q[102],q[92];
rz(-pi/256) q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[84],q[83];
rz(pi/512) q[83];
cx q[84],q[83];
rz(-pi/512) q[83];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
rz(-pi/256) q[85];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[103],q[102];
cx q[102],q[103];
cx q[103],q[102];
rz(-pi/1024) q[102];
rz(-0.023009712) q[103];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[102],q[92];
rz(-pi/128) q[83];
cx q[83],q[84];
rz(pi/128) q[84];
cx q[83],q[84];
rz(-pi/128) q[84];
cx q[85],q[84];
rz(pi/256) q[84];
cx q[85],q[84];
rz(-pi/256) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
rz(-pi/64) q[85];
cx q[85],q[86];
rz(pi/64) q[86];
cx q[85],q[86];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
rz(-0.29452431) q[66];
cx q[66],q[67];
rz(pi/32) q[67];
cx q[66],q[67];
cx q[66],q[65];
rz(pi/16) q[65];
cx q[66],q[65];
rz(-pi/16) q[65];
rz(-pi/32) q[67];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
rz(-pi/128) q[85];
rz(-pi/64) q[86];
cx q[85],q[86];
rz(pi/128) q[86];
cx q[85],q[86];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
rz(-0.3436117) q[66];
cx q[66],q[67];
rz(pi/64) q[67];
cx q[66],q[67];
cx q[66],q[65];
rz(pi/32) q[65];
cx q[66],q[65];
rz(-pi/32) q[65];
rz(-pi/64) q[67];
rz(-pi/8) q[73];
rz(-pi/128) q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[73],q[85];
rz(pi/8) q[85];
cx q[73],q[85];
rz(-pi/8) q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
rz(pi/16) q[73];
cx q[66],q[73];
rz(-pi/16) q[73];
rz(pi/4) q[85];
cx q[85],q[86];
rz(pi/4) q[86];
cx q[85],q[86];
sx q[85];
rz(pi/2) q[85];
rz(-pi/4) q[86];
rz(pi/1024) q[92];
cx q[102],q[92];
rz(-pi/1024) q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[103],q[102];
rz(pi/2048) q[102];
cx q[103],q[102];
rz(-pi/2048) q[102];
rz(-pi/512) q[92];
cx q[92],q[83];
rz(pi/512) q[83];
cx q[92],q[83];
rz(-pi/512) q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
rz(-pi/256) q[83];
cx q[83],q[84];
rz(pi/256) q[84];
cx q[83],q[84];
rz(-pi/256) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
rz(pi/128) q[66];
cx q[66],q[67];
rz(pi/128) q[67];
cx q[66],q[67];
cx q[66],q[65];
rz(pi/64) q[65];
cx q[66],q[65];
rz(-pi/64) q[65];
rz(-pi/128) q[67];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
rz(pi/32) q[73];
cx q[66],q[73];
rz(-pi/32) q[73];
rz(pi/8) q[85];
cx q[85],q[86];
rz(pi/8) q[86];
cx q[85],q[86];
cx q[85],q[84];
rz(pi/4) q[84];
cx q[85],q[84];
rz(-pi/4) q[84];
sx q[85];
rz(pi/2) q[85];
rz(-pi/8) q[86];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
rz(pi/16) q[73];
cx q[66],q[73];
rz(-pi/16) q[73];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
rz(pi/8) q[73];
cx q[66],q[73];
rz(-pi/8) q[73];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
rz(pi/4) q[73];
cx q[66],q[73];
sx q[66];
rz(pi/2) q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
rz(-pi/4) q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[103],q[102];
rz(pi/1024) q[102];
cx q[103],q[102];
rz(-pi/1024) q[102];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[102],q[92];
cx q[82],q[83];
rz(pi/4096) q[83];
cx q[82],q[83];
rz(-pi/4096) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[80],q[81];
rz(pi/8192) q[81];
cx q[80],q[81];
rz(-pi/8192) q[81];
rz(-pi/16384) q[82];
cx q[82],q[81];
rz(pi/16384) q[81];
cx q[82],q[81];
rz(-pi/16384) q[81];
cx q[72],q[81];
rz(pi/32768) q[81];
cx q[72],q[81];
rz(-pi/32768) q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
rz(-pi/4096) q[81];
rz(-pi/2048) q[83];
cx q[92],q[102];
cx q[102],q[92];
cx q[103],q[102];
rz(pi/512) q[102];
cx q[103],q[102];
rz(-pi/512) q[102];
cx q[83],q[92];
rz(pi/2048) q[92];
cx q[83],q[92];
rz(-pi/2048) q[92];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
rz(pi/4096) q[82];
cx q[81],q[82];
rz(-pi/4096) q[82];
rz(-pi/8192) q[83];
cx q[83],q[82];
rz(pi/8192) q[82];
cx q[83],q[82];
rz(-pi/8192) q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[72],q[81];
rz(pi/16384) q[81];
cx q[72],q[81];
rz(-pi/16384) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(-pi/8192) q[81];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
rz(-pi/4096) q[82];
rz(-pi/1024) q[92];
cx q[92],q[102];
rz(pi/1024) q[102];
cx q[92],q[102];
rz(-pi/1024) q[102];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi/512) q[84];
rz(-pi/2048) q[92];
cx q[92],q[102];
rz(pi/2048) q[102];
cx q[92],q[102];
rz(-pi/2048) q[102];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[103],q[102];
rz(pi/256) q[102];
cx q[103],q[102];
rz(-pi/256) q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[82],q[83];
rz(pi/4096) q[83];
cx q[82],q[83];
rz(-pi/4096) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
rz(pi/8192) q[82];
cx q[81],q[82];
rz(-pi/8192) q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(-pi/4096) q[82];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
rz(-0.0092038847) q[103];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[84],q[83];
rz(pi/512) q[83];
cx q[84],q[83];
rz(-pi/512) q[83];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
rz(-pi/256) q[85];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[103],q[102];
rz(pi/1024) q[102];
cx q[103],q[102];
rz(-pi/1024) q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
rz(-pi/128) q[83];
cx q[83],q[84];
rz(pi/128) q[84];
cx q[83],q[84];
rz(-pi/128) q[84];
cx q[85],q[84];
rz(pi/256) q[84];
cx q[85],q[84];
rz(-pi/256) q[84];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi/64) q[84];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
rz(-pi/128) q[73];
cx q[84],q[85];
rz(pi/64) q[85];
cx q[84],q[85];
rz(-pi/64) q[85];
cx q[73],q[85];
rz(pi/128) q[85];
cx q[73],q[85];
rz(-pi/128) q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
rz(-pi/32) q[85];
cx q[85],q[86];
rz(pi/32) q[86];
cx q[85],q[86];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
rz(-pi/16) q[73];
cx q[73],q[66];
rz(pi/16) q[66];
cx q[73],q[66];
rz(-pi/16) q[66];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
rz(pi/8) q[66];
cx q[66],q[65];
rz(pi/8) q[65];
cx q[66],q[65];
rz(-pi/8) q[65];
cx q[66],q[67];
rz(pi/4) q[67];
cx q[66],q[67];
sx q[66];
rz(pi/2) q[66];
rz(-pi/4) q[67];
rz(-0.14726216) q[85];
rz(-pi/32) q[86];
cx q[85],q[86];
rz(pi/64) q[86];
cx q[85],q[86];
cx q[85],q[73];
rz(pi/32) q[73];
cx q[85],q[73];
rz(-pi/32) q[73];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
rz(pi/16) q[66];
cx q[66],q[65];
rz(pi/16) q[65];
cx q[66],q[65];
rz(-pi/16) q[65];
cx q[66],q[67];
rz(pi/8) q[67];
cx q[66],q[67];
cx q[66],q[73];
rz(-pi/8) q[67];
rz(pi/4) q[73];
cx q[66],q[73];
sx q[66];
rz(pi/2) q[66];
rz(-pi/4) q[73];
rz(-pi/64) q[86];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
rz(-pi/2048) q[92];
cx q[92],q[102];
rz(pi/2048) q[102];
cx q[92],q[102];
rz(-pi/2048) q[102];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[103],q[102];
rz(pi/512) q[102];
cx q[103],q[102];
rz(-pi/512) q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[82],q[83];
rz(pi/4096) q[83];
cx q[82],q[83];
rz(-pi/4096) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(-pi/1024) q[92];
cx q[92],q[102];
rz(pi/1024) q[102];
cx q[92],q[102];
rz(-pi/1024) q[102];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi/2048) q[92];
cx q[92],q[102];
rz(pi/2048) q[102];
cx q[92],q[102];
rz(-pi/2048) q[102];
cx q[103],q[102];
cx q[102],q[103];
cx q[103],q[102];
rz(-pi/256) q[102];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[102],q[92];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
rz(-pi/512) q[83];
rz(pi/256) q[92];
cx q[102],q[92];
rz(-pi/256) q[92];
cx q[83],q[92];
rz(pi/512) q[92];
cx q[83],q[92];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi/1024) q[83];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
rz(-pi/512) q[92];
cx q[83],q[92];
rz(pi/1024) q[92];
cx q[83],q[92];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
rz(-pi/256) q[84];
rz(-pi/1024) q[92];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
rz(-pi/128) q[92];
cx q[92],q[83];
rz(pi/128) q[83];
cx q[92],q[83];
rz(-pi/128) q[83];
cx q[84],q[83];
rz(pi/256) q[83];
cx q[84],q[83];
rz(-pi/256) q[83];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
rz(-pi/512) q[84];
cx q[84],q[83];
rz(pi/512) q[83];
cx q[84],q[83];
rz(-pi/512) q[83];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
rz(-pi/64) q[85];
cx q[85],q[86];
rz(pi/64) q[86];
cx q[85],q[86];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
rz(-0.29452431) q[66];
cx q[66],q[65];
rz(pi/32) q[65];
cx q[66],q[65];
rz(-pi/32) q[65];
cx q[66],q[67];
rz(pi/16) q[67];
cx q[66],q[67];
rz(-pi/16) q[67];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi/128) q[85];
rz(-pi/64) q[86];
cx q[85],q[86];
rz(pi/128) q[86];
cx q[85],q[86];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
rz(-0.14726216) q[66];
cx q[66],q[65];
rz(pi/64) q[65];
cx q[66],q[65];
rz(-pi/64) q[65];
cx q[66],q[67];
rz(pi/32) q[67];
cx q[66],q[67];
rz(-pi/32) q[67];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
rz(pi/8) q[84];
cx q[84],q[83];
rz(pi/8) q[83];
cx q[84],q[83];
rz(-pi/8) q[83];
rz(-pi/256) q[85];
rz(-pi/128) q[86];
cx q[85],q[86];
rz(pi/256) q[86];
cx q[85],q[86];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
rz(-0.073631078) q[66];
cx q[66],q[65];
rz(pi/128) q[65];
cx q[66],q[65];
rz(-pi/128) q[65];
cx q[66],q[67];
rz(pi/64) q[67];
cx q[66],q[67];
rz(-pi/64) q[67];
cx q[84],q[85];
rz(pi/4) q[85];
cx q[84],q[85];
sx q[84];
rz(pi/2) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
rz(-pi/4) q[85];
cx q[85],q[73];
cx q[73],q[85];
rz(pi/8) q[73];
rz(-3*pi/16) q[85];
cx q[85],q[84];
rz(pi/16) q[84];
cx q[85],q[84];
rz(-pi/16) q[84];
cx q[85],q[73];
rz(-pi/8) q[73];
cx q[73],q[66];
cx q[66],q[73];
rz(pi/16) q[66];
rz(-0.29452431) q[73];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[73],q[85];
rz(pi/4) q[84];
cx q[84],q[83];
rz(pi/4) q[83];
cx q[84],q[83];
rz(-pi/4) q[83];
sx q[84];
rz(pi/2) q[84];
rz(pi/32) q[85];
cx q[73],q[85];
cx q[73],q[66];
rz(-pi/16) q[66];
rz(-pi/32) q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[84],q[85];
cx q[85],q[84];
rz(pi/8) q[84];
cx q[84],q[83];
rz(pi/8) q[83];
cx q[84],q[83];
rz(-pi/8) q[83];
rz(pi/4) q[85];
cx q[84],q[85];
sx q[84];
rz(pi/2) q[84];
rz(-pi/4) q[85];
rz(-pi/256) q[86];
barrier q[18],q[65],q[15],q[79],q[24],q[88],q[33],q[97],q[42],q[39],q[106],q[51],q[48],q[115],q[112],q[57],q[2],q[121],q[72],q[11],q[75],q[8],q[83],q[17],q[73],q[26],q[90],q[35],q[99],q[44],q[41],q[108],q[105],q[50],q[114],q[59],q[4],q[123],q[68],q[1],q[13],q[62],q[77],q[10],q[74],q[19],q[103],q[28],q[86],q[37],q[34],q[101],q[98],q[43],q[107],q[52],q[116],q[66],q[6],q[125],q[70],q[3],q[80],q[12],q[76],q[21],q[61],q[30],q[27],q[94],q[91],q[36],q[82],q[100],q[45],q[109],q[64],q[118],q[54],q[60],q[5],q[124],q[69],q[14],q[78],q[23],q[20],q[87],q[32],q[102],q[29],q[96],q[93],q[38],q[92],q[47],q[111],q[56],q[120],q[53],q[117],q[85],q[7],q[126],q[71],q[16],q[67],q[25],q[22],q[89],q[63],q[31],q[95],q[40],q[104],q[49],q[113],q[46],q[58],q[110],q[122],q[55],q[0],q[119],q[84],q[9],q[81];
measure q[84] -> meas[0];
measure q[85] -> meas[1];
measure q[83] -> meas[2];
measure q[66] -> meas[3];
measure q[73] -> meas[4];
measure q[67] -> meas[5];
measure q[65] -> meas[6];
measure q[86] -> meas[7];
measure q[92] -> meas[8];
measure q[102] -> meas[9];
measure q[103] -> meas[10];
measure q[82] -> meas[11];
measure q[81] -> meas[12];
measure q[72] -> meas[13];
measure q[80] -> meas[14];
measure q[62] -> meas[15];
measure q[63] -> meas[16];
measure q[61] -> meas[17];
measure q[64] -> meas[18];
measure q[54] -> meas[19];
