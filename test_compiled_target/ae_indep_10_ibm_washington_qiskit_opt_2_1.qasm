OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[10];
rz(-3*pi/2) q[61];
sx q[61];
rz(1.5646604) q[61];
rz(-pi) q[62];
sx q[62];
rz(2.2142974) q[62];
sx q[62];
cx q[61],q[62];
sx q[62];
rz(2.2142974) q[62];
sx q[62];
rz(-pi) q[62];
cx q[61],q[62];
rz(-pi) q[62];
sx q[62];
rz(2.2142974) q[62];
sx q[62];
rz(-3*pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[63],q[62];
sx q[62];
rz(1.2870023) q[62];
sx q[62];
rz(-pi) q[62];
cx q[63],q[62];
rz(-pi) q[62];
sx q[62];
rz(1.2870023) q[62];
sx q[62];
rz(-3*pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[63],q[62];
rz(-pi) q[62];
sx q[62];
rz(0.56758825) q[62];
sx q[62];
cx q[63],q[62];
sx q[62];
rz(0.56758825) q[62];
sx q[62];
rz(-pi) q[62];
rz(-pi/128) q[63];
rz(-0.036815539) q[64];
rz(-3*pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[72],q[62];
sx q[62];
rz(2.0064163) q[62];
sx q[62];
rz(-pi) q[62];
cx q[72],q[62];
rz(-pi) q[62];
sx q[62];
rz(2.0064163) q[62];
sx q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(-pi/64) q[62];
rz(-3*pi/2) q[79];
sx q[79];
rz(pi/2) q[79];
rz(-3*pi/2) q[80];
sx q[80];
rz(pi/2) q[80];
rz(-3*pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(-3*pi/2) q[82];
sx q[82];
rz(0.88357293) q[82];
cx q[82],q[81];
sx q[81];
rz(0.87124027) q[81];
sx q[81];
rz(-pi) q[81];
cx q[82],q[81];
rz(-pi) q[81];
sx q[81];
rz(0.87123975) q[81];
sx q[81];
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
cx q[72],q[81];
sx q[81];
rz(0.34336642) q[81];
sx q[81];
rz(-pi) q[81];
cx q[72],q[81];
rz(-pi/8) q[72];
rz(-pi) q[81];
sx q[81];
rz(0.34336645) q[81];
sx q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
rz(-3*pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[91],q[79];
rz(-pi) q[79];
sx q[79];
rz(2.4548618) q[79];
sx q[79];
cx q[91],q[79];
sx q[79];
rz(2.4548597) q[79];
sx q[79];
rz(-pi) q[79];
cx q[80],q[79];
rz(-pi) q[79];
sx q[79];
rz(1.768131) q[79];
sx q[79];
cx q[80],q[79];
sx q[79];
rz(1.768131) q[79];
sx q[79];
rz(-pi) q[79];
cx q[79],q[91];
rz(pi/2) q[80];
sx q[80];
rz(pi/2) q[80];
cx q[91],q[79];
cx q[79],q[91];
rz(pi/4) q[79];
cx q[79],q[80];
rz(pi/4) q[80];
cx q[79],q[80];
sx q[79];
rz(pi/2) q[79];
rz(-pi/4) q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[72],q[81];
rz(-pi/16) q[80];
rz(pi/8) q[81];
cx q[72],q[81];
rz(-pi/8) q[81];
cx q[80],q[81];
rz(pi/16) q[81];
cx q[80],q[81];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
rz(pi/8) q[79];
rz(-pi/16) q[81];
cx q[82],q[81];
rz(pi/32) q[81];
cx q[82],q[81];
rz(-pi/32) q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[62],q[72];
rz(pi/64) q[72];
cx q[62],q[72];
rz(-pi/64) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
rz(pi/128) q[62];
cx q[63],q[62];
rz(-pi/128) q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-pi/64) q[62];
cx q[64],q[63];
rz(pi/256) q[63];
cx q[64],q[63];
rz(-pi/256) q[63];
rz(-pi/32) q[72];
rz(pi/4) q[81];
cx q[81],q[80];
rz(pi/4) q[80];
cx q[81],q[80];
rz(-pi/4) q[80];
cx q[79],q[80];
rz(pi/8) q[80];
cx q[79],q[80];
rz(-pi/8) q[80];
sx q[81];
rz(pi/2) q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[79],q[80];
rz(pi/4) q[80];
cx q[79],q[80];
sx q[79];
rz(pi/2) q[79];
rz(-pi/4) q[80];
cx q[82],q[81];
rz(pi/16) q[81];
cx q[82],q[81];
rz(-pi/16) q[81];
cx q[72],q[81];
rz(pi/32) q[81];
cx q[72],q[81];
rz(-pi/32) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
rz(pi/64) q[72];
cx q[62],q[72];
rz(-pi/64) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
rz(pi/512) q[62];
cx q[61],q[62];
rz(-pi/512) q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(-pi/256) q[62];
cx q[64],q[63];
rz(pi/128) q[63];
cx q[64],q[63];
rz(-pi/128) q[63];
cx q[62],q[63];
rz(pi/256) q[63];
cx q[62],q[63];
rz(-pi/256) q[63];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(-pi/64) q[63];
rz(-pi/32) q[72];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
rz(-pi/16) q[80];
cx q[82],q[81];
rz(pi/8) q[81];
cx q[82],q[81];
rz(-pi/8) q[81];
cx q[80],q[81];
rz(pi/16) q[81];
cx q[80],q[81];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
rz(pi/8) q[79];
rz(-pi/16) q[81];
cx q[72],q[81];
rz(pi/32) q[81];
cx q[72],q[81];
rz(-pi/32) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
rz(pi/64) q[62];
cx q[63],q[62];
rz(-pi/64) q[62];
rz(-pi/128) q[72];
cx q[72],q[62];
rz(pi/128) q[62];
cx q[72],q[62];
rz(-pi/128) q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-pi/32) q[62];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(pi/4) q[81];
cx q[81],q[80];
rz(pi/4) q[80];
cx q[81],q[80];
rz(-pi/4) q[80];
cx q[79],q[80];
rz(pi/8) q[80];
cx q[79],q[80];
rz(-pi/8) q[80];
sx q[81];
rz(pi/2) q[81];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[79],q[80];
rz(pi/4) q[80];
cx q[79],q[80];
sx q[79];
rz(pi/2) q[79];
rz(-pi/4) q[80];
rz(-3*pi/16) q[82];
cx q[82],q[81];
rz(pi/16) q[81];
cx q[82],q[81];
rz(-pi/16) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
rz(pi/32) q[72];
cx q[62],q[72];
rz(-pi/32) q[72];
rz(-pi/64) q[81];
cx q[81],q[72];
rz(pi/64) q[72];
cx q[81],q[72];
rz(-pi/64) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(-pi/16) q[72];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
rz(-pi/32) q[80];
cx q[82],q[81];
rz(pi/8) q[81];
cx q[82],q[81];
rz(-pi/8) q[81];
cx q[72],q[81];
rz(pi/16) q[81];
cx q[72],q[81];
rz(-pi/16) q[81];
cx q[80],q[81];
rz(pi/32) q[81];
cx q[80],q[81];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
rz(-pi/16) q[79];
rz(-pi/32) q[81];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
rz(pi/4) q[81];
cx q[81],q[80];
rz(pi/4) q[80];
cx q[81],q[80];
rz(-pi/4) q[80];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/8) q[81];
cx q[81],q[80];
rz(pi/8) q[80];
cx q[81],q[80];
rz(-pi/8) q[80];
cx q[79],q[80];
rz(pi/16) q[80];
cx q[79],q[80];
rz(-pi/16) q[80];
cx q[79],q[80];
cx q[80],q[79];
cx q[79],q[80];
cx q[81],q[72];
rz(pi/4) q[72];
cx q[81],q[72];
rz(-pi/4) q[72];
sx q[81];
rz(pi/2) q[81];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[81];
rz(pi/8) q[81];
cx q[81],q[72];
rz(pi/8) q[72];
cx q[81],q[72];
rz(-pi/8) q[72];
cx q[81],q[80];
rz(pi/4) q[80];
cx q[81],q[80];
rz(-pi/4) q[80];
sx q[81];
rz(pi/2) q[81];
barrier q[18],q[82],q[15],q[61],q[24],q[88],q[33],q[97],q[42],q[39],q[106],q[51],q[48],q[115],q[112],q[57],q[2],q[121],q[66],q[11],q[75],q[8],q[79],q[17],q[63],q[26],q[90],q[35],q[99],q[44],q[41],q[108],q[105],q[50],q[114],q[59],q[4],q[123],q[68],q[1],q[13],q[65],q[77],q[10],q[74],q[19],q[83],q[28],q[92],q[37],q[34],q[101],q[98],q[43],q[107],q[52],q[116],q[81],q[6],q[125],q[70],q[3],q[67],q[12],q[76],q[21],q[85],q[30],q[27],q[94],q[64],q[36],q[103],q[100],q[45],q[109],q[54],q[118],q[80],q[60],q[5],q[124],q[69],q[14],q[78],q[23],q[20],q[87],q[32],q[84],q[29],q[96],q[93],q[38],q[102],q[47],q[111],q[56],q[120],q[53],q[117],q[91],q[7],q[126],q[71],q[16],q[62],q[25],q[22],q[89],q[86],q[31],q[95],q[40],q[104],q[49],q[113],q[46],q[58],q[110],q[122],q[55],q[0],q[119],q[72],q[9],q[73];
measure q[81] -> meas[0];
measure q[80] -> meas[1];
measure q[72] -> meas[2];
measure q[79] -> meas[3];
measure q[82] -> meas[4];
measure q[62] -> meas[5];
measure q[63] -> meas[6];
measure q[64] -> meas[7];
measure q[61] -> meas[8];
measure q[91] -> meas[9];
