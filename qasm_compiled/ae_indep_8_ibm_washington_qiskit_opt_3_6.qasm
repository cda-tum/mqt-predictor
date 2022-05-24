OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[8];
rz(-3*pi/2) q[54];
sx q[54];
rz(1.4235342) q[54];
rz(-3*pi/2) q[61];
sx q[61];
rz(5*pi/8) q[61];
rz(-3*pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(-3*pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
rz(-pi) q[64];
sx q[64];
rz(2.2142974) q[64];
sx q[64];
rz(-3*pi/2) q[65];
sx q[65];
rz(pi/2) q[65];
cx q[65],q[64];
sx q[64];
rz(2.2142974) q[64];
sx q[64];
rz(-pi) q[64];
cx q[65],q[64];
rz(-pi) q[64];
sx q[64];
rz(2.2142974) q[64];
sx q[64];
cx q[54],q[64];
sx q[64];
rz(1.2870023) q[64];
sx q[64];
rz(-pi) q[64];
cx q[54],q[64];
rz(-pi) q[64];
sx q[64];
rz(1.2870023) q[64];
sx q[64];
cx q[63],q[64];
rz(-pi) q[64];
sx q[64];
rz(0.56758825) q[64];
sx q[64];
cx q[63],q[64];
sx q[64];
rz(0.56758825) q[64];
sx q[64];
rz(-pi) q[64];
rz(-3*pi/2) q[66];
sx q[66];
rz(pi/2) q[66];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[65],q[64];
sx q[64];
rz(2.0064163) q[64];
sx q[64];
rz(-pi) q[64];
cx q[65],q[64];
rz(-pi) q[64];
sx q[64];
rz(2.0064163) q[64];
sx q[64];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
sx q[62];
rz(0.87124027) q[62];
sx q[62];
rz(-pi) q[62];
cx q[61],q[62];
rz(-pi) q[62];
sx q[62];
rz(0.87123975) q[62];
sx q[62];
cx q[63],q[62];
rz(-pi) q[62];
sx q[62];
rz(1.3991131) q[62];
sx q[62];
cx q[63],q[62];
rz(-pi) q[62];
sx q[62];
rz(0.17168327) q[62];
sx q[62];
rz(pi/4) q[63];
rz(-pi/16) q[65];
rz(-pi/128) q[66];
sx q[66];
rz(-pi/2) q[66];
rz(-pi) q[72];
sx q[72];
cx q[72],q[62];
rz(-pi/2) q[62];
sx q[72];
rz(-pi) q[72];
cx q[72],q[62];
rz(1.2274299) q[62];
sx q[72];
cx q[72],q[62];
rz(pi/2) q[62];
sx q[62];
rz(-pi/2) q[62];
cx q[63],q[62];
rz(pi/4) q[62];
cx q[63],q[62];
rz(-pi/4) q[62];
cx q[61],q[62];
rz(pi/8) q[62];
cx q[61],q[62];
rz(-pi/8) q[62];
sx q[63];
rz(pi/2) q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
rz(pi/4) q[62];
cx q[61],q[62];
sx q[61];
rz(pi/2) q[61];
rz(-pi/4) q[62];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(-pi/32) q[63];
cx q[65],q[64];
rz(pi/16) q[64];
cx q[65],q[64];
rz(-pi/16) q[64];
cx q[63],q[64];
rz(pi/32) q[64];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-pi/16) q[62];
rz(-pi/32) q[64];
cx q[54],q[64];
rz(pi/64) q[64];
cx q[54],q[64];
rz(-pi/64) q[64];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
rz(-pi/8) q[64];
cx q[64],q[63];
rz(pi/8) q[63];
cx q[64],q[63];
rz(-pi/8) q[63];
cx q[62],q[63];
rz(pi/16) q[63];
cx q[62],q[63];
rz(-pi/16) q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[54],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(pi/4) q[62];
cx q[62],q[61];
rz(pi/4) q[61];
cx q[62],q[61];
rz(-pi/4) q[61];
sx q[62];
rz(pi/2) q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(pi/8) q[62];
cx q[62],q[61];
rz(pi/8) q[61];
cx q[62],q[61];
rz(-pi/8) q[61];
cx q[62],q[63];
rz(pi/4) q[63];
cx q[62],q[63];
sx q[62];
rz(pi/2) q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(-pi/4) q[63];
rz(pi/32) q[64];
cx q[54],q[64];
rz(-pi/32) q[64];
sx q[65];
rz(-pi) q[65];
cx q[66],q[65];
rz(-pi/2) q[65];
sx q[66];
rz(-pi) q[66];
cx q[66],q[65];
rz(1.5462526) q[65];
sx q[66];
cx q[66],q[65];
x q[65];
rz(3.0925053) q[65];
cx q[65],q[64];
rz(pi/64) q[64];
cx q[65],q[64];
rz(-pi/64) q[64];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(-3*pi/16) q[63];
cx q[63],q[62];
rz(pi/16) q[62];
cx q[63],q[62];
rz(-pi/16) q[62];
cx q[63],q[64];
rz(pi/8) q[64];
cx q[63],q[64];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(pi/4) q[62];
cx q[62],q[61];
rz(pi/4) q[61];
cx q[62],q[61];
rz(-pi/4) q[61];
sx q[62];
rz(pi/2) q[62];
rz(-pi/8) q[64];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
rz(-0.29452431) q[64];
cx q[64],q[63];
rz(pi/32) q[63];
cx q[64],q[63];
rz(-pi/32) q[63];
cx q[64],q[65];
rz(pi/16) q[65];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
rz(pi/8) q[62];
cx q[62],q[61];
rz(pi/8) q[61];
cx q[62],q[61];
rz(-pi/8) q[61];
rz(pi/4) q[63];
cx q[62],q[63];
sx q[62];
rz(pi/2) q[62];
rz(-pi/4) q[63];
rz(-pi/16) q[65];
rz(-1.59534) q[66];
rz(-0.34336642) q[72];
sx q[72];
rz(-pi) q[72];
barrier q[15],q[67],q[12],q[79],q[76],q[21],q[85],q[30],q[94],q[39],q[103],q[36],q[100],q[45],q[109],q[63],q[118],q[61],q[8],q[5],q[66],q[69],q[14],q[78],q[23],q[87],q[32],q[96],q[29],q[41],q[93],q[105],q[38],q[102],q[47],q[111],q[56],q[1],q[120],q[62],q[54],q[7],q[126],q[71],q[16],q[80],q[25],q[89],q[22],q[34],q[98],q[31],q[95],q[40],q[104],q[49],q[113],q[58],q[55],q[122],q[0],q[119],q[3],q[72],q[9],q[73],q[18],q[82],q[27],q[91],q[24],q[88],q[33],q[97],q[42],q[106],q[51],q[48],q[115],q[60],q[112],q[57],q[124],q[2],q[121],q[65],q[11],q[75],q[20],q[84],q[17],q[81],q[26],q[90],q[35],q[99],q[44],q[108],q[53],q[50],q[117],q[114],q[59],q[4],q[123],q[68],q[13],q[77],q[10],q[74],q[86],q[19],q[83],q[28],q[92],q[37],q[101],q[46],q[43],q[110],q[107],q[52],q[116],q[64],q[6],q[125],q[70];
measure q[62] -> meas[0];
measure q[63] -> meas[1];
measure q[61] -> meas[2];
measure q[65] -> meas[3];
measure q[64] -> meas[4];
measure q[54] -> meas[5];
measure q[66] -> meas[6];
measure q[72] -> meas[7];
