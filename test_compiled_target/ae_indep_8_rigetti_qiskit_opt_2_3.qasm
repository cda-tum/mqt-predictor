OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg meas[8];
rz(-3*pi/2) q[20];
rx(pi/2) q[20];
rz(3.0434179) q[20];
rz(-3*pi/2) q[49];
rx(pi/2) q[49];
rz(pi/4) q[49];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rx(pi/2) q[57];
rz(2.4980915) q[57];
rx(pi/2) q[57];
rz(-pi/2) q[61];
rx(-pi/2) q[61];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
rz(-3*pi/2) q[70];
rx(pi/2) q[70];
rz(3.117049) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(0.92729522) q[57];
rx(-pi/2) q[57];
cz q[70],q[57];
rx(-pi/2) q[57];
rz(0.92729522) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[57];
rz(1.8545904) q[57];
rx(-pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[57];
rz(1.8545904) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[56],q[57];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(1.5217089) q[57];
rz(pi/2) q[57];
rx(pi/2) q[57];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(15*pi/16) q[56];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[56],q[63];
cz q[20],q[63];
rx(-pi/2) q[63];
rz(2.5740044) q[63];
rx(pi/2) q[63];
cz q[20],q[63];
rx(pi/2) q[63];
rz(2.5740044) q[63];
rx(-pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[63];
rz(1.1351764) q[63];
rx(-pi/2) q[63];
cz q[56],q[63];
rx(-pi/2) q[63];
rz(1.1351764) q[63];
rx(pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[63];
rz(2.2703524) q[63];
rx(-pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rx(-pi/2) q[63];
rz(0.69955657) q[63];
rx(-pi/2) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi) q[62];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
rz(7*pi/8) q[63];
cz q[63],q[62];
cz q[49],q[62];
rx(-pi/2) q[62];
rz(1.7424795) q[62];
rx(pi/2) q[62];
cz q[49],q[62];
rx(pi/2) q[62];
rz(1.7424796) q[62];
rx(-pi/2) q[62];
cz q[61],q[62];
rx(pi/2) q[62];
rz(2.7982262) q[62];
rx(-pi/2) q[62];
cz q[61],q[62];
rx(-pi/2) q[62];
rz(1.2274299) q[62];
rx(-pi/2) q[62];
cz q[62],q[61];
rx(pi/2) q[61];
rz(pi) q[61];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
cz q[61],q[62];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[61];
rx(pi/2) q[61];
rz(pi/2) q[61];
rx(pi/2) q[62];
rz(pi/2) q[62];
cz q[49],q[62];
rx(pi/4) q[62];
cz q[49],q[62];
rx(-pi/4) q[62];
cz q[63],q[62];
rx(pi/8) q[62];
cz q[63],q[62];
rx(3*pi/8) q[62];
rz(pi/2) q[62];
rx(pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(3*pi/4) q[62];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[62],q[63];
cz q[56],q[63];
cz q[62],q[49];
rx(pi/4) q[49];
cz q[62],q[49];
rx(-pi/4) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/16) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(-pi/16) q[63];
cz q[20],q[63];
rx(pi/32) q[63];
cz q[20],q[63];
rx(1.4726216) q[63];
rz(pi/2) q[63];
rx(pi/2) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
cz q[57],q[56];
rx(pi/64) q[56];
cz q[57],q[56];
rx(-pi/64) q[56];
cz q[57],q[56];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(1.4726216) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(5*pi/8) q[62];
cz q[62],q[49];
rx(pi/8) q[49];
cz q[62],q[49];
rx(3*pi/8) q[49];
rz(pi/2) q[49];
rx(pi/2) q[49];
rx(pi/2) q[63];
rz(pi/2) q[63];
rx(pi/4) q[63];
cz q[62],q[63];
cz q[49],q[62];
rz(-pi/2) q[49];
rx(-pi/2) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[49];
rx(pi/2) q[49];
rz(pi) q[49];
rx(pi/2) q[62];
rz(pi/2) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(pi/2) q[49];
rx(-pi/4) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi) q[20];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi/2) q[20];
rx(7*pi/16) q[63];
rz(pi/2) q[63];
rx(pi/2) q[63];
cz q[63],q[62];
rx(pi/16) q[62];
cz q[63],q[62];
rx(-pi/16) q[62];
rz(3*pi/8) q[63];
cz q[63],q[20];
rx(pi/8) q[20];
cz q[63],q[20];
rx(-pi/8) q[20];
cz q[63],q[62];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/4) q[62];
rz(pi/2) q[62];
rx(pi/2) q[62];
cz q[62],q[49];
rx(pi/4) q[49];
cz q[62],q[49];
rx(-pi/4) q[49];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[56],q[63];
rx(pi/32) q[63];
cz q[56],q[63];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rx(1.4726216) q[63];
rz(pi/2) q[63];
rx(pi/2) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
rz(15*pi/16) q[63];
cz q[63],q[56];
cz q[63],q[20];
rx(pi/16) q[20];
cz q[63],q[20];
rx(-pi/16) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(5*pi/8) q[62];
cz q[62],q[49];
rx(pi/8) q[49];
cz q[62],q[49];
rx(-pi/8) q[49];
rx(pi/2) q[63];
rz(pi/2) q[63];
rx(pi/4) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[49];
rz(-pi/2) q[49];
rx(-pi/2) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(pi) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[49];
rx(pi/2) q[49];
rz(pi) q[49];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(pi/4) q[63];
rz(pi/2) q[63];
rx(pi/2) q[63];
cz q[70],q[57];
rx(pi/128) q[57];
cz q[70],q[57];
rx(1.5462526) q[57];
rz(pi/2) q[57];
rx(pi/2) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(3.0925053) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
cz q[57],q[56];
rx(pi/64) q[56];
cz q[57],q[56];
rx(1.5217089) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/8) q[56];
rx(pi/2) q[63];
rz(1.6689711) q[63];
cz q[63],q[20];
rx(pi/32) q[20];
cz q[63],q[20];
rx(1.4726216) q[20];
rz(pi/2) q[20];
rx(pi/2) q[20];
cz q[63],q[62];
rx(pi/16) q[62];
cz q[63],q[62];
rx(-pi/16) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(pi) q[49];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
cz q[62],q[49];
rx(pi/2) q[49];
rz(pi/2) q[49];
rx(pi/2) q[62];
rz(pi/2) q[62];
cz q[49],q[62];
cz q[63],q[56];
rx(3*pi/8) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
cz q[63],q[62];
rx(pi/4) q[62];
cz q[63],q[62];
rx(pi/4) q[62];
rz(pi/2) q[62];
rx(pi/2) q[62];
rx(pi/2) q[63];
rz(pi/2) q[63];
rx(pi/2) q[70];
rz(pi/2) q[70];
barrier q[17],q[26],q[35],q[32],q[41],q[50],q[59],q[4],q[1],q[68],q[65],q[10],q[74],q[19],q[28],q[25],q[37],q[34],q[43],q[52],q[70],q[58],q[3],q[67],q[12],q[76],q[21],q[18],q[30],q[27],q[36],q[45],q[54],q[51],q[60],q[5],q[69],q[14],q[78],q[23],q[56],q[29],q[38],q[47],q[44],q[62],q[53],q[20],q[7],q[71],q[16],q[13],q[77],q[22],q[31],q[40],q[57],q[46],q[55],q[0],q[64],q[8],q[9],q[73],q[6],q[63],q[15],q[79],q[24],q[33],q[42],q[39],q[48],q[61],q[2],q[66],q[11],q[49],q[75],q[72];
measure q[63] -> meas[0];
measure q[62] -> meas[1];
measure q[56] -> meas[2];
measure q[49] -> meas[3];
measure q[20] -> meas[4];
measure q[57] -> meas[5];
measure q[70] -> meas[6];
measure q[61] -> meas[7];
