OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg meas[4];
rz(-pi/2) q[19];
rx(pi/2) q[19];
rz(3.2400137) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi) q[57];
rx(-2.7344077) q[57];
cz q[56],q[57];
rx(pi) q[56];
rx(0.92729522) q[57];
rz(pi/2) q[57];
cz q[56],q[57];
rz(2.8577985) q[56];
rx(pi/2) q[57];
rz(1.9779813) q[57];
rx(-pi/2) q[57];
rz(-pi) q[63];
rx(-0.56839927) q[63];
cz q[56],q[63];
rx(pi) q[56];
rx(1.2870023) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(-1.2870023) q[56];
cz q[19],q[56];
rx(pi) q[19];
rx(0.56758825) q[56];
cz q[19],q[56];
rx(1.538054) q[19];
rz(1.2506585) q[19];
rx(0.10370774) q[19];
rx(1.0032081) q[56];
rx(2.5731934) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi/4) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(-pi/2) q[56];
rz(-pi/2) q[56];
cz q[19],q[56];
rz(pi/2) q[19];
rx(pi) q[19];
rx(pi/4) q[56];
rz(pi/2) q[56];
cz q[19],q[56];
rz(-2.677945) q[19];
rx(pi/2) q[19];
rz(-pi/2) q[19];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(-pi/2) q[56];
cz q[56],q[57];
rx(pi/2) q[56];
rz(-pi/2) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(5*pi/8) q[56];
cz q[56],q[19];
rx(pi/8) q[19];
cz q[56],q[19];
rx(3*pi/8) q[19];
rz(pi/2) q[19];
rx(pi/2) q[19];
rz(pi/2) q[57];
cz q[56],q[57];
rx(pi/4) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/4) q[57];
rz(pi/2) q[57];
rx(pi/2) q[57];
rx(pi/2) q[63];
rz(pi/2) q[63];
barrier q[17],q[26],q[35],q[32],q[41],q[50],q[59],q[4],q[1],q[68],q[65],q[10],q[74],q[19],q[28],q[25],q[37],q[34],q[43],q[52],q[61],q[58],q[3],q[67],q[12],q[76],q[21],q[18],q[30],q[27],q[36],q[45],q[54],q[51],q[60],q[5],q[69],q[14],q[78],q[23],q[20],q[29],q[38],q[47],q[44],q[63],q[53],q[62],q[7],q[71],q[16],q[13],q[77],q[22],q[31],q[40],q[49],q[46],q[55],q[0],q[64],q[8],q[9],q[73],q[6],q[70],q[15],q[79],q[24],q[33],q[42],q[39],q[48],q[56],q[2],q[66],q[11],q[57],q[75],q[72];
measure q[56] -> meas[0];
measure q[57] -> meas[1];
measure q[19] -> meas[2];
measure q[63] -> meas[3];
