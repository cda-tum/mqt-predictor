OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg meas[3];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[62];
rx(0.50099722) q[62];
cz q[63],q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[63],q[62];
rx(0.50099722) q[62];
cz q[63],q[62];
rz(1.7104231) q[62];
cz q[63],q[56];
rx(0.50099722) q[56];
cz q[63],q[56];
rz(3.2812194) q[56];
rx(-4.5727622) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[63],q[62];
rx(-0.81646326) q[62];
cz q[63],q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[63],q[62];
rx(-0.81646326) q[62];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(-2.4240612) q[62];
cz q[63],q[56];
rx(-0.81646326) q[56];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(-2.4240612) q[56];
rx(2.2883278) q[63];
barrier q[41],q[50],q[59],q[4],q[1],q[68],q[65],q[10],q[74],q[19],q[28],q[25],q[37],q[34],q[43],q[52],q[61],q[58],q[3],q[67],q[12],q[76],q[21],q[18],q[30],q[27],q[36],q[45],q[54],q[51],q[60],q[5],q[69],q[14],q[78],q[23],q[20],q[29],q[38],q[47],q[44],q[63],q[53],q[62],q[7],q[71],q[16],q[13],q[77],q[22],q[31],q[40],q[49],q[46],q[55],q[0],q[64],q[9],q[73],q[6],q[70],q[15],q[79],q[24],q[33],q[42],q[39],q[48],q[57],q[2],q[66],q[11],q[56],q[75],q[8],q[72],q[17],q[26],q[35],q[32];
measure q[63] -> meas[0];
measure q[56] -> meas[1];
measure q[62] -> meas[2];
