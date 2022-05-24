OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg c[3];
creg meas[3];
rx(3*pi/4) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
rz(pi/2) q[63];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[56],q[63];
rx(-pi/4) q[63];
cz q[56],q[63];
rz(5*pi/8) q[56];
rx(3*pi/4) q[63];
rz(pi/2) q[63];
rx(pi/2) q[63];
cz q[62],q[63];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(3*pi/4) q[62];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[62],q[63];
cz q[56],q[63];
rx(-pi/8) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/8) q[63];
cz q[62],q[63];
rx(-pi/4) q[63];
cz q[62],q[63];
rx(pi/4) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi/2) q[56];
barrier q[70],q[15],q[79],q[24],q[21],q[30],q[39],q[48],q[57],q[54],q[62],q[8],q[72],q[17],q[14],q[26],q[78],q[23],q[32],q[41],q[50],q[47],q[56],q[1],q[65],q[10],q[74],q[7],q[19],q[16],q[25],q[34],q[43],q[40],q[49],q[58],q[3],q[67],q[12],q[76],q[9],q[73],q[18],q[27],q[36],q[33],q[45],q[42],q[51],q[60],q[5],q[69],q[2],q[66],q[11],q[75],q[20],q[29],q[38],q[35],q[44],q[53],q[63],q[59],q[71],q[4],q[68],q[13],q[77],q[22],q[31],q[28],q[37],q[46],q[55],q[0],q[52],q[64],q[61],q[6];
measure q[63] -> meas[0];
measure q[62] -> meas[1];
measure q[56] -> meas[2];
