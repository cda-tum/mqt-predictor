OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg meas[7];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[20];
rx(pi/2) q[20];
rz(pi/2) q[21];
rx(pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[48];
rx(pi/2) q[48];
rz(-pi) q[49];
rx(-pi/2) q[49];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
rz(-pi) q[63];
rx(-pi/2) q[63];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
rz(pi/2) q[71];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi/2) q[70];
cz q[70],q[57];
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
cz q[56],q[63];
rx(pi/2) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
rz(-pi/2) q[63];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(-pi/2) q[63];
rz(pi/2) q[63];
rx(-pi/2) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(-pi/2) q[63];
rx(pi/2) q[63];
cz q[20],q[63];
rx(-pi/2) q[20];
rx(pi/2) q[63];
cz q[20],q[63];
cz q[20],q[21];
rx(pi/2) q[21];
rz(pi/2) q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rx(pi/2) q[21];
cz q[10],q[21];
rx(-pi/2) q[10];
rz(pi/2) q[10];
rx(pi/2) q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rz(pi) q[10];
cz q[11],q[10];
rx(pi/2) q[10];
rz(pi) q[10];
rz(-pi/2) q[11];
rx(-pi/2) q[11];
cz q[10],q[11];
rx(pi/2) q[10];
rz(pi) q[10];
rx(pi/2) q[11];
rz(pi/2) q[11];
cz q[11],q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
cz q[11],q[48];
rz(pi/2) q[21];
cz q[48],q[49];
rx(pi/2) q[48];
rz(-pi/2) q[49];
rx(pi/2) q[49];
cz q[48],q[49];
rx(-pi/2) q[48];
rz(pi/2) q[48];
rx(pi/2) q[49];
cz q[48],q[49];
rx(pi/2) q[48];
rz(pi/2) q[48];
rz(pi/2) q[49];
cz q[49],q[62];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(-pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[63];
rz(pi/2) q[63];
barrier q[75],q[63],q[17],q[26],q[35],q[44],q[41],q[53],q[50],q[59],q[4],q[68],q[13],q[21],q[77],q[74],q[19],q[28],q[37],q[34],q[46],q[43],q[52],q[61],q[6],q[3],q[70],q[67],q[12],q[76],q[11],q[30],q[39],q[36],q[45],q[54],q[56],q[8],q[60],q[5],q[72],q[69],q[14],q[78],q[23],q[32],q[29],q[38],q[47],q[57],q[1],q[65],q[62],q[7],q[71],q[16],q[25],q[22],q[31],q[40],q[48],q[58],q[55],q[0],q[64],q[9],q[73],q[18],q[15],q[27],q[79],q[24],q[33],q[42],q[51],q[49],q[20],q[2],q[66],q[10];
measure q[63] -> meas[0];
measure q[62] -> meas[1];
measure q[49] -> meas[2];
measure q[11] -> meas[3];
measure q[20] -> meas[4];
measure q[70] -> meas[5];
measure q[71] -> meas[6];
