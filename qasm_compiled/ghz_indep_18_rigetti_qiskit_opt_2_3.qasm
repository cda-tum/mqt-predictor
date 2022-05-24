OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg meas[18];
rz(-pi/2) q[17];
rx(-pi/2) q[17];
rz(-pi/2) q[18];
rx(-pi/2) q[18];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
rz(-pi/2) q[20];
rx(-pi/2) q[20];
rz(-pi/2) q[24];
rx(-pi/2) q[24];
rz(-pi/2) q[26];
rx(-pi/2) q[26];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
rz(-pi/2) q[29];
rx(-pi/2) q[29];
rz(-pi/2) q[30];
rx(-pi/2) q[30];
rz(pi/2) q[48];
rx(pi/2) q[48];
rz(pi/2) q[48];
rz(-pi/2) q[49];
rx(-pi/2) q[49];
cz q[48],q[49];
rx(pi/2) q[49];
rz(pi/2) q[49];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rz(-pi/2) q[59];
rx(-pi/2) q[59];
rz(-pi/2) q[60];
rx(-pi/2) q[60];
rz(-pi/2) q[61];
rx(-pi/2) q[61];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
cz q[49],q[62];
rx(pi/2) q[62];
rz(pi) q[62];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi/2) q[20];
cz q[20],q[19];
rx(pi/2) q[19];
rz(pi/2) q[19];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi/2) q[18];
cz q[18],q[17];
rx(pi/2) q[17];
rz(pi) q[17];
cz q[17],q[30];
rx(pi/2) q[17];
rz(pi) q[17];
rx(pi/2) q[30];
rz(pi) q[30];
cz q[30],q[17];
rx(pi/2) q[17];
rz(pi/2) q[17];
rx(pi/2) q[30];
rz(pi) q[30];
cz q[17],q[30];
rx(pi/2) q[30];
rz(pi/2) q[30];
cz q[30],q[29];
rx(pi/2) q[29];
rz(pi/2) q[29];
cz q[29],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rz(pi/2) q[65];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
cz q[65],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[64],q[65];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[65];
rz(pi/2) q[65];
cz q[71],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rx(pi/2) q[64];
rz(pi/2) q[64];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
cz q[57],q[70];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[58],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rz(-pi/2) q[58];
rx(-pi/2) q[58];
cz q[57],q[58];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[58];
rz(pi/2) q[58];
cz q[58],q[57];
cz q[58],q[59];
rx(pi/2) q[59];
rz(pi/2) q[59];
cz q[59],q[60];
rx(pi/2) q[60];
rz(pi) q[60];
cz q[60],q[61];
rx(pi/2) q[60];
rz(pi) q[60];
rx(pi/2) q[61];
rz(pi) q[61];
cz q[61],q[60];
rx(pi/2) q[60];
rz(pi/2) q[60];
rx(pi/2) q[61];
rz(pi) q[61];
cz q[60],q[61];
rx(pi/2) q[61];
rz(pi/2) q[61];
cz q[61],q[62];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi/2) q[62];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[71],q[28];
cz q[27],q[28];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[28],q[27];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[27],q[28];
cz q[27],q[26];
cz q[25],q[26];
rz(-pi/2) q[25];
rx(-pi/2) q[25];
rx(pi/2) q[26];
rz(pi) q[26];
cz q[26],q[25];
rx(pi/2) q[25];
rz(pi/2) q[25];
rx(pi/2) q[26];
rz(pi) q[26];
cz q[25],q[26];
cz q[25],q[24];
rx(pi/2) q[24];
rz(pi/2) q[24];
rx(pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[28];
rz(pi/2) q[28];
barrier q[75],q[20],q[30],q[25],q[35],q[44],q[41],q[53],q[50],q[59],q[4],q[68],q[13],q[10],q[77],q[74],q[19],q[64],q[37],q[34],q[46],q[43],q[52],q[60],q[6],q[3],q[70],q[67],q[12],q[76],q[21],q[17],q[39],q[36],q[45],q[54],q[63],q[8],q[61],q[5],q[72],q[69],q[14],q[78],q[23],q[32],q[29],q[38],q[47],q[71],q[1],q[58],q[62],q[7],q[27],q[16],q[26],q[22],q[31],q[40],q[49],q[56],q[55],q[0],q[65],q[9],q[73],q[18],q[15],q[28],q[79],q[24],q[33],q[42],q[51],q[48],q[57],q[2],q[66],q[11];
measure q[24] -> meas[0];
measure q[25] -> meas[1];
measure q[27] -> meas[2];
measure q[71] -> meas[3];
measure q[63] -> meas[4];
measure q[61] -> meas[5];
measure q[59] -> meas[6];
measure q[58] -> meas[7];
measure q[65] -> meas[8];
measure q[64] -> meas[9];
measure q[29] -> meas[10];
measure q[30] -> meas[11];
measure q[18] -> meas[12];
measure q[19] -> meas[13];
measure q[20] -> meas[14];
measure q[62] -> meas[15];
measure q[49] -> meas[16];
measure q[48] -> meas[17];
