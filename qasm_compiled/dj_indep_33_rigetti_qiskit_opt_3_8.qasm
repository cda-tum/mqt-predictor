OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg c[32];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(-pi) q[11];
rx(-pi/2) q[11];
cz q[10],q[11];
rx(pi/2) q[10];
rz(-pi/2) q[11];
rx(pi/2) q[11];
cz q[10],q[11];
rx(-pi/2) q[10];
rx(pi/2) q[11];
cz q[10],q[11];
rx(pi) q[10];
rz(-pi) q[12];
rx(-pi/2) q[12];
cz q[11],q[12];
rx(pi/2) q[11];
rz(-pi/2) q[12];
rx(pi/2) q[12];
cz q[11],q[12];
rx(-pi/2) q[11];
rz(pi/2) q[11];
rx(pi/2) q[12];
cz q[11],q[12];
rx(-pi/2) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi) q[12];
rz(pi/2) q[18];
rx(pi/2) q[18];
rz(pi/2) q[19];
rx(pi/2) q[19];
rz(pi/2) q[19];
rz(-pi/2) q[20];
rx(pi/2) q[20];
rz(pi/2) q[20];
rx(-pi) q[21];
cz q[20],q[21];
cz q[10],q[21];
rx(-pi) q[10];
rz(-pi/2) q[10];
cz q[10],q[11];
rx(pi/2) q[10];
rz(-pi/2) q[11];
rx(pi/2) q[11];
cz q[10],q[11];
rx(-pi/2) q[10];
rz(pi/2) q[10];
rx(pi/2) q[11];
cz q[10],q[11];
cz q[10],q[21];
rx(-pi/2) q[11];
cz q[12],q[11];
rx(pi/2) q[11];
rz(pi) q[11];
rx(pi/2) q[12];
rz(pi) q[12];
cz q[11],q[12];
rx(pi/2) q[11];
rz(pi) q[11];
rx(pi/2) q[12];
rz(pi/2) q[12];
cz q[12],q[11];
rx(pi/2) q[11];
rz(pi) q[11];
cz q[11],q[10];
rx(pi/2) q[10];
rz(pi) q[10];
rx(pi/2) q[11];
rz(pi) q[11];
cz q[10],q[11];
rx(pi/2) q[10];
rz(pi) q[10];
rx(pi/2) q[11];
rz(pi/2) q[11];
cz q[11],q[10];
rx(pi/2) q[10];
rz(pi) q[10];
cz q[10],q[21];
rx(pi/2) q[10];
rz(pi/2) q[10];
rx(-pi) q[20];
rz(-pi/2) q[20];
rx(pi/2) q[21];
rz(pi/2) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rx(pi/2) q[21];
cz q[20],q[21];
rx(-pi/2) q[20];
rz(pi/2) q[20];
rx(pi/2) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rz(-0.48569525) q[20];
cz q[19],q[20];
rx(pi/2) q[19];
rz(pi/2) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rx(pi/2) q[19];
cz q[18],q[19];
rx(-pi/2) q[18];
rx(pi/2) q[19];
cz q[18],q[19];
rx(-pi/2) q[19];
rx(-pi/2) q[20];
rz(pi/2) q[21];
rz(pi/2) q[25];
rx(pi/2) q[25];
rz(pi/2) q[26];
rx(pi/2) q[26];
rz(pi/2) q[27];
rx(pi/2) q[27];
rz(pi/2) q[27];
rz(pi/2) q[28];
rx(-pi/2) q[28];
rz(pi/2) q[48];
rx(pi/2) q[48];
rz(-pi) q[49];
rx(-pi/2) q[49];
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
rz(pi) q[49];
rz(pi/2) q[50];
rx(pi/2) q[50];
rz(-pi) q[55];
rx(-pi/2) q[55];
rz(pi/2) q[56];
rx(pi/2) q[56];
rx(-pi/2) q[57];
rz(pi/2) q[57];
rz(-pi/2) q[58];
rx(-pi/2) q[58];
rz(-pi) q[59];
rx(-pi/2) q[59];
rz(-pi) q[60];
rx(-pi/2) q[60];
rz(pi/2) q[61];
rx(pi/2) q[61];
rz(pi/2) q[61];
rz(-pi) q[62];
rx(-pi/2) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(-pi/2) q[62];
rx(pi/2) q[62];
cz q[49],q[62];
rx(-pi/2) q[49];
rz(3*pi/2) q[49];
rx(pi/2) q[62];
cz q[49],q[62];
rz(pi/2) q[62];
rx(-pi/2) q[63];
rz(pi/2) q[63];
cz q[20],q[63];
rx(-pi/2) q[20];
rz(-pi/2) q[20];
rx(pi/2) q[63];
cz q[20],q[63];
rx(-pi/2) q[20];
rx(pi/2) q[63];
rz(-1.0851011) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rx(pi/2) q[62];
cz q[49],q[62];
rx(-pi/2) q[49];
rz(pi/2) q[49];
rx(pi/2) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(pi) q[49];
cz q[49],q[48];
rx(pi/2) q[48];
rz(pi) q[48];
rx(pi/2) q[49];
rz(pi) q[49];
cz q[48],q[49];
rx(pi/2) q[48];
rx(pi/2) q[49];
rz(pi) q[49];
cz q[49],q[48];
cz q[48],q[55];
rx(pi/2) q[48];
rx(pi/2) q[49];
rz(-pi/2) q[55];
rx(pi/2) q[55];
cz q[48],q[55];
rx(-pi/2) q[48];
rx(pi/2) q[55];
cz q[48],q[55];
rx(pi) q[48];
rz(pi/2) q[55];
rx(-pi/2) q[62];
rz(pi) q[62];
rx(-pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-pi/2) q[56];
rz(-2.9203923) q[56];
rx(pi/2) q[63];
cz q[56],q[63];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi/2) q[19];
rx(-pi/2) q[63];
rz(pi/2) q[63];
rx(-pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(-pi/2) q[63];
rx(pi/2) q[63];
cz q[62],q[63];
rx(-pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
cz q[62],q[63];
rx(-pi/2) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(-pi/2) q[62];
rx(pi/2) q[62];
cz q[49],q[62];
rx(-pi/2) q[49];
rz(pi/2) q[49];
rx(pi/2) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(pi) q[49];
cz q[48],q[49];
rz(-pi/2) q[48];
rx(-pi/2) q[48];
rx(pi/2) q[49];
rz(pi) q[49];
cz q[49],q[48];
rx(pi/2) q[48];
rz(pi) q[48];
rx(pi/2) q[49];
cz q[48],q[49];
rx(pi/2) q[48];
rx(-pi/2) q[62];
rz(pi) q[62];
rx(-pi/2) q[63];
rz(-pi/2) q[63];
cz q[63],q[56];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rx(pi/2) q[63];
cz q[62],q[63];
rx(-pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
cz q[62],q[63];
rx(-pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(-pi/2) q[62];
rx(pi/2) q[62];
cz q[49],q[62];
rx(-pi/2) q[49];
rz(pi/2) q[49];
rx(pi/2) q[62];
cz q[49],q[62];
rx(-pi/2) q[49];
rz(pi/2) q[49];
rx(pi/2) q[49];
cz q[48],q[49];
rx(pi/2) q[48];
rz(-pi/2) q[49];
rx(pi/2) q[49];
cz q[48],q[49];
rx(-pi/2) q[48];
rx(pi/2) q[49];
cz q[48],q[49];
rx(-pi/2) q[62];
rz(-pi) q[63];
cz q[63],q[56];
rx(pi/2) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
rz(-pi) q[63];
cz q[63],q[62];
rx(-pi/2) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rx(pi/2) q[62];
cz q[49],q[62];
rx(-pi/2) q[49];
rz(pi/2) q[49];
rx(pi/2) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(pi/2) q[49];
rz(pi) q[62];
cz q[63],q[56];
rx(pi/2) q[63];
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
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi/2) q[62];
cz q[63],q[56];
rx(pi/2) q[56];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(-pi/2) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rx(pi/2) q[57];
rz(-1.7919966) q[57];
cz q[58],q[57];
rx(pi/2) q[63];
rz(pi/2) q[63];
rz(pi/2) q[64];
rx(pi/2) q[64];
rz(pi/2) q[65];
rx(pi/2) q[65];
rz(pi/2) q[65];
rz(-pi) q[66];
rx(-pi/2) q[66];
rz(-pi/2) q[67];
rx(pi/2) q[67];
rz(pi/2) q[67];
rz(pi/2) q[68];
rx(pi/2) q[68];
rz(-pi/2) q[69];
rx(pi/2) q[69];
rz(pi/2) q[69];
rz(-pi) q[70];
rx(-pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(-pi/2) q[70];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[70];
rz(-3.6227927) q[70];
cz q[69],q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi/2) q[57];
rx(-pi/2) q[69];
rz(pi/2) q[69];
cz q[68],q[69];
rx(pi/2) q[68];
rx(pi/2) q[69];
cz q[68],q[69];
rx(-pi/2) q[68];
rz(pi/2) q[68];
rx(pi/2) q[69];
cz q[68],q[69];
rx(pi/2) q[68];
rz(pi) q[68];
rx(-pi/2) q[70];
rx(-pi/2) q[71];
rz(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
rz(-pi) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi) q[70];
cz q[69],q[70];
rx(pi/2) q[69];
rz(-pi/2) q[70];
rx(pi/2) q[70];
cz q[69],q[70];
rx(-pi/2) q[69];
rz(pi/2) q[69];
rx(pi/2) q[70];
cz q[69],q[70];
rx(pi/2) q[69];
rz(pi) q[69];
rz(pi) q[70];
rx(pi/2) q[71];
rz(-1.0895963) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(-pi/2) q[28];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi/2) q[70];
rx(-pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rx(pi/2) q[71];
cz q[64],q[71];
rx(-pi/2) q[64];
rz(-pi/2) q[64];
rx(pi/2) q[71];
cz q[64],q[71];
cz q[65],q[64];
rx(pi/2) q[65];
rz(pi/2) q[65];
rx(-pi/2) q[71];
cz q[71],q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[26];
cz q[25],q[26];
rx(pi/2) q[25];
rz(-pi/2) q[26];
rx(pi/2) q[26];
cz q[25],q[26];
rx(-pi/2) q[25];
rx(pi/2) q[26];
cz q[25],q[26];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[27];
rz(pi/2) q[27];
cz q[27],q[64];
rx(-pi) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(pi) q[26];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
cz q[27],q[26];
rx(pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[26],q[27];
rx(pi/2) q[27];
rz(-pi) q[27];
cz q[27],q[64];
rx(pi/2) q[27];
rz(-pi/2) q[27];
cz q[64],q[65];
rx(pi/2) q[64];
rx(pi/2) q[65];
cz q[64],q[65];
rx(-pi/2) q[64];
rx(pi/2) q[65];
cz q[64],q[65];
rx(-pi/2) q[65];
rz(-pi) q[65];
rx(pi/2) q[71];
rz(pi/2) q[71];
rz(pi/2) q[72];
rx(pi/2) q[72];
rz(-pi/2) q[77];
rx(-pi/2) q[77];
rz(-pi/2) q[78];
rx(pi/2) q[78];
rz(pi/2) q[78];
cz q[78],q[65];
rx(-pi) q[78];
rz(-pi/2) q[78];
rz(-pi) q[79];
rx(-pi/2) q[79];
cz q[78],q[79];
rx(pi/2) q[78];
rz(-pi/2) q[79];
rx(pi/2) q[79];
cz q[78],q[79];
rx(-pi/2) q[78];
rx(pi/2) q[79];
cz q[78],q[79];
rx(pi) q[78];
cz q[78],q[65];
cz q[65],q[66];
rx(pi/2) q[65];
rz(-pi/2) q[66];
rx(pi/2) q[66];
cz q[65],q[66];
rx(-pi/2) q[65];
rx(pi/2) q[66];
cz q[65],q[66];
rx(pi) q[65];
rx(-pi/2) q[66];
cz q[77],q[66];
rx(-pi) q[78];
rx(-pi/2) q[79];
rz(pi/2) q[79];
rx(-pi/2) q[79];
cz q[72],q[79];
rx(pi/2) q[72];
rz(-pi/2) q[79];
rx(pi/2) q[79];
cz q[72],q[79];
rx(-pi/2) q[72];
rx(pi/2) q[79];
cz q[72],q[79];
rx(pi/2) q[79];
rz(pi) q[79];
cz q[79],q[78];
rz(-pi/2) q[78];
rx(-pi/2) q[78];
rx(pi/2) q[79];
rz(pi) q[79];
cz q[78],q[79];
rx(pi/2) q[78];
rz(pi) q[78];
rx(pi/2) q[79];
rz(pi/2) q[79];
cz q[79],q[78];
rx(pi/2) q[78];
rz(pi) q[78];
cz q[78],q[77];
rx(pi/2) q[77];
rz(pi) q[77];
rx(pi/2) q[78];
rz(pi) q[78];
cz q[77],q[78];
rx(pi/2) q[77];
rz(pi) q[77];
rx(pi/2) q[78];
rz(pi/2) q[78];
cz q[78],q[77];
rx(pi/2) q[77];
rz(-pi) q[77];
cz q[77],q[66];
cz q[67],q[66];
cz q[65],q[66];
rz(-3*pi/2) q[65];
rx(pi/2) q[65];
rz(-pi/2) q[65];
rx(pi/2) q[66];
rz(pi) q[66];
rx(-pi) q[67];
cz q[66],q[67];
rx(pi/2) q[66];
rz(pi) q[66];
rz(-pi/2) q[67];
rx(-pi/2) q[67];
cz q[67],q[66];
rx(pi/2) q[66];
rz(pi/2) q[66];
rx(pi/2) q[67];
rz(pi) q[67];
cz q[66],q[67];
rx(pi/2) q[67];
rz(pi) q[67];
cz q[67],q[68];
rx(pi/2) q[67];
rz(pi) q[67];
rx(pi/2) q[68];
rz(pi) q[68];
cz q[68],q[67];
rx(pi/2) q[67];
rz(pi/2) q[67];
rx(pi/2) q[68];
rz(pi) q[68];
cz q[67],q[68];
rx(pi/2) q[68];
rz(pi) q[68];
cz q[68],q[69];
rx(pi/2) q[68];
rz(pi) q[68];
rx(pi/2) q[69];
rz(pi) q[69];
cz q[69],q[68];
rx(pi/2) q[68];
rz(pi/2) q[68];
rx(pi/2) q[69];
rz(pi) q[69];
cz q[68],q[69];
rx(pi/2) q[69];
rz(pi) q[69];
cz q[69],q[58];
rx(pi/2) q[58];
rz(pi) q[58];
rx(pi/2) q[69];
rz(pi) q[69];
cz q[58],q[69];
rx(pi/2) q[58];
rx(pi/2) q[69];
rz(pi/2) q[69];
cz q[69],q[58];
cz q[58],q[59];
rx(pi/2) q[58];
rz(-pi/2) q[59];
rx(pi/2) q[59];
cz q[58],q[59];
rx(-pi/2) q[58];
rz(pi/2) q[58];
rx(pi/2) q[59];
cz q[58],q[59];
rx(pi/2) q[58];
rz(pi/2) q[58];
rx(-pi/2) q[59];
rz(pi) q[59];
cz q[59],q[60];
rx(pi/2) q[59];
rz(-pi/2) q[60];
rx(pi/2) q[60];
cz q[59],q[60];
rx(-pi/2) q[59];
rz(pi/2) q[59];
rx(pi/2) q[60];
cz q[59],q[60];
rx(pi/2) q[59];
rz(pi/2) q[59];
rx(-pi/2) q[60];
cz q[61],q[60];
rx(pi/2) q[61];
rz(pi/2) q[61];
cz q[50],q[61];
rx(pi/2) q[50];
rx(pi/2) q[61];
cz q[50],q[61];
rx(-pi/2) q[50];
rx(pi/2) q[61];
cz q[50],q[61];
rx(pi/2) q[61];
rz(-pi) q[61];
cz q[61],q[60];
rx(pi/2) q[60];
rz(pi/2) q[60];
rx(pi/2) q[61];
rz(-pi/2) q[61];
rx(pi/2) q[77];
rz(-pi/2) q[77];
barrier q[49],q[69],q[62],q[0],q[71],q[9],q[73],q[19],q[15],q[25],q[79],q[24],q[33],q[42],q[51],q[55],q[56],q[2],q[65],q[12],q[75],q[21],q[17],q[26],q[35],q[44],q[41],q[53],q[61],q[58],q[4],q[70],q[13],q[10],q[78],q[74],q[18],q[28],q[37],q[34],q[46],q[43],q[52],q[50],q[6],q[3],q[57],q[66],q[11],q[76],q[60],q[30],q[39],q[36],q[45],q[54],q[20],q[8],q[59],q[5],q[77],q[67],q[14],q[72],q[23],q[32],q[29],q[38],q[47],q[63],q[1],q[64],q[48],q[7],q[68],q[16],q[27],q[22],q[31],q[40];
measure q[21] -> c[0];
measure q[12] -> c[1];
measure q[11] -> c[2];
measure q[10] -> c[3];
measure q[18] -> c[4];
measure q[20] -> c[5];
measure q[55] -> c[6];
measure q[19] -> c[7];
measure q[48] -> c[8];
measure q[49] -> c[9];
measure q[62] -> c[10];
measure q[63] -> c[11];
measure q[56] -> c[12];
measure q[69] -> c[13];
measure q[67] -> c[14];
measure q[57] -> c[15];
measure q[68] -> c[16];
measure q[28] -> c[17];
measure q[70] -> c[18];
measure q[64] -> c[19];
measure q[71] -> c[20];
measure q[25] -> c[21];
measure q[26] -> c[22];
measure q[27] -> c[23];
measure q[72] -> c[24];
measure q[79] -> c[25];
measure q[78] -> c[26];
measure q[77] -> c[27];
measure q[66] -> c[28];
measure q[65] -> c[29];
measure q[50] -> c[30];
measure q[61] -> c[31];
