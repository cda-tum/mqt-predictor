OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg meas[15];
rz(-pi/2) q[18];
rx(-pi/2) q[18];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
rz(-pi/2) q[20];
rx(-pi/2) q[20];
cz q[20],q[21];
rx(-2.8882322) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rz(pi/2) q[20];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
rz(pi/2) q[28];
rx(pi/2) q[28];
rz(pi/2) q[28];
rz(-pi/2) q[29];
rx(-pi/2) q[29];
cz q[29],q[30];
rx(-2.8882322) q[30];
cz q[29],q[30];
rx(pi/2) q[29];
rz(pi/2) q[29];
cz q[18],q[29];
rx(-2.8882322) q[29];
cz q[18],q[29];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[18],q[19];
rx(pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[29];
rz(pi/2) q[29];
rx(-2.9410554) q[29];
rz(-pi/2) q[31];
rx(-pi/2) q[31];
cz q[31],q[30];
rx(-2.8882322) q[30];
cz q[31],q[30];
rz(1.7713336) q[30];
cz q[29],q[30];
rx(-1.4840794) q[30];
cz q[29],q[30];
rz(-pi/2) q[29];
rx(-pi/2) q[29];
cz q[18],q[29];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[18],q[29];
rx(pi/2) q[18];
rz(pi/2) q[18];
rx(pi/2) q[29];
rz(pi/2) q[29];
rx(pi/2) q[31];
rz(pi) q[31];
cz q[24],q[31];
rz(-pi/2) q[24];
rx(-pi/2) q[24];
rx(pi/2) q[31];
rz(pi) q[31];
cz q[31],q[24];
rx(pi/2) q[24];
rz(pi) q[24];
rx(pi/2) q[31];
rz(pi) q[31];
cz q[24],q[31];
rx(pi/2) q[24];
rz(pi) q[24];
cz q[25],q[24];
rx(pi/2) q[24];
rz(pi) q[24];
rz(-pi/2) q[25];
rx(-pi/2) q[25];
cz q[24],q[25];
rx(pi/2) q[24];
rz(pi) q[24];
rx(pi/2) q[25];
rz(pi/2) q[25];
cz q[25],q[24];
rx(pi/2) q[24];
rz(pi) q[24];
cz q[25],q[26];
rx(-2.8882322) q[26];
cz q[25],q[26];
rx(8.0545189) q[25];
cz q[27],q[26];
rx(-2.8882322) q[26];
cz q[27],q[26];
rz(1.7713336) q[26];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[31];
rz(pi) q[31];
cz q[31],q[30];
rz(-pi/2) q[30];
rx(-pi/2) q[30];
rx(pi/2) q[31];
rz(pi) q[31];
cz q[30],q[31];
rx(pi/2) q[30];
rz(pi) q[30];
rx(pi/2) q[31];
rz(pi) q[31];
cz q[31],q[30];
rx(pi/2) q[30];
rz(pi/2) q[30];
rx(pi/2) q[31];
rz(pi) q[31];
cz q[24],q[31];
rx(pi/2) q[24];
rz(pi) q[24];
rx(pi/2) q[31];
rz(pi) q[31];
cz q[31],q[24];
rx(pi/2) q[24];
rz(pi) q[24];
rx(pi/2) q[31];
rz(pi) q[31];
cz q[24],q[31];
rx(pi/2) q[24];
rz(pi/2) q[24];
cz q[25],q[24];
rx(-1.4840794) q[24];
cz q[25],q[24];
rx(pi/2) q[24];
rz(pi/2) q[24];
rx(-0.64046853) q[24];
cz q[25],q[26];
rx(-1.4840794) q[26];
cz q[25],q[26];
rx(10.355106) q[25];
rx(pi/2) q[31];
rz(pi/2) q[31];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
cz q[62],q[61];
rx(-2.8882322) q[61];
cz q[62],q[61];
rz(-pi/2) q[61];
rx(-pi/2) q[61];
rx(pi/2) q[62];
rz(pi/2) q[62];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[62];
rx(-2.8882322) q[62];
cz q[63],q[62];
rz(3.3421299) q[62];
cz q[61],q[62];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[61];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(pi/2) q[62];
cz q[61],q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[63],q[62];
rx(-2.8882322) q[62];
cz q[63],q[62];
rz(1.7713336) q[62];
cz q[61],q[62];
rx(-1.4840794) q[62];
cz q[61],q[62];
rx(pi/2) q[61];
rz(pi) q[61];
cz q[63],q[20];
rx(-2.8882322) q[20];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi/2) q[20];
rx(-2.9410554) q[20];
cz q[20],q[21];
rz(-pi/2) q[20];
rx(-pi/2) q[20];
rz(-pi/2) q[21];
rx(-pi/2) q[21];
cz q[21],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rz(pi/2) q[20];
cz q[19],q[20];
rx(-2.8882322) q[20];
cz q[19],q[20];
rx(8.0545189) q[19];
cz q[19],q[18];
rx(-1.4840794) q[18];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi/2) q[18];
rx(-0.64046853) q[18];
rz(1.7713336) q[20];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[21],q[20];
rx(-1.4840794) q[20];
cz q[21],q[20];
cz q[19],q[20];
rx(-1.4840794) q[20];
cz q[19],q[20];
rx(10.355106) q[19];
rx(pi/2) q[20];
rz(pi/2) q[20];
rx(-0.64046853) q[20];
rx(pi/2) q[21];
rz(pi/2) q[21];
rx(8.0545189) q[63];
cz q[63],q[62];
rx(-1.4840794) q[62];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(-0.64046853) q[62];
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
rz(pi) q[62];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[20],q[63];
rz(-pi/2) q[20];
rx(-pi/2) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi/2) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
cz q[20],q[21];
rx(-1.4840794) q[21];
cz q[20],q[21];
rx(10.355106) q[20];
rx(pi/2) q[21];
rz(pi/2) q[21];
rx(-0.64046853) q[21];
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
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[57],q[70];
rx(-2.8882322) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[56],q[57];
rx(-2.8882322) q[57];
cz q[56],q[57];
rx(8.0545189) q[56];
cz q[56],q[63];
rx(pi/2) q[57];
rz(pi/2) q[57];
rx(-2.9410554) q[57];
rx(-1.4840794) q[63];
cz q[56],q[63];
rx(pi/2) q[63];
rz(pi/2) q[63];
rx(-0.64046853) q[63];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[70],q[71];
cz q[28],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
rx(-2.8882322) q[71];
cz q[28],q[71];
cz q[28],q[27];
rx(-2.8882322) q[27];
cz q[28],q[27];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(-2.9410554) q[27];
cz q[27],q[26];
rx(-1.4840794) q[26];
cz q[27],q[26];
rx(pi/2) q[26];
rz(pi/2) q[26];
rx(-0.64046853) q[26];
rz(pi/2) q[27];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(8.0545189) q[28];
rx(pi/2) q[71];
rz(pi/2) q[71];
rx(-2.9410554) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
cz q[57],q[70];
rx(-1.4840794) q[70];
cz q[57],q[70];
rz(pi/2) q[57];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[56],q[57];
rx(-1.4840794) q[57];
cz q[56],q[57];
rx(10.355106) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
rx(-0.64046853) q[57];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
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
rz(pi/2) q[71];
cz q[70],q[71];
cz q[28],q[71];
rx(-1.4840794) q[71];
cz q[28],q[71];
cz q[28],q[27];
rx(-1.4840794) q[27];
cz q[28],q[27];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(-0.64046853) q[27];
rx(10.355106) q[28];
rx(pi/2) q[71];
rz(pi/2) q[71];
rx(-0.64046853) q[71];
barrier q[41],q[50],q[59],q[4],q[1],q[68],q[65],q[10],q[74],q[29],q[28],q[31],q[37],q[34],q[43],q[52],q[61],q[58],q[3],q[67],q[12],q[76],q[62],q[19],q[24],q[27],q[36],q[45],q[54],q[51],q[60],q[5],q[69],q[14],q[78],q[23],q[21],q[18],q[38],q[47],q[44],q[20],q[53],q[63],q[7],q[70],q[16],q[13],q[77],q[22],q[25],q[40],q[49],q[46],q[55],q[0],q[64],q[9],q[73],q[6],q[71],q[15],q[79],q[30],q[33],q[42],q[39],q[48],q[57],q[2],q[66],q[11],q[56],q[75],q[8],q[72],q[17],q[26],q[35],q[32];
measure q[56] -> meas[0];
measure q[28] -> meas[1];
measure q[20] -> meas[2];
measure q[57] -> meas[3];
measure q[19] -> meas[4];
measure q[27] -> meas[5];
measure q[25] -> meas[6];
measure q[26] -> meas[7];
measure q[21] -> meas[8];
measure q[71] -> meas[9];
measure q[62] -> meas[10];
measure q[63] -> meas[11];
measure q[18] -> meas[12];
measure q[61] -> meas[13];
measure q[24] -> meas[14];
