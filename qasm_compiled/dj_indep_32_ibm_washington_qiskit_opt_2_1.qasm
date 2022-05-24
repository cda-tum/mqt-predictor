OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg c[31];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
rz(pi/2) q[33];
sx q[33];
rz(pi/2) q[33];
rz(-pi/2) q[34];
sx q[34];
rz(pi/2) q[34];
rz(-pi/2) q[38];
sx q[38];
rz(pi/2) q[38];
rz(pi/2) q[39];
sx q[39];
rz(pi/2) q[39];
rz(-pi/2) q[40];
sx q[40];
rz(-pi) q[40];
rz(-pi/2) q[41];
sx q[41];
rz(-pi) q[41];
rz(-pi/2) q[42];
sx q[42];
rz(-pi) q[42];
rz(-pi/2) q[43];
sx q[43];
rz(pi/2) q[43];
rz(pi/2) q[53];
sx q[53];
rz(pi) q[53];
rz(pi/2) q[54];
sx q[54];
rz(pi) q[54];
rz(pi/2) q[59];
sx q[59];
rz(pi) q[59];
rz(-pi/2) q[60];
sx q[60];
rz(-pi) q[60];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[63];
sx q[63];
rz(pi) q[63];
rz(-pi/2) q[65];
sx q[65];
rz(-pi) q[65];
rz(-3*pi/2) q[66];
sx q[66];
rz(-pi/2) q[66];
rz(-pi/2) q[67];
sx q[67];
rz(-pi) q[67];
cx q[67],q[66];
cx q[65],q[66];
sx q[65];
rz(-pi/2) q[65];
sx q[67];
rz(-pi/2) q[67];
rz(pi/2) q[72];
sx q[72];
rz(pi) q[72];
rz(pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
rz(-pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
rz(pi/2) q[84];
sx q[84];
rz(pi/2) q[84];
rz(-pi/2) q[85];
sx q[85];
rz(pi/2) q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[85],q[73];
cx q[73],q[85];
cx q[73],q[66];
rz(pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[66];
rz(pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
rz(-pi/2) q[86];
sx q[86];
rz(pi/2) q[86];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[73],q[66];
rz(-3*pi/2) q[73];
sx q[73];
rz(-pi/2) q[73];
rz(pi/2) q[87];
sx q[87];
rz(pi/2) q[87];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[73],q[66];
rz(pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[73],q[66];
rz(pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
rz(pi/2) q[102];
sx q[102];
rz(pi/2) q[102];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[73],q[85];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[85],q[73];
cx q[73],q[85];
cx q[73],q[66];
rz(pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[66];
rz(-3*pi/2) q[73];
sx q[73];
rz(-pi/2) q[73];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[102],q[92];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[73],q[66];
rz(-3*pi/2) q[73];
sx q[73];
rz(-pi/2) q[73];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[66];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
cx q[54],q[64];
sx q[54];
rz(pi/2) q[54];
sx q[63];
rz(pi/2) q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[72],q[62];
cx q[63],q[62];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
sx q[72];
rz(pi/2) q[72];
cx q[72],q[81];
rz(pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
cx q[81],q[72];
cx q[72],q[81];
cx q[72],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
sx q[60];
rz(-pi/2) q[60];
cx q[62],q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[59],q[60];
cx q[53],q[60];
sx q[53];
rz(pi/2) q[53];
cx q[53],q[60];
sx q[59];
rz(pi/2) q[59];
cx q[60],q[53];
cx q[53],q[60];
cx q[41],q[53];
sx q[41];
rz(-pi/2) q[41];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[42],q[41];
cx q[40],q[41];
sx q[40];
rz(-pi/2) q[40];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
cx q[38],q[39];
cx q[39],q[38];
cx q[38],q[39];
cx q[40],q[41];
rz(pi/2) q[40];
sx q[40];
rz(pi/2) q[40];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
cx q[33],q[39];
cx q[39],q[33];
cx q[33],q[39];
cx q[40],q[41];
rz(-3*pi/2) q[40];
sx q[40];
rz(-pi/2) q[40];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
sx q[42];
rz(-pi/2) q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[34],q[43];
cx q[42],q[41];
rz(-3*pi/2) q[42];
sx q[42];
rz(-pi/2) q[42];
cx q[43],q[34];
cx q[34],q[43];
cx q[24],q[34];
cx q[34],q[24];
cx q[24],q[34];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[34],q[43];
cx q[42],q[41];
cx q[40],q[41];
rz(pi/2) q[40];
sx q[40];
rz(pi/2) q[40];
rz(-3*pi/2) q[42];
sx q[42];
rz(-pi/2) q[42];
cx q[43],q[34];
cx q[34],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[41];
rz(pi/2) q[42];
sx q[42];
rz(pi/2) q[42];
cx q[53],q[41];
cx q[41],q[53];
cx q[53],q[41];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[110];
cx q[110],q[100];
cx q[100],q[110];
rz(-pi/2) q[115];
sx q[115];
rz(pi/2) q[115];
cx q[115],q[116];
cx q[116],q[115];
cx q[115],q[116];
rz(pi/2) q[117];
sx q[117];
rz(pi) q[117];
cx q[110],q[118];
cx q[118],q[110];
cx q[110],q[118];
cx q[117],q[118];
sx q[117];
rz(pi/2) q[117];
cx q[118],q[117];
cx q[117],q[118];
cx q[118],q[117];
cx q[116],q[117];
rz(-3*pi/2) q[116];
sx q[116];
rz(-pi/2) q[116];
barrier q[28],q[83],q[37],q[102],q[46],q[100],q[55],q[0],q[119],q[65],q[21],q[85],q[30],q[94],q[33],q[103],q[48],q[112],q[57],q[121],q[14],q[78],q[23],q[92],q[32],q[96],q[41],q[105],q[50],q[114],q[7],q[71],q[16],q[80],q[25],q[89],q[43],q[98],q[34],q[107],q[9],q[73],q[18],q[84],q[27],q[91],q[36],q[101],q[45],q[109],q[2],q[117],q[11],q[75],q[20],q[87],q[29],q[93],q[39],q[82],q[59],q[4],q[123],q[68],q[13],q[77],q[22],q[81],q[31],q[95],q[52],q[115],q[61],q[6],q[125],q[70],q[15],q[79],q[42],q[88],q[54],q[110],q[64],q[8],q[72],q[17],q[62],q[90],q[47],q[111],q[56],q[1],q[120],q[66],q[10],q[74],q[19],q[86],q[38],q[104],q[49],q[113],q[58],q[3],q[122],q[67],q[12],q[76],q[40],q[97],q[24],q[106],q[51],q[116],q[60],q[5],q[124],q[69],q[26],q[35],q[99],q[44],q[108],q[53],q[118],q[63],q[126];
measure q[67] -> c[0];
measure q[66] -> c[1];
measure q[87] -> c[2];
measure q[86] -> c[3];
measure q[81] -> c[4];
measure q[92] -> c[5];
measure q[83] -> c[6];
measure q[82] -> c[7];
measure q[84] -> c[8];
measure q[85] -> c[9];
measure q[73] -> c[10];
measure q[64] -> c[11];
measure q[54] -> c[12];
measure q[72] -> c[13];
measure q[63] -> c[14];
measure q[62] -> c[15];
measure q[60] -> c[16];
measure q[61] -> c[17];
measure q[59] -> c[18];
measure q[53] -> c[19];
measure q[41] -> c[20];
measure q[24] -> c[21];
measure q[38] -> c[22];
measure q[33] -> c[23];
measure q[39] -> c[24];
measure q[34] -> c[25];
measure q[43] -> c[26];
measure q[40] -> c[27];
measure q[42] -> c[28];
measure q[118] -> c[29];
measure q[116] -> c[30];
