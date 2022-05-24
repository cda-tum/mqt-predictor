OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[34];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[20],q[33];
cx q[20],q[21];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[25],q[26];
rz(pi/2) q[26];
sx q[26];
rz(pi/2) q[26];
cx q[26],q[27];
rz(pi/2) q[27];
sx q[27];
rz(pi/2) q[27];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[27];
rz(pi/2) q[26];
sx q[26];
rz(pi/2) q[26];
rz(pi/2) q[33];
sx q[33];
rz(pi/2) q[33];
cx q[33],q[39];
rz(pi/2) q[39];
sx q[39];
rz(pi/2) q[39];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
rz(pi/2) q[40];
sx q[40];
rz(pi/2) q[40];
rz(pi/2) q[42];
sx q[42];
rz(pi/2) q[42];
cx q[42],q[41];
rz(pi/2) q[41];
sx q[41];
rz(pi/2) q[41];
cx q[41],q[40];
rz(pi/2) q[40];
sx q[40];
rz(pi/2) q[40];
rz(pi/2) q[57];
sx q[57];
rz(pi/2) q[57];
rz(pi/2) q[59];
sx q[59];
rz(pi/2) q[59];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[65];
sx q[65];
rz(pi/2) q[65];
cx q[65],q[64];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[64],q[54];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[54],q[45];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[44],q[43];
rz(pi/2) q[43];
sx q[43];
rz(pi/2) q[43];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[34],q[43];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[41],q[53];
cx q[43],q[34];
cx q[34],q[43];
cx q[34],q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[25],q[26];
rz(pi/2) q[26];
sx q[26];
rz(pi/2) q[26];
cx q[53],q[41];
cx q[41],q[53];
cx q[65],q[66];
rz(pi/2) q[66];
sx q[66];
rz(pi/2) q[66];
rz(pi/2) q[71];
sx q[71];
rz(pi/2) q[71];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[59],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[71],q[77];
cx q[71],q[58];
rz(pi/2) q[58];
sx q[58];
rz(pi/2) q[58];
cx q[57],q[58];
cx q[58],q[57];
cx q[57],q[58];
rz(pi/2) q[57];
sx q[57];
rz(pi/2) q[57];
rz(pi/2) q[58];
sx q[58];
rz(pi/2) q[58];
cx q[59],q[58];
rz(pi/2) q[58];
sx q[58];
rz(pi/2) q[58];
cx q[58],q[57];
rz(pi/2) q[57];
sx q[57];
rz(pi/2) q[57];
rz(pi/2) q[77];
sx q[77];
rz(pi/2) q[77];
cx q[77],q[78];
rz(pi/2) q[78];
sx q[78];
rz(pi/2) q[78];
cx q[77],q[78];
cx q[78],q[77];
cx q[77],q[78];
cx q[71],q[77];
cx q[77],q[71];
cx q[71],q[77];
cx q[58],q[71];
cx q[71],q[58];
cx q[58],q[71];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
rz(pi/2) q[59];
sx q[59];
rz(pi/2) q[59];
cx q[60],q[59];
rz(pi/2) q[59];
sx q[59];
rz(pi/2) q[59];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[82];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(pi/2) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
rz(pi/2) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[85],q[86];
cx q[85],q[84];
rz(pi/2) q[84];
sx q[84];
rz(pi/2) q[84];
rz(pi/2) q[86];
sx q[86];
rz(pi/2) q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[83],q[92];
cx q[87],q[93];
rz(pi/2) q[93];
sx q[93];
rz(pi/2) q[93];
rz(pi/2) q[102];
sx q[102];
rz(pi/2) q[102];
cx q[102],q[103];
cx q[102],q[101];
rz(pi/2) q[101];
sx q[101];
rz(pi/2) q[101];
cx q[101],q[100];
rz(pi/2) q[100];
sx q[100];
rz(pi/2) q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[92];
rz(pi/2) q[103];
sx q[103];
rz(pi/2) q[103];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[103],q[104];
rz(pi/2) q[104];
sx q[104];
rz(pi/2) q[104];
cx q[104],q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[93],q[106];
cx q[106],q[93];
cx q[93],q[106];
rz(pi/2) q[106];
sx q[106];
rz(pi/2) q[106];
cx q[105],q[106];
rz(pi/2) q[106];
sx q[106];
rz(pi/2) q[106];
barrier q[52],q[62],q[6],q[125],q[70],q[15],q[79],q[23],q[88],q[54],q[118],q[65],q[8],q[53],q[17],q[72],q[27],q[90],q[47],q[111],q[56],q[1],q[120],q[64],q[10],q[74],q[19],q[82],q[39],q[105],q[49],q[113],q[57],q[3],q[122],q[67],q[12],q[76],q[33],q[97],q[34],q[93],q[51],q[115],q[61],q[5],q[124],q[69],q[35],q[99],q[45],q[108],q[41],q[117],q[81],q[126],q[77],q[28],q[92],q[37],q[100],q[46],q[110],q[55],q[0],q[119],q[63],q[20],q[73],q[30],q[94],q[40],q[103],q[48],q[112],q[71],q[121],q[14],q[59],q[24],q[86],q[32],q[96],q[42],q[104],q[50],q[114],q[7],q[16],q[80],q[25],q[89],q[43],q[98],q[60],q[107],q[116],q[9],q[66],q[18],q[84],q[26],q[91],q[36],q[102],q[44],q[109],q[2],q[83],q[11],q[75],q[22],q[85],q[29],q[106],q[38],q[101],q[58],q[4],q[123],q[68],q[13],q[78],q[21],q[87],q[31],q[95];
measure q[22] -> meas[0];
measure q[33] -> meas[1];
measure q[65] -> meas[2];
measure q[81] -> meas[3];
measure q[34] -> meas[4];
measure q[42] -> meas[5];
measure q[40] -> meas[6];
measure q[64] -> meas[7];
measure q[63] -> meas[8];
measure q[101] -> meas[9];
measure q[103] -> meas[10];
measure q[54] -> meas[11];
measure q[44] -> meas[12];
measure q[85] -> meas[13];
measure q[87] -> meas[14];
measure q[77] -> meas[15];
measure q[78] -> meas[16];
measure q[58] -> meas[17];
measure q[53] -> meas[18];
measure q[24] -> meas[19];
measure q[27] -> meas[20];
measure q[100] -> meas[21];
measure q[60] -> meas[22];
measure q[83] -> meas[23];
measure q[105] -> meas[24];
measure q[106] -> meas[25];
measure q[59] -> meas[26];
measure q[25] -> meas[27];
measure q[84] -> meas[28];
measure q[102] -> meas[29];
measure q[92] -> meas[30];
measure q[71] -> meas[31];
measure q[26] -> meas[32];
measure q[57] -> meas[33];
