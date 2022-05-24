OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[67];
x q[11];
x q[36];
rz(pi/2) q[69];
sx q[69];
rz(pi/2) q[69];
cx q[69],q[70];
cx q[70],q[74];
cx q[74],q[89];
cx q[89],q[74];
cx q[74],q[89];
cx q[89],q[88];
cx q[88],q[89];
cx q[89],q[88];
cx q[88],q[87];
cx q[87],q[88];
cx q[88],q[87];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[93],q[106];
cx q[106],q[93];
cx q[93],q[106];
cx q[106],q[105];
cx q[105],q[106];
cx q[106],q[105];
cx q[105],q[104];
cx q[103],q[104];
cx q[104],q[103];
cx q[103],q[104];
cx q[103],q[102];
cx q[102],q[92];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
cx q[72],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[60],q[53];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[41],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[48],q[49];
cx q[49],q[48];
cx q[48],q[49];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
rz(pi/2) q[49];
cx q[55],q[68];
cx q[68],q[67];
cx q[67],q[66];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[65],q[64];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[63],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[98],q[97];
cx q[97],q[98];
cx q[98],q[97];
cx q[97],q[96];
cx q[96],q[97];
cx q[97],q[96];
rz(-pi) q[96];
sx q[96];
rz(pi/2) q[96];
rz(pi/2) q[109];
cx q[96],q[109];
rz(pi/2) q[109];
sx q[96];
rz(-pi/2) q[96];
sx q[96];
cx q[96],q[109];
rz(-pi) q[109];
sx q[109];
rz(-pi/2) q[109];
rz(-pi) q[96];
sx q[96];
rz(-pi) q[96];
cx q[96],q[97];
cx q[97],q[98];
cx q[98],q[97];
cx q[97],q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[98],q[91];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
cx q[80],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[54],q[45];
cx q[45],q[44];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[34],q[43];
cx q[43],q[34];
cx q[34],q[43];
cx q[24],q[34];
cx q[34],q[24];
cx q[24],q[34];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[22],q[21];
cx q[21],q[20];
cx q[20],q[33];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[20];
cx q[33],q[39];
cx q[38],q[39];
cx q[39],q[38];
cx q[38],q[39];
cx q[37],q[38];
cx q[38],q[37];
cx q[37],q[38];
cx q[37],q[52];
cx q[52],q[37];
cx q[37],q[52];
cx q[52],q[56];
cx q[56],q[57];
cx q[57],q[56];
cx q[56],q[57];
cx q[57],q[58];
cx q[58],q[57];
cx q[57],q[58];
cx q[58],q[71];
cx q[71],q[58];
cx q[58],q[71];
cx q[71],q[77];
cx q[77],q[71];
cx q[71],q[77];
cx q[77],q[78];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[79],q[80];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
x q[81];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[93],q[106];
cx q[105],q[106];
cx q[106],q[105];
cx q[105],q[106];
cx q[104],q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[103],q[104];
cx q[104],q[103];
cx q[103],q[104];
cx q[103],q[102];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[92],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[45],q[46];
cx q[46],q[47];
cx q[47],q[35];
cx q[28],q[35];
cx q[35],q[28];
cx q[28],q[35];
cx q[29],q[28];
cx q[28],q[29];
cx q[29],q[28];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[17],q[30];
cx q[30],q[17];
cx q[17],q[30];
cx q[17],q[12];
rz(pi/2) q[12];
sx q[12];
rz(-pi) q[12];
cx q[11],q[12];
sx q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[12];
cx q[11],q[12];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[12];
sx q[12];
cx q[12],q[17];
cx q[17],q[12];
cx q[12],q[17];
cx q[17],q[30];
cx q[30],q[29];
cx q[28],q[29];
cx q[29],q[28];
cx q[28],q[29];
cx q[35],q[28];
cx q[28],q[35];
cx q[35],q[28];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[47],q[48];
rz(-pi) q[48];
sx q[48];
rz(pi/2) q[48];
cx q[48],q[49];
sx q[48];
rz(-pi/2) q[48];
sx q[48];
rz(pi/2) q[49];
cx q[48],q[49];
rz(-pi) q[48];
sx q[48];
rz(-pi) q[48];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[35],q[28];
cx q[28],q[27];
cx q[27],q[26];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[21],q[20];
cx q[20],q[33];
cx q[33],q[20];
cx q[20],q[33];
cx q[33],q[39];
cx q[39],q[33];
cx q[33],q[39];
cx q[39],q[38];
cx q[38],q[39];
cx q[39],q[38];
cx q[38],q[37];
cx q[37],q[38];
cx q[38],q[37];
cx q[37],q[52];
rz(-pi) q[49];
sx q[49];
rz(-pi/2) q[49];
cx q[52],q[37];
cx q[37],q[52];
cx q[52],q[56];
cx q[56],q[57];
cx q[57],q[56];
cx q[56],q[57];
cx q[57],q[58];
cx q[58],q[57];
cx q[57],q[58];
cx q[58],q[71];
cx q[71],q[58];
cx q[58],q[71];
cx q[77],q[71];
cx q[71],q[77];
cx q[77],q[71];
cx q[77],q[78];
cx q[78],q[77];
cx q[77],q[78];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[91],q[98];
cx q[98],q[99];
cx q[99],q[100];
cx q[100],q[99];
cx q[99],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[83];
cx q[83],q[82];
rz(pi/2) q[82];
sx q[82];
rz(-pi) q[82];
cx q[81],q[82];
sx q[81];
rz(-pi/2) q[81];
sx q[81];
rz(pi/2) q[82];
cx q[81],q[82];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[82];
sx q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
cx q[103],q[104];
cx q[104],q[103];
cx q[103],q[104];
cx q[104],q[111];
cx q[111],q[104];
cx q[104],q[111];
cx q[111],q[122];
cx q[122],q[121];
cx q[120],q[121];
cx q[121],q[120];
cx q[120],q[121];
cx q[119],q[120];
cx q[120],q[119];
cx q[119],q[120];
cx q[119],q[118];
cx q[118],q[110];
cx q[100],q[110];
cx q[110],q[100];
cx q[100],q[110];
cx q[100],q[99];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[99];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[91],q[79];
cx q[79],q[80];
cx q[80],q[79];
cx q[79],q[80];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[85],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[68];
cx q[68],q[67];
cx q[67],q[68];
cx q[55],q[68];
cx q[68],q[55];
cx q[55],q[68];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
cx q[49],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[51];
rz(pi/2) q[51];
sx q[51];
rz(-pi) q[51];
cx q[36],q[51];
sx q[36];
rz(-pi/2) q[36];
sx q[36];
rz(pi/2) q[51];
cx q[36],q[51];
rz(pi/2) q[36];
sx q[36];
rz(pi/2) q[51];
sx q[51];
cx q[51],q[50];
cx q[50],q[49];
cx q[49],q[50];
cx q[50],q[49];
cx q[55],q[49];
cx q[49],q[55];
cx q[55],q[49];
cx q[68],q[55];
cx q[55],q[68];
cx q[68],q[55];
cx q[68],q[67];
cx q[67],q[68];
cx q[68],q[67];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[41],q[40];
cx q[40],q[39];
barrier q[114],q[7],q[57],q[16],q[65],q[25],q[74],q[43],q[99],q[42],q[107],q[0],q[63],q[9],q[84],q[18],q[80],q[27],q[85],q[51],q[91],q[98],q[2],q[119],q[73],q[17],q[75],q[19],q[45],q[29],q[86],q[36],q[59],q[4],q[123],q[68],q[13],q[58],q[21],q[93],q[38],q[116],q[109],q[6],q[125],q[70],q[15],q[77],q[34],q[89],q[23],q[96],q[66],q[118],q[62],q[8],q[72],q[30],q[111],q[33],q[110],q[46],q[104],q[71],q[1],q[121],q[50],q[10],q[106],q[31],q[95],q[40],q[103],q[49],q[113],q[56],q[3],q[122],q[67],q[20],q[97],q[55],q[87],q[64],q[115],q[41],q[124],q[69],q[26],q[90],q[12],q[101],q[44],q[108],q[53],q[117],q[61],q[126],q[52],q[82],q[28],q[79],q[39],q[92],q[54],q[100],q[35],q[120],q[11],q[76],q[22],q[81],q[48],q[94],q[37],q[105],q[47],q[112],q[5],q[14],q[83],q[24],q[88],q[32],q[78],q[60],q[102];
measure q[39] -> meas[0];
measure q[40] -> meas[1];
measure q[41] -> meas[2];
measure q[62] -> meas[3];
measure q[64] -> meas[4];
measure q[51] -> meas[5];
measure q[36] -> meas[6];
measure q[50] -> meas[7];
measure q[85] -> meas[8];
measure q[91] -> meas[9];
measure q[100] -> meas[10];
measure q[118] -> meas[11];
measure q[119] -> meas[12];
measure q[122] -> meas[13];
measure q[111] -> meas[14];
measure q[80] -> meas[15];
measure q[81] -> meas[16];
measure q[82] -> meas[17];
measure q[101] -> meas[18];
measure q[99] -> meas[19];
measure q[98] -> meas[20];
measure q[52] -> meas[21];
measure q[21] -> meas[22];
measure q[24] -> meas[23];
measure q[25] -> meas[24];
measure q[26] -> meas[25];
measure q[27] -> meas[26];
measure q[28] -> meas[27];
measure q[35] -> meas[28];
measure q[49] -> meas[29];
measure q[48] -> meas[30];
measure q[30] -> meas[31];
measure q[17] -> meas[32];
measure q[11] -> meas[33];
measure q[12] -> meas[34];
measure q[47] -> meas[35];
measure q[46] -> meas[36];
measure q[45] -> meas[37];
measure q[92] -> meas[38];
measure q[102] -> meas[39];
measure q[93] -> meas[40];
measure q[83] -> meas[41];
measure q[71] -> meas[42];
measure q[37] -> meas[43];
measure q[20] -> meas[44];
measure q[19] -> meas[45];
measure q[22] -> meas[46];
measure q[23] -> meas[47];
measure q[54] -> meas[48];
measure q[65] -> meas[49];
measure q[78] -> meas[50];
measure q[96] -> meas[51];
measure q[109] -> meas[52];
measure q[63] -> meas[53];
measure q[73] -> meas[54];
measure q[67] -> meas[55];
measure q[68] -> meas[56];
measure q[55] -> meas[57];
measure q[53] -> meas[58];
measure q[61] -> meas[59];
measure q[72] -> meas[60];
measure q[79] -> meas[61];
measure q[110] -> meas[62];
measure q[103] -> meas[63];
measure q[106] -> meas[64];
measure q[70] -> meas[65];
measure q[69] -> meas[66];
