OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[58];
cx q[27],q[28];
cx q[28],q[27];
cx q[27],q[28];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[60];
rz(pi/2) q[95];
sx q[95];
rz(pi/2) q[95];
cx q[95],q[96];
cx q[96],q[97];
cx q[97],q[96];
cx q[96],q[97];
cx q[97],q[98];
cx q[98],q[99];
cx q[99],q[100];
cx q[100],q[99];
cx q[99],q[100];
cx q[100],q[110];
cx q[110],q[100];
cx q[100],q[110];
cx q[110],q[118];
cx q[118],q[110];
cx q[110],q[118];
cx q[118],q[119];
cx q[119],q[118];
cx q[118],q[119];
cx q[119],q[120];
cx q[120],q[121];
cx q[121],q[120];
cx q[120],q[121];
cx q[121],q[122];
cx q[122],q[121];
cx q[121],q[122];
cx q[111],q[122];
cx q[122],q[111];
cx q[111],q[122];
cx q[111],q[104];
cx q[104],q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[105],q[106];
cx q[106],q[105];
cx q[105],q[106];
cx q[93],q[106];
cx q[106],q[93];
cx q[93],q[106];
cx q[93],q[87];
cx q[87],q[88];
cx q[88],q[87];
cx q[87],q[88];
cx q[88],q[89];
cx q[89],q[88];
cx q[88],q[89];
cx q[89],q[74];
cx q[74],q[89];
cx q[89],q[74];
cx q[74],q[89];
cx q[89],q[88];
cx q[88],q[87];
cx q[87],q[86];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
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
cx q[118],q[110];
cx q[110],q[118];
cx q[100],q[110];
cx q[110],q[100];
cx q[100],q[110];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[101],q[102];
cx q[102],q[92];
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
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[68],q[67];
cx q[67],q[68];
cx q[68],q[67];
cx q[55],q[68];
cx q[68],q[55];
cx q[55],q[68];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
cx q[48],q[49];
cx q[49],q[48];
cx q[48],q[49];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[35],q[28];
cx q[28],q[27];
cx q[27],q[28];
cx q[28],q[27];
cx q[27],q[26];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[15],q[22];
cx q[22],q[15];
cx q[15],q[22];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[34];
cx q[34],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[46],q[45];
cx q[45],q[46];
cx q[45],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[65],q[66];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
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
cx q[100],q[99];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[99];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[53];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[41],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[44],q[43];
cx q[43],q[44];
cx q[43],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
cx q[39],q[38];
cx q[38],q[37];
cx q[37],q[52];
cx q[52],q[37];
cx q[37],q[52];
cx q[52],q[56];
cx q[56],q[52];
cx q[52],q[56];
cx q[56],q[57];
cx q[57],q[56];
cx q[56],q[57];
cx q[57],q[58];
cx q[71],q[58];
cx q[58],q[71];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[60],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[53],q[41];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
cx q[39],q[33];
cx q[33],q[20];
cx q[20],q[21];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[20];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[20];
cx q[20],q[33];
cx q[33],q[20];
cx q[20],q[33];
cx q[33],q[39];
cx q[39],q[33];
cx q[33],q[39];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
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
cx q[95],q[96];
cx q[96],q[95];
cx q[95],q[96];
cx q[94],q[95];
cx q[95],q[94];
cx q[94],q[95];
cx q[90],q[94];
cx q[94],q[90];
cx q[90],q[94];
cx q[90],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[77];
cx q[77],q[76];
cx q[76],q[77];
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
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[68],q[67];
cx q[67],q[68];
cx q[68],q[67];
cx q[55],q[68];
cx q[68],q[55];
cx q[55],q[68];
cx q[55],q[49];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
cx q[55],q[68];
cx q[68],q[67];
cx q[67],q[68];
cx q[68],q[67];
cx q[67],q[66];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[65],q[64];
cx q[64],q[63];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[62],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[80];
cx q[80],q[79];
cx q[79],q[91];
barrier q[114],q[7],q[53],q[16],q[79],q[26],q[88],q[34],q[99],q[46],q[107],q[0],q[64],q[9],q[85],q[18],q[72],q[27],q[91],q[36],q[61],q[56],q[2],q[120],q[73],q[11],q[77],q[19],q[84],q[29],q[106],q[50],q[59],q[4],q[123],q[65],q[13],q[76],q[24],q[82],q[37],q[116],q[60],q[6],q[125],q[70],q[21],q[78],q[23],q[87],q[43],q[109],q[66],q[118],q[62],q[8],q[81],q[17],q[80],q[38],q[35],q[48],q[122],q[52],q[1],q[111],q[54],q[10],q[89],q[31],q[96],q[39],q[93],q[67],q[113],q[71],q[3],q[121],q[101],q[20],q[90],q[44],q[105],q[51],q[115],q[58],q[124],q[69],q[25],q[94],q[47],q[119],q[41],q[108],q[42],q[117],q[97],q[126],q[63],q[83],q[28],q[110],q[57],q[92],q[45],q[100],q[68],q[102],q[12],q[75],q[22],q[86],q[30],q[95],q[40],q[103],q[55],q[112],q[5],q[14],q[49],q[15],q[74],q[32],q[98],q[33],q[104];
measure q[91] -> meas[0];
measure q[79] -> meas[1];
measure q[80] -> meas[2];
measure q[81] -> meas[3];
measure q[62] -> meas[4];
measure q[64] -> meas[5];
measure q[65] -> meas[6];
measure q[67] -> meas[7];
measure q[55] -> meas[8];
measure q[49] -> meas[9];
measure q[77] -> meas[10];
measure q[90] -> meas[11];
measure q[97] -> meas[12];
measure q[63] -> meas[13];
measure q[21] -> meas[14];
measure q[22] -> meas[15];
measure q[19] -> meas[16];
measure q[20] -> meas[17];
measure q[33] -> meas[18];
measure q[53] -> meas[19];
measure q[71] -> meas[20];
measure q[57] -> meas[21];
measure q[38] -> meas[22];
measure q[39] -> meas[23];
measure q[41] -> meas[24];
measure q[43] -> meas[25];
measure q[44] -> meas[26];
measure q[42] -> meas[27];
measure q[59] -> meas[28];
measure q[60] -> meas[29];
measure q[61] -> meas[30];
measure q[100] -> meas[31];
measure q[101] -> meas[32];
measure q[66] -> meas[33];
measure q[45] -> meas[34];
measure q[46] -> meas[35];
measure q[34] -> meas[36];
measure q[24] -> meas[37];
measure q[15] -> meas[38];
measure q[23] -> meas[39];
measure q[25] -> meas[40];
measure q[27] -> meas[41];
measure q[35] -> meas[42];
measure q[92] -> meas[43];
measure q[102] -> meas[44];
measure q[118] -> meas[45];
measure q[110] -> meas[46];
measure q[82] -> meas[47];
measure q[87] -> meas[48];
measure q[88] -> meas[49];
measure q[89] -> meas[50];
measure q[74] -> meas[51];
measure q[93] -> meas[52];
measure q[111] -> meas[53];
measure q[119] -> meas[54];
measure q[99] -> meas[55];
measure q[98] -> meas[56];
measure q[96] -> meas[57];
