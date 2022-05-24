OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[42];
x q[15];
rz(pi/2) q[26];
x q[28];
rz(pi/2) q[119];
sx q[119];
rz(pi/2) q[119];
cx q[119],q[118];
cx q[118],q[110];
cx q[100],q[110];
cx q[110],q[100];
cx q[100],q[110];
rz(pi/2) q[110];
cx q[99],q[100];
cx q[100],q[99];
cx q[99],q[100];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[99];
cx q[98],q[91];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[77],q[78];
cx q[78],q[77];
cx q[77],q[78];
cx q[71],q[77];
cx q[77],q[71];
cx q[71],q[77];
cx q[58],q[71];
cx q[71],q[58];
cx q[58],q[71];
cx q[58],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
cx q[33],q[39];
cx q[39],q[33];
cx q[33],q[39];
cx q[20],q[33];
cx q[33],q[20];
cx q[20],q[33];
cx q[20],q[21];
cx q[21],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(-pi) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[26];
sx q[25];
rz(-pi/2) q[25];
sx q[25];
rz(pi/2) q[26];
cx q[25],q[26];
rz(-pi) q[25];
sx q[25];
rz(-pi) q[25];
cx q[25],q[24];
cx q[24],q[34];
rz(-pi) q[26];
sx q[26];
rz(-pi/2) q[26];
cx q[34],q[24];
cx q[24],q[34];
cx q[34],q[43];
cx q[43],q[34];
cx q[34],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[63];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[61],q[60];
cx q[60],q[59];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[71],q[58];
cx q[58],q[71];
cx q[71],q[58];
cx q[71],q[77];
cx q[77],q[71];
cx q[71],q[77];
cx q[77],q[76];
cx q[76],q[77];
cx q[77],q[76];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[75],q[90];
cx q[90],q[94];
cx q[94],q[90];
cx q[90],q[94];
cx q[94],q[95];
cx q[95],q[94];
cx q[94],q[95];
cx q[95],q[96];
cx q[96],q[95];
cx q[95],q[96];
cx q[96],q[97];
cx q[97],q[96];
cx q[96],q[97];
cx q[97],q[98];
cx q[98],q[97];
cx q[97],q[98];
cx q[98],q[99];
cx q[99],q[100];
cx q[100],q[99];
cx q[99],q[100];
rz(-pi) q[100];
sx q[100];
rz(pi/2) q[100];
cx q[100],q[110];
sx q[100];
rz(-pi/2) q[100];
sx q[100];
rz(pi/2) q[110];
cx q[100],q[110];
rz(-pi) q[100];
sx q[100];
rz(-pi) q[100];
cx q[100],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
rz(-pi) q[110];
sx q[110];
rz(-pi/2) q[110];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[85],q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[93],q[106];
cx q[106],q[105];
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
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[82],q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
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
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[47],q[35];
rz(pi/2) q[35];
sx q[35];
rz(-pi) q[35];
cx q[28],q[35];
sx q[28];
rz(-pi/2) q[28];
sx q[28];
rz(pi/2) q[35];
cx q[28],q[35];
rz(pi/2) q[28];
sx q[28];
rz(pi/2) q[35];
sx q[35];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[48],q[49];
cx q[49],q[48];
cx q[48],q[49];
cx q[49],q[55];
cx q[55],q[68];
cx q[68],q[55];
cx q[55],q[68];
cx q[68],q[67];
cx q[67],q[68];
cx q[68],q[67];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[80];
cx q[79],q[80];
cx q[80],q[79];
cx q[79],q[80];
cx q[79],q[91];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[98],q[99];
cx q[99],q[100];
cx q[100],q[99];
cx q[99],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
cx q[103],q[104];
cx q[104],q[103];
cx q[103],q[104];
cx q[104],q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[105],q[106];
cx q[106],q[105];
cx q[105],q[106];
cx q[106],q[107];
cx q[107],q[106];
cx q[106],q[107];
cx q[107],q[108];
cx q[108],q[112];
cx q[112],q[108];
cx q[108],q[112];
cx q[112],q[126];
cx q[126],q[112];
cx q[112],q[126];
cx q[126],q[125];
cx q[125],q[126];
cx q[126],q[125];
cx q[125],q[124];
cx q[124],q[125];
cx q[125],q[124];
cx q[124],q[123];
cx q[123],q[122];
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
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
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
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[64],q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[45],q[44];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[34],q[43];
cx q[43],q[34];
cx q[34],q[43];
cx q[34],q[24];
cx q[24],q[23];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[23];
rz(pi/2) q[22];
sx q[22];
rz(-pi) q[22];
cx q[15],q[22];
sx q[15];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[22];
cx q[15],q[22];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[22];
sx q[22];
cx q[22],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[34];
cx q[34],q[24];
cx q[24],q[34];
cx q[34],q[43];
cx q[43],q[34];
cx q[34],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[46];
cx q[46],q[47];
barrier q[114],q[7],q[71],q[16],q[79],q[63],q[89],q[23],q[110],q[34],q[105],q[0],q[65],q[9],q[64],q[18],q[81],q[27],q[59],q[36],q[99],q[57],q[2],q[121],q[67],q[11],q[76],q[33],q[83],q[29],q[93],q[50],q[20],q[4],q[123],q[55],q[13],q[78],q[26],q[106],q[52],q[116],q[72],q[6],q[126],q[70],q[22],q[98],q[15],q[88],q[24],q[109],q[44],q[118],q[61],q[8],q[82],q[17],q[35],q[38],q[100],q[46],q[122],q[56],q[1],q[120],q[66],q[10],q[74],q[31],q[94],q[41],q[73],q[48],q[113],q[58],q[3],q[111],q[68],q[39],q[96],q[42],q[104],q[51],q[115],q[75],q[125],q[69],q[25],q[91],q[28],q[107],q[43],q[124],q[60],q[117],q[62],q[112],q[19],q[101],q[49],q[92],q[37],q[85],q[54],q[97],q[84],q[119],q[12],q[77],q[21],q[86],q[30],q[90],q[40],q[103],q[47],q[108],q[5],q[14],q[80],q[45],q[87],q[32],q[95],q[53],q[102];
measure q[47] -> meas[0];
measure q[46] -> meas[1];
measure q[45] -> meas[2];
measure q[22] -> meas[3];
measure q[15] -> meas[4];
measure q[23] -> meas[5];
measure q[24] -> meas[6];
measure q[44] -> meas[7];
measure q[64] -> meas[8];
measure q[73] -> meas[9];
measure q[111] -> meas[10];
measure q[123] -> meas[11];
measure q[124] -> meas[12];
measure q[107] -> meas[13];
measure q[98] -> meas[14];
measure q[79] -> meas[15];
measure q[81] -> meas[16];
measure q[84] -> meas[17];
measure q[49] -> meas[18];
measure q[28] -> meas[19];
measure q[35] -> meas[20];
measure q[83] -> meas[21];
measure q[92] -> meas[22];
measure q[102] -> meas[23];
measure q[104] -> meas[24];
measure q[106] -> meas[25];
measure q[85] -> meas[26];
measure q[99] -> meas[27];
measure q[110] -> meas[28];
measure q[91] -> meas[29];
measure q[75] -> meas[30];
measure q[60] -> meas[31];
measure q[61] -> meas[32];
measure q[63] -> meas[33];
measure q[25] -> meas[34];
measure q[26] -> meas[35];
measure q[21] -> meas[36];
measure q[20] -> meas[37];
measure q[59] -> meas[38];
measure q[97] -> meas[39];
measure q[118] -> meas[40];
measure q[119] -> meas[41];
