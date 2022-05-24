OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[43];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[23];
cx q[22],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[4],q[15];
cx q[15],q[4];
cx q[4],q[15];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[6],q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[8];
cx q[8],q[7];
cx q[7],q[8];
cx q[8],q[16];
cx q[16],q[8];
cx q[8],q[16];
cx q[23],q[24];
cx q[16],q[26];
cx q[26],q[16];
cx q[16],q[26];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[27],q[28];
rz(pi/2) q[28];
sx q[28];
rz(pi/2) q[28];
cx q[28],q[35];
cx q[35],q[28];
cx q[28],q[35];
rz(-pi) q[36];
x q[36];
rz(pi/2) q[43];
sx q[43];
rz(pi) q[43];
cx q[43],q[34];
rz(pi/2) q[34];
sx q[34];
rz(pi/2) q[34];
cx q[34],q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
sx q[43];
rz(pi/2) q[43];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(pi/2) q[44];
sx q[44];
rz(-pi) q[44];
cx q[43],q[44];
sx q[43];
rz(-pi/2) q[43];
sx q[43];
rz(pi/2) q[44];
cx q[43],q[44];
rz(pi/2) q[43];
sx q[43];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
rz(-pi) q[44];
sx q[44];
rz(pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
cx q[41],q[53];
rz(pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[63],q[64];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
rz(pi/2) q[66];
sx q[66];
rz(pi/2) q[66];
cx q[66],q[65];
rz(pi/2) q[65];
sx q[65];
rz(pi/2) q[65];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(pi/2) q[67];
sx q[67];
rz(pi/2) q[67];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
rz(pi/2) q[66];
sx q[66];
rz(pi/2) q[66];
rz(pi/2) q[68];
sx q[68];
rz(pi/2) q[68];
cx q[67],q[68];
cx q[68],q[67];
cx q[67],q[68];
cx q[68],q[69];
cx q[69],q[68];
cx q[68],q[69];
cx q[69],q[70];
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
cx q[62],q[61];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[61],q[60];
cx q[53],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
cx q[73],q[66];
rz(pi/2) q[66];
sx q[66];
rz(pi/2) q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
rz(pi/2) q[66];
sx q[66];
rz(pi/2) q[66];
cx q[67],q[68];
cx q[68],q[67];
cx q[67],q[68];
cx q[55],q[68];
cx q[68],q[55];
cx q[55],q[68];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
cx q[50],q[49];
cx q[49],q[50];
cx q[50],q[49];
cx q[50],q[51];
rz(-pi/2) q[51];
sx q[51];
rz(-2.9276986) q[51];
sx q[51];
cx q[36],q[51];
sx q[36];
rz(-pi/2) q[36];
sx q[36];
rz(pi/2) q[51];
cx q[36],q[51];
rz(pi/2) q[36];
sx q[36];
rz(-2.9276986) q[36];
sx q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[49];
cx q[49],q[50];
cx q[50],q[49];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
cx q[55],q[68];
cx q[68],q[55];
cx q[55],q[68];
cx q[68],q[67];
cx q[67],q[68];
cx q[68],q[67];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[81],q[82];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[85];
sx q[85];
rz(pi/2) q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[73],q[66];
rz(pi/2) q[66];
sx q[66];
rz(pi/2) q[66];
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
rz(pi/2) q[46];
sx q[46];
rz(pi/2) q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
rz(pi/2) q[47];
sx q[47];
rz(pi/2) q[47];
cx q[35],q[47];
rz(pi/2) q[47];
sx q[47];
rz(pi/2) q[47];
rz(pi/2) q[86];
sx q[86];
rz(pi/2) q[86];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
rz(pi/2) q[85];
sx q[85];
rz(pi/2) q[85];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
rz(pi/2) q[90];
sx q[90];
rz(pi/2) q[90];
cx q[90],q[75];
rz(pi/2) q[75];
sx q[75];
rz(pi/2) q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[77];
cx q[77],q[76];
cx q[76],q[77];
cx q[77],q[78];
rz(pi/2) q[78];
sx q[78];
rz(pi/2) q[78];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
rz(-1.5289023) q[92];
sx q[92];
rz(-pi/2) q[92];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[90],q[94];
cx q[94],q[90];
cx q[90],q[94];
cx q[94],q[95];
rz(pi/2) q[95];
sx q[95];
rz(pi/2) q[95];
cx q[95],q[96];
cx q[96],q[95];
cx q[95],q[96];
cx q[96],q[97];
cx q[97],q[96];
cx q[96],q[97];
rz(pi/2) q[97];
sx q[97];
rz(pi/2) q[97];
cx q[91],q[98];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
rz(pi/2) q[100];
sx q[100];
rz(pi/2) q[100];
rz(pi/2) q[101];
sx q[101];
rz(pi/2) q[101];
cx q[101],q[102];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
rz(pi/2) q[101];
sx q[101];
rz(pi/2) q[101];
rz(pi/2) q[102];
sx q[102];
rz(pi/2) q[102];
cx q[99],q[100];
cx q[100],q[99];
cx q[99],q[100];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[99];
cx q[98],q[97];
rz(pi/2) q[97];
sx q[97];
rz(pi/2) q[97];
rz(pi/2) q[99];
sx q[99];
rz(pi/2) q[99];
rz(pi/2) q[103];
sx q[103];
rz(pi/2) q[103];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
rz(pi/2) q[102];
sx q[102];
rz(-pi) q[102];
cx q[92],q[102];
rz(pi/2) q[102];
sx q[92];
rz(-pi/2) q[92];
sx q[92];
cx q[92],q[102];
sx q[102];
rz(-0.041894027) q[102];
sx q[92];
cx q[83],q[92];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[85];
rz(pi/2) q[85];
sx q[85];
rz(pi/2) q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
rz(pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
cx q[66],q[73];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
rz(pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[101];
rz(pi/2) q[101];
sx q[101];
rz(pi/2) q[101];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[100],q[99];
rz(pi/2) q[99];
sx q[99];
rz(pi/2) q[99];
rz(pi/2) q[104];
sx q[104];
rz(pi/2) q[104];
rz(pi/2) q[105];
sx q[105];
rz(pi/2) q[105];
cx q[104],q[105];
cx q[105],q[104];
cx q[104],q[105];
rz(pi/2) q[104];
sx q[104];
rz(pi/2) q[104];
cx q[103],q[104];
rz(pi/2) q[104];
sx q[104];
rz(pi/2) q[104];
cx q[93],q[106];
rz(pi/2) q[106];
sx q[106];
rz(pi/2) q[106];
cx q[105],q[106];
cx q[106],q[105];
cx q[105],q[106];
rz(pi/2) q[105];
sx q[105];
rz(pi/2) q[105];
rz(pi/2) q[106];
sx q[106];
rz(pi/2) q[106];
cx q[93],q[87];
cx q[87],q[93];
cx q[93],q[87];
cx q[93],q[106];
rz(pi/2) q[106];
sx q[106];
rz(pi/2) q[106];
cx q[106],q[105];
rz(pi/2) q[105];
sx q[105];
rz(pi/2) q[105];
cx q[104],q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[106],q[105];
cx q[105],q[106];
cx q[106],q[105];
cx q[93],q[106];
cx q[106],q[93];
cx q[93],q[106];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[88],q[87];
cx q[87],q[88];
cx q[88],q[87];
cx q[88],q[89];
rz(pi/2) q[89];
sx q[89];
rz(pi/2) q[89];
cx q[74],q[89];
cx q[89],q[74];
cx q[74],q[89];
cx q[74],q[70];
rz(pi/2) q[70];
sx q[70];
rz(pi/2) q[70];
barrier q[52],q[61],q[5],q[125],q[70],q[6],q[78],q[24],q[87],q[64],q[118],q[65],q[7],q[62],q[17],q[81],q[16],q[94],q[46],q[111],q[56],q[1],q[120],q[63],q[10],q[89],q[19],q[72],q[40],q[105],q[49],q[113],q[58],q[3],q[122],q[51],q[12],q[75],q[33],q[96],q[43],q[104],q[36],q[115],q[53],q[4],q[124],q[68],q[28],q[101],q[54],q[108],q[60],q[117],q[85],q[126],q[71],q[35],q[82],q[37],q[98],q[47],q[110],q[55],q[0],q[119],q[67],q[21],q[106],q[30],q[90],q[39],q[83],q[48],q[112],q[57],q[121],q[14],q[91],q[23],q[73],q[32],q[95],q[42],q[88],q[50],q[114],q[27],q[8],q[80],q[25],q[74],q[34],q[99],q[44],q[107],q[116],q[9],q[93],q[18],q[92],q[26],q[79],q[102],q[100],q[41],q[109],q[2],q[69],q[11],q[77],q[20],q[84],q[29],q[86],q[38],q[103],q[59],q[15],q[123],q[45],q[13],q[76],q[22],q[66],q[31],q[97];
measure q[94] -> meas[0];
measure q[77] -> meas[1];
measure q[106] -> meas[2];
measure q[45] -> meas[3];
measure q[22] -> meas[4];
measure q[23] -> meas[5];
measure q[44] -> meas[6];
measure q[34] -> meas[7];
measure q[93] -> meas[8];
measure q[51] -> meas[9];
measure q[6] -> meas[10];
measure q[81] -> meas[11];
measure q[92] -> meas[12];
measure q[98] -> meas[13];
measure q[103] -> meas[14];
measure q[105] -> meas[15];
measure q[104] -> meas[16];
measure q[88] -> meas[17];
measure q[82] -> meas[18];
measure q[65] -> meas[19];
measure q[85] -> meas[20];
measure q[97] -> meas[21];
measure q[69] -> meas[22];
measure q[63] -> meas[23];
measure q[67] -> meas[24];
measure q[74] -> meas[25];
measure q[36] -> meas[26];
measure q[83] -> meas[27];
measure q[91] -> meas[28];
measure q[27] -> meas[29];
measure q[102] -> meas[30];
measure q[62] -> meas[31];
measure q[61] -> meas[32];
measure q[41] -> meas[33];
measure q[35] -> meas[34];
measure q[53] -> meas[35];
measure q[70] -> meas[36];
measure q[66] -> meas[37];
measure q[60] -> meas[38];
measure q[24] -> meas[39];
measure q[47] -> meas[40];
measure q[100] -> meas[41];
measure q[99] -> meas[42];
