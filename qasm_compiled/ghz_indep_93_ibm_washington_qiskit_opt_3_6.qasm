OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[93];
cx q[17],q[30];
cx q[30],q[17];
cx q[17],q[30];
x q[35];
x q[37];
rz(pi/2) q[58];
sx q[58];
rz(pi/2) q[58];
cx q[58],q[59];
cx q[59],q[60];
cx q[60],q[61];
rz(-pi) q[61];
sx q[61];
rz(pi/2) q[61];
rz(pi/2) q[62];
cx q[61],q[62];
sx q[61];
rz(-pi/2) q[61];
sx q[61];
rz(pi/2) q[62];
cx q[61],q[62];
rz(-pi) q[61];
sx q[61];
rz(-pi) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[41],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[34];
cx q[24],q[34];
cx q[34],q[24];
cx q[24],q[34];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[16],q[26];
cx q[26],q[16];
cx q[16],q[26];
rz(-pi) q[62];
sx q[62];
rz(-pi/2) q[62];
cx q[8],q[16];
cx q[16],q[8];
cx q[8],q[16];
cx q[8],q[7];
cx q[6],q[7];
cx q[7],q[6];
cx q[6],q[7];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[4],q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[1],q[0];
cx q[0],q[14];
cx q[14],q[0];
cx q[0],q[14];
cx q[14],q[18];
cx q[18],q[14];
cx q[14],q[18];
cx q[18],q[19];
cx q[19],q[18];
cx q[18],q[19];
x q[18];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[23];
x q[22];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[27],q[28];
cx q[28],q[27];
cx q[27],q[28];
cx q[28],q[29];
cx q[29],q[28];
cx q[28],q[29];
cx q[29],q[30];
cx q[30],q[31];
cx q[30],q[17];
cx q[17],q[30];
cx q[30],q[17];
cx q[29],q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[31],q[32];
cx q[32],q[36];
cx q[36],q[51];
cx q[51],q[50];
cx q[50],q[49];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
cx q[55],q[68];
cx q[68],q[67];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
rz(pi/2) q[67];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
rz(pi/2) q[74];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[87];
rz(pi/2) q[89];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
x q[97];
cx q[93],q[106];
cx q[106],q[93];
cx q[93],q[106];
cx q[106],q[105];
cx q[105],q[106];
cx q[106],q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[105],q[104];
cx q[104],q[103];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
rz(pi/2) q[103];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[82],q[81];
cx q[81],q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[62],q[63];
cx q[63],q[64];
cx q[64],q[54];
cx q[54],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[47];
rz(pi/2) q[47];
sx q[47];
rz(-pi) q[47];
cx q[35],q[47];
sx q[35];
rz(-pi/2) q[35];
sx q[35];
rz(pi/2) q[47];
cx q[35],q[47];
rz(pi/2) q[35];
sx q[35];
rz(pi/2) q[47];
sx q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[43],q[34];
cx q[34],q[24];
cx q[24],q[34];
cx q[34],q[24];
cx q[24],q[23];
rz(pi/2) q[23];
sx q[23];
rz(-pi) q[23];
cx q[22],q[23];
sx q[22];
rz(-pi/2) q[22];
sx q[22];
rz(pi/2) q[23];
cx q[22],q[23];
rz(pi/2) q[22];
sx q[22];
cx q[15],q[22];
cx q[22],q[15];
cx q[15],q[22];
rz(pi/2) q[23];
sx q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[25];
cx q[25],q[26];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[27],q[28];
cx q[28],q[35];
cx q[35],q[28];
cx q[28],q[35];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[47],q[48];
cx q[48],q[49];
cx q[49],q[48];
cx q[48],q[49];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
cx q[55],q[68];
cx q[68],q[55];
cx q[55],q[68];
cx q[68],q[69];
cx q[69],q[70];
cx q[70],q[69];
cx q[69],q[70];
rz(-pi) q[70];
sx q[70];
rz(pi/2) q[70];
cx q[70],q[74];
sx q[70];
rz(-pi/2) q[70];
sx q[70];
rz(pi/2) q[74];
cx q[70],q[74];
rz(-pi) q[70];
sx q[70];
rz(-pi) q[70];
cx q[70],q[69];
cx q[68],q[69];
cx q[69],q[68];
cx q[68],q[69];
cx q[55],q[68];
cx q[68],q[55];
cx q[55],q[68];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
cx q[49],q[48];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[43],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
cx q[39],q[33];
cx q[20],q[33];
cx q[33],q[20];
cx q[20],q[33];
cx q[20],q[19];
rz(pi/2) q[19];
sx q[19];
rz(-pi) q[19];
cx q[18],q[19];
sx q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[19];
cx q[18],q[19];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[19];
sx q[19];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
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
cx q[64],q[65];
cx q[65],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[85];
rz(-pi) q[74];
sx q[74];
rz(-pi/2) q[74];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
cx q[83],q[92];
cx q[92],q[102];
rz(-pi) q[102];
sx q[102];
rz(pi/2) q[102];
cx q[102],q[103];
sx q[102];
rz(-pi/2) q[102];
sx q[102];
rz(pi/2) q[103];
cx q[102],q[103];
rz(-pi) q[102];
sx q[102];
rz(-pi) q[102];
rz(-pi) q[103];
sx q[103];
rz(-pi/2) q[103];
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
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[73],q[66];
rz(-pi) q[66];
sx q[66];
rz(pi/2) q[66];
cx q[66],q[67];
sx q[66];
rz(-pi/2) q[66];
sx q[66];
rz(pi/2) q[67];
cx q[66],q[67];
rz(-pi) q[66];
sx q[66];
rz(-pi) q[66];
cx q[66],q[73];
rz(-pi) q[67];
sx q[67];
rz(-pi/2) q[67];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[102],q[101];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[100],q[110];
rz(pi/2) q[112];
x q[115];
cx q[110],q[118];
cx q[118],q[110];
cx q[110],q[118];
cx q[118],q[117];
cx q[117],q[118];
cx q[118],q[117];
cx q[117],q[116];
rz(pi/2) q[116];
sx q[116];
rz(-pi) q[116];
cx q[115],q[116];
sx q[115];
rz(-pi/2) q[115];
sx q[115];
rz(pi/2) q[116];
cx q[115],q[116];
rz(pi/2) q[115];
sx q[115];
rz(pi/2) q[116];
sx q[116];
cx q[117],q[116];
cx q[116],q[117];
cx q[117],q[116];
cx q[117],q[118];
cx q[118],q[110];
cx q[110],q[118];
cx q[118],q[110];
cx q[100],q[110];
cx q[110],q[100];
cx q[100],q[110];
cx q[99],q[100];
cx q[100],q[99];
cx q[99],q[100];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[99];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[91],q[79];
cx q[79],q[80];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[86],q[87];
cx q[87],q[88];
rz(-pi) q[88];
sx q[88];
rz(pi/2) q[88];
cx q[88],q[89];
sx q[88];
rz(-pi/2) q[88];
sx q[88];
rz(pi/2) q[89];
cx q[88],q[89];
rz(-pi) q[88];
sx q[88];
rz(-pi) q[88];
cx q[88],q[87];
cx q[87],q[88];
cx q[88],q[87];
cx q[87],q[93];
rz(-pi) q[89];
sx q[89];
rz(-pi/2) q[89];
cx q[93],q[87];
cx q[87],q[93];
cx q[93],q[106];
cx q[106],q[93];
cx q[93],q[106];
cx q[106],q[105];
cx q[105],q[106];
cx q[106],q[105];
cx q[104],q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[104],q[103];
cx q[103],q[104];
cx q[104],q[103];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
cx q[102],q[101];
cx q[101],q[100];
cx q[100],q[99];
cx q[99],q[98];
rz(pi/2) q[98];
sx q[98];
rz(-pi) q[98];
cx q[97],q[98];
sx q[97];
rz(-pi/2) q[97];
sx q[97];
rz(pi/2) q[98];
cx q[97],q[98];
rz(pi/2) q[97];
sx q[97];
rz(pi/2) q[98];
sx q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[98],q[91];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[79],q[78];
cx q[77],q[78];
cx q[78],q[77];
cx q[77],q[78];
cx q[77],q[71];
cx q[58],q[71];
cx q[71],q[58];
cx q[58],q[71];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[59],q[60];
cx q[60],q[53];
cx q[53],q[41];
cx q[41],q[42];
cx q[42],q[41];
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
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[46],q[47];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[35],q[28];
cx q[28],q[35];
cx q[35],q[28];
cx q[27],q[28];
cx q[28],q[27];
cx q[27],q[28];
cx q[27],q[26];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[21],q[20];
cx q[28],q[29];
cx q[29],q[28];
cx q[28],q[29];
cx q[33],q[20];
cx q[20],q[33];
cx q[33],q[20];
cx q[33],q[39];
cx q[39],q[33];
cx q[33],q[39];
cx q[39],q[38];
rz(pi/2) q[38];
sx q[38];
rz(-pi) q[38];
cx q[37],q[38];
sx q[37];
rz(-pi/2) q[37];
sx q[37];
rz(pi/2) q[38];
cx q[37],q[38];
rz(pi/2) q[37];
sx q[37];
rz(pi/2) q[38];
sx q[38];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
cx q[38],q[39];
cx q[39],q[33];
cx q[33],q[39];
cx q[39],q[33];
cx q[20],q[33];
cx q[33],q[20];
cx q[20],q[33];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[15],q[22];
cx q[22],q[15];
cx q[15],q[22];
cx q[4],q[15];
cx q[15],q[4];
cx q[4],q[15];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[7];
cx q[7],q[8];
cx q[8],q[16];
cx q[16],q[8];
cx q[8],q[16];
cx q[16],q[26];
cx q[26],q[16];
cx q[16],q[26];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[27],q[28];
cx q[28],q[35];
cx q[35],q[28];
cx q[28],q[35];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[87],q[93];
cx q[93],q[106];
cx q[106],q[107];
cx q[107],q[106];
cx q[106],q[107];
cx q[107],q[108];
cx q[108],q[107];
cx q[107],q[108];
rz(-pi) q[108];
sx q[108];
rz(pi/2) q[108];
cx q[108],q[112];
sx q[108];
rz(-pi/2) q[108];
sx q[108];
rz(pi/2) q[112];
cx q[108],q[112];
rz(-pi) q[108];
sx q[108];
rz(-pi) q[108];
cx q[108],q[107];
cx q[107],q[106];
cx q[105],q[106];
cx q[106],q[105];
cx q[105],q[106];
cx q[104],q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[104],q[111];
rz(-pi) q[112];
sx q[112];
rz(-pi/2) q[112];
cx q[111],q[122];
cx q[122],q[111];
cx q[111],q[122];
cx q[122],q[121];
cx q[121],q[122];
cx q[122],q[121];
cx q[121],q[120];
cx q[120],q[119];
barrier q[114],q[1],q[59],q[16],q[80],q[22],q[102],q[7],q[99],q[39],q[104],q[33],q[45],q[9],q[83],q[14],q[82],q[29],q[97],q[36],q[101],q[57],q[3],q[122],q[66],q[11],q[75],q[18],q[84],q[48],q[88],q[50],q[58],q[6],q[123],q[68],q[13],q[78],q[40],q[103],q[52],q[115],q[72],q[27],q[125],q[49],q[25],q[91],q[23],q[89],q[35],q[109],q[44],q[118],q[63],q[8],q[62],q[17],q[85],q[37],q[65],q[28],q[121],q[56],q[2],q[120],q[64],q[10],q[70],q[31],q[95],q[47],q[112],q[55],q[113],q[71],q[15],q[111],q[73],q[19],q[79],q[43],q[86],q[51],q[117],q[61],q[124],q[74],q[54],q[90],q[46],q[100],q[42],q[107],q[60],q[98],q[41],q[126],q[21],q[92],q[20],q[105],q[38],q[110],q[26],q[116],q[34],q[119],q[12],q[76],q[30],q[67],q[87],q[94],q[4],q[81],q[69],q[108],q[5],q[0],q[77],q[24],q[106],q[32],q[96],q[53],q[93];
measure q[119] -> meas[0];
measure q[120] -> meas[1];
measure q[121] -> meas[2];
measure q[104] -> meas[3];
measure q[107] -> meas[4];
measure q[108] -> meas[5];
measure q[112] -> meas[6];
measure q[93] -> meas[7];
measure q[87] -> meas[8];
measure q[27] -> meas[9];
measure q[6] -> meas[10];
measure q[4] -> meas[11];
measure q[38] -> meas[12];
measure q[37] -> meas[13];
measure q[40] -> meas[14];
measure q[20] -> meas[15];
measure q[26] -> meas[16];
measure q[47] -> meas[17];
measure q[53] -> meas[18];
measure q[60] -> meas[19];
measure q[59] -> meas[20];
measure q[77] -> meas[21];
measure q[79] -> meas[22];
measure q[97] -> meas[23];
measure q[99] -> meas[24];
measure q[100] -> meas[25];
measure q[101] -> meas[26];
measure q[102] -> meas[27];
measure q[89] -> meas[28];
measure q[88] -> meas[29];
measure q[85] -> meas[30];
measure q[91] -> meas[31];
measure q[98] -> meas[32];
measure q[117] -> meas[33];
measure q[115] -> meas[34];
measure q[116] -> meas[35];
measure q[110] -> meas[36];
measure q[103] -> meas[37];
measure q[66] -> meas[38];
measure q[67] -> meas[39];
measure q[65] -> meas[40];
measure q[105] -> meas[41];
measure q[92] -> meas[42];
measure q[82] -> meas[43];
measure q[83] -> meas[44];
measure q[64] -> meas[45];
measure q[54] -> meas[46];
measure q[25] -> meas[47];
measure q[21] -> meas[48];
measure q[18] -> meas[49];
measure q[19] -> meas[50];
measure q[39] -> meas[51];
measure q[34] -> meas[52];
measure q[49] -> meas[53];
measure q[70] -> meas[54];
measure q[74] -> meas[55];
measure q[69] -> meas[56];
measure q[48] -> meas[57];
measure q[29] -> meas[58];
measure q[16] -> meas[59];
measure q[24] -> meas[60];
measure q[22] -> meas[61];
measure q[23] -> meas[62];
measure q[42] -> meas[63];
measure q[46] -> meas[64];
measure q[28] -> meas[65];
measure q[35] -> meas[66];
measure q[44] -> meas[67];
measure q[45] -> meas[68];
measure q[63] -> meas[69];
measure q[62] -> meas[70];
measure q[80] -> meas[71];
measure q[81] -> meas[72];
measure q[106] -> meas[73];
measure q[73] -> meas[74];
measure q[68] -> meas[75];
measure q[55] -> meas[76];
measure q[50] -> meas[77];
measure q[51] -> meas[78];
measure q[36] -> meas[79];
measure q[32] -> meas[80];
measure q[31] -> meas[81];
measure q[17] -> meas[82];
measure q[30] -> meas[83];
measure q[33] -> meas[84];
measure q[1] -> meas[85];
measure q[7] -> meas[86];
measure q[43] -> meas[87];
measure q[41] -> meas[88];
measure q[72] -> meas[89];
measure q[61] -> meas[90];
measure q[58] -> meas[91];
measure q[71] -> meas[92];
