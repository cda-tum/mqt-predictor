OPENQASM 2.0;
include "qelib1.inc";

qreg node[127];
creg meas[70];
sx node[45];
sx node[46];
sx node[47];
sx node[48];
sx node[49];
sx node[54];
sx node[55];
sx node[58];
sx node[59];
sx node[60];
sx node[61];
sx node[62];
sx node[63];
sx node[64];
sx node[65];
sx node[66];
sx node[67];
sx node[68];
sx node[69];
sx node[70];
sx node[71];
sx node[72];
sx node[73];
sx node[74];
sx node[75];
sx node[76];
sx node[77];
sx node[78];
sx node[79];
sx node[80];
sx node[81];
sx node[82];
sx node[83];
sx node[84];
sx node[85];
sx node[86];
sx node[87];
sx node[88];
sx node[89];
sx node[90];
sx node[91];
sx node[92];
sx node[93];
sx node[94];
sx node[95];
sx node[96];
sx node[97];
sx node[98];
sx node[99];
sx node[100];
sx node[101];
sx node[102];
sx node[103];
sx node[104];
sx node[105];
sx node[106];
sx node[107];
sx node[108];
sx node[110];
sx node[111];
sx node[112];
sx node[118];
sx node[119];
sx node[120];
sx node[121];
sx node[122];
sx node[123];
sx node[124];
sx node[125];
sx node[126];
rz(0.5*pi) node[47];
rz(0.5*pi) node[48];
rz(0.5*pi) node[54];
rz(0.5*pi) node[55];
rz(0.5*pi) node[59];
rz(0.5*pi) node[68];
rz(0.5*pi) node[71];
rz(0.5*pi) node[75];
rz(0.5*pi) node[83];
rz(0.5*pi) node[88];
rz(0.5*pi) node[92];
rz(0.5*pi) node[98];
rz(0.5*pi) node[99];
rz(0.5*pi) node[108];
rz(0.5*pi) node[111];
rz(0.5*pi) node[119];
rz(0.5*pi) node[126];
sx node[47];
sx node[48];
sx node[54];
sx node[55];
sx node[59];
sx node[68];
sx node[71];
sx node[75];
sx node[83];
sx node[88];
sx node[92];
sx node[98];
sx node[99];
sx node[108];
sx node[111];
sx node[119];
sx node[126];
cx node[47],node[46];
cx node[48],node[49];
cx node[54],node[64];
cx node[68],node[69];
cx node[71],node[77];
cx node[75],node[90];
cx node[83],node[82];
cx node[98],node[91];
cx node[92],node[102];
cx node[108],node[107];
cx node[111],node[122];
cx node[119],node[118];
cx node[126],node[125];
sx node[46];
cx node[49],node[48];
cx node[71],node[58];
sx node[64];
cx node[68],node[67];
sx node[69];
sx node[77];
sx node[82];
cx node[83],node[84];
sx node[90];
sx node[91];
cx node[98],node[97];
sx node[102];
cx node[111],node[104];
sx node[107];
cx node[112],node[108];
sx node[118];
cx node[119],node[120];
sx node[122];
sx node[125];
rz(2.5*pi) node[46];
cx node[48],node[49];
cx node[59],node[58];
rz(2.5*pi) node[64];
sx node[67];
rz(2.5*pi) node[69];
rz(2.5*pi) node[77];
rz(2.5*pi) node[82];
sx node[84];
rz(2.5*pi) node[90];
rz(2.5*pi) node[91];
rz(2.5*pi) node[102];
sx node[104];
rz(2.5*pi) node[107];
cx node[108],node[112];
rz(2.5*pi) node[118];
sx node[120];
rz(2.5*pi) node[122];
rz(2.5*pi) node[125];
sx node[46];
cx node[47],node[48];
cx node[49],node[55];
sx node[58];
cx node[59],node[60];
sx node[64];
rz(2.5*pi) node[67];
sx node[69];
sx node[77];
sx node[82];
rz(2.5*pi) node[84];
sx node[90];
sx node[91];
sx node[102];
rz(2.5*pi) node[104];
sx node[107];
cx node[112],node[108];
sx node[118];
rz(2.5*pi) node[120];
sx node[122];
sx node[125];
rz(1.5*pi) node[46];
cx node[55],node[49];
rz(2.5*pi) node[58];
sx node[60];
rz(1.5*pi) node[64];
sx node[67];
rz(1.5*pi) node[69];
rz(1.5*pi) node[77];
rz(1.5*pi) node[82];
sx node[84];
rz(1.5*pi) node[90];
rz(1.5*pi) node[91];
rz(1.5*pi) node[102];
sx node[104];
rz(1.5*pi) node[107];
rz(1.5*pi) node[118];
sx node[120];
rz(1.5*pi) node[122];
rz(1.5*pi) node[125];
cx node[46],node[45];
cx node[49],node[55];
sx node[58];
rz(2.5*pi) node[60];
cx node[64],node[63];
rz(1.5*pi) node[67];
cx node[69],node[70];
cx node[77],node[76];
cx node[91],node[79];
cx node[82],node[81];
rz(1.5*pi) node[84];
cx node[90],node[94];
cx node[102],node[101];
rz(1.5*pi) node[104];
cx node[107],node[106];
cx node[118],node[110];
rz(1.5*pi) node[120];
cx node[122],node[121];
cx node[125],node[124];
cx node[54],node[45];
cx node[55],node[68];
rz(1.5*pi) node[58];
sx node[60];
sx node[63];
sx node[70];
cx node[75],node[76];
sx node[79];
sx node[81];
cx node[84],node[85];
cx node[92],node[102];
sx node[94];
sx node[101];
cx node[104],node[105];
cx node[120],node[121];
cx node[123],node[122];
cx node[124],node[125];
sx node[45];
cx node[68],node[55];
rz(1.5*pi) node[60];
rz(2.5*pi) node[63];
rz(2.5*pi) node[70];
sx node[76];
rz(2.5*pi) node[79];
rz(2.5*pi) node[81];
sx node[85];
cx node[102],node[92];
rz(2.5*pi) node[94];
rz(2.5*pi) node[101];
sx node[105];
sx node[121];
cx node[122],node[123];
cx node[125],node[124];
rz(2.5*pi) node[45];
cx node[55],node[68];
sx node[63];
sx node[70];
rz(2.5*pi) node[76];
sx node[79];
sx node[81];
rz(2.5*pi) node[85];
cx node[92],node[102];
sx node[94];
sx node[101];
rz(2.5*pi) node[105];
rz(2.5*pi) node[121];
cx node[123],node[122];
cx node[124],node[125];
sx node[45];
rz(1.5*pi) node[63];
cx node[68],node[69];
rz(1.5*pi) node[70];
sx node[76];
rz(1.5*pi) node[79];
rz(1.5*pi) node[81];
sx node[85];
rz(1.5*pi) node[94];
rz(1.5*pi) node[101];
cx node[103],node[102];
sx node[105];
cx node[122],node[111];
sx node[121];
cx node[125],node[126];
rz(1.5*pi) node[45];
cx node[63],node[62];
cx node[69],node[68];
cx node[70],node[74];
rz(1.5*pi) node[76];
cx node[81],node[80];
rz(1.5*pi) node[85];
cx node[94],node[95];
cx node[101],node[100];
cx node[102],node[103];
rz(1.5*pi) node[105];
cx node[111],node[122];
rz(1.5*pi) node[121];
cx node[126],node[125];
sx node[62];
cx node[68],node[69];
cx node[85],node[73];
sx node[74];
cx node[79],node[80];
sx node[95];
sx node[100];
cx node[103],node[102];
cx node[105],node[106];
cx node[122],node[111];
cx node[125],node[126];
rz(2.5*pi) node[62];
cx node[69],node[70];
sx node[73];
rz(2.5*pi) node[74];
cx node[78],node[79];
sx node[80];
rz(2.5*pi) node[95];
rz(2.5*pi) node[100];
cx node[103],node[104];
sx node[106];
cx node[126],node[112];
sx node[62];
cx node[70],node[69];
rz(2.5*pi) node[73];
sx node[74];
cx node[79],node[78];
rz(2.5*pi) node[80];
sx node[95];
sx node[100];
cx node[104],node[103];
rz(2.5*pi) node[106];
cx node[112],node[126];
rz(1.5*pi) node[62];
cx node[69],node[70];
sx node[73];
rz(1.5*pi) node[74];
cx node[78],node[79];
sx node[80];
rz(1.5*pi) node[95];
rz(1.5*pi) node[100];
cx node[103],node[104];
sx node[106];
cx node[126],node[112];
cx node[62],node[61];
cx node[70],node[74];
rz(1.5*pi) node[73];
rz(1.5*pi) node[80];
cx node[95],node[96];
cx node[100],node[110];
cx node[104],node[105];
rz(1.5*pi) node[106];
cx node[125],node[126];
cx node[60],node[61];
cx node[73],node[66];
cx node[74],node[70];
cx node[79],node[80];
cx node[93],node[106];
sx node[96];
cx node[99],node[100];
cx node[105],node[104];
sx node[110];
cx node[126],node[125];
sx node[61];
cx node[67],node[66];
cx node[70],node[74];
cx node[80],node[79];
cx node[106],node[93];
rz(2.5*pi) node[96];
cx node[100],node[99];
cx node[104],node[105];
rz(2.5*pi) node[110];
cx node[125],node[126];
rz(2.5*pi) node[61];
sx node[66];
cx node[89],node[74];
cx node[79],node[80];
cx node[93],node[106];
sx node[96];
cx node[99],node[100];
cx node[111],node[104];
sx node[110];
cx node[126],node[112];
sx node[61];
rz(2.5*pi) node[66];
cx node[74],node[89];
cx node[80],node[81];
rz(1.5*pi) node[96];
cx node[100],node[101];
cx node[104],node[111];
cx node[106],node[107];
rz(1.5*pi) node[110];
rz(1.5*pi) node[61];
sx node[66];
cx node[89],node[74];
cx node[81],node[80];
cx node[96],node[97];
cx node[101],node[100];
cx node[111],node[104];
cx node[107],node[106];
rz(1.5*pi) node[66];
cx node[74],node[70];
cx node[80],node[81];
cx node[89],node[88];
sx node[97];
cx node[100],node[101];
cx node[104],node[103];
cx node[106],node[107];
cx node[70],node[74];
cx node[88],node[89];
rz(2.5*pi) node[97];
cx node[101],node[102];
cx node[103],node[104];
cx node[108],node[107];
cx node[74],node[70];
cx node[89],node[88];
sx node[97];
sx node[102];
cx node[104],node[103];
cx node[107],node[108];
cx node[70],node[69];
cx node[88],node[87];
rz(1.5*pi) node[97];
rz(2.5*pi) node[102];
cx node[108],node[107];
cx node[69],node[70];
cx node[87],node[88];
sx node[102];
cx node[112],node[108];
cx node[70],node[69];
cx node[88],node[87];
rz(1.5*pi) node[102];
cx node[126],node[112];
cx node[69],node[68];
cx node[74],node[70];
cx node[87],node[93];
cx node[89],node[88];
cx node[103],node[102];
cx node[112],node[108];
cx node[125],node[126];
cx node[68],node[69];
cx node[70],node[74];
cx node[93],node[87];
cx node[88],node[89];
cx node[102],node[103];
cx node[69],node[68];
cx node[74],node[70];
cx node[87],node[93];
cx node[89],node[88];
cx node[103],node[102];
cx node[68],node[55];
cx node[70],node[69];
cx node[88],node[87];
cx node[93],node[106];
cx node[101],node[102];
cx node[55],node[68];
cx node[69],node[70];
cx node[87],node[88];
cx node[106],node[93];
cx node[103],node[102];
cx node[68],node[55];
cx node[70],node[69];
cx node[88],node[87];
cx node[93],node[106];
sx node[102];
cx node[55],node[49];
cx node[87],node[93];
cx node[89],node[88];
rz(2.5*pi) node[102];
cx node[105],node[106];
cx node[49],node[55];
cx node[93],node[87];
cx node[88],node[89];
sx node[102];
cx node[106],node[107];
cx node[55],node[49];
cx node[87],node[93];
cx node[89],node[88];
rz(1.5*pi) node[102];
cx node[105],node[106];
cx node[48],node[49];
cx node[106],node[107];
cx node[47],node[48];
cx node[108],node[107];
cx node[48],node[49];
cx node[107],node[108];
cx node[49],node[55];
cx node[108],node[107];
cx node[55],node[49];
cx node[107],node[106];
cx node[112],node[108];
cx node[49],node[55];
cx node[106],node[107];
cx node[108],node[112];
cx node[107],node[106];
cx node[112],node[108];
cx node[93],node[106];
cx node[107],node[108];
cx node[126],node[112];
sx node[106];
sx node[108];
cx node[125],node[126];
rz(2.5*pi) node[106];
rz(2.5*pi) node[108];
cx node[126],node[112];
sx node[106];
sx node[108];
sx node[112];
rz(1.5*pi) node[106];
rz(1.5*pi) node[108];
rz(2.5*pi) node[112];
cx node[107],node[106];
sx node[112];
cx node[106],node[107];
rz(1.5*pi) node[112];
cx node[107],node[106];
cx node[106],node[93];
cx node[93],node[106];
cx node[106],node[93];
cx node[93],node[87];
cx node[87],node[93];
cx node[93],node[87];
cx node[87],node[86];
cx node[106],node[93];
cx node[86],node[87];
cx node[93],node[106];
cx node[87],node[86];
cx node[106],node[93];
cx node[86],node[85];
cx node[93],node[87];
cx node[85],node[86];
cx node[87],node[88];
cx node[86],node[85];
cx node[93],node[87];
cx node[85],node[73];
cx node[87],node[88];
cx node[73],node[85];
sx node[88];
cx node[85],node[73];
rz(2.5*pi) node[88];
cx node[73],node[66];
sx node[88];
cx node[66],node[73];
rz(1.5*pi) node[88];
cx node[73],node[66];
cx node[88],node[87];
cx node[66],node[67];
cx node[87],node[86];
cx node[67],node[66];
cx node[86],node[87];
cx node[66],node[67];
cx node[87],node[86];
cx node[67],node[68];
cx node[86],node[85];
cx node[68],node[67];
cx node[85],node[86];
cx node[67],node[68];
cx node[86],node[85];
cx node[68],node[55];
cx node[85],node[84];
cx node[55],node[68];
cx node[84],node[85];
cx node[68],node[55];
cx node[85],node[84];
cx node[55],node[49];
cx node[69],node[68];
cx node[84],node[83];
cx node[49],node[48];
sx node[68];
cx node[83],node[84];
cx node[55],node[49];
rz(2.5*pi) node[68];
cx node[84],node[83];
cx node[49],node[48];
sx node[68];
cx node[83],node[82];
cx node[49],node[48];
rz(1.5*pi) node[68];
cx node[82],node[83];
sx node[48];
cx node[49],node[55];
cx node[83],node[82];
rz(2.5*pi) node[48];
cx node[55],node[49];
sx node[48];
cx node[49],node[55];
rz(1.5*pi) node[48];
cx node[55],node[68];
cx node[68],node[55];
cx node[55],node[68];
cx node[68],node[67];
cx node[67],node[68];
cx node[68],node[67];
cx node[67],node[66];
cx node[66],node[67];
cx node[67],node[66];
cx node[66],node[65];
sx node[65];
rz(2.5*pi) node[65];
sx node[65];
rz(1.5*pi) node[65];
cx node[65],node[64];
cx node[64],node[65];
cx node[65],node[64];
cx node[64],node[63];
cx node[63],node[64];
cx node[64],node[63];
cx node[63],node[62];
cx node[62],node[63];
cx node[63],node[62];
cx node[62],node[72];
sx node[72];
rz(2.5*pi) node[72];
sx node[72];
rz(1.5*pi) node[72];
cx node[72],node[81];
sx node[81];
rz(2.5*pi) node[81];
sx node[81];
rz(1.5*pi) node[81];
cx node[81],node[82];
sx node[82];
rz(2.5*pi) node[82];
sx node[82];
rz(1.5*pi) node[82];
barrier node[47],node[46],node[105],node[92],node[54],node[65],node[126],node[124],node[125],node[106],node[45],node[71],node[77],node[84],node[83],node[93],node[107],node[100],node[68],node[70],node[49],node[108],node[101],node[103],node[112],node[64],node[74],node[69],node[88],node[85],node[63],node[66],node[48],node[122],node[123],node[55],node[59],node[58],node[119],node[118],node[120],node[80],node[75],node[90],node[87],node[62],node[99],node[102],node[104],node[111],node[98],node[91],node[94],node[95],node[60],node[86],node[72],node[81],node[76],node[67],node[78],node[79],node[110],node[73],node[121],node[61],node[89],node[82],node[96],node[97];
measure node[47] -> meas[0];
measure node[46] -> meas[1];
measure node[105] -> meas[2];
measure node[92] -> meas[3];
measure node[54] -> meas[4];
measure node[65] -> meas[5];
measure node[126] -> meas[6];
measure node[124] -> meas[7];
measure node[125] -> meas[8];
measure node[106] -> meas[9];
measure node[45] -> meas[10];
measure node[71] -> meas[11];
measure node[77] -> meas[12];
measure node[84] -> meas[13];
measure node[83] -> meas[14];
measure node[93] -> meas[15];
measure node[107] -> meas[16];
measure node[100] -> meas[17];
measure node[68] -> meas[18];
measure node[70] -> meas[19];
measure node[49] -> meas[20];
measure node[108] -> meas[21];
measure node[101] -> meas[22];
measure node[103] -> meas[23];
measure node[112] -> meas[24];
measure node[64] -> meas[25];
measure node[74] -> meas[26];
measure node[69] -> meas[27];
measure node[88] -> meas[28];
measure node[85] -> meas[29];
measure node[63] -> meas[30];
measure node[66] -> meas[31];
measure node[48] -> meas[32];
measure node[122] -> meas[33];
measure node[123] -> meas[34];
measure node[55] -> meas[35];
measure node[59] -> meas[36];
measure node[58] -> meas[37];
measure node[119] -> meas[38];
measure node[118] -> meas[39];
measure node[120] -> meas[40];
measure node[80] -> meas[41];
measure node[75] -> meas[42];
measure node[90] -> meas[43];
measure node[87] -> meas[44];
measure node[62] -> meas[45];
measure node[99] -> meas[46];
measure node[102] -> meas[47];
measure node[104] -> meas[48];
measure node[111] -> meas[49];
measure node[98] -> meas[50];
measure node[91] -> meas[51];
measure node[94] -> meas[52];
measure node[95] -> meas[53];
measure node[60] -> meas[54];
measure node[86] -> meas[55];
measure node[72] -> meas[56];
measure node[81] -> meas[57];
measure node[76] -> meas[58];
measure node[67] -> meas[59];
measure node[78] -> meas[60];
measure node[79] -> meas[61];
measure node[110] -> meas[62];
measure node[73] -> meas[63];
measure node[121] -> meas[64];
measure node[61] -> meas[65];
measure node[89] -> meas[66];
measure node[82] -> meas[67];
measure node[96] -> meas[68];
measure node[97] -> meas[69];
