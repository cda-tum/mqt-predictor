OPENQASM 2.0;
include "qelib1.inc";

qreg node[127];
creg meas[44];
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
rz(0.5*pi) node[75];
rz(0.5*pi) node[85];
rz(0.5*pi) node[92];
rz(0.5*pi) node[99];
rz(0.5*pi) node[103];
rz(0.5*pi) node[107];
rz(0.5*pi) node[111];
cx node[126],node[112];
rz(0.5*pi) node[118];
rz(0.5*pi) node[121];
rz(0.5*pi) node[123];
sx node[75];
sx node[85];
sx node[92];
sx node[99];
sx node[103];
sx node[107];
sx node[111];
cx node[112],node[126];
sx node[118];
sx node[121];
sx node[123];
cx node[75],node[76];
cx node[85],node[86];
cx node[99],node[100];
cx node[103],node[102];
cx node[107],node[106];
cx node[126],node[112];
cx node[118],node[119];
cx node[75],node[90];
sx node[76];
cx node[85],node[84];
sx node[86];
cx node[99],node[98];
sx node[100];
sx node[102];
cx node[103],node[104];
sx node[106];
cx node[107],node[108];
sx node[119];
cx node[90],node[75];
rz(2.5*pi) node[76];
sx node[84];
rz(2.5*pi) node[86];
sx node[98];
rz(2.5*pi) node[100];
rz(2.5*pi) node[102];
sx node[104];
rz(2.5*pi) node[106];
cx node[112],node[108];
rz(2.5*pi) node[119];
cx node[75],node[90];
sx node[76];
rz(2.5*pi) node[84];
sx node[86];
rz(2.5*pi) node[98];
sx node[100];
sx node[102];
rz(2.5*pi) node[104];
sx node[106];
cx node[108],node[112];
sx node[119];
rz(1.5*pi) node[76];
sx node[84];
rz(1.5*pi) node[86];
cx node[90],node[94];
sx node[98];
rz(1.5*pi) node[100];
rz(1.5*pi) node[102];
sx node[104];
rz(1.5*pi) node[106];
cx node[112],node[108];
rz(1.5*pi) node[119];
cx node[76],node[77];
rz(1.5*pi) node[84];
cx node[86],node[87];
cx node[94],node[90];
rz(1.5*pi) node[98];
cx node[102],node[101];
rz(1.5*pi) node[104];
cx node[106],node[105];
cx node[108],node[107];
cx node[119],node[120];
cx node[75],node[76];
sx node[77];
cx node[84],node[83];
sx node[87];
cx node[90],node[94];
cx node[98],node[97];
cx node[100],node[101];
cx node[104],node[105];
cx node[107],node[108];
cx node[118],node[119];
cx node[121],node[120];
cx node[76],node[75];
rz(2.5*pi) node[77];
sx node[83];
rz(2.5*pi) node[87];
cx node[94],node[95];
sx node[97];
sx node[101];
sx node[105];
cx node[108],node[107];
cx node[119],node[118];
sx node[120];
cx node[121],node[122];
cx node[75],node[76];
sx node[77];
rz(2.5*pi) node[83];
sx node[87];
cx node[95],node[94];
rz(2.5*pi) node[97];
rz(2.5*pi) node[101];
rz(2.5*pi) node[105];
cx node[107],node[106];
cx node[118],node[119];
rz(2.5*pi) node[120];
cx node[123],node[122];
rz(1.5*pi) node[77];
sx node[83];
rz(1.5*pi) node[87];
cx node[94],node[95];
sx node[97];
sx node[101];
sx node[105];
cx node[106],node[107];
cx node[110],node[118];
sx node[120];
sx node[122];
cx node[123],node[124];
cx node[77],node[78];
rz(1.5*pi) node[83];
cx node[87],node[86];
cx node[90],node[94];
rz(1.5*pi) node[97];
rz(1.5*pi) node[101];
rz(1.5*pi) node[105];
cx node[107],node[106];
cx node[118],node[110];
rz(1.5*pi) node[120];
rz(2.5*pi) node[122];
sx node[124];
cx node[76],node[77];
sx node[78];
cx node[83],node[82];
cx node[86],node[87];
cx node[94],node[90];
cx node[97],node[96];
cx node[106],node[105];
cx node[110],node[118];
cx node[119],node[120];
sx node[122];
rz(2.5*pi) node[124];
cx node[77],node[76];
rz(2.5*pi) node[78];
sx node[82];
cx node[87],node[86];
cx node[90],node[94];
sx node[96];
cx node[105],node[106];
cx node[120],node[119];
rz(1.5*pi) node[122];
sx node[124];
cx node[76],node[77];
sx node[78];
rz(2.5*pi) node[82];
cx node[86],node[85];
rz(2.5*pi) node[96];
cx node[106],node[105];
cx node[119],node[120];
rz(1.5*pi) node[124];
rz(1.5*pi) node[78];
sx node[82];
cx node[85],node[86];
cx node[93],node[106];
sx node[96];
cx node[105],node[104];
cx node[118],node[119];
cx node[120],node[121];
cx node[124],node[125];
cx node[78],node[79];
rz(1.5*pi) node[82];
cx node[86],node[85];
cx node[106],node[93];
rz(1.5*pi) node[96];
cx node[104],node[105];
cx node[119],node[118];
cx node[121],node[120];
sx node[125];
cx node[77],node[78];
cx node[82],node[81];
cx node[85],node[84];
cx node[93],node[106];
cx node[95],node[96];
cx node[105],node[104];
cx node[118],node[119];
cx node[120],node[121];
rz(2.5*pi) node[125];
cx node[78],node[77];
sx node[81];
cx node[84],node[85];
cx node[96],node[95];
cx node[104],node[103];
cx node[106],node[105];
cx node[119],node[120];
cx node[121],node[122];
sx node[125];
cx node[77],node[78];
rz(2.5*pi) node[81];
cx node[85],node[84];
cx node[95],node[96];
cx node[103],node[104];
cx node[105],node[106];
cx node[120],node[119];
cx node[122],node[121];
rz(1.5*pi) node[125];
sx node[81];
cx node[84],node[83];
cx node[104],node[103];
cx node[106],node[105];
cx node[119],node[120];
cx node[121],node[122];
rz(1.5*pi) node[81];
cx node[83],node[84];
cx node[103],node[102];
cx node[105],node[104];
cx node[120],node[121];
cx node[122],node[123];
cx node[81],node[80];
cx node[84],node[83];
cx node[102],node[103];
cx node[104],node[105];
cx node[121],node[120];
cx node[123],node[122];
sx node[80];
cx node[83],node[82];
cx node[103],node[102];
cx node[105],node[104];
cx node[120],node[121];
cx node[122],node[123];
rz(2.5*pi) node[80];
cx node[82],node[83];
cx node[102],node[101];
cx node[104],node[103];
cx node[123],node[124];
sx node[80];
cx node[83],node[82];
cx node[101],node[102];
cx node[103],node[104];
cx node[124],node[123];
rz(1.5*pi) node[80];
cx node[82],node[81];
cx node[102],node[101];
cx node[104],node[103];
cx node[123],node[124];
cx node[80],node[79];
cx node[81],node[82];
cx node[101],node[100];
cx node[103],node[102];
cx node[124],node[125];
sx node[79];
cx node[82],node[81];
cx node[100],node[101];
cx node[102],node[103];
cx node[125],node[124];
rz(2.5*pi) node[79];
cx node[81],node[80];
cx node[101],node[100];
cx node[103],node[102];
cx node[124],node[125];
sx node[79];
cx node[80],node[81];
cx node[92],node[102];
cx node[100],node[99];
cx node[124],node[123];
cx node[125],node[126];
rz(1.5*pi) node[79];
cx node[81],node[80];
cx node[99],node[100];
sx node[102];
cx node[123],node[124];
cx node[126],node[125];
cx node[78],node[79];
cx node[100],node[99];
rz(2.5*pi) node[102];
cx node[124],node[123];
cx node[125],node[126];
cx node[79],node[78];
cx node[99],node[98];
sx node[102];
cx node[123],node[122];
cx node[126],node[125];
cx node[78],node[79];
cx node[98],node[99];
rz(1.5*pi) node[102];
cx node[122],node[123];
cx node[125],node[124];
cx node[80],node[79];
cx node[99],node[98];
cx node[102],node[101];
cx node[123],node[122];
cx node[124],node[125];
sx node[79];
cx node[98],node[97];
cx node[101],node[102];
cx node[122],node[121];
cx node[125],node[124];
rz(2.5*pi) node[79];
cx node[97],node[98];
cx node[102],node[101];
cx node[121],node[122];
sx node[79];
cx node[92],node[102];
cx node[98],node[97];
cx node[101],node[100];
cx node[122],node[121];
rz(1.5*pi) node[79];
cx node[102],node[92];
cx node[100],node[101];
cx node[111],node[122];
cx node[121],node[120];
cx node[79],node[91];
cx node[92],node[102];
cx node[101],node[100];
cx node[120],node[121];
sx node[122];
cx node[100],node[99];
cx node[102],node[101];
cx node[121],node[120];
rz(2.5*pi) node[122];
cx node[99],node[100];
cx node[101],node[102];
cx node[120],node[119];
sx node[122];
cx node[100],node[99];
cx node[102],node[101];
cx node[119],node[120];
rz(1.5*pi) node[122];
cx node[99],node[98];
cx node[120],node[119];
cx node[122],node[123];
cx node[98],node[99];
cx node[119],node[118];
cx node[123],node[122];
cx node[99],node[98];
cx node[118],node[119];
cx node[122],node[123];
cx node[98],node[97];
cx node[119],node[118];
cx node[124],node[123];
cx node[97],node[98];
cx node[118],node[110];
cx node[123],node[124];
cx node[98],node[97];
cx node[110],node[118];
cx node[124],node[123];
cx node[97],node[96];
cx node[118],node[110];
cx node[123],node[122];
cx node[124],node[125];
cx node[96],node[97];
cx node[110],node[100];
cx node[122],node[123];
cx node[125],node[124];
cx node[97],node[96];
cx node[100],node[110];
cx node[123],node[122];
cx node[124],node[125];
cx node[96],node[95];
cx node[97],node[98];
cx node[110],node[100];
cx node[111],node[122];
cx node[125],node[126];
cx node[95],node[94];
cx node[100],node[99];
cx node[126],node[112];
sx node[122];
cx node[96],node[95];
cx node[99],node[100];
rz(2.5*pi) node[122];
cx node[125],node[126];
cx node[95],node[94];
cx node[100],node[99];
cx node[126],node[112];
sx node[122];
sx node[94];
cx node[99],node[98];
cx node[101],node[100];
sx node[112];
rz(1.5*pi) node[122];
rz(2.5*pi) node[94];
sx node[98];
cx node[100],node[101];
rz(2.5*pi) node[112];
sx node[94];
rz(2.5*pi) node[98];
cx node[101],node[100];
sx node[112];
rz(1.5*pi) node[94];
sx node[98];
cx node[100],node[99];
rz(1.5*pi) node[112];
cx node[94],node[90];
rz(1.5*pi) node[98];
cx node[99],node[100];
cx node[95],node[94];
cx node[100],node[99];
cx node[94],node[95];
cx node[99],node[98];
cx node[95],node[94];
cx node[98],node[99];
cx node[94],node[90];
cx node[99],node[98];
sx node[90];
cx node[98],node[91];
rz(2.5*pi) node[90];
sx node[91];
sx node[90];
rz(2.5*pi) node[91];
rz(1.5*pi) node[90];
sx node[91];
rz(1.5*pi) node[91];
barrier node[97],node[75],node[86],node[87],node[108],node[107],node[126],node[118],node[120],node[119],node[105],node[104],node[80],node[79],node[123],node[121],node[111],node[125],node[102],node[92],node[124],node[100],node[110],node[76],node[77],node[85],node[84],node[83],node[122],node[98],node[96],node[82],node[81],node[112],node[99],node[95],node[106],node[93],node[101],node[94],node[78],node[91],node[90],node[103];
measure node[97] -> meas[0];
measure node[75] -> meas[1];
measure node[86] -> meas[2];
measure node[87] -> meas[3];
measure node[108] -> meas[4];
measure node[107] -> meas[5];
measure node[126] -> meas[6];
measure node[118] -> meas[7];
measure node[120] -> meas[8];
measure node[119] -> meas[9];
measure node[105] -> meas[10];
measure node[104] -> meas[11];
measure node[80] -> meas[12];
measure node[79] -> meas[13];
measure node[123] -> meas[14];
measure node[121] -> meas[15];
measure node[111] -> meas[16];
measure node[125] -> meas[17];
measure node[102] -> meas[18];
measure node[92] -> meas[19];
measure node[124] -> meas[20];
measure node[100] -> meas[21];
measure node[110] -> meas[22];
measure node[76] -> meas[23];
measure node[77] -> meas[24];
measure node[85] -> meas[25];
measure node[84] -> meas[26];
measure node[83] -> meas[27];
measure node[122] -> meas[28];
measure node[98] -> meas[29];
measure node[96] -> meas[30];
measure node[82] -> meas[31];
measure node[81] -> meas[32];
measure node[112] -> meas[33];
measure node[99] -> meas[34];
measure node[95] -> meas[35];
measure node[106] -> meas[36];
measure node[93] -> meas[37];
measure node[101] -> meas[38];
measure node[94] -> meas[39];
measure node[78] -> meas[40];
measure node[91] -> meas[41];
measure node[90] -> meas[42];
measure node[103] -> meas[43];
