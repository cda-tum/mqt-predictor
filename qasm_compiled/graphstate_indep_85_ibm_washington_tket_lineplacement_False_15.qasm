OPENQASM 2.0;
include "qelib1.inc";

qreg node[127];
creg meas[85];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
sx node[7];
sx node[8];
sx node[14];
sx node[16];
sx node[18];
sx node[19];
sx node[20];
sx node[21];
sx node[22];
sx node[23];
sx node[24];
sx node[25];
sx node[26];
sx node[27];
sx node[28];
sx node[29];
sx node[30];
sx node[31];
sx node[32];
sx node[36];
sx node[37];
sx node[38];
sx node[39];
sx node[40];
sx node[41];
sx node[42];
sx node[43];
sx node[44];
sx node[45];
sx node[46];
sx node[47];
sx node[48];
sx node[49];
sx node[50];
sx node[51];
sx node[52];
sx node[56];
sx node[57];
sx node[58];
sx node[62];
sx node[63];
sx node[64];
sx node[65];
sx node[66];
sx node[67];
sx node[68];
sx node[72];
sx node[79];
sx node[80];
sx node[81];
sx node[83];
sx node[84];
sx node[85];
sx node[86];
sx node[87];
sx node[91];
sx node[92];
sx node[93];
sx node[98];
sx node[99];
sx node[100];
sx node[102];
sx node[103];
sx node[104];
sx node[106];
sx node[107];
sx node[108];
sx node[110];
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
rz(0.5*pi) node[5];
rz(0.5*pi) node[14];
rz(0.5*pi) node[19];
rz(0.5*pi) node[25];
rz(0.5*pi) node[28];
rz(0.5*pi) node[32];
rz(0.5*pi) node[39];
rz(0.5*pi) node[46];
rz(0.5*pi) node[48];
rz(0.5*pi) node[51];
rz(0.5*pi) node[52];
rz(0.5*pi) node[58];
rz(0.5*pi) node[62];
rz(0.5*pi) node[67];
rz(0.5*pi) node[80];
rz(0.5*pi) node[84];
rz(0.5*pi) node[93];
rz(0.5*pi) node[100];
rz(0.5*pi) node[103];
rz(0.5*pi) node[121];
rz(0.5*pi) node[125];
sx node[5];
sx node[14];
sx node[19];
sx node[25];
sx node[28];
sx node[32];
sx node[39];
sx node[46];
sx node[48];
sx node[51];
sx node[52];
sx node[58];
sx node[62];
sx node[67];
sx node[80];
sx node[84];
sx node[93];
sx node[100];
sx node[103];
sx node[121];
sx node[125];
cx node[5],node[4];
cx node[19],node[20];
cx node[25],node[26];
cx node[28],node[29];
cx node[32],node[31];
cx node[52],node[37];
cx node[39],node[40];
cx node[46],node[45];
cx node[48],node[49];
cx node[58],node[57];
cx node[62],node[63];
cx node[67],node[66];
cx node[80],node[79];
cx node[84],node[83];
cx node[93],node[106];
cx node[100],node[110];
cx node[121],node[120];
cx node[125],node[124];
sx node[4];
cx node[5],node[6];
cx node[19],node[18];
sx node[20];
cx node[25],node[24];
sx node[26];
sx node[29];
sx node[31];
sx node[37];
sx node[40];
sx node[45];
cx node[48],node[47];
sx node[49];
cx node[52],node[56];
sx node[57];
cx node[58],node[59];
cx node[62],node[72];
sx node[63];
sx node[66];
cx node[67],node[68];
sx node[79];
cx node[80],node[81];
sx node[83];
cx node[84],node[85];
cx node[93],node[87];
sx node[106];
sx node[110];
sx node[120];
cx node[121],node[122];
sx node[124];
cx node[125],node[126];
rz(2.5*pi) node[4];
sx node[6];
cx node[14],node[18];
rz(2.5*pi) node[20];
sx node[24];
rz(2.5*pi) node[26];
rz(2.5*pi) node[29];
rz(2.5*pi) node[31];
rz(2.5*pi) node[37];
rz(2.5*pi) node[40];
rz(2.5*pi) node[45];
cx node[46],node[47];
rz(2.5*pi) node[49];
rz(2.5*pi) node[57];
cx node[59],node[58];
rz(2.5*pi) node[63];
rz(2.5*pi) node[66];
sx node[68];
rz(2.5*pi) node[79];
sx node[81];
rz(2.5*pi) node[83];
sx node[85];
rz(2.5*pi) node[106];
rz(2.5*pi) node[110];
rz(2.5*pi) node[120];
sx node[122];
rz(2.5*pi) node[124];
sx node[126];
sx node[4];
rz(2.5*pi) node[6];
sx node[18];
sx node[20];
rz(2.5*pi) node[24];
sx node[26];
sx node[29];
sx node[31];
sx node[37];
sx node[40];
sx node[45];
sx node[47];
sx node[49];
sx node[57];
cx node[58],node[59];
sx node[63];
sx node[66];
rz(2.5*pi) node[68];
sx node[79];
rz(2.5*pi) node[81];
sx node[83];
rz(2.5*pi) node[85];
sx node[106];
sx node[110];
sx node[120];
rz(2.5*pi) node[122];
sx node[124];
rz(2.5*pi) node[126];
rz(1.5*pi) node[4];
sx node[6];
rz(2.5*pi) node[18];
rz(1.5*pi) node[20];
sx node[24];
rz(1.5*pi) node[26];
rz(1.5*pi) node[29];
rz(1.5*pi) node[31];
rz(1.5*pi) node[37];
rz(1.5*pi) node[40];
rz(1.5*pi) node[45];
rz(2.5*pi) node[47];
rz(1.5*pi) node[49];
rz(1.5*pi) node[57];
cx node[59],node[60];
rz(1.5*pi) node[63];
rz(1.5*pi) node[66];
sx node[68];
rz(1.5*pi) node[79];
sx node[81];
rz(1.5*pi) node[83];
sx node[85];
rz(1.5*pi) node[106];
rz(1.5*pi) node[110];
rz(1.5*pi) node[120];
sx node[122];
rz(1.5*pi) node[124];
sx node[126];
cx node[4],node[3];
rz(1.5*pi) node[6];
sx node[18];
cx node[20],node[21];
rz(1.5*pi) node[24];
cx node[26],node[27];
cx node[31],node[30];
cx node[37],node[38];
cx node[40],node[41];
cx node[45],node[44];
sx node[47];
cx node[49],node[50];
cx node[57],node[56];
cx node[60],node[59];
cx node[63],node[64];
cx node[66],node[65];
rz(1.5*pi) node[68];
cx node[79],node[91];
rz(1.5*pi) node[81];
cx node[83],node[92];
rz(1.5*pi) node[85];
cx node[106],node[107];
cx node[110],node[118];
cx node[120],node[119];
rz(1.5*pi) node[122];
cx node[124],node[123];
rz(1.5*pi) node[126];
sx node[3];
cx node[6],node[7];
rz(1.5*pi) node[18];
cx node[24],node[23];
cx node[28],node[27];
cx node[29],node[30];
cx node[39],node[38];
sx node[41];
sx node[44];
rz(1.5*pi) node[47];
cx node[51],node[50];
cx node[68],node[55];
sx node[56];
cx node[59],node[60];
sx node[65];
cx node[81],node[72];
cx node[85],node[86];
sx node[91];
sx node[92];
cx node[126],node[112];
sx node[118];
cx node[122],node[123];
rz(2.5*pi) node[3];
sx node[7];
sx node[23];
sx node[27];
sx node[30];
cx node[51],node[36];
sx node[38];
rz(2.5*pi) node[41];
rz(2.5*pi) node[44];
sx node[50];
cx node[60],node[53];
cx node[55],node[68];
rz(2.5*pi) node[56];
rz(2.5*pi) node[65];
sx node[72];
sx node[86];
rz(2.5*pi) node[91];
rz(2.5*pi) node[92];
sx node[112];
rz(2.5*pi) node[118];
sx node[123];
sx node[3];
rz(2.5*pi) node[7];
rz(2.5*pi) node[23];
rz(2.5*pi) node[27];
rz(2.5*pi) node[30];
cx node[32],node[36];
rz(2.5*pi) node[38];
sx node[41];
sx node[44];
rz(2.5*pi) node[50];
cx node[53],node[60];
cx node[68],node[55];
sx node[56];
sx node[65];
rz(2.5*pi) node[72];
rz(2.5*pi) node[86];
sx node[91];
sx node[92];
rz(2.5*pi) node[112];
sx node[118];
rz(2.5*pi) node[123];
rz(1.5*pi) node[3];
sx node[7];
sx node[23];
sx node[27];
sx node[30];
sx node[36];
sx node[38];
rz(1.5*pi) node[41];
rz(1.5*pi) node[44];
cx node[55],node[49];
sx node[50];
cx node[60],node[53];
rz(1.5*pi) node[56];
rz(1.5*pi) node[65];
sx node[72];
sx node[86];
rz(1.5*pi) node[91];
rz(1.5*pi) node[92];
sx node[112];
rz(1.5*pi) node[118];
sx node[123];
cx node[3],node[2];
rz(1.5*pi) node[7];
rz(1.5*pi) node[23];
rz(1.5*pi) node[27];
rz(1.5*pi) node[30];
rz(2.5*pi) node[36];
rz(1.5*pi) node[38];
cx node[41],node[42];
cx node[49],node[55];
rz(1.5*pi) node[50];
cx node[65],node[64];
rz(1.5*pi) node[72];
rz(1.5*pi) node[86];
cx node[91],node[98];
cx node[92],node[102];
rz(1.5*pi) node[112];
cx node[118],node[119];
rz(1.5*pi) node[123];
sx node[2];
cx node[7],node[8];
cx node[23],node[22];
sx node[36];
cx node[53],node[41];
sx node[42];
cx node[55],node[49];
sx node[64];
cx node[86],node[87];
sx node[98];
cx node[103],node[102];
cx node[112],node[108];
sx node[119];
rz(2.5*pi) node[2];
sx node[8];
sx node[22];
rz(1.5*pi) node[36];
cx node[41],node[53];
rz(2.5*pi) node[42];
cx node[49],node[48];
rz(2.5*pi) node[64];
sx node[87];
rz(2.5*pi) node[98];
sx node[102];
cx node[103],node[104];
sx node[108];
rz(2.5*pi) node[119];
sx node[2];
rz(2.5*pi) node[8];
rz(2.5*pi) node[22];
cx node[53],node[41];
sx node[42];
cx node[48],node[49];
sx node[64];
rz(2.5*pi) node[87];
sx node[98];
rz(2.5*pi) node[102];
cx node[104],node[103];
rz(2.5*pi) node[108];
sx node[119];
rz(1.5*pi) node[2];
sx node[8];
sx node[22];
cx node[41],node[40];
rz(1.5*pi) node[42];
cx node[49],node[48];
rz(1.5*pi) node[64];
sx node[87];
rz(1.5*pi) node[98];
sx node[102];
cx node[103],node[104];
sx node[108];
rz(1.5*pi) node[119];
cx node[2],node[1];
rz(1.5*pi) node[8];
rz(1.5*pi) node[22];
cx node[40],node[41];
cx node[42],node[43];
cx node[48],node[47];
rz(1.5*pi) node[87];
cx node[98],node[99];
rz(1.5*pi) node[102];
cx node[104],node[103];
rz(1.5*pi) node[108];
sx node[1];
cx node[8],node[16];
cx node[22],node[21];
cx node[41],node[40];
cx node[44],node[43];
cx node[47],node[48];
cx node[100],node[99];
cx node[103],node[102];
cx node[108],node[107];
rz(2.5*pi) node[1];
sx node[21];
cx node[40],node[39];
sx node[43];
cx node[48],node[47];
sx node[99];
cx node[102],node[103];
sx node[107];
sx node[1];
rz(2.5*pi) node[21];
cx node[47],node[35];
cx node[39],node[40];
rz(2.5*pi) node[43];
rz(2.5*pi) node[99];
cx node[103],node[102];
rz(2.5*pi) node[107];
rz(1.5*pi) node[1];
sx node[21];
cx node[35],node[47];
cx node[40],node[39];
sx node[43];
cx node[102],node[92];
sx node[99];
sx node[107];
cx node[1],node[0];
rz(1.5*pi) node[21];
cx node[39],node[33];
cx node[47],node[35];
rz(1.5*pi) node[43];
cx node[92],node[102];
rz(1.5*pi) node[99];
rz(1.5*pi) node[107];
cx node[35],node[28];
cx node[33],node[39];
cx node[102],node[92];
cx node[28],node[35];
cx node[39],node[33];
cx node[92],node[83];
cx node[33],node[20];
cx node[35],node[28];
cx node[83],node[92];
cx node[20],node[33];
cx node[28],node[27];
cx node[92],node[83];
cx node[33],node[20];
cx node[27],node[28];
cx node[83],node[82];
cx node[20],node[19];
cx node[28],node[27];
cx node[82],node[83];
cx node[19],node[20];
cx node[27],node[26];
cx node[83],node[82];
cx node[20],node[19];
cx node[26],node[27];
cx node[82],node[81];
cx node[19],node[18];
cx node[27],node[26];
cx node[81],node[82];
cx node[26],node[16];
cx node[18],node[19];
cx node[82],node[81];
sx node[16];
cx node[19],node[18];
cx node[81],node[72];
cx node[18],node[14];
rz(2.5*pi) node[16];
cx node[72],node[81];
cx node[14],node[18];
sx node[16];
cx node[81],node[72];
cx node[18],node[14];
rz(1.5*pi) node[16];
cx node[72],node[62];
cx node[14],node[0];
cx node[62],node[72];
sx node[0];
cx node[72],node[62];
rz(2.5*pi) node[0];
cx node[62],node[61];
sx node[0];
cx node[61],node[62];
rz(1.5*pi) node[0];
cx node[62],node[61];
cx node[61],node[60];
cx node[60],node[61];
cx node[61],node[60];
cx node[60],node[53];
cx node[53],node[60];
cx node[60],node[53];
cx node[53],node[41];
cx node[41],node[53];
cx node[53],node[41];
cx node[41],node[40];
cx node[40],node[41];
cx node[41],node[40];
cx node[40],node[39];
cx node[39],node[40];
cx node[40],node[39];
cx node[39],node[33];
cx node[33],node[39];
cx node[39],node[33];
cx node[33],node[20];
cx node[20],node[33];
cx node[33],node[20];
cx node[20],node[19];
cx node[19],node[20];
cx node[20],node[19];
cx node[18],node[19];
sx node[19];
rz(2.5*pi) node[19];
sx node[19];
rz(1.5*pi) node[19];
barrier node[72],node[63],node[49],node[55],node[51],node[50],node[5],node[4],node[6],node[32],node[31],node[52],node[37],node[67],node[66],node[80],node[79],node[25],node[27],node[3],node[84],node[92],node[93],node[106],node[85],node[36],node[102],node[2],node[1],node[86],node[87],node[125],node[124],node[14],node[57],node[35],node[29],node[121],node[120],node[126],node[112],node[82],node[65],node[56],node[41],node[53],node[91],node[98],node[104],node[103],node[24],node[23],node[100],node[110],node[33],node[39],node[46],node[45],node[60],node[42],node[48],node[122],node[99],node[7],node[8],node[64],node[38],node[0],node[81],node[18],node[20],node[22],node[44],node[26],node[118],node[28],node[30],node[16],node[43],node[21],node[19],node[123],node[108],node[107],node[119];
measure node[72] -> meas[0];
measure node[63] -> meas[1];
measure node[49] -> meas[2];
measure node[55] -> meas[3];
measure node[51] -> meas[4];
measure node[50] -> meas[5];
measure node[5] -> meas[6];
measure node[4] -> meas[7];
measure node[6] -> meas[8];
measure node[32] -> meas[9];
measure node[31] -> meas[10];
measure node[52] -> meas[11];
measure node[37] -> meas[12];
measure node[67] -> meas[13];
measure node[66] -> meas[14];
measure node[80] -> meas[15];
measure node[79] -> meas[16];
measure node[25] -> meas[17];
measure node[27] -> meas[18];
measure node[3] -> meas[19];
measure node[84] -> meas[20];
measure node[92] -> meas[21];
measure node[93] -> meas[22];
measure node[106] -> meas[23];
measure node[85] -> meas[24];
measure node[36] -> meas[25];
measure node[102] -> meas[26];
measure node[2] -> meas[27];
measure node[1] -> meas[28];
measure node[86] -> meas[29];
measure node[87] -> meas[30];
measure node[125] -> meas[31];
measure node[124] -> meas[32];
measure node[14] -> meas[33];
measure node[57] -> meas[34];
measure node[35] -> meas[35];
measure node[29] -> meas[36];
measure node[121] -> meas[37];
measure node[120] -> meas[38];
measure node[126] -> meas[39];
measure node[112] -> meas[40];
measure node[82] -> meas[41];
measure node[65] -> meas[42];
measure node[56] -> meas[43];
measure node[41] -> meas[44];
measure node[53] -> meas[45];
measure node[91] -> meas[46];
measure node[98] -> meas[47];
measure node[104] -> meas[48];
measure node[103] -> meas[49];
measure node[24] -> meas[50];
measure node[23] -> meas[51];
measure node[100] -> meas[52];
measure node[110] -> meas[53];
measure node[33] -> meas[54];
measure node[39] -> meas[55];
measure node[46] -> meas[56];
measure node[45] -> meas[57];
measure node[60] -> meas[58];
measure node[42] -> meas[59];
measure node[48] -> meas[60];
measure node[122] -> meas[61];
measure node[99] -> meas[62];
measure node[7] -> meas[63];
measure node[8] -> meas[64];
measure node[64] -> meas[65];
measure node[38] -> meas[66];
measure node[0] -> meas[67];
measure node[81] -> meas[68];
measure node[18] -> meas[69];
measure node[20] -> meas[70];
measure node[22] -> meas[71];
measure node[44] -> meas[72];
measure node[26] -> meas[73];
measure node[118] -> meas[74];
measure node[28] -> meas[75];
measure node[30] -> meas[76];
measure node[16] -> meas[77];
measure node[43] -> meas[78];
measure node[21] -> meas[79];
measure node[19] -> meas[80];
measure node[123] -> meas[81];
measure node[108] -> meas[82];
measure node[107] -> meas[83];
measure node[119] -> meas[84];
