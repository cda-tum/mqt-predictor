OPENQASM 2.0;
include "qelib1.inc";

qreg node[127];
creg meas[71];
sx node[107];
rz(0.5*pi) node[107];
sx node[107];
cx node[107],node[108];
cx node[108],node[112];
cx node[112],node[126];
cx node[126],node[125];
cx node[125],node[124];
cx node[124],node[123];
cx node[123],node[122];
cx node[122],node[111];
cx node[111],node[104];
cx node[104],node[105];
cx node[105],node[106];
cx node[106],node[93];
cx node[93],node[87];
cx node[87],node[86];
cx node[86],node[85];
cx node[85],node[73];
cx node[73],node[66];
cx node[66],node[65];
cx node[65],node[64];
cx node[64],node[54];
cx node[54],node[45];
cx node[45],node[44];
cx node[44],node[43];
cx node[43],node[34];
cx node[34],node[24];
cx node[24],node[23];
cx node[23],node[22];
cx node[22],node[15];
cx node[15],node[4];
cx node[4],node[3];
cx node[3],node[2];
cx node[2],node[1];
cx node[1],node[0];
cx node[0],node[14];
cx node[14],node[18];
cx node[18],node[19];
cx node[19],node[20];
cx node[20],node[21];
cx node[33],node[20];
cx node[20],node[33];
cx node[33],node[20];
cx node[21],node[20];
cx node[39],node[33];
cx node[33],node[39];
cx node[39],node[33];
cx node[20],node[33];
cx node[40],node[39];
cx node[39],node[40];
cx node[40],node[39];
cx node[33],node[39];
cx node[39],node[38];
cx node[38],node[37];
cx node[37],node[52];
cx node[52],node[56];
cx node[56],node[57];
cx node[57],node[58];
cx node[58],node[59];
cx node[59],node[60];
cx node[60],node[53];
cx node[53],node[41];
cx node[41],node[42];
cx node[42],node[43];
cx node[43],node[42];
cx node[42],node[43];
cx node[43],node[34];
cx node[34],node[43];
cx node[43],node[34];
cx node[34],node[24];
cx node[24],node[34];
cx node[34],node[24];
cx node[24],node[25];
cx node[25],node[26];
cx node[26],node[16];
cx node[16],node[8];
cx node[27],node[26];
cx node[8],node[7];
cx node[26],node[27];
cx node[7],node[6];
cx node[27],node[26];
cx node[6],node[5];
cx node[26],node[16];
cx node[28],node[27];
cx node[16],node[26];
cx node[27],node[28];
cx node[26],node[16];
cx node[28],node[27];
cx node[16],node[8];
cx node[27],node[26];
cx node[29],node[28];
cx node[8],node[16];
cx node[26],node[27];
cx node[28],node[29];
cx node[16],node[8];
cx node[27],node[26];
cx node[29],node[28];
cx node[8],node[7];
cx node[26],node[16];
cx node[28],node[27];
cx node[30],node[29];
cx node[7],node[8];
cx node[16],node[26];
cx node[27],node[28];
cx node[29],node[30];
cx node[8],node[7];
cx node[26],node[16];
cx node[28],node[27];
cx node[30],node[29];
cx node[7],node[6];
cx node[16],node[8];
cx node[27],node[26];
cx node[35],node[28];
cx node[6],node[7];
cx node[8],node[16];
cx node[26],node[27];
cx node[28],node[35];
cx node[7],node[6];
cx node[16],node[8];
cx node[27],node[26];
cx node[35],node[28];
cx node[5],node[6];
cx node[8],node[7];
cx node[26],node[16];
cx node[28],node[27];
cx node[47],node[35];
cx node[7],node[8];
cx node[16],node[26];
cx node[27],node[28];
cx node[35],node[47];
cx node[8],node[7];
cx node[26],node[16];
cx node[28],node[27];
cx node[47],node[35];
cx node[6],node[7];
cx node[16],node[8];
cx node[27],node[26];
cx node[29],node[28];
cx node[46],node[47];
cx node[8],node[16];
cx node[26],node[27];
cx node[28],node[29];
cx node[47],node[46];
cx node[16],node[8];
cx node[27],node[26];
cx node[29],node[28];
cx node[46],node[47];
cx node[7],node[8];
cx node[26],node[16];
cx node[28],node[27];
cx node[16],node[26];
cx node[27],node[28];
cx node[26],node[16];
cx node[28],node[27];
cx node[8],node[16];
cx node[27],node[26];
cx node[35],node[28];
cx node[26],node[27];
cx node[28],node[35];
cx node[27],node[26];
cx node[35],node[28];
cx node[16],node[26];
cx node[28],node[27];
cx node[47],node[35];
cx node[27],node[28];
cx node[35],node[47];
cx node[28],node[27];
cx node[47],node[35];
cx node[26],node[27];
cx node[35],node[28];
cx node[28],node[35];
cx node[35],node[28];
cx node[27],node[28];
cx node[28],node[29];
cx node[29],node[28];
cx node[28],node[29];
cx node[29],node[30];
cx node[30],node[17];
cx node[29],node[30];
cx node[30],node[17];
cx node[17],node[12];
cx node[12],node[11];
cx node[11],node[10];
barrier node[10],node[11],node[12],node[17],node[29],node[27],node[26],node[16],node[8],node[7],node[6],node[5],node[35],node[47],node[28],node[46],node[30],node[25],node[24],node[41],node[53],node[60],node[59],node[58],node[57],node[56],node[52],node[37],node[38],node[39],node[33],node[20],node[21],node[40],node[19],node[18],node[14],node[0],node[1],node[2],node[3],node[4],node[15],node[22],node[23],node[34],node[43],node[42],node[44],node[45],node[54],node[64],node[65],node[66],node[73],node[85],node[86],node[87],node[93],node[106],node[105],node[104],node[111],node[122],node[123],node[124],node[125],node[126],node[112],node[108],node[107];
measure node[10] -> meas[0];
measure node[11] -> meas[1];
measure node[12] -> meas[2];
measure node[17] -> meas[3];
measure node[29] -> meas[4];
measure node[27] -> meas[5];
measure node[26] -> meas[6];
measure node[16] -> meas[7];
measure node[8] -> meas[8];
measure node[7] -> meas[9];
measure node[6] -> meas[10];
measure node[5] -> meas[11];
measure node[35] -> meas[12];
measure node[47] -> meas[13];
measure node[28] -> meas[14];
measure node[46] -> meas[15];
measure node[30] -> meas[16];
measure node[25] -> meas[17];
measure node[24] -> meas[18];
measure node[41] -> meas[19];
measure node[53] -> meas[20];
measure node[60] -> meas[21];
measure node[59] -> meas[22];
measure node[58] -> meas[23];
measure node[57] -> meas[24];
measure node[56] -> meas[25];
measure node[52] -> meas[26];
measure node[37] -> meas[27];
measure node[38] -> meas[28];
measure node[39] -> meas[29];
measure node[33] -> meas[30];
measure node[20] -> meas[31];
measure node[21] -> meas[32];
measure node[40] -> meas[33];
measure node[19] -> meas[34];
measure node[18] -> meas[35];
measure node[14] -> meas[36];
measure node[0] -> meas[37];
measure node[1] -> meas[38];
measure node[2] -> meas[39];
measure node[3] -> meas[40];
measure node[4] -> meas[41];
measure node[15] -> meas[42];
measure node[22] -> meas[43];
measure node[23] -> meas[44];
measure node[34] -> meas[45];
measure node[43] -> meas[46];
measure node[42] -> meas[47];
measure node[44] -> meas[48];
measure node[45] -> meas[49];
measure node[54] -> meas[50];
measure node[64] -> meas[51];
measure node[65] -> meas[52];
measure node[66] -> meas[53];
measure node[73] -> meas[54];
measure node[85] -> meas[55];
measure node[86] -> meas[56];
measure node[87] -> meas[57];
measure node[93] -> meas[58];
measure node[106] -> meas[59];
measure node[105] -> meas[60];
measure node[104] -> meas[61];
measure node[111] -> meas[62];
measure node[122] -> meas[63];
measure node[123] -> meas[64];
measure node[124] -> meas[65];
measure node[125] -> meas[66];
measure node[126] -> meas[67];
measure node[112] -> meas[68];
measure node[108] -> meas[69];
measure node[107] -> meas[70];
