OPENQASM 2.0;
include "qelib1.inc";

qreg node[86];
creg meas[79];
sx node[5];
rz(0.5*pi) node[5];
sx node[5];
cx node[5],node[4];
cx node[4],node[3];
cx node[3],node[2];
cx node[2],node[1];
cx node[1],node[0];
cx node[0],node[14];
cx node[14],node[18];
cx node[18],node[19];
cx node[19],node[20];
cx node[20],node[21];
cx node[21],node[22];
cx node[22],node[15];
cx node[23],node[22];
cx node[22],node[23];
cx node[23],node[22];
cx node[15],node[22];
cx node[24],node[23];
cx node[23],node[24];
cx node[24],node[23];
cx node[22],node[23];
cx node[25],node[24];
cx node[24],node[25];
cx node[25],node[24];
cx node[23],node[24];
cx node[24],node[34];
cx node[34],node[43];
cx node[43],node[42];
cx node[42],node[41];
cx node[41],node[40];
cx node[40],node[39];
cx node[39],node[33];
cx node[33],node[39];
cx node[39],node[38];
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
cx node[61],node[60];
cx node[60],node[61];
cx node[61],node[60];
cx node[53],node[60];
cx node[62],node[61];
cx node[61],node[62];
cx node[62],node[61];
cx node[60],node[61];
cx node[63],node[62];
cx node[62],node[63];
cx node[63],node[62];
cx node[61],node[62];
cx node[62],node[72];
cx node[72],node[81];
cx node[81],node[82];
cx node[82],node[83];
cx node[83],node[84];
cx node[84],node[85];
cx node[85],node[73];
cx node[73],node[66];
cx node[66],node[65];
cx node[65],node[64];
cx node[64],node[54];
cx node[54],node[45];
cx node[45],node[44];
cx node[46],node[45];
cx node[45],node[46];
cx node[46],node[45];
cx node[44],node[45];
cx node[47],node[46];
cx node[46],node[47];
cx node[47],node[46];
cx node[45],node[46];
cx node[46],node[47];
cx node[47],node[35];
cx node[46],node[47];
cx node[47],node[35];
cx node[35],node[28];
cx node[48],node[47];
cx node[28],node[27];
cx node[47],node[48];
cx node[27],node[26];
cx node[29],node[28];
cx node[48],node[47];
cx node[26],node[16];
cx node[28],node[29];
cx node[47],node[35];
cx node[16],node[8];
cx node[29],node[28];
cx node[35],node[47];
cx node[8],node[7];
cx node[28],node[27];
cx node[30],node[29];
cx node[47],node[35];
cx node[7],node[6];
cx node[27],node[28];
cx node[29],node[30];
cx node[28],node[27];
cx node[30],node[29];
cx node[17],node[30];
cx node[27],node[26];
cx node[29],node[28];
cx node[30],node[17];
cx node[26],node[27];
cx node[28],node[29];
cx node[17],node[30];
cx node[27],node[26];
cx node[29],node[28];
cx node[26],node[16];
cx node[28],node[27];
cx node[30],node[29];
cx node[16],node[26];
cx node[27],node[28];
cx node[29],node[30];
cx node[26],node[16];
cx node[28],node[27];
cx node[30],node[29];
cx node[16],node[8];
cx node[27],node[26];
cx node[29],node[28];
cx node[31],node[30];
cx node[8],node[16];
cx node[26],node[27];
cx node[28],node[29];
cx node[30],node[31];
cx node[16],node[8];
cx node[27],node[26];
cx node[29],node[28];
cx node[31],node[30];
cx node[8],node[7];
cx node[26],node[16];
cx node[28],node[27];
cx node[30],node[29];
cx node[32],node[31];
cx node[7],node[8];
cx node[16],node[26];
cx node[27],node[28];
cx node[29],node[30];
cx node[31],node[32];
cx node[8],node[7];
cx node[26],node[16];
cx node[28],node[27];
cx node[30],node[29];
cx node[32],node[31];
cx node[6],node[7];
cx node[16],node[8];
cx node[27],node[26];
cx node[29],node[28];
cx node[36],node[32];
cx node[8],node[16];
cx node[26],node[27];
cx node[28],node[29];
cx node[32],node[36];
cx node[16],node[8];
cx node[27],node[26];
cx node[29],node[28];
cx node[36],node[32];
cx node[7],node[8];
cx node[26],node[16];
cx node[28],node[27];
cx node[51],node[36];
cx node[16],node[26];
cx node[27],node[28];
cx node[36],node[51];
cx node[26],node[16];
cx node[28],node[27];
cx node[51],node[36];
cx node[8],node[16];
cx node[27],node[26];
cx node[35],node[28];
cx node[50],node[51];
cx node[26],node[27];
cx node[28],node[35];
cx node[51],node[50];
cx node[27],node[26];
cx node[35],node[28];
cx node[50],node[51];
cx node[16],node[26];
cx node[28],node[27];
cx node[49],node[50];
cx node[27],node[28];
cx node[50],node[49];
cx node[28],node[27];
cx node[49],node[50];
cx node[26],node[27];
cx node[55],node[49];
cx node[27],node[28];
cx node[49],node[55];
cx node[28],node[27];
cx node[55],node[49];
cx node[27],node[28];
cx node[68],node[55];
cx node[28],node[29];
cx node[55],node[68];
cx node[29],node[28];
cx node[68],node[55];
cx node[28],node[29];
cx node[67],node[68];
cx node[29],node[30];
cx node[68],node[67];
cx node[30],node[29];
cx node[67],node[68];
cx node[29],node[30];
cx node[30],node[17];
cx node[17],node[30];
cx node[30],node[17];
cx node[17],node[12];
cx node[31],node[30];
cx node[12],node[11];
cx node[30],node[31];
cx node[11],node[10];
cx node[13],node[12];
cx node[31],node[30];
cx node[10],node[9];
cx node[12],node[13];
cx node[30],node[17];
cx node[32],node[31];
cx node[13],node[12];
cx node[17],node[30];
cx node[31],node[32];
cx node[12],node[11];
cx node[30],node[17];
cx node[32],node[31];
cx node[11],node[12];
cx node[31],node[30];
cx node[36],node[32];
cx node[12],node[11];
cx node[30],node[31];
cx node[32],node[36];
cx node[11],node[10];
cx node[17],node[12];
cx node[31],node[30];
cx node[36],node[32];
cx node[10],node[11];
cx node[12],node[17];
cx node[32],node[31];
cx node[51],node[36];
cx node[11],node[10];
cx node[17],node[12];
cx node[31],node[32];
cx node[36],node[51];
cx node[9],node[10];
cx node[12],node[11];
cx node[30],node[17];
cx node[32],node[31];
cx node[51],node[36];
cx node[11],node[12];
cx node[17],node[30];
cx node[36],node[32];
cx node[50],node[51];
cx node[12],node[11];
cx node[30],node[17];
cx node[32],node[36];
cx node[51],node[50];
cx node[10],node[11];
cx node[17],node[12];
cx node[31],node[30];
cx node[36],node[32];
cx node[50],node[51];
cx node[12],node[17];
cx node[30],node[31];
cx node[51],node[36];
cx node[49],node[50];
cx node[17],node[12];
cx node[31],node[30];
cx node[36],node[51];
cx node[50],node[49];
cx node[11],node[12];
cx node[30],node[17];
cx node[32],node[31];
cx node[51],node[36];
cx node[49],node[50];
cx node[17],node[30];
cx node[31],node[32];
cx node[55],node[49];
cx node[50],node[51];
cx node[30],node[17];
cx node[32],node[31];
cx node[49],node[55];
cx node[51],node[50];
cx node[12],node[17];
cx node[31],node[30];
cx node[36],node[32];
cx node[55],node[49];
cx node[50],node[51];
cx node[30],node[31];
cx node[32],node[36];
cx node[49],node[50];
cx node[68],node[55];
cx node[31],node[30];
cx node[36],node[32];
cx node[50],node[49];
cx node[55],node[68];
cx node[17],node[30];
cx node[32],node[31];
cx node[51],node[36];
cx node[49],node[50];
cx node[68],node[55];
cx node[31],node[32];
cx node[36],node[51];
cx node[55],node[49];
cx node[69],node[68];
cx node[32],node[31];
cx node[51],node[36];
cx node[49],node[55];
cx node[68],node[69];
cx node[30],node[31];
cx node[36],node[32];
cx node[55],node[49];
cx node[50],node[51];
cx node[69],node[68];
cx node[32],node[36];
cx node[51],node[50];
cx node[68],node[55];
cx node[70],node[69];
cx node[36],node[32];
cx node[50],node[51];
cx node[55],node[68];
cx node[69],node[70];
cx node[31],node[32];
cx node[51],node[36];
cx node[49],node[50];
cx node[68],node[55];
cx node[70],node[69];
cx node[36],node[51];
cx node[50],node[49];
cx node[69],node[68];
cx node[74],node[70];
cx node[51],node[36];
cx node[49],node[50];
cx node[68],node[69];
cx node[70],node[74];
cx node[32],node[36];
cx node[55],node[49];
cx node[69],node[68];
cx node[74],node[70];
cx node[36],node[51];
cx node[49],node[55];
cx node[70],node[69];
cx node[55],node[49];
cx node[51],node[50];
cx node[69],node[70];
cx node[36],node[51];
cx node[68],node[55];
cx node[70],node[69];
cx node[51],node[50];
cx node[55],node[68];
cx node[50],node[49];
cx node[68],node[55];
cx node[49],node[55];
cx node[69],node[68];
cx node[68],node[69];
cx node[69],node[68];
cx node[55],node[68];
barrier node[68],node[55],node[49],node[50],node[36],node[32],node[31],node[30],node[17],node[12],node[11],node[10],node[9],node[51],node[69],node[13],node[70],node[26],node[16],node[8],node[7],node[6],node[27],node[35],node[28],node[29],node[67],node[74],node[47],node[46],node[45],node[44],node[48],node[54],node[64],node[65],node[66],node[73],node[85],node[84],node[83],node[82],node[81],node[72],node[62],node[61],node[60],node[53],node[63],node[59],node[58],node[57],node[56],node[52],node[37],node[38],node[33],node[39],node[40],node[41],node[42],node[43],node[34],node[24],node[23],node[22],node[15],node[25],node[21],node[20],node[19],node[18],node[14],node[0],node[1],node[2],node[3],node[4],node[5];
measure node[68] -> meas[0];
measure node[55] -> meas[1];
measure node[49] -> meas[2];
measure node[50] -> meas[3];
measure node[36] -> meas[4];
measure node[32] -> meas[5];
measure node[31] -> meas[6];
measure node[30] -> meas[7];
measure node[17] -> meas[8];
measure node[12] -> meas[9];
measure node[11] -> meas[10];
measure node[10] -> meas[11];
measure node[9] -> meas[12];
measure node[51] -> meas[13];
measure node[69] -> meas[14];
measure node[13] -> meas[15];
measure node[70] -> meas[16];
measure node[26] -> meas[17];
measure node[16] -> meas[18];
measure node[8] -> meas[19];
measure node[7] -> meas[20];
measure node[6] -> meas[21];
measure node[27] -> meas[22];
measure node[35] -> meas[23];
measure node[28] -> meas[24];
measure node[29] -> meas[25];
measure node[67] -> meas[26];
measure node[74] -> meas[27];
measure node[47] -> meas[28];
measure node[46] -> meas[29];
measure node[45] -> meas[30];
measure node[44] -> meas[31];
measure node[48] -> meas[32];
measure node[54] -> meas[33];
measure node[64] -> meas[34];
measure node[65] -> meas[35];
measure node[66] -> meas[36];
measure node[73] -> meas[37];
measure node[85] -> meas[38];
measure node[84] -> meas[39];
measure node[83] -> meas[40];
measure node[82] -> meas[41];
measure node[81] -> meas[42];
measure node[72] -> meas[43];
measure node[62] -> meas[44];
measure node[61] -> meas[45];
measure node[60] -> meas[46];
measure node[53] -> meas[47];
measure node[63] -> meas[48];
measure node[59] -> meas[49];
measure node[58] -> meas[50];
measure node[57] -> meas[51];
measure node[56] -> meas[52];
measure node[52] -> meas[53];
measure node[37] -> meas[54];
measure node[38] -> meas[55];
measure node[33] -> meas[56];
measure node[39] -> meas[57];
measure node[40] -> meas[58];
measure node[41] -> meas[59];
measure node[42] -> meas[60];
measure node[43] -> meas[61];
measure node[34] -> meas[62];
measure node[24] -> meas[63];
measure node[23] -> meas[64];
measure node[22] -> meas[65];
measure node[15] -> meas[66];
measure node[25] -> meas[67];
measure node[21] -> meas[68];
measure node[20] -> meas[69];
measure node[19] -> meas[70];
measure node[18] -> meas[71];
measure node[14] -> meas[72];
measure node[0] -> meas[73];
measure node[1] -> meas[74];
measure node[2] -> meas[75];
measure node[3] -> meas[76];
measure node[4] -> meas[77];
measure node[5] -> meas[78];
