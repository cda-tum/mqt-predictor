OPENQASM 2.0;
include "qelib1.inc";

qreg node[82];
creg meas[48];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
sx node[7];
sx node[8];
sx node[10];
sx node[11];
sx node[12];
sx node[14];
sx node[16];
sx node[17];
sx node[20];
sx node[21];
sx node[22];
sx node[23];
sx node[24];
sx node[25];
sx node[26];
sx node[28];
sx node[29];
sx node[30];
sx node[33];
sx node[35];
sx node[37];
sx node[38];
sx node[39];
sx node[45];
sx node[46];
sx node[47];
sx node[52];
sx node[54];
sx node[56];
sx node[57];
sx node[58];
sx node[62];
sx node[63];
sx node[64];
sx node[71];
sx node[72];
sx node[77];
sx node[78];
sx node[79];
sx node[80];
sx node[81];
rz(0.5*pi) node[2];
rz(0.5*pi) node[4];
rz(0.5*pi) node[12];
rz(0.5*pi) node[20];
rz(0.5*pi) node[23];
rz(0.5*pi) node[26];
rz(0.5*pi) node[30];
rz(0.5*pi) node[45];
rz(0.5*pi) node[47];
rz(0.5*pi) node[58];
rz(0.5*pi) node[64];
rz(0.5*pi) node[79];
rz(0.5*pi) node[81];
sx node[2];
sx node[4];
sx node[12];
sx node[20];
sx node[23];
sx node[26];
sx node[30];
sx node[45];
sx node[47];
sx node[58];
sx node[64];
sx node[79];
sx node[81];
cx node[4],node[5];
cx node[12],node[11];
cx node[26],node[16];
cx node[20],node[21];
cx node[23],node[24];
cx node[30],node[29];
cx node[47],node[35];
cx node[58],node[57];
cx node[64],node[63];
cx node[79],node[78];
cx node[4],node[3];
sx node[5];
sx node[11];
sx node[16];
cx node[30],node[17];
cx node[20],node[33];
sx node[21];
sx node[24];
cx node[26],node[25];
sx node[29];
sx node[35];
cx node[47],node[46];
sx node[57];
cx node[58],node[71];
sx node[63];
sx node[78];
cx node[79],node[80];
cx node[2],node[3];
rz(2.5*pi) node[5];
rz(2.5*pi) node[11];
cx node[12],node[17];
rz(2.5*pi) node[16];
rz(2.5*pi) node[21];
rz(2.5*pi) node[24];
rz(2.5*pi) node[29];
sx node[33];
rz(2.5*pi) node[35];
cx node[45],node[46];
rz(2.5*pi) node[57];
rz(2.5*pi) node[63];
rz(2.5*pi) node[78];
cx node[81],node[80];
cx node[2],node[1];
sx node[3];
sx node[5];
sx node[11];
sx node[16];
sx node[17];
sx node[21];
sx node[24];
sx node[29];
rz(2.5*pi) node[33];
sx node[35];
cx node[45],node[54];
sx node[46];
sx node[57];
sx node[63];
cx node[81],node[72];
sx node[78];
sx node[80];
sx node[1];
rz(2.5*pi) node[3];
rz(1.5*pi) node[5];
rz(1.5*pi) node[11];
rz(1.5*pi) node[16];
rz(2.5*pi) node[17];
rz(1.5*pi) node[21];
rz(1.5*pi) node[24];
rz(1.5*pi) node[29];
sx node[33];
rz(1.5*pi) node[35];
rz(2.5*pi) node[46];
cx node[64],node[54];
rz(1.5*pi) node[57];
rz(1.5*pi) node[63];
rz(1.5*pi) node[78];
rz(2.5*pi) node[80];
rz(2.5*pi) node[1];
sx node[3];
cx node[5],node[6];
cx node[11],node[10];
sx node[17];
cx node[21],node[22];
cx node[24],node[25];
cx node[29],node[28];
rz(1.5*pi) node[33];
sx node[46];
sx node[54];
cx node[57],node[56];
cx node[63],node[62];
cx node[78],node[77];
sx node[80];
sx node[1];
rz(1.5*pi) node[3];
sx node[6];
sx node[10];
rz(1.5*pi) node[17];
cx node[23],node[22];
sx node[25];
cx node[35],node[28];
cx node[33],node[39];
rz(1.5*pi) node[46];
rz(2.5*pi) node[54];
sx node[56];
sx node[62];
sx node[77];
rz(1.5*pi) node[80];
rz(1.5*pi) node[1];
rz(2.5*pi) node[6];
rz(2.5*pi) node[10];
sx node[22];
rz(2.5*pi) node[25];
sx node[28];
sx node[39];
sx node[54];
rz(2.5*pi) node[56];
rz(2.5*pi) node[62];
rz(2.5*pi) node[77];
cx node[1],node[0];
sx node[6];
sx node[10];
rz(2.5*pi) node[22];
sx node[25];
rz(2.5*pi) node[28];
rz(2.5*pi) node[39];
rz(1.5*pi) node[54];
sx node[56];
sx node[62];
sx node[77];
sx node[0];
rz(1.5*pi) node[6];
rz(1.5*pi) node[10];
sx node[22];
rz(1.5*pi) node[25];
sx node[28];
sx node[39];
rz(1.5*pi) node[56];
rz(1.5*pi) node[62];
rz(1.5*pi) node[77];
rz(2.5*pi) node[0];
cx node[6],node[7];
rz(1.5*pi) node[22];
rz(1.5*pi) node[28];
rz(1.5*pi) node[39];
cx node[56],node[52];
cx node[62],node[72];
cx node[77],node[71];
sx node[0];
sx node[7];
cx node[39],node[38];
sx node[52];
sx node[71];
sx node[72];
rz(1.5*pi) node[0];
rz(2.5*pi) node[7];
sx node[38];
rz(2.5*pi) node[52];
rz(2.5*pi) node[71];
rz(2.5*pi) node[72];
cx node[0],node[14];
sx node[7];
rz(2.5*pi) node[38];
sx node[52];
sx node[71];
sx node[72];
rz(1.5*pi) node[7];
cx node[14],node[18];
sx node[38];
rz(1.5*pi) node[52];
rz(1.5*pi) node[71];
rz(1.5*pi) node[72];
cx node[7],node[8];
cx node[18],node[14];
cx node[52],node[37];
rz(1.5*pi) node[38];
cx node[16],node[8];
cx node[14],node[18];
cx node[38],node[37];
sx node[8];
cx node[18],node[19];
sx node[37];
rz(2.5*pi) node[8];
cx node[19],node[18];
rz(2.5*pi) node[37];
sx node[8];
cx node[18],node[19];
sx node[37];
rz(1.5*pi) node[8];
cx node[19],node[20];
rz(1.5*pi) node[37];
cx node[20],node[19];
cx node[19],node[20];
cx node[20],node[21];
cx node[21],node[20];
cx node[20],node[21];
cx node[21],node[22];
cx node[22],node[21];
cx node[21],node[22];
cx node[22],node[23];
cx node[23],node[22];
cx node[22],node[23];
cx node[23],node[24];
cx node[24],node[23];
cx node[23],node[24];
cx node[24],node[25];
cx node[25],node[24];
cx node[24],node[25];
cx node[25],node[26];
cx node[26],node[25];
cx node[25],node[26];
cx node[26],node[27];
cx node[27],node[26];
cx node[26],node[27];
cx node[27],node[28];
cx node[28],node[27];
cx node[27],node[28];
cx node[28],node[29];
cx node[29],node[28];
cx node[28],node[29];
cx node[29],node[30];
cx node[30],node[29];
cx node[29],node[30];
cx node[30],node[17];
cx node[17],node[30];
cx node[30],node[17];
cx node[17],node[12];
cx node[12],node[17];
cx node[17],node[12];
cx node[12],node[11];
cx node[11],node[12];
cx node[12],node[11];
cx node[10],node[11];
sx node[11];
rz(2.5*pi) node[11];
sx node[11];
rz(1.5*pi) node[11];
barrier node[19],node[20],node[4],node[5],node[6],node[7],node[29],node[28],node[58],node[57],node[33],node[39],node[47],node[35],node[2],node[3],node[45],node[46],node[56],node[52],node[38],node[64],node[63],node[79],node[78],node[25],node[16],node[27],node[81],node[80],node[1],node[22],node[23],node[37],node[77],node[24],node[62],node[72],node[54],node[8],node[71],node[17],node[12],node[0],node[30],node[10],node[11],node[21];
measure node[19] -> meas[0];
measure node[20] -> meas[1];
measure node[4] -> meas[2];
measure node[5] -> meas[3];
measure node[6] -> meas[4];
measure node[7] -> meas[5];
measure node[29] -> meas[6];
measure node[28] -> meas[7];
measure node[58] -> meas[8];
measure node[57] -> meas[9];
measure node[33] -> meas[10];
measure node[39] -> meas[11];
measure node[47] -> meas[12];
measure node[35] -> meas[13];
measure node[2] -> meas[14];
measure node[3] -> meas[15];
measure node[45] -> meas[16];
measure node[46] -> meas[17];
measure node[56] -> meas[18];
measure node[52] -> meas[19];
measure node[38] -> meas[20];
measure node[64] -> meas[21];
measure node[63] -> meas[22];
measure node[79] -> meas[23];
measure node[78] -> meas[24];
measure node[25] -> meas[25];
measure node[16] -> meas[26];
measure node[27] -> meas[27];
measure node[81] -> meas[28];
measure node[80] -> meas[29];
measure node[1] -> meas[30];
measure node[22] -> meas[31];
measure node[23] -> meas[32];
measure node[37] -> meas[33];
measure node[77] -> meas[34];
measure node[24] -> meas[35];
measure node[62] -> meas[36];
measure node[72] -> meas[37];
measure node[54] -> meas[38];
measure node[8] -> meas[39];
measure node[71] -> meas[40];
measure node[17] -> meas[41];
measure node[12] -> meas[42];
measure node[0] -> meas[43];
measure node[30] -> meas[44];
measure node[10] -> meas[45];
measure node[11] -> meas[46];
measure node[21] -> meas[47];
