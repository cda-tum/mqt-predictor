OPENQASM 2.0;
include "qelib1.inc";

qreg node[126];
creg c[60];
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
sx node[15];
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
sx node[33];
sx node[34];
sx node[37];
sx node[38];
sx node[39];
sx node[40];
sx node[41];
sx node[42];
sx node[43];
sx node[44];
sx node[45];
sx node[52];
sx node[53];
sx node[56];
sx node[57];
sx node[59];
sx node[60];
sx node[61];
sx node[62];
sx node[63];
sx node[72];
sx node[80];
sx node[81];
sx node[82];
sx node[83];
sx node[84];
sx node[92];
sx node[101];
sx node[102];
sx node[103];
sx node[104];
sx node[105];
sx node[111];
sx node[121];
sx node[122];
sx node[123];
sx node[124];
sx node[125];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[28];
rz(0.5*pi) node[33];
rz(0.5*pi) node[34];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[42];
rz(0.5*pi) node[43];
rz(0.5*pi) node[44];
rz(0.5*pi) node[45];
rz(0.5*pi) node[52];
rz(0.5*pi) node[53];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[59];
rz(0.5*pi) node[60];
rz(0.5*pi) node[61];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rz(0.5*pi) node[72];
rz(0.5*pi) node[80];
rz(0.5*pi) node[81];
rz(0.5*pi) node[82];
rz(0.5*pi) node[83];
rz(0.5*pi) node[84];
rz(0.5*pi) node[92];
rz(0.5*pi) node[101];
rz(0.5*pi) node[102];
rz(0.5*pi) node[103];
rz(0.5*pi) node[104];
rz(0.5*pi) node[105];
rz(0.5*pi) node[111];
rz(0.5*pi) node[121];
rz(0.5*pi) node[122];
rz(0.5*pi) node[123];
rz(3.5*pi) node[124];
rz(0.5*pi) node[125];
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
sx node[15];
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
sx node[33];
sx node[34];
sx node[37];
sx node[38];
sx node[39];
sx node[40];
sx node[41];
sx node[42];
sx node[43];
sx node[44];
sx node[45];
sx node[52];
sx node[53];
sx node[56];
sx node[57];
sx node[59];
sx node[60];
sx node[61];
sx node[62];
sx node[63];
sx node[72];
sx node[80];
sx node[81];
sx node[82];
sx node[83];
sx node[84];
sx node[92];
sx node[101];
sx node[102];
sx node[103];
sx node[104];
sx node[105];
sx node[111];
sx node[121];
sx node[122];
sx node[123];
sx node[124];
sx node[125];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[28];
rz(0.5*pi) node[33];
rz(0.5*pi) node[34];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[42];
rz(0.5*pi) node[43];
rz(0.5*pi) node[44];
rz(0.5*pi) node[45];
rz(0.5*pi) node[52];
rz(0.5*pi) node[53];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[59];
rz(0.5*pi) node[60];
rz(0.5*pi) node[61];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rz(0.5*pi) node[72];
rz(0.5*pi) node[80];
rz(0.5*pi) node[81];
rz(0.5*pi) node[82];
rz(0.5*pi) node[83];
rz(0.5*pi) node[84];
rz(0.5*pi) node[92];
rz(0.5*pi) node[101];
rz(0.5*pi) node[102];
rz(0.5*pi) node[103];
rz(0.5*pi) node[104];
rz(0.5*pi) node[105];
rz(0.5*pi) node[111];
rz(0.5*pi) node[121];
rz(0.5*pi) node[122];
rz(0.5*pi) node[123];
rz(0.5*pi) node[125];
cx node[123],node[124];
rz(0.5*pi) node[123];
cx node[125],node[124];
sx node[123];
rz(0.5*pi) node[125];
rz(3.5*pi) node[123];
sx node[125];
sx node[123];
rz(3.5*pi) node[125];
rz(1.0*pi) node[123];
sx node[125];
cx node[124],node[123];
rz(1.0*pi) node[125];
cx node[123],node[124];
cx node[124],node[123];
cx node[122],node[123];
rz(0.5*pi) node[122];
sx node[122];
rz(3.5*pi) node[122];
sx node[122];
rz(1.0*pi) node[122];
cx node[123],node[122];
cx node[122],node[123];
cx node[123],node[122];
cx node[111],node[122];
rz(0.5*pi) node[111];
cx node[121],node[122];
sx node[111];
rz(0.5*pi) node[121];
rz(3.5*pi) node[111];
sx node[121];
sx node[111];
rz(3.5*pi) node[121];
rz(1.0*pi) node[111];
sx node[121];
cx node[122],node[111];
rz(1.0*pi) node[121];
cx node[111],node[122];
cx node[122],node[111];
cx node[104],node[111];
rz(0.5*pi) node[104];
sx node[104];
rz(3.5*pi) node[104];
sx node[104];
rz(1.0*pi) node[104];
cx node[111],node[104];
cx node[104],node[111];
cx node[111],node[104];
cx node[103],node[104];
rz(0.5*pi) node[103];
cx node[105],node[104];
sx node[103];
rz(0.5*pi) node[105];
rz(3.5*pi) node[103];
sx node[105];
sx node[103];
rz(3.5*pi) node[105];
rz(1.0*pi) node[103];
sx node[105];
cx node[104],node[103];
rz(1.0*pi) node[105];
cx node[103],node[104];
cx node[104],node[103];
cx node[102],node[103];
rz(0.5*pi) node[102];
sx node[102];
rz(3.5*pi) node[102];
sx node[102];
rz(1.0*pi) node[102];
cx node[103],node[102];
cx node[102],node[103];
cx node[103],node[102];
cx node[92],node[102];
rz(0.5*pi) node[92];
cx node[101],node[102];
sx node[92];
rz(0.5*pi) node[101];
rz(3.5*pi) node[92];
sx node[101];
sx node[92];
rz(3.5*pi) node[101];
rz(1.0*pi) node[92];
sx node[101];
cx node[102],node[92];
rz(1.0*pi) node[101];
cx node[92],node[102];
cx node[102],node[92];
cx node[83],node[92];
rz(0.5*pi) node[83];
sx node[83];
rz(3.5*pi) node[83];
sx node[83];
rz(1.0*pi) node[83];
cx node[92],node[83];
cx node[83],node[92];
cx node[92],node[83];
cx node[82],node[83];
rz(0.5*pi) node[82];
cx node[84],node[83];
sx node[82];
rz(0.5*pi) node[84];
rz(3.5*pi) node[82];
sx node[84];
sx node[82];
rz(3.5*pi) node[84];
rz(1.0*pi) node[82];
sx node[84];
cx node[83],node[82];
rz(1.0*pi) node[84];
cx node[82],node[83];
cx node[83],node[82];
cx node[81],node[82];
rz(0.5*pi) node[81];
sx node[81];
rz(3.5*pi) node[81];
sx node[81];
rz(1.0*pi) node[81];
cx node[82],node[81];
cx node[81],node[82];
cx node[82],node[81];
cx node[72],node[81];
rz(0.5*pi) node[72];
cx node[80],node[81];
sx node[72];
rz(0.5*pi) node[80];
rz(3.5*pi) node[72];
sx node[80];
sx node[72];
rz(3.5*pi) node[80];
rz(1.0*pi) node[72];
sx node[80];
cx node[81],node[72];
rz(1.0*pi) node[80];
cx node[72],node[81];
cx node[81],node[72];
cx node[62],node[72];
rz(0.5*pi) node[62];
sx node[62];
rz(3.5*pi) node[62];
sx node[62];
rz(1.0*pi) node[62];
cx node[72],node[62];
cx node[62],node[72];
cx node[72],node[62];
cx node[61],node[62];
rz(0.5*pi) node[61];
cx node[63],node[62];
sx node[61];
rz(0.5*pi) node[63];
rz(3.5*pi) node[61];
sx node[63];
sx node[61];
rz(3.5*pi) node[63];
rz(1.0*pi) node[61];
sx node[63];
cx node[62],node[61];
rz(1.0*pi) node[63];
cx node[61],node[62];
cx node[62],node[61];
cx node[60],node[61];
rz(0.5*pi) node[60];
sx node[60];
rz(3.5*pi) node[60];
sx node[60];
rz(1.0*pi) node[60];
cx node[61],node[60];
cx node[60],node[61];
cx node[61],node[60];
cx node[53],node[60];
rz(0.5*pi) node[53];
cx node[59],node[60];
sx node[53];
rz(0.5*pi) node[59];
rz(3.5*pi) node[53];
sx node[59];
sx node[53];
rz(3.5*pi) node[59];
rz(1.0*pi) node[53];
sx node[59];
cx node[60],node[53];
rz(1.0*pi) node[59];
cx node[53],node[60];
cx node[60],node[53];
cx node[41],node[53];
rz(0.5*pi) node[41];
sx node[41];
rz(3.5*pi) node[41];
sx node[41];
rz(1.0*pi) node[41];
cx node[53],node[41];
cx node[41],node[53];
cx node[53],node[41];
cx node[40],node[41];
rz(0.5*pi) node[40];
cx node[42],node[41];
sx node[40];
rz(0.5*pi) node[42];
rz(3.5*pi) node[40];
sx node[42];
sx node[40];
rz(3.5*pi) node[42];
rz(1.0*pi) node[40];
sx node[42];
cx node[41],node[40];
rz(1.0*pi) node[42];
cx node[40],node[41];
cx node[43],node[42];
cx node[41],node[40];
cx node[42],node[43];
cx node[39],node[40];
cx node[43],node[42];
rz(0.5*pi) node[39];
cx node[42],node[41];
cx node[44],node[43];
sx node[39];
cx node[41],node[42];
cx node[43],node[44];
rz(3.5*pi) node[39];
cx node[42],node[41];
cx node[44],node[43];
sx node[39];
cx node[43],node[42];
cx node[45],node[44];
rz(1.0*pi) node[39];
cx node[42],node[43];
cx node[44],node[45];
cx node[40],node[39];
cx node[43],node[42];
cx node[45],node[44];
cx node[39],node[40];
cx node[44],node[43];
cx node[40],node[39];
cx node[43],node[44];
cx node[33],node[39];
cx node[41],node[40];
cx node[44],node[43];
rz(0.5*pi) node[33];
cx node[38],node[39];
cx node[40],node[41];
sx node[33];
rz(0.5*pi) node[38];
cx node[41],node[40];
rz(3.5*pi) node[33];
sx node[38];
cx node[42],node[41];
sx node[33];
rz(3.5*pi) node[38];
cx node[41],node[42];
rz(1.0*pi) node[33];
sx node[38];
cx node[42],node[41];
cx node[39],node[33];
rz(1.0*pi) node[38];
cx node[43],node[42];
cx node[33],node[39];
cx node[37],node[38];
cx node[42],node[43];
cx node[39],node[33];
cx node[38],node[37];
cx node[43],node[42];
cx node[20],node[33];
cx node[37],node[38];
rz(0.5*pi) node[20];
cx node[52],node[37];
cx node[38],node[39];
sx node[20];
cx node[37],node[52];
cx node[39],node[38];
rz(3.5*pi) node[20];
cx node[52],node[37];
cx node[38],node[39];
sx node[20];
cx node[37],node[38];
cx node[56],node[52];
rz(1.0*pi) node[20];
cx node[38],node[37];
cx node[52],node[56];
cx node[33],node[20];
cx node[37],node[38];
cx node[56],node[52];
cx node[20],node[33];
cx node[52],node[37];
cx node[57],node[56];
cx node[33],node[20];
cx node[37],node[52];
cx node[56],node[57];
cx node[19],node[20];
cx node[39],node[33];
cx node[52],node[37];
cx node[57],node[56];
rz(0.5*pi) node[19];
cx node[21],node[20];
cx node[33],node[39];
cx node[56],node[52];
sx node[19];
rz(0.5*pi) node[21];
cx node[39],node[33];
cx node[52],node[56];
rz(3.5*pi) node[19];
sx node[21];
cx node[38],node[39];
cx node[56],node[52];
sx node[19];
rz(3.5*pi) node[21];
cx node[39],node[38];
rz(1.0*pi) node[19];
sx node[21];
cx node[38],node[39];
cx node[20],node[19];
rz(1.0*pi) node[21];
cx node[37],node[38];
cx node[19],node[20];
cx node[22],node[21];
cx node[38],node[37];
cx node[20],node[19];
cx node[21],node[22];
cx node[37],node[38];
cx node[18],node[19];
cx node[22],node[21];
cx node[52],node[37];
rz(0.5*pi) node[18];
cx node[21],node[20];
cx node[23],node[22];
cx node[37],node[52];
sx node[18];
cx node[20],node[21];
cx node[22],node[23];
cx node[52],node[37];
rz(3.5*pi) node[18];
cx node[21],node[20];
cx node[23],node[22];
sx node[18];
cx node[22],node[21];
cx node[24],node[23];
rz(1.0*pi) node[18];
cx node[21],node[22];
cx node[23],node[24];
cx node[19],node[18];
cx node[22],node[21];
cx node[24],node[23];
cx node[18],node[19];
cx node[23],node[22];
cx node[25],node[24];
cx node[19],node[18];
cx node[22],node[23];
cx node[24],node[25];
cx node[14],node[18];
cx node[20],node[19];
cx node[23],node[22];
cx node[25],node[24];
rz(0.5*pi) node[14];
cx node[19],node[20];
cx node[24],node[23];
cx node[26],node[25];
sx node[14];
cx node[20],node[19];
cx node[23],node[24];
cx node[25],node[26];
rz(3.5*pi) node[14];
cx node[21],node[20];
cx node[24],node[23];
cx node[26],node[25];
sx node[14];
cx node[20],node[21];
cx node[34],node[24];
cx node[27],node[26];
rz(1.0*pi) node[14];
cx node[21],node[20];
cx node[24],node[34];
cx node[26],node[27];
cx node[18],node[14];
cx node[22],node[21];
cx node[34],node[24];
cx node[27],node[26];
cx node[14],node[18];
cx node[21],node[22];
cx node[28],node[27];
cx node[18],node[14];
cx node[22],node[21];
cx node[27],node[28];
cx node[0],node[14];
cx node[19],node[18];
cx node[23],node[22];
cx node[28],node[27];
rz(0.5*pi) node[0];
cx node[18],node[19];
cx node[22],node[23];
sx node[0];
cx node[19],node[18];
cx node[23],node[22];
rz(3.5*pi) node[0];
cx node[20],node[19];
cx node[24],node[23];
sx node[0];
cx node[19],node[20];
cx node[23],node[24];
rz(1.0*pi) node[0];
cx node[20],node[19];
cx node[24],node[23];
cx node[14],node[0];
cx node[21],node[20];
cx node[25],node[24];
cx node[0],node[14];
cx node[20],node[21];
cx node[24],node[25];
cx node[14],node[0];
cx node[21],node[20];
cx node[25],node[24];
cx node[1],node[0];
cx node[18],node[14];
cx node[22],node[21];
cx node[26],node[25];
rz(0.5*pi) node[1];
cx node[14],node[18];
cx node[21],node[22];
cx node[25],node[26];
sx node[1];
cx node[18],node[14];
cx node[22],node[21];
cx node[26],node[25];
rz(3.5*pi) node[1];
cx node[19],node[18];
cx node[23],node[22];
cx node[27],node[26];
sx node[1];
cx node[18],node[19];
cx node[22],node[23];
cx node[26],node[27];
rz(1.0*pi) node[1];
cx node[19],node[18];
cx node[23],node[22];
cx node[27],node[26];
cx node[2],node[1];
cx node[20],node[19];
cx node[24],node[23];
cx node[1],node[2];
cx node[19],node[20];
cx node[23],node[24];
cx node[2],node[1];
cx node[20],node[19];
cx node[24],node[23];
cx node[1],node[0];
cx node[3],node[2];
cx node[33],node[20];
cx node[25],node[24];
rz(0.5*pi) node[1];
cx node[2],node[3];
cx node[20],node[33];
cx node[24],node[25];
sx node[1];
cx node[3],node[2];
cx node[33],node[20];
cx node[25],node[24];
rz(3.5*pi) node[1];
cx node[4],node[3];
cx node[26],node[25];
cx node[39],node[33];
sx node[1];
cx node[3],node[4];
cx node[25],node[26];
cx node[33],node[39];
rz(1.0*pi) node[1];
cx node[4],node[3];
cx node[26],node[25];
cx node[39],node[33];
cx node[2],node[1];
cx node[5],node[4];
cx node[40],node[39];
cx node[1],node[2];
cx node[4],node[5];
cx node[39],node[40];
cx node[2],node[1];
cx node[5],node[4];
cx node[40],node[39];
cx node[1],node[0];
cx node[3],node[2];
cx node[6],node[5];
cx node[41],node[40];
rz(0.5*pi) node[1];
cx node[2],node[3];
cx node[5],node[6];
cx node[40],node[41];
sx node[1];
cx node[3],node[2];
cx node[6],node[5];
cx node[41],node[40];
rz(3.5*pi) node[1];
cx node[4],node[3];
cx node[7],node[6];
cx node[42],node[41];
sx node[1];
cx node[3],node[4];
cx node[6],node[7];
cx node[41],node[42];
rz(1.0*pi) node[1];
cx node[4],node[3];
cx node[7],node[6];
cx node[42],node[41];
cx node[2],node[1];
cx node[15],node[4];
cx node[8],node[7];
cx node[1],node[2];
cx node[4],node[15];
cx node[7],node[8];
cx node[2],node[1];
cx node[15],node[4];
cx node[8],node[7];
cx node[1],node[0];
cx node[3],node[2];
cx node[16],node[8];
rz(0.5*pi) node[1];
cx node[2],node[3];
cx node[8],node[16];
sx node[1];
cx node[3],node[2];
cx node[16],node[8];
rz(3.5*pi) node[1];
cx node[4],node[3];
sx node[1];
cx node[3],node[4];
rz(1.0*pi) node[1];
cx node[4],node[3];
cx node[2],node[1];
cx node[5],node[4];
cx node[1],node[2];
cx node[4],node[5];
cx node[2],node[1];
cx node[5],node[4];
cx node[1],node[0];
cx node[3],node[2];
cx node[6],node[5];
rz(0.5*pi) node[1];
cx node[2],node[3];
cx node[5],node[6];
sx node[1];
cx node[3],node[2];
cx node[6],node[5];
rz(3.5*pi) node[1];
cx node[4],node[3];
cx node[7],node[6];
sx node[1];
cx node[3],node[4];
cx node[6],node[7];
rz(1.0*pi) node[1];
cx node[4],node[3];
cx node[7],node[6];
cx node[2],node[1];
cx node[5],node[4];
cx node[8],node[7];
cx node[1],node[2];
cx node[4],node[5];
cx node[7],node[8];
cx node[2],node[1];
cx node[5],node[4];
cx node[8],node[7];
cx node[1],node[0];
cx node[3],node[2];
cx node[6],node[5];
cx node[14],node[0];
rz(0.5*pi) node[1];
cx node[2],node[3];
cx node[5],node[6];
sx node[1];
cx node[3],node[2];
cx node[6],node[5];
rz(0.5*pi) node[14];
rz(3.5*pi) node[1];
cx node[4],node[3];
cx node[7],node[6];
sx node[14];
sx node[1];
cx node[3],node[4];
cx node[6],node[7];
rz(3.5*pi) node[14];
rz(1.0*pi) node[1];
cx node[4],node[3];
cx node[7],node[6];
sx node[14];
cx node[2],node[1];
cx node[5],node[4];
rz(1.0*pi) node[14];
cx node[1],node[2];
cx node[4],node[5];
cx node[18],node[14];
cx node[2],node[1];
cx node[5],node[4];
cx node[14],node[18];
cx node[1],node[0];
cx node[3],node[2];
cx node[6],node[5];
cx node[18],node[14];
cx node[14],node[0];
rz(0.5*pi) node[1];
cx node[2],node[3];
cx node[5],node[6];
cx node[19],node[18];
sx node[1];
cx node[3],node[2];
cx node[6],node[5];
rz(0.5*pi) node[14];
cx node[18],node[19];
rz(3.5*pi) node[1];
cx node[4],node[3];
sx node[14];
cx node[19],node[18];
sx node[1];
cx node[3],node[4];
rz(3.5*pi) node[14];
cx node[20],node[19];
rz(1.0*pi) node[1];
cx node[4],node[3];
sx node[14];
cx node[19],node[20];
cx node[2],node[1];
cx node[5],node[4];
rz(1.0*pi) node[14];
cx node[20],node[19];
cx node[1],node[2];
cx node[4],node[5];
cx node[18],node[14];
cx node[21],node[20];
cx node[2],node[1];
cx node[5],node[4];
cx node[14],node[18];
cx node[20],node[21];
cx node[1],node[0];
cx node[3],node[2];
cx node[18],node[14];
cx node[21],node[20];
cx node[14],node[0];
rz(0.5*pi) node[1];
cx node[2],node[3];
cx node[19],node[18];
cx node[22],node[21];
sx node[1];
cx node[3],node[2];
rz(0.5*pi) node[14];
cx node[18],node[19];
cx node[21],node[22];
rz(3.5*pi) node[1];
cx node[4],node[3];
sx node[14];
cx node[19],node[18];
cx node[22],node[21];
sx node[1];
cx node[3],node[4];
rz(3.5*pi) node[14];
cx node[20],node[19];
cx node[23],node[22];
rz(1.0*pi) node[1];
cx node[4],node[3];
sx node[14];
cx node[19],node[20];
cx node[22],node[23];
cx node[2],node[1];
rz(1.0*pi) node[14];
cx node[20],node[19];
cx node[23],node[22];
cx node[1],node[2];
cx node[18],node[14];
cx node[21],node[20];
cx node[24],node[23];
cx node[2],node[1];
cx node[14],node[18];
cx node[20],node[21];
cx node[23],node[24];
cx node[1],node[0];
cx node[3],node[2];
cx node[18],node[14];
cx node[21],node[20];
cx node[24],node[23];
cx node[14],node[0];
rz(0.5*pi) node[1];
cx node[2],node[3];
cx node[19],node[18];
cx node[22],node[21];
cx node[25],node[24];
sx node[1];
cx node[3],node[2];
rz(0.5*pi) node[14];
cx node[18],node[19];
cx node[21],node[22];
cx node[24],node[25];
rz(3.5*pi) node[1];
sx node[14];
cx node[19],node[18];
cx node[22],node[21];
cx node[25],node[24];
sx node[1];
rz(3.5*pi) node[14];
cx node[20],node[19];
cx node[23],node[22];
rz(1.0*pi) node[1];
sx node[14];
cx node[19],node[20];
cx node[22],node[23];
cx node[2],node[1];
rz(1.0*pi) node[14];
cx node[20],node[19];
cx node[23],node[22];
cx node[1],node[2];
cx node[18],node[14];
cx node[33],node[20];
cx node[24],node[23];
cx node[2],node[1];
cx node[14],node[18];
cx node[20],node[33];
cx node[23],node[24];
cx node[18],node[14];
cx node[33],node[20];
cx node[24],node[23];
cx node[14],node[0];
cx node[19],node[18];
cx node[39],node[33];
rz(0.5*pi) node[14];
cx node[18],node[19];
cx node[33],node[39];
sx node[14];
cx node[19],node[18];
cx node[39],node[33];
rz(3.5*pi) node[14];
cx node[20],node[19];
cx node[38],node[39];
sx node[14];
cx node[19],node[20];
cx node[39],node[38];
rz(1.0*pi) node[14];
cx node[20],node[19];
cx node[38],node[39];
cx node[18],node[14];
cx node[21],node[20];
cx node[37],node[38];
cx node[14],node[18];
cx node[20],node[21];
cx node[38],node[37];
cx node[18],node[14];
cx node[21],node[20];
cx node[37],node[38];
cx node[14],node[0];
cx node[19],node[18];
cx node[22],node[21];
cx node[1],node[0];
rz(0.5*pi) node[14];
cx node[18],node[19];
cx node[21],node[22];
rz(0.5*pi) node[1];
sx node[14];
cx node[19],node[18];
cx node[22],node[21];
sx node[1];
rz(3.5*pi) node[14];
cx node[20],node[19];
cx node[23],node[22];
rz(3.5*pi) node[1];
sx node[14];
cx node[19],node[20];
cx node[22],node[23];
sx node[1];
rz(1.0*pi) node[14];
cx node[20],node[19];
cx node[23],node[22];
rz(1.0*pi) node[1];
cx node[18],node[14];
cx node[33],node[20];
cx node[14],node[18];
cx node[20],node[33];
cx node[18],node[14];
cx node[33],node[20];
cx node[14],node[0];
cx node[19],node[18];
cx node[39],node[33];
rz(0.5*pi) node[14];
cx node[18],node[19];
cx node[33],node[39];
sx node[14];
cx node[19],node[18];
cx node[39],node[33];
rz(3.5*pi) node[14];
cx node[20],node[19];
cx node[40],node[39];
sx node[14];
cx node[19],node[20];
cx node[39],node[40];
rz(1.0*pi) node[14];
cx node[20],node[19];
cx node[40],node[39];
cx node[18],node[14];
cx node[33],node[20];
cx node[41],node[40];
cx node[14],node[18];
cx node[20],node[33];
cx node[40],node[41];
cx node[18],node[14];
cx node[33],node[20];
cx node[41],node[40];
cx node[14],node[0];
cx node[19],node[18];
cx node[39],node[33];
rz(0.5*pi) node[14];
cx node[18],node[19];
cx node[33],node[39];
sx node[14];
cx node[19],node[18];
cx node[39],node[33];
rz(3.5*pi) node[14];
cx node[20],node[19];
cx node[38],node[39];
sx node[14];
cx node[19],node[20];
cx node[39],node[38];
rz(1.0*pi) node[14];
cx node[20],node[19];
cx node[38],node[39];
cx node[18],node[14];
cx node[21],node[20];
cx node[14],node[18];
cx node[20],node[21];
cx node[18],node[14];
cx node[21],node[20];
cx node[14],node[0];
cx node[19],node[18];
cx node[22],node[21];
rz(0.5*pi) node[14];
cx node[18],node[19];
cx node[21],node[22];
sx node[14];
cx node[19],node[18];
cx node[22],node[21];
rz(3.5*pi) node[14];
cx node[20],node[19];
sx node[14];
cx node[19],node[20];
rz(1.0*pi) node[14];
cx node[20],node[19];
cx node[18],node[14];
cx node[33],node[20];
cx node[14],node[18];
cx node[20],node[33];
cx node[18],node[14];
cx node[33],node[20];
cx node[14],node[0];
cx node[19],node[18];
cx node[39],node[33];
rz(0.5*pi) node[14];
cx node[18],node[19];
cx node[33],node[39];
sx node[14];
cx node[19],node[18];
cx node[39],node[33];
rz(3.5*pi) node[14];
cx node[20],node[19];
cx node[40],node[39];
sx node[14];
cx node[19],node[20];
cx node[39],node[40];
rz(1.0*pi) node[14];
cx node[20],node[19];
cx node[40],node[39];
cx node[18],node[14];
cx node[33],node[20];
cx node[14],node[18];
cx node[20],node[33];
cx node[18],node[14];
cx node[33],node[20];
cx node[14],node[0];
cx node[19],node[18];
cx node[39],node[33];
rz(0.5*pi) node[14];
cx node[18],node[19];
cx node[33],node[39];
sx node[14];
cx node[19],node[18];
cx node[39],node[33];
rz(3.5*pi) node[14];
cx node[20],node[19];
sx node[14];
cx node[19],node[20];
rz(1.0*pi) node[14];
cx node[20],node[19];
cx node[18],node[14];
cx node[21],node[20];
cx node[14],node[18];
cx node[20],node[21];
cx node[18],node[14];
cx node[21],node[20];
cx node[14],node[0];
cx node[19],node[18];
rz(0.5*pi) node[14];
cx node[18],node[19];
sx node[14];
cx node[19],node[18];
rz(3.5*pi) node[14];
cx node[20],node[19];
sx node[14];
cx node[19],node[20];
rz(1.0*pi) node[14];
cx node[20],node[19];
cx node[18],node[14];
cx node[33],node[20];
cx node[14],node[18];
cx node[20],node[33];
cx node[18],node[14];
cx node[33],node[20];
cx node[14],node[0];
cx node[19],node[18];
rz(0.5*pi) node[14];
cx node[18],node[19];
sx node[14];
cx node[19],node[18];
rz(3.5*pi) node[14];
cx node[20],node[19];
sx node[14];
cx node[19],node[20];
rz(1.0*pi) node[14];
cx node[20],node[19];
cx node[18],node[14];
cx node[14],node[18];
cx node[18],node[14];
cx node[14],node[0];
cx node[19],node[18];
rz(0.5*pi) node[14];
cx node[18],node[19];
sx node[14];
cx node[19],node[18];
rz(3.5*pi) node[14];
sx node[14];
rz(1.0*pi) node[14];
cx node[18],node[14];
cx node[14],node[18];
cx node[18],node[14];
cx node[14],node[0];
rz(0.5*pi) node[14];
sx node[14];
rz(3.5*pi) node[14];
sx node[14];
rz(1.0*pi) node[14];
barrier node[124],node[125],node[123],node[122],node[121],node[111],node[104],node[105],node[103],node[102],node[101],node[92],node[83],node[84],node[82],node[81],node[80],node[72],node[62],node[63],node[61],node[60],node[59],node[53],node[44],node[45],node[43],node[56],node[57],node[52],node[34],node[28],node[27],node[26],node[42],node[16],node[15],node[8],node[7],node[6],node[5],node[25],node[4],node[24],node[3],node[37],node[2],node[23],node[41],node[38],node[1],node[22],node[40],node[39],node[21],node[33],node[20],node[19],node[18],node[14],node[0];
measure node[124] -> c[0];
measure node[125] -> c[1];
measure node[123] -> c[2];
measure node[122] -> c[3];
measure node[121] -> c[4];
measure node[111] -> c[5];
measure node[104] -> c[6];
measure node[105] -> c[7];
measure node[103] -> c[8];
measure node[102] -> c[9];
measure node[101] -> c[10];
measure node[92] -> c[11];
measure node[83] -> c[12];
measure node[84] -> c[13];
measure node[82] -> c[14];
measure node[81] -> c[15];
measure node[80] -> c[16];
measure node[72] -> c[17];
measure node[62] -> c[18];
measure node[63] -> c[19];
measure node[61] -> c[20];
measure node[60] -> c[21];
measure node[59] -> c[22];
measure node[53] -> c[23];
measure node[44] -> c[24];
measure node[45] -> c[25];
measure node[43] -> c[26];
measure node[56] -> c[27];
measure node[57] -> c[28];
measure node[52] -> c[29];
measure node[34] -> c[30];
measure node[28] -> c[31];
measure node[27] -> c[32];
measure node[26] -> c[33];
measure node[42] -> c[34];
measure node[16] -> c[35];
measure node[15] -> c[36];
measure node[8] -> c[37];
measure node[7] -> c[38];
measure node[6] -> c[39];
measure node[5] -> c[40];
measure node[25] -> c[41];
measure node[4] -> c[42];
measure node[24] -> c[43];
measure node[3] -> c[44];
measure node[37] -> c[45];
measure node[2] -> c[46];
measure node[23] -> c[47];
measure node[41] -> c[48];
measure node[38] -> c[49];
measure node[1] -> c[50];
measure node[22] -> c[51];
measure node[40] -> c[52];
measure node[39] -> c[53];
measure node[21] -> c[54];
measure node[33] -> c[55];
measure node[20] -> c[56];
measure node[19] -> c[57];
measure node[18] -> c[58];
measure node[14] -> c[59];
