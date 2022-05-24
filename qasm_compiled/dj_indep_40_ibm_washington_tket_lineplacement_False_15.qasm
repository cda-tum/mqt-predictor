OPENQASM 2.0;
include "qelib1.inc";

qreg node[62];
creg c[39];
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
sx node[59];
sx node[60];
sx node[61];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(3.5*pi) node[4];
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
rz(0.5*pi) node[59];
rz(0.5*pi) node[60];
rz(0.5*pi) node[61];
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
sx node[59];
sx node[60];
sx node[61];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
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
rz(0.5*pi) node[59];
rz(0.5*pi) node[60];
rz(0.5*pi) node[61];
cx node[3],node[4];
rz(0.5*pi) node[3];
cx node[5],node[4];
sx node[3];
cx node[15],node[4];
rz(0.5*pi) node[5];
rz(3.5*pi) node[3];
sx node[5];
rz(0.5*pi) node[15];
sx node[3];
rz(3.5*pi) node[5];
sx node[15];
rz(1.0*pi) node[3];
sx node[5];
rz(3.5*pi) node[15];
cx node[4],node[3];
rz(1.0*pi) node[5];
sx node[15];
cx node[3],node[4];
cx node[6],node[5];
rz(1.0*pi) node[15];
cx node[4],node[3];
cx node[5],node[6];
cx node[22],node[15];
cx node[2],node[3];
cx node[6],node[5];
cx node[15],node[22];
rz(0.5*pi) node[2];
cx node[7],node[6];
cx node[22],node[15];
sx node[2];
cx node[15],node[4];
cx node[6],node[7];
cx node[21],node[22];
rz(3.5*pi) node[2];
cx node[4],node[15];
cx node[7],node[6];
cx node[22],node[21];
sx node[2];
cx node[15],node[4];
cx node[8],node[7];
cx node[21],node[22];
rz(1.0*pi) node[2];
cx node[7],node[8];
cx node[22],node[15];
cx node[3],node[2];
cx node[8],node[7];
cx node[15],node[22];
cx node[2],node[3];
cx node[16],node[8];
cx node[22],node[15];
cx node[3],node[2];
cx node[8],node[16];
cx node[23],node[22];
cx node[1],node[2];
cx node[4],node[3];
cx node[16],node[8];
cx node[22],node[23];
rz(0.5*pi) node[1];
cx node[3],node[4];
cx node[23],node[22];
sx node[1];
cx node[4],node[3];
cx node[24],node[23];
rz(3.5*pi) node[1];
cx node[5],node[4];
cx node[23],node[24];
sx node[1];
cx node[4],node[5];
cx node[24],node[23];
rz(1.0*pi) node[1];
cx node[5],node[4];
cx node[25],node[24];
cx node[2],node[1];
cx node[6],node[5];
cx node[24],node[25];
cx node[1],node[2];
cx node[5],node[6];
cx node[25],node[24];
cx node[2],node[1];
cx node[6],node[5];
cx node[26],node[25];
cx node[0],node[1];
cx node[3],node[2];
cx node[7],node[6];
cx node[25],node[26];
rz(0.5*pi) node[0];
cx node[2],node[3];
cx node[6],node[7];
cx node[26],node[25];
sx node[0];
cx node[3],node[2];
cx node[7],node[6];
cx node[27],node[26];
rz(3.5*pi) node[0];
cx node[4],node[3];
cx node[8],node[7];
cx node[26],node[27];
sx node[0];
cx node[3],node[4];
cx node[7],node[8];
cx node[27],node[26];
rz(1.0*pi) node[0];
cx node[4],node[3];
cx node[8],node[7];
cx node[28],node[27];
cx node[14],node[0];
cx node[15],node[4];
cx node[27],node[28];
cx node[0],node[14];
cx node[4],node[15];
cx node[28],node[27];
cx node[14],node[0];
cx node[15],node[4];
cx node[0],node[1];
cx node[18],node[14];
cx node[22],node[15];
rz(0.5*pi) node[0];
cx node[14],node[18];
cx node[15],node[22];
sx node[0];
cx node[18],node[14];
cx node[22],node[15];
rz(3.5*pi) node[0];
cx node[19],node[18];
cx node[23],node[22];
sx node[0];
cx node[18],node[19];
cx node[22],node[23];
rz(1.0*pi) node[0];
cx node[19],node[18];
cx node[23],node[22];
cx node[14],node[0];
cx node[20],node[19];
cx node[24],node[23];
cx node[0],node[14];
cx node[19],node[20];
cx node[23],node[24];
cx node[14],node[0];
cx node[20],node[19];
cx node[24],node[23];
cx node[0],node[1];
cx node[18],node[14];
cx node[33],node[20];
cx node[34],node[24];
rz(0.5*pi) node[0];
cx node[14],node[18];
cx node[20],node[33];
cx node[24],node[34];
sx node[0];
cx node[18],node[14];
cx node[33],node[20];
cx node[34],node[24];
rz(3.5*pi) node[0];
cx node[19],node[18];
cx node[39],node[33];
cx node[43],node[34];
sx node[0];
cx node[18],node[19];
cx node[33],node[39];
cx node[34],node[43];
rz(1.0*pi) node[0];
cx node[19],node[18];
cx node[39],node[33];
cx node[43],node[34];
cx node[14],node[0];
cx node[20],node[19];
cx node[40],node[39];
cx node[42],node[43];
cx node[0],node[14];
cx node[19],node[20];
cx node[39],node[40];
cx node[43],node[42];
cx node[14],node[0];
cx node[20],node[19];
cx node[40],node[39];
cx node[42],node[43];
cx node[0],node[1];
cx node[18],node[14];
cx node[33],node[20];
cx node[41],node[40];
rz(0.5*pi) node[0];
cx node[14],node[18];
cx node[20],node[33];
cx node[40],node[41];
sx node[0];
cx node[18],node[14];
cx node[33],node[20];
cx node[41],node[40];
rz(3.5*pi) node[0];
cx node[19],node[18];
cx node[39],node[33];
cx node[53],node[41];
sx node[0];
cx node[18],node[19];
cx node[33],node[39];
cx node[41],node[53];
rz(1.0*pi) node[0];
cx node[19],node[18];
cx node[39],node[33];
cx node[53],node[41];
cx node[14],node[0];
cx node[20],node[19];
cx node[38],node[39];
cx node[60],node[53];
cx node[0],node[14];
cx node[19],node[20];
cx node[39],node[38];
cx node[53],node[60];
cx node[14],node[0];
cx node[20],node[19];
cx node[38],node[39];
cx node[60],node[53];
cx node[0],node[1];
cx node[18],node[14];
cx node[33],node[20];
cx node[37],node[38];
cx node[61],node[60];
rz(0.5*pi) node[0];
cx node[2],node[1];
cx node[14],node[18];
cx node[20],node[33];
cx node[38],node[37];
cx node[60],node[61];
sx node[0];
rz(0.5*pi) node[2];
cx node[18],node[14];
cx node[33],node[20];
cx node[37],node[38];
cx node[61],node[60];
rz(3.5*pi) node[0];
sx node[2];
cx node[19],node[18];
cx node[39],node[33];
cx node[52],node[37];
sx node[0];
rz(3.5*pi) node[2];
cx node[18],node[19];
cx node[33],node[39];
cx node[37],node[52];
rz(1.0*pi) node[0];
sx node[2];
cx node[19],node[18];
cx node[39],node[33];
cx node[52],node[37];
cx node[14],node[0];
rz(1.0*pi) node[2];
cx node[20],node[19];
cx node[40],node[39];
cx node[56],node[52];
cx node[0],node[14];
cx node[3],node[2];
cx node[19],node[20];
cx node[39],node[40];
cx node[52],node[56];
cx node[14],node[0];
cx node[2],node[3];
cx node[20],node[19];
cx node[40],node[39];
cx node[56],node[52];
cx node[3],node[2];
cx node[18],node[14];
cx node[33],node[20];
cx node[41],node[40];
cx node[2],node[1];
cx node[4],node[3];
cx node[14],node[18];
cx node[20],node[33];
cx node[40],node[41];
rz(0.5*pi) node[2];
cx node[3],node[4];
cx node[18],node[14];
cx node[33],node[20];
cx node[41],node[40];
sx node[2];
cx node[4],node[3];
cx node[19],node[18];
cx node[39],node[33];
cx node[53],node[41];
rz(3.5*pi) node[2];
cx node[15],node[4];
cx node[18],node[19];
cx node[33],node[39];
cx node[41],node[53];
sx node[2];
cx node[4],node[15];
cx node[19],node[18];
cx node[39],node[33];
cx node[53],node[41];
rz(1.0*pi) node[2];
cx node[15],node[4];
cx node[20],node[19];
cx node[38],node[39];
cx node[60],node[53];
cx node[3],node[2];
cx node[22],node[15];
cx node[19],node[20];
cx node[39],node[38];
cx node[53],node[60];
cx node[2],node[3];
cx node[15],node[22];
cx node[20],node[19];
cx node[38],node[39];
cx node[60],node[53];
cx node[3],node[2];
cx node[22],node[15];
cx node[33],node[20];
cx node[37],node[38];
cx node[59],node[60];
cx node[2],node[1];
cx node[4],node[3];
cx node[20],node[33];
cx node[23],node[22];
cx node[38],node[37];
cx node[60],node[59];
cx node[0],node[1];
rz(0.5*pi) node[2];
cx node[3],node[4];
cx node[33],node[20];
cx node[22],node[23];
cx node[37],node[38];
cx node[59],node[60];
rz(0.5*pi) node[0];
sx node[2];
cx node[4],node[3];
cx node[23],node[22];
cx node[39],node[33];
cx node[52],node[37];
sx node[0];
rz(3.5*pi) node[2];
cx node[5],node[4];
cx node[24],node[23];
cx node[33],node[39];
cx node[37],node[52];
rz(3.5*pi) node[0];
sx node[2];
cx node[4],node[5];
cx node[23],node[24];
cx node[39],node[33];
cx node[52],node[37];
sx node[0];
rz(1.0*pi) node[2];
cx node[5],node[4];
cx node[24],node[23];
cx node[40],node[39];
rz(1.0*pi) node[0];
cx node[3],node[2];
cx node[6],node[5];
cx node[25],node[24];
cx node[39],node[40];
cx node[14],node[0];
cx node[2],node[3];
cx node[5],node[6];
cx node[24],node[25];
cx node[40],node[39];
cx node[0],node[14];
cx node[3],node[2];
cx node[6],node[5];
cx node[25],node[24];
cx node[41],node[40];
cx node[14],node[0];
cx node[2],node[1];
cx node[4],node[3];
cx node[7],node[6];
cx node[26],node[25];
cx node[40],node[41];
rz(0.5*pi) node[2];
cx node[3],node[4];
cx node[6],node[7];
cx node[18],node[14];
cx node[25],node[26];
cx node[41],node[40];
sx node[2];
cx node[4],node[3];
cx node[7],node[6];
cx node[14],node[18];
cx node[26],node[25];
cx node[53],node[41];
rz(3.5*pi) node[2];
cx node[15],node[4];
cx node[18],node[14];
cx node[27],node[26];
cx node[41],node[53];
sx node[2];
cx node[4],node[15];
cx node[19],node[18];
cx node[26],node[27];
cx node[53],node[41];
rz(1.0*pi) node[2];
cx node[15],node[4];
cx node[18],node[19];
cx node[27],node[26];
cx node[60],node[53];
cx node[3],node[2];
cx node[22],node[15];
cx node[19],node[18];
cx node[53],node[60];
cx node[2],node[3];
cx node[15],node[22];
cx node[20],node[19];
cx node[60],node[53];
cx node[3],node[2];
cx node[22],node[15];
cx node[19],node[20];
cx node[2],node[1];
cx node[4],node[3];
cx node[20],node[19];
cx node[23],node[22];
cx node[0],node[1];
rz(0.5*pi) node[2];
cx node[3],node[4];
cx node[33],node[20];
cx node[22],node[23];
rz(0.5*pi) node[0];
sx node[2];
cx node[4],node[3];
cx node[20],node[33];
cx node[23],node[22];
sx node[0];
rz(3.5*pi) node[2];
cx node[5],node[4];
cx node[33],node[20];
cx node[24],node[23];
rz(3.5*pi) node[0];
sx node[2];
cx node[4],node[5];
cx node[23],node[24];
cx node[39],node[33];
sx node[0];
rz(1.0*pi) node[2];
cx node[5],node[4];
cx node[24],node[23];
cx node[33],node[39];
rz(1.0*pi) node[0];
cx node[3],node[2];
cx node[6],node[5];
cx node[34],node[24];
cx node[39],node[33];
cx node[14],node[0];
cx node[2],node[3];
cx node[5],node[6];
cx node[24],node[34];
cx node[38],node[39];
cx node[0],node[14];
cx node[3],node[2];
cx node[6],node[5];
cx node[34],node[24];
cx node[39],node[38];
cx node[14],node[0];
cx node[2],node[1];
cx node[4],node[3];
cx node[43],node[34];
cx node[38],node[39];
rz(0.5*pi) node[2];
cx node[3],node[4];
cx node[18],node[14];
cx node[34],node[43];
cx node[37],node[38];
sx node[2];
cx node[4],node[3];
cx node[14],node[18];
cx node[43],node[34];
cx node[38],node[37];
rz(3.5*pi) node[2];
cx node[15],node[4];
cx node[18],node[14];
cx node[37],node[38];
cx node[44],node[43];
sx node[2];
cx node[4],node[15];
cx node[19],node[18];
cx node[43],node[44];
rz(1.0*pi) node[2];
cx node[15],node[4];
cx node[18],node[19];
cx node[44],node[43];
cx node[3],node[2];
cx node[22],node[15];
cx node[19],node[18];
cx node[45],node[44];
cx node[2],node[3];
cx node[15],node[22];
cx node[20],node[19];
cx node[44],node[45];
cx node[3],node[2];
cx node[22],node[15];
cx node[19],node[20];
cx node[45],node[44];
cx node[2],node[1];
cx node[4],node[3];
cx node[20],node[19];
cx node[23],node[22];
cx node[0],node[1];
rz(0.5*pi) node[2];
cx node[3],node[4];
cx node[33],node[20];
cx node[22],node[23];
rz(0.5*pi) node[0];
sx node[2];
cx node[4],node[3];
cx node[20],node[33];
cx node[23],node[22];
sx node[0];
rz(3.5*pi) node[2];
cx node[15],node[4];
cx node[33],node[20];
cx node[24],node[23];
rz(3.5*pi) node[0];
sx node[2];
cx node[4],node[15];
cx node[23],node[24];
cx node[39],node[33];
sx node[0];
rz(1.0*pi) node[2];
cx node[15],node[4];
cx node[24],node[23];
cx node[33],node[39];
rz(1.0*pi) node[0];
cx node[3],node[2];
cx node[22],node[15];
cx node[34],node[24];
cx node[39],node[33];
cx node[14],node[0];
cx node[2],node[3];
cx node[15],node[22];
cx node[24],node[34];
cx node[40],node[39];
cx node[0],node[14];
cx node[3],node[2];
cx node[22],node[15];
cx node[34],node[24];
cx node[39],node[40];
cx node[14],node[0];
cx node[4],node[3];
cx node[23],node[22];
cx node[43],node[34];
cx node[40],node[39];
cx node[0],node[1];
cx node[3],node[4];
cx node[18],node[14];
cx node[22],node[23];
cx node[34],node[43];
cx node[41],node[40];
rz(0.5*pi) node[0];
cx node[2],node[1];
cx node[4],node[3];
cx node[14],node[18];
cx node[23],node[22];
cx node[43],node[34];
cx node[40],node[41];
sx node[0];
rz(0.5*pi) node[2];
cx node[5],node[4];
cx node[18],node[14];
cx node[24],node[23];
cx node[41],node[40];
cx node[44],node[43];
rz(3.5*pi) node[0];
sx node[2];
cx node[4],node[5];
cx node[19],node[18];
cx node[23],node[24];
cx node[53],node[41];
cx node[43],node[44];
sx node[0];
rz(3.5*pi) node[2];
cx node[5],node[4];
cx node[18],node[19];
cx node[24],node[23];
cx node[41],node[53];
cx node[44],node[43];
rz(1.0*pi) node[0];
sx node[2];
cx node[19],node[18];
cx node[25],node[24];
cx node[53],node[41];
cx node[14],node[0];
rz(1.0*pi) node[2];
cx node[20],node[19];
cx node[24],node[25];
cx node[0],node[14];
cx node[3],node[2];
cx node[19],node[20];
cx node[25],node[24];
cx node[14],node[0];
cx node[2],node[3];
cx node[20],node[19];
cx node[26],node[25];
cx node[3],node[2];
cx node[18],node[14];
cx node[33],node[20];
cx node[25],node[26];
cx node[2],node[1];
cx node[4],node[3];
cx node[14],node[18];
cx node[20],node[33];
cx node[26],node[25];
rz(0.5*pi) node[2];
cx node[3],node[4];
cx node[18],node[14];
cx node[33],node[20];
sx node[2];
cx node[4],node[3];
cx node[19],node[18];
cx node[39],node[33];
rz(3.5*pi) node[2];
cx node[15],node[4];
cx node[18],node[19];
cx node[33],node[39];
sx node[2];
cx node[4],node[15];
cx node[19],node[18];
cx node[39],node[33];
rz(1.0*pi) node[2];
cx node[15],node[4];
cx node[20],node[19];
cx node[38],node[39];
cx node[3],node[2];
cx node[22],node[15];
cx node[19],node[20];
cx node[39],node[38];
cx node[2],node[3];
cx node[15],node[22];
cx node[20],node[19];
cx node[38],node[39];
cx node[3],node[2];
cx node[22],node[15];
cx node[33],node[20];
cx node[2],node[1];
cx node[4],node[3];
cx node[20],node[33];
cx node[23],node[22];
cx node[0],node[1];
rz(0.5*pi) node[2];
cx node[3],node[4];
cx node[33],node[20];
cx node[22],node[23];
rz(0.5*pi) node[0];
sx node[2];
cx node[4],node[3];
cx node[23],node[22];
cx node[39],node[33];
sx node[0];
rz(3.5*pi) node[2];
cx node[15],node[4];
cx node[24],node[23];
cx node[33],node[39];
rz(3.5*pi) node[0];
sx node[2];
cx node[4],node[15];
cx node[23],node[24];
cx node[39],node[33];
sx node[0];
rz(1.0*pi) node[2];
cx node[15],node[4];
cx node[24],node[23];
cx node[40],node[39];
rz(1.0*pi) node[0];
cx node[3],node[2];
cx node[22],node[15];
cx node[34],node[24];
cx node[39],node[40];
cx node[14],node[0];
cx node[2],node[3];
cx node[15],node[22];
cx node[24],node[34];
cx node[40],node[39];
cx node[0],node[14];
cx node[3],node[2];
cx node[22],node[15];
cx node[34],node[24];
cx node[41],node[40];
cx node[14],node[0];
cx node[4],node[3];
cx node[23],node[22];
cx node[43],node[34];
cx node[40],node[41];
cx node[0],node[1];
cx node[3],node[4];
cx node[18],node[14];
cx node[22],node[23];
cx node[34],node[43];
cx node[41],node[40];
rz(0.5*pi) node[0];
cx node[2],node[1];
cx node[4],node[3];
cx node[14],node[18];
cx node[23],node[22];
cx node[43],node[34];
sx node[0];
rz(0.5*pi) node[2];
cx node[15],node[4];
cx node[18],node[14];
cx node[24],node[23];
rz(3.5*pi) node[0];
sx node[2];
cx node[4],node[15];
cx node[19],node[18];
cx node[23],node[24];
sx node[0];
rz(3.5*pi) node[2];
cx node[15],node[4];
cx node[18],node[19];
cx node[24],node[23];
rz(1.0*pi) node[0];
sx node[2];
cx node[22],node[15];
cx node[19],node[18];
cx node[25],node[24];
cx node[14],node[0];
rz(1.0*pi) node[2];
cx node[15],node[22];
cx node[20],node[19];
cx node[24],node[25];
cx node[0],node[14];
cx node[3],node[2];
cx node[22],node[15];
cx node[19],node[20];
cx node[25],node[24];
cx node[14],node[0];
cx node[2],node[3];
cx node[20],node[19];
cx node[23],node[22];
cx node[3],node[2];
cx node[18],node[14];
cx node[33],node[20];
cx node[22],node[23];
cx node[2],node[1];
cx node[4],node[3];
cx node[14],node[18];
cx node[20],node[33];
cx node[23],node[22];
rz(0.5*pi) node[2];
cx node[3],node[4];
cx node[18],node[14];
cx node[33],node[20];
cx node[24],node[23];
sx node[2];
cx node[4],node[3];
cx node[19],node[18];
cx node[23],node[24];
cx node[39],node[33];
rz(3.5*pi) node[2];
cx node[15],node[4];
cx node[18],node[19];
cx node[24],node[23];
cx node[33],node[39];
sx node[2];
cx node[4],node[15];
cx node[19],node[18];
cx node[34],node[24];
cx node[39],node[33];
rz(1.0*pi) node[2];
cx node[15],node[4];
cx node[20],node[19];
cx node[24],node[34];
cx node[40],node[39];
cx node[3],node[2];
cx node[22],node[15];
cx node[19],node[20];
cx node[34],node[24];
cx node[39],node[40];
cx node[2],node[3];
cx node[15],node[22];
cx node[20],node[19];
cx node[40],node[39];
cx node[3],node[2];
cx node[22],node[15];
cx node[33],node[20];
cx node[2],node[1];
cx node[4],node[3];
cx node[20],node[33];
cx node[23],node[22];
cx node[0],node[1];
rz(0.5*pi) node[2];
cx node[3],node[4];
cx node[33],node[20];
cx node[22],node[23];
rz(0.5*pi) node[0];
sx node[2];
cx node[4],node[3];
cx node[23],node[22];
cx node[39],node[33];
sx node[0];
rz(3.5*pi) node[2];
cx node[15],node[4];
cx node[24],node[23];
cx node[33],node[39];
rz(3.5*pi) node[0];
sx node[2];
cx node[4],node[15];
cx node[23],node[24];
cx node[39],node[33];
sx node[0];
rz(1.0*pi) node[2];
cx node[15],node[4];
cx node[24],node[23];
rz(1.0*pi) node[0];
cx node[3],node[2];
cx node[22],node[15];
cx node[14],node[0];
cx node[2],node[3];
cx node[15],node[22];
cx node[0],node[14];
cx node[3],node[2];
cx node[22],node[15];
cx node[14],node[0];
cx node[4],node[3];
cx node[23],node[22];
cx node[0],node[1];
cx node[3],node[4];
cx node[18],node[14];
cx node[22],node[23];
rz(0.5*pi) node[0];
cx node[2],node[1];
cx node[4],node[3];
cx node[14],node[18];
cx node[23],node[22];
sx node[0];
rz(0.5*pi) node[2];
cx node[15],node[4];
cx node[18],node[14];
rz(3.5*pi) node[0];
sx node[2];
cx node[4],node[15];
cx node[19],node[18];
sx node[0];
rz(3.5*pi) node[2];
cx node[15],node[4];
cx node[18],node[19];
rz(1.0*pi) node[0];
sx node[2];
cx node[22],node[15];
cx node[19],node[18];
cx node[14],node[0];
rz(1.0*pi) node[2];
cx node[15],node[22];
cx node[20],node[19];
cx node[0],node[14];
cx node[3],node[2];
cx node[22],node[15];
cx node[19],node[20];
cx node[14],node[0];
cx node[2],node[3];
cx node[20],node[19];
cx node[3],node[2];
cx node[18],node[14];
cx node[33],node[20];
cx node[2],node[1];
cx node[4],node[3];
cx node[14],node[18];
cx node[20],node[33];
cx node[0],node[1];
rz(0.5*pi) node[2];
cx node[3],node[4];
cx node[18],node[14];
cx node[33],node[20];
rz(0.5*pi) node[0];
sx node[2];
cx node[4],node[3];
cx node[19],node[18];
sx node[0];
rz(3.5*pi) node[2];
cx node[15],node[4];
cx node[18],node[19];
rz(3.5*pi) node[0];
sx node[2];
cx node[4],node[15];
cx node[19],node[18];
sx node[0];
rz(1.0*pi) node[2];
cx node[15],node[4];
cx node[20],node[19];
rz(1.0*pi) node[0];
cx node[3],node[2];
cx node[19],node[20];
cx node[14],node[0];
cx node[2],node[3];
cx node[20],node[19];
cx node[0],node[14];
cx node[3],node[2];
cx node[14],node[0];
cx node[4],node[3];
cx node[0],node[1];
cx node[3],node[4];
cx node[18],node[14];
rz(0.5*pi) node[0];
cx node[2],node[1];
cx node[4],node[3];
cx node[14],node[18];
sx node[0];
rz(0.5*pi) node[2];
cx node[18],node[14];
rz(3.5*pi) node[0];
sx node[2];
cx node[19],node[18];
sx node[0];
rz(3.5*pi) node[2];
cx node[18],node[19];
rz(1.0*pi) node[0];
sx node[2];
cx node[19],node[18];
cx node[14],node[0];
rz(1.0*pi) node[2];
cx node[0],node[14];
cx node[3],node[2];
cx node[14],node[0];
cx node[2],node[3];
cx node[3],node[2];
cx node[18],node[14];
cx node[2],node[1];
cx node[14],node[18];
cx node[0],node[1];
rz(0.5*pi) node[2];
cx node[18],node[14];
rz(0.5*pi) node[0];
sx node[2];
sx node[0];
rz(3.5*pi) node[2];
rz(3.5*pi) node[0];
sx node[2];
sx node[0];
rz(1.0*pi) node[2];
rz(1.0*pi) node[0];
cx node[14],node[0];
cx node[0],node[14];
cx node[14],node[0];
cx node[0],node[1];
rz(0.5*pi) node[0];
sx node[0];
rz(3.5*pi) node[0];
sx node[0];
rz(1.0*pi) node[0];
barrier node[28],node[16],node[21],node[8],node[42],node[61],node[56],node[59],node[52],node[60],node[27],node[7],node[45],node[37],node[6],node[44],node[53],node[26],node[5],node[38],node[41],node[43],node[25],node[34],node[40],node[39],node[24],node[23],node[22],node[33],node[20],node[15],node[4],node[19],node[18],node[3],node[2],node[14],node[0],node[1];
measure node[28] -> c[0];
measure node[16] -> c[1];
measure node[21] -> c[2];
measure node[8] -> c[3];
measure node[42] -> c[4];
measure node[61] -> c[5];
measure node[56] -> c[6];
measure node[59] -> c[7];
measure node[52] -> c[8];
measure node[60] -> c[9];
measure node[27] -> c[10];
measure node[7] -> c[11];
measure node[45] -> c[12];
measure node[37] -> c[13];
measure node[6] -> c[14];
measure node[44] -> c[15];
measure node[53] -> c[16];
measure node[26] -> c[17];
measure node[5] -> c[18];
measure node[38] -> c[19];
measure node[41] -> c[20];
measure node[43] -> c[21];
measure node[25] -> c[22];
measure node[34] -> c[23];
measure node[40] -> c[24];
measure node[39] -> c[25];
measure node[24] -> c[26];
measure node[23] -> c[27];
measure node[22] -> c[28];
measure node[33] -> c[29];
measure node[20] -> c[30];
measure node[15] -> c[31];
measure node[4] -> c[32];
measure node[19] -> c[33];
measure node[18] -> c[34];
measure node[3] -> c[35];
measure node[2] -> c[36];
measure node[14] -> c[37];
measure node[0] -> c[38];
