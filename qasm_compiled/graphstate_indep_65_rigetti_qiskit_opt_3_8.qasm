OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg meas[65];
rz(-pi/2) q[2];
rx(-pi/2) q[2];
rz(pi/2) q[3];
rx(pi/2) q[3];
rz(pi/2) q[4];
rx(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
rx(pi/2) q[5];
rz(pi/2) q[5];
cz q[4],q[5];
rx(-pi/2) q[4];
cz q[3],q[4];
rx(pi/2) q[3];
rx(pi/2) q[4];
cz q[3],q[4];
rx(-pi/2) q[3];
rx(pi/2) q[4];
cz q[3],q[4];
cz q[3],q[2];
rx(-pi/2) q[4];
rz(-pi/2) q[4];
cz q[5],q[4];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
rx(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[10];
rx(pi/2) q[10];
rz(pi/2) q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi) q[12];
rx(-pi/2) q[12];
rz(-pi/2) q[13];
rx(-pi/2) q[13];
rz(-pi/2) q[14];
rx(-pi/2) q[14];
rz(-pi) q[15];
rx(-pi/2) q[15];
rz(pi/2) q[16];
rx(pi/2) q[16];
rz(pi/2) q[16];
rz(-pi/2) q[17];
rx(-pi/2) q[17];
rz(-pi/2) q[18];
rx(-pi/2) q[18];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rx(pi/2) q[19];
rz(-pi) q[20];
rx(-pi/2) q[20];
cz q[19],q[20];
rx(pi/2) q[19];
rz(-pi/2) q[20];
rx(pi/2) q[20];
cz q[19],q[20];
rx(-pi/2) q[19];
rz(-pi) q[19];
rx(pi/2) q[20];
cz q[19],q[20];
rz(-0.8059157) q[20];
rx(pi/2) q[20];
rz(-pi) q[21];
rx(-pi/2) q[21];
rz(-pi) q[22];
rx(-pi/2) q[22];
rz(pi/2) q[23];
rx(pi/2) q[23];
rz(pi/2) q[23];
rz(pi/2) q[24];
rx(pi/2) q[24];
rz(-pi) q[25];
rx(-pi/2) q[25];
rz(pi/2) q[26];
rx(pi/2) q[26];
rz(pi/2) q[27];
rx(pi/2) q[27];
rz(pi/2) q[27];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
cz q[27],q[28];
rx(-pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-pi/2) q[27];
rz(-pi) q[29];
rx(-pi/2) q[29];
cz q[18],q[29];
rx(pi/2) q[18];
rz(-pi/2) q[29];
rx(pi/2) q[29];
cz q[18],q[29];
rx(-pi/2) q[18];
rz(-pi/2) q[18];
rx(pi/2) q[29];
cz q[18],q[29];
cz q[18],q[17];
cz q[17],q[16];
cz q[23],q[16];
rx(-pi/2) q[29];
rz(pi) q[29];
rz(-pi) q[30];
rx(-pi/2) q[30];
cz q[29],q[30];
rx(pi/2) q[29];
rz(-pi/2) q[30];
rx(pi/2) q[30];
cz q[29],q[30];
rx(-pi/2) q[29];
rz(pi/2) q[29];
rx(pi/2) q[30];
cz q[29],q[30];
cz q[18],q[29];
rx(-pi/2) q[30];
rz(pi) q[30];
rz(-pi) q[31];
rx(-pi/2) q[31];
cz q[30],q[31];
rx(pi/2) q[30];
rz(-pi/2) q[31];
rx(pi/2) q[31];
cz q[30],q[31];
rx(-pi/2) q[30];
rz(pi/2) q[30];
rx(pi/2) q[31];
cz q[30],q[31];
rx(pi/2) q[30];
rz(pi) q[30];
cz q[29],q[30];
rx(pi/2) q[29];
rz(pi) q[29];
rx(pi/2) q[30];
rz(pi) q[30];
cz q[30],q[29];
rx(pi/2) q[29];
rz(pi) q[29];
rx(pi/2) q[30];
rz(pi) q[30];
cz q[29],q[30];
rx(pi/2) q[30];
rz(pi) q[30];
rz(-pi/2) q[31];
rx(-pi/2) q[31];
cz q[24],q[31];
rx(pi/2) q[24];
rz(-pi/2) q[31];
rx(pi/2) q[31];
cz q[24],q[31];
rx(-pi/2) q[24];
rz(3*pi/2) q[24];
rx(pi/2) q[31];
cz q[24],q[31];
cz q[24],q[25];
rx(pi/2) q[24];
rz(-pi/2) q[25];
rx(pi/2) q[25];
cz q[24],q[25];
rx(-pi/2) q[24];
rz(pi/2) q[24];
rx(pi/2) q[25];
cz q[24],q[25];
rx(pi/2) q[24];
rz(pi/2) q[24];
rz(pi/2) q[25];
rx(-pi/2) q[31];
rz(-pi/2) q[32];
rx(-pi/2) q[32];
rz(pi/2) q[33];
rx(pi/2) q[33];
rz(pi/2) q[33];
cz q[32],q[33];
rx(pi/2) q[32];
rz(pi/2) q[34];
rx(pi/2) q[34];
rz(pi/2) q[34];
rz(pi/2) q[35];
rx(pi/2) q[35];
rz(pi/2) q[35];
rz(pi/2) q[36];
rx(pi/2) q[36];
rz(pi/2) q[36];
cz q[36],q[35];
cz q[35],q[34];
cz q[33],q[34];
rz(pi/2) q[37];
rx(pi/2) q[37];
rz(pi/2) q[37];
cz q[26],q[37];
cz q[36],q[37];
rz(pi/2) q[38];
rx(pi/2) q[38];
rz(pi/2) q[38];
cz q[25],q[38];
rz(-pi) q[39];
rx(-pi/2) q[39];
cz q[32],q[39];
rx(pi/2) q[32];
rz(-pi/2) q[39];
rx(pi/2) q[39];
cz q[32],q[39];
rx(-pi/2) q[32];
rz(pi/2) q[32];
rx(pi/2) q[39];
cz q[32],q[39];
rx(pi/2) q[32];
rz(pi/2) q[32];
rz(pi/2) q[39];
cz q[39],q[38];
rz(-pi/2) q[41];
rx(-pi/2) q[41];
rz(pi/2) q[47];
rx(pi/2) q[47];
rz(pi/2) q[47];
rz(-pi/2) q[48];
rx(-pi/2) q[48];
rz(-pi/2) q[49];
rx(-pi/2) q[49];
rz(-pi/2) q[50];
rx(-pi/2) q[50];
cz q[50],q[49];
rx(pi/2) q[49];
rz(pi) q[49];
rx(pi/2) q[50];
rz(-pi/2) q[51];
rx(-pi/2) q[51];
rz(-pi/2) q[52];
rx(-pi/2) q[52];
cz q[52],q[51];
rx(pi/2) q[51];
rz(pi) q[51];
rx(pi/2) q[52];
rz(-pi) q[53];
rx(-pi/2) q[53];
cz q[52],q[53];
rx(pi/2) q[52];
rz(-pi/2) q[53];
rx(pi/2) q[53];
cz q[52],q[53];
rx(-pi/2) q[52];
rz(pi/2) q[52];
rx(pi/2) q[53];
cz q[52],q[53];
rx(-pi/2) q[53];
cz q[42],q[53];
rz(-pi/2) q[42];
rx(-pi/2) q[42];
rx(pi/2) q[53];
rz(pi) q[53];
cz q[53],q[42];
rx(pi/2) q[42];
rz(pi/2) q[42];
rx(pi/2) q[53];
rz(pi) q[53];
cz q[42],q[53];
cz q[42],q[41];
rx(pi/2) q[53];
rz(pi) q[53];
rz(pi/2) q[54];
rx(pi/2) q[54];
rz(-pi/2) q[55];
rx(-pi/2) q[55];
cz q[48],q[55];
cz q[48],q[49];
rx(pi/2) q[48];
rz(pi) q[48];
rx(pi/2) q[49];
rz(pi) q[49];
cz q[49],q[48];
rx(pi/2) q[48];
rz(pi) q[48];
rx(pi/2) q[49];
rz(pi) q[49];
cz q[48],q[49];
cz q[48],q[11];
rx(-pi/2) q[11];
cz q[10],q[11];
rx(pi/2) q[10];
rx(pi/2) q[11];
cz q[10],q[11];
rx(-pi/2) q[10];
rz(pi/2) q[10];
rx(pi/2) q[11];
cz q[10],q[11];
rx(pi/2) q[10];
cz q[10],q[21];
rx(pi/2) q[10];
cz q[11],q[12];
rx(pi/2) q[11];
rz(-pi/2) q[12];
rx(pi/2) q[12];
cz q[11],q[12];
rx(-pi/2) q[11];
rz(pi) q[11];
rx(pi/2) q[12];
cz q[11],q[12];
rx(-pi/2) q[11];
rz(pi) q[12];
rz(-pi/2) q[21];
rx(pi/2) q[21];
cz q[10],q[21];
rx(-pi/2) q[10];
rz(3*pi/2) q[10];
rx(pi/2) q[21];
cz q[10],q[21];
cz q[10],q[11];
rx(pi/2) q[10];
rz(-pi/2) q[11];
rx(pi/2) q[11];
cz q[10],q[11];
rx(-pi/2) q[10];
rz(3*pi/2) q[10];
rx(pi/2) q[11];
cz q[10],q[11];
rz(pi) q[11];
cz q[12],q[11];
rx(pi/2) q[11];
rz(pi) q[11];
rx(pi/2) q[12];
rz(pi) q[12];
cz q[11],q[12];
rx(pi/2) q[11];
rz(pi/2) q[11];
rx(pi/2) q[12];
rz(pi) q[12];
cz q[12],q[11];
rx(-pi/2) q[11];
rx(-pi/2) q[21];
rz(pi) q[21];
cz q[21],q[22];
rx(pi/2) q[21];
rz(-pi/2) q[22];
rx(pi/2) q[22];
cz q[21],q[22];
rx(-pi/2) q[21];
rz(pi/2) q[21];
rx(pi/2) q[22];
cz q[21],q[22];
rx(-pi/2) q[21];
rz(pi/2) q[21];
rx(pi/2) q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rz(-pi/2) q[21];
rx(pi/2) q[21];
cz q[10],q[21];
rx(-pi/2) q[10];
rz(3*pi/2) q[10];
rx(pi/2) q[21];
cz q[10],q[21];
cz q[10],q[11];
rx(pi/2) q[10];
rx(pi/2) q[11];
cz q[10],q[11];
rx(-pi/2) q[10];
rz(pi/2) q[10];
rx(pi/2) q[11];
cz q[10],q[11];
rx(pi/2) q[10];
rz(pi) q[10];
rz(pi) q[11];
rz(-2.3454553) q[21];
rx(-pi/2) q[21];
cz q[20],q[21];
rx(-pi/2) q[20];
rz(-pi/2) q[20];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[20],q[21];
rx(2.3454553) q[20];
rz(-pi/2) q[20];
rx(pi/2) q[21];
rz(0.8059157) q[21];
rz(pi/2) q[22];
cz q[23],q[22];
rx(pi/2) q[49];
rz(pi) q[49];
rx(pi/2) q[55];
rz(pi) q[55];
cz q[48],q[55];
rx(pi/2) q[48];
rz(pi) q[48];
rx(pi/2) q[55];
rz(pi) q[55];
cz q[55],q[48];
rx(pi/2) q[48];
rz(pi) q[48];
rx(pi/2) q[55];
rz(pi/2) q[55];
cz q[48],q[55];
rx(-pi/2) q[55];
rz(pi/2) q[56];
rx(pi/2) q[56];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
rz(-pi/2) q[58];
rx(-pi/2) q[58];
rz(-pi/2) q[59];
rx(-pi/2) q[59];
cz q[59],q[58];
rx(pi/2) q[58];
rx(pi/2) q[59];
rz(pi) q[59];
rz(pi/2) q[60];
rx(pi/2) q[60];
rz(-pi) q[61];
rx(-pi/2) q[61];
cz q[50],q[61];
rx(pi/2) q[50];
rz(-pi/2) q[61];
rx(pi/2) q[61];
cz q[50],q[61];
rx(-pi/2) q[50];
rz(pi/2) q[50];
rx(pi/2) q[61];
cz q[50],q[61];
rx(pi/2) q[50];
rz(pi) q[50];
rz(-pi/2) q[61];
rx(-pi/2) q[61];
cz q[60],q[61];
rx(pi/2) q[60];
rz(-pi/2) q[61];
rx(pi/2) q[61];
cz q[60],q[61];
rx(-pi/2) q[60];
rz(pi/2) q[60];
rx(pi/2) q[61];
cz q[60],q[61];
rx(pi/2) q[60];
rz(pi) q[60];
cz q[60],q[59];
rx(pi/2) q[59];
rz(pi) q[59];
rx(pi/2) q[60];
rz(pi) q[60];
cz q[59],q[60];
rx(pi/2) q[59];
rz(pi) q[59];
rx(pi/2) q[60];
rz(pi) q[60];
cz q[60],q[59];
rx(pi/2) q[59];
rz(pi) q[59];
rx(pi/2) q[60];
rx(-pi/2) q[61];
rz(2.9604525) q[61];
rx(-pi/2) q[61];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
cz q[49],q[62];
cz q[49],q[50];
rx(pi/2) q[49];
rz(pi) q[49];
rx(pi/2) q[50];
rz(pi) q[50];
cz q[50],q[49];
rx(pi/2) q[49];
rz(pi) q[49];
rx(pi/2) q[50];
rz(pi) q[50];
cz q[49],q[50];
cz q[48],q[49];
rx(pi/2) q[48];
rx(pi/2) q[50];
rz(pi) q[50];
cz q[50],q[51];
rx(pi/2) q[50];
rz(pi) q[50];
rx(pi/2) q[51];
rz(pi) q[51];
cz q[51],q[50];
rx(pi/2) q[50];
rz(pi) q[50];
rx(pi/2) q[51];
rz(pi) q[51];
cz q[50],q[51];
rx(pi/2) q[50];
rx(pi/2) q[51];
rz(pi) q[51];
rx(pi/2) q[62];
rz(-pi) q[63];
rx(-pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(-pi/2) q[63];
rx(pi/2) q[63];
cz q[62],q[63];
rx(-pi/2) q[62];
rz(0.65124239) q[62];
rx(pi/2) q[63];
cz q[62],q[63];
cz q[49],q[62];
rx(pi/2) q[49];
rx(pi/2) q[62];
cz q[61],q[62];
rx(-pi/2) q[61];
rz(-pi/2) q[61];
rx(pi/2) q[62];
cz q[61],q[62];
rx(0.65124239) q[61];
cz q[60],q[61];
rx(pi/2) q[60];
rz(-pi/2) q[61];
rx(pi/2) q[61];
cz q[60],q[61];
rx(-pi/2) q[60];
rz(3*pi/2) q[60];
rx(pi/2) q[61];
cz q[60],q[61];
rx(-pi/2) q[61];
rz(pi/2) q[61];
rx(-pi/2) q[61];
cz q[50],q[61];
rx(pi/2) q[50];
rz(-pi/2) q[61];
rx(pi/2) q[61];
cz q[50],q[61];
rx(-pi/2) q[50];
rz(pi/2) q[50];
rx(pi/2) q[61];
cz q[50],q[61];
rx(pi/2) q[50];
rz(pi) q[50];
cz q[51],q[50];
rx(pi/2) q[50];
rz(pi) q[50];
rx(pi/2) q[51];
rz(pi) q[51];
cz q[50],q[51];
rx(pi/2) q[50];
rz(pi) q[50];
rx(pi/2) q[51];
rz(pi/2) q[51];
cz q[51],q[50];
rx(pi/2) q[50];
rz(pi/2) q[50];
cz q[51],q[52];
rx(pi/2) q[52];
rz(pi) q[52];
cz q[53],q[52];
rx(pi/2) q[52];
rz(pi) q[52];
rx(pi/2) q[53];
rz(pi) q[53];
cz q[52],q[53];
rx(pi/2) q[52];
rz(pi) q[52];
rx(pi/2) q[53];
rz(pi/2) q[53];
cz q[53],q[52];
rx(pi/2) q[52];
rz(pi/2) q[52];
cz q[53],q[54];
cz q[54],q[55];
rx(pi/2) q[54];
rx(pi/2) q[55];
cz q[54],q[55];
rx(-pi/2) q[54];
rz(pi/2) q[54];
rx(pi/2) q[55];
cz q[54],q[55];
cz q[41],q[54];
rx(pi/2) q[41];
rz(pi) q[41];
rx(pi/2) q[54];
rz(pi) q[54];
cz q[54],q[41];
rx(pi/2) q[41];
rz(pi) q[41];
rx(pi/2) q[54];
rz(pi) q[54];
cz q[41],q[54];
rx(pi/2) q[41];
rx(pi/2) q[54];
rz(pi) q[54];
rz(pi) q[55];
cz q[12],q[55];
rx(pi/2) q[12];
rz(pi) q[12];
rx(pi/2) q[55];
rz(pi) q[55];
cz q[55],q[12];
rx(pi/2) q[12];
rz(pi) q[12];
rx(pi/2) q[55];
rz(pi) q[55];
cz q[12],q[55];
cz q[12],q[13];
rx(pi/2) q[12];
rz(pi) q[12];
cz q[14],q[13];
rx(pi/2) q[13];
rz(pi) q[13];
rx(pi/2) q[14];
cz q[14],q[15];
rx(pi/2) q[14];
rz(-pi/2) q[15];
rx(pi/2) q[15];
cz q[14],q[15];
rx(-pi/2) q[14];
rz(pi/2) q[14];
rx(pi/2) q[15];
cz q[14],q[15];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[15];
cz q[15],q[8];
cz q[2],q[13];
rx(pi/2) q[13];
rz(pi) q[13];
rx(pi/2) q[2];
rz(pi) q[2];
cz q[13],q[2];
rx(pi/2) q[13];
rz(pi) q[13];
rx(pi/2) q[2];
rz(pi/2) q[2];
cz q[2],q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rx(pi/2) q[55];
rz(pi) q[55];
cz q[54],q[55];
rx(pi/2) q[54];
cz q[55],q[12];
rx(pi/2) q[12];
rz(pi) q[12];
rx(pi/2) q[55];
rz(pi) q[55];
cz q[12],q[55];
rx(pi/2) q[12];
rz(pi) q[12];
rx(pi/2) q[55];
rz(-pi/2) q[55];
cz q[55],q[12];
rx(pi/2) q[12];
rz(pi) q[12];
cz q[12],q[11];
rx(pi/2) q[11];
rz(pi) q[11];
rx(pi/2) q[12];
rz(pi) q[12];
cz q[11],q[12];
rx(pi/2) q[11];
rz(pi) q[11];
rx(pi/2) q[12];
rz(pi/2) q[12];
cz q[12],q[11];
rx(pi/2) q[11];
rz(pi) q[11];
cz q[13],q[12];
rx(-pi/2) q[12];
rx(-pi/2) q[55];
rz(-pi/2) q[61];
rx(-pi/2) q[61];
cz q[60],q[61];
rx(pi/2) q[60];
rz(-pi/2) q[61];
rx(pi/2) q[61];
cz q[60],q[61];
rx(-pi/2) q[60];
rz(pi/2) q[60];
rx(pi/2) q[61];
cz q[60],q[61];
rx(-pi/2) q[61];
rz(-pi/2) q[61];
rx(pi/2) q[62];
rz(0.18114012) q[62];
rz(pi/2) q[63];
cz q[9],q[8];
cz q[9],q[10];
cz q[11],q[10];
rx(pi/2) q[10];
rx(pi/2) q[11];
cz q[11],q[12];
rx(pi/2) q[11];
rx(pi/2) q[12];
cz q[11],q[12];
rx(-pi/2) q[11];
rz(pi/2) q[11];
rx(pi/2) q[12];
cz q[11],q[12];
rx(-pi/2) q[11];
rz(pi/2) q[11];
rx(pi/2) q[11];
cz q[10],q[11];
rx(pi/2) q[10];
rz(-pi/2) q[11];
rx(pi/2) q[11];
cz q[10],q[11];
rx(-pi/2) q[10];
rz(pi/2) q[10];
rx(pi/2) q[11];
cz q[10],q[11];
rx(pi/2) q[10];
rz(pi) q[10];
rz(pi/2) q[11];
rz(pi/2) q[12];
cz q[21],q[10];
rx(pi/2) q[10];
rz(pi) q[10];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rz(pi) q[10];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[21],q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[64];
rx(pi/2) q[64];
rz(pi/2) q[65];
rx(pi/2) q[65];
rz(-pi/2) q[66];
rx(-pi/2) q[66];
rz(-pi/2) q[67];
rx(-pi/2) q[67];
cz q[66],q[67];
rz(-pi/2) q[68];
rx(-pi/2) q[68];
cz q[67],q[68];
rx(pi/2) q[67];
rz(pi) q[67];
rx(pi/2) q[68];
rz(pi) q[68];
rz(-pi) q[69];
rx(-pi/2) q[69];
cz q[58],q[69];
rx(pi/2) q[58];
rz(-pi/2) q[69];
rx(pi/2) q[69];
cz q[58],q[69];
rx(-pi/2) q[58];
rz(pi/2) q[58];
rx(pi/2) q[69];
cz q[58],q[69];
cz q[59],q[58];
cz q[57],q[58];
rx(pi/2) q[57];
cz q[59],q[60];
rx(pi/2) q[59];
rz(pi) q[59];
rx(pi/2) q[60];
rz(pi) q[60];
cz q[60],q[59];
rx(pi/2) q[59];
rz(pi/2) q[59];
rx(pi/2) q[60];
rz(pi) q[60];
cz q[59],q[60];
rx(pi/2) q[60];
rz(pi/2) q[60];
rz(pi) q[69];
cz q[69],q[68];
rx(pi/2) q[68];
rz(pi) q[68];
rx(pi/2) q[69];
rz(pi) q[69];
cz q[68],q[69];
rx(pi/2) q[68];
rz(pi) q[68];
rx(pi/2) q[69];
rz(pi) q[69];
cz q[69],q[68];
rx(pi/2) q[68];
rz(pi) q[68];
cz q[68],q[67];
rx(pi/2) q[67];
rz(pi) q[67];
rx(pi/2) q[68];
rz(pi) q[68];
cz q[67],q[68];
rx(pi/2) q[67];
rz(pi) q[67];
rx(pi/2) q[68];
rz(-pi/2) q[68];
cz q[68],q[67];
rx(pi/2) q[67];
rz(pi) q[67];
rx(-pi/2) q[68];
rx(pi/2) q[69];
rz(pi/2) q[70];
rx(pi/2) q[70];
rz(-pi) q[71];
rx(-pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(-pi/2) q[71];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(-pi/2) q[70];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(-pi/2) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(-pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
cz q[56],q[63];
rx(-pi/2) q[57];
rz(-3*pi/2) q[57];
rx(-pi/2) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rx(pi/2) q[63];
cz q[20],q[63];
rx(-pi/2) q[20];
rz(pi/2) q[20];
rx(pi/2) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[21],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rz(pi/2) q[20];
rx(pi/2) q[21];
rz(pi/2) q[21];
cz q[21],q[20];
rx(-pi/2) q[20];
rz(-pi/2) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rx(-pi/2) q[63];
rx(-pi/2) q[70];
rz(pi) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[28],q[71];
cz q[28],q[29];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(-pi/2) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(-pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(-pi/2) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[19],q[56];
rx(-pi/2) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(-pi/2) q[19];
rx(pi/2) q[19];
cz q[18],q[19];
rx(-pi/2) q[18];
rz(pi/2) q[18];
rx(pi/2) q[19];
cz q[18],q[19];
cz q[17],q[18];
rx(pi/2) q[17];
rz(pi) q[17];
rx(pi/2) q[18];
rz(pi) q[18];
cz q[18],q[17];
rx(pi/2) q[17];
rz(pi/2) q[17];
rx(pi/2) q[18];
rz(pi) q[18];
cz q[17],q[18];
cz q[17],q[30];
rx(pi/2) q[18];
rz(pi) q[18];
rx(-pi/2) q[19];
rz(pi) q[19];
cz q[19],q[20];
rx(pi/2) q[19];
rx(pi/2) q[20];
cz q[19],q[20];
rx(-pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[20];
cz q[19],q[20];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi/2) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[18],q[19];
rx(pi/2) q[19];
rz(pi/2) q[19];
rx(-pi/2) q[20];
rz(-pi/2) q[20];
rx(pi/2) q[30];
rz(pi) q[30];
cz q[31],q[30];
rx(pi/2) q[30];
rz(pi) q[30];
rx(pi/2) q[31];
rz(pi) q[31];
cz q[30],q[31];
rx(pi/2) q[30];
rz(pi) q[30];
rx(pi/2) q[31];
rz(pi/2) q[31];
cz q[31],q[30];
rx(pi/2) q[30];
rz(pi) q[30];
rx(pi/2) q[56];
rz(pi) q[57];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rx(pi/2) q[70];
rz(pi/2) q[70];
cz q[70],q[57];
rx(-pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(-pi/2) q[71];
rx(pi/2) q[71];
cz q[64],q[71];
rx(-pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
rx(-pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
cz q[69],q[70];
rx(pi/2) q[69];
rx(pi/2) q[70];
cz q[69],q[70];
rx(-pi/2) q[69];
rz(pi) q[69];
rx(pi/2) q[70];
cz q[69],q[70];
rx(-pi/2) q[69];
rx(-pi/2) q[70];
rz(-pi/2) q[72];
rx(-pi/2) q[72];
rz(pi/2) q[73];
rx(pi/2) q[73];
rz(pi/2) q[73];
rz(pi/2) q[74];
rx(pi/2) q[74];
rz(pi/2) q[74];
cz q[74],q[73];
cz q[73],q[72];
rx(pi/2) q[72];
rz(pi) q[72];
rz(-pi/2) q[75];
rx(-pi/2) q[75];
cz q[74],q[75];
rx(pi/2) q[75];
rz(-pi) q[76];
rx(-pi/2) q[76];
cz q[75],q[76];
rx(pi/2) q[75];
rz(-pi/2) q[76];
rx(pi/2) q[76];
cz q[75],q[76];
rx(-pi/2) q[75];
rz(pi/2) q[75];
rx(pi/2) q[76];
cz q[75],q[76];
rx(pi/2) q[75];
rz(pi/2) q[75];
rz(pi/2) q[76];
rz(pi/2) q[77];
rx(pi/2) q[77];
rz(3.6677513) q[77];
cz q[66],q[77];
rx(pi/2) q[66];
rz(pi) q[66];
cz q[67],q[66];
rx(pi/2) q[66];
rz(pi) q[66];
rx(pi/2) q[67];
rz(pi) q[67];
cz q[66],q[67];
rx(pi/2) q[66];
rx(pi/2) q[67];
rz(-pi/2) q[67];
cz q[67],q[66];
rx(-pi/2) q[67];
rx(pi/2) q[77];
rz(-pi) q[78];
rx(-pi/2) q[78];
cz q[65],q[78];
rx(pi/2) q[65];
rz(-pi/2) q[78];
rx(pi/2) q[78];
cz q[65],q[78];
rx(-pi/2) q[65];
rz(pi/2) q[65];
rx(pi/2) q[78];
cz q[65],q[78];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[64],q[65];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[65],q[64];
rx(pi/2) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[64],q[65];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[71],q[64];
cz q[65],q[64];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rx(pi/2) q[71];
rz(-pi/2) q[71];
cz q[71],q[70];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[28],q[27];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[27],q[28];
rx(pi/2) q[28];
rz(pi/2) q[28];
cz q[29],q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[30],q[29];
rx(pi/2) q[29];
rz(pi) q[29];
rx(pi/2) q[30];
rz(pi) q[30];
cz q[29],q[30];
rx(pi/2) q[29];
rz(pi) q[29];
rx(pi/2) q[30];
rz(pi/2) q[30];
cz q[30],q[29];
rx(pi/2) q[29];
rz(pi/2) q[29];
cz q[18],q[29];
cz q[28],q[29];
rx(-pi/2) q[71];
rx(-pi/2) q[78];
rz(pi) q[78];
rz(-pi) q[79];
rx(-pi/2) q[79];
cz q[78],q[79];
rx(pi/2) q[78];
rz(-pi/2) q[79];
rx(pi/2) q[79];
cz q[78],q[79];
rx(-pi/2) q[78];
rz(1.8848511) q[78];
rx(pi/2) q[79];
cz q[78],q[79];
rx(-pi/2) q[78];
cz q[77],q[78];
rx(-pi/2) q[77];
rz(-pi/2) q[77];
rx(pi/2) q[78];
rz(pi/2) q[78];
cz q[77],q[78];
rx(-1.8848511) q[77];
rz(pi/2) q[77];
cz q[66],q[77];
rx(pi/2) q[66];
rx(pi/2) q[77];
cz q[66],q[77];
rx(-pi/2) q[66];
rz(3*pi/2) q[66];
rx(pi/2) q[77];
cz q[66],q[77];
cz q[66],q[67];
rx(pi/2) q[66];
rz(-pi/2) q[67];
rx(pi/2) q[67];
cz q[66],q[67];
rx(-pi/2) q[66];
rz(pi/2) q[66];
rx(pi/2) q[67];
cz q[66],q[67];
rx(pi/2) q[66];
rz(pi/2) q[66];
rx(-pi/2) q[67];
rz(pi) q[67];
cz q[67],q[68];
rx(pi/2) q[67];
rz(-pi/2) q[68];
rx(pi/2) q[68];
cz q[67],q[68];
rx(-pi/2) q[67];
rz(pi/2) q[67];
rx(pi/2) q[68];
cz q[67],q[68];
rx(pi/2) q[67];
rz(pi/2) q[67];
rx(-pi/2) q[68];
rz(pi) q[68];
cz q[68],q[69];
rx(pi/2) q[68];
rz(-pi/2) q[69];
rx(pi/2) q[69];
cz q[68],q[69];
rx(-pi/2) q[68];
rz(pi/2) q[68];
rx(pi/2) q[69];
cz q[68],q[69];
rx(pi/2) q[68];
rz(pi/2) q[68];
rz(pi) q[69];
cz q[58],q[69];
rx(pi/2) q[58];
rz(pi) q[58];
rx(pi/2) q[69];
rz(pi) q[69];
cz q[69],q[58];
rx(pi/2) q[58];
rz(pi/2) q[58];
rx(pi/2) q[69];
rz(pi) q[69];
cz q[58],q[69];
cz q[59],q[58];
rx(pi/2) q[69];
rz(pi/2) q[69];
rz(pi/2) q[77];
cz q[77],q[76];
rx(-0.52615869) q[78];
rz(pi/2) q[78];
rx(pi/2) q[78];
rz(pi) q[79];
cz q[79],q[72];
rx(pi/2) q[72];
rz(pi) q[72];
rx(pi/2) q[79];
rz(pi) q[79];
cz q[72],q[79];
rx(pi/2) q[72];
rz(pi) q[72];
rx(pi/2) q[79];
rz(-pi/2) q[79];
cz q[79],q[72];
rx(pi/2) q[72];
rz(pi/2) q[72];
rx(-pi/2) q[79];
cz q[78],q[79];
rx(pi/2) q[78];
rz(-pi/2) q[79];
rx(pi/2) q[79];
cz q[78],q[79];
rx(-pi/2) q[78];
rz(pi/2) q[78];
rx(pi/2) q[79];
cz q[78],q[79];
cz q[65],q[78];
rx(pi/2) q[65];
rz(pi) q[65];
rx(pi/2) q[78];
rz(pi) q[78];
cz q[78],q[65];
rx(pi/2) q[65];
rz(-pi/2) q[65];
rx(pi/2) q[78];
rz(pi) q[78];
cz q[65],q[78];
rx(-pi/2) q[65];
cz q[64],q[65];
rx(pi/2) q[64];
rz(-pi/2) q[65];
rx(pi/2) q[65];
cz q[64],q[65];
rx(-pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[65];
cz q[64],q[65];
rx(pi/2) q[64];
cz q[64],q[71];
rx(pi/2) q[64];
rx(-pi/2) q[65];
rz(-pi/2) q[65];
rz(-pi/2) q[71];
rx(pi/2) q[71];
cz q[64],q[71];
rx(-pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi/2) q[64];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(-pi/2) q[71];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(-pi/2) q[70];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(-pi/2) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(3*pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[57];
rz(-pi/2) q[63];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[56];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(-pi/2) q[63];
rx(pi/2) q[63];
cz q[62],q[63];
rx(-pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
cz q[62],q[63];
rx(-pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(-pi/2) q[62];
rx(pi/2) q[62];
cz q[49],q[62];
rx(-pi/2) q[49];
rz(pi/2) q[49];
rx(pi/2) q[62];
cz q[49],q[62];
rx(-pi/2) q[49];
rz(pi/2) q[49];
rx(pi/2) q[49];
cz q[48],q[49];
rx(pi/2) q[48];
rz(-pi/2) q[49];
rx(pi/2) q[49];
cz q[48],q[49];
rx(-pi/2) q[48];
rz(3*pi/2) q[48];
rx(pi/2) q[49];
cz q[48],q[49];
cz q[48],q[55];
rx(pi/2) q[48];
rz(pi/2) q[49];
rz(-pi/2) q[55];
rx(pi/2) q[55];
cz q[48],q[55];
rx(-pi/2) q[48];
rz(pi/2) q[48];
rx(pi/2) q[55];
cz q[48],q[55];
rx(pi/2) q[48];
rz(pi/2) q[48];
rz(-pi/2) q[55];
rx(-pi/2) q[55];
cz q[54],q[55];
rx(pi/2) q[54];
rz(-pi/2) q[55];
rx(pi/2) q[55];
cz q[54],q[55];
rx(-pi/2) q[54];
rz(pi/2) q[54];
rx(pi/2) q[55];
cz q[54],q[55];
rx(-pi/2) q[54];
rz(pi/2) q[54];
rx(pi/2) q[54];
cz q[41],q[54];
rx(pi/2) q[41];
rz(-pi/2) q[54];
rx(pi/2) q[54];
cz q[41],q[54];
rx(-pi/2) q[41];
rz(pi/2) q[41];
rx(pi/2) q[54];
cz q[41],q[54];
rx(pi/2) q[41];
rz(pi) q[41];
cz q[40],q[41];
rz(-pi/2) q[40];
rx(-pi/2) q[40];
rx(pi/2) q[41];
rz(pi) q[41];
cz q[41],q[40];
rx(pi/2) q[40];
rz(pi/2) q[40];
rx(pi/2) q[41];
rz(pi) q[41];
cz q[40],q[41];
cz q[40],q[47];
rx(pi/2) q[41];
rz(pi/2) q[41];
cz q[47],q[4];
rz(pi/2) q[54];
rz(pi/2) q[55];
rz(pi/2) q[62];
rz(pi/2) q[63];
rz(pi/2) q[70];
rz(pi/2) q[71];
rx(pi/2) q[78];
rz(pi/2) q[78];
rx(-pi/2) q[79];
rz(-pi/2) q[79];
barrier q[9],q[73],q[25],q[14],q[26],q[29],q[33],q[52],q[59],q[50],q[78],q[13],q[66],q[22],q[76],q[17],q[19],q[28],q[35],q[44],q[55],q[53],q[60],q[51],q[3],q[64],q[2],q[79],q[11],q[74],q[10],q[70],q[37],q[46],q[43],q[42],q[62],q[6],q[27],q[4],q[67],q[58],q[56],q[75],q[12],q[31],q[32],q[36],q[45],q[48],q[61],q[8],q[63],q[40],q[5],q[69],q[15],q[65],q[23],q[39],q[20],q[38],q[47],q[57],q[1],q[72],q[21],q[7],q[68],q[16],q[24],q[18],q[34],q[30],q[41],q[54],q[77],q[49],q[0],q[71];
measure q[3] -> meas[0];
measure q[5] -> meas[1];
measure q[60] -> meas[2];
measure q[54] -> meas[3];
measure q[74] -> meas[4];
measure q[73] -> meas[5];
measure q[26] -> meas[6];
measure q[70] -> meas[7];
measure q[40] -> meas[8];
measure q[47] -> meas[9];
measure q[50] -> meas[10];
measure q[49] -> meas[11];
measure q[51] -> meas[12];
measure q[77] -> meas[13];
measure q[68] -> meas[14];
measure q[21] -> meas[15];
measure q[20] -> meas[16];
measure q[19] -> meas[17];
measure q[76] -> meas[18];
measure q[53] -> meas[19];
measure q[48] -> meas[20];
measure q[23] -> meas[21];
measure q[16] -> meas[22];
measure q[13] -> meas[23];
measure q[66] -> meas[24];
measure q[67] -> meas[25];
measure q[79] -> meas[26];
measure q[42] -> meas[27];
measure q[59] -> meas[28];
measure q[25] -> meas[29];
measure q[10] -> meas[30];
measure q[36] -> meas[31];
measure q[35] -> meas[32];
measure q[37] -> meas[33];
measure q[27] -> meas[34];
measure q[30] -> meas[35];
measure q[4] -> meas[36];
measure q[71] -> meas[37];
measure q[64] -> meas[38];
measure q[62] -> meas[39];
measure q[61] -> meas[40];
measure q[17] -> meas[41];
measure q[57] -> meas[42];
measure q[39] -> meas[43];
measure q[33] -> meas[44];
measure q[15] -> meas[45];
measure q[2] -> meas[46];
measure q[18] -> meas[47];
measure q[34] -> meas[48];
measure q[56] -> meas[49];
measure q[9] -> meas[50];
measure q[8] -> meas[51];
measure q[28] -> meas[52];
measure q[78] -> meas[53];
measure q[69] -> meas[54];
measure q[63] -> meas[55];
measure q[22] -> meas[56];
measure q[55] -> meas[57];
measure q[12] -> meas[58];
measure q[29] -> meas[59];
measure q[65] -> meas[60];
measure q[38] -> meas[61];
measure q[11] -> meas[62];
measure q[31] -> meas[63];
measure q[58] -> meas[64];
