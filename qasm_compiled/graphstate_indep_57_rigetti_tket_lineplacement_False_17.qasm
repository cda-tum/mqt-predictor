OPENQASM 2.0;
include "qelib1.inc";

qreg node[80];
creg meas[57];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[28];
rz(0.5*pi) node[29];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[32];
rz(0.5*pi) node[33];
rz(0.5*pi) node[34];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[42];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[50];
rz(0.5*pi) node[51];
rz(0.5*pi) node[52];
rz(0.5*pi) node[53];
rz(0.5*pi) node[54];
rz(0.5*pi) node[55];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[58];
rz(0.5*pi) node[59];
rz(0.5*pi) node[60];
rz(0.5*pi) node[61];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[4];
rx(0.5*pi) node[5];
rx(0.5*pi) node[6];
rx(0.5*pi) node[7];
rx(0.5*pi) node[8];
rx(0.5*pi) node[9];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rx(0.5*pi) node[16];
rx(0.5*pi) node[17];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rx(0.5*pi) node[20];
rx(0.5*pi) node[24];
rx(0.5*pi) node[25];
rx(0.5*pi) node[26];
rx(0.5*pi) node[27];
rx(0.5*pi) node[28];
rx(0.5*pi) node[29];
rx(0.5*pi) node[30];
rx(0.5*pi) node[31];
rx(0.5*pi) node[32];
rx(0.5*pi) node[33];
rx(0.5*pi) node[34];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[39];
rx(0.5*pi) node[42];
rx(0.5*pi) node[48];
rx(0.5*pi) node[49];
rx(0.5*pi) node[50];
rx(0.5*pi) node[51];
rx(0.5*pi) node[52];
rx(0.5*pi) node[53];
rx(0.5*pi) node[54];
rx(0.5*pi) node[55];
rx(0.5*pi) node[56];
rx(0.5*pi) node[57];
rx(0.5*pi) node[58];
rx(0.5*pi) node[59];
rx(0.5*pi) node[60];
rx(0.5*pi) node[61];
rx(0.5*pi) node[62];
rx(0.5*pi) node[63];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rx(0.5*pi) node[70];
rx(0.5*pi) node[71];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rx(0.5*pi) node[74];
rx(0.5*pi) node[75];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[28];
rz(0.5*pi) node[29];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[32];
rz(0.5*pi) node[33];
rz(0.5*pi) node[34];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[42];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[50];
rz(0.5*pi) node[51];
rz(0.5*pi) node[52];
rz(0.5*pi) node[53];
rz(0.5*pi) node[54];
rz(0.5*pi) node[55];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[58];
rz(0.5*pi) node[59];
rz(0.5*pi) node[60];
rz(0.5*pi) node[61];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[2],node[1];
cz node[9],node[10];
cz node[55],node[12];
cz node[17],node[16];
cz node[20],node[63];
cz node[25],node[26];
cz node[30],node[29];
cz node[34],node[35];
cz node[50],node[51];
cz node[57],node[58];
cz node[66],node[65];
cz node[79],node[78];
rz(0.5*pi) node[1];
cz node[2],node[3];
cz node[9],node[8];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
rz(0.5*pi) node[16];
cz node[17],node[18];
cz node[20],node[19];
cz node[25],node[38];
rz(0.5*pi) node[26];
rz(0.5*pi) node[29];
cz node[30],node[31];
cz node[34],node[33];
rz(0.5*pi) node[35];
cz node[79],node[36];
cz node[55],node[48];
cz node[50],node[49];
rz(0.5*pi) node[51];
rz(0.5*pi) node[58];
rz(0.5*pi) node[63];
rz(0.5*pi) node[65];
cz node[66],node[77];
rz(0.5*pi) node[78];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[8];
rx(0.5*pi) node[10];
rx(0.5*pi) node[12];
rx(0.5*pi) node[16];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[25];
rx(0.5*pi) node[26];
rx(0.5*pi) node[29];
rz(0.5*pi) node[31];
rz(0.5*pi) node[33];
rx(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rx(0.5*pi) node[51];
rz(0.5*pi) node[55];
rx(0.5*pi) node[58];
rx(0.5*pi) node[63];
rx(0.5*pi) node[65];
rz(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[8];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
rz(0.5*pi) node[16];
rx(0.5*pi) node[17];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rx(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[29];
rx(0.5*pi) node[31];
rx(0.5*pi) node[33];
rz(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[38];
rx(0.5*pi) node[48];
rx(0.5*pi) node[49];
rz(0.5*pi) node[51];
rx(0.5*pi) node[55];
rz(0.5*pi) node[58];
rz(0.5*pi) node[63];
rz(0.5*pi) node[65];
rx(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[8];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[29];
rz(0.5*pi) node[31];
rz(0.5*pi) node[33];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[51];
rz(0.5*pi) node[55];
rz(0.5*pi) node[58];
rz(0.5*pi) node[63];
rz(0.5*pi) node[65];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rx(0.5*pi) node[1];
rz(0.5*pi) node[3];
cz node[8],node[15];
rx(0.5*pi) node[10];
rx(0.5*pi) node[12];
rx(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
cz node[38],node[25];
rx(0.5*pi) node[26];
rx(0.5*pi) node[29];
rz(0.5*pi) node[31];
rz(0.5*pi) node[33];
rx(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rx(0.5*pi) node[51];
rx(0.5*pi) node[58];
rx(0.5*pi) node[63];
rx(0.5*pi) node[65];
rz(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[1];
rx(0.5*pi) node[3];
rz(0.5*pi) node[8];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[29];
rx(0.5*pi) node[31];
rx(0.5*pi) node[33];
rz(0.5*pi) node[35];
rx(0.5*pi) node[36];
rz(0.5*pi) node[38];
rx(0.5*pi) node[48];
rx(0.5*pi) node[49];
rz(0.5*pi) node[51];
rz(0.5*pi) node[58];
rz(0.5*pi) node[63];
rz(0.5*pi) node[65];
rx(0.5*pi) node[77];
rz(0.5*pi) node[78];
cz node[1],node[0];
rz(0.5*pi) node[3];
rx(0.5*pi) node[8];
cz node[10],node[11];
rx(0.5*pi) node[15];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rx(0.5*pi) node[25];
cz node[26],node[37];
cz node[29],node[28];
rz(0.5*pi) node[31];
rz(0.5*pi) node[33];
cz node[35],node[72];
rz(0.5*pi) node[36];
rx(0.5*pi) node[38];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
cz node[51],node[52];
cz node[63],node[56];
cz node[58],node[59];
cz node[65],node[64];
rz(0.5*pi) node[77];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[3],node[4];
rz(0.5*pi) node[8];
rz(0.5*pi) node[11];
rz(0.5*pi) node[15];
cz node[19],node[18];
cz node[24],node[31];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[28];
cz node[32],node[33];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
cz node[49],node[48];
rz(0.5*pi) node[52];
rz(0.5*pi) node[56];
rz(0.5*pi) node[59];
rz(0.5*pi) node[63];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[72];
cz node[78],node[77];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[4];
cz node[15],node[8];
rx(0.5*pi) node[11];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
cz node[25],node[38];
rx(0.5*pi) node[26];
rx(0.5*pi) node[28];
rz(0.5*pi) node[31];
cz node[32],node[39];
rz(0.5*pi) node[33];
rx(0.5*pi) node[37];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rx(0.5*pi) node[52];
rx(0.5*pi) node[56];
rx(0.5*pi) node[59];
rx(0.5*pi) node[63];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rx(0.5*pi) node[72];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[4];
rz(0.5*pi) node[8];
rz(0.5*pi) node[11];
rz(0.5*pi) node[15];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[28];
rx(0.5*pi) node[31];
rx(0.5*pi) node[33];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rx(0.5*pi) node[48];
rx(0.5*pi) node[49];
rz(0.5*pi) node[52];
rz(0.5*pi) node[56];
rz(0.5*pi) node[59];
rz(0.5*pi) node[63];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[72];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[4];
rx(0.5*pi) node[8];
rz(0.5*pi) node[11];
rx(0.5*pi) node[15];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rx(0.5*pi) node[25];
rz(0.5*pi) node[28];
rz(0.5*pi) node[31];
rz(0.5*pi) node[33];
rz(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[39];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[52];
rz(0.5*pi) node[56];
rz(0.5*pi) node[59];
rz(0.5*pi) node[64];
rz(0.5*pi) node[72];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[4];
rz(0.5*pi) node[8];
rx(0.5*pi) node[11];
rz(0.5*pi) node[15];
rz(0.5*pi) node[18];
rz(0.5*pi) node[25];
rx(0.5*pi) node[28];
rz(0.5*pi) node[31];
rz(0.5*pi) node[33];
rx(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[48];
rx(0.5*pi) node[52];
rx(0.5*pi) node[56];
rx(0.5*pi) node[59];
rx(0.5*pi) node[64];
rx(0.5*pi) node[72];
rz(0.5*pi) node[77];
rx(0.5*pi) node[4];
cz node[8],node[15];
rz(0.5*pi) node[11];
rx(0.5*pi) node[18];
cz node[38],node[25];
rz(0.5*pi) node[28];
rx(0.5*pi) node[31];
rx(0.5*pi) node[33];
rz(0.5*pi) node[37];
rx(0.5*pi) node[48];
rz(0.5*pi) node[52];
rz(0.5*pi) node[56];
rz(0.5*pi) node[59];
rz(0.5*pi) node[64];
rz(0.5*pi) node[72];
rx(0.5*pi) node[77];
rz(0.5*pi) node[4];
cz node[12],node[11];
rz(0.5*pi) node[15];
rz(0.5*pi) node[18];
rz(0.5*pi) node[25];
cz node[27],node[28];
rz(0.5*pi) node[31];
rz(0.5*pi) node[33];
cz node[36],node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[48];
cz node[52],node[53];
cz node[57],node[56];
cz node[59],node[60];
cz node[72],node[73];
rz(0.5*pi) node[77];
cz node[4],node[5];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rx(0.5*pi) node[15];
rz(0.5*pi) node[18];
rx(0.5*pi) node[25];
cz node[27],node[64];
rz(0.5*pi) node[28];
rz(0.5*pi) node[37];
rx(0.5*pi) node[38];
rz(0.5*pi) node[48];
rz(0.5*pi) node[53];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[60];
rz(0.5*pi) node[73];
rz(0.5*pi) node[77];
rz(0.5*pi) node[5];
rx(0.5*pi) node[11];
rx(0.5*pi) node[12];
rz(0.5*pi) node[15];
rx(0.5*pi) node[18];
rz(0.5*pi) node[25];
rz(0.5*pi) node[27];
rx(0.5*pi) node[28];
rx(0.5*pi) node[37];
rz(0.5*pi) node[38];
rx(0.5*pi) node[48];
rx(0.5*pi) node[53];
rx(0.5*pi) node[56];
rx(0.5*pi) node[57];
rx(0.5*pi) node[60];
rz(0.5*pi) node[64];
rx(0.5*pi) node[73];
rx(0.5*pi) node[77];
rx(0.5*pi) node[5];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
cz node[15],node[14];
rz(0.5*pi) node[18];
rz(0.5*pi) node[25];
rx(0.5*pi) node[27];
rz(0.5*pi) node[28];
rz(0.5*pi) node[37];
cz node[39],node[38];
rz(0.5*pi) node[48];
rz(0.5*pi) node[53];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[60];
rx(0.5*pi) node[64];
rz(0.5*pi) node[73];
rz(0.5*pi) node[77];
rz(0.5*pi) node[5];
rz(0.5*pi) node[11];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[25];
rz(0.5*pi) node[27];
rz(0.5*pi) node[28];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[53];
rz(0.5*pi) node[56];
cz node[60],node[61];
rz(0.5*pi) node[64];
rz(0.5*pi) node[73];
rz(0.5*pi) node[5];
rx(0.5*pi) node[11];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[25];
rx(0.5*pi) node[28];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[39];
rx(0.5*pi) node[53];
rx(0.5*pi) node[56];
rz(0.5*pi) node[60];
rz(0.5*pi) node[61];
rz(0.5*pi) node[64];
rx(0.5*pi) node[73];
rx(0.5*pi) node[5];
rz(0.5*pi) node[11];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[24],node[25];
rz(0.5*pi) node[28];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[53];
rz(0.5*pi) node[56];
rx(0.5*pi) node[60];
rx(0.5*pi) node[61];
rx(0.5*pi) node[64];
rz(0.5*pi) node[73];
rz(0.5*pi) node[5];
cz node[14],node[15];
rz(0.5*pi) node[25];
rz(0.5*pi) node[37];
cz node[38],node[39];
cz node[53],node[42];
rz(0.5*pi) node[56];
rz(0.5*pi) node[60];
rz(0.5*pi) node[61];
rz(0.5*pi) node[64];
cz node[73],node[74];
cz node[5],node[6];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[25];
rx(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[42];
rz(0.5*pi) node[53];
rx(0.5*pi) node[56];
cz node[61],node[60];
rz(0.5*pi) node[64];
rz(0.5*pi) node[74];
rz(0.5*pi) node[6];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[25];
rz(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[39];
rx(0.5*pi) node[42];
rx(0.5*pi) node[53];
rz(0.5*pi) node[56];
rz(0.5*pi) node[60];
rz(0.5*pi) node[61];
rx(0.5*pi) node[64];
rx(0.5*pi) node[74];
rx(0.5*pi) node[6];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[25];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[42];
rz(0.5*pi) node[53];
rx(0.5*pi) node[60];
rx(0.5*pi) node[61];
rz(0.5*pi) node[64];
rz(0.5*pi) node[74];
rz(0.5*pi) node[6];
cz node[15],node[14];
rx(0.5*pi) node[25];
cz node[39],node[38];
rz(0.5*pi) node[42];
rz(0.5*pi) node[60];
rz(0.5*pi) node[61];
rz(0.5*pi) node[74];
rz(0.5*pi) node[6];
rz(0.5*pi) node[14];
rz(0.5*pi) node[25];
rz(0.5*pi) node[38];
rx(0.5*pi) node[42];
cz node[60],node[61];
rx(0.5*pi) node[74];
rx(0.5*pi) node[6];
rx(0.5*pi) node[14];
rx(0.5*pi) node[38];
rz(0.5*pi) node[42];
rz(0.5*pi) node[61];
rz(0.5*pi) node[74];
rz(0.5*pi) node[6];
rz(0.5*pi) node[14];
rz(0.5*pi) node[38];
cz node[42],node[53];
rx(0.5*pi) node[61];
cz node[74],node[75];
cz node[14],node[1];
cz node[6],node[7];
rz(0.5*pi) node[38];
rz(0.5*pi) node[42];
rz(0.5*pi) node[53];
rz(0.5*pi) node[61];
rz(0.5*pi) node[75];
rz(0.5*pi) node[1];
rz(0.5*pi) node[7];
rz(0.5*pi) node[14];
rx(0.5*pi) node[38];
rx(0.5*pi) node[42];
rx(0.5*pi) node[53];
rx(0.5*pi) node[75];
rx(0.5*pi) node[1];
rx(0.5*pi) node[7];
rx(0.5*pi) node[14];
rz(0.5*pi) node[38];
rz(0.5*pi) node[42];
rz(0.5*pi) node[53];
rz(0.5*pi) node[75];
rz(0.5*pi) node[1];
rz(0.5*pi) node[7];
rz(0.5*pi) node[14];
cz node[53],node[42];
rz(0.5*pi) node[75];
cz node[1],node[14];
rz(0.5*pi) node[42];
rz(0.5*pi) node[53];
rx(0.5*pi) node[75];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rx(0.5*pi) node[42];
rx(0.5*pi) node[53];
rz(0.5*pi) node[75];
rx(0.5*pi) node[1];
rx(0.5*pi) node[14];
rz(0.5*pi) node[42];
rz(0.5*pi) node[53];
cz node[75],node[76];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
cz node[42],node[53];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
cz node[14],node[1];
rz(0.5*pi) node[53];
rx(0.5*pi) node[75];
rx(0.5*pi) node[76];
rz(0.5*pi) node[1];
rx(0.5*pi) node[53];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rx(0.5*pi) node[1];
rz(0.5*pi) node[53];
cz node[76],node[75];
rz(0.5*pi) node[1];
cz node[53],node[54];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[1];
rz(0.5*pi) node[53];
rz(0.5*pi) node[54];
rx(0.5*pi) node[75];
rx(0.5*pi) node[76];
rx(0.5*pi) node[1];
rx(0.5*pi) node[53];
rx(0.5*pi) node[54];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[1];
rz(0.5*pi) node[53];
rz(0.5*pi) node[54];
cz node[75],node[76];
cz node[0],node[1];
cz node[54],node[53];
rz(0.5*pi) node[76];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[53];
rz(0.5*pi) node[54];
rx(0.5*pi) node[76];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[53];
rx(0.5*pi) node[54];
rz(0.5*pi) node[76];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[53];
rz(0.5*pi) node[54];
cz node[76],node[77];
cz node[1],node[0];
cz node[53],node[54];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[54];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[54];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[54];
cz node[77],node[76];
cz node[0],node[1];
cz node[54],node[55];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[54];
rz(0.5*pi) node[55];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[54];
rx(0.5*pi) node[55];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[54];
rz(0.5*pi) node[55];
cz node[76],node[77];
cz node[7],node[0];
rz(0.5*pi) node[1];
cz node[55],node[54];
rz(0.5*pi) node[77];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[7];
rz(0.5*pi) node[54];
rz(0.5*pi) node[55];
rx(0.5*pi) node[77];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[7];
rx(0.5*pi) node[54];
rx(0.5*pi) node[55];
rz(0.5*pi) node[77];
rz(0.5*pi) node[0];
rz(0.5*pi) node[7];
rz(0.5*pi) node[54];
rz(0.5*pi) node[55];
cz node[77],node[78];
rz(0.5*pi) node[0];
cz node[6],node[7];
cz node[54],node[55];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rx(0.5*pi) node[0];
rz(0.5*pi) node[7];
rz(0.5*pi) node[55];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[0];
rx(0.5*pi) node[7];
rx(0.5*pi) node[55];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[7];
rz(0.5*pi) node[55];
cz node[78],node[77];
cz node[7],node[0];
cz node[55],node[48];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[0];
rz(0.5*pi) node[48];
rz(0.5*pi) node[55];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[0];
rx(0.5*pi) node[48];
rx(0.5*pi) node[55];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[0];
rz(0.5*pi) node[48];
rz(0.5*pi) node[55];
cz node[77],node[78];
rz(0.5*pi) node[0];
cz node[48],node[55];
rz(0.5*pi) node[78];
rx(0.5*pi) node[0];
rz(0.5*pi) node[48];
rz(0.5*pi) node[55];
rx(0.5*pi) node[78];
rz(0.5*pi) node[0];
rx(0.5*pi) node[48];
rx(0.5*pi) node[55];
rz(0.5*pi) node[78];
rz(0.5*pi) node[48];
rz(0.5*pi) node[55];
cz node[78],node[65];
cz node[55],node[48];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[48];
rz(0.5*pi) node[55];
rx(0.5*pi) node[65];
rx(0.5*pi) node[78];
rx(0.5*pi) node[48];
rx(0.5*pi) node[55];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[48];
rz(0.5*pi) node[55];
cz node[65],node[78];
cz node[48],node[49];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rx(0.5*pi) node[65];
rx(0.5*pi) node[78];
rx(0.5*pi) node[48];
rx(0.5*pi) node[49];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
cz node[78],node[65];
cz node[49],node[48];
rz(0.5*pi) node[65];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rx(0.5*pi) node[65];
rx(0.5*pi) node[48];
rx(0.5*pi) node[49];
rz(0.5*pi) node[65];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
cz node[65],node[64];
cz node[48],node[49];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rx(0.5*pi) node[48];
rx(0.5*pi) node[49];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
cz node[64],node[65];
cz node[49],node[62];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[49];
rz(0.5*pi) node[62];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rx(0.5*pi) node[49];
rx(0.5*pi) node[62];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[49];
rz(0.5*pi) node[62];
cz node[65],node[64];
cz node[62],node[49];
rz(0.5*pi) node[64];
rz(0.5*pi) node[49];
rz(0.5*pi) node[62];
rx(0.5*pi) node[64];
rx(0.5*pi) node[49];
rx(0.5*pi) node[62];
rz(0.5*pi) node[64];
rz(0.5*pi) node[49];
rz(0.5*pi) node[62];
cz node[64],node[71];
cz node[49],node[62];
rz(0.5*pi) node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[49];
rz(0.5*pi) node[62];
rx(0.5*pi) node[64];
rx(0.5*pi) node[71];
rx(0.5*pi) node[49];
rx(0.5*pi) node[62];
rz(0.5*pi) node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[49];
rz(0.5*pi) node[62];
cz node[71],node[64];
cz node[62],node[63];
rz(0.5*pi) node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rx(0.5*pi) node[64];
rx(0.5*pi) node[71];
rx(0.5*pi) node[62];
rx(0.5*pi) node[63];
rz(0.5*pi) node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
cz node[64],node[71];
cz node[63],node[62];
rz(0.5*pi) node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rx(0.5*pi) node[64];
rx(0.5*pi) node[71];
rx(0.5*pi) node[62];
rx(0.5*pi) node[63];
rz(0.5*pi) node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
cz node[71],node[70];
cz node[62],node[63];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rx(0.5*pi) node[70];
rx(0.5*pi) node[71];
rx(0.5*pi) node[62];
rx(0.5*pi) node[63];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
cz node[70],node[71];
rz(0.5*pi) node[63];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
rx(0.5*pi) node[63];
rx(0.5*pi) node[70];
rx(0.5*pi) node[71];
rz(0.5*pi) node[63];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
cz node[71],node[70];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
rx(0.5*pi) node[70];
rx(0.5*pi) node[71];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
cz node[70],node[57];
rz(0.5*pi) node[57];
rz(0.5*pi) node[70];
rx(0.5*pi) node[57];
rx(0.5*pi) node[70];
rz(0.5*pi) node[57];
rz(0.5*pi) node[70];
cz node[57],node[70];
rz(0.5*pi) node[57];
rz(0.5*pi) node[70];
rx(0.5*pi) node[57];
rx(0.5*pi) node[70];
rz(0.5*pi) node[57];
rz(0.5*pi) node[70];
cz node[70],node[57];
rz(0.5*pi) node[57];
rz(0.5*pi) node[70];
rx(0.5*pi) node[57];
rx(0.5*pi) node[70];
rz(0.5*pi) node[57];
rz(0.5*pi) node[70];
cz node[57],node[56];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rx(0.5*pi) node[56];
rx(0.5*pi) node[57];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
cz node[56],node[57];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rx(0.5*pi) node[56];
rx(0.5*pi) node[57];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
cz node[57],node[56];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rx(0.5*pi) node[56];
rx(0.5*pi) node[57];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
cz node[56],node[63];
rz(0.5*pi) node[56];
rz(0.5*pi) node[63];
rx(0.5*pi) node[56];
rx(0.5*pi) node[63];
rz(0.5*pi) node[56];
rz(0.5*pi) node[63];
cz node[63],node[56];
rz(0.5*pi) node[56];
rz(0.5*pi) node[63];
rx(0.5*pi) node[56];
rx(0.5*pi) node[63];
rz(0.5*pi) node[56];
rz(0.5*pi) node[63];
cz node[56],node[63];
cz node[56],node[57];
rz(0.5*pi) node[63];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rx(0.5*pi) node[63];
rx(0.5*pi) node[56];
rx(0.5*pi) node[57];
rz(0.5*pi) node[63];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
cz node[63],node[62];
cz node[57],node[56];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rx(0.5*pi) node[62];
rx(0.5*pi) node[63];
rx(0.5*pi) node[56];
rx(0.5*pi) node[57];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
cz node[62],node[63];
cz node[56],node[57];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rx(0.5*pi) node[62];
rx(0.5*pi) node[63];
rx(0.5*pi) node[56];
rx(0.5*pi) node[57];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
cz node[63],node[62];
cz node[57],node[70];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rz(0.5*pi) node[57];
rx(0.5*pi) node[62];
rx(0.5*pi) node[63];
rz(0.5*pi) node[70];
rx(0.5*pi) node[57];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rx(0.5*pi) node[70];
cz node[62],node[49];
rz(0.5*pi) node[57];
rz(0.5*pi) node[70];
rz(0.5*pi) node[49];
cz node[70],node[57];
rz(0.5*pi) node[62];
rx(0.5*pi) node[49];
rz(0.5*pi) node[57];
rx(0.5*pi) node[62];
rz(0.5*pi) node[70];
rz(0.5*pi) node[49];
rx(0.5*pi) node[57];
rz(0.5*pi) node[62];
rx(0.5*pi) node[70];
cz node[49],node[62];
rz(0.5*pi) node[57];
rz(0.5*pi) node[70];
rz(0.5*pi) node[49];
cz node[57],node[70];
rz(0.5*pi) node[62];
rx(0.5*pi) node[49];
rx(0.5*pi) node[62];
rz(0.5*pi) node[70];
rz(0.5*pi) node[49];
rz(0.5*pi) node[62];
rx(0.5*pi) node[70];
cz node[62],node[49];
rz(0.5*pi) node[70];
rz(0.5*pi) node[49];
rz(0.5*pi) node[62];
cz node[70],node[71];
rx(0.5*pi) node[49];
rx(0.5*pi) node[62];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
rz(0.5*pi) node[49];
rz(0.5*pi) node[62];
rx(0.5*pi) node[70];
rx(0.5*pi) node[71];
cz node[49],node[48];
cz node[61],node[62];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[61];
rz(0.5*pi) node[62];
cz node[71],node[70];
rx(0.5*pi) node[48];
rx(0.5*pi) node[49];
rx(0.5*pi) node[61];
rx(0.5*pi) node[62];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[61];
rz(0.5*pi) node[62];
rx(0.5*pi) node[70];
rx(0.5*pi) node[71];
cz node[48],node[49];
cz node[62],node[61];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[61];
rz(0.5*pi) node[62];
cz node[70],node[71];
rx(0.5*pi) node[48];
rx(0.5*pi) node[49];
rx(0.5*pi) node[61];
rx(0.5*pi) node[62];
rz(0.5*pi) node[71];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[61];
rz(0.5*pi) node[62];
rx(0.5*pi) node[71];
cz node[49],node[48];
cz node[61],node[62];
rz(0.5*pi) node[71];
rz(0.5*pi) node[48];
rz(0.5*pi) node[62];
cz node[71],node[64];
rx(0.5*pi) node[48];
rx(0.5*pi) node[62];
rz(0.5*pi) node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[48];
rz(0.5*pi) node[62];
rx(0.5*pi) node[64];
rx(0.5*pi) node[71];
cz node[48],node[55];
cz node[62],node[63];
rz(0.5*pi) node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[48];
rz(0.5*pi) node[55];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
cz node[64],node[71];
rx(0.5*pi) node[48];
rx(0.5*pi) node[55];
rx(0.5*pi) node[62];
rx(0.5*pi) node[63];
rz(0.5*pi) node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[48];
rz(0.5*pi) node[55];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rx(0.5*pi) node[64];
rx(0.5*pi) node[71];
cz node[55],node[48];
cz node[63],node[62];
rz(0.5*pi) node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[48];
rz(0.5*pi) node[55];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
cz node[71],node[64];
rx(0.5*pi) node[48];
rx(0.5*pi) node[55];
rx(0.5*pi) node[62];
rx(0.5*pi) node[63];
rz(0.5*pi) node[64];
rz(0.5*pi) node[48];
rz(0.5*pi) node[55];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rx(0.5*pi) node[64];
cz node[48],node[55];
cz node[62],node[63];
rz(0.5*pi) node[64];
cz node[64],node[27];
rz(0.5*pi) node[55];
rz(0.5*pi) node[63];
rz(0.5*pi) node[27];
rx(0.5*pi) node[55];
rx(0.5*pi) node[63];
rz(0.5*pi) node[64];
rx(0.5*pi) node[27];
rz(0.5*pi) node[55];
rz(0.5*pi) node[63];
rx(0.5*pi) node[64];
cz node[55],node[12];
rz(0.5*pi) node[27];
cz node[63],node[56];
rz(0.5*pi) node[64];
rz(0.5*pi) node[12];
cz node[27],node[64];
rz(0.5*pi) node[55];
rz(0.5*pi) node[56];
rz(0.5*pi) node[63];
rx(0.5*pi) node[12];
rz(0.5*pi) node[27];
rx(0.5*pi) node[55];
rx(0.5*pi) node[56];
rx(0.5*pi) node[63];
rz(0.5*pi) node[64];
rz(0.5*pi) node[12];
rx(0.5*pi) node[27];
rz(0.5*pi) node[55];
rz(0.5*pi) node[56];
rz(0.5*pi) node[63];
rx(0.5*pi) node[64];
cz node[12],node[55];
rz(0.5*pi) node[27];
cz node[56],node[63];
rz(0.5*pi) node[64];
rz(0.5*pi) node[12];
cz node[64],node[27];
rz(0.5*pi) node[55];
rz(0.5*pi) node[56];
rz(0.5*pi) node[63];
rx(0.5*pi) node[12];
rz(0.5*pi) node[27];
rx(0.5*pi) node[55];
rx(0.5*pi) node[56];
rx(0.5*pi) node[63];
rz(0.5*pi) node[12];
rx(0.5*pi) node[27];
rz(0.5*pi) node[55];
rz(0.5*pi) node[56];
rz(0.5*pi) node[63];
cz node[55],node[12];
rz(0.5*pi) node[27];
cz node[63],node[56];
rz(0.5*pi) node[12];
cz node[27],node[26];
rz(0.5*pi) node[56];
rx(0.5*pi) node[12];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rx(0.5*pi) node[56];
rz(0.5*pi) node[12];
rx(0.5*pi) node[26];
rx(0.5*pi) node[27];
rz(0.5*pi) node[56];
cz node[12],node[13];
cz node[56],node[19];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[19];
cz node[26],node[27];
rz(0.5*pi) node[56];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[19];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rx(0.5*pi) node[56];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[19];
rx(0.5*pi) node[26];
rx(0.5*pi) node[27];
rz(0.5*pi) node[56];
cz node[13],node[12];
cz node[19],node[56];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[19];
cz node[27],node[26];
rz(0.5*pi) node[56];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[19];
rz(0.5*pi) node[26];
rx(0.5*pi) node[56];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[19];
rx(0.5*pi) node[26];
rz(0.5*pi) node[56];
cz node[12],node[13];
cz node[56],node[19];
rz(0.5*pi) node[26];
rz(0.5*pi) node[13];
rz(0.5*pi) node[19];
cz node[26],node[37];
rx(0.5*pi) node[13];
rx(0.5*pi) node[19];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[13];
rz(0.5*pi) node[19];
rx(0.5*pi) node[26];
rx(0.5*pi) node[37];
cz node[13],node[2];
cz node[19],node[18];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[2];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
cz node[37],node[26];
rx(0.5*pi) node[2];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[2];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rx(0.5*pi) node[26];
rx(0.5*pi) node[37];
cz node[2],node[1];
cz node[18],node[19];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
cz node[26],node[37];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rz(0.5*pi) node[37];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rx(0.5*pi) node[37];
rz(0.5*pi) node[1];
cz node[13],node[2];
cz node[19],node[18];
rz(0.5*pi) node[37];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[18];
cz node[37],node[38];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[18];
rz(0.5*pi) node[38];
rz(0.5*pi) node[2];
rz(0.5*pi) node[18];
rx(0.5*pi) node[38];
cz node[2],node[1];
cz node[18],node[17];
rz(0.5*pi) node[38];
rz(0.5*pi) node[1];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[38];
rx(0.5*pi) node[1];
rx(0.5*pi) node[17];
rx(0.5*pi) node[18];
rx(0.5*pi) node[38];
rz(0.5*pi) node[1];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[38];
rz(0.5*pi) node[1];
cz node[17],node[18];
rx(0.5*pi) node[1];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[1];
rx(0.5*pi) node[17];
rx(0.5*pi) node[18];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
cz node[18],node[17];
rz(0.5*pi) node[17];
rx(0.5*pi) node[17];
rz(0.5*pi) node[17];
rz(0.5*pi) node[17];
rx(0.5*pi) node[17];
rz(0.5*pi) node[17];
cz node[16],node[17];
rz(0.5*pi) node[17];
rx(0.5*pi) node[17];
rz(0.5*pi) node[17];
rz(0.5*pi) node[17];
rx(0.5*pi) node[17];
rz(0.5*pi) node[17];
barrier node[20],node[62],node[9],node[10],node[39],node[27],node[50],node[51],node[57],node[58],node[66],node[78],node[30],node[29],node[24],node[31],node[2],node[14],node[63],node[59],node[18],node[16],node[79],node[77],node[34],node[35],node[52],node[3],node[64],node[28],node[4],node[5],node[54],node[55],node[56],node[19],node[6],node[0],node[72],node[73],node[36],node[26],node[17],node[49],node[32],node[33],node[42],node[37],node[25],node[38],node[65],node[74],node[13],node[11],node[1],node[48],node[76];
measure node[20] -> meas[0];
measure node[62] -> meas[1];
measure node[9] -> meas[2];
measure node[10] -> meas[3];
measure node[39] -> meas[4];
measure node[27] -> meas[5];
measure node[50] -> meas[6];
measure node[51] -> meas[7];
measure node[57] -> meas[8];
measure node[58] -> meas[9];
measure node[66] -> meas[10];
measure node[78] -> meas[11];
measure node[30] -> meas[12];
measure node[29] -> meas[13];
measure node[24] -> meas[14];
measure node[31] -> meas[15];
measure node[2] -> meas[16];
measure node[14] -> meas[17];
measure node[63] -> meas[18];
measure node[59] -> meas[19];
measure node[18] -> meas[20];
measure node[16] -> meas[21];
measure node[79] -> meas[22];
measure node[77] -> meas[23];
measure node[34] -> meas[24];
measure node[35] -> meas[25];
measure node[52] -> meas[26];
measure node[3] -> meas[27];
measure node[64] -> meas[28];
measure node[28] -> meas[29];
measure node[4] -> meas[30];
measure node[5] -> meas[31];
measure node[54] -> meas[32];
measure node[55] -> meas[33];
measure node[56] -> meas[34];
measure node[19] -> meas[35];
measure node[6] -> meas[36];
measure node[0] -> meas[37];
measure node[72] -> meas[38];
measure node[73] -> meas[39];
measure node[36] -> meas[40];
measure node[26] -> meas[41];
measure node[17] -> meas[42];
measure node[49] -> meas[43];
measure node[32] -> meas[44];
measure node[33] -> meas[45];
measure node[42] -> meas[46];
measure node[37] -> meas[47];
measure node[25] -> meas[48];
measure node[38] -> meas[49];
measure node[65] -> meas[50];
measure node[74] -> meas[51];
measure node[13] -> meas[52];
measure node[11] -> meas[53];
measure node[1] -> meas[54];
measure node[48] -> meas[55];
measure node[76] -> meas[56];
