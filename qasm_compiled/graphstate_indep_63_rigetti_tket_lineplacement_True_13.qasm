OPENQASM 2.0;
include "qelib1.inc";

qreg node[80];
creg meas[63];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
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
rz(0.5*pi) node[29];
rz(0.5*pi) node[30];
rz(0.5*pi) node[32];
rz(0.5*pi) node[33];
rz(0.5*pi) node[34];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[42];
rz(0.5*pi) node[43];
rz(0.5*pi) node[44];
rz(0.5*pi) node[45];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
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
rz(0.5*pi) node[67];
rz(0.5*pi) node[68];
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
rx(0.5*pi) node[16];
rx(0.5*pi) node[17];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rx(0.5*pi) node[20];
rx(0.5*pi) node[21];
rx(0.5*pi) node[22];
rx(0.5*pi) node[23];
rx(0.5*pi) node[24];
rx(0.5*pi) node[25];
rx(0.5*pi) node[26];
rx(0.5*pi) node[27];
rx(0.5*pi) node[28];
rx(0.5*pi) node[29];
rx(0.5*pi) node[30];
rx(0.5*pi) node[32];
rx(0.5*pi) node[33];
rx(0.5*pi) node[34];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[40];
rx(0.5*pi) node[41];
rx(0.5*pi) node[42];
rx(0.5*pi) node[43];
rx(0.5*pi) node[44];
rx(0.5*pi) node[45];
rx(0.5*pi) node[46];
rx(0.5*pi) node[47];
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
rx(0.5*pi) node[67];
rx(0.5*pi) node[68];
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
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
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
rz(0.5*pi) node[29];
rz(0.5*pi) node[30];
rz(0.5*pi) node[32];
rz(0.5*pi) node[33];
rz(0.5*pi) node[34];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[42];
rz(0.5*pi) node[43];
rz(0.5*pi) node[44];
rz(0.5*pi) node[45];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
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
rz(0.5*pi) node[67];
rz(0.5*pi) node[68];
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
cz node[18],node[17];
cz node[21],node[22];
rz(0.5*pi) node[23];
cz node[31],node[24];
cz node[27],node[26];
cz node[34],node[33];
cz node[37],node[38];
cz node[40],node[47];
cz node[53],node[42];
cz node[45],node[44];
cz node[62],node[49];
cz node[57],node[56];
cz node[59],node[58];
cz node[65],node[66];
cz node[67],node[68];
cz node[73],node[72];
rz(0.5*pi) node[17];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rx(0.5*pi) node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[26];
cz node[27],node[28];
rz(0.5*pi) node[31];
rz(0.5*pi) node[33];
cz node[37],node[36];
rz(0.5*pi) node[38];
cz node[40],node[41];
rz(0.5*pi) node[42];
rz(0.5*pi) node[44];
cz node[45],node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[49];
cz node[53],node[52];
rz(0.5*pi) node[56];
cz node[57],node[70];
rz(0.5*pi) node[58];
cz node[59],node[60];
cz node[62],node[63];
cz node[65],node[64];
rz(0.5*pi) node[66];
rz(0.5*pi) node[68];
rz(0.5*pi) node[72];
cz node[73],node[74];
rx(0.5*pi) node[17];
rx(0.5*pi) node[21];
rx(0.5*pi) node[22];
rz(0.5*pi) node[23];
rx(0.5*pi) node[24];
rx(0.5*pi) node[26];
rz(0.5*pi) node[28];
rx(0.5*pi) node[31];
rx(0.5*pi) node[33];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[38];
rz(0.5*pi) node[41];
rx(0.5*pi) node[42];
rx(0.5*pi) node[44];
rz(0.5*pi) node[46];
rx(0.5*pi) node[47];
rx(0.5*pi) node[49];
rz(0.5*pi) node[52];
rx(0.5*pi) node[56];
rz(0.5*pi) node[57];
rx(0.5*pi) node[58];
rz(0.5*pi) node[60];
rz(0.5*pi) node[63];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rx(0.5*pi) node[66];
rx(0.5*pi) node[68];
rz(0.5*pi) node[70];
rx(0.5*pi) node[72];
rz(0.5*pi) node[74];
rz(0.5*pi) node[17];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[24];
rz(0.5*pi) node[26];
rx(0.5*pi) node[28];
rz(0.5*pi) node[31];
rz(0.5*pi) node[33];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[38];
rx(0.5*pi) node[41];
rz(0.5*pi) node[42];
rz(0.5*pi) node[44];
rx(0.5*pi) node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[49];
rx(0.5*pi) node[52];
rz(0.5*pi) node[56];
rx(0.5*pi) node[57];
rz(0.5*pi) node[58];
rx(0.5*pi) node[60];
rx(0.5*pi) node[63];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[68];
rx(0.5*pi) node[70];
rz(0.5*pi) node[72];
rx(0.5*pi) node[74];
rz(0.5*pi) node[17];
cz node[22],node[21];
cz node[24],node[31];
rz(0.5*pi) node[26];
rz(0.5*pi) node[28];
rz(0.5*pi) node[33];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[41];
rz(0.5*pi) node[42];
rz(0.5*pi) node[44];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[49];
rz(0.5*pi) node[52];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[58];
rz(0.5*pi) node[60];
rz(0.5*pi) node[63];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[68];
rz(0.5*pi) node[70];
rz(0.5*pi) node[72];
rz(0.5*pi) node[74];
rx(0.5*pi) node[17];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[24];
rx(0.5*pi) node[26];
rz(0.5*pi) node[28];
rz(0.5*pi) node[31];
rx(0.5*pi) node[33];
rz(0.5*pi) node[36];
rx(0.5*pi) node[38];
rz(0.5*pi) node[41];
rx(0.5*pi) node[42];
rx(0.5*pi) node[44];
rz(0.5*pi) node[46];
rx(0.5*pi) node[47];
rx(0.5*pi) node[49];
rz(0.5*pi) node[52];
rx(0.5*pi) node[56];
rx(0.5*pi) node[58];
rz(0.5*pi) node[60];
rz(0.5*pi) node[63];
rz(0.5*pi) node[64];
rx(0.5*pi) node[66];
rx(0.5*pi) node[68];
rz(0.5*pi) node[70];
rx(0.5*pi) node[72];
rz(0.5*pi) node[74];
rz(0.5*pi) node[17];
rx(0.5*pi) node[21];
rx(0.5*pi) node[22];
rx(0.5*pi) node[24];
rz(0.5*pi) node[26];
rx(0.5*pi) node[28];
rx(0.5*pi) node[31];
rz(0.5*pi) node[33];
rx(0.5*pi) node[36];
rz(0.5*pi) node[38];
rx(0.5*pi) node[41];
rz(0.5*pi) node[42];
rz(0.5*pi) node[44];
rx(0.5*pi) node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[49];
rx(0.5*pi) node[52];
rz(0.5*pi) node[56];
rz(0.5*pi) node[58];
rx(0.5*pi) node[60];
rx(0.5*pi) node[63];
rx(0.5*pi) node[64];
rz(0.5*pi) node[66];
rz(0.5*pi) node[68];
rx(0.5*pi) node[70];
rz(0.5*pi) node[72];
rx(0.5*pi) node[74];
cz node[17],node[30];
cz node[56],node[19];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[24];
cz node[38],node[25];
rz(0.5*pi) node[28];
rz(0.5*pi) node[31];
cz node[33],node[32];
cz node[72],node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[41];
cz node[42],node[43];
rz(0.5*pi) node[46];
cz node[49],node[48];
rz(0.5*pi) node[52];
rz(0.5*pi) node[58];
rz(0.5*pi) node[60];
rz(0.5*pi) node[63];
rz(0.5*pi) node[64];
cz node[66],node[77];
rz(0.5*pi) node[70];
rz(0.5*pi) node[74];
rz(0.5*pi) node[17];
rz(0.5*pi) node[19];
cz node[63],node[20];
cz node[21],node[22];
cz node[31],node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[30];
rz(0.5*pi) node[32];
rz(0.5*pi) node[35];
rz(0.5*pi) node[38];
cz node[54],node[41];
rz(0.5*pi) node[43];
cz node[47],node[46];
rz(0.5*pi) node[48];
cz node[52],node[51];
rz(0.5*pi) node[56];
rx(0.5*pi) node[58];
cz node[61],node[60];
cz node[64],node[71];
rz(0.5*pi) node[66];
cz node[74],node[75];
rz(0.5*pi) node[77];
rx(0.5*pi) node[17];
rx(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[24];
rx(0.5*pi) node[25];
rx(0.5*pi) node[30];
rz(0.5*pi) node[31];
rx(0.5*pi) node[32];
rx(0.5*pi) node[35];
rx(0.5*pi) node[38];
rz(0.5*pi) node[41];
rx(0.5*pi) node[43];
rz(0.5*pi) node[46];
rx(0.5*pi) node[48];
cz node[61],node[50];
rz(0.5*pi) node[51];
cz node[54],node[55];
rx(0.5*pi) node[56];
rz(0.5*pi) node[58];
rz(0.5*pi) node[60];
rx(0.5*pi) node[66];
rz(0.5*pi) node[71];
rz(0.5*pi) node[75];
rx(0.5*pi) node[77];
rz(0.5*pi) node[17];
rz(0.5*pi) node[19];
rx(0.5*pi) node[20];
rx(0.5*pi) node[21];
rx(0.5*pi) node[22];
rx(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[30];
rx(0.5*pi) node[31];
rz(0.5*pi) node[32];
rz(0.5*pi) node[35];
rz(0.5*pi) node[38];
rx(0.5*pi) node[41];
rz(0.5*pi) node[43];
rx(0.5*pi) node[46];
rz(0.5*pi) node[48];
rz(0.5*pi) node[50];
rx(0.5*pi) node[51];
rz(0.5*pi) node[55];
rz(0.5*pi) node[56];
cz node[69],node[58];
rx(0.5*pi) node[60];
rz(0.5*pi) node[66];
rx(0.5*pi) node[71];
rx(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[35];
cz node[39],node[38];
rz(0.5*pi) node[41];
rz(0.5*pi) node[43];
rz(0.5*pi) node[46];
rz(0.5*pi) node[48];
rx(0.5*pi) node[50];
rz(0.5*pi) node[51];
rx(0.5*pi) node[55];
rz(0.5*pi) node[58];
rz(0.5*pi) node[60];
cz node[67],node[66];
rz(0.5*pi) node[69];
rz(0.5*pi) node[71];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rx(0.5*pi) node[19];
rz(0.5*pi) node[20];
cz node[22],node[23];
rz(0.5*pi) node[24];
rx(0.5*pi) node[25];
rx(0.5*pi) node[30];
rx(0.5*pi) node[35];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[41];
rx(0.5*pi) node[43];
rz(0.5*pi) node[46];
rx(0.5*pi) node[48];
rz(0.5*pi) node[50];
rz(0.5*pi) node[51];
rz(0.5*pi) node[55];
rx(0.5*pi) node[58];
rz(0.5*pi) node[60];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rx(0.5*pi) node[69];
rz(0.5*pi) node[71];
rz(0.5*pi) node[75];
rx(0.5*pi) node[77];
rz(0.5*pi) node[19];
rx(0.5*pi) node[20];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rx(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[30];
rz(0.5*pi) node[35];
rx(0.5*pi) node[38];
rx(0.5*pi) node[39];
rx(0.5*pi) node[41];
rz(0.5*pi) node[43];
rx(0.5*pi) node[46];
rz(0.5*pi) node[48];
rz(0.5*pi) node[50];
rx(0.5*pi) node[51];
rz(0.5*pi) node[55];
rz(0.5*pi) node[58];
rx(0.5*pi) node[60];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
rz(0.5*pi) node[69];
rx(0.5*pi) node[71];
rx(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[20];
rx(0.5*pi) node[22];
rx(0.5*pi) node[23];
rz(0.5*pi) node[24];
cz node[26],node[25];
cz node[30],node[29];
cz node[36],node[35];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[41];
cz node[44],node[43];
rz(0.5*pi) node[46];
rx(0.5*pi) node[50];
rz(0.5*pi) node[51];
rx(0.5*pi) node[55];
cz node[58],node[69];
rz(0.5*pi) node[60];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[71];
rz(0.5*pi) node[75];
cz node[77],node[76];
cz node[20],node[19];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[29];
rz(0.5*pi) node[30];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
cz node[38],node[39];
rz(0.5*pi) node[43];
rz(0.5*pi) node[50];
rz(0.5*pi) node[55];
rz(0.5*pi) node[58];
cz node[66],node[67];
rz(0.5*pi) node[69];
cz node[71],node[70];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
cz node[23],node[22];
rx(0.5*pi) node[25];
rx(0.5*pi) node[26];
rx(0.5*pi) node[29];
rx(0.5*pi) node[30];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rx(0.5*pi) node[43];
cz node[55],node[48];
cz node[51],node[50];
rx(0.5*pi) node[58];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rx(0.5*pi) node[69];
rz(0.5*pi) node[70];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rx(0.5*pi) node[19];
rx(0.5*pi) node[20];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[29];
rz(0.5*pi) node[30];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rx(0.5*pi) node[38];
rx(0.5*pi) node[39];
rz(0.5*pi) node[43];
rz(0.5*pi) node[48];
rz(0.5*pi) node[50];
rz(0.5*pi) node[58];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
rz(0.5*pi) node[69];
rx(0.5*pi) node[70];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rx(0.5*pi) node[22];
rx(0.5*pi) node[23];
rz(0.5*pi) node[25];
rz(0.5*pi) node[29];
rz(0.5*pi) node[35];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[43];
rx(0.5*pi) node[48];
rx(0.5*pi) node[50];
cz node[69],node[58];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[70];
rz(0.5*pi) node[76];
rz(0.5*pi) node[19];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rx(0.5*pi) node[25];
rx(0.5*pi) node[29];
rx(0.5*pi) node[35];
cz node[39],node[38];
rx(0.5*pi) node[43];
rz(0.5*pi) node[48];
rz(0.5*pi) node[50];
rz(0.5*pi) node[58];
cz node[67],node[66];
rz(0.5*pi) node[70];
rx(0.5*pi) node[76];
rx(0.5*pi) node[19];
cz node[22],node[23];
rz(0.5*pi) node[25];
rz(0.5*pi) node[29];
rz(0.5*pi) node[35];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[43];
rz(0.5*pi) node[48];
rz(0.5*pi) node[50];
rx(0.5*pi) node[58];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rx(0.5*pi) node[70];
rz(0.5*pi) node[76];
rz(0.5*pi) node[19];
rz(0.5*pi) node[23];
rz(0.5*pi) node[25];
cz node[28],node[29];
rz(0.5*pi) node[35];
rx(0.5*pi) node[38];
rx(0.5*pi) node[39];
rx(0.5*pi) node[48];
rx(0.5*pi) node[50];
rz(0.5*pi) node[58];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
rz(0.5*pi) node[70];
cz node[75],node[76];
rz(0.5*pi) node[19];
rx(0.5*pi) node[23];
rx(0.5*pi) node[25];
rz(0.5*pi) node[29];
rx(0.5*pi) node[35];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[48];
rz(0.5*pi) node[50];
rz(0.5*pi) node[58];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[76];
rx(0.5*pi) node[19];
rz(0.5*pi) node[23];
rz(0.5*pi) node[25];
rx(0.5*pi) node[29];
cz node[32],node[39];
rz(0.5*pi) node[35];
rx(0.5*pi) node[58];
cz node[66],node[77];
cz node[68],node[67];
rx(0.5*pi) node[76];
cz node[23],node[16];
rz(0.5*pi) node[19];
cz node[38],node[25];
rz(0.5*pi) node[29];
rz(0.5*pi) node[32];
cz node[34],node[35];
rz(0.5*pi) node[39];
rz(0.5*pi) node[58];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[68];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[16];
cz node[18],node[19];
rz(0.5*pi) node[23];
rz(0.5*pi) node[25];
rz(0.5*pi) node[29];
rx(0.5*pi) node[32];
rz(0.5*pi) node[34];
rz(0.5*pi) node[35];
rz(0.5*pi) node[38];
rx(0.5*pi) node[39];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
rx(0.5*pi) node[68];
rz(0.5*pi) node[76];
rx(0.5*pi) node[77];
rx(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rx(0.5*pi) node[23];
rx(0.5*pi) node[25];
rx(0.5*pi) node[29];
rz(0.5*pi) node[32];
rx(0.5*pi) node[34];
rx(0.5*pi) node[35];
rx(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[68];
rx(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[16];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rz(0.5*pi) node[23];
rz(0.5*pi) node[25];
rz(0.5*pi) node[29];
cz node[39],node[32];
rz(0.5*pi) node[34];
rz(0.5*pi) node[35];
rz(0.5*pi) node[38];
cz node[77],node[66];
cz node[67],node[68];
rz(0.5*pi) node[76];
cz node[16],node[23];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
cz node[25],node[38];
rz(0.5*pi) node[32];
cz node[35],node[34];
rz(0.5*pi) node[39];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[68];
rz(0.5*pi) node[77];
rz(0.5*pi) node[16];
cz node[19],node[18];
rz(0.5*pi) node[23];
rz(0.5*pi) node[25];
rx(0.5*pi) node[32];
rz(0.5*pi) node[34];
rz(0.5*pi) node[35];
rz(0.5*pi) node[38];
rx(0.5*pi) node[39];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
rx(0.5*pi) node[68];
rx(0.5*pi) node[77];
rx(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rx(0.5*pi) node[23];
rx(0.5*pi) node[25];
rz(0.5*pi) node[32];
rx(0.5*pi) node[34];
rx(0.5*pi) node[35];
rx(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[68];
rz(0.5*pi) node[77];
rz(0.5*pi) node[16];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rz(0.5*pi) node[23];
rz(0.5*pi) node[25];
cz node[32],node[39];
rz(0.5*pi) node[34];
rz(0.5*pi) node[35];
rz(0.5*pi) node[38];
cz node[66],node[77];
cz node[68],node[67];
cz node[23],node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
cz node[38],node[25];
cz node[34],node[35];
rz(0.5*pi) node[39];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[77];
rz(0.5*pi) node[16];
cz node[18],node[19];
rz(0.5*pi) node[23];
rz(0.5*pi) node[25];
rz(0.5*pi) node[35];
rz(0.5*pi) node[38];
rx(0.5*pi) node[39];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
rx(0.5*pi) node[77];
rx(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rx(0.5*pi) node[23];
rx(0.5*pi) node[25];
rx(0.5*pi) node[35];
rx(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[77];
rz(0.5*pi) node[16];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rz(0.5*pi) node[23];
rz(0.5*pi) node[25];
rz(0.5*pi) node[35];
rz(0.5*pi) node[38];
cz node[67],node[66];
cz node[77],node[78];
cz node[16],node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
cz node[25],node[24];
cz node[35],node[36];
cz node[39],node[38];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[78];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
cz node[19],node[56];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
rx(0.5*pi) node[78];
rx(0.5*pi) node[16];
rx(0.5*pi) node[17];
rz(0.5*pi) node[19];
rx(0.5*pi) node[24];
rx(0.5*pi) node[25];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[38];
rx(0.5*pi) node[39];
rz(0.5*pi) node[56];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[78];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rx(0.5*pi) node[19];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rx(0.5*pi) node[56];
cz node[66],node[67];
cz node[78],node[79];
cz node[17],node[16];
rz(0.5*pi) node[19];
cz node[24],node[25];
cz node[36],node[35];
cz node[38],node[39];
rz(0.5*pi) node[56];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
cz node[56],node[19];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rx(0.5*pi) node[16];
rx(0.5*pi) node[17];
rz(0.5*pi) node[19];
rx(0.5*pi) node[24];
rx(0.5*pi) node[25];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[38];
rx(0.5*pi) node[39];
rz(0.5*pi) node[56];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rx(0.5*pi) node[19];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rx(0.5*pi) node[56];
cz node[67],node[66];
cz node[77],node[78];
rz(0.5*pi) node[79];
cz node[16],node[17];
rz(0.5*pi) node[19];
cz node[25],node[24];
cz node[35],node[36];
cz node[39],node[38];
rz(0.5*pi) node[56];
rz(0.5*pi) node[66];
rz(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
cz node[19],node[56];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rx(0.5*pi) node[66];
rx(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[16];
rx(0.5*pi) node[17];
rz(0.5*pi) node[19];
rx(0.5*pi) node[24];
rx(0.5*pi) node[25];
rx(0.5*pi) node[36];
rx(0.5*pi) node[38];
rz(0.5*pi) node[56];
rz(0.5*pi) node[66];
rz(0.5*pi) node[78];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rx(0.5*pi) node[19];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rx(0.5*pi) node[56];
cz node[66],node[65];
cz node[78],node[79];
cz node[17],node[30];
rz(0.5*pi) node[19];
cz node[24],node[31];
cz node[36],node[37];
rz(0.5*pi) node[56];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[24];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[56],node[57];
rx(0.5*pi) node[65];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rx(0.5*pi) node[24];
rx(0.5*pi) node[30];
rx(0.5*pi) node[31];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[57];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[24];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[57];
cz node[65],node[78];
rz(0.5*pi) node[79];
cz node[31],node[24];
cz node[37],node[36];
rz(0.5*pi) node[57];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[24];
rz(0.5*pi) node[31];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[57],node[58];
rx(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[24];
rx(0.5*pi) node[31];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[57];
rz(0.5*pi) node[58];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[24];
rz(0.5*pi) node[31];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[57];
rx(0.5*pi) node[58];
cz node[66],node[65];
rz(0.5*pi) node[78];
cz node[24],node[31];
cz node[36],node[37];
rz(0.5*pi) node[57];
rz(0.5*pi) node[58];
rz(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[24];
rz(0.5*pi) node[31];
rz(0.5*pi) node[37];
cz node[56],node[57];
rz(0.5*pi) node[58];
rx(0.5*pi) node[65];
rz(0.5*pi) node[78];
rx(0.5*pi) node[24];
rx(0.5*pi) node[31];
rx(0.5*pi) node[37];
rz(0.5*pi) node[57];
rx(0.5*pi) node[58];
rz(0.5*pi) node[65];
rz(0.5*pi) node[24];
rz(0.5*pi) node[31];
rz(0.5*pi) node[37];
rx(0.5*pi) node[57];
rz(0.5*pi) node[58];
cz node[65],node[78];
cz node[37],node[26];
rz(0.5*pi) node[31];
rz(0.5*pi) node[57];
rz(0.5*pi) node[78];
rz(0.5*pi) node[26];
rx(0.5*pi) node[31];
cz node[57],node[58];
rx(0.5*pi) node[78];
rx(0.5*pi) node[26];
rz(0.5*pi) node[31];
rz(0.5*pi) node[58];
rz(0.5*pi) node[78];
rz(0.5*pi) node[26];
cz node[30],node[31];
rx(0.5*pi) node[58];
rz(0.5*pi) node[78];
cz node[26],node[25];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[58];
rx(0.5*pi) node[78];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rx(0.5*pi) node[30];
rx(0.5*pi) node[31];
rz(0.5*pi) node[58];
rz(0.5*pi) node[78];
rx(0.5*pi) node[25];
rx(0.5*pi) node[26];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rx(0.5*pi) node[58];
cz node[79],node[78];
cz node[17],node[30];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[31];
rz(0.5*pi) node[58];
rz(0.5*pi) node[78];
cz node[17],node[18];
rz(0.5*pi) node[25];
cz node[37],node[26];
rz(0.5*pi) node[30];
rx(0.5*pi) node[31];
cz node[69],node[58];
rx(0.5*pi) node[78];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rx(0.5*pi) node[25];
rz(0.5*pi) node[26];
rx(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[58];
rz(0.5*pi) node[78];
rx(0.5*pi) node[17];
rx(0.5*pi) node[18];
rz(0.5*pi) node[25];
rx(0.5*pi) node[26];
rz(0.5*pi) node[30];
rx(0.5*pi) node[58];
rz(0.5*pi) node[78];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[26];
cz node[30],node[31];
rz(0.5*pi) node[58];
rx(0.5*pi) node[78];
cz node[18],node[17];
cz node[26],node[25];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[58];
rz(0.5*pi) node[78];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[25];
rx(0.5*pi) node[30];
rx(0.5*pi) node[31];
rx(0.5*pi) node[58];
rx(0.5*pi) node[17];
rx(0.5*pi) node[18];
rx(0.5*pi) node[25];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[58];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[25];
rz(0.5*pi) node[31];
cz node[17],node[18];
rz(0.5*pi) node[25];
rx(0.5*pi) node[31];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rx(0.5*pi) node[25];
rz(0.5*pi) node[31];
rx(0.5*pi) node[17];
rx(0.5*pi) node[18];
rz(0.5*pi) node[25];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
cz node[25],node[24];
cz node[18],node[19];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rx(0.5*pi) node[24];
rx(0.5*pi) node[25];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[24];
cz node[38],node[25];
cz node[19],node[18];
rx(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[38];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[24];
rx(0.5*pi) node[25];
rx(0.5*pi) node[38];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
cz node[31],node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[38];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[24];
cz node[25],node[38];
rz(0.5*pi) node[31];
cz node[18],node[19];
rx(0.5*pi) node[24];
rz(0.5*pi) node[25];
rx(0.5*pi) node[31];
rz(0.5*pi) node[38];
rz(0.5*pi) node[19];
rz(0.5*pi) node[24];
rx(0.5*pi) node[25];
rz(0.5*pi) node[31];
rx(0.5*pi) node[38];
rx(0.5*pi) node[19];
cz node[24],node[31];
rz(0.5*pi) node[25];
rz(0.5*pi) node[38];
rz(0.5*pi) node[19];
rz(0.5*pi) node[24];
cz node[38],node[25];
rz(0.5*pi) node[31];
cz node[19],node[20];
rx(0.5*pi) node[24];
rz(0.5*pi) node[25];
rx(0.5*pi) node[31];
rz(0.5*pi) node[20];
rz(0.5*pi) node[24];
rx(0.5*pi) node[25];
rz(0.5*pi) node[31];
rx(0.5*pi) node[20];
cz node[31],node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[20];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
cz node[31],node[30];
cz node[20],node[21];
rx(0.5*pi) node[24];
rx(0.5*pi) node[25];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rx(0.5*pi) node[30];
rx(0.5*pi) node[31];
rx(0.5*pi) node[20];
rx(0.5*pi) node[21];
cz node[24],node[25];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[25];
cz node[30],node[31];
cz node[19],node[20];
rz(0.5*pi) node[21];
rx(0.5*pi) node[25];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[20];
rx(0.5*pi) node[21];
rz(0.5*pi) node[25];
rx(0.5*pi) node[30];
rx(0.5*pi) node[31];
rx(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[25];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[20];
rx(0.5*pi) node[25];
cz node[31],node[30];
cz node[20],node[21];
rz(0.5*pi) node[25];
rz(0.5*pi) node[30];
rz(0.5*pi) node[21];
rx(0.5*pi) node[30];
rx(0.5*pi) node[21];
rz(0.5*pi) node[30];
cz node[30],node[17];
rz(0.5*pi) node[21];
rz(0.5*pi) node[17];
rz(0.5*pi) node[21];
rz(0.5*pi) node[30];
rx(0.5*pi) node[17];
rx(0.5*pi) node[21];
rx(0.5*pi) node[30];
rz(0.5*pi) node[17];
rz(0.5*pi) node[21];
rz(0.5*pi) node[30];
cz node[17],node[30];
cz node[22],node[21];
rz(0.5*pi) node[17];
rz(0.5*pi) node[21];
cz node[22],node[23];
rz(0.5*pi) node[30];
rx(0.5*pi) node[17];
rx(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rx(0.5*pi) node[30];
rz(0.5*pi) node[17];
rz(0.5*pi) node[21];
rx(0.5*pi) node[22];
rx(0.5*pi) node[23];
rz(0.5*pi) node[30];
cz node[30],node[17];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[17];
rx(0.5*pi) node[21];
cz node[23],node[22];
rx(0.5*pi) node[17];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[17];
rx(0.5*pi) node[22];
rx(0.5*pi) node[23];
rz(0.5*pi) node[17];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rx(0.5*pi) node[17];
cz node[22],node[23];
rz(0.5*pi) node[17];
rz(0.5*pi) node[23];
rx(0.5*pi) node[23];
rz(0.5*pi) node[23];
cz node[23],node[16];
rz(0.5*pi) node[16];
rz(0.5*pi) node[23];
rx(0.5*pi) node[16];
rx(0.5*pi) node[23];
rz(0.5*pi) node[16];
rz(0.5*pi) node[23];
cz node[16],node[23];
rz(0.5*pi) node[16];
rz(0.5*pi) node[23];
rx(0.5*pi) node[16];
rx(0.5*pi) node[23];
rz(0.5*pi) node[16];
rz(0.5*pi) node[23];
cz node[23],node[16];
rz(0.5*pi) node[16];
rx(0.5*pi) node[16];
rz(0.5*pi) node[16];
cz node[16],node[17];
rz(0.5*pi) node[17];
rx(0.5*pi) node[17];
rz(0.5*pi) node[17];
rz(0.5*pi) node[17];
rx(0.5*pi) node[17];
rz(0.5*pi) node[17];
barrier node[56],node[23],node[73],node[72],node[36],node[32],node[37],node[33],node[57],node[18],node[62],node[49],node[53],node[42],node[65],node[68],node[27],node[26],node[45],node[44],node[63],node[20],node[77],node[66],node[59],node[69],node[40],node[47],node[39],node[35],node[34],node[52],node[54],node[41],node[19],node[24],node[55],node[43],node[64],node[61],node[60],node[71],node[51],node[70],node[31],node[79],node[38],node[46],node[48],node[16],node[21],node[28],node[17],node[30],node[29],node[50],node[74],node[25],node[67],node[78],node[58],node[75],node[76];
measure node[56] -> meas[0];
measure node[23] -> meas[1];
measure node[73] -> meas[2];
measure node[72] -> meas[3];
measure node[36] -> meas[4];
measure node[32] -> meas[5];
measure node[37] -> meas[6];
measure node[33] -> meas[7];
measure node[57] -> meas[8];
measure node[18] -> meas[9];
measure node[62] -> meas[10];
measure node[49] -> meas[11];
measure node[53] -> meas[12];
measure node[42] -> meas[13];
measure node[65] -> meas[14];
measure node[68] -> meas[15];
measure node[27] -> meas[16];
measure node[26] -> meas[17];
measure node[45] -> meas[18];
measure node[44] -> meas[19];
measure node[63] -> meas[20];
measure node[20] -> meas[21];
measure node[77] -> meas[22];
measure node[66] -> meas[23];
measure node[59] -> meas[24];
measure node[69] -> meas[25];
measure node[40] -> meas[26];
measure node[47] -> meas[27];
measure node[39] -> meas[28];
measure node[35] -> meas[29];
measure node[34] -> meas[30];
measure node[52] -> meas[31];
measure node[54] -> meas[32];
measure node[41] -> meas[33];
measure node[19] -> meas[34];
measure node[24] -> meas[35];
measure node[55] -> meas[36];
measure node[43] -> meas[37];
measure node[64] -> meas[38];
measure node[61] -> meas[39];
measure node[60] -> meas[40];
measure node[71] -> meas[41];
measure node[51] -> meas[42];
measure node[70] -> meas[43];
measure node[31] -> meas[44];
measure node[79] -> meas[45];
measure node[38] -> meas[46];
measure node[46] -> meas[47];
measure node[48] -> meas[48];
measure node[16] -> meas[49];
measure node[21] -> meas[50];
measure node[28] -> meas[51];
measure node[17] -> meas[52];
measure node[30] -> meas[53];
measure node[29] -> meas[54];
measure node[50] -> meas[55];
measure node[74] -> meas[56];
measure node[25] -> meas[57];
measure node[67] -> meas[58];
measure node[78] -> meas[59];
measure node[58] -> meas[60];
measure node[75] -> meas[61];
measure node[76] -> meas[62];
