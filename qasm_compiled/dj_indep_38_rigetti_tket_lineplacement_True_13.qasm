OPENQASM 2.0;
include "qelib1.inc";

qreg node[80];
creg c[37];
rz(3.5*pi) node[8];
rz(3.5*pi) node[9];
rz(3.5*pi) node[10];
rz(3.5*pi) node[11];
rz(3.5*pi) node[12];
rz(3.5*pi) node[14];
rz(3.5*pi) node[15];
rz(3.5*pi) node[16];
rz(3.5*pi) node[17];
rz(3.5*pi) node[18];
rz(3.5*pi) node[19];
rz(3.5*pi) node[20];
rz(3.5*pi) node[21];
rz(3.5*pi) node[22];
rz(3.5*pi) node[23];
rz(3.5*pi) node[24];
rz(3.5*pi) node[25];
rz(3.5*pi) node[26];
rz(3.5*pi) node[27];
rz(3.5*pi) node[28];
rz(3.5*pi) node[29];
rz(3.5*pi) node[30];
rz(3.5*pi) node[31];
rz(3.5*pi) node[32];
rz(3.5*pi) node[36];
rz(3.5*pi) node[37];
rz(3.5*pi) node[38];
rz(3.5*pi) node[39];
rz(3.5*pi) node[48];
rz(3.5*pi) node[56];
rz(3.5*pi) node[64];
rz(3.5*pi) node[65];
rz(3.5*pi) node[66];
rz(3.5*pi) node[70];
rz(3.5*pi) node[71];
rz(3.5*pi) node[77];
rz(0.5*pi) node[78];
rz(3.5*pi) node[79];
rx(1.5*pi) node[8];
rx(1.5*pi) node[9];
rx(1.5*pi) node[10];
rx(1.5*pi) node[11];
rx(1.5*pi) node[12];
rx(1.5*pi) node[14];
rx(1.5*pi) node[15];
rx(1.5*pi) node[16];
rx(1.5*pi) node[17];
rx(1.5*pi) node[18];
rx(1.5*pi) node[19];
rx(1.5*pi) node[20];
rx(1.5*pi) node[21];
rx(1.5*pi) node[22];
rx(1.5*pi) node[23];
rx(1.5*pi) node[24];
rx(1.5*pi) node[25];
rx(1.5*pi) node[26];
rx(1.5*pi) node[27];
rx(1.5*pi) node[28];
rx(1.5*pi) node[29];
rx(1.5*pi) node[30];
rx(1.5*pi) node[31];
rx(1.5*pi) node[32];
rx(1.5*pi) node[36];
rx(1.5*pi) node[37];
rx(1.5*pi) node[38];
rx(1.5*pi) node[39];
rx(1.5*pi) node[48];
rx(1.5*pi) node[56];
rx(1.5*pi) node[64];
rx(1.5*pi) node[65];
rx(1.5*pi) node[66];
rx(1.5*pi) node[70];
rx(1.5*pi) node[71];
rx(1.5*pi) node[77];
rx(1.5*pi) node[78];
rx(1.5*pi) node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[77],node[78];
rx(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[79],node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[79];
rx(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[65],node[78];
rx(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rz(0.5*pi) node[65];
cz node[78],node[65];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
cz node[65],node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
cz node[78],node[65];
rz(0.5*pi) node[65];
rx(0.5*pi) node[65];
rz(0.5*pi) node[65];
rz(0.5*pi) node[65];
rx(0.5*pi) node[65];
rz(0.5*pi) node[65];
cz node[64],node[65];
rx(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[64];
rx(0.5*pi) node[65];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rx(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[64];
rx(0.5*pi) node[65];
rz(0.5*pi) node[65];
cz node[66],node[65];
rz(0.5*pi) node[65];
rx(0.5*pi) node[66];
rx(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[65];
cz node[65],node[64];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
cz node[64],node[65];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
cz node[65],node[64];
rz(0.5*pi) node[64];
rx(0.5*pi) node[64];
rz(0.5*pi) node[64];
rz(0.5*pi) node[64];
rx(0.5*pi) node[64];
rz(0.5*pi) node[64];
cz node[27],node[64];
rx(0.5*pi) node[27];
rz(0.5*pi) node[64];
rz(0.5*pi) node[27];
rx(0.5*pi) node[64];
rz(0.5*pi) node[27];
rz(0.5*pi) node[64];
rx(0.5*pi) node[27];
rz(0.5*pi) node[64];
rz(0.5*pi) node[27];
rx(0.5*pi) node[64];
cz node[28],node[27];
rz(0.5*pi) node[64];
rz(0.5*pi) node[27];
cz node[71],node[64];
rx(0.5*pi) node[27];
rz(0.5*pi) node[64];
rx(0.5*pi) node[71];
rz(0.5*pi) node[27];
rx(0.5*pi) node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[64];
rx(0.5*pi) node[71];
rx(0.5*pi) node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[64];
cz node[70],node[71];
cz node[27],node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[27];
rz(0.5*pi) node[64];
rx(0.5*pi) node[71];
rx(0.5*pi) node[27];
rx(0.5*pi) node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[27];
rz(0.5*pi) node[64];
cz node[28],node[27];
rz(0.5*pi) node[64];
rz(0.5*pi) node[27];
rx(0.5*pi) node[28];
rx(0.5*pi) node[64];
rx(0.5*pi) node[27];
rz(0.5*pi) node[28];
rz(0.5*pi) node[64];
rz(0.5*pi) node[27];
rz(0.5*pi) node[28];
cz node[27],node[64];
rx(0.5*pi) node[28];
rz(0.5*pi) node[27];
rz(0.5*pi) node[28];
rz(0.5*pi) node[64];
rx(0.5*pi) node[27];
cz node[29],node[28];
rx(0.5*pi) node[64];
rz(0.5*pi) node[27];
rz(0.5*pi) node[28];
rz(0.5*pi) node[29];
rz(0.5*pi) node[64];
rx(0.5*pi) node[28];
rx(0.5*pi) node[29];
rz(0.5*pi) node[64];
rz(0.5*pi) node[28];
rz(0.5*pi) node[29];
rx(0.5*pi) node[64];
cz node[28],node[29];
rz(0.5*pi) node[64];
rz(0.5*pi) node[28];
rz(0.5*pi) node[29];
cz node[71],node[64];
rx(0.5*pi) node[28];
rx(0.5*pi) node[29];
rz(0.5*pi) node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[28];
rz(0.5*pi) node[29];
rx(0.5*pi) node[64];
rx(0.5*pi) node[71];
cz node[29],node[28];
rz(0.5*pi) node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[28];
rz(0.5*pi) node[64];
cz node[70],node[71];
rx(0.5*pi) node[28];
rx(0.5*pi) node[64];
rx(0.5*pi) node[70];
rz(0.5*pi) node[71];
rz(0.5*pi) node[28];
rz(0.5*pi) node[64];
rz(0.5*pi) node[70];
rx(0.5*pi) node[71];
rz(0.5*pi) node[71];
cz node[71],node[64];
rz(0.5*pi) node[64];
rx(0.5*pi) node[64];
rz(0.5*pi) node[64];
cz node[64],node[27];
rz(0.5*pi) node[27];
rz(0.5*pi) node[64];
rx(0.5*pi) node[27];
rx(0.5*pi) node[64];
rz(0.5*pi) node[27];
rz(0.5*pi) node[64];
cz node[27],node[64];
rz(0.5*pi) node[27];
rz(0.5*pi) node[64];
rx(0.5*pi) node[27];
rx(0.5*pi) node[64];
rz(0.5*pi) node[27];
rz(0.5*pi) node[64];
cz node[64],node[27];
rz(0.5*pi) node[27];
rx(0.5*pi) node[27];
rz(0.5*pi) node[27];
rz(0.5*pi) node[27];
rx(0.5*pi) node[27];
rz(0.5*pi) node[27];
cz node[26],node[27];
rx(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[26];
rx(0.5*pi) node[27];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rx(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[26];
rx(0.5*pi) node[27];
rz(0.5*pi) node[27];
cz node[28],node[27];
rz(0.5*pi) node[27];
rx(0.5*pi) node[28];
rx(0.5*pi) node[27];
rz(0.5*pi) node[28];
rz(0.5*pi) node[27];
cz node[27],node[26];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rx(0.5*pi) node[26];
rx(0.5*pi) node[27];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
cz node[26],node[27];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rx(0.5*pi) node[26];
rx(0.5*pi) node[27];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
cz node[27],node[26];
rz(0.5*pi) node[26];
rx(0.5*pi) node[26];
rz(0.5*pi) node[26];
rz(0.5*pi) node[26];
rx(0.5*pi) node[26];
rz(0.5*pi) node[26];
cz node[25],node[26];
rx(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[25];
rx(0.5*pi) node[26];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rx(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[25];
rx(0.5*pi) node[26];
rz(0.5*pi) node[26];
cz node[37],node[26];
rz(0.5*pi) node[26];
rx(0.5*pi) node[37];
rx(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
cz node[26],node[25];
rx(0.5*pi) node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rx(0.5*pi) node[25];
rx(0.5*pi) node[26];
cz node[36],node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[25],node[26];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[25];
rx(0.5*pi) node[26];
cz node[37],node[36];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[26],node[25];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[25];
rx(0.5*pi) node[26];
cz node[36],node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[25];
rx(0.5*pi) node[37];
rx(0.5*pi) node[25];
rz(0.5*pi) node[37];
rz(0.5*pi) node[25];
cz node[37],node[26];
cz node[24],node[25];
rz(0.5*pi) node[26];
rx(0.5*pi) node[24];
rz(0.5*pi) node[25];
rx(0.5*pi) node[26];
rz(0.5*pi) node[24];
rx(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rx(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[24];
rx(0.5*pi) node[25];
cz node[31],node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[24];
cz node[38],node[25];
rz(0.5*pi) node[31];
rx(0.5*pi) node[24];
rz(0.5*pi) node[25];
rx(0.5*pi) node[31];
rx(0.5*pi) node[38];
rz(0.5*pi) node[24];
rx(0.5*pi) node[25];
rz(0.5*pi) node[31];
rz(0.5*pi) node[38];
cz node[24],node[31];
rz(0.5*pi) node[25];
rz(0.5*pi) node[38];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[31];
rx(0.5*pi) node[38];
rx(0.5*pi) node[24];
rx(0.5*pi) node[25];
rx(0.5*pi) node[31];
rz(0.5*pi) node[38];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[31];
cz node[39],node[38];
cz node[31],node[24];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[24];
rz(0.5*pi) node[31];
rx(0.5*pi) node[38];
rx(0.5*pi) node[39];
rx(0.5*pi) node[24];
rx(0.5*pi) node[31];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[24];
rz(0.5*pi) node[31];
cz node[38],node[39];
cz node[24],node[25];
cz node[30],node[31];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rx(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rx(0.5*pi) node[38];
rx(0.5*pi) node[39];
rz(0.5*pi) node[24];
rx(0.5*pi) node[25];
rx(0.5*pi) node[30];
rx(0.5*pi) node[31];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
cz node[39],node[38];
rx(0.5*pi) node[24];
rz(0.5*pi) node[25];
cz node[31],node[30];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[24];
rx(0.5*pi) node[25];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rx(0.5*pi) node[38];
rx(0.5*pi) node[39];
rz(0.5*pi) node[25];
rx(0.5*pi) node[30];
rx(0.5*pi) node[31];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
cz node[38],node[25];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
cz node[32],node[39];
rz(0.5*pi) node[25];
cz node[30],node[31];
rz(0.5*pi) node[32];
rx(0.5*pi) node[38];
rz(0.5*pi) node[39];
rx(0.5*pi) node[25];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rx(0.5*pi) node[32];
rz(0.5*pi) node[38];
rx(0.5*pi) node[39];
rz(0.5*pi) node[25];
rx(0.5*pi) node[30];
rx(0.5*pi) node[31];
rz(0.5*pi) node[32];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[25];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
cz node[39],node[32];
rx(0.5*pi) node[38];
cz node[31],node[24];
rx(0.5*pi) node[25];
rz(0.5*pi) node[32];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[31];
rx(0.5*pi) node[32];
rx(0.5*pi) node[39];
rx(0.5*pi) node[24];
rx(0.5*pi) node[31];
rz(0.5*pi) node[32];
rz(0.5*pi) node[39];
rz(0.5*pi) node[24];
rz(0.5*pi) node[31];
cz node[32],node[39];
cz node[24],node[31];
rz(0.5*pi) node[39];
rz(0.5*pi) node[24];
rz(0.5*pi) node[31];
rx(0.5*pi) node[39];
rx(0.5*pi) node[24];
rx(0.5*pi) node[31];
rz(0.5*pi) node[39];
rz(0.5*pi) node[24];
rz(0.5*pi) node[31];
cz node[39],node[38];
cz node[31],node[24];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[24];
rz(0.5*pi) node[31];
rx(0.5*pi) node[38];
rx(0.5*pi) node[39];
rx(0.5*pi) node[24];
rx(0.5*pi) node[31];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[24];
rz(0.5*pi) node[31];
cz node[38],node[39];
cz node[24],node[25];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rx(0.5*pi) node[24];
rz(0.5*pi) node[25];
rx(0.5*pi) node[38];
rx(0.5*pi) node[39];
rz(0.5*pi) node[24];
rx(0.5*pi) node[25];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
cz node[39],node[38];
rx(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[38];
rz(0.5*pi) node[24];
rx(0.5*pi) node[25];
rx(0.5*pi) node[38];
rz(0.5*pi) node[25];
rz(0.5*pi) node[38];
cz node[26],node[25];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rx(0.5*pi) node[25];
rx(0.5*pi) node[26];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[25];
cz node[37],node[26];
rx(0.5*pi) node[25];
rz(0.5*pi) node[26];
rx(0.5*pi) node[37];
rz(0.5*pi) node[25];
rx(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[26];
cz node[26],node[25];
rz(0.5*pi) node[25];
rx(0.5*pi) node[25];
rz(0.5*pi) node[25];
rz(0.5*pi) node[25];
rx(0.5*pi) node[25];
rz(0.5*pi) node[25];
cz node[38],node[25];
rz(0.5*pi) node[25];
rx(0.5*pi) node[38];
rx(0.5*pi) node[25];
rz(0.5*pi) node[38];
rz(0.5*pi) node[25];
cz node[25],node[24];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rx(0.5*pi) node[24];
rx(0.5*pi) node[25];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
cz node[24],node[25];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rx(0.5*pi) node[24];
rx(0.5*pi) node[25];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
cz node[25],node[24];
rz(0.5*pi) node[24];
rx(0.5*pi) node[24];
rz(0.5*pi) node[24];
cz node[24],node[31];
rz(0.5*pi) node[24];
rz(0.5*pi) node[31];
rx(0.5*pi) node[24];
rx(0.5*pi) node[31];
rz(0.5*pi) node[24];
rz(0.5*pi) node[31];
cz node[31],node[24];
rz(0.5*pi) node[24];
rz(0.5*pi) node[31];
rx(0.5*pi) node[24];
rx(0.5*pi) node[31];
rz(0.5*pi) node[24];
rz(0.5*pi) node[31];
cz node[24],node[31];
rz(0.5*pi) node[31];
rx(0.5*pi) node[31];
rz(0.5*pi) node[31];
cz node[31],node[30];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rx(0.5*pi) node[30];
rx(0.5*pi) node[31];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
cz node[30],node[31];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rx(0.5*pi) node[30];
rx(0.5*pi) node[31];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
cz node[31],node[30];
rz(0.5*pi) node[30];
rx(0.5*pi) node[30];
rz(0.5*pi) node[30];
rz(0.5*pi) node[30];
rx(0.5*pi) node[30];
rz(0.5*pi) node[30];
cz node[17],node[30];
rx(0.5*pi) node[17];
rz(0.5*pi) node[30];
rz(0.5*pi) node[17];
rx(0.5*pi) node[30];
rz(0.5*pi) node[17];
rz(0.5*pi) node[30];
rx(0.5*pi) node[17];
rz(0.5*pi) node[17];
cz node[30],node[17];
rz(0.5*pi) node[17];
rz(0.5*pi) node[30];
rx(0.5*pi) node[17];
rx(0.5*pi) node[30];
rz(0.5*pi) node[17];
rz(0.5*pi) node[30];
cz node[17],node[30];
rz(0.5*pi) node[17];
rz(0.5*pi) node[30];
rx(0.5*pi) node[17];
rx(0.5*pi) node[30];
rz(0.5*pi) node[17];
rz(0.5*pi) node[30];
cz node[30],node[17];
rz(0.5*pi) node[17];
rx(0.5*pi) node[17];
rz(0.5*pi) node[17];
rz(0.5*pi) node[17];
rx(0.5*pi) node[17];
rz(0.5*pi) node[17];
cz node[16],node[17];
rx(0.5*pi) node[16];
rz(0.5*pi) node[17];
rz(0.5*pi) node[16];
rx(0.5*pi) node[17];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rx(0.5*pi) node[16];
rz(0.5*pi) node[17];
rz(0.5*pi) node[16];
rx(0.5*pi) node[17];
cz node[23],node[16];
rz(0.5*pi) node[17];
rz(0.5*pi) node[16];
cz node[18],node[17];
rz(0.5*pi) node[23];
rx(0.5*pi) node[16];
rz(0.5*pi) node[17];
rx(0.5*pi) node[18];
rx(0.5*pi) node[23];
rz(0.5*pi) node[16];
rx(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[23];
cz node[16],node[23];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rx(0.5*pi) node[18];
rz(0.5*pi) node[23];
rx(0.5*pi) node[16];
rx(0.5*pi) node[17];
rz(0.5*pi) node[18];
rx(0.5*pi) node[23];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
cz node[19],node[18];
rz(0.5*pi) node[23];
cz node[23],node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[16];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rz(0.5*pi) node[23];
rx(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rx(0.5*pi) node[23];
rz(0.5*pi) node[16];
cz node[18],node[19];
rz(0.5*pi) node[23];
cz node[16],node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
cz node[22],node[23];
rx(0.5*pi) node[16];
rz(0.5*pi) node[17];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[16];
rx(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rx(0.5*pi) node[22];
rx(0.5*pi) node[23];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
cz node[19],node[18];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rx(0.5*pi) node[16];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
cz node[23],node[22];
rz(0.5*pi) node[16];
rx(0.5*pi) node[17];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rx(0.5*pi) node[22];
rx(0.5*pi) node[23];
cz node[18],node[17];
cz node[20],node[19];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[17];
rx(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
cz node[22],node[23];
rx(0.5*pi) node[17];
rz(0.5*pi) node[18];
rx(0.5*pi) node[19];
rx(0.5*pi) node[20];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rx(0.5*pi) node[22];
rx(0.5*pi) node[23];
rz(0.5*pi) node[17];
rx(0.5*pi) node[18];
cz node[19],node[20];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
cz node[23],node[16];
rx(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rx(0.5*pi) node[19];
rx(0.5*pi) node[20];
rz(0.5*pi) node[23];
rx(0.5*pi) node[16];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rx(0.5*pi) node[23];
rz(0.5*pi) node[16];
cz node[20],node[19];
rz(0.5*pi) node[23];
cz node[16],node[23];
rz(0.5*pi) node[19];
rz(0.5*pi) node[16];
rx(0.5*pi) node[19];
rz(0.5*pi) node[23];
rx(0.5*pi) node[16];
rz(0.5*pi) node[19];
rx(0.5*pi) node[23];
rz(0.5*pi) node[16];
cz node[19],node[18];
rz(0.5*pi) node[23];
cz node[23],node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[16];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rz(0.5*pi) node[23];
rx(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rx(0.5*pi) node[23];
rz(0.5*pi) node[16];
cz node[18],node[19];
rz(0.5*pi) node[23];
cz node[16],node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rx(0.5*pi) node[16];
rz(0.5*pi) node[17];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rz(0.5*pi) node[16];
rx(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
cz node[19],node[18];
rx(0.5*pi) node[16];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[16];
rx(0.5*pi) node[17];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
cz node[18],node[17];
cz node[56],node[19];
rz(0.5*pi) node[17];
rx(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[56];
rx(0.5*pi) node[17];
rz(0.5*pi) node[18];
rx(0.5*pi) node[19];
rx(0.5*pi) node[56];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[56];
rz(0.5*pi) node[17];
rx(0.5*pi) node[18];
cz node[19],node[56];
rx(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[56];
rz(0.5*pi) node[17];
rx(0.5*pi) node[19];
rx(0.5*pi) node[56];
rz(0.5*pi) node[19];
rz(0.5*pi) node[56];
cz node[56],node[19];
rz(0.5*pi) node[19];
rx(0.5*pi) node[19];
rz(0.5*pi) node[19];
cz node[19],node[18];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
cz node[18],node[19];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
cz node[19],node[18];
rz(0.5*pi) node[18];
rx(0.5*pi) node[18];
rz(0.5*pi) node[18];
cz node[18],node[17];
rz(0.5*pi) node[17];
rx(0.5*pi) node[18];
rx(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[17];
cz node[17],node[16];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rx(0.5*pi) node[16];
rx(0.5*pi) node[17];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
cz node[16],node[17];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rx(0.5*pi) node[16];
rx(0.5*pi) node[17];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
cz node[17],node[16];
rz(0.5*pi) node[16];
rx(0.5*pi) node[16];
rz(0.5*pi) node[16];
cz node[16],node[23];
rz(0.5*pi) node[16];
rz(0.5*pi) node[23];
rx(0.5*pi) node[16];
rx(0.5*pi) node[23];
rz(0.5*pi) node[16];
rz(0.5*pi) node[23];
cz node[23],node[16];
rz(0.5*pi) node[16];
rz(0.5*pi) node[23];
rx(0.5*pi) node[16];
rx(0.5*pi) node[23];
rz(0.5*pi) node[16];
rz(0.5*pi) node[23];
cz node[16],node[23];
rz(0.5*pi) node[23];
rx(0.5*pi) node[23];
rz(0.5*pi) node[23];
cz node[23],node[22];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rx(0.5*pi) node[22];
rx(0.5*pi) node[23];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
cz node[22],node[23];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rx(0.5*pi) node[22];
rx(0.5*pi) node[23];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
cz node[23],node[22];
rz(0.5*pi) node[22];
rx(0.5*pi) node[22];
rz(0.5*pi) node[22];
rz(0.5*pi) node[22];
rx(0.5*pi) node[22];
rz(0.5*pi) node[22];
cz node[9],node[22];
rx(0.5*pi) node[9];
rz(0.5*pi) node[22];
rz(0.5*pi) node[9];
rx(0.5*pi) node[22];
rz(0.5*pi) node[9];
rz(0.5*pi) node[22];
rx(0.5*pi) node[9];
rz(0.5*pi) node[22];
rz(0.5*pi) node[9];
rx(0.5*pi) node[22];
rz(0.5*pi) node[22];
cz node[21],node[22];
rx(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[21];
rx(0.5*pi) node[22];
rz(0.5*pi) node[22];
cz node[22],node[9];
rz(0.5*pi) node[9];
rz(0.5*pi) node[22];
rx(0.5*pi) node[9];
rx(0.5*pi) node[22];
rz(0.5*pi) node[9];
rz(0.5*pi) node[22];
cz node[9],node[22];
rz(0.5*pi) node[9];
rz(0.5*pi) node[22];
rx(0.5*pi) node[9];
rx(0.5*pi) node[22];
rz(0.5*pi) node[9];
rz(0.5*pi) node[22];
cz node[22],node[9];
rz(0.5*pi) node[9];
rx(0.5*pi) node[9];
rz(0.5*pi) node[9];
rz(0.5*pi) node[9];
rx(0.5*pi) node[9];
rz(0.5*pi) node[9];
cz node[8],node[9];
rx(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[8];
rx(0.5*pi) node[9];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rx(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[8];
rx(0.5*pi) node[9];
cz node[15],node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[8];
cz node[10],node[9];
rz(0.5*pi) node[15];
rx(0.5*pi) node[8];
rz(0.5*pi) node[9];
rx(0.5*pi) node[10];
rx(0.5*pi) node[15];
rz(0.5*pi) node[8];
rx(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[15];
cz node[8],node[15];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rx(0.5*pi) node[10];
rz(0.5*pi) node[15];
rx(0.5*pi) node[8];
rx(0.5*pi) node[9];
rz(0.5*pi) node[10];
rx(0.5*pi) node[15];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
cz node[11],node[10];
rz(0.5*pi) node[15];
cz node[15],node[8];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[8];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[15];
rx(0.5*pi) node[8];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[15];
rz(0.5*pi) node[8];
cz node[10],node[11];
rz(0.5*pi) node[15];
cz node[8],node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
cz node[14],node[15];
rx(0.5*pi) node[8];
rz(0.5*pi) node[9];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[8];
rx(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
cz node[11],node[10];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
cz node[15],node[14];
rz(0.5*pi) node[8];
rx(0.5*pi) node[9];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
cz node[10],node[9];
cz node[12],node[11];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[9];
rx(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
cz node[14],node[15];
rx(0.5*pi) node[9];
rz(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(0.5*pi) node[12];
rz(0.5*pi) node[15];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rx(0.5*pi) node[15];
rz(0.5*pi) node[9];
rx(0.5*pi) node[10];
cz node[11],node[12];
rz(0.5*pi) node[15];
cz node[15],node[8];
rx(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rx(0.5*pi) node[11];
rx(0.5*pi) node[12];
rz(0.5*pi) node[15];
rx(0.5*pi) node[8];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rx(0.5*pi) node[15];
rz(0.5*pi) node[8];
cz node[12],node[11];
rz(0.5*pi) node[15];
cz node[8],node[15];
rz(0.5*pi) node[11];
rz(0.5*pi) node[8];
rx(0.5*pi) node[11];
rz(0.5*pi) node[15];
rx(0.5*pi) node[8];
rz(0.5*pi) node[11];
rx(0.5*pi) node[15];
rz(0.5*pi) node[8];
cz node[11],node[10];
rz(0.5*pi) node[15];
cz node[15],node[8];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[8];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(0.5*pi) node[8];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[8];
cz node[10],node[11];
cz node[8],node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[8];
rz(0.5*pi) node[9];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[8];
rx(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[9];
cz node[11],node[10];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[9];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
cz node[10],node[9];
cz node[48],node[11];
rz(0.5*pi) node[9];
rx(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[48];
rx(0.5*pi) node[9];
rz(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(0.5*pi) node[48];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[48];
rz(0.5*pi) node[9];
rx(0.5*pi) node[10];
cz node[11],node[48];
rx(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[48];
rz(0.5*pi) node[9];
rx(0.5*pi) node[11];
rx(0.5*pi) node[48];
rz(0.5*pi) node[11];
rz(0.5*pi) node[48];
cz node[48],node[11];
rz(0.5*pi) node[11];
rx(0.5*pi) node[11];
rz(0.5*pi) node[11];
cz node[11],node[10];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
cz node[10],node[11];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
cz node[11],node[10];
rz(0.5*pi) node[10];
rx(0.5*pi) node[10];
rz(0.5*pi) node[10];
cz node[10],node[9];
rz(0.5*pi) node[9];
rx(0.5*pi) node[10];
rx(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[9];
barrier node[77],node[79],node[78],node[65],node[66],node[64],node[71],node[29],node[70],node[27],node[28],node[26],node[36],node[31],node[32],node[24],node[39],node[25],node[37],node[38],node[30],node[23],node[20],node[16],node[56],node[17],node[19],node[18],node[22],node[21],node[14],node[12],node[15],node[48],node[8],node[11],node[10],node[9];
measure node[77] -> c[0];
measure node[79] -> c[1];
measure node[78] -> c[2];
measure node[65] -> c[3];
measure node[66] -> c[4];
measure node[64] -> c[5];
measure node[71] -> c[6];
measure node[29] -> c[7];
measure node[70] -> c[8];
measure node[27] -> c[9];
measure node[28] -> c[10];
measure node[26] -> c[11];
measure node[36] -> c[12];
measure node[31] -> c[13];
measure node[32] -> c[14];
measure node[24] -> c[15];
measure node[39] -> c[16];
measure node[25] -> c[17];
measure node[37] -> c[18];
measure node[38] -> c[19];
measure node[30] -> c[20];
measure node[23] -> c[21];
measure node[20] -> c[22];
measure node[16] -> c[23];
measure node[56] -> c[24];
measure node[17] -> c[25];
measure node[19] -> c[26];
measure node[18] -> c[27];
measure node[22] -> c[28];
measure node[21] -> c[29];
measure node[14] -> c[30];
measure node[12] -> c[31];
measure node[15] -> c[32];
measure node[48] -> c[33];
measure node[8] -> c[34];
measure node[11] -> c[35];
measure node[10] -> c[36];
