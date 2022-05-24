OPENQASM 2.0;
include "qelib1.inc";

qreg node[80];
creg meas[40];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[42];
rz(0.5*pi) node[43];
rz(0.5*pi) node[44];
rz(0.5*pi) node[45];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[54];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[68];
rz(0.5*pi) node[69];
rz(0.5*pi) node[70];
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
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[19];
rx(0.5*pi) node[20];
rx(0.5*pi) node[21];
rx(0.5*pi) node[40];
rx(0.5*pi) node[41];
rx(0.5*pi) node[42];
rx(0.5*pi) node[43];
rx(0.5*pi) node[44];
rx(0.5*pi) node[45];
rx(0.5*pi) node[46];
rx(0.5*pi) node[47];
rx(0.5*pi) node[54];
rx(0.5*pi) node[56];
rx(0.5*pi) node[57];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
rx(0.5*pi) node[68];
rx(0.5*pi) node[69];
rx(0.5*pi) node[70];
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
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[42];
rz(0.5*pi) node[43];
rz(0.5*pi) node[44];
rz(0.5*pi) node[45];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[54];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[68];
rz(0.5*pi) node[69];
rz(0.5*pi) node[70];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[79],node[72];
rz(0.5*pi) node[72];
rx(0.5*pi) node[72];
rz(0.5*pi) node[72];
cz node[72],node[73];
rz(0.5*pi) node[73];
rx(0.5*pi) node[73];
rz(0.5*pi) node[73];
cz node[73],node[74];
rz(0.5*pi) node[74];
rx(0.5*pi) node[74];
rz(0.5*pi) node[74];
cz node[74],node[75];
rz(0.5*pi) node[75];
rx(0.5*pi) node[75];
rz(0.5*pi) node[75];
cz node[75],node[76];
rz(0.5*pi) node[76];
rx(0.5*pi) node[76];
rz(0.5*pi) node[76];
cz node[76],node[77];
rz(0.5*pi) node[77];
rx(0.5*pi) node[77];
rz(0.5*pi) node[77];
cz node[77],node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[78],node[65];
rz(0.5*pi) node[65];
rx(0.5*pi) node[65];
rz(0.5*pi) node[65];
cz node[65],node[66];
rz(0.5*pi) node[66];
rx(0.5*pi) node[66];
rz(0.5*pi) node[66];
cz node[66],node[67];
rz(0.5*pi) node[67];
rx(0.5*pi) node[67];
rz(0.5*pi) node[67];
cz node[67],node[68];
rz(0.5*pi) node[68];
rx(0.5*pi) node[68];
rz(0.5*pi) node[68];
cz node[68],node[69];
rz(0.5*pi) node[69];
rx(0.5*pi) node[69];
rz(0.5*pi) node[69];
cz node[69],node[70];
rz(0.5*pi) node[70];
rx(0.5*pi) node[70];
rz(0.5*pi) node[70];
cz node[70],node[57];
rz(0.5*pi) node[57];
rx(0.5*pi) node[57];
rz(0.5*pi) node[57];
cz node[57],node[56];
rz(0.5*pi) node[56];
rx(0.5*pi) node[56];
rz(0.5*pi) node[56];
cz node[56],node[19];
rz(0.5*pi) node[19];
rx(0.5*pi) node[19];
rz(0.5*pi) node[19];
cz node[19],node[20];
rz(0.5*pi) node[20];
rx(0.5*pi) node[20];
rz(0.5*pi) node[20];
cz node[20],node[21];
rz(0.5*pi) node[21];
rx(0.5*pi) node[21];
rz(0.5*pi) node[21];
cz node[21],node[10];
rz(0.5*pi) node[10];
rx(0.5*pi) node[10];
rz(0.5*pi) node[10];
cz node[10],node[11];
rz(0.5*pi) node[11];
rx(0.5*pi) node[11];
rz(0.5*pi) node[11];
cz node[11],node[12];
rz(0.5*pi) node[12];
rx(0.5*pi) node[12];
rz(0.5*pi) node[12];
cz node[12],node[13];
rz(0.5*pi) node[13];
rx(0.5*pi) node[13];
rz(0.5*pi) node[13];
cz node[13],node[2];
rz(0.5*pi) node[2];
rx(0.5*pi) node[2];
rz(0.5*pi) node[2];
cz node[2],node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[1],node[0];
rz(0.5*pi) node[0];
rx(0.5*pi) node[0];
rz(0.5*pi) node[0];
cz node[0],node[7];
rz(0.5*pi) node[7];
rx(0.5*pi) node[7];
rz(0.5*pi) node[7];
cz node[7],node[6];
rz(0.5*pi) node[6];
rx(0.5*pi) node[6];
rz(0.5*pi) node[6];
cz node[6],node[5];
rz(0.5*pi) node[5];
rx(0.5*pi) node[5];
rz(0.5*pi) node[5];
cz node[5],node[4];
rz(0.5*pi) node[4];
rx(0.5*pi) node[4];
rz(0.5*pi) node[4];
cz node[4],node[3];
rz(0.5*pi) node[3];
rx(0.5*pi) node[3];
rz(0.5*pi) node[3];
cz node[3],node[40];
rz(0.5*pi) node[40];
rx(0.5*pi) node[40];
rz(0.5*pi) node[40];
cz node[40],node[47];
rz(0.5*pi) node[47];
rx(0.5*pi) node[47];
rz(0.5*pi) node[47];
cz node[47],node[46];
rz(0.5*pi) node[46];
rx(0.5*pi) node[46];
rz(0.5*pi) node[46];
cz node[46],node[45];
rz(0.5*pi) node[45];
rx(0.5*pi) node[45];
rz(0.5*pi) node[45];
cz node[45],node[44];
rz(0.5*pi) node[44];
rx(0.5*pi) node[44];
rz(0.5*pi) node[44];
cz node[44],node[43];
rz(0.5*pi) node[43];
rx(0.5*pi) node[43];
rz(0.5*pi) node[43];
cz node[43],node[42];
rz(0.5*pi) node[42];
rx(0.5*pi) node[42];
rz(0.5*pi) node[42];
cz node[42],node[41];
rz(0.5*pi) node[41];
rx(0.5*pi) node[41];
rz(0.5*pi) node[41];
cz node[41],node[54];
rz(0.5*pi) node[54];
rx(0.5*pi) node[54];
rz(0.5*pi) node[54];
barrier node[54],node[41],node[42],node[43],node[44],node[45],node[46],node[47],node[40],node[3],node[4],node[5],node[6],node[7],node[0],node[1],node[2],node[13],node[12],node[11],node[10],node[21],node[20],node[19],node[56],node[57],node[70],node[69],node[68],node[67],node[66],node[65],node[78],node[77],node[76],node[75],node[74],node[73],node[72],node[79];
measure node[54] -> meas[0];
measure node[41] -> meas[1];
measure node[42] -> meas[2];
measure node[43] -> meas[3];
measure node[44] -> meas[4];
measure node[45] -> meas[5];
measure node[46] -> meas[6];
measure node[47] -> meas[7];
measure node[40] -> meas[8];
measure node[3] -> meas[9];
measure node[4] -> meas[10];
measure node[5] -> meas[11];
measure node[6] -> meas[12];
measure node[7] -> meas[13];
measure node[0] -> meas[14];
measure node[1] -> meas[15];
measure node[2] -> meas[16];
measure node[13] -> meas[17];
measure node[12] -> meas[18];
measure node[11] -> meas[19];
measure node[10] -> meas[20];
measure node[21] -> meas[21];
measure node[20] -> meas[22];
measure node[19] -> meas[23];
measure node[56] -> meas[24];
measure node[57] -> meas[25];
measure node[70] -> meas[26];
measure node[69] -> meas[27];
measure node[68] -> meas[28];
measure node[67] -> meas[29];
measure node[66] -> meas[30];
measure node[65] -> meas[31];
measure node[78] -> meas[32];
measure node[77] -> meas[33];
measure node[76] -> meas[34];
measure node[75] -> meas[35];
measure node[74] -> meas[36];
measure node[73] -> meas[37];
measure node[72] -> meas[38];
measure node[79] -> meas[39];
