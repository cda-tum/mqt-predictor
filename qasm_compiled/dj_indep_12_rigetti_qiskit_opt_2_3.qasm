OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg c[11];
rz(-pi/2) q[11];
rx(-pi/2) q[11];
rx(-pi) q[12];
rz(pi/2) q[13];
rx(-pi/2) q[13];
cz q[13],q[12];
cz q[11],q[12];
rx(pi/2) q[11];
rz(pi) q[11];
cz q[11],q[12];
rx(pi/2) q[11];
rz(pi) q[11];
rz(-pi/2) q[12];
rx(-pi/2) q[12];
cz q[12],q[11];
rx(pi/2) q[11];
rz(pi) q[11];
rx(pi/2) q[12];
rz(pi) q[12];
cz q[11],q[12];
rx(pi/2) q[11];
rz(pi) q[11];
cz q[10],q[11];
rz(-pi/2) q[10];
rx(-pi/2) q[10];
rx(pi/2) q[11];
rz(pi) q[11];
cz q[11],q[10];
rx(pi/2) q[10];
rz(pi) q[10];
rx(pi/2) q[11];
rz(pi) q[11];
cz q[10],q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rx(pi/2) q[12];
rz(pi/2) q[12];
rx(pi/2) q[13];
rz(-pi/2) q[13];
rz(-pi/2) q[18];
rx(-pi/2) q[18];
rz(3*pi/2) q[20];
rz(-pi/2) q[21];
rx(-pi/2) q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rz(pi) q[10];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[21],q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[10],q[21];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[21],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[21],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[20],q[19];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[19],q[20];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[21],q[20];
rx(pi/2) q[20];
rz(-pi) q[20];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[20],q[21];
rx(pi/2) q[21];
rz(pi/2) q[21];
rz(3*pi/2) q[27];
rz(3*pi/2) q[28];
rz(3*pi/2) q[29];
cz q[18],q[29];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[18];
rx(pi/2) q[18];
rz(pi/2) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[18],q[29];
cz q[18],q[19];
rx(-pi) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[28],q[29];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[28],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
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
cz q[29],q[18];
rz(-pi/2) q[18];
rx(-pi/2) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[18],q[29];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[18];
rx(pi/2) q[18];
rz(pi/2) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[28],q[29];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[28];
rx(pi/2) q[28];
rz(pi/2) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[28],q[29];
rx(pi/2) q[29];
rz(pi) q[29];
rz(pi/2) q[56];
rx(-pi/2) q[56];
cz q[56],q[19];
rx(pi/2) q[56];
rz(pi/2) q[57];
rx(-pi/2) q[57];
rz(3*pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
cz q[56],q[19];
cz q[18],q[19];
rx(-pi) q[18];
cz q[29],q[18];
rz(-pi/2) q[18];
rx(-pi/2) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[18],q[29];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi/2) q[29];
cz q[29],q[18];
rx(pi/2) q[18];
rz(-pi) q[18];
cz q[18],q[19];
rx(pi/2) q[18];
rz(-pi/2) q[18];
cz q[20],q[19];
rx(pi/2) q[20];
rz(-pi/2) q[20];
rx(-pi) q[56];
cz q[57],q[56];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(-pi) q[56];
cz q[56],q[19];
rx(pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[56];
rz(-pi/2) q[56];
rx(pi/2) q[63];
rz(pi/2) q[63];
barrier q[49],q[58],q[55],q[0],q[64],q[9],q[73],q[27],q[15],q[18],q[79],q[24],q[33],q[42],q[51],q[48],q[56],q[2],q[66],q[12],q[75],q[20],q[17],q[26],q[35],q[44],q[41],q[53],q[50],q[59],q[4],q[68],q[13],q[11],q[77],q[74],q[21],q[29],q[37],q[34],q[46],q[43],q[52],q[61],q[6],q[3],q[70],q[67],q[19],q[76],q[10],q[30],q[39],q[36],q[45],q[54],q[57],q[8],q[60],q[5],q[72],q[69],q[14],q[78],q[23],q[32],q[28],q[38],q[47],q[63],q[1],q[65],q[62],q[7],q[71],q[16],q[25],q[22],q[31],q[40];
measure q[13] -> c[0];
measure q[12] -> c[1];
measure q[27] -> c[2];
measure q[28] -> c[3];
measure q[63] -> c[4];
measure q[57] -> c[5];
measure q[29] -> c[6];
measure q[18] -> c[7];
measure q[21] -> c[8];
measure q[20] -> c[9];
measure q[56] -> c[10];
