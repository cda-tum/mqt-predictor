OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg meas[7];
rx(-pi/2) q[19];
rz(2.1867167) q[19];
rx(-pi/2) q[19];
rx(-pi/2) q[20];
rz(2.4382268) q[20];
rx(-pi/2) q[20];
rx(-pi/2) q[56];
rz(1.1027038) q[56];
rx(-pi/2) q[56];
rx(-pi/2) q[57];
rz(0.19802069) q[57];
rx(-pi/2) q[57];
rx(pi/2) q[62];
rz(0.31161179) q[62];
rx(pi/2) q[62];
rx(-pi/2) q[63];
rz(2.2881757) q[63];
rx(pi/2) q[63];
cz q[63],q[56];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
cz q[63],q[20];
cz q[63],q[62];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
cz q[56],q[19];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[57],q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rz(-pi/2) q[20];
rx(-pi/2) q[20];
cz q[63],q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
cz q[20],q[19];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[56],q[19];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
cz q[56],q[19];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[19],q[20];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[20],q[19];
rx(pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[63],q[62];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[70];
rz(0.54227512) q[70];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rz(0.72227413) q[57];
rx(pi/2) q[57];
cz q[57],q[70];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[19],q[56];
rx(-pi/2) q[19];
rz(1.5946227) q[19];
rx(-pi/2) q[19];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[57],q[56];
rx(-pi/2) q[57];
rz(0.0028387759) q[57];
rx(pi/2) q[57];
cz q[57],q[70];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
cz q[63],q[56];
rx(pi/2) q[63];
rz(0.049582854) q[63];
rx(pi/2) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi) q[20];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[63],q[56];
rx(-pi/2) q[63];
rz(1.6457644) q[63];
rx(pi/2) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[56],q[19];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[56],q[19];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(0.34670352) q[62];
rx(-pi/2) q[62];
rx(-pi/2) q[63];
rz(1.9453325) q[63];
rx(pi/2) q[63];
cz q[62],q[63];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[56],q[19];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[19],q[20];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[20],q[19];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[19],q[20];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[20],q[19];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[63],q[62];
rx(pi/2) q[63];
rz(0.70221577) q[63];
rx(-pi/2) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi) q[62];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[56],q[19];
cz q[20],q[19];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[19],q[20];
rx(pi/2) q[20];
rz(pi/2) q[20];
cz q[20],q[63];
rx(pi/2) q[20];
rz(1.0382379) q[20];
rx(-pi/2) q[20];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[56],q[19];
rx(pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi/2) q[56];
cz q[56],q[63];
rx(pi/2) q[56];
rz(0.20693068) q[56];
rx(-pi/2) q[56];
cz q[56],q[63];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(0.62543629) q[19];
rx(-pi/2) q[19];
rx(pi/2) q[63];
rz(pi/2) q[63];
rx(pi/2) q[70];
rz(pi/2) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[70],q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(2.2299348) q[56];
rx(pi/2) q[56];
rx(-pi/2) q[57];
rz(0.82860923) q[57];
rx(-pi/2) q[57];
rx(-pi/2) q[70];
rz(0.55953512) q[70];
rx(pi/2) q[70];
barrier q[57],q[3],q[15],q[79],q[12],q[76],q[21],q[30],q[39],q[36],q[45],q[54],q[62],q[8],q[72],q[5],q[69],q[14],q[78],q[23],q[32],q[29],q[41],q[38],q[47],q[20],q[1],q[65],q[19],q[7],q[71],q[16],q[25],q[34],q[31],q[40],q[49],q[58],q[55],q[67],q[0],q[64],q[9],q[73],q[18],q[27],q[24],q[33],q[42],q[51],q[48],q[60],q[56],q[2],q[66],q[11],q[75],q[63],q[17],q[26],q[35],q[44],q[53],q[50],q[59],q[4],q[68],q[13],q[10],q[77],q[22],q[74],q[70],q[28],q[37],q[46],q[43],q[52],q[61],q[6];
measure q[62] -> meas[0];
measure q[20] -> meas[1];
measure q[63] -> meas[2];
measure q[19] -> meas[3];
measure q[70] -> meas[4];
measure q[56] -> meas[5];
measure q[57] -> meas[6];
