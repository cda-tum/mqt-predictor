OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[12];
rz(-pi) q[18];
x q[18];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
cx q[33],q[39];
cx q[39],q[33];
cx q[33],q[39];
cx q[20],q[33];
cx q[33],q[20];
cx q[20],q[33];
cx q[20],q[19];
rz(-pi/2) q[19];
sx q[19];
rz(-2.9276986) q[19];
sx q[19];
cx q[18],q[19];
sx q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[19];
cx q[18],q[19];
rz(pi/2) q[18];
sx q[18];
rz(-2.9276986) q[18];
sx q[19];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[33];
cx q[33],q[20];
cx q[20],q[33];
cx q[33],q[39];
cx q[39],q[33];
cx q[33],q[39];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
rz(pi/2) q[40];
sx q[40];
rz(pi/2) q[40];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
cx q[79],q[78];
rz(pi/2) q[78];
sx q[78];
rz(pi/2) q[78];
cx q[78],q[77];
rz(pi/2) q[77];
sx q[77];
rz(pi/2) q[77];
cx q[71],q[77];
cx q[77],q[71];
cx q[71],q[77];
cx q[58],q[71];
cx q[71],q[58];
cx q[58],q[71];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[41],q[40];
rz(pi/2) q[40];
sx q[40];
rz(pi/2) q[40];
rz(pi/2) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[84],q[83];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[83],q[92];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
rz(pi/2) q[102];
sx q[102];
rz(pi/2) q[102];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[83],q[92];
rz(pi/2) q[103];
sx q[103];
rz(pi/2) q[103];
cx q[103],q[104];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
cx q[102],q[92];
rz(pi/2) q[103];
sx q[103];
rz(pi/2) q[103];
rz(pi/2) q[104];
sx q[104];
rz(pi/2) q[104];
cx q[104],q[103];
rz(pi/2) q[103];
sx q[103];
rz(pi/2) q[103];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
barrier q[103],q[37],q[101],q[46],q[110],q[55],q[0],q[52],q[119],q[64],q[79],q[6],q[125],q[70],q[15],q[80],q[24],q[88],q[21],q[85],q[30],q[94],q[39],q[102],q[48],q[112],q[57],q[54],q[121],q[118],q[63],q[8],q[62],q[17],q[72],q[14],q[26],q[78],q[95],q[90],q[23],q[87],q[32],q[96],q[60],q[105],q[50],q[47],q[114],q[111],q[56],q[1],q[120],q[65],q[10],q[74],q[7],q[18],q[84],q[16],q[81],q[25],q[89],q[34],q[98],q[43],q[53],q[107],q[104],q[49],q[116],q[113],q[71],q[3],q[122],q[67],q[12],q[76],q[9],q[73],q[40],q[82],q[27],q[91],q[36],q[33],q[100],q[45],q[97],q[42],q[109],q[106],q[51],q[115],q[19],q[5],q[124],q[69],q[2],q[66],q[11],q[75],q[20],q[83],q[29],q[93],q[38],q[35],q[92],q[99],q[44],q[108],q[59],q[117],q[61],q[126],q[58],q[77],q[4],q[123],q[68],q[13],q[41],q[22],q[86],q[31],q[28];
measure q[83] -> meas[0];
measure q[84] -> meas[1];
measure q[102] -> meas[2];
measure q[104] -> meas[3];
measure q[19] -> meas[4];
measure q[79] -> meas[5];
measure q[103] -> meas[6];
measure q[78] -> meas[7];
measure q[18] -> meas[8];
measure q[41] -> meas[9];
measure q[92] -> meas[10];
measure q[40] -> meas[11];
