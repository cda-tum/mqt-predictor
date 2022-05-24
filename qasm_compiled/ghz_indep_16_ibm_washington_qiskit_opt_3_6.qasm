OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[16];
x q[17];
x q[41];
rz(pi/2) q[112];
sx q[112];
rz(pi/2) q[112];
cx q[112],q[108];
cx q[108],q[107];
cx q[106],q[107];
cx q[107],q[106];
cx q[106],q[107];
cx q[105],q[106];
cx q[106],q[105];
cx q[105],q[106];
cx q[104],q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[103],q[104];
cx q[104],q[103];
cx q[103],q[104];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[100],q[99];
cx q[99],q[98];
cx q[98],q[91];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[28],q[35];
cx q[35],q[28];
cx q[28],q[35];
cx q[29],q[28];
cx q[28],q[29];
cx q[29],q[28];
cx q[29],q[30];
rz(pi/2) q[30];
sx q[30];
rz(-pi) q[30];
cx q[17],q[30];
sx q[17];
rz(-pi/2) q[17];
sx q[17];
rz(pi/2) q[30];
cx q[17],q[30];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[30];
sx q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[29],q[28];
cx q[35],q[28];
cx q[28],q[35];
cx q[35],q[28];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[48],q[49];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
cx q[55],q[68];
cx q[68],q[55];
cx q[55],q[68];
cx q[68],q[67];
cx q[67],q[68];
cx q[68],q[67];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[65];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[60],q[53];
rz(pi/2) q[53];
sx q[53];
rz(-pi) q[53];
cx q[41],q[53];
sx q[41];
rz(-pi/2) q[41];
sx q[41];
rz(pi/2) q[53];
cx q[41],q[53];
rz(pi/2) q[41];
sx q[41];
rz(pi/2) q[53];
sx q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[62];
cx q[62],q[72];
cx q[72],q[81];
barrier q[103],q[114],q[46],q[111],q[56],q[1],q[120],q[53],q[10],q[7],q[74],q[71],q[16],q[79],q[25],q[89],q[34],q[98],q[31],q[43],q[95],q[100],q[40],q[105],q[66],q[113],q[58],q[3],q[122],q[0],q[68],q[64],q[9],q[73],q[18],q[82],q[27],q[30],q[36],q[101],q[33],q[97],q[42],q[107],q[51],q[115],q[60],q[57],q[124],q[2],q[121],q[69],q[67],q[11],q[75],q[20],q[84],q[48],q[93],q[26],q[90],q[35],q[99],q[44],q[108],q[41],q[50],q[117],q[72],q[59],q[126],q[4],q[123],q[55],q[13],q[77],q[22],q[86],q[19],q[83],q[28],q[92],q[37],q[102],q[45],q[110],q[49],q[52],q[119],q[116],q[62],q[6],q[125],q[70],q[15],q[91],q[12],q[24],q[76],q[88],q[21],q[85],q[17],q[94],q[39],q[104],q[47],q[54],q[112],q[109],q[65],q[118],q[63],q[8],q[81],q[5],q[29],q[80],q[14],q[78],q[23],q[87],q[32],q[96],q[106],q[61],q[38];
measure q[81] -> meas[0];
measure q[72] -> meas[1];
measure q[62] -> meas[2];
measure q[61] -> meas[3];
measure q[41] -> meas[4];
measure q[53] -> meas[5];
measure q[66] -> meas[6];
measure q[48] -> meas[7];
measure q[29] -> meas[8];
measure q[17] -> meas[9];
measure q[30] -> meas[10];
measure q[98] -> meas[11];
measure q[99] -> meas[12];
measure q[100] -> meas[13];
measure q[108] -> meas[14];
measure q[112] -> meas[15];
