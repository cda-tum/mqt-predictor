OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[25];
rz(-pi) q[23];
x q[23];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[24];
rz(pi/2) q[24];
sx q[24];
rz(-0.11046123) q[24];
sx q[24];
cx q[23],q[24];
sx q[23];
rz(-pi/2) q[23];
sx q[23];
rz(pi/2) q[24];
cx q[23],q[24];
rz(-pi/2) q[23];
sx q[23];
rz(-3.0311314) q[23];
rz(-pi/2) q[24];
sx q[24];
rz(pi/2) q[27];
sx q[27];
rz(pi/2) q[27];
rz(pi/2) q[28];
sx q[28];
rz(pi/2) q[28];
rz(-pi/2) q[29];
cx q[28],q[35];
sx q[28];
rz(-pi/2) q[28];
cx q[28],q[29];
sx q[28];
rz(-pi/2) q[28];
sx q[28];
rz(pi/2) q[29];
cx q[28],q[29];
sx q[28];
rz(-pi) q[28];
cx q[27],q[28];
cx q[27],q[26];
rz(pi/2) q[26];
sx q[26];
rz(pi/2) q[26];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[25],q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[27];
rz(pi/2) q[28];
sx q[28];
rz(pi/2) q[28];
cx q[27],q[28];
cx q[28],q[27];
cx q[27],q[28];
sx q[29];
rz(pi/2) q[29];
rz(pi/2) q[35];
sx q[35];
rz(pi/2) q[35];
rz(pi/2) q[47];
sx q[47];
rz(pi/2) q[47];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
rz(pi/2) q[35];
sx q[35];
rz(pi/2) q[35];
cx q[28],q[35];
cx q[47],q[35];
rz(pi/2) q[35];
sx q[35];
rz(pi/2) q[35];
rz(pi/2) q[62];
sx q[62];
rz(-pi/2) q[62];
rz(-pi/2) q[63];
rz(pi/2) q[76];
sx q[76];
rz(pi/2) q[76];
cx q[76],q[77];
rz(pi/2) q[77];
sx q[77];
rz(pi/2) q[77];
cx q[77],q[78];
cx q[76],q[77];
cx q[77],q[76];
cx q[76],q[77];
rz(pi/2) q[78];
sx q[78];
rz(pi/2) q[78];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
rz(pi/2) q[84];
sx q[84];
rz(pi/2) q[84];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[91],q[98];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[79],q[80];
rz(pi/2) q[80];
sx q[80];
rz(pi/2) q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[72],q[81];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
rz(pi/2) q[79];
sx q[79];
rz(pi/2) q[79];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
sx q[62];
rz(pi/2) q[62];
cx q[62],q[63];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[63];
cx q[62],q[63];
rz(-pi/2) q[62];
sx q[62];
rz(-pi) q[62];
rz(-pi) q[63];
sx q[63];
rz(pi/2) q[63];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[91],q[79];
rz(pi/2) q[79];
sx q[79];
rz(pi/2) q[79];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
rz(pi/2) q[78];
sx q[78];
rz(pi/2) q[78];
cx q[77],q[78];
rz(pi/2) q[78];
sx q[78];
rz(pi/2) q[78];
cx q[79],q[80];
cx q[80],q[79];
cx q[79],q[80];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
rz(pi/2) q[110];
sx q[110];
rz(pi/2) q[110];
rz(-pi) q[117];
x q[117];
cx q[110],q[118];
cx q[110],q[100];
rz(pi/2) q[100];
sx q[100];
rz(pi/2) q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
rz(-pi/2) q[118];
sx q[118];
rz(-2.9276986) q[118];
sx q[118];
cx q[117],q[118];
sx q[117];
rz(-pi/2) q[117];
sx q[117];
rz(pi/2) q[118];
cx q[117],q[118];
rz(pi/2) q[117];
sx q[117];
rz(-2.9276986) q[117];
sx q[118];
cx q[110],q[118];
cx q[118],q[110];
cx q[110],q[118];
rz(pi/2) q[110];
sx q[110];
rz(pi/2) q[110];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
rz(pi/2) q[102];
sx q[102];
rz(pi/2) q[102];
cx q[92],q[83];
cx q[84],q[83];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[102];
rz(pi/2) q[102];
sx q[102];
rz(pi/2) q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[110];
rz(pi/2) q[110];
sx q[110];
rz(pi/2) q[110];
barrier q[100],q[37],q[101],q[46],q[118],q[55],q[0],q[52],q[119],q[64],q[61],q[6],q[125],q[70],q[15],q[98],q[23],q[88],q[21],q[85],q[30],q[94],q[39],q[103],q[48],q[112],q[57],q[54],q[121],q[117],q[92],q[8],q[80],q[17],q[78],q[14],q[25],q[81],q[95],q[90],q[24],q[87],q[32],q[96],q[41],q[105],q[50],q[35],q[114],q[111],q[56],q[1],q[120],q[65],q[10],q[74],q[7],q[19],q[84],q[16],q[62],q[28],q[89],q[34],q[91],q[43],q[40],q[107],q[104],q[49],q[116],q[113],q[58],q[3],q[122],q[67],q[12],q[77],q[9],q[73],q[18],q[72],q[26],q[79],q[36],q[33],q[83],q[45],q[97],q[42],q[109],q[106],q[51],q[115],q[60],q[5],q[124],q[69],q[2],q[66],q[11],q[75],q[20],q[82],q[27],q[93],q[38],q[47],q[102],q[99],q[44],q[108],q[53],q[110],q[63],q[126],q[59],q[71],q[4],q[123],q[68],q[13],q[76],q[22],q[86],q[31],q[29];
measure q[28] -> meas[0];
measure q[23] -> meas[1];
measure q[79] -> meas[2];
measure q[91] -> meas[3];
measure q[29] -> meas[4];
measure q[47] -> meas[5];
measure q[118] -> meas[6];
measure q[117] -> meas[7];
measure q[83] -> meas[8];
measure q[63] -> meas[9];
measure q[62] -> meas[10];
measure q[77] -> meas[11];
measure q[76] -> meas[12];
measure q[26] -> meas[13];
measure q[27] -> meas[14];
measure q[35] -> meas[15];
measure q[81] -> meas[16];
measure q[78] -> meas[17];
measure q[92] -> meas[18];
measure q[100] -> meas[19];
measure q[84] -> meas[20];
measure q[110] -> meas[21];
measure q[82] -> meas[22];
measure q[25] -> meas[23];
measure q[24] -> meas[24];
