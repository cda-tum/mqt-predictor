OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[24];
rz(pi/2) q[28];
sx q[28];
rz(pi/2) q[28];
rz(-pi/2) q[29];
rz(pi/2) q[35];
sx q[35];
rz(pi/2) q[35];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[45],q[46];
rz(pi/2) q[46];
sx q[46];
rz(pi/2) q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[28],q[35];
cx q[35],q[28];
cx q[28],q[35];
rz(-pi) q[28];
sx q[28];
rz(pi/2) q[28];
cx q[28],q[29];
sx q[28];
rz(-pi/2) q[28];
sx q[28];
rz(pi/2) q[29];
cx q[28],q[29];
rz(-pi/2) q[28];
sx q[28];
rz(-pi) q[28];
rz(-pi) q[29];
sx q[29];
rz(pi/2) q[29];
rz(pi/2) q[35];
sx q[35];
rz(pi/2) q[35];
cx q[28],q[35];
rz(pi/2) q[47];
sx q[47];
rz(pi/2) q[47];
rz(pi/2) q[48];
sx q[48];
rz(pi/2) q[48];
cx q[48],q[49];
rz(pi/2) q[49];
sx q[49];
rz(pi/2) q[49];
rz(pi/2) q[52];
sx q[52];
rz(pi/2) q[52];
rz(pi/2) q[55];
sx q[55];
rz(pi/2) q[55];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
rz(pi/2) q[49];
sx q[49];
rz(pi/2) q[49];
cx q[48],q[49];
cx q[55],q[49];
rz(pi/2) q[49];
sx q[49];
rz(pi/2) q[49];
rz(pi/2) q[57];
sx q[57];
rz(pi/2) q[57];
rz(pi/2) q[59];
sx q[59];
rz(pi/2) q[59];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[64],q[54];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[46],q[47];
rz(pi/2) q[47];
sx q[47];
rz(pi/2) q[47];
cx q[47],q[35];
rz(pi/2) q[35];
sx q[35];
rz(pi/2) q[35];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[54],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
rz(pi/2) q[78];
sx q[78];
rz(pi/2) q[78];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[91],q[79];
rz(pi/2) q[79];
sx q[79];
rz(pi/2) q[79];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[77],q[78];
cx q[78],q[77];
cx q[77],q[78];
cx q[71],q[77];
cx q[77],q[71];
cx q[71],q[77];
cx q[71],q[58];
rz(pi/2) q[58];
sx q[58];
rz(pi/2) q[58];
cx q[57],q[58];
cx q[58],q[57];
cx q[57],q[58];
cx q[56],q[57];
cx q[57],q[56];
cx q[56],q[57];
rz(pi/2) q[56];
sx q[56];
rz(pi/2) q[56];
cx q[52],q[56];
cx q[52],q[37];
rz(pi/2) q[37];
sx q[37];
rz(pi/2) q[37];
cx q[37],q[38];
cx q[38],q[37];
cx q[37],q[38];
cx q[38],q[39];
cx q[39],q[38];
cx q[38],q[39];
cx q[39],q[40];
rz(pi/2) q[56];
sx q[56];
rz(pi/2) q[56];
rz(pi/2) q[58];
sx q[58];
rz(pi/2) q[58];
cx q[59],q[58];
rz(pi/2) q[58];
sx q[58];
rz(pi/2) q[58];
cx q[58],q[71];
cx q[59],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[53],q[41];
rz(pi/2) q[41];
sx q[41];
rz(pi/2) q[41];
cx q[41],q[40];
rz(pi/2) q[40];
sx q[40];
rz(pi/2) q[40];
cx q[71],q[58];
cx q[58],q[71];
rz(pi/2) q[79];
sx q[79];
rz(pi/2) q[79];
cx q[80],q[79];
rz(pi/2) q[79];
sx q[79];
rz(pi/2) q[79];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[77],q[78];
cx q[78],q[77];
cx q[77],q[78];
rz(pi/2) q[77];
sx q[77];
rz(pi/2) q[77];
cx q[71],q[77];
rz(pi/2) q[77];
sx q[77];
rz(pi/2) q[77];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[99];
cx q[99],q[100];
cx q[100],q[99];
cx q[99],q[100];
cx q[100],q[101];
cx q[92],q[102];
rz(pi/2) q[102];
sx q[102];
rz(pi/2) q[102];
cx q[102],q[101];
rz(pi/2) q[101];
sx q[101];
rz(pi/2) q[101];
barrier q[83],q[39],q[101],q[29],q[110],q[49],q[0],q[52],q[119],q[63],q[61],q[6],q[125],q[70],q[15],q[58],q[24],q[88],q[21],q[85],q[30],q[94],q[38],q[103],q[48],q[112],q[71],q[46],q[121],q[118],q[54],q[8],q[80],q[17],q[62],q[14],q[26],q[77],q[95],q[90],q[23],q[87],q[32],q[96],q[41],q[105],q[50],q[45],q[114],q[111],q[57],q[1],q[120],q[65],q[10],q[74],q[7],q[19],q[82],q[16],q[72],q[25],q[89],q[34],q[91],q[43],q[40],q[107],q[104],q[55],q[116],q[113],q[56],q[3],q[122],q[67],q[12],q[76],q[9],q[73],q[18],q[81],q[27],q[100],q[36],q[33],q[99],q[92],q[97],q[42],q[109],q[106],q[51],q[115],q[53],q[5],q[124],q[69],q[2],q[66],q[11],q[75],q[20],q[84],q[28],q[93],q[37],q[47],q[102],q[98],q[44],q[108],q[60],q[117],q[64],q[126],q[59],q[78],q[4],q[123],q[68],q[13],q[79],q[22],q[86],q[31],q[35];
measure q[92] -> meas[0];
measure q[29] -> meas[1];
measure q[28] -> meas[2];
measure q[100] -> meas[3];
measure q[58] -> meas[4];
measure q[52] -> meas[5];
measure q[56] -> meas[6];
measure q[63] -> meas[7];
measure q[46] -> meas[8];
measure q[39] -> meas[9];
measure q[102] -> meas[10];
measure q[80] -> meas[11];
measure q[59] -> meas[12];
measure q[71] -> meas[13];
measure q[77] -> meas[14];
measure q[47] -> meas[15];
measure q[48] -> meas[16];
measure q[55] -> meas[17];
measure q[35] -> meas[18];
measure q[53] -> meas[19];
measure q[101] -> meas[20];
measure q[41] -> meas[21];
measure q[40] -> meas[22];
measure q[49] -> meas[23];
