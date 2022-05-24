OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[13];
x q[33];
sx q[37];
rz(0.30627733) q[37];
sx q[37];
sx q[38];
rz(0.29284273) q[38];
sx q[38];
sx q[39];
rz(0.28103493) q[39];
sx q[39];
cx q[33],q[39];
sx q[39];
rz(0.28103493) q[39];
sx q[39];
cx q[39],q[38];
sx q[38];
rz(0.29284273) q[38];
sx q[38];
cx q[38],q[37];
sx q[37];
rz(0.30627733) q[37];
sx q[37];
cx q[39],q[33];
cx q[38],q[39];
cx q[37],q[52];
cx q[52],q[37];
cx q[37],q[52];
cx q[38],q[37];
cx q[37],q[38];
cx q[38],q[37];
cx q[52],q[56];
cx q[56],q[52];
cx q[52],q[56];
cx q[37],q[52];
cx q[52],q[37];
cx q[37],q[52];
cx q[56],q[57];
cx q[57],q[56];
cx q[56],q[57];
cx q[52],q[56];
cx q[56],q[52];
cx q[52],q[56];
cx q[57],q[58];
cx q[58],q[57];
cx q[57],q[58];
cx q[56],q[57];
cx q[57],q[56];
cx q[56],q[57];
sx q[59];
rz(0.46364763) q[59];
sx q[59];
sx q[64];
rz(pi/6) q[64];
sx q[64];
sx q[71];
rz(0.32175053) q[71];
sx q[71];
cx q[58],q[71];
cx q[58],q[57];
sx q[71];
rz(0.32175053) q[71];
sx q[71];
sx q[77];
rz(1.9913307) q[77];
sx q[77];
rz(-pi) q[77];
cx q[71],q[77];
cx q[77],q[71];
cx q[71],q[77];
rz(pi/2) q[71];
sx q[71];
rz(pi/2) q[71];
sx q[78];
rz(1.9583931) q[78];
sx q[78];
rz(-pi) q[78];
cx q[77],q[78];
cx q[78],q[77];
cx q[77],q[78];
rz(pi/2) q[77];
sx q[77];
rz(pi/2) q[77];
sx q[79];
rz(1.9321635) q[79];
sx q[79];
rz(-pi) q[79];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
rz(pi/2) q[78];
sx q[78];
rz(pi/2) q[78];
rz(-pi/4) q[79];
sx q[79];
rz(pi/2) q[79];
sx q[84];
rz(pi/4) q[84];
sx q[84];
sx q[85];
rz(0.61547971) q[85];
sx q[85];
sx q[91];
rz(-2.8017557) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[79],q[91];
sx q[79];
rz(-pi/2) q[79];
sx q[79];
rz(pi/2) q[91];
cx q[79],q[91];
rz(-2.8017557) q[79];
sx q[79];
cx q[79],q[78];
sx q[78];
rz(0.36136713) q[78];
sx q[78];
cx q[78],q[77];
sx q[77];
rz(0.38759673) q[77];
sx q[77];
cx q[77],q[71];
sx q[71];
rz(0.42053433) q[71];
sx q[71];
cx q[58],q[71];
cx q[71],q[58];
cx q[58],q[71];
cx q[58],q[59];
sx q[59];
rz(0.46364763) q[59];
sx q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[64];
sx q[64];
rz(pi/6) q[64];
sx q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[73];
cx q[71],q[77];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[85];
cx q[77],q[71];
cx q[71],q[77];
cx q[77],q[78];
cx q[78],q[77];
cx q[77],q[78];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
x q[79];
sx q[85];
rz(0.61547971) q[85];
sx q[85];
cx q[85],q[84];
sx q[84];
rz(pi/4) q[84];
sx q[84];
sx q[91];
rz(-pi/4) q[91];
sx q[91];
cx q[79],q[91];
sx q[79];
rz(-pi/2) q[79];
sx q[79];
rz(pi/2) q[91];
cx q[79],q[91];
rz(pi/2) q[79];
sx q[79];
cx q[78],q[79];
cx q[77],q[78];
cx q[71],q[77];
cx q[58],q[71];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[63],q[62];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[73],q[66];
cx q[85],q[73];
cx q[84],q[85];
rz(pi/2) q[91];
sx q[91];
barrier q[111],q[37],q[1],q[120],q[63],q[60],q[7],q[126],q[79],q[16],q[80],q[25],q[89],q[34],q[31],q[98],q[95],q[40],q[104],q[49],q[113],q[56],q[3],q[122],q[55],q[67],q[0],q[119],q[73],q[9],q[65],q[18],q[82],q[27],q[24],q[78],q[88],q[33],q[97],q[42],q[106],q[51],q[115],q[58],q[124],q[52],q[2],q[121],q[64],q[11],q[75],q[20],q[17],q[84],q[81],q[26],q[93],q[90],q[35],q[99],q[44],q[108],q[53],q[117],q[50],q[114],q[66],q[4],q[123],q[68],q[57],q[13],q[10],q[62],q[22],q[74],q[19],q[86],q[83],q[28],q[92],q[91],q[101],q[46],q[110],q[43],q[107],q[38],q[116],q[59],q[6],q[125],q[70],q[15],q[12],q[77],q[76],q[21],q[85],q[30],q[94],q[39],q[103],q[36],q[48],q[100],q[112],q[45],q[109],q[54],q[118],q[61],q[8],q[5],q[72],q[69],q[14],q[71],q[23],q[87],q[32],q[96],q[29],q[41],q[105],q[102],q[47];
measure q[84] -> meas[0];
measure q[85] -> meas[1];
measure q[73] -> meas[2];
measure q[66] -> meas[3];
measure q[62] -> meas[4];
measure q[71] -> meas[5];
measure q[77] -> meas[6];
measure q[78] -> meas[7];
measure q[79] -> meas[8];
measure q[91] -> meas[9];
measure q[57] -> meas[10];
measure q[39] -> meas[11];
measure q[33] -> meas[12];
