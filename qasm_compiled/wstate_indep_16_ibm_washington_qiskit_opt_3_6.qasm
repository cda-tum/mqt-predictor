OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[16];
sx q[46];
rz(0.36136713) q[46];
sx q[46];
sx q[47];
rz(0.33983693) q[47];
sx q[47];
sx q[53];
rz(pi/6) q[53];
sx q[53];
sx q[60];
rz(2.186276) q[60];
sx q[60];
rz(-pi) q[60];
sx q[61];
rz(0.46364763) q[61];
sx q[61];
sx q[62];
rz(0.42053433) q[62];
sx q[62];
sx q[64];
rz(0.38759673) q[64];
sx q[64];
sx q[66];
rz(0.32175053) q[66];
sx q[66];
sx q[72];
rz(pi/4) q[72];
sx q[72];
sx q[73];
rz(0.30627733) q[73];
sx q[73];
sx q[84];
rz(0.28103493) q[84];
sx q[84];
sx q[85];
rz(0.29284273) q[85];
sx q[85];
sx q[102];
rz(0.27054973) q[102];
sx q[102];
sx q[103];
rz(0.26115743) q[103];
sx q[103];
sx q[104];
rz(0.25268023) q[104];
sx q[104];
x q[105];
cx q[105],q[104];
sx q[104];
rz(0.25268023) q[104];
sx q[104];
cx q[104],q[103];
sx q[103];
rz(0.26115743) q[103];
sx q[103];
cx q[103],q[102];
sx q[102];
rz(0.27054973) q[102];
sx q[102];
cx q[104],q[105];
cx q[103],q[104];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[83],q[84];
sx q[84];
rz(0.28103493) q[84];
sx q[84];
cx q[84],q[85];
sx q[85];
rz(0.29284273) q[85];
sx q[85];
cx q[85],q[73];
sx q[73];
rz(0.30627733) q[73];
sx q[73];
cx q[73],q[66];
sx q[66];
rz(0.32175053) q[66];
sx q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[68];
cx q[68],q[67];
cx q[67],q[68];
cx q[55],q[68];
cx q[68],q[55];
cx q[55],q[68];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
cx q[48],q[49];
cx q[49],q[48];
cx q[48],q[49];
cx q[48],q[47];
sx q[47];
rz(0.33983693) q[47];
sx q[47];
cx q[47],q[46];
sx q[46];
rz(0.36136713) q[46];
sx q[46];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[64];
sx q[64];
rz(0.38759673) q[64];
sx q[64];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[63],q[62];
sx q[62];
rz(0.42053433) q[62];
sx q[62];
cx q[62],q[61];
sx q[61];
rz(0.46364763) q[61];
sx q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
rz(-1.7252939) q[60];
sx q[60];
cx q[53],q[60];
sx q[53];
rz(-pi/2) q[53];
sx q[53];
rz(pi/2) q[60];
cx q[53],q[60];
rz(-pi/2) q[53];
sx q[53];
rz(-2.9870951) q[53];
sx q[53];
rz(pi/2) q[53];
rz(-2*pi/3) q[60];
sx q[60];
rz(-pi) q[60];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
sx q[61];
rz(0.61547971) q[61];
sx q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[72];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
sx q[72];
rz(pi/4) q[72];
sx q[72];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[83],q[92];
cx q[84],q[83];
cx q[85],q[84];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[68],q[67];
cx q[67],q[68];
cx q[68],q[67];
cx q[55],q[68];
cx q[68],q[55];
cx q[55],q[68];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
cx q[48],q[49];
cx q[47],q[48];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[54],q[45];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[63],q[64];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
rz(pi/2) q[60];
cx q[53],q[60];
sx q[53];
rz(-pi/2) q[53];
sx q[53];
rz(pi/2) q[60];
cx q[53],q[60];
rz(-pi) q[53];
sx q[53];
rz(-pi) q[53];
rz(-pi) q[60];
sx q[60];
rz(-pi/2) q[60];
cx q[61],q[60];
cx q[62],q[61];
cx q[72],q[62];
barrier q[111],q[56],q[1],q[120],q[65],q[53],q[7],q[126],q[71],q[16],q[80],q[25],q[89],q[34],q[31],q[98],q[95],q[40],q[104],q[68],q[113],q[58],q[3],q[122],q[67],q[73],q[0],q[119],q[63],q[9],q[49],q[18],q[82],q[27],q[24],q[91],q[88],q[33],q[97],q[42],q[106],q[51],q[115],q[62],q[124],q[57],q[2],q[121],q[48],q[11],q[75],q[20],q[17],q[84],q[81],q[26],q[93],q[90],q[35],q[99],q[44],q[108],q[61],q[117],q[50],q[114],q[59],q[4],q[123],q[66],q[38],q[13],q[10],q[77],q[22],q[74],q[19],q[86],q[102],q[28],q[103],q[37],q[101],q[64],q[110],q[43],q[107],q[52],q[116],q[60],q[6],q[125],q[70],q[15],q[12],q[79],q[76],q[21],q[85],q[30],q[94],q[39],q[92],q[36],q[55],q[100],q[112],q[47],q[109],q[46],q[118],q[54],q[8],q[5],q[72],q[69],q[14],q[78],q[23],q[87],q[32],q[96],q[29],q[41],q[105],q[83],q[45];
measure q[72] -> meas[0];
measure q[62] -> meas[1];
measure q[61] -> meas[2];
measure q[60] -> meas[3];
measure q[53] -> meas[4];
measure q[63] -> meas[5];
measure q[64] -> meas[6];
measure q[45] -> meas[7];
measure q[48] -> meas[8];
measure q[49] -> meas[9];
measure q[85] -> meas[10];
measure q[84] -> meas[11];
measure q[83] -> meas[12];
measure q[92] -> meas[13];
measure q[104] -> meas[14];
measure q[105] -> meas[15];
