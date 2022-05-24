OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg c[8];
rz(-pi/2) q[23];
sx q[23];
rz(-pi) q[23];
rz(-3*pi/2) q[24];
sx q[24];
rz(-pi/2) q[24];
cx q[23],q[24];
sx q[23];
rz(-pi/2) q[23];
rz(-pi/2) q[25];
sx q[25];
rz(-pi) q[25];
cx q[25],q[24];
sx q[25];
rz(-pi/2) q[25];
cx q[24],q[34];
cx q[34],q[24];
cx q[24],q[34];
cx q[34],q[43];
cx q[43],q[34];
cx q[34],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(pi/2) q[80];
sx q[80];
rz(pi) q[80];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[80],q[81];
sx q[80];
rz(pi/2) q[80];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(-pi/2) q[84];
sx q[84];
rz(-pi) q[84];
cx q[84],q[83];
sx q[84];
rz(-pi/2) q[84];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[83],q[92];
cx q[92],q[83];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
rz(-pi/2) q[101];
sx q[101];
rz(pi/2) q[101];
rz(-pi/2) q[102];
sx q[102];
rz(-pi) q[102];
cx q[102],q[92];
sx q[102];
rz(-pi/2) q[102];
rz(-pi/2) q[103];
sx q[103];
rz(pi/2) q[103];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
cx q[102],q[92];
rz(-3*pi/2) q[102];
sx q[102];
rz(-pi/2) q[102];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[92];
rz(-3*pi/2) q[102];
sx q[102];
rz(-pi/2) q[102];
barrier q[13],q[77],q[22],q[86],q[31],q[28],q[95],q[83],q[37],q[102],q[46],q[110],q[55],q[0],q[119],q[52],q[54],q[116],q[61],q[6],q[125],q[70],q[15],q[79],q[92],q[21],q[88],q[85],q[30],q[94],q[39],q[101],q[48],q[112],q[57],q[121],q[45],q[118],q[64],q[8],q[62],q[17],q[14],q[72],q[78],q[23],q[90],q[87],q[32],q[96],q[41],q[105],q[50],q[114],q[47],q[111],q[56],q[1],q[120],q[65],q[10],q[7],q[74],q[19],q[71],q[16],q[82],q[80],q[25],q[89],q[24],q[98],q[34],q[107],q[40],q[104],q[49],q[113],q[58],q[3],q[122],q[67],q[12],q[9],q[76],q[73],q[18],q[81],q[27],q[91],q[36],q[100],q[33],q[44],q[97],q[109],q[42],q[106],q[51],q[115],q[60],q[5],q[124],q[2],q[69],q[66],q[11],q[75],q[20],q[84],q[29],q[93],q[26],q[38],q[103],q[35],q[99],q[43],q[108],q[53],q[126],q[117],q[63],q[59],q[4],q[123],q[68];
measure q[23] -> c[0];
measure q[25] -> c[1];
measure q[80] -> c[2];
measure q[84] -> c[3];
measure q[103] -> c[4];
measure q[101] -> c[5];
measure q[102] -> c[6];
measure q[83] -> c[7];
