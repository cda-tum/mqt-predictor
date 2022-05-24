OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg c[34];
rz(pi/2) q[39];
sx q[39];
rz(pi/2) q[39];
rz(-pi/2) q[40];
sx q[40];
rz(-pi) q[40];
rz(-3*pi/2) q[41];
sx q[41];
rz(-pi/2) q[41];
cx q[40],q[41];
sx q[40];
rz(-pi/2) q[40];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
cx q[40],q[41];
rz(pi/2) q[40];
sx q[40];
rz(pi/2) q[40];
x q[41];
rz(-pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
rz(-pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
rz(pi/2) q[46];
sx q[46];
rz(pi/2) q[46];
rz(-pi/2) q[53];
cx q[41],q[53];
sx q[41];
rz(-pi/2) q[41];
sx q[41];
rz(pi/2) q[53];
cx q[41],q[53];
rz(-pi) q[41];
sx q[41];
rz(-pi/2) q[53];
sx q[53];
rz(-pi) q[53];
rz(-pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(-pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[53],q[60];
cx q[60],q[53];
rz(-3*pi/2) q[53];
sx q[53];
rz(-pi/2) q[53];
rz(-pi) q[60];
x q[60];
rz(pi/2) q[61];
cx q[60],q[61];
sx q[60];
rz(-pi/2) q[60];
sx q[60];
rz(pi/2) q[61];
cx q[60],q[61];
sx q[60];
rz(pi/2) q[61];
sx q[61];
rz(-pi) q[61];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(-pi/2) q[63];
sx q[63];
rz(-pi) q[63];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
rz(-pi/2) q[65];
sx q[65];
rz(pi/2) q[65];
x q[66];
rz(-pi/2) q[67];
sx q[67];
rz(-pi) q[67];
rz(-pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(-pi/2) q[73];
sx q[73];
rz(-pi) q[73];
rz(pi/2) q[79];
sx q[79];
rz(pi/2) q[79];
rz(pi/2) q[80];
sx q[80];
rz(pi/2) q[80];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[81];
rz(-pi/2) q[84];
rz(-pi/2) q[85];
sx q[85];
rz(-pi) q[85];
rz(pi/2) q[86];
sx q[86];
rz(pi) q[86];
rz(pi/2) q[87];
sx q[87];
rz(pi) q[87];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
rz(pi/2) q[93];
sx q[93];
rz(pi/2) q[93];
rz(-pi/2) q[100];
sx q[100];
rz(pi/2) q[100];
rz(-pi/2) q[101];
sx q[101];
rz(pi/2) q[101];
rz(-pi/2) q[102];
sx q[102];
rz(pi/2) q[102];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[92],q[83];
rz(-3*pi/2) q[92];
sx q[92];
rz(-pi/2) q[92];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[92],q[83];
rz(-3*pi/2) q[92];
sx q[92];
rz(-pi/2) q[92];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[83];
rz(-3*pi/2) q[92];
sx q[92];
rz(-pi/2) q[92];
rz(-pi/2) q[103];
sx q[103];
rz(pi/2) q[103];
rz(-pi/2) q[110];
sx q[110];
rz(pi/2) q[110];
cx q[100],q[110];
cx q[110],q[100];
cx q[100],q[110];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
cx q[92],q[83];
rz(-3*pi/2) q[92];
sx q[92];
rz(-pi/2) q[92];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[83];
rz(-3*pi/2) q[92];
sx q[92];
rz(-pi/2) q[92];
rz(-pi/2) q[118];
sx q[118];
rz(pi/2) q[118];
cx q[110],q[118];
cx q[118],q[110];
cx q[110],q[118];
cx q[100],q[110];
cx q[110],q[100];
cx q[100],q[110];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[118],q[110];
cx q[110],q[118];
cx q[118],q[110];
cx q[110],q[100];
cx q[100],q[110];
cx q[110],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[92],q[83];
rz(-3*pi/2) q[92];
sx q[92];
rz(-pi/2) q[92];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[83];
x q[83];
cx q[83],q[84];
sx q[83];
rz(-pi/2) q[83];
sx q[83];
rz(pi/2) q[84];
cx q[83],q[84];
rz(-pi) q[83];
sx q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[79],q[80];
cx q[80],q[79];
cx q[79],q[80];
rz(-pi/2) q[84];
sx q[84];
rz(-pi) q[84];
cx q[85],q[84];
cx q[83],q[84];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[83],q[84];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[83],q[84];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[83],q[84];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[83],q[84];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[83],q[84];
rz(-3*pi/2) q[83];
sx q[83];
rz(-pi/2) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[84];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
sx q[85];
rz(-pi/2) q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
sx q[73];
rz(-pi/2) q[73];
cx q[86],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
rz(pi/2) q[73];
cx q[66],q[73];
sx q[66];
rz(-pi/2) q[66];
sx q[66];
rz(pi/2) q[73];
cx q[66],q[73];
sx q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[54],q[64];
rz(-3*pi/2) q[54];
sx q[54];
rz(-pi/2) q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[63],q[64];
cx q[54],q[64];
rz(-3*pi/2) q[54];
sx q[54];
rz(-pi/2) q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[54],q[64];
rz(-3*pi/2) q[54];
sx q[54];
rz(-pi/2) q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[64];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
sx q[63];
rz(-pi/2) q[63];
x q[64];
rz(pi/2) q[65];
sx q[65];
rz(-pi) q[65];
cx q[64],q[65];
sx q[64];
rz(-pi/2) q[64];
sx q[64];
rz(pi/2) q[65];
cx q[64],q[65];
rz(-pi) q[64];
sx q[64];
rz(-pi) q[64];
rz(pi/2) q[65];
sx q[65];
x q[65];
rz(-pi) q[66];
sx q[66];
cx q[65],q[66];
sx q[65];
rz(-pi/2) q[65];
sx q[65];
rz(pi/2) q[66];
cx q[65],q[66];
rz(-pi/2) q[66];
sx q[66];
rz(-pi) q[66];
cx q[67],q[66];
sx q[67];
rz(-pi/2) q[67];
rz(pi/2) q[73];
sx q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
sx q[86];
rz(pi/2) q[86];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[87],q[86];
sx q[87];
rz(pi/2) q[87];
cx q[87],q[93];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[93],q[87];
cx q[87],q[93];
cx q[87],q[86];
rz(pi/2) q[87];
sx q[87];
rz(pi/2) q[87];
barrier q[28],q[92],q[37],q[110],q[54],q[100],q[55],q[0],q[119],q[64],q[21],q[84],q[30],q[94],q[40],q[101],q[48],q[112],q[57],q[121],q[14],q[78],q[23],q[93],q[32],q[96],q[86],q[105],q[50],q[114],q[7],q[71],q[16],q[61],q[25],q[89],q[34],q[98],q[43],q[107],q[9],q[73],q[18],q[80],q[27],q[91],q[36],q[103],q[44],q[109],q[2],q[66],q[11],q[75],q[20],q[79],q[29],q[87],q[38],q[118],q[59],q[4],q[123],q[68],q[13],q[77],q[22],q[85],q[31],q[95],q[52],q[116],q[60],q[6],q[125],q[70],q[15],q[72],q[24],q[88],q[45],q[102],q[63],q[8],q[82],q[17],q[81],q[90],q[47],q[111],q[56],q[1],q[120],q[65],q[10],q[74],q[19],q[62],q[39],q[104],q[49],q[113],q[58],q[3],q[122],q[67],q[12],q[76],q[33],q[97],q[42],q[106],q[51],q[115],q[53],q[5],q[124],q[69],q[26],q[35],q[99],q[46],q[108],q[41],q[117],q[83],q[126];
measure q[39] -> c[0];
measure q[40] -> c[1];
measure q[41] -> c[2];
measure q[60] -> c[3];
measure q[53] -> c[4];
measure q[118] -> c[5];
measure q[110] -> c[6];
measure q[103] -> c[7];
measure q[100] -> c[8];
measure q[101] -> c[9];
measure q[102] -> c[10];
measure q[92] -> c[11];
measure q[79] -> c[12];
measure q[84] -> c[13];
measure q[80] -> c[14];
measure q[61] -> c[15];
measure q[62] -> c[16];
measure q[72] -> c[17];
measure q[81] -> c[18];
measure q[82] -> c[19];
measure q[83] -> c[20];
measure q[73] -> c[21];
measure q[85] -> c[22];
measure q[66] -> c[23];
measure q[44] -> c[24];
measure q[63] -> c[25];
measure q[46] -> c[26];
measure q[45] -> c[27];
measure q[54] -> c[28];
measure q[64] -> c[29];
measure q[65] -> c[30];
measure q[67] -> c[31];
measure q[93] -> c[32];
measure q[87] -> c[33];
