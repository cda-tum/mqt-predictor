OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg c[5];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
rz(pi/2) q[100];
sx q[100];
rz(pi/2) q[100];
rz(pi/2) q[101];
sx q[101];
rz(pi/2) q[101];
x q[102];
rz(-5*pi/16) q[102];
cx q[102],q[101];
rz(7*pi/16) q[101];
cx q[102],q[101];
rz(-7*pi/16) q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
rz(-pi/32) q[100];
rz(pi/2) q[101];
sx q[101];
rz(pi/2) q[101];
cx q[102],q[92];
rz(-pi/8) q[92];
cx q[102],q[92];
rz(pi/8) q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
rz(3*pi/4) q[92];
cx q[92],q[83];
rz(-pi/4) q[83];
cx q[92],q[83];
rz(pi/4) q[83];
rz(pi/2) q[103];
sx q[103];
rz(pi/2) q[103];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
cx q[92],q[102];
rz(-pi/2) q[102];
cx q[92],q[102];
rz(pi/2) q[102];
cx q[103],q[102];
cx q[102],q[103];
cx q[103],q[102];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
rz(-pi/16) q[101];
rz(pi/4) q[103];
cx q[103],q[102];
rz(pi/4) q[102];
cx q[103],q[102];
rz(-pi/4) q[102];
sx q[103];
rz(pi/2) q[103];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
rz(pi/8) q[92];
cx q[92],q[102];
rz(pi/8) q[102];
cx q[92],q[102];
rz(-pi/8) q[102];
cx q[101],q[102];
rz(pi/16) q[102];
cx q[101],q[102];
rz(-pi/16) q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[100],q[101];
rz(pi/32) q[101];
cx q[100],q[101];
rz(-pi/32) q[101];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
rz(-3*pi/16) q[101];
cx q[103],q[102];
cx q[102],q[103];
cx q[103],q[102];
rz(pi/8) q[103];
cx q[92],q[102];
rz(pi/4) q[102];
cx q[92],q[102];
rz(-pi/4) q[102];
cx q[103],q[102];
rz(pi/8) q[102];
cx q[103],q[102];
rz(-pi/8) q[102];
cx q[101],q[102];
rz(pi/16) q[102];
cx q[101],q[102];
rz(-pi/16) q[102];
sx q[92];
rz(pi/2) q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[103],q[102];
rz(pi/4) q[102];
cx q[103],q[102];
rz(-pi/4) q[102];
cx q[101],q[102];
rz(pi/8) q[102];
cx q[101],q[102];
rz(-pi/8) q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
rz(pi/4) q[102];
sx q[103];
rz(pi/2) q[103];
cx q[102],q[103];
rz(pi/4) q[103];
cx q[102],q[103];
sx q[102];
rz(pi/2) q[102];
rz(-pi/4) q[103];
barrier q[92],q[37],q[100],q[46],q[110],q[55],q[0],q[119],q[64],q[61],q[6],q[125],q[70],q[15],q[79],q[24],q[21],q[88],q[33],q[85],q[30],q[97],q[94],q[39],q[103],q[48],q[112],q[57],q[121],q[54],q[118],q[63],q[8],q[72],q[17],q[14],q[81],q[26],q[23],q[90],q[87],q[32],q[96],q[41],q[105],q[50],q[114],q[47],q[111],q[123],q[56],q[1],q[120],q[65],q[10],q[74],q[19],q[16],q[101],q[80],q[25],q[89],q[34],q[98],q[43],q[107],q[40],q[52],q[104],q[116],q[49],q[113],q[58],q[3],q[122],q[67],q[12],q[28],q[9],q[76],q[73],q[18],q[82],q[27],q[91],q[36],q[102],q[45],q[109],q[42],q[106],q[51],q[115],q[60],q[5],q[124],q[2],q[69],q[66],q[11],q[78],q[75],q[20],q[84],q[29],q[93],q[38],q[83],q[35],q[99],q[44],q[108],q[53],q[117],q[62],q[7],q[59],q[126],q[4],q[71],q[68],q[13],q[77],q[22],q[86],q[31],q[95];
measure q[100] -> c[0];
measure q[92] -> c[1];
measure q[101] -> c[2];
measure q[103] -> c[3];
measure q[102] -> c[4];
