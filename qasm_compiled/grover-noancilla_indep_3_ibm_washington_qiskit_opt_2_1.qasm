OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[3];
x q[91];
rz(pi/2) q[98];
sx q[98];
rz(3*pi/4) q[98];
cx q[98],q[91];
rz(-pi/4) q[91];
cx q[98],q[91];
rz(pi/4) q[91];
rz(pi/2) q[99];
sx q[99];
rz(pi/2) q[99];
cx q[99],q[98];
cx q[98],q[99];
rz(-pi/4) q[98];
cx q[98],q[91];
rz(pi/4) q[91];
cx q[98],q[91];
rz(-pi/4) q[91];
cx q[99],q[98];
rz(-pi/4) q[98];
cx q[98],q[91];
rz(-pi/4) q[91];
cx q[98],q[91];
rz(pi/4) q[91];
sx q[98];
rz(-pi) q[98];
rz(-pi) q[99];
cx q[98],q[99];
sx q[98];
rz(-pi/2) q[98];
rz(-pi) q[99];
barrier q[6],q[125],q[58],q[70],q[3],q[122],q[67],q[12],q[76],q[21],q[85],q[30],q[27],q[94],q[91],q[36],q[100],q[45],q[109],q[54],q[118],q[63],q[60],q[5],q[124],q[69],q[14],q[78],q[23],q[20],q[87],q[84],q[29],q[96],q[93],q[38],q[102],q[47],q[111],q[56],q[120],q[53],q[117],q[62],q[7],q[126],q[71],q[16],q[13],q[80],q[25],q[77],q[22],q[89],q[86],q[31],q[95],q[40],q[104],q[49],q[113],q[46],q[110],q[55],q[0],q[119],q[64],q[9],q[73],q[18],q[15],q[82],q[79],q[24],q[88],q[33],q[97],q[42],q[106],q[39],q[51],q[103],q[115],q[48],q[112],q[57],q[2],q[121],q[66],q[11],q[8],q[75],q[72],q[17],q[81],q[26],q[90],q[35],q[98],q[32],q[44],q[108],q[41],q[105],q[50],q[114],q[59],q[4],q[123],q[1],q[68],q[65],q[10],q[74],q[19],q[83],q[28],q[92],q[37],q[101],q[34],q[99],q[43],q[107],q[52],q[116],q[61];
measure q[98] -> meas[0];
measure q[99] -> meas[1];
measure q[91] -> meas[2];
