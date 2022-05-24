OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[5];
sx q[79];
rz(-0.27272224) q[79];
sx q[79];
sx q[91];
rz(1.4089345) q[91];
sx q[91];
sx q[97];
rz(1.8694596) q[97];
sx q[97];
rz(-pi) q[98];
sx q[98];
rz(1.3364007) q[98];
sx q[98];
cx q[98],q[97];
rz(pi/2) q[97];
sx q[97];
rz(pi/2) q[97];
cx q[98],q[91];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[91],q[79];
rz(pi/2) q[79];
sx q[79];
rz(pi/2) q[79];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
cx q[97],q[98];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
rz(-pi) q[99];
sx q[99];
rz(0.42771586) q[99];
sx q[99];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[99];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
cx q[91],q[98];
sx q[91];
rz(0.20379187) q[91];
sx q[91];
rz(-pi) q[91];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
cx q[97],q[98];
cx q[98],q[97];
cx q[97],q[98];
rz(pi/2) q[97];
sx q[97];
rz(pi/2) q[97];
cx q[98],q[91];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[98],q[97];
rz(pi/2) q[97];
sx q[97];
rz(pi/2) q[97];
sx q[98];
rz(1.4120965) q[98];
sx q[98];
rz(-pi) q[98];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[79],q[91];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
cx q[99],q[98];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
cx q[98],q[97];
cx q[97],q[98];
cx q[98],q[97];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
cx q[99],q[98];
cx q[97],q[98];
sx q[97];
rz(1.3113235) q[97];
sx q[97];
rz(-pi) q[97];
sx q[98];
rz(1.5030265) q[98];
sx q[98];
sx q[99];
rz(1.4548174) q[99];
sx q[99];
rz(-pi) q[99];
cx q[99],q[98];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[98],q[91];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[79],q[91];
cx q[98],q[91];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[91],q[79];
cx q[97],q[98];
cx q[98],q[97];
cx q[97],q[98];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
cx q[91],q[98];
cx q[97],q[98];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[98],q[91];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[79],q[91];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
rz(pi/2) q[99];
sx q[99];
rz(pi/2) q[99];
cx q[98],q[99];
sx q[98];
rz(1.3647544) q[98];
sx q[98];
rz(-pi) q[98];
rz(pi/2) q[99];
sx q[99];
rz(pi/2) q[99];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[99];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
cx q[97],q[98];
sx q[97];
rz(2.6704459) q[97];
sx q[97];
rz(-pi) q[97];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[79],q[91];
sx q[79];
rz(-2.6119158) q[79];
sx q[79];
cx q[98],q[91];
sx q[91];
rz(0.42989556) q[91];
sx q[91];
rz(-pi) q[98];
sx q[98];
rz(1.2230698) q[98];
sx q[98];
cx q[98],q[97];
cx q[97],q[98];
cx q[98],q[97];
rz(pi/2) q[97];
sx q[97];
rz(pi/2) q[97];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
cx q[99],q[98];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[99];
cx q[98],q[91];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[79];
rz(pi/2) q[79];
sx q[79];
rz(pi/2) q[79];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[98],q[97];
cx q[98],q[91];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
rz(pi/2) q[79];
sx q[79];
rz(pi/2) q[79];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
sx q[98];
rz(2.1938182) q[98];
sx q[98];
rz(-pi) q[98];
cx q[99],q[98];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[91];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[98],q[97];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[91],q[79];
rz(pi/2) q[79];
sx q[79];
rz(pi/2) q[79];
sx q[91];
rz(2.3787036) q[91];
sx q[91];
rz(-pi) q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[91],q[79];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[98],q[97];
rz(pi/2) q[97];
sx q[97];
rz(pi/2) q[97];
cx q[98],q[91];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
rz(-pi) q[98];
sx q[98];
rz(1.480182) q[98];
sx q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[98],q[91];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
cx q[97],q[98];
sx q[97];
rz(0.2540887) q[97];
sx q[97];
rz(-pi) q[97];
sx q[98];
rz(-0.24130116) q[98];
sx q[98];
barrier q[92],q[37],q[101],q[46],q[110],q[43],q[55],q[107],q[119],q[52],q[116],q[61],q[6],q[125],q[70],q[15],q[12],q[97],q[76],q[21],q[85],q[30],q[94],q[39],q[103],q[36],q[48],q[112],q[45],q[109],q[54],q[118],q[63],q[8],q[5],q[72],q[69],q[14],q[78],q[23],q[87],q[32],q[96],q[41],q[105],q[38],q[102],q[47],q[111],q[56],q[1],q[120],q[65],q[10],q[62],q[7],q[126],q[74],q[71],q[16],q[80],q[25],q[89],q[34],q[99],q[31],q[95],q[40],q[104],q[49],q[113],q[58],q[3],q[122],q[0],q[67],q[64],q[9],q[73],q[18],q[82],q[27],q[91],q[24],q[88],q[100],q[33],q[79],q[42],q[106],q[51],q[115],q[60],q[57],q[124],q[2],q[121],q[66],q[11],q[75],q[20],q[84],q[17],q[29],q[81],q[93],q[26],q[90],q[35],q[19],q[98],q[44],q[108],q[53],q[50],q[117],q[114],q[59],q[4],q[123],q[68],q[13],q[77],q[22],q[86],q[83],q[28];
measure q[99] -> meas[0];
measure q[79] -> meas[1];
measure q[91] -> meas[2];
measure q[97] -> meas[3];
measure q[98] -> meas[4];
