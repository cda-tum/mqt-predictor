OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[4];
rz(-pi) q[61];
sx q[61];
rz(2.9669986) q[61];
sx q[61];
rz(-pi) q[62];
sx q[62];
rz(3.1351352) q[62];
sx q[62];
rz(-pi) q[72];
sx q[72];
rz(2.6142168) q[72];
sx q[72];
rz(-pi) q[81];
sx q[81];
rz(2.3929282) q[81];
sx q[81];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[61];
rz(-pi) q[62];
sx q[62];
rz(2.7614805) q[62];
sx q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[62];
rz(-pi) q[72];
sx q[72];
rz(2.6857483) q[72];
sx q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[61],q[62];
cx q[72],q[81];
rz(-pi) q[72];
sx q[72];
rz(2.6569536) q[72];
sx q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
rz(-pi) q[81];
sx q[81];
rz(2.8271351) q[81];
sx q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
rz(-pi) q[62];
sx q[62];
rz(3.0585417) q[62];
sx q[62];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[61],q[62];
rz(-pi) q[61];
sx q[61];
rz(2.9075865) q[61];
sx q[61];
rz(-pi) q[62];
sx q[62];
rz(2.2268576) q[62];
sx q[62];
rz(-pi) q[81];
sx q[81];
rz(2.4633906) q[81];
sx q[81];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[61];
rz(-pi) q[62];
sx q[62];
rz(2.5437102) q[62];
sx q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[62];
rz(-pi) q[72];
sx q[72];
rz(2.6876093) q[72];
sx q[72];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[62],q[72];
rz(-pi) q[62];
sx q[62];
rz(2.9202491) q[62];
sx q[62];
rz(-pi) q[72];
sx q[72];
rz(2.2402546) q[72];
sx q[72];
barrier q[15],q[12],q[79],q[76],q[21],q[88],q[85],q[30],q[94],q[39],q[103],q[48],q[112],q[45],q[109],q[54],q[118],q[63],q[8],q[5],q[61],q[17],q[69],q[14],q[81],q[78],q[23],q[87],q[32],q[96],q[41],q[105],q[38],q[102],q[47],q[111],q[56],q[1],q[120],q[65],q[10],q[7],q[74],q[71],q[16],q[80],q[25],q[89],q[34],q[98],q[31],q[43],q[95],q[107],q[40],q[104],q[49],q[113],q[58],q[3],q[122],q[0],q[67],q[64],q[9],q[73],q[18],q[82],q[27],q[91],q[24],q[36],q[100],q[33],q[97],q[42],q[106],q[51],q[115],q[60],q[57],q[124],q[2],q[121],q[66],q[11],q[75],q[20],q[84],q[29],q[93],q[26],q[90],q[35],q[99],q[44],q[108],q[53],q[50],q[117],q[72],q[114],q[59],q[126],q[4],q[123],q[68],q[13],q[77],q[22],q[86],q[19],q[83],q[28],q[92],q[37],q[101],q[46],q[110],q[55],q[52],q[119],q[116],q[62],q[6],q[125],q[70];
measure q[61] -> meas[0];
measure q[81] -> meas[1];
measure q[62] -> meas[2];
measure q[72] -> meas[3];
