OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[12];
rz(pi/2) q[41];
sx q[41];
rz(pi/2) q[41];
rz(pi/2) q[42];
sx q[42];
rz(pi/2) q[42];
cx q[41],q[42];
rz(-3.61246870101387) q[42];
cx q[41],q[42];
rz(pi/2) q[43];
sx q[43];
rz(pi/2) q[43];
cx q[43],q[42];
rz(-3.61246870101387) q[42];
cx q[43],q[42];
rz(pi/2) q[42];
sx q[42];
rz(7.88666297899717) q[42];
sx q[42];
rz(5*pi/2) q[42];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[45],q[44];
rz(-3.61246870101387) q[44];
cx q[45],q[44];
rz(pi/2) q[44];
sx q[44];
rz(7.88666297899717) q[44];
sx q[44];
rz(5*pi/2) q[44];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[62],q[63];
rz(-3.61246870101387) q[63];
cx q[62],q[63];
rz(pi/2) q[64];
sx q[64];
rz(pi) q[64];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[82],q[81];
rz(-3.61246870101387) q[81];
cx q[82],q[81];
cx q[72],q[81];
rz(-3.61246870101387) q[81];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
rz(-3.61246870101387) q[62];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[60],q[53];
rz(-3.61246870101387) q[53];
cx q[60],q[53];
rz(pi/2) q[53];
sx q[53];
rz(7.88666297899717) q[53];
sx q[53];
rz(5*pi/2) q[53];
cx q[53],q[41];
rz(-1.80127652144224) q[41];
cx q[53],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[44],q[43];
rz(-1.80127652144224) q[43];
cx q[44],q[43];
rz(pi/2) q[43];
sx q[43];
rz(9.45398646550747) q[43];
sx q[43];
rz(5*pi/2) q[43];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(-pi/2) q[60];
sx q[60];
rz(-1.6034777) q[60];
sx q[60];
rz(pi/2) q[62];
sx q[62];
rz(7.88666297899717) q[62];
sx q[62];
rz(5*pi/2) q[62];
cx q[64],q[63];
rz(-3.61246870101387) q[63];
cx q[64],q[63];
rz(pi/2) q[63];
sx q[63];
rz(7.88666297899717) q[63];
sx q[63];
rz(5*pi/2) q[63];
cx q[64],q[54];
rz(-3.61246870101387) q[54];
cx q[64],q[54];
rz(pi/2) q[54];
sx q[54];
rz(7.88666297899717) q[54];
sx q[54];
rz(5*pi/2) q[54];
cx q[54],q[45];
rz(-1.80127652144224) q[45];
cx q[54],q[45];
rz(pi/2) q[45];
sx q[45];
rz(9.45398646550747) q[45];
sx q[45];
rz(5*pi/2) q[45];
sx q[64];
rz(7.88666297899717) q[64];
sx q[64];
rz(3*pi) q[64];
rz(pi/2) q[81];
sx q[81];
rz(7.88666297899717) q[81];
sx q[81];
rz(5*pi/2) q[81];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[83],q[82];
rz(-3.61246870101387) q[82];
cx q[83],q[82];
rz(pi/2) q[82];
sx q[82];
rz(7.88666297899717) q[82];
sx q[82];
rz(5*pi/2) q[82];
cx q[82],q[81];
rz(-1.80127652144224) q[81];
cx q[82],q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
rz(-1.80127652144224) q[62];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[62];
rz(9.45398646550747) q[62];
sx q[62];
rz(5*pi/2) q[62];
cx q[64],q[63];
rz(-1.80127652144224) q[63];
cx q[64],q[63];
rz(pi/2) q[63];
sx q[63];
rz(9.45398646550747) q[63];
sx q[63];
rz(5*pi/2) q[63];
cx q[64],q[54];
rz(-1.80127652144224) q[54];
cx q[64],q[54];
rz(pi/2) q[54];
sx q[54];
rz(9.45398646550747) q[54];
sx q[54];
rz(5*pi/2) q[54];
sx q[64];
rz(9.45398646550747) q[64];
sx q[64];
rz(5*pi/2) q[64];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
rz(-3.61246870101387) q[82];
cx q[81],q[82];
rz(-pi/2) q[81];
sx q[81];
rz(-1.6034777) q[81];
sx q[81];
cx q[81],q[72];
rz(-1.80127652144224) q[72];
cx q[81],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
rz(-1.80127652144224) q[61];
cx q[60],q[61];
cx q[60],q[53];
rz(-1.80127652144224) q[53];
cx q[60],q[53];
rz(pi/2) q[53];
sx q[53];
rz(9.45398646550747) q[53];
sx q[53];
rz(5*pi/2) q[53];
sx q[60];
rz(9.45398646550747) q[60];
sx q[60];
rz(5*pi/2) q[60];
rz(pi/2) q[61];
sx q[61];
rz(9.45398646550747) q[61];
sx q[61];
rz(5*pi/2) q[61];
rz(pi/2) q[82];
sx q[82];
rz(7.88666297899717) q[82];
sx q[82];
rz(5*pi/2) q[82];
cx q[82],q[83];
rz(-1.80127652144224) q[83];
cx q[82],q[83];
cx q[81],q[82];
rz(-1.80127652144224) q[82];
cx q[81],q[82];
sx q[81];
rz(9.45398646550747) q[81];
sx q[81];
rz(5*pi/2) q[81];
rz(pi/2) q[82];
sx q[82];
rz(9.45398646550747) q[82];
sx q[82];
rz(5*pi/2) q[82];
rz(pi/2) q[83];
sx q[83];
rz(9.45398646550747) q[83];
sx q[83];
rz(5*pi/2) q[83];
barrier q[5],q[63],q[69],q[14],q[78],q[23],q[87],q[32],q[96],q[53],q[105],q[38],q[102],q[47],q[111],q[56],q[1],q[120],q[65],q[81],q[7],q[126],q[74],q[71],q[16],q[80],q[25],q[89],q[34],q[98],q[31],q[95],q[40],q[104],q[49],q[113],q[58],q[3],q[55],q[122],q[0],q[67],q[64],q[9],q[73],q[18],q[83],q[27],q[91],q[24],q[88],q[33],q[97],q[43],q[106],q[51],q[115],q[62],q[57],q[124],q[2],q[121],q[66],q[11],q[75],q[20],q[84],q[17],q[29],q[72],q[93],q[26],q[90],q[35],q[99],q[42],q[108],q[41],q[50],q[117],q[114],q[59],q[4],q[123],q[68],q[13],q[77],q[10],q[22],q[86],q[19],q[82],q[28],q[92],q[37],q[101],q[46],q[45],q[110],q[107],q[52],q[119],q[116],q[60],q[6],q[125],q[70],q[15],q[79],q[12],q[76],q[21],q[85],q[30],q[94],q[39],q[36],q[103],q[48],q[100],q[54],q[112],q[109],q[44],q[118],q[61],q[8];
measure q[60] -> meas[0];
measure q[64] -> meas[1];
measure q[54] -> meas[2];
measure q[81] -> meas[3];
measure q[82] -> meas[4];
measure q[63] -> meas[5];
measure q[83] -> meas[6];
measure q[72] -> meas[7];
measure q[45] -> meas[8];
measure q[53] -> meas[9];
measure q[61] -> meas[10];
measure q[43] -> meas[11];
