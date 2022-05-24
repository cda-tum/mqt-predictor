OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[11];
sx q[53];
rz(2.3792517) q[53];
sx q[53];
rz(-pi) q[53];
sx q[60];
rz(1.2099402) q[60];
sx q[60];
sx q[61];
rz(-2.5714528) q[61];
sx q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
sx q[62];
rz(-0.045476454) q[62];
sx q[62];
sx q[63];
rz(-2.8865552) q[63];
sx q[63];
sx q[72];
rz(2.4172162) q[72];
sx q[72];
rz(-pi) q[72];
cx q[62],q[72];
cx q[62],q[61];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[61],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[60],q[53];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[72],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[62],q[61];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[60],q[53];
rz(pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[72],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[53],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
rz(-pi) q[81];
sx q[81];
rz(2.6903751) q[81];
sx q[81];
cx q[72],q[81];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
sx q[82];
rz(2.4415203) q[82];
sx q[82];
rz(-pi) q[82];
cx q[81],q[82];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[62],q[63];
cx q[62],q[61];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
sx q[83];
rz(0.99110017) q[83];
sx q[83];
rz(-pi) q[83];
cx q[82],q[83];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[72];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[72],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[61],q[62];
cx q[61],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
cx q[60],q[53];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[60],q[53];
rz(pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[62],q[63];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[61],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[53],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[62],q[63];
rz(-pi) q[84];
sx q[84];
rz(2.1009738) q[84];
sx q[84];
cx q[83],q[84];
rz(pi/2) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[82],q[83];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[81],q[82];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[62],q[63];
cx q[62],q[61];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
sx q[85];
rz(0.32588777) q[85];
sx q[85];
rz(-pi) q[85];
cx q[84],q[85];
sx q[84];
rz(1.8227579) q[84];
sx q[84];
rz(-pi) q[84];
rz(pi/2) q[85];
sx q[85];
rz(pi/2) q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
rz(pi/2) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[83],q[84];
sx q[83];
rz(3.0013855) q[83];
sx q[83];
rz(-pi) q[83];
rz(pi/2) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[82],q[83];
sx q[82];
rz(2.9861249) q[82];
sx q[82];
rz(-pi) q[82];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[81],q[82];
sx q[81];
rz(2.5915752) q[81];
sx q[81];
rz(-pi) q[81];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
rz(-pi) q[72];
sx q[72];
rz(0.31842289) q[72];
sx q[72];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
rz(-pi) q[62];
sx q[62];
rz(2.8639869) q[62];
sx q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[61],q[62];
cx q[61],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
cx q[60],q[53];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[53];
rz(pi/2) q[53];
sx q[53];
rz(-pi/2) q[53];
cx q[62],q[72];
sx q[62];
rz(1.6746415) q[62];
sx q[62];
rz(-pi) q[62];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[61],q[62];
rz(-pi) q[61];
sx q[61];
rz(0.97975289) q[61];
sx q[61];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
rz(-pi) q[60];
sx q[60];
rz(2.8897048) q[60];
sx q[60];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[53],q[60];
sx q[53];
rz(0.25221579) q[53];
sx q[53];
sx q[60];
rz(1.3649353) q[60];
sx q[60];
barrier q[121],q[66],q[11],q[75],q[20],q[69],q[53],q[29],q[26],q[93],q[124],q[90],q[35],q[99],q[44],q[108],q[81],q[117],q[85],q[126],q[59],q[4],q[123],q[68],q[13],q[77],q[22],q[19],q[86],q[62],q[28],q[95],q[92],q[37],q[101],q[46],q[110],q[55],q[119],q[52],q[116],q[82],q[6],q[125],q[70],q[15],q[12],q[79],q[24],q[76],q[21],q[88],q[60],q[30],q[94],q[39],q[103],q[48],q[112],q[45],q[109],q[54],q[118],q[61],q[8],q[84],q[17],q[14],q[63],q[78],q[23],q[87],q[32],q[96],q[41],q[105],q[38],q[50],q[102],q[114],q[47],q[111],q[56],q[1],q[120],q[65],q[10],q[7],q[74],q[71],q[16],q[80],q[25],q[89],q[34],q[98],q[31],q[43],q[107],q[40],q[104],q[49],q[113],q[58],q[3],q[122],q[0],q[67],q[64],q[9],q[73],q[18],q[72],q[27],q[91],q[36],q[100],q[33],q[97],q[42],q[106],q[51],q[115],q[83],q[5],q[57],q[2];
measure q[85] -> meas[0];
measure q[84] -> meas[1];
measure q[83] -> meas[2];
measure q[82] -> meas[3];
measure q[81] -> meas[4];
measure q[63] -> meas[5];
measure q[72] -> meas[6];
measure q[62] -> meas[7];
measure q[61] -> meas[8];
measure q[53] -> meas[9];
measure q[60] -> meas[10];
