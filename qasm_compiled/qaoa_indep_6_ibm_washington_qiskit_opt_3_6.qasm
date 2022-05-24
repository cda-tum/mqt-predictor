OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[6];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[81],q[82];
rz(-1.75673946673406) q[82];
cx q[81],q[82];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[79],q[80];
cx q[80],q[79];
cx q[79],q[80];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
rz(pi/2) q[99];
sx q[99];
rz(pi/2) q[99];
rz(pi/2) q[100];
sx q[100];
rz(pi/2) q[100];
cx q[99],q[100];
rz(-1.75673946673406) q[100];
cx q[99],q[100];
cx q[98],q[99];
rz(-1.75673946673406) q[99];
cx q[98],q[99];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[91],q[79];
rz(-1.75673946673406) q[79];
cx q[91],q[79];
rz(pi/2) q[79];
sx q[79];
rz(12.0455905594774) q[79];
sx q[79];
rz(5*pi/2) q[79];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
rz(pi/2) q[91];
sx q[91];
rz(12.0455905594774) q[91];
sx q[91];
rz(5*pi/2) q[91];
rz(pi/2) q[99];
sx q[99];
rz(12.0455905594774) q[99];
sx q[99];
rz(5*pi/2) q[99];
rz(pi/2) q[101];
sx q[101];
rz(pi/2) q[101];
cx q[101],q[100];
rz(-1.75673946673406) q[100];
cx q[101],q[100];
rz(pi/2) q[100];
sx q[100];
rz(12.0455905594774) q[100];
sx q[100];
rz(5*pi/2) q[100];
cx q[99],q[100];
rz(5.08712887134137) q[100];
cx q[99],q[100];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[99];
cx q[91],q[98];
rz(5.08712887134137) q[98];
cx q[91],q[98];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
rz(pi/2) q[98];
sx q[98];
rz(14.8674278535498) q[98];
sx q[98];
rz(5*pi/2) q[98];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[92],q[83];
rz(-1.75673946673406) q[83];
cx q[92],q[83];
rz(pi/2) q[83];
sx q[83];
rz(12.0455905594774) q[83];
sx q[83];
rz(5*pi/2) q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[80],q[81];
rz(5.08712887134137) q[81];
cx q[80],q[81];
cx q[79],q[80];
rz(5.08712887134137) q[80];
cx q[79],q[80];
rz(pi/2) q[79];
sx q[79];
rz(14.8674278535498) q[79];
sx q[79];
rz(5*pi/2) q[79];
rz(pi/2) q[80];
sx q[80];
rz(14.8674278535498) q[80];
sx q[80];
rz(5*pi/2) q[80];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
rz(pi/2) q[92];
sx q[92];
rz(12.0455905594774) q[92];
sx q[92];
rz(5*pi/2) q[92];
cx q[92],q[102];
rz(5.08712887134137) q[102];
cx q[92],q[102];
rz(pi/2) q[102];
sx q[102];
rz(14.8674278535498) q[102];
sx q[102];
rz(5*pi/2) q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[83],q[82];
rz(5.08712887134137) q[82];
cx q[83],q[82];
rz(pi/2) q[82];
sx q[82];
rz(14.8674278535498) q[82];
sx q[82];
rz(5*pi/2) q[82];
rz(pi/2) q[83];
sx q[83];
rz(14.8674278535498) q[83];
sx q[83];
rz(5*pi/2) q[83];
barrier q[5],q[72],q[69],q[14],q[78],q[23],q[87],q[32],q[96],q[41],q[105],q[38],q[100],q[47],q[111],q[56],q[1],q[120],q[65],q[62],q[7],q[126],q[74],q[71],q[16],q[81],q[25],q[89],q[34],q[79],q[31],q[95],q[40],q[104],q[49],q[113],q[58],q[3],q[55],q[122],q[0],q[67],q[64],q[9],q[73],q[18],q[82],q[27],q[99],q[24],q[88],q[33],q[97],q[42],q[106],q[51],q[115],q[60],q[57],q[124],q[2],q[121],q[66],q[11],q[75],q[20],q[84],q[17],q[29],q[80],q[93],q[26],q[90],q[35],q[98],q[44],q[108],q[53],q[50],q[117],q[114],q[59],q[4],q[123],q[68],q[13],q[77],q[10],q[22],q[86],q[19],q[92],q[28],q[101],q[37],q[83],q[46],q[43],q[110],q[107],q[52],q[119],q[116],q[61],q[6],q[125],q[70],q[15],q[91],q[12],q[76],q[21],q[85],q[30],q[94],q[39],q[36],q[103],q[48],q[102],q[45],q[112],q[109],q[54],q[118],q[63],q[8];
measure q[79] -> meas[0];
measure q[83] -> meas[1];
measure q[80] -> meas[2];
measure q[98] -> meas[3];
measure q[82] -> meas[4];
measure q[102] -> meas[5];
