OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[8];
sx q[44];
rz(2.846624) q[44];
sx q[44];
rz(-pi) q[44];
sx q[45];
rz(2.4885139) q[45];
sx q[45];
rz(-pi) q[45];
sx q[54];
rz(1.6534708) q[54];
sx q[54];
sx q[63];
rz(1.5101881) q[63];
sx q[63];
rz(-pi) q[63];
sx q[64];
rz(1.4569349) q[64];
sx q[64];
rz(-pi) q[64];
cx q[54],q[64];
cx q[54],q[45];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[54],q[45];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
sx q[65];
rz(1.1979301) q[65];
sx q[65];
rz(-pi) q[65];
cx q[64],q[65];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[45],q[44];
cx q[64],q[65];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[45],q[44];
cx q[64],q[65];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[44];
cx q[64],q[63];
rz(pi/2) q[65];
sx q[65];
rz(pi/2) q[65];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[45],q[44];
rz(pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
cx q[64],q[63];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
rz(pi/2) q[65];
sx q[65];
rz(pi/2) q[65];
sx q[66];
rz(-2.901823) q[66];
sx q[66];
rz(-pi/2) q[66];
cx q[65],q[66];
sx q[65];
rz(-pi/2) q[65];
sx q[65];
rz(pi/2) q[66];
cx q[65],q[66];
rz(-pi) q[65];
sx q[65];
rz(-pi/2) q[65];
cx q[64],q[65];
sx q[64];
rz(-pi/2) q[64];
sx q[64];
rz(pi/2) q[65];
cx q[64],q[65];
rz(pi/2) q[64];
sx q[64];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[64],q[63];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[64],q[54];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
sx q[65];
rz(-3.1262231) q[65];
sx q[65];
rz(-pi/2) q[65];
sx q[66];
rz(0.46364761) q[66];
sx q[66];
rz(-pi/2) q[66];
sx q[73];
rz(-2.1982512) q[73];
sx q[73];
rz(pi/2) q[73];
cx q[66],q[73];
sx q[66];
rz(-pi/2) q[66];
sx q[66];
rz(pi/2) q[73];
cx q[66],q[73];
sx q[66];
rz(pi/2) q[66];
cx q[65],q[66];
sx q[65];
rz(-pi/2) q[65];
sx q[65];
rz(pi/2) q[66];
cx q[65],q[66];
rz(-pi) q[65];
sx q[65];
rz(-pi) q[65];
cx q[64],q[65];
sx q[64];
rz(1.4754079) q[64];
sx q[64];
rz(-pi) q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
cx q[54],q[45];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[44],q[45];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[44],q[45];
cx q[54],q[45];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(-1.3565815) q[64];
sx q[64];
rz(-pi/2) q[64];
rz(pi/2) q[65];
cx q[64],q[65];
sx q[64];
rz(-pi/2) q[64];
sx q[64];
rz(pi/2) q[65];
cx q[64],q[65];
rz(pi/2) q[64];
sx q[64];
rz(-pi) q[64];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
rz(-pi) q[45];
sx q[45];
rz(2.7883648) q[45];
sx q[45];
cx q[64],q[54];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[44],q[45];
sx q[44];
rz(2.2029835) q[44];
sx q[44];
rz(-pi) q[44];
sx q[45];
rz(1.9697545) q[45];
sx q[45];
rz(-pi) q[64];
sx q[64];
rz(2.946853) q[64];
sx q[64];
rz(1.1741067) q[65];
sx q[65];
rz(-1.6546544) q[65];
sx q[65];
rz(1.3734454) q[65];
rz(-1.6460002) q[66];
sx q[66];
rz(-2.9397966) q[66];
sx q[66];
rz(0.076755435) q[66];
rz(-0.59705777) q[73];
sx q[73];
rz(-0.54379483) q[73];
sx q[73];
rz(-2.0976977) q[73];
barrier q[121],q[44],q[11],q[75],q[20],q[69],q[84],q[29],q[26],q[93],q[124],q[90],q[35],q[99],q[54],q[108],q[53],q[117],q[62],q[126],q[59],q[4],q[123],q[68],q[13],q[77],q[22],q[19],q[86],q[83],q[28],q[95],q[92],q[37],q[101],q[46],q[110],q[55],q[119],q[52],q[116],q[61],q[6],q[125],q[70],q[15],q[12],q[79],q[24],q[76],q[21],q[88],q[85],q[30],q[94],q[39],q[103],q[48],q[112],q[63],q[109],q[73],q[118],q[64],q[8],q[72],q[17],q[14],q[81],q[78],q[23],q[87],q[32],q[96],q[41],q[105],q[38],q[50],q[102],q[114],q[47],q[111],q[56],q[1],q[120],q[65],q[10],q[7],q[74],q[71],q[16],q[80],q[25],q[89],q[34],q[98],q[31],q[43],q[107],q[40],q[104],q[49],q[113],q[58],q[3],q[122],q[0],q[67],q[66],q[9],q[45],q[18],q[82],q[27],q[91],q[36],q[100],q[33],q[97],q[42],q[106],q[51],q[115],q[60],q[5],q[57],q[2];
measure q[73] -> meas[0];
measure q[66] -> meas[1];
measure q[63] -> meas[2];
measure q[65] -> meas[3];
measure q[54] -> meas[4];
measure q[64] -> meas[5];
measure q[44] -> meas[6];
measure q[45] -> meas[7];
