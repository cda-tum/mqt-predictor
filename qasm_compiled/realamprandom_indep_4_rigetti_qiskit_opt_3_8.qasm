OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg meas[4];
rx(pi/2) q[18];
rz(2.1671961) q[18];
rx(pi/2) q[18];
rx(pi/2) q[19];
rz(0.44010271) q[19];
rx(-pi/2) q[19];
cz q[19],q[18];
rz(-pi/2) q[18];
rx(-pi/2) q[18];
rx(-pi/2) q[29];
rz(1.0651731) q[29];
rx(pi/2) q[56];
rz(1.8507908) q[56];
rx(pi/2) q[56];
cz q[19],q[56];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[19],q[18];
rx(pi/2) q[18];
rz(2.4665469) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
cz q[18],q[29];
rx(-pi/2) q[18];
rz(-pi/2) q[18];
rx(pi/2) q[19];
rz(4.2911081) q[19];
cz q[19],q[56];
rx(-pi/2) q[19];
rx(pi/2) q[29];
rz(3.4310715) q[29];
cz q[18],q[29];
rx(0.57204847) q[18];
rz(-pi/2) q[18];
cz q[18],q[19];
rx(-pi/2) q[18];
rz(-1.2853365) q[18];
rx(pi/2) q[19];
rz(pi/2) q[19];
cz q[18],q[19];
rx(1.9768739) q[18];
rz(-4.5969789) q[18];
rx(-1.1175472) q[19];
rz(-pi/2) q[19];
rx(1.9133327) q[29];
rz(3.8670493) q[29];
cz q[29],q[18];
rx(pi/2) q[29];
rz(pi) q[29];
rx(2.9053683) q[56];
rz(pi/2) q[56];
cz q[19],q[56];
rx(-pi/2) q[19];
rz(-pi/2) q[19];
rx(pi/2) q[56];
rz(pi/2) q[56];
cz q[19],q[56];
rx(-2.9053683) q[19];
rz(pi/2) q[19];
rx(-1.2127889) q[19];
cz q[18],q[19];
rx(-pi/2) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[18],q[19];
cz q[18],q[29];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi/2) q[29];
cz q[29],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(0.54549875) q[56];
rz(-0.12142937) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rx(pi/2) q[56];
cz q[19],q[56];
rx(-pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi/2) q[19];
cz q[18],q[19];
rx(1.0999872) q[18];
rz(pi/2) q[56];
cz q[56],q[19];
rx(pi/2) q[19];
rz(pi/2) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rx(pi/2) q[19];
cz q[18],q[19];
rx(-pi/2) q[18];
rz(pi/2) q[18];
rx(pi/2) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi) q[18];
rz(-pi/2) q[19];
cz q[29],q[18];
rx(1.7257289) q[18];
rx(pi/2) q[29];
rz(2.5657202) q[29];
rx(pi/2) q[29];
rx(pi/2) q[56];
rz(2.4345037) q[56];
rx(pi/2) q[56];
cz q[19],q[56];
rx(-pi/2) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(-pi/2) q[19];
rx(pi/2) q[19];
cz q[18],q[19];
rx(-pi/2) q[18];
rz(pi/2) q[18];
rx(pi/2) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi) q[18];
cz q[18],q[29];
rz(pi/2) q[19];
cz q[18],q[19];
rx(1.5221463) q[18];
rx(-pi/2) q[19];
rx(pi/2) q[29];
rz(pi/2) q[29];
cz q[18],q[29];
rx(pi/2) q[18];
rx(pi/2) q[29];
cz q[18],q[29];
rx(-pi/2) q[18];
rz(3*pi/2) q[18];
rx(pi/2) q[29];
cz q[18],q[29];
cz q[18],q[19];
rx(pi/2) q[18];
rx(pi/2) q[19];
cz q[18],q[19];
rx(-pi/2) q[18];
rz(3*pi/2) q[18];
rx(pi/2) q[19];
cz q[18],q[19];
rz(pi/2) q[19];
rz(pi/2) q[29];
rz(pi/2) q[56];
rx(pi/2) q[56];
rz(pi/2) q[56];
cz q[56],q[19];
rx(pi/2) q[19];
rz(pi/2) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rx(pi/2) q[19];
cz q[18],q[19];
rx(-pi/2) q[18];
rx(pi/2) q[19];
cz q[18],q[19];
rx(-pi/2) q[19];
rz(-pi/2) q[19];
cz q[56],q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(0.49195731) q[18];
rx(-pi/2) q[18];
rx(pi/2) q[19];
rz(1.5376277) q[19];
rx(pi/2) q[19];
rx(pi/2) q[56];
rz(0.21601699) q[56];
rx(-pi/2) q[56];
barrier q[33],q[42],q[51],q[60],q[57],q[2],q[69],q[66],q[11],q[75],q[20],q[19],q[26],q[35],q[44],q[53],q[50],q[62],q[59],q[4],q[68],q[13],q[77],q[22],q[29],q[28],q[37],q[46],q[55],q[52],q[61],q[6],q[70],q[15],q[79],q[12],q[24],q[76],q[21],q[30],q[39],q[48],q[45],q[54],q[63],q[8],q[72],q[5],q[17],q[14],q[78],q[23],q[32],q[41],q[38],q[47],q[18],q[1],q[65],q[10],q[74],q[7],q[71],q[16],q[25],q[34],q[31],q[43],q[40],q[49],q[58],q[3],q[67],q[0],q[64],q[9],q[73],q[56],q[27],q[36];
measure q[29] -> meas[0];
measure q[56] -> meas[1];
measure q[18] -> meas[2];
measure q[19] -> meas[3];
