OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg meas[3];
rz(-pi/2) q[56];
rx(pi/2) q[56];
rz(0.070479509) q[56];
rz(-pi/2) q[62];
rx(pi/2) q[62];
rz(1.2543404) q[62];
rx(-0.045403725) q[63];
cz q[62],q[63];
rx(pi) q[62];
rx(1.1817921) q[63];
rz(pi) q[63];
cz q[62],q[63];
rz(1.2543404) q[62];
rx(1.6162001) q[63];
cz q[56],q[63];
rx(pi) q[56];
rx(1.18175) q[63];
cz q[56],q[63];
rx(-pi) q[56];
rz(0.42266863) q[56];
rx(0.38927715) q[63];
rz(-0.44719013) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[63];
cz q[56],q[63];
rx(pi) q[56];
rx(1.1817932) q[63];
cz q[56],q[63];
rx(1.1256788) q[56];
rz(1.6153171) q[56];
rx(1.6638219) q[56];
rx(0.38717915) q[63];
rz(-2.0179865) q[63];
cz q[62],q[63];
rx(0.81054681) q[63];
cz q[62],q[63];
rx(pi) q[62];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-pi/2) q[56];
rz(-pi/2) q[56];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-pi) q[56];
rx(-pi/2) q[63];
rz(pi/2) q[63];
rx(-1.6442799) q[63];
cz q[62],q[63];
rz(pi/2) q[62];
rx(pi) q[62];
rx(0.81051798) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(-2.2509718) q[62];
rz(2.1549617) q[62];
rx(0.59837281) q[62];
rx(-1.7311716) q[63];
cz q[56],q[63];
rz(pi/2) q[56];
rx(pi) q[56];
rx(0.81054759) q[63];
rz(pi/2) q[63];
cz q[56],q[63];
rz(-2.3825949) q[56];
rx(pi/2) q[56];
rz(-2.2764583) q[56];
rx(pi/2) q[63];
rz(0.52710484) q[63];
rx(2.2764583) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(-pi/2) q[63];
rx(pi/2) q[63];
cz q[62],q[63];
rx(-pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
cz q[62],q[63];
rx(-pi/2) q[63];
rz(-0.21786702) q[63];
cz q[63],q[56];
rx(1.1970209) q[56];
cz q[63],q[56];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rx(-pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(-pi/2) q[63];
rx(pi/2) q[63];
cz q[62],q[63];
rx(0.37359452) q[62];
rz(-1.1464804) q[62];
rx(pi/2) q[63];
cz q[62],q[63];
rx(1.8400301) q[62];
rz(-2.102935) q[62];
rx(pi/2) q[63];
rz(-pi/2) q[63];
rx(-1.3282748) q[63];
cz q[56],q[63];
rz(pi/2) q[56];
rx(pi) q[56];
rx(1.197016) q[63];
rz(1.0760671) q[63];
cz q[56],q[63];
rz(-1.9524266) q[56];
rx(-0.49810345) q[56];
rx(1.5103549) q[63];
rz(1.4592922) q[63];
barrier q[30],q[39],q[48],q[57],q[54],q[62],q[8],q[72],q[17],q[14],q[26],q[23],q[32],q[41],q[50],q[47],q[63],q[1],q[65],q[10],q[74],q[19],q[16],q[25],q[34],q[43],q[40],q[52],q[49],q[58],q[3],q[67],q[12],q[9],q[76],q[73],q[18],q[27],q[36],q[45],q[42],q[51],q[60],q[5],q[2],q[69],q[66],q[11],q[78],q[75],q[20],q[29],q[38],q[35],q[44],q[53],q[56],q[7],q[59],q[4],q[71],q[68],q[13],q[77],q[22],q[31],q[28],q[37],q[46],q[55],q[0],q[64],q[61],q[6],q[70],q[15],q[79],q[24],q[21],q[33];
measure q[63] -> meas[0];
measure q[56] -> meas[1];
measure q[62] -> meas[2];
