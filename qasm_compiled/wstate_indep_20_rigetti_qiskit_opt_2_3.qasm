OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg meas[20];
rx(-pi/2) q[2];
rz(1.3258177) q[2];
rx(pi/2) q[2];
rz(-pi/2) q[3];
rx(-pi/2) q[3];
rx(-pi/2) q[12];
rz(1.2897614) q[12];
rx(pi/2) q[12];
rx(-pi/2) q[13];
rz(1.3328552) q[13];
rx(pi/2) q[13];
rx(-pi/2) q[17];
rz(0.95531662) q[17];
rx(pi/2) q[17];
rx(-pi/2) q[18];
rz(1.1831996) q[18];
rx(pi/2) q[18];
rx(-pi/2) q[19];
rz(1.150262) q[19];
rx(pi/2) q[19];
rx(pi/2) q[20];
rz(pi/6) q[20];
rx(pi/2) q[20];
rx(-pi/2) q[28];
rz(1.2309594) q[28];
rx(pi/2) q[28];
rx(-pi/2) q[29];
rz(1.2094292) q[29];
rx(pi/2) q[29];
rx(-pi/2) q[30];
rz(pi/4) q[30];
rx(pi/2) q[30];
rx(-pi/2) q[40];
rz(1.3181161) q[40];
rx(pi/2) q[40];
rx(-pi/2) q[41];
rz(1.3096389) q[41];
rx(pi/2) q[41];
rx(-pi/2) q[48];
rz(1.3452829) q[48];
rx(pi/2) q[48];
rx(pi) q[49];
cz q[49],q[48];
rx(pi/2) q[48];
rz(1.3452829) q[48];
rx(-pi/2) q[48];
rz(-pi/2) q[49];
rx(-pi/2) q[49];
rx(-pi/2) q[54];
rz(1.3002466) q[54];
rx(pi/2) q[54];
rx(-pi/2) q[55];
rz(1.339319) q[55];
rx(pi/2) q[55];
cz q[48],q[55];
cz q[48],q[49];
rz(-pi/2) q[48];
rx(-pi/2) q[48];
rx(pi/2) q[55];
rz(2.9101153) q[55];
rx(pi/2) q[55];
cz q[12],q[55];
rz(-pi/2) q[12];
rx(-pi/2) q[12];
rz(-pi/2) q[55];
rx(-pi/2) q[55];
cz q[55],q[12];
rx(pi/2) q[12];
rz(pi) q[12];
rx(pi/2) q[55];
rz(pi) q[55];
cz q[12],q[55];
cz q[12],q[13];
rx(pi/2) q[13];
rz(1.3328552) q[13];
rx(-pi/2) q[13];
cz q[13],q[2];
rx(pi/2) q[2];
rz(1.3258177) q[2];
rx(-pi/2) q[2];
cz q[2],q[3];
rz(-pi/2) q[2];
rx(-pi/2) q[2];
rx(pi/2) q[3];
rz(pi) q[3];
cz q[3],q[2];
rx(pi/2) q[2];
rz(pi) q[2];
rx(pi/2) q[3];
rz(pi) q[3];
cz q[2],q[3];
rx(pi/2) q[3];
rz(pi) q[3];
cz q[3],q[40];
rx(pi/2) q[40];
rz(1.3181161) q[40];
rx(-pi/2) q[40];
cz q[40],q[41];
rx(pi/2) q[41];
rz(1.3096389) q[41];
rx(-pi/2) q[41];
cz q[41],q[54];
rx(pi/2) q[54];
rz(2.8710429) q[54];
rx(pi/2) q[54];
rx(pi/2) q[55];
rz(pi) q[55];
cz q[55],q[48];
rx(pi/2) q[48];
rz(pi) q[48];
rx(pi/2) q[55];
rz(pi) q[55];
cz q[48],q[55];
rx(pi/2) q[48];
rz(pi) q[48];
rx(pi/2) q[55];
rz(pi) q[55];
cz q[55],q[48];
rx(pi/2) q[48];
rz(pi/2) q[48];
rx(pi/2) q[55];
rz(pi) q[55];
cz q[12],q[55];
rx(pi/2) q[12];
rz(pi) q[12];
cz q[13],q[12];
rx(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/2) q[13];
rx(-pi/2) q[13];
cz q[2],q[13];
rx(pi/2) q[13];
rz(pi) q[13];
rx(pi/2) q[2];
rz(pi) q[2];
cz q[13],q[2];
rx(pi/2) q[13];
rz(pi) q[13];
rx(pi/2) q[2];
rz(pi) q[2];
cz q[2],q[13];
rx(pi/2) q[13];
rz(pi/2) q[13];
rx(pi/2) q[2];
rz(pi) q[2];
cz q[3],q[2];
rx(pi/2) q[2];
rz(pi/2) q[2];
rx(pi/2) q[3];
rz(pi) q[3];
cz q[40],q[3];
rx(pi/2) q[3];
rz(pi/2) q[3];
rz(-pi/2) q[40];
rx(-pi/2) q[40];
cz q[41],q[40];
rx(pi/2) q[40];
rz(pi/2) q[40];
rx(pi/2) q[55];
rz(pi) q[55];
cz q[55],q[54];
rz(-pi/2) q[54];
rx(-pi/2) q[54];
rx(pi/2) q[55];
rz(pi) q[55];
cz q[54],q[55];
rx(pi/2) q[54];
rz(pi) q[54];
rx(pi/2) q[55];
rz(pi) q[55];
cz q[55],q[54];
cz q[41],q[54];
rz(-pi/2) q[41];
rx(-pi/2) q[41];
rx(pi/2) q[54];
rz(pi) q[54];
cz q[54],q[41];
rx(pi/2) q[41];
rz(pi/2) q[41];
rx(pi/2) q[54];
rz(pi) q[54];
cz q[41],q[54];
cz q[55],q[48];
rx(pi/2) q[48];
rz(1.2897614) q[48];
rx(-pi/2) q[48];
cz q[48],q[49];
rz(-pi/2) q[48];
rx(-pi/2) q[48];
rx(pi/2) q[49];
rz(pi) q[49];
cz q[49],q[48];
rx(pi/2) q[48];
rz(pi) q[48];
rx(pi/2) q[49];
rz(pi) q[49];
cz q[48],q[49];
rx(pi/2) q[49];
rz(pi) q[49];
cz q[55],q[54];
rx(pi/2) q[54];
rz(pi/2) q[54];
rx(pi/2) q[55];
rz(pi) q[55];
cz q[48],q[55];
rx(pi/2) q[48];
rz(pi) q[48];
rx(pi/2) q[55];
rz(pi) q[55];
cz q[55],q[48];
rx(pi/2) q[48];
rz(pi) q[48];
rx(pi/2) q[55];
rz(pi) q[55];
cz q[48],q[55];
rx(pi/2) q[48];
rz(pi) q[48];
rx(pi/2) q[55];
rz(pi/2) q[55];
rx(-pi/2) q[57];
rz(1.1071487) q[57];
rx(pi/2) q[57];
rx(-pi/2) q[62];
rz(1.2779536) q[62];
rx(pi/2) q[62];
cz q[49],q[62];
cz q[49],q[48];
rx(pi/2) q[48];
rz(pi/2) q[48];
rx(pi/2) q[49];
rz(pi) q[49];
rx(pi/2) q[62];
rz(1.2779536) q[62];
rx(-pi/2) q[62];
rx(-pi/2) q[63];
rz(1.264519) q[63];
rx(pi/2) q[63];
cz q[62],q[63];
cz q[62],q[49];
rx(pi/2) q[49];
rz(pi/2) q[49];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
rx(pi/2) q[63];
rz(2.8353153) q[63];
rx(pi/2) q[63];
cz q[56],q[63];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
rx(-pi/2) q[70];
rz(1.2490458) q[70];
rx(pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[70];
rz(1.2490458) q[70];
rx(-pi/2) q[70];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[70],q[71];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(1.2309594) q[28];
rx(-pi/2) q[28];
cz q[28],q[29];
rx(pi/2) q[29];
rz(1.2094292) q[29];
rx(-pi/2) q[29];
cz q[29],q[18];
rx(pi/2) q[18];
rz(1.1831996) q[18];
rx(-pi/2) q[18];
cz q[18],q[19];
rx(pi/2) q[19];
rz(1.150262) q[19];
rx(-pi/2) q[19];
cz q[19],q[56];
cz q[19],q[20];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
rz(-pi/2) q[20];
rx(-pi/2) q[20];
cz q[20],q[19];
rx(pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[19],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[56];
rz(1.1071487) q[56];
rx(-pi/2) q[56];
cz q[56],q[19];
rx(pi/2) q[19];
rz(5*pi/6) q[19];
rx(pi/2) q[19];
cz q[18],q[19];
rz(-pi/2) q[18];
rx(-pi/2) q[18];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[18],q[19];
cz q[18],q[17];
rx(pi/2) q[17];
rz(0.95531662) q[17];
rx(-pi/2) q[17];
cz q[17],q[30];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[30];
rz(pi/4) q[30];
rx(-pi/2) q[30];
cz q[56],q[63];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[57],q[56];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[63];
rz(pi) q[63];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[28],q[71];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
cz q[29],q[28];
rx(pi/2) q[28];
rz(pi/2) q[28];
rz(-pi/2) q[29];
rx(-pi/2) q[29];
cz q[18],q[29];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[18],q[19];
cz q[20],q[19];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[29];
rz(pi/2) q[29];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi/2) q[20];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[63],q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[56],q[19];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi) q[18];
cz q[17],q[18];
rz(-pi/2) q[17];
rx(-pi/2) q[17];
rx(pi/2) q[18];
rz(pi/2) q[18];
rx(pi/2) q[19];
rz(pi/2) q[19];
cz q[30],q[17];
rx(pi/2) q[17];
rz(pi/2) q[17];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[71];
rz(pi/2) q[71];
barrier q[25],q[34],q[43],q[52],q[61],q[58],q[13],q[67],q[49],q[76],q[21],q[56],q[30],q[27],q[36],q[45],q[48],q[51],q[60],q[5],q[69],q[14],q[78],q[23],q[18],q[29],q[38],q[47],q[44],q[62],q[53],q[63],q[7],q[57],q[16],q[2],q[77],q[22],q[31],q[40],q[37],q[55],q[46],q[12],q[0],q[64],q[9],q[6],q[73],q[71],q[15],q[79],q[24],q[33],q[42],q[39],q[41],q[19],q[3],q[66],q[11],q[70],q[8],q[75],q[72],q[17],q[26],q[35],q[32],q[54],q[50],q[59],q[4],q[1],q[68],q[65],q[10],q[74],q[20],q[28];
measure q[30] -> meas[0];
measure q[17] -> meas[1];
measure q[18] -> meas[2];
measure q[19] -> meas[3];
measure q[20] -> meas[4];
measure q[56] -> meas[5];
measure q[29] -> meas[6];
measure q[28] -> meas[7];
measure q[71] -> meas[8];
measure q[70] -> meas[9];
measure q[63] -> meas[10];
measure q[49] -> meas[11];
measure q[48] -> meas[12];
measure q[54] -> meas[13];
measure q[40] -> meas[14];
measure q[3] -> meas[15];
measure q[2] -> meas[16];
measure q[12] -> meas[17];
measure q[41] -> meas[18];
measure q[55] -> meas[19];
