OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg meas[7];
rx(-pi/2) q[20];
rz(1.3855018) q[20];
rx(-pi/2) q[20];
rx(pi/2) q[21];
rz(1.2238741) q[21];
rx(pi/2) q[21];
rx(pi/2) q[56];
rz(0.88910775) q[56];
rx(pi/2) q[56];
rx(pi/2) q[57];
rz(-2.5912191) q[57];
rx(-pi/2) q[62];
rz(0.75510345) q[62];
rx(-pi/2) q[62];
rx(-pi/2) q[63];
rz(0.68289205) q[63];
rx(-pi/2) q[63];
cz q[20],q[63];
cz q[20],q[21];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[20],q[63];
rz(-pi/2) q[20];
rx(-pi/2) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(-0.8059157) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
cz q[20],q[21];
rx(pi/2) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
cz q[63],q[62];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
rz(3.9377301) q[63];
cz q[63],q[56];
rx(-1.357138) q[56];
rz(pi/2) q[56];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(-pi/2) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[56],q[57];
rx(-1.1174915) q[56];
rz(-pi/2) q[56];
rx(1.7844546) q[57];
rz(pi/2) q[57];
rx(-pi/2) q[63];
cz q[20],q[63];
rx(-pi/2) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(-0.7961374) q[20];
cz q[21],q[20];
rx(pi/2) q[63];
rz(-pi/2) q[63];
cz q[63],q[62];
rz(-pi/2) q[62];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-pi/2) q[56];
rz(-pi/2) q[56];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[56],q[63];
rx(1.7333896) q[56];
rz(-pi/2) q[56];
rx(-2.0241011) q[63];
rz(-pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rx(pi/2) q[63];
cz q[62],q[63];
rx(-pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
cz q[62],q[63];
rz(pi) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
cz q[21],q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[21],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rz(-pi/2) q[21];
rx(-pi/2) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[21];
rz(pi/2) q[21];
cz q[21],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
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
cz q[20],q[63];
rx(pi/2) q[62];
rz(pi/2) q[62];
cz q[62],q[63];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[20],q[63];
cz q[21],q[20];
rx(-3.0727529) q[63];
rz(pi/2) q[63];
rx(pi/2) q[70];
rz(-1.5802635) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rz(-pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(2.928543) q[57];
rz(-pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(-pi/2) q[56];
rx(pi/2) q[57];
rz(3.0514025) q[57];
cz q[56],q[57];
rx(1.1661825) q[56];
rz(-pi/2) q[56];
cz q[56],q[63];
rx(-pi/2) q[56];
rz(0.045291845) q[56];
rx(1.6326391) q[57];
rz(-2.5420969) q[57];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(0.98974841) q[56];
rz(-1.6531797) q[56];
rx(pi/2) q[63];
rz(0.19156409) q[63];
cz q[62],q[63];
rx(-pi/2) q[62];
rz(1.7491116) q[62];
rx(pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi/2) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
cz q[21],q[20];
rx(-pi/2) q[21];
rz(0.43172086) q[21];
rx(pi/2) q[21];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[63],q[20];
rx(-pi/2) q[20];
rz(1.4500923) q[20];
rx(pi/2) q[20];
rx(pi/2) q[63];
rz(2.7216929) q[63];
rx(-pi/2) q[63];
rx(1.7344015) q[70];
rz(pi/2) q[70];
barrier q[57],q[8],q[72],q[5],q[69],q[14],q[78],q[23],q[32],q[41],q[38],q[47],q[62],q[1],q[65],q[10],q[21],q[74],q[7],q[71],q[16],q[25],q[34],q[31],q[40],q[49],q[58],q[3],q[67],q[0],q[64],q[9],q[73],q[18],q[27],q[24],q[33],q[42],q[51],q[60],q[63],q[2],q[66],q[11],q[75],q[70],q[17],q[29],q[26],q[35],q[44],q[53],q[50],q[59],q[4],q[68],q[13],q[77],q[22],q[19],q[28],q[37],q[46],q[43],q[55],q[52],q[61],q[6],q[20],q[15],q[12],q[79],q[76],q[56],q[30],q[39],q[36],q[48],q[45],q[54];
measure q[70] -> meas[0];
measure q[57] -> meas[1];
measure q[56] -> meas[2];
measure q[62] -> meas[3];
measure q[21] -> meas[4];
measure q[63] -> meas[5];
measure q[20] -> meas[6];
