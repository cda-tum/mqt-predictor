OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg meas[32];
rx(-pi/2) q[10];
rz(1.1071487) q[10];
rx(pi/2) q[10];
rx(-pi/2) q[11];
rz(1.3452829) q[11];
rx(pi/2) q[11];
rx(-pi/2) q[12];
rz(1.3508083) q[12];
rx(pi/2) q[12];
rx(-pi/2) q[18];
rz(1.2094292) q[18];
rx(pi/2) q[18];
rx(-pi/2) q[19];
rz(1.1831996) q[19];
rx(pi/2) q[19];
rz(-pi) q[20];
rx(-pi/2) q[20];
rx(-pi/2) q[21];
rz(1.150262) q[21];
rx(pi/2) q[21];
rx(-pi/2) q[22];
rz(0.95531662) q[22];
rx(pi/2) q[22];
rx(-pi/2) q[23];
rz(pi/4) q[23];
rx(pi/2) q[23];
rx(-pi/2) q[26];
rz(1.3902111) q[26];
rx(pi/2) q[26];
rx(-pi/2) q[27];
rz(1.3871923) q[27];
rx(pi/2) q[27];
rz(-pi) q[28];
rx(-pi/2) q[28];
rx(-pi/2) q[29];
rz(1.2309594) q[29];
rx(pi/2) q[29];
rx(pi) q[36];
rx(-pi/2) q[37];
rz(1.3930857) q[37];
rx(pi/2) q[37];
cz q[36],q[37];
rz(-pi/2) q[36];
rx(-pi/2) q[36];
rx(pi/2) q[37];
rz(1.3930857) q[37];
rx(-pi/2) q[37];
cz q[37],q[26];
rx(pi/2) q[26];
rz(1.3902111) q[26];
rx(-pi/2) q[26];
cz q[26],q[27];
rz(pi/2) q[27];
rx(0.18360403) q[27];
cz q[37],q[36];
rx(pi/2) q[36];
rz(pi/2) q[36];
rz(-pi/2) q[37];
rx(-pi/2) q[37];
cz q[26],q[37];
rx(pi/2) q[37];
rz(pi/2) q[37];
rz(-pi/2) q[40];
rx(0.21005573) q[40];
rx(-pi/2) q[41];
rz(1.3652274) q[41];
rx(pi/2) q[41];
rz(pi/2) q[42];
rx(pi/2) q[42];
rz(-pi) q[48];
rx(-pi/2) q[48];
rz(pi/2) q[49];
rx(pi/2) q[49];
rx(-pi/2) q[50];
rz(1.3734008) q[50];
rx(pi/2) q[50];
rz(-pi) q[51];
rx(-pi/2) q[51];
rx(-pi/2) q[52];
rz(1.3694384) q[52];
rx(pi/2) q[52];
rz(-pi) q[53];
rx(-pi/2) q[53];
rz(-pi) q[54];
rx(-pi/2) q[54];
rx(-pi/2) q[55];
rz(1.3559465) q[55];
rx(pi/2) q[55];
rz(pi/2) q[56];
rx(pi/2) q[56];
rx(-pi/2) q[57];
rz(1.3840169) q[57];
rx(pi/2) q[57];
rx(-pi/2) q[62];
rz(1.377138) q[62];
rx(pi/2) q[62];
rx(-pi/2) q[63];
rz(1.3806707) q[63];
rx(pi/2) q[63];
rz(-pi) q[64];
rx(-pi/2) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(-pi/2) q[64];
rx(pi/2) q[64];
cz q[27],q[64];
rx(-pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[64];
cz q[27],q[64];
cz q[26],q[27];
rz(-pi/2) q[26];
rx(-pi/2) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[27],q[26];
rx(pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
cz q[27],q[28];
rx(pi/2) q[27];
rz(-pi/2) q[28];
rx(pi/2) q[28];
cz q[27],q[28];
rx(-pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(-pi/2) q[28];
rz(pi) q[28];
rx(-pi/2) q[64];
rz(pi) q[64];
rx(-pi/2) q[65];
rz(1.3181161) q[65];
rx(pi/2) q[65];
rx(-pi/2) q[67];
rz(1.2779536) q[67];
rx(pi/2) q[67];
rx(-pi/2) q[68];
rz(1.264519) q[68];
rx(pi/2) q[68];
rz(-pi) q[69];
rx(-pi/2) q[69];
rz(pi/2) q[70];
rx(pi/2) q[70];
rz(-pi) q[71];
rx(-pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(-pi/2) q[71];
rx(pi/2) q[71];
cz q[64],q[71];
rx(-pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(-pi/2) q[71];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(-pi/2) q[70];
cz q[70],q[57];
rx(-pi/2) q[57];
rz(-1.3840169) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
cz q[56],q[63];
rx(-1.3258177) q[57];
rx(pi/2) q[63];
rz(1.3806707) q[63];
rx(-pi/2) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(2.9479343) q[62];
rx(pi/2) q[62];
cz q[61],q[62];
rz(-pi/2) q[61];
rx(-pi/2) q[61];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
cz q[62],q[61];
rx(pi/2) q[61];
rz(pi/2) q[61];
rx(pi/2) q[62];
cz q[61],q[62];
cz q[61],q[50];
rz(pi/2) q[50];
rx(0.19739553) q[50];
cz q[50],q[51];
rx(pi/2) q[50];
rz(-pi/2) q[51];
rx(pi/2) q[51];
cz q[50],q[51];
rx(-pi/2) q[50];
rz(pi/2) q[50];
rx(pi/2) q[51];
cz q[50],q[51];
rx(-1.339319) q[50];
rz(pi/2) q[50];
rx(-pi/2) q[50];
cz q[49],q[50];
rx(pi/2) q[49];
rz(-pi/2) q[50];
rx(pi/2) q[50];
cz q[49],q[50];
rx(-pi/2) q[49];
rx(pi/2) q[50];
cz q[49],q[50];
rx(-1.3328552) q[50];
rz(pi) q[50];
rz(pi) q[51];
cz q[51],q[52];
rz(pi/2) q[52];
rx(0.20135793) q[52];
cz q[52],q[53];
rx(pi/2) q[52];
rz(-pi/2) q[53];
rx(pi/2) q[53];
cz q[52],q[53];
rx(-pi/2) q[52];
rz(pi/2) q[52];
rx(pi/2) q[53];
cz q[52],q[53];
rx(-pi/2) q[53];
rz(pi) q[53];
cz q[53],q[54];
rx(pi/2) q[53];
rz(-pi/2) q[54];
rx(pi/2) q[54];
cz q[53],q[54];
rx(-pi/2) q[53];
rz(pi) q[53];
rx(pi/2) q[54];
cz q[53],q[54];
rx(-pi/2) q[53];
rz(pi) q[54];
cz q[54],q[41];
rx(1.5491603) q[41];
rz(1.3652742) q[41];
rx(-3.1371766) q[41];
cz q[40],q[41];
rx(-pi/2) q[40];
rz(-pi/2) q[40];
rx(pi/2) q[41];
rz(pi) q[41];
cz q[40],q[41];
rx(-2.9316894) q[40];
rz(pi/2) q[40];
rx(0.21005573) q[41];
rx(pi/2) q[54];
rz(pi) q[54];
cz q[41],q[54];
rx(pi/2) q[41];
rz(pi) q[41];
rx(pi/2) q[54];
rz(pi) q[54];
cz q[54],q[41];
rx(pi/2) q[41];
rz(pi/2) q[41];
rx(pi/2) q[54];
rz(pi) q[54];
cz q[41],q[54];
rx(pi/2) q[54];
rz(pi) q[54];
cz q[54],q[55];
rx(pi/2) q[55];
rz(1.3559465) q[55];
rx(-pi/2) q[55];
cz q[55],q[12];
rx(pi/2) q[12];
rz(1.3508083) q[12];
rx(-pi/2) q[12];
cz q[12],q[11];
rz(pi/2) q[11];
rx(0.22551343) q[11];
cz q[11],q[48];
rx(pi/2) q[11];
rz(-2.9105089) q[12];
rx(-pi/2) q[12];
rz(-pi/2) q[48];
rx(pi/2) q[48];
cz q[11],q[48];
rx(-pi/2) q[11];
rz(pi/2) q[11];
rx(pi/2) q[48];
cz q[11],q[48];
rx(pi/6) q[11];
rz(0.54817577) q[11];
rx(pi/2) q[11];
rz(pi) q[48];
cz q[48],q[49];
rx(pi/2) q[49];
rz(1.339319) q[49];
rx(-pi/2) q[49];
cz q[49],q[50];
rx(0.23794113) q[50];
rx(-2.8198421) q[71];
rz(pi/2) q[71];
rx(-pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(-pi/2) q[71];
rx(pi/2) q[71];
cz q[28],q[71];
rx(-pi/2) q[28];
rx(pi/2) q[71];
cz q[28],q[71];
rx(-pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(-pi/2) q[70];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[63],q[56];
rx(pi/2) q[56];
rx(-pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rx(pi/2) q[63];
cz q[62],q[63];
rx(-pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[61],q[62];
rx(-pi/2) q[61];
cz q[50],q[61];
rx(pi/2) q[50];
rx(pi/2) q[61];
cz q[50],q[61];
rx(-pi/2) q[50];
rz(pi/2) q[50];
rx(pi/2) q[61];
cz q[50],q[61];
rx(pi/2) q[50];
rz(pi) q[50];
cz q[51],q[50];
rx(pi/2) q[50];
rz(pi/2) q[50];
cz q[51],q[52];
rx(pi/2) q[51];
rz(pi) q[51];
rx(pi/2) q[52];
rz(pi) q[52];
cz q[52],q[51];
rx(pi/2) q[51];
rz(pi/2) q[51];
rx(pi/2) q[52];
cz q[51],q[52];
cz q[52],q[53];
rx(pi/2) q[52];
rz(-pi/2) q[53];
rx(pi/2) q[53];
cz q[52],q[53];
rx(-pi/2) q[52];
rx(pi/2) q[53];
cz q[52],q[53];
rx(-pi/2) q[53];
rz(pi/2) q[53];
rx(-pi/2) q[53];
cz q[42],q[53];
rx(pi/2) q[42];
rz(-pi/2) q[53];
rx(pi/2) q[53];
cz q[42],q[53];
rx(-pi/2) q[42];
rz(pi/2) q[42];
rx(pi/2) q[53];
cz q[42],q[53];
rx(pi/2) q[42];
rz(pi) q[42];
cz q[41],q[42];
rx(0.92215795) q[41];
rz(-pi/2) q[41];
cz q[40],q[41];
rx(-pi/2) q[40];
rz(-pi/2) q[40];
rx(pi/2) q[41];
rz(-pi/2) q[41];
cz q[40],q[41];
rx(-2.2194347) q[40];
rz(-pi) q[40];
rx(-1.8018801) q[41];
rx(pi/2) q[42];
rz(pi/2) q[42];
rz(pi/2) q[53];
cz q[54],q[41];
rz(pi/2) q[41];
rx(pi/2) q[41];
rz(pi/2) q[41];
rx(pi/2) q[54];
rz(pi) q[54];
cz q[55],q[54];
rx(pi/2) q[54];
rz(pi/2) q[54];
rx(0.92215795) q[55];
rz(-pi/2) q[55];
cz q[12],q[55];
rx(-pi/2) q[12];
rz(-pi/2) q[12];
rx(pi/2) q[55];
rz(-pi/2) q[55];
cz q[12],q[55];
rx(-2.2194347) q[12];
rz(-pi) q[12];
rx(-1.8018801) q[55];
cz q[48],q[55];
rx(pi/2) q[48];
rz(pi) q[48];
cz q[49],q[48];
rx(pi/2) q[48];
rz(pi/2) q[48];
rz(pi/2) q[55];
rx(pi/2) q[55];
rz(pi/2) q[55];
rz(pi) q[61];
cz q[61],q[62];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[61];
rx(pi/2) q[61];
rz(pi/2) q[61];
rx(pi/2) q[62];
cz q[61],q[62];
rx(-pi/2) q[63];
rz(pi/2) q[63];
rx(-pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(-pi/2) q[63];
rx(pi/2) q[63];
cz q[62],q[63];
rx(-pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[49],q[62];
rz(-pi/2) q[49];
rx(-pi/2) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[49];
rx(pi/2) q[49];
rz(pi/2) q[49];
rx(pi/2) q[62];
cz q[49],q[62];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(-pi/2) q[63];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
cz q[56],q[63];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(-pi/2) q[63];
rx(pi/2) q[63];
cz q[62],q[63];
rx(-pi/2) q[62];
rx(pi/2) q[63];
cz q[62],q[63];
rx(-pi/2) q[63];
rz(pi/2) q[63];
rx(-pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(-pi/2) q[63];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rz(pi/2) q[63];
rx(-pi/2) q[70];
rz(-pi/2) q[70];
cz q[57],q[70];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(2.896614) q[70];
rx(pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(-pi/2) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[70];
rx(-pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(-pi/2) q[71];
rx(pi/2) q[71];
cz q[64],q[71];
rx(-pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[64],q[65];
rx(pi/2) q[65];
rz(1.3181161) q[65];
rx(-pi/2) q[65];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(-pi/2) q[71];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
rx(-pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
rx(-pi/2) q[77];
rz(1.2897614) q[77];
rx(pi/2) q[77];
rx(-pi/2) q[78];
rz(1.3096389) q[78];
rx(pi/2) q[78];
cz q[65],q[78];
cz q[65],q[64];
rx(pi/2) q[64];
rz(pi/2) q[64];
rz(pi/2) q[65];
rx(pi/2) q[65];
rx(0.76815336) q[78];
rz(1.3871923) q[78];
rx(0.18677947) q[78];
rx(-pi/2) q[79];
rz(1.3002466) q[79];
cz q[78],q[79];
rx(-pi/2) q[78];
rz(-1.3002466) q[78];
rx(pi/2) q[79];
rz(pi/2) q[79];
cz q[78],q[79];
rx(pi/2) q[78];
cz q[78],q[77];
rx(pi/2) q[77];
rz(2.8605577) q[77];
rx(pi/2) q[77];
cz q[66],q[77];
rz(-pi/2) q[66];
rx(-pi/2) q[66];
rz(-pi/2) q[77];
rx(-pi/2) q[77];
cz q[77],q[66];
rx(pi/2) q[66];
rz(pi) q[66];
rx(pi/2) q[77];
rz(pi) q[77];
cz q[66],q[77];
cz q[66],q[67];
rx(pi/2) q[67];
rz(1.2779536) q[67];
rx(-pi/2) q[67];
cz q[67],q[68];
rz(pi/2) q[68];
rx(0.30627733) q[68];
cz q[68],q[69];
rx(pi/2) q[68];
rz(-pi/2) q[69];
rx(pi/2) q[69];
cz q[68],q[69];
rx(-pi/2) q[68];
rz(pi/2) q[68];
rx(pi/2) q[69];
cz q[68],q[69];
rx(-pi/2) q[69];
rz(pi) q[69];
cz q[69],q[70];
rx(pi/2) q[69];
rz(-pi/2) q[70];
rx(pi/2) q[70];
cz q[69],q[70];
rx(-pi/2) q[69];
rz(pi/2) q[69];
rx(pi/2) q[70];
cz q[69],q[70];
rx(-pi/2) q[69];
rz(pi/2) q[69];
rx(pi/2) q[69];
rx(-pi/2) q[70];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rx(-pi/2) q[70];
cz q[71],q[28];
rx(pi/2) q[28];
rz(1.2490458) q[28];
rx(-pi/2) q[28];
cz q[28],q[29];
rx(pi/2) q[29];
rz(1.2309594) q[29];
rx(-pi/2) q[29];
cz q[29],q[18];
rx(pi/2) q[18];
rz(1.2094292) q[18];
rx(-pi/2) q[18];
cz q[18],q[19];
rz(pi/2) q[19];
rx(0.38759673) q[19];
cz q[19],q[20];
rx(pi/2) q[19];
rz(-pi/2) q[20];
rx(pi/2) q[20];
cz q[19],q[20];
rx(-pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[20];
cz q[19],q[20];
rx(-pi/2) q[19];
rz(pi) q[20];
cz q[20],q[21];
rx(pi/2) q[21];
rz(1.150262) q[21];
rx(-pi/2) q[21];
cz q[21],q[10];
rx(-2.8922326) q[10];
rz(1.6935677) q[10];
rx(2.693351) q[10];
cz q[10],q[11];
rx(-pi/2) q[10];
rz(-2.8493331) q[10];
rx(pi/2) q[11];
rz(pi/2) q[11];
cz q[10],q[11];
rx(1.1299132) q[10];
rz(-0.61411247) q[10];
rx(2.3710654) q[11];
rz(pi/2) q[11];
cz q[21],q[10];
rx(pi/2) q[10];
rz(pi) q[10];
rz(-pi/2) q[21];
rx(-pi/2) q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rx(pi/2) q[21];
rz(pi/2) q[21];
cz q[21],q[10];
cz q[21],q[22];
rx(-pi/2) q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rx(pi/2) q[21];
cz q[10],q[21];
rx(-pi/2) q[10];
rz(3*pi/2) q[10];
rx(pi/2) q[21];
cz q[10],q[21];
rz(-pi/2) q[21];
rx(pi/2) q[22];
rz(0.95531662) q[22];
rx(-pi/2) q[22];
cz q[22],q[23];
rx(pi/2) q[23];
rz(pi/4) q[23];
rx(-pi/2) q[23];
rx(pi/2) q[77];
rz(pi/2) q[77];
rx(-pi/2) q[78];
cz q[65],q[78];
rx(pi/2) q[65];
rx(pi/2) q[78];
cz q[65],q[78];
rx(-pi/2) q[65];
rz(pi/2) q[65];
rx(pi/2) q[78];
cz q[65],q[78];
rx(-pi/2) q[78];
rz(1.3881742) q[78];
rx(-pi/2) q[78];
rx(0.51647432) q[79];
rz(-pi/2) q[79];
cz q[78],q[79];
rx(-pi/2) q[78];
rz(-3*pi/2) q[78];
rx(pi/2) q[79];
rz(-pi/2) q[79];
cz q[78],q[79];
rx(-2.8726688) q[78];
cz q[65],q[78];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[66],q[65];
rx(pi/2) q[65];
rz(pi/2) q[65];
rx(pi/2) q[66];
rz(pi) q[66];
cz q[67],q[66];
rx(pi/2) q[66];
rz(pi/2) q[66];
cz q[67],q[68];
rz(-pi/2) q[67];
rx(-pi/2) q[67];
rx(pi/2) q[68];
rz(pi) q[68];
cz q[68],q[67];
rx(pi/2) q[67];
rz(pi/2) q[67];
rx(pi/2) q[68];
cz q[67],q[68];
cz q[68],q[69];
rx(pi/2) q[68];
rz(-pi/2) q[69];
rx(pi/2) q[69];
cz q[68],q[69];
rx(-pi/2) q[68];
rx(pi/2) q[69];
cz q[68],q[69];
cz q[69],q[70];
rx(pi/2) q[69];
rx(pi/2) q[70];
cz q[69],q[70];
rx(-pi/2) q[69];
rz(pi/2) q[69];
rx(pi/2) q[70];
cz q[69],q[70];
rx(pi/2) q[69];
rz(pi/2) q[69];
rz(pi) q[70];
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
rz(pi/2) q[18];
rx(pi/2) q[18];
cz q[18],q[19];
rx(pi/2) q[18];
rz(-pi/2) q[19];
rx(pi/2) q[19];
cz q[18],q[19];
rx(-pi/2) q[18];
rx(pi/2) q[19];
cz q[18],q[19];
rx(-pi/2) q[19];
cz q[20],q[19];
rx(pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[21],q[20];
rx(pi/2) q[20];
rz(pi/2) q[20];
rx(-pi/2) q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rz(-pi/2) q[21];
rx(pi/2) q[21];
cz q[10],q[21];
rx(-pi/2) q[10];
rz(3.3757029) q[10];
rx(pi/2) q[21];
cz q[10],q[21];
rx(-pi/2) q[10];
cz q[10],q[11];
rx(-pi/2) q[10];
rz(-pi/2) q[10];
rx(pi/2) q[11];
rz(-pi/2) q[11];
cz q[10],q[11];
rx(-2.6484446) q[10];
rz(-pi/2) q[10];
rx(-1.8049066) q[11];
rx(-pi/2) q[21];
cz q[21],q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[22],q[21];
rx(pi/2) q[21];
rz(pi/2) q[21];
rz(-pi/2) q[22];
rx(-pi/2) q[22];
cz q[23],q[22];
rx(pi/2) q[22];
rz(pi/2) q[22];
rx(pi/2) q[29];
rz(pi/2) q[29];
rx(pi/2) q[71];
rz(pi/2) q[71];
rz(pi/2) q[78];
rx(pi/2) q[78];
rz(pi/2) q[78];
rx(-1.3881742) q[79];
barrier q[25],q[34],q[43],q[40],q[49],q[58],q[3],q[70],q[55],q[76],q[11],q[19],q[30],q[63],q[36],q[45],q[52],q[56],q[60],q[5],q[67],q[14],q[78],q[23],q[18],q[29],q[38],q[47],q[44],q[64],q[51],q[50],q[7],q[68],q[16],q[13],q[66],q[22],q[31],q[54],q[37],q[69],q[46],q[12],q[0],q[26],q[9],q[6],q[73],q[28],q[15],q[65],q[24],q[33],q[53],q[39],q[21],q[62],q[2],q[77],q[48],q[61],q[8],q[75],q[72],q[17],q[57],q[35],q[32],q[41],q[42],q[59],q[4],q[1],q[71],q[79],q[10],q[74],q[20],q[27];
measure q[23] -> meas[0];
measure q[22] -> meas[1];
measure q[21] -> meas[2];
measure q[10] -> meas[3];
measure q[11] -> meas[4];
measure q[20] -> meas[5];
measure q[19] -> meas[6];
measure q[29] -> meas[7];
measure q[28] -> meas[8];
measure q[71] -> meas[9];
measure q[70] -> meas[10];
measure q[66] -> meas[11];
measure q[65] -> meas[12];
measure q[78] -> meas[13];
measure q[79] -> meas[14];
measure q[64] -> meas[15];
measure q[69] -> meas[16];
measure q[56] -> meas[17];
measure q[48] -> meas[18];
measure q[55] -> meas[19];
measure q[12] -> meas[20];
measure q[54] -> meas[21];
measure q[41] -> meas[22];
measure q[40] -> meas[23];
measure q[42] -> meas[24];
measure q[50] -> meas[25];
measure q[61] -> meas[26];
measure q[62] -> meas[27];
measure q[63] -> meas[28];
measure q[57] -> meas[29];
measure q[37] -> meas[30];
measure q[36] -> meas[31];
