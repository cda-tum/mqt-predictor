OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[62];
sx q[14];
rz(0.14484353) q[14];
sx q[14];
sx q[18];
rz(1.7171836) q[18];
sx q[18];
rz(-pi) q[18];
sx q[19];
rz(0.14334753) q[19];
sx q[19];
sx q[22];
rz(1.9321635) q[22];
sx q[22];
rz(-pi) q[22];
sx q[23];
rz(0.14189703) q[23];
sx q[23];
sx q[24];
rz(1.9106333) q[24];
sx q[24];
rz(-pi) q[24];
sx q[25];
rz(0.38759673) q[25];
sx q[25];
sx q[29];
rz(2.034444) q[29];
sx q[29];
rz(-pi) q[29];
sx q[30];
rz(1.9913307) q[30];
sx q[30];
rz(-pi) q[30];
sx q[31];
rz(0.17771063) q[31];
sx q[31];
sx q[32];
rz(0.18058523) q[32];
sx q[32];
sx q[33];
rz(0.14798143) q[33];
sx q[33];
sx q[34];
rz(1.7316169) q[34];
sx q[34];
rz(-pi) q[34];
sx q[35];
rz(2*pi/3) q[35];
sx q[35];
rz(-pi) q[35];
sx q[36];
rz(0.18360403) q[36];
sx q[36];
sx q[37];
rz(0.15309593) q[37];
sx q[37];
sx q[38];
rz(0.15133263) q[38];
sx q[38];
sx q[39];
rz(0.14962893) q[39];
sx q[39];
sx q[40];
rz(0.15492233) q[40];
sx q[40];
sx q[41];
rz(0.15681573) q[41];
sx q[41];
sx q[42];
rz(0.15878023) q[42];
sx q[42];
sx q[43];
rz(0.14048973) q[43];
sx q[43];
sx q[44];
rz(1.7457654) q[44];
sx q[44];
rz(-pi) q[44];
sx q[46];
rz(3*pi/4) q[46];
sx q[46];
rz(-pi) q[46];
sx q[47];
rz(2.186276) q[47];
sx q[47];
rz(-pi) q[47];
sx q[51];
rz(0.18677943) q[51];
sx q[51];
sx q[52];
rz(0.23794113) q[52];
sx q[52];
sx q[53];
rz(0.16294153) q[53];
sx q[53];
sx q[56];
rz(0.23147733) q[56];
sx q[56];
sx q[57];
rz(1.815775) q[57];
sx q[57];
rz(-pi) q[57];
sx q[58];
rz(1.8770737) q[58];
sx q[58];
rz(-pi) q[58];
sx q[60];
rz(1.8925469) q[60];
sx q[60];
rz(-pi) q[60];
sx q[61];
rz(0.25268023) q[61];
sx q[61];
sx q[62];
rz(0.13650633) q[62];
sx q[62];
sx q[65];
rz(1.7099198) q[65];
sx q[65];
rz(-pi) q[65];
sx q[66];
rz(0.13779623) q[66];
sx q[66];
sx q[67];
rz(0.19365833) q[67];
sx q[67];
sx q[68];
rz(0.19012563) q[68];
sx q[68];
sx q[71];
rz(1.7359451) q[71];
sx q[71];
rz(-pi) q[71];
cx q[58],q[71];
cx q[71],q[58];
cx q[58],q[71];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
rz(pi/2) q[59];
sx q[59];
rz(pi/2) q[59];
sx q[72];
rz(0.13525193) q[72];
sx q[72];
sx q[73];
rz(1.7681919) q[73];
sx q[73];
rz(-pi) q[73];
sx q[76];
rz(1.7963098) q[76];
sx q[76];
rz(-pi) q[76];
sx q[77];
rz(1.8636391) q[77];
sx q[77];
rz(-pi) q[77];
sx q[78];
rz(0.13403153) q[78];
sx q[78];
sx q[79];
rz(0.13284363) q[79];
sx q[79];
sx q[80];
rz(0.13168673) q[80];
sx q[80];
sx q[81];
rz(0.13055953) q[81];
sx q[81];
sx q[82];
rz(1.743147) q[82];
sx q[82];
rz(-pi) q[82];
sx q[83];
rz(0.12946073) q[83];
sx q[83];
sx q[84];
rz(0.12838933) q[84];
sx q[84];
sx q[85];
rz(0.12734403) q[85];
sx q[85];
x q[86];
cx q[86],q[85];
sx q[85];
rz(0.12734403) q[85];
sx q[85];
cx q[85],q[84];
sx q[84];
rz(0.12838933) q[84];
sx q[84];
cx q[84],q[83];
sx q[83];
rz(0.12946073) q[83];
sx q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[82],q[81];
sx q[81];
rz(0.13055953) q[81];
sx q[81];
cx q[81],q[80];
sx q[80];
rz(0.13168673) q[80];
sx q[80];
cx q[80],q[79];
sx q[79];
rz(0.13284363) q[79];
sx q[79];
cx q[79],q[78];
sx q[78];
rz(0.13403153) q[78];
sx q[78];
cx q[85],q[86];
cx q[84],q[85];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[82],q[83];
cx q[81],q[82];
cx q[80],q[81];
cx q[79],q[80];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[79],q[80];
cx q[80],q[79];
cx q[79],q[80];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[79],q[80];
cx q[80],q[79];
cx q[79],q[80];
cx q[81],q[72];
sx q[72];
rz(0.13525193) q[72];
sx q[72];
cx q[72],q[62];
sx q[62];
rz(0.13650633) q[62];
sx q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[66];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
rz(pi/2) q[65];
sx q[65];
rz(pi/2) q[65];
sx q[66];
rz(0.13779623) q[66];
sx q[66];
cx q[66],q[65];
sx q[65];
rz(0.13912343) q[65];
sx q[65];
cx q[81],q[80];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[64],q[63];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[44],q[43];
sx q[43];
rz(0.14048973) q[43];
sx q[43];
cx q[34],q[43];
cx q[43],q[34];
cx q[34],q[43];
cx q[24],q[34];
cx q[34],q[24];
cx q[24],q[34];
cx q[24],q[23];
sx q[23];
rz(0.14189703) q[23];
sx q[23];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[20],q[21];
cx q[21],q[20];
cx q[20],q[21];
cx q[20],q[19];
sx q[19];
rz(0.14334753) q[19];
sx q[19];
cx q[18],q[19];
cx q[19],q[18];
cx q[18],q[19];
cx q[18],q[14];
sx q[14];
rz(0.14484353) q[14];
sx q[14];
cx q[14],q[18];
cx q[18],q[14];
cx q[14],q[18];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[18],q[19];
sx q[19];
rz(0.14638723) q[19];
sx q[19];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[33];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[19];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
sx q[33];
rz(0.14798143) q[33];
sx q[33];
cx q[33],q[39];
cx q[34],q[24];
cx q[24],q[34];
cx q[34],q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
sx q[39];
rz(0.14962893) q[39];
sx q[39];
cx q[39],q[38];
sx q[38];
rz(0.15133263) q[38];
sx q[38];
cx q[38],q[37];
sx q[37];
rz(0.15309593) q[37];
sx q[37];
cx q[37],q[38];
cx q[38],q[37];
cx q[37],q[38];
cx q[38],q[39];
cx q[39],q[38];
cx q[38],q[39];
cx q[39],q[40];
cx q[39],q[38];
cx q[38],q[39];
cx q[39],q[38];
sx q[40];
rz(0.15492233) q[40];
sx q[40];
cx q[40],q[41];
sx q[41];
rz(0.15681573) q[41];
sx q[41];
cx q[41],q[42];
sx q[42];
rz(0.15878023) q[42];
sx q[42];
rz(pi/2) q[43];
sx q[43];
rz(pi/2) q[43];
cx q[42],q[43];
sx q[43];
rz(0.16082053) q[43];
sx q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[53];
sx q[53];
rz(0.16294153) q[53];
sx q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
cx q[60],q[59];
sx q[59];
rz(0.16514873) q[59];
sx q[59];
cx q[58],q[59];
cx q[59],q[58];
cx q[58],q[59];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[44],q[45];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[71],q[58];
cx q[58],q[71];
cx q[71],q[58];
cx q[71],q[77];
cx q[77],q[71];
cx q[71],q[77];
cx q[77],q[78];
cx q[78],q[77];
cx q[77],q[78];
cx q[76],q[77];
cx q[77],q[76];
cx q[76],q[77];
rz(pi/2) q[77];
sx q[77];
rz(pi/2) q[77];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
rz(pi/2) q[84];
sx q[84];
rz(pi/2) q[84];
sx q[91];
rz(1.7907844) q[91];
sx q[91];
rz(-pi) q[91];
sx q[92];
rz(1.8319538) q[92];
sx q[92];
rz(-pi) q[92];
sx q[97];
rz(1.8518313) q[97];
sx q[97];
rz(-pi) q[97];
sx q[99];
rz(1.7856462) q[99];
sx q[99];
rz(-pi) q[99];
sx q[100];
rz(1.7382445) q[100];
sx q[100];
rz(-pi) q[100];
cx q[99],q[100];
cx q[100],q[99];
cx q[99],q[100];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[99];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[79],q[91];
sx q[91];
rz(0.16744813) q[91];
sx q[91];
cx q[98],q[91];
cx q[91],q[98];
cx q[98],q[91];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[99],q[98];
cx q[98],q[99];
cx q[99],q[98];
cx q[100],q[99];
cx q[99],q[100];
cx q[100],q[99];
rz(pi/2) q[99];
sx q[99];
rz(pi/2) q[99];
sx q[101];
rz(0.16984633) q[101];
sx q[101];
cx q[100],q[101];
sx q[101];
rz(0.16984633) q[101];
sx q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[83],q[84];
sx q[84];
rz(0.17235063) q[84];
sx q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[64],q[54];
sx q[54];
rz(0.17496903) q[54];
sx q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[35],q[47];
rz(pi/2) q[46];
sx q[46];
rz(pi/2) q[46];
cx q[47],q[35];
cx q[35],q[47];
cx q[28],q[35];
cx q[35],q[28];
cx q[28],q[35];
cx q[29],q[28];
cx q[28],q[29];
cx q[29],q[28];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[28],q[29];
cx q[29],q[28];
cx q[28],q[29];
rz(pi/2) q[28];
sx q[28];
rz(pi/2) q[28];
rz(pi/2) q[29];
sx q[29];
rz(pi/2) q[29];
cx q[30],q[31];
sx q[31];
rz(0.17771063) q[31];
sx q[31];
cx q[31],q[32];
sx q[32];
rz(0.18058523) q[32];
sx q[32];
cx q[32],q[36];
sx q[36];
rz(0.18360403) q[36];
sx q[36];
cx q[36],q[51];
rz(pi/2) q[47];
sx q[47];
rz(pi/2) q[47];
sx q[51];
rz(0.18677943) q[51];
sx q[51];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[49],q[50];
cx q[50],q[49];
cx q[49],q[50];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
cx q[55],q[68];
sx q[68];
rz(0.19012563) q[68];
sx q[68];
cx q[68],q[67];
sx q[67];
rz(0.19365833) q[67];
sx q[67];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
rz(pi/2) q[85];
sx q[85];
rz(pi/2) q[85];
cx q[73],q[85];
sx q[85];
rz(0.19739553) q[85];
sx q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
sx q[103];
rz(0.20135793) q[103];
sx q[103];
cx q[102],q[103];
sx q[103];
rz(0.20135793) q[103];
sx q[103];
sx q[104];
rz(1.7808521) q[104];
sx q[104];
rz(-pi) q[104];
cx q[103],q[104];
cx q[104],q[103];
cx q[103],q[104];
rz(pi/2) q[103];
sx q[103];
rz(pi/2) q[103];
sx q[105];
rz(0.20556893) q[105];
sx q[105];
cx q[104],q[105];
sx q[105];
rz(0.20556893) q[105];
sx q[105];
cx q[104],q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[104],q[103];
sx q[103];
rz(0.21005573) q[103];
sx q[103];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[100],q[99];
sx q[99];
rz(0.21484983) q[99];
sx q[99];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[99];
cx q[98],q[91];
sx q[91];
rz(0.21998803) q[91];
sx q[91];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[78],q[77];
sx q[77];
rz(0.22551343) q[77];
sx q[77];
cx q[77],q[71];
cx q[71],q[77];
cx q[77],q[71];
cx q[58],q[71];
cx q[71],q[58];
cx q[58],q[71];
cx q[57],q[58];
cx q[58],q[57];
cx q[57],q[58];
cx q[57],q[56];
sx q[56];
rz(0.23147733) q[56];
sx q[56];
cx q[56],q[52];
sx q[52];
rz(0.23794113) q[52];
sx q[52];
cx q[52],q[56];
cx q[56],q[52];
cx q[52],q[56];
cx q[56],q[57];
cx q[57],q[56];
cx q[56],q[57];
rz(pi/2) q[58];
sx q[58];
rz(pi/2) q[58];
cx q[57],q[58];
sx q[58];
rz(0.24497863) q[58];
sx q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[58],q[59];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[60],q[61];
sx q[61];
rz(0.25268023) q[61];
sx q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[72];
rz(pi/2) q[71];
sx q[71];
rz(pi/2) q[71];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[81];
cx q[77],q[78];
cx q[78],q[77];
cx q[77],q[78];
rz(pi/2) q[78];
sx q[78];
rz(pi/2) q[78];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[92];
sx q[92];
rz(0.26115743) q[92];
sx q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[97],q[98];
cx q[98],q[97];
cx q[97],q[98];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
sx q[110];
rz(1.8413461) q[110];
sx q[110];
rz(-pi) q[110];
cx q[100],q[110];
cx q[110],q[100];
cx q[100],q[110];
rz(pi/2) q[100];
sx q[100];
rz(pi/2) q[100];
cx q[101],q[100];
sx q[100];
rz(0.27054973) q[100];
sx q[100];
cx q[99],q[100];
cx q[100],q[99];
cx q[99],q[100];
cx q[99],q[98];
sx q[98];
rz(0.28103493) q[98];
sx q[98];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[79],q[78];
sx q[78];
rz(0.29284273) q[78];
sx q[78];
cx q[78],q[77];
cx q[77],q[78];
cx q[78],q[77];
cx q[77],q[71];
sx q[71];
rz(0.30627733) q[71];
sx q[71];
cx q[71],q[58];
cx q[58],q[71];
cx q[71],q[58];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[60],q[53];
sx q[53];
rz(0.32175053) q[53];
sx q[53];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[34],q[43];
cx q[43],q[34];
cx q[34],q[43];
cx q[34],q[24];
sx q[24];
rz(0.33983693) q[24];
sx q[24];
cx q[24],q[23];
sx q[23];
rz(0.36136713) q[23];
sx q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[25];
sx q[25];
rz(0.38759673) q[25];
sx q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[27],q[28];
sx q[28];
rz(0.42053433) q[28];
sx q[28];
cx q[28],q[29];
sx q[29];
rz(0.46364763) q[29];
sx q[29];
cx q[29],q[28];
cx q[28],q[29];
cx q[29],q[28];
cx q[35],q[28];
cx q[28],q[35];
cx q[35],q[28];
cx q[35],q[47];
cx q[43],q[44];
cx q[43],q[34];
cx q[34],q[43];
cx q[43],q[34];
cx q[24],q[34];
cx q[34],q[24];
cx q[24],q[34];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[20],q[21];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[20];
cx q[18],q[19];
cx q[19],q[18];
cx q[18],q[19];
cx q[18],q[14];
cx q[14],q[18];
cx q[19],q[18];
cx q[20],q[19];
cx q[33],q[20];
cx q[34],q[24];
cx q[24],q[34];
cx q[34],q[24];
cx q[39],q[33];
cx q[38],q[39];
cx q[39],q[38];
cx q[38],q[39];
cx q[38],q[37];
cx q[37],q[38];
cx q[39],q[38];
cx q[40],q[39];
cx q[41],q[40];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
sx q[47];
rz(pi/6) q[47];
sx q[47];
cx q[47],q[46];
sx q[46];
rz(0.61547971) q[46];
sx q[46];
cx q[46],q[45];
sx q[45];
rz(pi/4) q[45];
sx q[45];
cx q[53],q[41];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[58],q[59];
cx q[58],q[71];
cx q[71],q[58];
cx q[58],q[71];
cx q[71],q[77];
cx q[77],q[71];
cx q[71],q[77];
cx q[77],q[78];
cx q[78],q[77];
cx q[77],q[78];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[98],q[91];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[99];
cx q[99],q[100];
cx q[100],q[99];
cx q[99],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[102],q[101];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[84],q[83];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[28],q[35];
cx q[35],q[28];
cx q[28],q[35];
cx q[29],q[28];
cx q[28],q[29];
cx q[29],q[28];
cx q[30],q[29];
cx q[31],q[30];
cx q[32],q[31];
cx q[36],q[32];
cx q[36],q[51];
cx q[51],q[36];
cx q[36],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[49];
cx q[49],q[50];
cx q[50],q[49];
cx q[55],q[49];
cx q[68],q[55];
cx q[67],q[68];
cx q[68],q[67];
cx q[67],q[68];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[85],q[73];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[103],q[102];
cx q[103],q[104];
cx q[104],q[103];
cx q[103],q[104];
cx q[105],q[104];
cx q[104],q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[103],q[104];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[100],q[110];
cx q[110],q[100];
cx q[99],q[100];
cx q[100],q[99];
cx q[99],q[100];
cx q[99],q[98];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[97];
cx q[97],q[98];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[77],q[78];
cx q[71],q[77];
cx q[77],q[71];
cx q[71],q[77];
cx q[58],q[71];
cx q[71],q[58];
cx q[58],q[71];
cx q[57],q[58];
cx q[58],q[57];
cx q[57],q[58];
cx q[56],q[57];
cx q[56],q[52];
cx q[52],q[56];
cx q[56],q[57];
cx q[57],q[56];
cx q[56],q[57];
cx q[58],q[57];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[60],q[59];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[83],q[82];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[101],q[102];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[99],q[100];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[99];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[79],q[91];
cx q[79],q[78];
cx q[78],q[79];
cx q[79],q[78];
cx q[77],q[78];
cx q[71],q[77];
cx q[77],q[71];
cx q[71],q[77];
cx q[58],q[71];
cx q[71],q[58];
cx q[58],q[71];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[53],q[60];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[43],q[42];
cx q[34],q[43];
cx q[24],q[34];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[27],q[26];
cx q[28],q[27];
cx q[35],q[28];
cx q[28],q[35];
cx q[35],q[28];
cx q[47],q[35];
cx q[46],q[47];
cx q[45],q[46];
cx q[54],q[45];
barrier q[111],q[57],q[1],q[120],q[44],q[31],q[95],q[40],q[97],q[51],q[113],q[42],q[3],q[122],q[103],q[34],q[88],q[33],q[78],q[53],q[106],q[55],q[115],q[43],q[124],q[17],q[98],q[24],q[90],q[46],q[79],q[30],q[108],q[99],q[117],q[10],q[74],q[18],q[72],q[28],q[100],q[39],q[65],q[54],q[91],q[12],q[52],q[23],q[85],q[35],q[94],q[37],q[104],q[48],q[112],q[5],q[69],q[19],q[62],q[14],q[87],q[32],q[96],q[41],q[110],q[66],q[7],q[126],q[92],q[16],q[76],q[27],q[89],q[71],q[101],q[50],q[0],q[119],q[59],q[9],q[105],q[20],q[29],q[25],q[56],q[82],q[2],q[121],q[64],q[11],q[75],q[22],q[81],q[93],q[36],q[114],q[77],q[4],q[123],q[73],q[13],q[60],q[26],q[86],q[21],q[107],q[58],q[116],q[102],q[6],q[125],q[70],q[15],q[80],q[49],q[84],q[67],q[109],q[68],q[118],q[61],q[8],q[63],q[47],q[38],q[83],q[45];
measure q[54] -> meas[0];
measure q[45] -> meas[1];
measure q[46] -> meas[2];
measure q[47] -> meas[3];
measure q[35] -> meas[4];
measure q[27] -> meas[5];
measure q[26] -> meas[6];
measure q[34] -> meas[7];
measure q[43] -> meas[8];
measure q[42] -> meas[9];
measure q[60] -> meas[10];
measure q[78] -> meas[11];
measure q[91] -> meas[12];
measure q[100] -> meas[13];
measure q[102] -> meas[14];
measure q[82] -> meas[15];
measure q[58] -> meas[16];
measure q[57] -> meas[17];
measure q[52] -> meas[18];
measure q[56] -> meas[19];
measure q[79] -> meas[20];
measure q[97] -> meas[21];
measure q[110] -> meas[22];
measure q[104] -> meas[23];
measure q[105] -> meas[24];
measure q[103] -> meas[25];
measure q[73] -> meas[26];
measure q[55] -> meas[27];
measure q[49] -> meas[28];
measure q[32] -> meas[29];
measure q[31] -> meas[30];
measure q[30] -> meas[31];
measure q[29] -> meas[32];
measure q[65] -> meas[33];
measure q[84] -> meas[34];
measure q[92] -> meas[35];
measure q[99] -> meas[36];
measure q[71] -> meas[37];
measure q[53] -> meas[38];
measure q[41] -> meas[39];
measure q[40] -> meas[40];
measure q[39] -> meas[41];
measure q[38] -> meas[42];
measure q[37] -> meas[43];
measure q[33] -> meas[44];
measure q[20] -> meas[45];
measure q[19] -> meas[46];
measure q[18] -> meas[47];
measure q[14] -> meas[48];
measure q[21] -> meas[49];
measure q[44] -> meas[50];
measure q[64] -> meas[51];
measure q[66] -> meas[52];
measure q[63] -> meas[53];
measure q[62] -> meas[54];
measure q[80] -> meas[55];
measure q[76] -> meas[56];
measure q[98] -> meas[57];
measure q[72] -> meas[58];
measure q[81] -> meas[59];
measure q[85] -> meas[60];
measure q[86] -> meas[61];
