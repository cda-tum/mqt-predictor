OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[84];
sx q[4];
rz(1.7681919) q[4];
sx q[4];
rz(-pi) q[4];
sx q[8];
rz(0.13284363) q[8];
sx q[8];
sx q[11];
rz(0.18058523) q[11];
sx q[11];
sx q[12];
rz(0.17771063) q[12];
sx q[12];
sx q[15];
rz(1.7721543) q[15];
sx q[15];
rz(-pi) q[15];
cx q[4],q[15];
cx q[15],q[4];
cx q[4],q[15];
sx q[16];
rz(1.7048279) q[16];
sx q[16];
rz(-pi) q[16];
sx q[17];
rz(1.7544004) q[17];
sx q[17];
rz(-pi) q[17];
sx q[20];
rz(0.14334753) q[20];
sx q[20];
sx q[21];
rz(2.034444) q[21];
sx q[21];
rz(-pi) q[21];
sx q[22];
rz(1.9106333) q[22];
sx q[22];
rz(-pi) q[22];
cx q[15],q[22];
cx q[22],q[15];
cx q[15],q[22];
cx q[4],q[15];
cx q[15],q[4];
cx q[4],q[15];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[15],q[22];
cx q[22],q[15];
cx q[15],q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[4],q[15];
cx q[15],q[4];
cx q[4],q[15];
sx q[24];
rz(0.17496903) q[24];
sx q[24];
sx q[25];
rz(1.7763653) q[25];
sx q[25];
rz(-pi) q[25];
sx q[26];
rz(0.13168673) q[26];
sx q[26];
sx q[27];
rz(0.13055953) q[27];
sx q[27];
sx q[28];
rz(0.12946073) q[28];
sx q[28];
sx q[29];
rz(1.8925469) q[29];
sx q[29];
rz(-pi) q[29];
sx q[30];
rz(1.7808521) q[30];
sx q[30];
rz(-pi) q[30];
sx q[32];
rz(0.21484983) q[32];
sx q[32];
sx q[33];
rz(0.14189703) q[33];
sx q[33];
sx q[34];
rz(1.760922) q[34];
sx q[34];
rz(-pi) q[34];
sx q[35];
rz(1.6991857) q[35];
sx q[35];
rz(-pi) q[35];
sx q[36];
rz(0.21998803) q[36];
sx q[36];
sx q[37];
rz(1.9913307) q[37];
sx q[37];
rz(-pi) q[37];
sx q[38];
rz(1.9321635) q[38];
sx q[38];
rz(-pi) q[38];
sx q[39];
rz(0.14048973) q[39];
sx q[39];
sx q[40];
rz(1.7644547) q[40];
sx q[40];
rz(-pi) q[40];
sx q[41];
rz(1.6961242) q[41];
sx q[41];
rz(-pi) q[41];
sx q[42];
rz(1.6971202) q[42];
sx q[42];
rz(-pi) q[42];
sx q[43];
rz(1.743147) q[43];
sx q[43];
rz(-pi) q[43];
sx q[44];
rz(1.7099198) q[44];
sx q[44];
rz(-pi) q[44];
sx q[45];
rz(0.11274753) q[45];
sx q[45];
sx q[46];
rz(1.7060483) q[46];
sx q[46];
rz(-pi) q[46];
sx q[47];
rz(1.7073027) q[47];
sx q[47];
rz(-pi) q[47];
sx q[48];
rz(1.7085926) q[48];
sx q[48];
rz(-pi) q[48];
sx q[49];
rz(1.6981404) q[49];
sx q[49];
rz(-pi) q[49];
cx q[48],q[49];
cx q[49],q[48];
cx q[48],q[49];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[35],q[47];
rz(pi/2) q[46];
sx q[46];
rz(pi/2) q[46];
cx q[47],q[35];
cx q[35],q[47];
rz(pi/2) q[47];
sx q[47];
rz(pi/2) q[47];
rz(pi/2) q[48];
sx q[48];
rz(pi/2) q[48];
rz(pi/2) q[49];
sx q[49];
rz(pi/2) q[49];
sx q[51];
rz(0.22551343) q[51];
sx q[51];
sx q[52];
rz(1.9583931) q[52];
sx q[52];
rz(-pi) q[52];
cx q[37],q[52];
cx q[52],q[37];
cx q[37],q[52];
sx q[53];
rz(0.11347103) q[53];
sx q[53];
sx q[54];
rz(1.7171836) q[54];
sx q[54];
rz(-pi) q[54];
sx q[55];
rz(0.16984633) q[55];
sx q[55];
sx q[58];
rz(0.11420873) q[58];
sx q[58];
x q[59];
sx q[60];
rz(0.10932663) q[60];
sx q[60];
cx q[59],q[60];
sx q[60];
rz(0.10932663) q[60];
sx q[60];
sx q[61];
rz(0.10998583) q[61];
sx q[61];
cx q[60],q[61];
cx q[60],q[59];
sx q[61];
rz(0.10998583) q[61];
sx q[61];
sx q[62];
rz(0.11065723) q[62];
sx q[62];
cx q[61],q[62];
cx q[61],q[60];
sx q[62];
rz(0.11065723) q[62];
sx q[62];
sx q[63];
rz(1.7156399) q[63];
sx q[63];
rz(-pi) q[63];
sx q[64];
rz(2*pi/3) q[64];
sx q[64];
rz(-pi) q[64];
sx q[65];
rz(1.7575758) q[65];
sx q[65];
rz(-pi) q[65];
sx q[66];
rz(0.12156703) q[66];
sx q[66];
sx q[67];
rz(1.7359451) q[67];
sx q[67];
rz(-pi) q[67];
sx q[68];
rz(0.14798143) q[68];
sx q[68];
sx q[69];
rz(1.7382445) q[69];
sx q[69];
rz(-pi) q[69];
sx q[70];
rz(0.14962893) q[70];
sx q[70];
sx q[72];
rz(1.8770737) q[72];
sx q[72];
rz(-pi) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
sx q[73];
rz(0.12067853) q[73];
sx q[73];
sx q[76];
rz(0.12435503) q[76];
sx q[76];
sx q[77];
rz(2.186276) q[77];
sx q[77];
rz(-pi) q[77];
sx q[78];
rz(0.11496093) q[78];
sx q[78];
sx q[79];
rz(1.8413461) q[79];
sx q[79];
rz(-pi) q[79];
sx q[80];
rz(0.11134103) q[80];
sx q[80];
sx q[81];
rz(1.682834) q[81];
sx q[81];
rz(-pi) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[80];
sx q[80];
rz(0.11134103) q[80];
sx q[80];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[62];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[80],q[81];
sx q[81];
rz(0.11203763) q[81];
sx q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[54],q[45];
sx q[45];
rz(0.11274753) q[45];
sx q[45];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[41],q[53];
sx q[53];
rz(0.11347103) q[53];
sx q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[58];
sx q[58];
rz(0.11420873) q[58];
sx q[58];
cx q[58],q[71];
cx q[71],q[58];
cx q[58],q[71];
cx q[71],q[77];
cx q[77],q[71];
cx q[71],q[77];
cx q[77],q[78];
sx q[78];
rz(0.11496093) q[78];
sx q[78];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[41],q[42];
cx q[41],q[53];
rz(pi/2) q[43];
sx q[43];
rz(pi/2) q[43];
rz(pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
cx q[53],q[41];
cx q[41],q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[59],q[60];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[58],q[71];
cx q[71],q[58];
cx q[58],q[71];
cx q[77],q[71];
cx q[77],q[78];
cx q[78],q[77];
cx q[77],q[78];
sx q[82];
rz(1.8319538) q[82];
sx q[82];
rz(-pi) q[82];
sx q[83];
rz(1.7316169) q[83];
sx q[83];
rz(-pi) q[83];
sx q[84];
rz(1.7337379) q[84];
sx q[84];
rz(-pi) q[84];
sx q[85];
rz(0.11980933) q[85];
sx q[85];
sx q[87];
rz(0.15492233) q[87];
sx q[87];
sx q[88];
rz(0.15309593) q[88];
sx q[88];
sx q[89];
rz(0.15133263) q[89];
sx q[89];
sx q[91];
rz(3*pi/4) q[91];
sx q[91];
rz(-pi) q[91];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
sx q[92];
rz(0.11895853) q[92];
sx q[92];
sx q[95];
rz(0.12340443) q[95];
sx q[95];
sx q[96];
rz(1.8636391) q[96];
sx q[96];
rz(-pi) q[96];
sx q[97];
rz(0.12247543) q[97];
sx q[97];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
sx q[99];
rz(1.8518313) q[99];
sx q[99];
rz(-pi) q[99];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[99];
cx q[91],q[98];
cx q[98],q[91];
cx q[91],q[98];
sx q[100];
rz(0.11572823) q[100];
sx q[100];
cx q[99],q[100];
sx q[100];
rz(0.11572823) q[100];
sx q[100];
cx q[99],q[98];
sx q[102];
rz(1.7276121) q[102];
sx q[102];
rz(-pi) q[102];
sx q[103];
rz(1.815775) q[103];
sx q[103];
rz(-pi) q[103];
sx q[104];
rz(1.8087375) q[104];
sx q[104];
rz(-pi) q[104];
sx q[105];
rz(1.8022737) q[105];
sx q[105];
rz(-pi) q[105];
sx q[110];
rz(1.7295766) q[110];
sx q[110];
rz(-pi) q[110];
cx q[100],q[110];
cx q[110],q[100];
cx q[100],q[110];
sx q[111];
rz(0.25268023) q[111];
sx q[111];
sx q[116];
rz(0.11731003) q[116];
sx q[116];
sx q[117];
rz(1.688922) q[117];
sx q[117];
rz(-pi) q[117];
sx q[118];
rz(0.11651103) q[118];
sx q[118];
cx q[110],q[118];
sx q[118];
rz(0.11651103) q[118];
sx q[118];
cx q[117],q[118];
cx q[118],q[117];
cx q[117],q[118];
cx q[117],q[116];
sx q[116];
rz(0.11731003) q[116];
sx q[116];
cx q[116],q[117];
cx q[117],q[116];
cx q[116],q[117];
rz(pi/2) q[118];
sx q[118];
rz(pi/2) q[118];
cx q[117],q[118];
sx q[118];
rz(0.11812563) q[118];
sx q[118];
cx q[110],q[118];
cx q[118],q[110];
cx q[110],q[118];
cx q[100],q[110];
cx q[110],q[100];
cx q[100],q[110];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[102],q[92];
sx q[92];
rz(0.11895853) q[92];
sx q[92];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[84],q[85];
sx q[85];
rz(0.11980933) q[85];
sx q[85];
cx q[85],q[73];
sx q[73];
rz(0.12067853) q[73];
sx q[73];
cx q[73],q[66];
sx q[66];
rz(0.12156703) q[66];
sx q[66];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[98],q[91];
cx q[91],q[98];
cx q[98],q[91];
cx q[98],q[97];
sx q[97];
rz(0.12247543) q[97];
sx q[97];
cx q[96],q[97];
cx q[97],q[96];
cx q[96],q[97];
cx q[96],q[95];
sx q[95];
rz(0.12340443) q[95];
sx q[95];
cx q[94],q[95];
cx q[95],q[94];
cx q[94],q[95];
cx q[90],q[94];
cx q[94],q[90];
cx q[90],q[94];
cx q[75],q[90];
cx q[90],q[75];
cx q[75],q[90];
cx q[75],q[76];
sx q[76];
rz(0.12435503) q[76];
sx q[76];
cx q[76],q[77];
cx q[77],q[76];
cx q[76],q[77];
cx q[71],q[77];
cx q[77],q[71];
cx q[71],q[77];
cx q[71],q[58];
cx q[58],q[71];
cx q[71],q[58];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[43];
sx q[43];
rz(0.12532783) q[43];
sx q[43];
cx q[43],q[44];
sx q[44];
rz(0.12632383) q[44];
sx q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[46];
sx q[46];
rz(0.12734403) q[46];
sx q[46];
cx q[46],q[47];
sx q[47];
rz(0.12838933) q[47];
sx q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[28];
sx q[28];
rz(0.12946073) q[28];
sx q[28];
cx q[28],q[27];
sx q[27];
rz(0.13055953) q[27];
sx q[27];
cx q[27],q[26];
sx q[26];
rz(0.13168673) q[26];
sx q[26];
cx q[16],q[26];
cx q[26],q[16];
cx q[16],q[26];
cx q[16],q[8];
rz(pi/2) q[26];
sx q[26];
rz(pi/2) q[26];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[28],q[35];
cx q[35],q[28];
cx q[28],q[35];
cx q[27],q[28];
cx q[28],q[27];
cx q[27],q[28];
rz(pi/2) q[27];
sx q[27];
rz(pi/2) q[27];
sx q[8];
rz(0.13284363) q[8];
sx q[8];
cx q[8],q[16];
cx q[16],q[8];
cx q[8],q[16];
cx q[16],q[26];
sx q[26];
rz(0.13403153) q[26];
sx q[26];
cx q[26],q[27];
sx q[27];
rz(0.13525193) q[27];
sx q[27];
cx q[99],q[100];
cx q[100],q[99];
cx q[99],q[100];
cx q[100],q[110];
cx q[110],q[100];
cx q[100],q[110];
cx q[118],q[110];
cx q[118],q[117];
cx q[117],q[118];
cx q[118],q[117];
cx q[116],q[117];
cx q[117],q[116];
cx q[116],q[117];
cx q[117],q[116];
cx q[118],q[117];
cx q[110],q[118];
cx q[118],q[110];
cx q[110],q[118];
cx q[100],q[110];
cx q[110],q[100];
cx q[100],q[110];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
rz(pi/2) q[100];
sx q[100];
rz(pi/2) q[100];
cx q[102],q[101];
rz(pi/2) q[110];
sx q[110];
rz(pi/2) q[110];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[84],q[83];
cx q[85],q[84];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
cx q[91],q[79];
cx q[79],q[91];
cx q[91],q[79];
cx q[98],q[91];
cx q[98],q[97];
cx q[97],q[98];
cx q[98],q[97];
cx q[91],q[98];
cx q[96],q[97];
cx q[95],q[96];
cx q[96],q[95];
cx q[95],q[96];
cx q[94],q[95];
cx q[95],q[94];
cx q[94],q[95];
cx q[90],q[94];
cx q[94],q[90];
cx q[90],q[94];
cx q[75],q[90];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[77],q[76];
cx q[76],q[77];
cx q[77],q[76];
cx q[71],q[77];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
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
cx q[60],q[53];
cx q[53],q[60];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[42],q[41];
cx q[43],q[42];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[45],q[44];
cx q[46],q[45];
cx q[45],q[54];
cx q[47],q[46];
cx q[35],q[47];
cx q[28],q[35];
cx q[28],q[27];
cx q[27],q[28];
cx q[28],q[27];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[16],q[26];
cx q[26],q[16];
cx q[16],q[26];
cx q[16],q[8];
cx q[35],q[28];
cx q[28],q[35];
cx q[35],q[28];
cx q[47],q[35];
cx q[35],q[47];
cx q[47],q[35];
cx q[47],q[48];
sx q[48];
rz(0.13650633) q[48];
sx q[48];
cx q[48],q[49];
sx q[49];
rz(0.13779623) q[49];
sx q[49];
cx q[48],q[49];
cx q[49],q[48];
cx q[48],q[49];
cx q[54],q[45];
cx q[45],q[54];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[77],q[76];
cx q[76],q[77];
cx q[77],q[76];
cx q[77],q[78];
cx q[78],q[77];
cx q[77],q[78];
cx q[76],q[77];
cx q[77],q[76];
cx q[76],q[77];
rz(pi/2) q[76];
sx q[76];
rz(pi/2) q[76];
rz(pi/2) q[77];
sx q[77];
rz(pi/2) q[77];
cx q[79],q[78];
cx q[78],q[79];
cx q[79],q[78];
cx q[79],q[80];
cx q[8],q[16];
cx q[26],q[16];
cx q[27],q[26];
cx q[27],q[28];
cx q[28],q[27];
cx q[27],q[28];
cx q[28],q[35];
cx q[35],q[28];
cx q[28],q[35];
cx q[47],q[35];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[45];
sx q[45];
rz(0.13912343) q[45];
sx q[45];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[40],q[39];
sx q[39];
rz(0.14048973) q[39];
sx q[39];
cx q[39],q[33];
sx q[33];
rz(0.14189703) q[33];
sx q[33];
cx q[33],q[20];
sx q[20];
rz(0.14334753) q[20];
sx q[20];
cx q[20],q[33];
cx q[33],q[20];
cx q[20],q[33];
cx q[33],q[39];
cx q[39],q[33];
cx q[33],q[39];
cx q[49],q[48];
cx q[48],q[49];
cx q[49],q[48];
cx q[48],q[49];
cx q[48],q[47];
cx q[47],q[48];
cx q[48],q[47];
cx q[46],q[47];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[40],q[41];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
cx q[33],q[39];
cx q[33],q[20];
cx q[20],q[33];
cx q[33],q[39];
cx q[39],q[33];
cx q[33],q[39];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
cx q[38],q[39];
cx q[39],q[38];
cx q[38],q[39];
cx q[38],q[37];
cx q[37],q[38];
cx q[38],q[37];
cx q[39],q[33];
cx q[33],q[39];
cx q[39],q[33];
cx q[33],q[20];
cx q[20],q[33];
cx q[33],q[20];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[39],q[38];
cx q[38],q[39];
cx q[39],q[38];
cx q[33],q[39];
cx q[39],q[33];
cx q[33],q[39];
cx q[20],q[33];
cx q[33],q[20];
cx q[20],q[33];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[41],q[53];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[52],q[37];
cx q[37],q[52];
cx q[52],q[37];
cx q[37],q[38];
cx q[38],q[37];
cx q[37],q[38];
cx q[38],q[39];
cx q[39],q[38];
cx q[38],q[39];
cx q[33],q[39];
cx q[39],q[33];
cx q[33],q[39];
rz(pi/2) q[33];
sx q[33];
rz(pi/2) q[33];
rz(pi/2) q[39];
sx q[39];
rz(pi/2) q[39];
cx q[53],q[41];
cx q[41],q[53];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[61],q[62];
cx q[61],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
sx q[62];
rz(0.14484353) q[62];
sx q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[63],q[64];
cx q[63],q[62];
sx q[64];
rz(0.14638723) q[64];
sx q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
rz(pi/2) q[66];
sx q[66];
rz(pi/2) q[66];
cx q[67],q[68];
sx q[68];
rz(0.14798143) q[68];
sx q[68];
cx q[68],q[69];
cx q[69],q[68];
cx q[68],q[69];
rz(pi/2) q[68];
sx q[68];
rz(pi/2) q[68];
cx q[69],q[70];
sx q[70];
rz(0.14962893) q[70];
sx q[70];
cx q[70],q[74];
cx q[74],q[70];
cx q[70],q[74];
cx q[74],q[89];
cx q[80],q[79];
cx q[79],q[80];
rz(pi/2) q[79];
sx q[79];
rz(pi/2) q[79];
rz(pi/2) q[80];
sx q[80];
rz(pi/2) q[80];
sx q[89];
rz(0.15133263) q[89];
sx q[89];
cx q[89],q[88];
sx q[88];
rz(0.15309593) q[88];
sx q[88];
cx q[88],q[87];
sx q[87];
rz(0.15492233) q[87];
sx q[87];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[102],q[92];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[92],q[102];
cx q[102],q[92];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[100];
sx q[100];
rz(0.15681573) q[100];
sx q[100];
cx q[100],q[110];
sx q[110];
rz(0.15878023) q[110];
sx q[110];
cx q[100],q[110];
cx q[110],q[100];
cx q[100],q[110];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[110],q[100];
cx q[100],q[110];
cx q[110],q[100];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[102],q[92];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
sx q[92];
rz(0.16082053) q[92];
sx q[92];
cx q[92],q[83];
sx q[83];
rz(0.16294153) q[83];
sx q[83];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[66];
sx q[66];
rz(0.16514873) q[66];
sx q[66];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[65];
cx q[67],q[68];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
sx q[68];
rz(0.16744813) q[68];
sx q[68];
cx q[68],q[55];
sx q[55];
rz(0.16984633) q[55];
sx q[55];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
cx q[48],q[49];
cx q[49],q[48];
cx q[48],q[49];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[46],q[45];
sx q[45];
rz(0.17235063) q[45];
sx q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[34],q[43];
cx q[43],q[34];
cx q[34],q[43];
cx q[34],q[24];
sx q[24];
rz(0.17496903) q[24];
sx q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[27],q[28];
cx q[28],q[27];
cx q[27],q[28];
cx q[28],q[29];
cx q[29],q[28];
cx q[28],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[17],q[30];
cx q[30],q[17];
cx q[17],q[30];
cx q[17],q[12];
sx q[12];
rz(0.17771063) q[12];
sx q[12];
cx q[12],q[11];
sx q[11];
rz(0.18058523) q[11];
sx q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[17];
cx q[17],q[12];
cx q[12],q[17];
rz(pi/2) q[30];
sx q[30];
rz(pi/2) q[30];
cx q[17],q[30];
sx q[30];
rz(0.18360403) q[30];
sx q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[29],q[28];
cx q[28],q[29];
cx q[29],q[28];
cx q[35],q[28];
cx q[28],q[35];
cx q[35],q[28];
cx q[35],q[47];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
rz(pi/2) q[43];
sx q[43];
rz(pi/2) q[43];
cx q[47],q[35];
cx q[35],q[47];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
rz(pi/2) q[46];
sx q[46];
rz(pi/2) q[46];
cx q[47],q[46];
sx q[46];
rz(0.18677943) q[46];
sx q[46];
cx q[46],q[45];
sx q[45];
rz(0.19012563) q[45];
sx q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[43];
sx q[43];
rz(0.19365833) q[43];
sx q[43];
cx q[43],q[34];
cx q[34],q[43];
cx q[43],q[34];
cx q[34],q[24];
sx q[24];
rz(0.19739553) q[24];
sx q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[22];
sx q[22];
rz(0.20135793) q[22];
sx q[22];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[22],q[23];
sx q[23];
rz(0.20556893) q[23];
sx q[23];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[67],q[68];
cx q[68],q[67];
cx q[67],q[68];
cx q[69],q[68];
cx q[69],q[70];
cx q[70],q[69];
cx q[69],q[70];
cx q[74],q[70];
cx q[89],q[74];
cx q[88],q[89];
cx q[87],q[88];
cx q[88],q[87];
cx q[87],q[88];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[93],q[106];
cx q[106],q[93];
cx q[93],q[106];
cx q[106],q[105];
cx q[105],q[106];
cx q[106],q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[105],q[104];
cx q[104],q[103];
cx q[103],q[104];
cx q[104],q[103];
cx q[102],q[103];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[102],q[101];
rz(pi/2) q[104];
sx q[104];
rz(pi/2) q[104];
rz(pi/2) q[105];
sx q[105];
rz(pi/2) q[105];
rz(pi/2) q[106];
sx q[106];
rz(pi/2) q[106];
cx q[92],q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
cx q[66],q[73];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[45],q[54];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
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
cx q[29],q[30];
cx q[30],q[29];
cx q[17],q[30];
cx q[30],q[17];
cx q[17],q[30];
cx q[12],q[17];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[17];
cx q[17],q[12];
cx q[12],q[17];
cx q[30],q[17];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[29],q[28];
cx q[28],q[29];
cx q[29],q[28];
cx q[28],q[35];
rz(pi/2) q[30];
sx q[30];
rz(pi/2) q[30];
cx q[35],q[28];
cx q[28],q[35];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[46],q[47];
cx q[45],q[46];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[43],q[44];
cx q[34],q[43];
cx q[24],q[34];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[15];
cx q[15],q[22];
cx q[22],q[15];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[28],q[27];
cx q[27],q[28];
cx q[28],q[27];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[28],q[29];
cx q[29],q[28];
cx q[28],q[29];
cx q[27],q[28];
cx q[28],q[27];
cx q[27],q[28];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[24],q[34];
cx q[29],q[30];
cx q[29],q[28];
sx q[30];
rz(0.21005573) q[30];
sx q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[29],q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[31],q[32];
cx q[31],q[30];
sx q[32];
rz(0.21484983) q[32];
sx q[32];
cx q[32],q[36];
cx q[32],q[31];
cx q[34],q[24];
cx q[24],q[34];
cx q[34],q[43];
sx q[36];
rz(0.21998803) q[36];
sx q[36];
cx q[36],q[51];
cx q[36],q[32];
cx q[43],q[34];
cx q[34],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[53];
sx q[51];
rz(0.22551343) q[51];
sx q[51];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[36],q[51];
cx q[49],q[50];
cx q[50],q[49];
cx q[49],q[50];
cx q[51],q[36];
cx q[36],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[53],q[41];
cx q[41],q[53];
cx q[53],q[60];
cx q[55],q[49];
cx q[49],q[55];
cx q[55],q[49];
cx q[50],q[49];
cx q[49],q[50];
cx q[50],q[49];
cx q[55],q[68];
cx q[60],q[53];
cx q[53],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[68],q[55];
cx q[55],q[68];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
cx q[68],q[69];
cx q[69],q[68];
cx q[68],q[69];
cx q[55],q[68];
cx q[68],q[55];
cx q[55],q[68];
cx q[69],q[70];
cx q[70],q[69];
cx q[69],q[70];
cx q[68],q[69];
cx q[69],q[68];
cx q[68],q[69];
cx q[70],q[74];
cx q[74],q[70];
cx q[70],q[74];
cx q[69],q[70];
cx q[70],q[69];
cx q[69],q[70];
cx q[74],q[89];
cx q[89],q[74];
cx q[74],q[89];
cx q[70],q[74];
cx q[74],q[70];
cx q[70],q[74];
cx q[89],q[88];
cx q[88],q[89];
cx q[89],q[88];
cx q[74],q[89];
cx q[88],q[87];
cx q[87],q[88];
cx q[88],q[87];
cx q[87],q[93];
cx q[89],q[74];
cx q[74],q[89];
cx q[89],q[88];
cx q[88],q[89];
cx q[89],q[88];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[93],q[87];
cx q[87],q[93];
cx q[88],q[87];
cx q[87],q[88];
cx q[88],q[87];
cx q[93],q[106];
sx q[106];
rz(0.23147733) q[106];
sx q[106];
cx q[106],q[105];
sx q[105];
rz(0.23794113) q[105];
sx q[105];
cx q[105],q[104];
sx q[104];
rz(0.24497863) q[104];
sx q[104];
cx q[104],q[111];
sx q[111];
rz(0.25268023) q[111];
sx q[111];
cx q[104],q[111];
cx q[111],q[104];
cx q[104],q[111];
cx q[103],q[104];
cx q[104],q[103];
cx q[103],q[104];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
cx q[102],q[92];
cx q[104],q[111];
cx q[111],q[104];
cx q[104],q[111];
sx q[92];
rz(0.26115743) q[92];
sx q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[80];
sx q[80];
rz(0.27054973) q[80];
sx q[80];
cx q[80],q[79];
sx q[79];
rz(0.28103493) q[79];
sx q[79];
cx q[93],q[87];
cx q[106],q[93];
cx q[105],q[106];
cx q[104],q[105];
cx q[104],q[103];
cx q[103],q[104];
cx q[104],q[103];
cx q[102],q[103];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[98],q[91];
cx q[91],q[98];
rz(pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[79],q[91];
sx q[91];
rz(0.29284273) q[91];
sx q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[91],q[79];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[62],q[61];
sx q[61];
rz(0.30627733) q[61];
sx q[61];
cx q[61],q[60];
sx q[60];
rz(0.32175053) q[60];
sx q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[34];
cx q[34],q[43];
cx q[43],q[34];
cx q[34],q[24];
cx q[24],q[34];
cx q[34],q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[23],q[22];
sx q[22];
rz(0.33983693) q[22];
sx q[22];
cx q[22],q[21];
sx q[21];
rz(0.36136713) q[21];
sx q[21];
cx q[21],q[20];
sx q[20];
rz(0.38759673) q[20];
sx q[20];
cx q[20],q[33];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[24],q[34];
sx q[33];
rz(0.42053433) q[33];
sx q[33];
cx q[33],q[39];
cx q[34],q[24];
cx q[24],q[34];
cx q[34],q[43];
sx q[39];
rz(0.46364763) q[39];
sx q[39];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
cx q[43],q[34];
cx q[34],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[80],q[81];
cx q[79],q[80];
cx q[91],q[79];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[61],q[62];
cx q[60],q[61];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[53];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[34];
cx q[34],q[43];
cx q[43],q[34];
cx q[34],q[24];
cx q[24],q[34];
cx q[34],q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[22],q[23];
cx q[21],q[22];
cx q[20],q[21];
cx q[33],q[20];
cx q[33],q[39];
cx q[39],q[33];
cx q[33],q[39];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
cx q[53],q[41];
cx q[41],q[53];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[53],q[60];
cx q[53],q[41];
sx q[60];
rz(pi/6) q[60];
sx q[60];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[60];
cx q[53],q[60];
cx q[58],q[59];
cx q[59],q[58];
cx q[58],q[59];
cx q[58],q[71];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[71],q[58];
cx q[58],q[71];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[71],q[77];
cx q[71],q[58];
sx q[77];
rz(0.61547971) q[77];
sx q[77];
cx q[77],q[76];
sx q[76];
rz(pi/4) q[76];
sx q[76];
cx q[77],q[71];
cx q[76],q[77];
barrier q[80],q[56],q[1],q[120],q[44],q[29],q[42],q[34],q[105],q[51],q[113],q[78],q[3],q[122],q[67],q[11],q[111],q[40],q[90],q[65],q[88],q[93],q[115],q[39],q[124],q[46],q[33],q[16],q[95],q[26],q[72],q[37],q[108],q[75],q[102],q[10],q[55],q[19],q[85],q[25],q[86],q[41],q[99],q[50],q[104],q[17],q[64],q[58],q[84],q[31],q[96],q[38],q[103],q[52],q[112],q[5],q[54],q[14],q[118],q[4],q[100],q[32],q[62],q[35],q[106],q[92],q[7],q[126],q[59],q[27],q[63],q[30],q[70],q[43],q[83],q[45],q[0],q[119],q[71],q[9],q[98],q[18],q[79],q[8],q[76],q[57],q[2],q[121],q[97],q[47],q[94],q[81],q[73],q[89],q[36],q[114],q[53],q[15],q[123],q[68],q[13],q[77],q[22],q[74],q[12],q[107],q[20],q[110],q[82],q[6],q[125],q[69],q[28],q[91],q[87],q[116],q[60],q[109],q[49],q[117],q[66],q[24],q[61],q[23],q[21],q[101],q[48];
measure q[76] -> meas[0];
measure q[77] -> meas[1];
measure q[71] -> meas[2];
measure q[58] -> meas[3];
measure q[41] -> meas[4];
measure q[20] -> meas[5];
measure q[21] -> meas[6];
measure q[22] -> meas[7];
measure q[23] -> meas[8];
measure q[61] -> meas[9];
measure q[62] -> meas[10];
measure q[72] -> meas[11];
measure q[91] -> meas[12];
measure q[79] -> meas[13];
measure q[80] -> meas[14];
measure q[103] -> meas[15];
measure q[105] -> meas[16];
measure q[106] -> meas[17];
measure q[93] -> meas[18];
measure q[87] -> meas[19];
measure q[32] -> meas[20];
measure q[31] -> meas[21];
measure q[30] -> meas[22];
measure q[28] -> meas[23];
measure q[15] -> meas[24];
measure q[34] -> meas[25];
measure q[43] -> meas[26];
measure q[44] -> meas[27];
measure q[46] -> meas[28];
measure q[47] -> meas[29];
measure q[17] -> meas[30];
measure q[11] -> meas[31];
measure q[12] -> meas[32];
measure q[45] -> meas[33];
measure q[54] -> meas[34];
measure q[67] -> meas[35];
measure q[73] -> meas[36];
measure q[85] -> meas[37];
measure q[104] -> meas[38];
measure q[101] -> meas[39];
measure q[100] -> meas[40];
measure q[111] -> meas[41];
measure q[70] -> meas[42];
measure q[69] -> meas[43];
measure q[68] -> meas[44];
measure q[49] -> meas[45];
measure q[66] -> meas[46];
measure q[81] -> meas[47];
measure q[40] -> meas[48];
measure q[38] -> meas[49];
measure q[37] -> meas[50];
measure q[52] -> meas[51];
measure q[48] -> meas[52];
measure q[50] -> meas[53];
measure q[27] -> meas[54];
measure q[24] -> meas[55];
measure q[16] -> meas[56];
measure q[8] -> meas[57];
measure q[25] -> meas[58];
measure q[26] -> meas[59];
measure q[51] -> meas[60];
measure q[65] -> meas[61];
measure q[35] -> meas[62];
measure q[64] -> meas[63];
measure q[42] -> meas[64];
measure q[90] -> meas[65];
measure q[97] -> meas[66];
measure q[98] -> meas[67];
measure q[84] -> meas[68];
measure q[86] -> meas[69];
measure q[102] -> meas[70];
measure q[110] -> meas[71];
measure q[117] -> meas[72];
measure q[116] -> meas[73];
measure q[118] -> meas[74];
measure q[78] -> meas[75];
measure q[75] -> meas[76];
measure q[60] -> meas[77];
measure q[33] -> meas[78];
measure q[63] -> meas[79];
measure q[92] -> meas[80];
measure q[82] -> meas[81];
measure q[39] -> meas[82];
measure q[53] -> meas[83];
