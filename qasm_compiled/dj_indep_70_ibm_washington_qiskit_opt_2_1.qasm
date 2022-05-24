OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg c[69];
rz(-pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/2) q[11];
sx q[11];
rz(-pi) q[11];
rz(-3*pi/2) q[12];
sx q[12];
rz(-pi/2) q[12];
cx q[11],q[12];
sx q[11];
rz(-pi/2) q[11];
cx q[10],q[11];
cx q[11],q[10];
cx q[10],q[11];
cx q[9],q[10];
cx q[10],q[9];
cx q[9],q[10];
rz(-pi/2) q[13];
sx q[13];
rz(-pi) q[13];
rz(-pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[12],q[17];
cx q[17],q[12];
cx q[12],q[17];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(-pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(-pi/2) q[26];
sx q[26];
rz(pi/2) q[26];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
rz(-pi/2) q[27];
sx q[27];
rz(-pi) q[27];
rz(-pi/2) q[28];
sx q[28];
rz(pi/2) q[28];
rz(-pi/2) q[29];
sx q[29];
rz(pi/2) q[29];
rz(-pi/2) q[30];
sx q[30];
rz(pi/2) q[30];
cx q[17],q[30];
cx q[30],q[17];
rz(-3*pi/2) q[17];
sx q[17];
rz(-pi/2) q[17];
rz(-pi/2) q[31];
sx q[31];
rz(-pi) q[31];
cx q[31],q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[29],q[28];
cx q[28],q[29];
cx q[29],q[28];
cx q[27],q[28];
sx q[27];
rz(-pi/2) q[27];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
sx q[31];
rz(-pi/2) q[31];
rz(-pi/2) q[34];
sx q[34];
rz(pi/2) q[34];
rz(-pi/2) q[35];
sx q[35];
rz(pi/2) q[35];
cx q[28],q[35];
cx q[35],q[28];
cx q[28],q[35];
rz(-pi/2) q[38];
sx q[38];
rz(pi/2) q[38];
rz(pi/2) q[39];
sx q[39];
rz(pi/2) q[39];
rz(-pi/2) q[40];
sx q[40];
rz(pi/2) q[40];
rz(-pi/2) q[41];
sx q[41];
rz(pi/2) q[41];
rz(pi/2) q[42];
sx q[42];
rz(pi/2) q[42];
rz(pi/2) q[43];
sx q[43];
rz(pi/2) q[43];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
cx q[38],q[39];
cx q[39],q[38];
cx q[38],q[39];
rz(pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
rz(-pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
rz(pi/2) q[46];
sx q[46];
rz(pi/2) q[46];
rz(pi/2) q[47];
sx q[47];
rz(pi/2) q[47];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
rz(-pi/2) q[48];
sx q[48];
rz(-pi) q[48];
cx q[48],q[47];
sx q[48];
rz(-pi/2) q[48];
rz(-pi/2) q[49];
sx q[49];
rz(pi/2) q[49];
cx q[48],q[49];
cx q[49],q[48];
cx q[48],q[49];
cx q[48],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[44],q[45];
rz(pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[44],q[45];
rz(-3*pi/2) q[44];
sx q[44];
rz(-pi/2) q[44];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[45];
rz(-3*pi/2) q[44];
sx q[44];
rz(-pi/2) q[44];
rz(-3*pi/2) q[48];
sx q[48];
rz(-pi/2) q[48];
rz(-pi/2) q[50];
sx q[50];
rz(pi/2) q[50];
rz(pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[39],q[40];
cx q[40],q[39];
cx q[39],q[40];
cx q[39],q[38];
cx q[38],q[39];
cx q[39],q[38];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[41],q[53];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[44],q[45];
rz(pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[34],q[43];
cx q[43],q[34];
cx q[34],q[43];
cx q[44],q[45];
rz(pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[45];
rz(-3*pi/2) q[44];
sx q[44];
rz(-pi/2) q[44];
cx q[53],q[41];
cx q[41],q[53];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[40],q[39];
cx q[39],q[40];
cx q[40],q[39];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
rz(-pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(-pi/2) q[55];
sx q[55];
rz(pi/2) q[55];
rz(-pi/2) q[58];
sx q[58];
rz(pi/2) q[58];
rz(pi/2) q[59];
sx q[59];
rz(pi/2) q[59];
rz(-pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[60];
cx q[58],q[59];
cx q[59],q[58];
cx q[58],q[59];
rz(-pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[60];
cx q[59],q[58];
cx q[58],q[59];
cx q[59],q[58];
cx q[61],q[62];
rz(-3*pi/2) q[61];
sx q[61];
rz(-pi/2) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[61],q[62];
rz(-3*pi/2) q[61];
sx q[61];
rz(-pi/2) q[61];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(pi/2) q[65];
sx q[65];
rz(pi/2) q[65];
rz(pi/2) q[66];
sx q[66];
rz(pi/2) q[66];
rz(pi/2) q[67];
sx q[67];
rz(pi) q[67];
rz(pi/2) q[68];
sx q[68];
rz(pi/2) q[68];
rz(-pi/2) q[69];
sx q[69];
rz(pi/2) q[69];
cx q[68],q[69];
cx q[69],q[68];
cx q[68],q[69];
cx q[55],q[68];
cx q[68],q[55];
cx q[55],q[68];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
cx q[48],q[49];
cx q[49],q[48];
cx q[48],q[49];
cx q[49],q[50];
cx q[50],q[49];
cx q[49],q[50];
cx q[55],q[68];
cx q[68],q[55];
cx q[55],q[68];
cx q[68],q[69];
cx q[69],q[68];
cx q[68],q[69];
rz(pi/2) q[72];
sx q[72];
rz(pi) q[72];
cx q[72],q[62];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(pi/2) q[73];
sx q[73];
rz(pi) q[73];
rz(-pi/2) q[78];
sx q[78];
rz(pi/2) q[78];
rz(-pi/2) q[79];
sx q[79];
rz(pi/2) q[79];
rz(pi/2) q[80];
sx q[80];
rz(pi) q[80];
rz(pi/2) q[81];
sx q[81];
rz(pi) q[81];
cx q[81],q[72];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[80],q[81];
sx q[80];
rz(pi/2) q[80];
cx q[79],q[80];
cx q[80],q[79];
cx q[79],q[80];
cx q[78],q[79];
cx q[79],q[78];
cx q[78],q[79];
cx q[80],q[81];
cx q[72],q[81];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
rz(-3*pi/2) q[80];
sx q[80];
rz(-pi/2) q[80];
cx q[79],q[80];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[81];
rz(-3*pi/2) q[80];
sx q[80];
rz(-pi/2) q[80];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(-pi/2) q[84];
sx q[84];
rz(pi/2) q[84];
rz(pi/2) q[85];
sx q[85];
rz(pi/2) q[85];
rz(pi/2) q[86];
sx q[86];
rz(pi/2) q[86];
rz(pi/2) q[87];
sx q[87];
rz(pi/2) q[87];
rz(-pi/2) q[91];
sx q[91];
rz(pi/2) q[91];
cx q[79],q[91];
cx q[91],q[79];
cx q[79],q[91];
cx q[80],q[79];
cx q[79],q[80];
cx q[80],q[79];
rz(-pi/2) q[92];
sx q[92];
rz(-pi) q[92];
cx q[92],q[83];
sx q[92];
rz(-pi/2) q[92];
rz(pi/2) q[98];
sx q[98];
rz(pi/2) q[98];
rz(pi/2) q[99];
sx q[99];
rz(pi/2) q[99];
rz(-pi/2) q[100];
sx q[100];
rz(pi/2) q[100];
rz(pi/2) q[101];
sx q[101];
rz(pi/2) q[101];
rz(-pi/2) q[102];
sx q[102];
rz(pi/2) q[102];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[92],q[83];
rz(-3*pi/2) q[92];
sx q[92];
rz(-pi/2) q[92];
rz(pi/2) q[103];
sx q[103];
rz(pi/2) q[103];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[92],q[83];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
rz(-pi/2) q[104];
sx q[104];
rz(pi/2) q[104];
cx q[103],q[104];
cx q[104],q[103];
cx q[103],q[104];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[92],q[83];
rz(-3*pi/2) q[92];
sx q[92];
rz(-pi/2) q[92];
rz(-pi/2) q[105];
sx q[105];
rz(pi/2) q[105];
cx q[104],q[105];
cx q[105],q[104];
cx q[104],q[105];
cx q[103],q[104];
cx q[104],q[103];
cx q[103],q[104];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[92],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
rz(-3*pi/2) q[92];
sx q[92];
rz(-pi/2) q[92];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[99],q[100];
cx q[100],q[99];
cx q[99],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[98],q[99];
cx q[99],q[98];
cx q[98],q[99];
rz(-pi/2) q[106];
sx q[106];
rz(pi/2) q[106];
cx q[93],q[106];
cx q[106],q[93];
cx q[93],q[106];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[86],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[83],q[84];
rz(-3*pi/2) q[83];
sx q[83];
rz(-pi/2) q[83];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
rz(-3*pi/2) q[86];
sx q[86];
rz(-pi/2) q[86];
cx q[87],q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[92],q[83];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[92],q[83];
rz(-3*pi/2) q[92];
sx q[92];
rz(-pi/2) q[92];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[83];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[93],q[87];
cx q[87],q[93];
cx q[93],q[87];
rz(-pi/2) q[110];
sx q[110];
rz(pi/2) q[110];
cx q[100],q[110];
cx q[110],q[100];
cx q[100],q[110];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[92],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
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
cx q[85],q[84];
rz(pi/2) q[85];
sx q[85];
rz(pi/2) q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[86],q[85];
cx q[84],q[85];
rz(pi/2) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(pi/2) q[86];
sx q[86];
rz(pi/2) q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[86],q[85];
cx q[84],q[85];
cx q[73],q[85];
sx q[73];
rz(pi/2) q[73];
rz(pi/2) q[84];
sx q[84];
rz(pi/2) q[84];
rz(pi/2) q[86];
sx q[86];
rz(pi/2) q[86];
rz(-3*pi/2) q[92];
sx q[92];
rz(-pi/2) q[92];
cx q[99],q[100];
cx q[100],q[99];
cx q[99],q[100];
cx q[100],q[101];
cx q[101],q[100];
cx q[100],q[101];
cx q[101],q[102];
cx q[102],q[101];
cx q[101],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[84],q[85];
rz(pi/2) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[67],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[45],q[54];
rz(-3*pi/2) q[45];
sx q[45];
rz(-pi/2) q[45];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[64],q[54];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[64],q[54];
rz(-3*pi/2) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[44],q[45];
rz(pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[48],q[47];
cx q[46],q[47];
rz(-3*pi/2) q[46];
sx q[46];
rz(-pi/2) q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[46],q[47];
rz(pi/2) q[46];
sx q[46];
rz(pi/2) q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[46],q[47];
rz(-3*pi/2) q[46];
sx q[46];
rz(-pi/2) q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[46],q[47];
rz(-3*pi/2) q[46];
sx q[46];
rz(-pi/2) q[46];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[47],q[35];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[28];
cx q[28],q[35];
cx q[35],q[28];
cx q[27],q[28];
rz(pi/2) q[27];
sx q[27];
rz(pi/2) q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[35],q[28];
rz(pi/2) q[35];
sx q[35];
rz(pi/2) q[35];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[47],q[35];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[28];
rz(pi/2) q[35];
sx q[35];
rz(pi/2) q[35];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[28];
cx q[27],q[28];
rz(-3*pi/2) q[27];
sx q[27];
rz(-pi/2) q[27];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[27];
rz(-3*pi/2) q[35];
sx q[35];
rz(-pi/2) q[35];
rz(-3*pi/2) q[48];
sx q[48];
rz(-pi/2) q[48];
cx q[49],q[48];
cx q[48],q[49];
cx q[49],q[48];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[35],q[28];
cx q[29],q[28];
rz(-3*pi/2) q[29];
sx q[29];
rz(-pi/2) q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[29],q[28];
cx q[27],q[28];
rz(pi/2) q[27];
sx q[27];
rz(pi/2) q[27];
rz(-3*pi/2) q[29];
sx q[29];
rz(-pi/2) q[29];
rz(-3*pi/2) q[35];
sx q[35];
rz(-pi/2) q[35];
cx q[55],q[49];
cx q[49],q[55];
cx q[55],q[49];
cx q[48],q[49];
cx q[49],q[48];
cx q[48],q[49];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[35],q[28];
rz(-3*pi/2) q[35];
sx q[35];
rz(-pi/2) q[35];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
sx q[67];
rz(pi/2) q[67];
cx q[68],q[55];
cx q[55],q[68];
cx q[68],q[55];
cx q[49],q[55];
cx q[55],q[49];
cx q[49],q[55];
cx q[48],q[49];
cx q[49],q[48];
cx q[48],q[49];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[35],q[28];
cx q[28],q[29];
cx q[29],q[28];
cx q[28],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[17];
cx q[17],q[30];
cx q[30],q[17];
cx q[17],q[12];
cx q[12],q[17];
cx q[13],q[12];
cx q[11],q[12];
rz(-3*pi/2) q[11];
sx q[11];
rz(-pi/2) q[11];
cx q[10],q[11];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[12];
rz(-3*pi/2) q[11];
sx q[11];
rz(-pi/2) q[11];
sx q[13];
rz(-pi/2) q[13];
rz(-3*pi/2) q[17];
sx q[17];
rz(-pi/2) q[17];
cx q[12],q[17];
cx q[17],q[12];
cx q[12],q[17];
cx q[17],q[30];
cx q[30],q[17];
cx q[17],q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[29],q[28];
cx q[28],q[29];
cx q[29],q[28];
cx q[28],q[27];
cx q[27],q[28];
cx q[28],q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[23],q[24];
rz(-3*pi/2) q[23];
sx q[23];
rz(-pi/2) q[23];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[24];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[35];
sx q[35];
rz(pi/2) q[35];
rz(pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
barrier q[30],q[105],q[37],q[99],q[55],q[102],q[47],q[0],q[119],q[81],q[21],q[82],q[17],q[94],q[40],q[103],q[69],q[112],q[57],q[121],q[14],q[79],q[26],q[86],q[32],q[96],q[60],q[110],q[48],q[114],q[7],q[71],q[16],q[78],q[28],q[89],q[53],q[84],q[43],q[107],q[11],q[85],q[18],q[83],q[25],q[65],q[36],q[100],q[44],q[109],q[2],q[73],q[9],q[75],q[20],q[80],q[29],q[106],q[42],q[104],q[58],q[4],q[123],q[35],q[13],q[77],q[23],q[87],q[31],q[95],q[52],q[116],q[59],q[6],q[125],q[70],q[15],q[91],q[22],q[88],q[54],q[118],q[64],q[8],q[62],q[12],q[72],q[90],q[46],q[111],q[56],q[1],q[120],q[63],q[10],q[74],q[19],q[92],q[39],q[98],q[50],q[113],q[61],q[3],q[122],q[67],q[24],q[76],q[33],q[97],q[38],q[93],q[51],q[115],q[45],q[5],q[124],q[68],q[27],q[49],q[101],q[41],q[108],q[34],q[117],q[66],q[126];
measure q[9] -> c[0];
measure q[31] -> c[1];
measure q[17] -> c[2];
measure q[25] -> c[3];
measure q[69] -> c[4];
measure q[50] -> c[5];
measure q[38] -> c[6];
measure q[60] -> c[7];
measure q[39] -> c[8];
measure q[34] -> c[9];
measure q[40] -> c[10];
measure q[53] -> c[11];
measure q[58] -> c[12];
measure q[59] -> c[13];
measure q[66] -> c[14];
measure q[61] -> c[15];
measure q[62] -> c[16];
measure q[72] -> c[17];
measure q[78] -> c[18];
measure q[91] -> c[19];
measure q[81] -> c[20];
measure q[79] -> c[21];
measure q[105] -> c[22];
measure q[104] -> c[23];
measure q[103] -> c[24];
measure q[98] -> c[25];
measure q[110] -> c[26];
measure q[93] -> c[27];
measure q[80] -> c[28];
measure q[99] -> c[29];
measure q[100] -> c[30];
measure q[101] -> c[31];
measure q[102] -> c[32];
measure q[82] -> c[33];
measure q[87] -> c[34];
measure q[92] -> c[35];
measure q[86] -> c[36];
measure q[83] -> c[37];
measure q[85] -> c[38];
measure q[84] -> c[39];
measure q[67] -> c[40];
measure q[73] -> c[41];
measure q[54] -> c[42];
measure q[63] -> c[43];
measure q[65] -> c[44];
measure q[64] -> c[45];
measure q[41] -> c[46];
measure q[68] -> c[47];
measure q[42] -> c[48];
measure q[43] -> c[49];
measure q[44] -> c[50];
measure q[45] -> c[51];
measure q[26] -> c[52];
measure q[46] -> c[53];
measure q[55] -> c[54];
measure q[49] -> c[55];
measure q[27] -> c[56];
measure q[48] -> c[57];
measure q[30] -> c[58];
measure q[29] -> c[59];
measure q[28] -> c[60];
measure q[47] -> c[61];
measure q[35] -> c[62];
measure q[13] -> c[63];
measure q[10] -> c[64];
measure q[12] -> c[65];
measure q[11] -> c[66];
measure q[22] -> c[67];
measure q[23] -> c[68];
