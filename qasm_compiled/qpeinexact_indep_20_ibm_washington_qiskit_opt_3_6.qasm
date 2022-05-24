OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg c[19];
rz(pi/2) q[41];
sx q[41];
rz(pi/2) q[41];
rz(pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
x q[62];
rz(-0.00515024) q[62];
cx q[62],q[61];
rz(1.3464786) q[61];
cx q[62],q[61];
rz(-1.3464786) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(pi/2) q[65];
sx q[65];
rz(pi/2) q[65];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
rz(pi/2) q[66];
sx q[66];
rz(pi/2) q[66];
rz(pi/2) q[67];
sx q[67];
rz(pi/2) q[67];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
rz(-0.44863545) q[72];
cx q[62],q[72];
cx q[62],q[63];
rz(-0.8972709) q[63];
cx q[62],q[63];
rz(0.8972709) q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
rz(1.34705085) q[63];
cx q[62],q[63];
rz(-1.34705085) q[63];
rz(0.44863545) q[72];
rz(pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
rz(-0.44749096) q[72];
cx q[62],q[72];
cx q[62],q[61];
rz(-0.8949819) q[61];
cx q[62],q[61];
rz(0.8949819) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
rz(0.44749096) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
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
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(-4.0548864) q[83];
rz(-pi) q[84];
sx q[84];
rz(pi/2) q[85];
sx q[85];
rz(pi/2) q[85];
rz(pi/2) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[83],q[92];
rz(1.3516288) q[92];
cx q[83],q[92];
sx q[83];
cx q[84],q[83];
rz(pi/2) q[83];
sx q[84];
rz(-pi) q[84];
cx q[84],q[83];
rz(1.1324613) q[83];
sx q[84];
cx q[84],q[83];
rz(0.43833501) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(1.4243012) q[84];
cx q[84],q[85];
rz(-0.87667) q[85];
cx q[84],q[85];
cx q[84],q[83];
rz(1.3882526) q[83];
cx q[84],q[83];
rz(-1.3882526) q[83];
rz(0.87667) q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[84],q[85];
rz(-0.36508743) q[85];
cx q[84],q[85];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
rz(2.1905246) q[83];
cx q[83],q[82];
rz(-0.73017485) q[82];
cx q[83],q[82];
rz(0.73017485) q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(0.36508743) q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
rz(-0.018407769) q[73];
rz(-1.3516288) q[92];
rz(pi/2) q[102];
sx q[102];
rz(pi/2) q[102];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[83],q[92];
rz(-1.4603497) q[92];
cx q[83],q[92];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
rz(-0.220893235) q[62];
cx q[62],q[61];
rz(0.220893235) q[61];
cx q[62],q[61];
rz(-0.220893235) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[62],q[72];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
rz(pi/4) q[54];
sx q[54];
rz(-pi) q[54];
rz(-1.3253594) q[65];
cx q[65],q[66];
rz(0.441786465) q[66];
cx q[65],q[66];
cx q[65],q[64];
rz(0.88357293382213) q[64];
cx q[65],q[64];
rz(-0.883572933822129) q[64];
rz(-0.441786465) q[66];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
rz(7*pi/16) q[66];
cx q[66],q[67];
rz(-7*pi/16) q[67];
cx q[66],q[67];
rz(7*pi/16) q[67];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
rz(-3*pi/8) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[64],q[54];
rz(-pi/2) q[54];
sx q[64];
rz(-pi) q[64];
cx q[64],q[54];
rz(3*pi/8) q[54];
sx q[64];
cx q[64],q[54];
x q[54];
rz(-3*pi/4) q[54];
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
rz(-pi/4) q[42];
cx q[42],q[41];
rz(pi/4) q[41];
cx q[42],q[41];
rz(-pi/4) q[41];
cx q[53],q[41];
cx q[41],q[53];
cx q[53],q[41];
rz(-pi/524288) q[41];
sx q[41];
rz(-pi) q[41];
rz(pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
rz(-0.3436117) q[60];
rz(-7*pi/8) q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
rz(-7.1905349e-05) q[67];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[62],q[72];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(-pi/8) q[63];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
rz(-4.6142142) q[64];
cx q[72],q[62];
cx q[62],q[72];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(-3*pi/16) q[61];
rz(-5*pi/4) q[72];
cx q[72],q[62];
rz(pi/4) q[62];
cx q[72],q[62];
rz(-pi/4) q[62];
cx q[63],q[62];
rz(pi/8) q[62];
cx q[63],q[62];
rz(-pi/8) q[62];
cx q[61],q[62];
rz(pi/16) q[62];
cx q[61],q[62];
rz(-pi/16) q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(1.3230994) q[62];
sx q[62];
cx q[64],q[63];
rz(pi/32) q[63];
cx q[64],q[63];
rz(-pi/32) q[63];
sx q[72];
rz(-pi) q[72];
cx q[72],q[62];
rz(pi/2) q[62];
sx q[72];
rz(-pi) q[72];
cx q[72],q[62];
rz(pi/4) q[62];
sx q[72];
cx q[72],q[62];
x q[62];
rz(3*pi/4) q[62];
cx q[61],q[62];
rz(pi/8) q[62];
cx q[61],q[62];
rz(-pi/8) q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[60],q[61];
rz(pi/64) q[61];
cx q[60],q[61];
rz(-pi/64) q[61];
rz(pi/4) q[62];
cx q[64],q[63];
rz(pi/16) q[63];
cx q[64],q[63];
rz(-pi/16) q[63];
rz(-2.1084975) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
rz(pi/4) q[72];
cx q[62],q[72];
sx q[62];
rz(pi/2) q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(-pi/4) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
rz(pi/32) q[61];
cx q[60],q[61];
rz(-pi/32) q[61];
cx q[64],q[63];
rz(pi/8) q[63];
cx q[64],q[63];
rz(-pi/8) q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[60],q[61];
rz(pi/16) q[61];
cx q[60],q[61];
rz(-pi/16) q[61];
cx q[64],q[63];
rz(pi/4) q[63];
cx q[64],q[63];
rz(-pi/4) q[63];
sx q[64];
rz(-pi) q[64];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(-pi/262144) q[53];
sx q[53];
rz(-pi) q[53];
rz(-pi/16384) q[60];
rz(-pi/8) q[61];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(-pi/256) q[81];
rz(1.4603497) q[92];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
rz(-pi/128) q[83];
cx q[83],q[82];
rz(pi/128) q[82];
cx q[83],q[82];
rz(-pi/128) q[82];
cx q[81],q[82];
rz(pi/256) q[82];
cx q[81],q[82];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(-pi/128) q[72];
rz(-pi/256) q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(-pi/64) q[82];
cx q[82],q[81];
rz(pi/64) q[81];
cx q[82],q[81];
rz(-pi/64) q[81];
cx q[72],q[81];
rz(pi/128) q[81];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(-pi/64) q[62];
rz(-pi/128) q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(-pi/32) q[81];
cx q[81],q[72];
rz(pi/32) q[72];
cx q[81],q[72];
rz(-pi/32) q[72];
cx q[62],q[72];
rz(pi/64) q[72];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
rz(pi/8) q[62];
cx q[61],q[62];
rz(-pi/8) q[62];
rz(-pi/32) q[63];
rz(-pi/64) q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(-3*pi/16) q[72];
cx q[72],q[62];
rz(pi/16) q[62];
cx q[72],q[62];
rz(-pi/16) q[62];
cx q[63],q[62];
rz(pi/32) q[62];
cx q[63],q[62];
rz(-pi/32) q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(1.3230994) q[63];
sx q[63];
cx q[64],q[63];
rz(pi/2) q[63];
sx q[64];
rz(-pi) q[64];
cx q[64],q[63];
rz(pi/4) q[63];
sx q[64];
cx q[64],q[63];
x q[63];
rz(3*pi/4) q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-pi/16) q[63];
rz(-0.53770119) q[64];
sx q[64];
rz(-pi) q[64];
cx q[72],q[62];
rz(pi/8) q[62];
cx q[72],q[62];
rz(-pi/8) q[62];
cx q[63],q[62];
rz(pi/16) q[62];
cx q[63],q[62];
rz(-pi/16) q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(7*pi/8) q[62];
sx q[62];
rz(1.3230994) q[63];
sx q[63];
cx q[64],q[63];
rz(pi/2) q[63];
sx q[64];
rz(-pi) q[64];
cx q[64],q[63];
rz(pi/4) q[63];
sx q[64];
cx q[64],q[63];
rz(pi/4) q[63];
sx q[63];
rz(-pi/2) q[63];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[63];
rz(-pi) q[63];
cx q[63],q[62];
rz(3*pi/8) q[62];
sx q[63];
cx q[63],q[62];
x q[62];
rz(7*pi/8) q[62];
rz(3*pi/4) q[63];
rz(-2.1084975) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[63],q[64];
rz(pi/4) q[64];
cx q[63],q[64];
sx q[63];
rz(pi/2) q[63];
rz(-pi/4) q[64];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi/1024) q[83];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
rz(pi/512) q[85];
cx q[73],q[85];
rz(-pi/512) q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[83],q[84];
rz(pi/1024) q[84];
cx q[83],q[84];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(-pi/1024) q[84];
rz(-pi/2048) q[85];
cx q[85],q[84];
rz(pi/2048) q[84];
cx q[85],q[84];
rz(-pi/2048) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
rz(pi/256) q[85];
cx q[73],q[85];
rz(-pi/256) q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
rz(-pi/1024) q[85];
rz(-pi/4096) q[92];
cx q[92],q[83];
rz(pi/4096) q[83];
cx q[92],q[83];
rz(-pi/4096) q[83];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
rz(-pi/8192) q[92];
cx q[92],q[83];
rz(pi/8192) q[83];
cx q[92],q[83];
cx q[102],q[92];
rz(-pi/8192) q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[60],q[61];
rz(pi/16384) q[61];
cx q[60],q[61];
rz(-pi/16384) q[61];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
rz(-pi/8192) q[61];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(-pi/65536) q[63];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
rz(-pi/32768) q[65];
cx q[65],q[64];
rz(pi/32768) q[64];
cx q[65],q[64];
rz(-pi/32768) q[64];
cx q[63],q[64];
rz(pi/65536) q[64];
cx q[63],q[64];
rz(-pi/65536) q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
rz(-0.0013422332) q[64];
rz(-pi/512) q[83];
cx q[83],q[84];
rz(pi/512) q[84];
cx q[83],q[84];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(-pi/512) q[84];
cx q[85],q[84];
rz(pi/1024) q[84];
cx q[85],q[84];
rz(-pi/1024) q[84];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
rz(-pi/128) q[85];
cx q[85],q[84];
rz(pi/128) q[84];
cx q[85],q[84];
cx q[73],q[85];
rz(-pi/128) q[84];
cx q[85],q[73];
cx q[73],q[85];
rz(-pi/64) q[73];
cx q[73],q[66];
rz(pi/64) q[66];
cx q[73],q[66];
rz(-pi/64) q[66];
rz(-pi/512) q[85];
cx q[92],q[102];
cx q[102],q[92];
rz(-pi/2048) q[92];
cx q[92],q[83];
rz(pi/2048) q[83];
cx q[92],q[83];
rz(-pi/2048) q[83];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
rz(-pi/4096) q[92];
cx q[92],q[83];
rz(pi/4096) q[83];
cx q[92],q[83];
cx q[102],q[92];
rz(-pi/4096) q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[61],q[62];
rz(pi/8192) q[62];
cx q[61],q[62];
rz(-pi/8192) q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-pi/32768) q[62];
cx q[64],q[63];
rz(pi/16384) q[63];
cx q[64],q[63];
rz(-pi/16384) q[63];
cx q[62],q[63];
rz(pi/32768) q[63];
cx q[62],q[63];
rz(-pi/32768) q[63];
rz(pi/4) q[72];
sx q[72];
rz(-pi) q[72];
rz(pi/4) q[81];
sx q[81];
rz(-pi) q[81];
rz(pi/4) q[82];
sx q[82];
rz(-pi) q[82];
rz(-pi/256) q[83];
cx q[83],q[84];
rz(pi/256) q[84];
cx q[83],q[84];
rz(-pi/256) q[84];
cx q[85],q[84];
rz(pi/512) q[84];
cx q[85],q[84];
rz(-pi/512) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
rz(-pi) q[83];
sx q[83];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
rz(-0.085902924) q[66];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
rz(-pi/128) q[85];
cx q[85],q[73];
rz(pi/128) q[73];
cx q[85],q[73];
rz(-pi/128) q[73];
cx q[66],q[73];
rz(pi/256) q[73];
cx q[66],q[73];
rz(-pi/256) q[73];
cx q[92],q[102];
cx q[102],q[92];
rz(-0.047553404) q[102];
rz(-1.5738643) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[92],q[83];
rz(-pi/2) q[83];
sx q[92];
rz(-pi) q[92];
cx q[92],q[83];
rz(1.5677284) q[83];
sx q[92];
cx q[92],q[83];
rz(pi/2) q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-0.88357293) q[83];
sx q[83];
rz(-pi/2) q[83];
cx q[83],q[82];
rz(-pi/2) q[82];
sx q[83];
rz(-pi) q[83];
cx q[83],q[82];
rz(1.4726216) q[82];
sx q[83];
cx q[83],q[82];
rz(pi/16) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[82],q[81];
rz(-pi/2) q[81];
sx q[82];
rz(-pi) q[82];
cx q[82],q[81];
rz(7*pi/16) q[81];
sx q[82];
cx q[82],q[81];
rz(pi/8) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[81],q[72];
rz(-pi/2) q[72];
sx q[81];
rz(-pi) q[81];
cx q[81],q[72];
rz(3*pi/8) q[72];
sx q[81];
cx q[81],q[72];
x q[72];
rz(-3*pi/4) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/4) q[61];
cx q[61],q[60];
rz(pi/4) q[60];
cx q[61],q[60];
rz(-pi/4) q[60];
sx q[61];
rz(pi/2) q[61];
rz(2.6869837) q[81];
sx q[81];
rz(-pi) q[81];
rz(-9*pi/16) q[82];
sx q[82];
rz(-pi) q[82];
rz(-2.4543693) q[83];
sx q[83];
rz(-pi) q[83];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
rz(-pi/64) q[84];
sx q[84];
rz(-pi/2) q[84];
cx q[84],q[83];
rz(-pi/2) q[83];
sx q[84];
rz(-pi) q[84];
cx q[84],q[83];
rz(1.5217089) q[83];
sx q[84];
cx q[84],q[83];
rz(0.88357293) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[83],q[82];
rz(-pi/2) q[82];
sx q[83];
rz(-pi) q[83];
cx q[83],q[82];
rz(1.4726216) q[82];
sx q[83];
cx q[83],q[82];
rz(-1.4363567) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[82],q[81];
rz(-pi/2) q[81];
sx q[82];
rz(-pi) q[82];
cx q[82],q[81];
rz(7*pi/16) q[81];
sx q[82];
cx q[82],q[81];
x q[81];
rz(2.2942846) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
rz(pi/8) q[61];
cx q[61],q[60];
rz(pi/8) q[60];
cx q[61],q[60];
rz(-pi/8) q[60];
cx q[61],q[62];
rz(pi/4) q[62];
cx q[61],q[62];
sx q[61];
rz(pi/2) q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(-pi/4) q[62];
rz(0.78463117) q[72];
sx q[72];
rz(-pi) q[72];
rz(-0.91983781) q[82];
rz(-2.4543693) q[83];
rz(-1.6198837) q[84];
rz(-pi/512) q[85];
cx q[85],q[73];
rz(pi/512) q[73];
cx q[85],q[73];
rz(-pi/512) q[73];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
rz(pi/128) q[73];
cx q[66],q[73];
rz(-pi/128) q[73];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi/256) q[85];
cx q[85],q[73];
rz(pi/256) q[73];
cx q[85],q[73];
rz(-pi/256) q[73];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
rz(pi/64) q[73];
cx q[66],q[73];
rz(-pi/64) q[73];
sx q[73];
rz(-pi) q[73];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
rz(-pi/128) q[85];
sx q[85];
rz(-pi/2) q[85];
cx q[85],q[73];
rz(-pi/2) q[73];
sx q[85];
rz(-pi) q[85];
cx q[85],q[73];
rz(1.5462526) q[73];
sx q[85];
cx q[85],q[73];
rz(-pi) q[73];
x q[73];
rz(-1.59534) q[85];
x q[92];
rz(1.5677284) q[92];
cx q[102],q[92];
rz(pi/2048) q[92];
cx q[102],q[92];
rz(-pi/2048) q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[102],q[92];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(-pi/4) q[81];
sx q[81];
rz(-pi/2) q[81];
cx q[81],q[72];
rz(-pi/2) q[72];
sx q[81];
rz(-pi) q[81];
cx q[81],q[72];
rz(1.5700293) q[72];
sx q[81];
cx q[81],q[72];
x q[72];
rz(-2.3569615) q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[64],q[63];
rz(pi/8192) q[63];
cx q[64],q[63];
rz(-pi/8192) q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-3*pi/4) q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
rz(-pi/16384) q[72];
cx q[72],q[62];
rz(pi/16384) q[62];
cx q[72],q[62];
rz(-pi/16384) q[62];
rz(0.78079622) q[82];
rz(pi/1024) q[92];
cx q[102],q[92];
rz(-pi/1024) q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[82],q[83];
rz(pi/2048) q[83];
cx q[82],q[83];
sx q[82];
rz(-pi) q[82];
rz(-pi/2048) q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[83],q[92];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
rz(pi/4096) q[65];
cx q[64],q[65];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-pi/4096) q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[67],q[66];
rz(pi/131072) q[66];
cx q[67],q[66];
rz(-pi/131072) q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
sx q[66];
rz(-pi) q[66];
sx q[67];
rz(-pi/2) q[67];
cx q[67],q[66];
rz(-pi/2) q[66];
sx q[67];
rz(-pi) q[67];
cx q[67],q[66];
rz(pi/2) q[66];
sx q[67];
cx q[67],q[66];
x q[66];
rz(3.1409215) q[66];
cx q[66],q[65];
rz(pi/32768) q[65];
cx q[66],q[65];
rz(-pi/32768) q[65];
rz(-1.5708443) q[67];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
rz(0.78424768) q[62];
cx q[62],q[63];
rz(pi/8192) q[63];
cx q[62],q[63];
sx q[62];
rz(-pi) q[62];
rz(-pi/8192) q[63];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[66],q[65];
rz(pi/16384) q[65];
cx q[66],q[65];
rz(-pi/16384) q[65];
rz(3.1400587) q[72];
sx q[72];
rz(-0.29452431) q[73];
cx q[92],q[83];
cx q[83],q[92];
cx q[102],q[92];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[73],q[85];
rz(-pi/64) q[84];
sx q[84];
rz(-pi) q[84];
rz(pi/32) q[85];
cx q[73],q[85];
rz(-pi/32) q[85];
sx q[85];
rz(-pi/2) q[85];
cx q[85],q[84];
rz(-pi/2) q[84];
sx q[85];
rz(-pi) q[85];
cx q[85],q[84];
rz(1.5217089) q[84];
sx q[85];
cx q[85],q[84];
x q[84];
rz(3.0925053) q[84];
rz(-pi/2) q[85];
rz(pi/512) q[92];
cx q[102],q[92];
rz(-pi/512) q[92];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[102],q[92];
rz(-pi/4) q[83];
sx q[83];
rz(-pi/2) q[83];
cx q[83],q[82];
rz(-pi/2) q[82];
sx q[83];
rz(-pi) q[83];
cx q[83],q[82];
rz(1.5677284) q[82];
sx q[83];
cx q[83],q[82];
x q[82];
rz(-2.3592625) q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
rz(-pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[81],q[72];
rz(-pi/2) q[72];
sx q[81];
rz(-pi) q[81];
cx q[81],q[72];
rz(1.5692623) q[72];
sx q[81];
cx q[81],q[72];
rz(0.78386418) q[72];
sx q[72];
rz(-pi/2) q[72];
cx q[72],q[62];
rz(-pi/2) q[62];
sx q[72];
rz(-pi) q[72];
cx q[72],q[62];
rz(1.5700293) q[62];
sx q[72];
cx q[72],q[62];
x q[62];
rz(-2.3569615) q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
sx q[60];
rz(-pi/2) q[60];
cx q[60],q[53];
rz(-pi/2) q[53];
sx q[60];
rz(-pi) q[60];
cx q[60],q[53];
rz(pi/2) q[53];
sx q[60];
cx q[60],q[53];
rz(1.1984225e-05) q[53];
sx q[53];
rz(pi/2) q[53];
cx q[53],q[41];
rz(-pi/2) q[41];
sx q[53];
rz(-pi) q[53];
cx q[53],q[41];
rz(pi/2) q[41];
sx q[53];
cx q[53],q[41];
x q[41];
rz(3.1415867) q[41];
rz(-pi/2) q[53];
rz(-pi/2) q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[66],q[65];
rz(pi/8192) q[65];
cx q[66],q[65];
rz(-pi/8192) q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(0.78463117) q[63];
sx q[63];
rz(pi/2) q[63];
rz(-pi/131072) q[64];
rz(-pi/2048) q[72];
sx q[72];
cx q[73],q[66];
rz(pi/16) q[66];
cx q[73],q[66];
rz(3*pi/16) q[66];
sx q[66];
rz(-pi) q[66];
cx q[73],q[85];
rz(0.78846612) q[81];
sx q[81];
rz(-2.3623304) q[83];
cx q[85],q[73];
cx q[73],q[85];
rz(-0.88357293) q[73];
sx q[73];
rz(-pi/2) q[73];
cx q[73],q[66];
rz(-pi/2) q[66];
sx q[73];
rz(-pi) q[73];
cx q[73],q[66];
rz(1.4726216) q[66];
sx q[73];
cx q[73],q[66];
x q[66];
rz(-13*pi/16) q[66];
rz(-2.4543693) q[73];
rz(pi/256) q[92];
cx q[102],q[92];
rz(-pi/256) q[92];
cx q[83],q[92];
rz(pi/512) q[92];
cx q[83],q[92];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi/256) q[84];
rz(-pi/512) q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[102],q[92];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
rz(-pi/4) q[82];
sx q[82];
rz(-pi/2) q[82];
cx q[82],q[81];
rz(-pi/2) q[81];
sx q[82];
rz(-pi) q[82];
cx q[82],q[81];
rz(1.5677284) q[81];
sx q[82];
cx q[82],q[81];
rz(pi/1024) q[81];
sx q[81];
rz(-pi/2) q[81];
cx q[81],q[72];
rz(-pi/2) q[72];
sx q[81];
rz(-pi) q[81];
cx q[81],q[72];
rz(1.5692623) q[72];
sx q[81];
cx q[81],q[72];
rz(0.78386418) q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
rz(3*pi/4) q[62];
sx q[62];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[63];
rz(-pi) q[63];
cx q[63],q[62];
rz(1.5700293) q[62];
sx q[63];
cx q[63],q[62];
x q[62];
rz(3*pi/4) q[62];
rz(2.3554275) q[63];
x q[81];
rz(3*pi/4) q[81];
rz(-2.3623304) q[82];
rz(pi/128) q[92];
cx q[102],q[92];
rz(-pi/128) q[92];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[84],q[83];
rz(pi/256) q[83];
cx q[84],q[83];
rz(-pi/256) q[83];
cx q[82],q[83];
rz(pi/512) q[83];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
rz(-pi/256) q[72];
rz(-pi/1024) q[82];
rz(-pi/512) q[83];
cx q[82],q[83];
rz(pi/1024) q[83];
cx q[82],q[83];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
rz(-pi/1024) q[83];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
rz(-3*pi/8) q[84];
sx q[84];
rz(-pi/2) q[84];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
rz(pi/4) q[83];
sx q[83];
rz(-pi) q[83];
cx q[84],q[83];
rz(-pi/2) q[83];
sx q[84];
rz(-pi) q[84];
cx q[84],q[83];
rz(3*pi/8) q[83];
sx q[84];
cx q[84],q[83];
rz(-pi) q[83];
x q[83];
cx q[83],q[82];
rz(pi/4) q[82];
cx q[83],q[82];
rz(-pi/4) q[82];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
rz(-7*pi/8) q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
rz(pi/16) q[73];
cx q[66],q[73];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[64],q[65];
rz(pi/131072) q[65];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(-pi/131072) q[65];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
rz(-pi/16) q[73];
rz(-pi/128) q[85];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
rz(-pi/64) q[83];
cx q[83],q[84];
rz(pi/64) q[84];
cx q[83],q[84];
rz(-pi/64) q[84];
cx q[85],q[84];
rz(pi/128) q[84];
cx q[85],q[84];
rz(-pi/128) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[72],q[81];
rz(pi/256) q[81];
cx q[72],q[81];
rz(-pi/256) q[81];
rz(-pi/512) q[82];
cx q[82],q[81];
rz(pi/512) q[81];
cx q[82],q[81];
rz(-pi/512) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
rz(2.3439226) q[81];
sx q[81];
rz(-pi/128) q[82];
sx q[82];
rz(-pi) q[82];
rz(-pi/32) q[84];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
rz(-pi/64) q[73];
cx q[84],q[85];
rz(pi/32) q[85];
cx q[84],q[85];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi/32) q[85];
cx q[73],q[85];
rz(pi/64) q[85];
cx q[73],q[85];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
rz(-0.29452431) q[66];
rz(-pi/8) q[73];
rz(-pi/64) q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
sx q[83];
rz(-pi/2) q[83];
cx q[83],q[82];
rz(-pi/2) q[82];
sx q[83];
rz(-pi) q[83];
cx q[83],q[82];
rz(1.5462526) q[82];
sx q[83];
cx q[83],q[82];
rz(0.80994186) q[82];
sx q[82];
rz(-pi/2) q[82];
cx q[82],q[81];
rz(-pi/2) q[81];
sx q[82];
rz(-pi) q[82];
cx q[82],q[81];
rz(1.5585245) q[81];
sx q[82];
cx q[82],q[81];
rz(0.77312632) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(-0.010737866) q[81];
x q[82];
rz(3*pi/4) q[82];
rz(-pi/2) q[83];
rz(-pi/16) q[84];
rz(pi/8) q[85];
cx q[73],q[85];
rz(-pi/8) q[85];
cx q[84],q[85];
rz(pi/16) q[85];
cx q[84],q[85];
rz(-pi/16) q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
rz(pi/32) q[73];
cx q[66],q[73];
rz(-pi/32) q[73];
sx q[73];
rz(-pi) q[73];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(1.3230994) q[83];
sx q[83];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
rz(-3*pi/8) q[84];
sx q[84];
rz(-pi/2) q[84];
rz(-pi/64) q[85];
sx q[85];
rz(-pi/2) q[85];
cx q[85],q[73];
rz(-pi/2) q[73];
sx q[85];
rz(-pi) q[85];
cx q[85],q[73];
rz(1.5217089) q[73];
sx q[85];
cx q[85],q[73];
rz(-pi) q[73];
x q[73];
rz(-1.6198837) q[85];
sx q[92];
rz(pi/2) q[92];
cx q[92],q[83];
rz(pi/2) q[83];
sx q[92];
rz(-pi) q[92];
cx q[92],q[83];
rz(pi/4) q[83];
sx q[92];
cx q[92],q[83];
sx q[83];
cx q[84],q[83];
rz(-pi/2) q[83];
sx q[84];
rz(-pi) q[84];
cx q[84],q[83];
rz(3*pi/8) q[83];
sx q[84];
cx q[84],q[83];
rz(-pi) q[83];
x q[83];
rz(-7*pi/8) q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
sx q[84];
rz(-pi/2) q[84];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
rz(pi/16) q[73];
cx q[66],q[73];
rz(3*pi/16) q[73];
sx q[73];
rz(-pi) q[73];
rz(-0.88357293) q[85];
sx q[85];
rz(-pi/2) q[85];
cx q[85],q[73];
rz(-pi/2) q[73];
sx q[85];
rz(-pi) q[85];
cx q[85],q[73];
rz(1.4726216) q[73];
sx q[85];
cx q[85],q[73];
x q[73];
rz(-3*pi/4) q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
rz(-2.4543693) q[85];
rz(-2.1084975) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[83],q[92];
rz(pi/4) q[92];
cx q[83],q[92];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(-pi/128) q[83];
sx q[83];
rz(-pi) q[83];
cx q[84],q[83];
rz(-pi/2) q[83];
sx q[84];
rz(-pi) q[84];
cx q[84],q[83];
rz(1.5462526) q[83];
sx q[84];
cx q[84],q[83];
x q[83];
rz(3.117049) q[83];
rz(-1.6198837) q[84];
cx q[84],q[85];
rz(pi/64) q[85];
cx q[84],q[85];
rz(-pi/64) q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
rz(-pi/16) q[73];
rz(-3*pi/8) q[85];
sx q[85];
rz(-pi/2) q[85];
rz(-pi/4) q[92];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
rz(pi/4) q[84];
sx q[84];
rz(-pi) q[84];
cx q[85],q[84];
rz(-pi/2) q[84];
sx q[85];
rz(-pi) q[85];
cx q[85],q[84];
rz(3*pi/8) q[84];
sx q[85];
cx q[85],q[84];
x q[84];
rz(-3*pi/4) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
rz(pi/4) q[83];
cx q[83],q[82];
rz(pi/4) q[82];
cx q[83],q[82];
rz(-pi/4) q[82];
sx q[83];
rz(pi/2) q[83];
rz(-pi/32) q[84];
rz(-7*pi/8) q[85];
cx q[73],q[85];
rz(pi/16) q[85];
cx q[73],q[85];
rz(-pi/16) q[85];
cx q[84],q[85];
rz(pi/32) q[85];
cx q[84],q[85];
rz(-pi/32) q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
rz(-3*pi/8) q[84];
sx q[84];
rz(-pi/2) q[84];
rz(-5*pi/16) q[85];
sx q[85];
rz(-pi/2) q[85];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[81],q[82];
rz(pi/2048) q[82];
cx q[81],q[82];
cx q[81],q[72];
rz(pi/1024) q[72];
cx q[81],q[72];
rz(-pi/1024) q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-pi/65536) q[62];
cx q[62],q[61];
rz(pi/65536) q[61];
cx q[62],q[61];
rz(-pi/65536) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(-pi/32768) q[61];
cx q[61],q[60];
rz(pi/32768) q[60];
cx q[61],q[60];
rz(-pi/32768) q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(-pi/262144) q[63];
cx q[63],q[64];
rz(pi/262144) q[64];
cx q[63],q[64];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-7.1905349e-05) q[62];
cx q[62],q[61];
rz(pi/131072) q[61];
cx q[62],q[61];
rz(-pi/131072) q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[62],q[61];
rz(pi/65536) q[61];
cx q[62],q[61];
rz(-pi/65536) q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(-0.0005752428) q[63];
rz(-pi/262144) q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
rz(pi/16384) q[64];
cx q[63],q[64];
rz(-pi/16384) q[64];
cx q[81],q[72];
rz(pi/512) q[72];
cx q[81],q[72];
rz(-pi/512) q[72];
rz(-pi/2048) q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(-pi/4) q[81];
sx q[81];
rz(-pi/2) q[81];
rz(pi/4) q[83];
sx q[83];
rz(-pi) q[83];
cx q[84],q[83];
rz(-pi/2) q[83];
sx q[84];
rz(-pi) q[84];
cx q[84],q[83];
rz(3*pi/8) q[83];
sx q[84];
cx q[84],q[83];
rz(-2.1084975) q[83];
sx q[83];
rz(-pi) q[83];
rz(-5*pi/8) q[84];
sx q[84];
rz(-pi) q[84];
cx q[85],q[84];
rz(-pi/2) q[84];
sx q[85];
rz(-pi) q[85];
cx q[85],q[84];
rz(7*pi/16) q[84];
sx q[85];
cx q[85],q[84];
rz(pi/8) q[84];
sx q[84];
rz(pi/2) q[84];
rz(-13*pi/16) q[85];
sx q[92];
rz(pi/2) q[92];
cx q[92],q[83];
rz(pi/2) q[83];
sx q[92];
rz(-pi) q[92];
cx q[92],q[83];
rz(pi/4) q[83];
sx q[92];
cx q[92],q[83];
sx q[83];
cx q[84],q[83];
rz(-pi/2) q[83];
sx q[84];
rz(-pi) q[84];
cx q[84],q[83];
rz(3*pi/8) q[83];
sx q[84];
cx q[84],q[83];
rz(-pi) q[83];
x q[83];
rz(-7*pi/8) q[84];
rz(-2.1084975) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[83],q[92];
rz(pi/4) q[92];
cx q[83],q[92];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
rz(-pi/256) q[83];
rz(-pi/4) q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[83],q[92];
rz(pi/256) q[92];
cx q[83],q[92];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
rz(-0.84730806) q[83];
sx q[83];
rz(-pi) q[83];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
rz(pi/4) q[84];
sx q[84];
rz(-pi) q[84];
rz(-0.95720401) q[85];
cx q[85],q[73];
rz(pi/128) q[73];
cx q[85],q[73];
rz(-pi/128) q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[63],q[64];
rz(pi/8192) q[64];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-0.0023009712) q[62];
cx q[62],q[72];
rz(-0.78606928) q[63];
rz(-pi/8192) q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
rz(pi/32768) q[64];
cx q[63],q[64];
rz(-pi/32768) q[64];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[63],q[64];
rz(pi/16384) q[64];
cx q[63],q[64];
sx q[63];
rz(-pi/2) q[63];
rz(-pi/16384) q[64];
cx q[66],q[67];
cx q[67],q[66];
cx q[66],q[67];
rz(pi/4096) q[72];
cx q[62],q[72];
cx q[62],q[61];
rz(pi/2048) q[61];
cx q[62],q[61];
rz(-pi/2048) q[61];
rz(-pi/4096) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(pi/4) q[62];
sx q[62];
rz(-pi) q[62];
cx q[63],q[62];
rz(-pi/2) q[62];
sx q[63];
rz(-pi) q[63];
cx q[63],q[62];
rz(1.5704128) q[62];
sx q[63];
cx q[63],q[62];
x q[62];
rz(0.78309719) q[62];
cx q[62],q[61];
rz(pi/4096) q[61];
cx q[62],q[61];
rz(-pi/4096) q[61];
sx q[62];
rz(-2.356578) q[63];
rz(0.7823302) q[72];
sx q[72];
rz(-pi) q[72];
cx q[81],q[72];
rz(-pi/2) q[72];
sx q[81];
rz(-pi) q[81];
cx q[81],q[72];
rz(1.5677284) q[72];
sx q[81];
cx q[81],q[72];
rz(0.78846612) q[72];
sx q[72];
rz(-pi/2) q[72];
cx q[72],q[62];
rz(-pi/2) q[62];
sx q[72];
rz(-pi) q[72];
cx q[72],q[62];
rz(1.5692623) q[62];
sx q[72];
cx q[72],q[62];
rz(1.5692623) q[62];
rz(-1.5677284) q[72];
sx q[72];
rz(-pi) q[72];
rz(-2.3623304) q[81];
sx q[81];
rz(-pi) q[81];
cx q[85],q[73];
rz(pi/64) q[73];
cx q[85],q[73];
rz(-pi/64) q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
sx q[85];
rz(-pi/2) q[85];
cx q[85],q[84];
rz(-pi/2) q[84];
sx q[85];
rz(-pi) q[85];
cx q[85],q[84];
rz(1.4726216) q[84];
sx q[85];
cx q[85],q[84];
rz(-1.4363567) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[84],q[83];
rz(-pi/2) q[83];
sx q[84];
rz(-pi) q[84];
cx q[84],q[83];
rz(7*pi/16) q[83];
sx q[84];
cx q[84],q[83];
x q[83];
rz(2.2942846) q[83];
rz(-0.91983781) q[84];
rz(-2.4543693) q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
sx q[73];
rz(-pi) q[73];
sx q[85];
rz(-pi/2) q[85];
rz(-pi/256) q[92];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
sx q[82];
rz(-pi/2) q[82];
cx q[82],q[81];
rz(-pi/2) q[81];
sx q[82];
rz(-pi) q[82];
cx q[82],q[81];
rz(1.5646604) q[81];
sx q[82];
cx q[82],q[81];
rz(1.5769322) q[81];
sx q[81];
rz(-pi/2) q[81];
cx q[81],q[72];
rz(-pi/2) q[72];
sx q[81];
rz(-pi) q[81];
cx q[81],q[72];
rz(1.5677284) q[72];
sx q[81];
cx q[81],q[72];
rz(1.5677284) q[72];
x q[81];
rz(pi/2) q[81];
rz(-pi/2) q[82];
rz(pi/8) q[92];
cx q[92],q[102];
rz(pi/8) q[102];
cx q[92],q[102];
rz(-pi/8) q[102];
cx q[92],q[83];
rz(pi/4) q[83];
cx q[92],q[83];
rz(-pi/4) q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(-0.018407769) q[83];
rz(-pi/256) q[84];
sx q[84];
rz(-pi) q[84];
cx q[85],q[84];
rz(-pi/2) q[84];
sx q[85];
rz(-pi) q[85];
cx q[85],q[84];
rz(1.5585245) q[84];
sx q[85];
cx q[85],q[84];
x q[84];
rz(3.1293208) q[84];
cx q[83],q[84];
rz(pi/512) q[84];
cx q[83],q[84];
rz(-pi/512) q[84];
rz(-1.59534) q[85];
sx q[85];
rz(-pi/2) q[85];
cx q[85],q[73];
rz(-pi/2) q[73];
sx q[85];
rz(-pi) q[85];
cx q[85],q[73];
rz(1.5462526) q[73];
sx q[85];
cx q[85],q[73];
x q[73];
rz(3.0925053) q[73];
cx q[73],q[66];
rz(pi/64) q[66];
cx q[73],q[66];
rz(-pi/64) q[66];
rz(-1.59534) q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
cx q[83],q[84];
rz(pi/256) q[84];
cx q[83],q[84];
rz(-pi/256) q[84];
cx q[85],q[73];
cx q[73],q[85];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi/32) q[83];
cx q[83],q[82];
rz(pi/32) q[82];
cx q[83],q[82];
rz(-pi/32) q[82];
rz(-pi/128) q[84];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[84],q[85];
rz(pi/128) q[85];
cx q[84],q[85];
rz(-pi/128) q[85];
sx q[92];
rz(pi/2) q[92];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
rz(-pi/64) q[83];
cx q[83],q[82];
rz(pi/64) q[82];
cx q[83],q[82];
rz(-pi/64) q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
rz(-pi/16) q[92];
cx q[92],q[102];
rz(pi/16) q[102];
cx q[92],q[102];
rz(-pi/16) q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
rz(pi/8) q[83];
cx q[83],q[82];
rz(pi/8) q[82];
cx q[83],q[82];
rz(-pi/8) q[82];
cx q[83],q[84];
rz(pi/4) q[84];
cx q[83],q[84];
sx q[83];
rz(pi/2) q[83];
rz(-pi/4) q[84];
rz(-pi/32) q[92];
cx q[92],q[102];
rz(pi/32) q[102];
cx q[92],q[102];
rz(-pi/32) q[102];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
rz(pi/16) q[83];
cx q[83],q[82];
rz(pi/16) q[82];
cx q[83],q[82];
rz(-pi/16) q[82];
cx q[83],q[84];
rz(pi/8) q[84];
cx q[83],q[84];
cx q[83],q[92];
rz(-pi/8) q[84];
rz(pi/4) q[92];
cx q[83],q[92];
sx q[83];
rz(pi/2) q[83];
rz(-pi/4) q[92];
barrier q[112],q[57],q[2],q[121],q[102],q[11],q[8],q[75],q[67],q[106],q[17],q[65],q[26],q[90],q[35],q[99],q[45],q[108],q[83],q[105],q[50],q[114],q[59],q[4],q[123],q[1],q[68],q[13],q[60],q[10],q[77],q[74],q[19],q[73],q[28],q[63],q[37],q[101],q[34],q[98],q[44],q[107],q[52],q[116],q[41],q[6],q[125],q[3],q[70],q[84],q[12],q[76],q[21],q[62],q[30],q[94],q[27],q[91],q[103],q[36],q[100],q[54],q[109],q[82],q[118],q[92],q[64],q[5],q[124],q[69],q[14],q[78],q[23],q[87],q[20],q[32],q[61],q[96],q[29],q[93],q[38],q[85],q[47],q[111],q[56],q[81],q[120],q[117],q[42],q[7],q[126],q[71],q[16],q[80],q[25],q[89],q[22],q[86],q[31],q[95],q[40],q[104],q[49],q[46],q[113],q[58],q[110],q[55],q[122],q[0],q[119],q[53],q[9],q[66],q[18],q[72],q[15],q[79],q[115],q[24],q[88],q[33],q[97],q[43],q[39],q[51],q[48];
measure q[41] -> c[0];
measure q[67] -> c[1];
measure q[53] -> c[2];
measure q[60] -> c[3];
measure q[65] -> c[4];
measure q[64] -> c[5];
measure q[63] -> c[6];
measure q[61] -> c[7];
measure q[62] -> c[8];
measure q[72] -> c[9];
measure q[66] -> c[10];
measure q[73] -> c[11];
measure q[85] -> c[12];
measure q[81] -> c[13];
measure q[102] -> c[14];
measure q[82] -> c[15];
measure q[84] -> c[16];
measure q[92] -> c[17];
measure q[83] -> c[18];
