OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg c[17];
rz(pi/2) q[41];
sx q[41];
rz(pi/2) q[41];
rz(pi/2) q[42];
sx q[42];
rz(pi/2) q[42];
rz(pi/2) q[43];
sx q[43];
rz(pi/2) q[43];
rz(pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
rz(pi/2) q[59];
sx q[59];
rz(pi/2) q[59];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
x q[62];
rz(-1.3579205) q[62];
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
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
rz(1.3464876) q[72];
cx q[62],q[72];
cx q[62],q[63];
rz(-0.448617475) q[63];
cx q[62],q[63];
rz(0.448617475) q[63];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[54],q[64];
cx q[62],q[63];
rz(-0.89723495) q[63];
cx q[62],q[63];
rz(0.89723495) q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[62],q[63];
rz(1.34712275) q[63];
cx q[62],q[63];
rz(-1.34712275) q[63];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
rz(-0.447347145) q[63];
cx q[62],q[63];
rz(0.447347145) q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[62],q[63];
rz(-0.8946943) q[63];
cx q[62],q[63];
rz(0.8946943) q[63];
rz(-1.3464876) q[72];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
rz(1.35220405) q[72];
cx q[62],q[72];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
rz(-0.081300975) q[53];
cx q[53],q[41];
rz(-0.437184525) q[41];
cx q[53],q[41];
rz(0.437184525) q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
rz(-0.87436905) q[60];
cx q[53],q[60];
cx q[53],q[41];
rz(1.39285455) q[41];
cx q[53],q[41];
rz(-1.39285455) q[41];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(0.87436905) q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[53],q[41];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
rz(-pi/2) q[53];
sx q[53];
rz(-pi) q[53];
rz(2.4911848) q[60];
cx q[60],q[59];
rz(-0.355883545) q[59];
cx q[60],q[59];
rz(0.355883545) q[59];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
rz(-0.7117671) q[61];
cx q[60],q[61];
sx q[60];
rz(pi/2) q[60];
cx q[60],q[53];
rz(-pi/2) q[53];
sx q[60];
rz(-pi) q[60];
cx q[60],q[53];
rz(0.14726218) q[53];
sx q[60];
cx q[60],q[53];
rz(-0.29452431) q[53];
cx q[53],q[41];
rz(0.294524311274043) q[41];
cx q[53],q[41];
rz(-0.294524311274043) q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
rz(-pi/256) q[43];
rz(1.4235341) q[60];
rz(0.7117671) q[61];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
rz(-1.35220405) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(-pi/128) q[53];
sx q[53];
rz(-pi) q[53];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(-3*pi/16) q[61];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
rz(3*pi/16) q[62];
cx q[61],q[62];
rz(-3*pi/16) q[62];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(pi/8) q[64];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(-pi/8) q[82];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[82],q[83];
rz(3*pi/8) q[83];
cx q[82],q[83];
cx q[82],q[81];
rz(-pi/4) q[81];
cx q[82],q[81];
rz(pi/4) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(-3*pi/8) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/4) q[72];
cx q[72],q[62];
rz(pi/4) q[62];
cx q[72],q[62];
rz(-pi/4) q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[64],q[63];
rz(pi/8) q[63];
cx q[64],q[63];
rz(-pi/8) q[63];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[64],q[63];
rz(pi/4) q[63];
cx q[64],q[63];
rz(-pi/4) q[63];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(pi/16) q[63];
cx q[63],q[62];
rz(pi/16) q[62];
cx q[63],q[62];
rz(-pi/16) q[62];
cx q[63],q[64];
rz(pi/8) q[64];
cx q[63],q[64];
rz(-pi/8) q[64];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[63],q[64];
rz(pi/4) q[64];
cx q[63],q[64];
sx q[63];
rz(pi/2) q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-pi/4) q[64];
cx q[64],q[65];
cx q[65],q[64];
rz(pi/32) q[64];
cx q[64],q[63];
rz(pi/32) q[63];
cx q[64],q[63];
rz(-pi/32) q[63];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[64],q[54];
rz(pi/16) q[54];
cx q[64],q[54];
rz(-pi/16) q[54];
rz(pi/8) q[65];
cx q[64],q[65];
cx q[64],q[63];
rz(pi/4) q[63];
cx q[64],q[63];
rz(-pi/4) q[63];
sx q[64];
rz(pi/2) q[64];
rz(-pi/8) q[65];
rz(-pi/64) q[72];
cx q[72],q[62];
rz(pi/64) q[62];
cx q[72],q[62];
rz(-pi/64) q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
sx q[60];
rz(-pi/2) q[60];
cx q[60],q[53];
rz(-pi/2) q[53];
sx q[60];
rz(-pi) q[60];
cx q[60],q[53];
rz(1.5462526) q[53];
sx q[60];
cx q[60],q[53];
x q[53];
rz(3.117049) q[53];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[43],q[42];
rz(pi/256) q[42];
cx q[43],q[42];
rz(-pi/256) q[42];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
rz(-0.0092038847) q[42];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(-pi/512) q[53];
cx q[53],q[41];
rz(pi/512) q[41];
cx q[53],q[41];
rz(-pi/512) q[41];
cx q[42],q[41];
rz(pi/1024) q[41];
cx q[42],q[41];
rz(2.3531265) q[41];
sx q[41];
rz(-pi/2) q[60];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[60];
cx q[53],q[60];
rz(2.3538935) q[59];
cx q[60],q[53];
cx q[53],q[60];
rz(-0.78693214) q[53];
sx q[53];
rz(pi/2) q[53];
cx q[53],q[41];
rz(-pi/2) q[41];
sx q[53];
rz(-pi) q[53];
cx q[53],q[41];
rz(1.5692623) q[41];
sx q[53];
cx q[53],q[41];
rz(pi/4) q[41];
x q[53];
rz(2.3546605) q[53];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(-0.29452431) q[64];
cx q[64],q[54];
rz(pi/32) q[54];
cx q[64],q[54];
rz(-pi/32) q[54];
cx q[64],q[65];
rz(pi/16) q[65];
cx q[64],q[65];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(pi/8) q[63];
cx q[63],q[62];
rz(pi/8) q[62];
cx q[63],q[62];
rz(-pi/8) q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[63],q[64];
rz(pi/4) q[64];
cx q[63],q[64];
sx q[63];
rz(pi/2) q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-pi/64) q[63];
sx q[63];
rz(-pi) q[63];
rz(-pi/4) q[64];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[64],q[63];
rz(-pi/2) q[63];
sx q[64];
rz(-pi) q[64];
cx q[64],q[63];
rz(1.5217089) q[63];
sx q[64];
cx q[64],q[63];
x q[63];
rz(3.0925053) q[63];
rz(-1.6689711) q[64];
rz(-pi/16) q[65];
cx q[64],q[65];
rz(pi/32) q[65];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-pi/16) q[62];
cx q[62],q[61];
rz(pi/16) q[61];
cx q[62],q[61];
rz(-pi/16) q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(2.3439226) q[61];
sx q[61];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(pi/8) q[64];
cx q[64],q[54];
rz(pi/8) q[54];
cx q[64],q[54];
rz(-pi/8) q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[64],q[63];
rz(pi/4) q[63];
cx q[64],q[63];
rz(-pi/4) q[63];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(-pi/128) q[63];
cx q[63],q[62];
rz(pi/128) q[62];
cx q[63],q[62];
rz(-0.80994186) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[62],q[61];
rz(-pi/2) q[61];
sx q[62];
rz(-pi) q[62];
cx q[62],q[61];
rz(1.5585245) q[61];
sx q[62];
cx q[62],q[61];
rz(0.77312632) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[42],q[41];
rz(pi/512) q[41];
cx q[42],q[41];
rz(3.1354567) q[41];
sx q[41];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
rz(-pi/256) q[44];
rz(-1.5738643) q[53];
sx q[53];
rz(pi/2) q[53];
cx q[53],q[41];
rz(-pi/2) q[41];
sx q[53];
rz(-pi) q[53];
cx q[53],q[41];
rz(1.5677284) q[41];
sx q[53];
cx q[53],q[41];
rz(pi/2) q[41];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
x q[53];
rz(1.5677284) q[53];
cx q[59],q[60];
rz(pi/4096) q[60];
cx q[59],q[60];
sx q[59];
rz(-pi/4096) q[60];
x q[62];
rz(3*pi/4) q[62];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-pi/64) q[64];
rz(-pi/32) q[65];
cx q[64],q[65];
rz(pi/64) q[65];
cx q[64],q[65];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-pi/32) q[62];
cx q[62],q[61];
rz(pi/32) q[61];
cx q[62],q[61];
rz(-pi/32) q[61];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-pi/128) q[64];
rz(-pi/64) q[65];
cx q[64],q[65];
rz(pi/128) q[65];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-pi/64) q[62];
cx q[62],q[61];
rz(pi/64) q[61];
cx q[62],q[61];
rz(-pi/64) q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
rz(-pi/4) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[60],q[59];
rz(-pi/2) q[59];
sx q[60];
rz(-pi) q[60];
cx q[60],q[59];
rz(1.5692623) q[59];
sx q[60];
cx q[60],q[59];
rz(0.78386418) q[59];
x q[60];
rz(3*pi/4) q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(-1.5738643) q[53];
sx q[53];
rz(pi/2) q[53];
rz(-pi/16384) q[60];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
rz(-pi/16) q[54];
cx q[54],q[45];
rz(pi/16) q[45];
cx q[54],q[45];
rz(-pi/16) q[45];
rz(-pi/128) q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[44],q[45];
rz(pi/256) q[45];
cx q[44],q[45];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
rz(-pi/128) q[43];
cx q[43],q[42];
rz(pi/128) q[42];
cx q[43],q[42];
rz(-pi/128) q[42];
rz(-pi/512) q[44];
sx q[44];
rz(-pi) q[44];
rz(-pi/256) q[45];
sx q[45];
rz(-pi/2) q[45];
cx q[45],q[44];
rz(-pi/2) q[44];
sx q[45];
rz(-pi) q[45];
cx q[45],q[44];
rz(1.5646604) q[44];
sx q[45];
cx q[45],q[44];
x q[44];
rz(3.1354567) q[44];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
rz(-pi) q[41];
sx q[41];
rz(-pi/2) q[45];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(-0.036815539) q[44];
cx q[44],q[43];
rz(pi/256) q[43];
cx q[44],q[43];
rz(0.77312632) q[43];
sx q[43];
rz(pi/2) q[43];
sx q[44];
rz(-pi) q[44];
rz(-pi/64) q[45];
sx q[45];
rz(-pi) q[45];
cx q[53],q[41];
rz(-pi/2) q[41];
sx q[53];
rz(-pi) q[53];
cx q[53],q[41];
rz(1.5677284) q[41];
sx q[53];
cx q[53],q[41];
rz(pi/2) q[41];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
rz(-0.00067111659) q[41];
rz(2.3500586) q[42];
sx q[42];
cx q[43],q[42];
rz(pi/2) q[42];
sx q[43];
rz(-pi) q[43];
cx q[43],q[42];
rz(1.5646604) q[42];
sx q[43];
cx q[43],q[42];
x q[42];
rz(2.3500586) q[42];
rz(2.3439226) q[43];
sx q[43];
rz(-pi) q[43];
x q[53];
rz(1.5677284) q[53];
rz(pi/8) q[64];
cx q[64],q[63];
rz(pi/8) q[63];
cx q[64],q[63];
rz(-pi/8) q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[64],q[65];
rz(pi/4) q[65];
cx q[64],q[65];
sx q[64];
rz(pi/2) q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-0.68722339) q[64];
cx q[64],q[54];
rz(pi/32) q[54];
cx q[64],q[54];
rz(-pi/32) q[54];
sx q[54];
rz(-pi/2) q[54];
cx q[54],q[45];
rz(-pi/2) q[45];
sx q[54];
rz(-pi) q[54];
cx q[54],q[45];
rz(1.5217089) q[45];
sx q[54];
cx q[54],q[45];
rz(pi/64) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[45],q[44];
rz(-pi/2) q[44];
sx q[45];
rz(-pi) q[45];
cx q[45],q[44];
rz(1.5462526) q[44];
sx q[45];
cx q[45],q[44];
rz(pi/128) q[44];
sx q[44];
rz(pi/2) q[44];
cx q[44],q[43];
rz(-pi/2) q[43];
sx q[44];
rz(-pi) q[44];
cx q[44],q[43];
rz(1.5585245) q[43];
sx q[44];
cx q[44],q[43];
x q[43];
rz(3.1293208) q[43];
rz(-1.59534) q[44];
sx q[44];
rz(-pi) q[44];
rz(-1.6198837) q[45];
sx q[45];
rz(-pi) q[45];
rz(-0.88357293) q[54];
sx q[54];
rz(-pi) q[54];
cx q[64],q[63];
rz(pi/16) q[63];
cx q[64],q[63];
rz(-pi/16) q[63];
rz(-pi/4) q[65];
cx q[64],q[65];
rz(pi/8) q[65];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(pi/4) q[63];
cx q[63],q[62];
rz(pi/4) q[62];
cx q[63],q[62];
rz(-pi/4) q[62];
sx q[63];
rz(pi/2) q[63];
rz(-pi/4) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[64],q[54];
rz(-pi/2) q[54];
sx q[64];
rz(-pi) q[64];
cx q[64],q[54];
rz(1.4726216) q[54];
sx q[64];
cx q[64],q[54];
rz(-0.68722339) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[54],q[45];
rz(-pi/2) q[45];
sx q[54];
rz(-pi) q[54];
cx q[54],q[45];
rz(1.5217089) q[45];
sx q[54];
cx q[54],q[45];
rz(pi/64) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[45],q[44];
rz(-pi/2) q[44];
sx q[45];
rz(-pi) q[45];
cx q[45],q[44];
rz(1.5462526) q[44];
sx q[45];
cx q[45],q[44];
rz(pi/128) q[44];
sx q[44];
rz(pi/2) q[44];
rz(-pi/2) q[45];
rz(-pi/2) q[54];
rz(-13*pi/16) q[64];
rz(-pi/8) q[65];
cx q[64],q[65];
rz(pi/16) q[65];
cx q[64],q[65];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(pi/8) q[63];
cx q[63],q[62];
rz(pi/8) q[62];
cx q[63],q[62];
rz(-pi/8) q[62];
cx q[63],q[64];
rz(pi/4) q[64];
cx q[63],q[64];
sx q[63];
rz(pi/2) q[63];
rz(-pi/4) q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
rz(-pi/32) q[64];
rz(-pi/16) q[65];
cx q[64],q[65];
rz(pi/32) q[65];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(-pi/16) q[63];
cx q[63],q[62];
rz(pi/16) q[62];
cx q[63],q[62];
rz(-pi/16) q[62];
cx q[62],q[72];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
rz(-pi/64) q[64];
rz(-pi/32) q[65];
cx q[64],q[65];
rz(pi/64) q[65];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[54],q[64];
rz(0.49087385) q[63];
cx q[64],q[54];
rz(pi/8) q[54];
cx q[54],q[45];
rz(pi/8) q[45];
cx q[54],q[45];
rz(-pi/8) q[45];
rz(pi/4) q[64];
cx q[54],q[64];
sx q[54];
rz(pi/2) q[54];
rz(-pi/4) q[64];
rz(-pi/64) q[65];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
sx q[45];
rz(-pi/2) q[45];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
rz(-pi/4) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[72],q[62];
cx q[62],q[72];
rz(-0.0026844664) q[62];
cx q[62],q[61];
rz(pi/8192) q[61];
cx q[62],q[61];
rz(-pi/8192) q[61];
cx q[60],q[61];
rz(pi/16384) q[61];
cx q[60],q[61];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
rz(-0.0011504856) q[59];
rz(-pi/16384) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[41],q[53];
rz(pi/32768) q[53];
cx q[41],q[53];
rz(-pi/32768) q[53];
cx q[62],q[61];
rz(pi/4096) q[61];
cx q[62],q[61];
rz(-pi/4096) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[59],q[60];
rz(pi/8192) q[60];
cx q[59],q[60];
rz(-pi/8192) q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[41],q[53];
rz(pi/16384) q[53];
cx q[41],q[53];
rz(-pi/16384) q[53];
cx q[62],q[61];
rz(pi/2048) q[61];
cx q[62],q[61];
rz(-pi/2048) q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[59],q[60];
rz(pi/4096) q[60];
cx q[59],q[60];
rz(-pi/4096) q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[41],q[53];
rz(pi/8192) q[53];
cx q[41],q[53];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
rz(-pi/2) q[41];
sx q[41];
rz(-pi) q[41];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
sx q[42];
rz(-pi/2) q[42];
rz(-pi/8192) q[53];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
rz(-1.5738643) q[53];
sx q[53];
rz(-pi/2) q[53];
cx q[53],q[41];
rz(pi/2) q[41];
sx q[53];
rz(-pi) q[53];
cx q[53],q[41];
rz(1.5677284) q[41];
sx q[53];
cx q[53],q[41];
rz(1.5646604) q[41];
sx q[41];
rz(-pi) q[41];
cx q[42],q[41];
rz(-pi/2) q[41];
sx q[42];
rz(-pi) q[42];
cx q[42],q[41];
rz(1.5646604) q[41];
sx q[42];
cx q[42],q[41];
x q[41];
rz(3.1354567) q[41];
rz(-pi/2) q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
rz(-0.0023009712) q[42];
rz(-pi/256) q[43];
sx q[43];
rz(-pi) q[43];
cx q[44],q[43];
rz(-pi/2) q[43];
sx q[44];
rz(-pi) q[44];
cx q[44],q[43];
rz(1.5585245) q[43];
sx q[44];
cx q[44],q[43];
x q[43];
rz(3.1293208) q[43];
rz(-1.59534) q[44];
sx q[44];
rz(-pi) q[44];
cx q[45],q[44];
rz(-pi/2) q[44];
sx q[45];
rz(-pi) q[45];
cx q[45],q[44];
rz(1.5462526) q[44];
sx q[45];
cx q[45],q[44];
x q[44];
rz(3.117049) q[44];
rz(-pi/2) q[45];
cx q[45],q[54];
x q[53];
rz(-pi/1024) q[53];
cx q[54],q[45];
cx q[45],q[54];
rz(pi/4) q[61];
sx q[61];
rz(-pi) q[61];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[63],q[62];
rz(pi/32) q[62];
cx q[63],q[62];
rz(-pi/32) q[62];
sx q[63];
rz(-pi) q[63];
cx q[64],q[63];
rz(-pi/2) q[63];
sx q[64];
rz(-pi) q[64];
cx q[64],q[63];
rz(7*pi/16) q[63];
sx q[64];
cx q[64],q[63];
x q[63];
rz(-13*pi/16) q[63];
rz(-7*pi/8) q[64];
cx q[64],q[65];
rz(pi/8) q[65];
cx q[64],q[65];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
rz(pi/4) q[54];
cx q[54],q[45];
rz(pi/4) q[45];
cx q[54],q[45];
rz(-pi/4) q[45];
sx q[54];
rz(pi/2) q[54];
rz(-pi/64) q[64];
sx q[64];
rz(-pi/2) q[64];
rz(-pi/8) q[65];
sx q[72];
rz(-pi) q[72];
rz(-pi/65536) q[81];
sx q[81];
rz(-pi/2) q[81];
cx q[81],q[72];
rz(-pi/2) q[72];
sx q[81];
rz(-pi) q[81];
cx q[81],q[72];
rz(pi/2) q[72];
sx q[81];
cx q[81],q[72];
rz(-pi) q[72];
x q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(-0.78549404) q[62];
sx q[62];
rz(-pi/2) q[62];
cx q[62],q[61];
rz(-pi/2) q[61];
sx q[62];
rz(-pi) q[62];
cx q[62],q[61];
rz(1.5707005) q[61];
sx q[62];
cx q[62],q[61];
x q[61];
rz(-3.1421679) q[61];
cx q[61],q[60];
rz(pi/16384) q[60];
cx q[61],q[60];
rz(-pi/16384) q[60];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[60];
rz(-0.0046019424) q[60];
cx q[60],q[53];
rz(pi/2048) q[53];
cx q[60],q[53];
rz(-pi/2048) q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[53],q[41];
cx q[42],q[41];
rz(pi/4096) q[41];
cx q[42],q[41];
rz(-pi/4096) q[41];
cx q[60],q[53];
rz(pi/1024) q[53];
cx q[60],q[53];
rz(-pi/1024) q[53];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[42],q[41];
rz(pi/2048) q[41];
cx q[42],q[41];
rz(-pi/2048) q[41];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
rz(-pi/1024) q[43];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[53],q[41];
rz(-pi/512) q[41];
cx q[41],q[42];
rz(pi/512) q[42];
cx q[41],q[42];
rz(-pi/512) q[42];
cx q[43],q[42];
rz(pi/1024) q[42];
cx q[43],q[42];
rz(-pi/1024) q[42];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
rz(-pi/256) q[42];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[42],q[43];
rz(pi/256) q[43];
cx q[42],q[43];
rz(-pi/256) q[43];
rz(-pi/512) q[44];
cx q[44],q[43];
rz(pi/512) q[43];
cx q[44],q[43];
rz(-pi/512) q[43];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
rz(-0.036815539) q[43];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(-pi/128) q[54];
sx q[54];
rz(-pi) q[54];
rz(pi/4) q[60];
sx q[60];
rz(-pi) q[60];
sx q[61];
rz(-pi/2) q[61];
cx q[61],q[60];
rz(-pi/2) q[60];
sx q[61];
rz(-pi) q[61];
cx q[61],q[60];
rz(1.5704128) q[60];
sx q[61];
cx q[61],q[60];
x q[60];
rz(-2.3584955) q[60];
cx q[60],q[53];
rz(pi/4096) q[53];
cx q[60],q[53];
rz(-pi/4096) q[53];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[60],q[53];
rz(pi/2048) q[53];
cx q[60],q[53];
rz(-pi/2048) q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[53],q[41];
rz(-2.356578) q[61];
rz(-2.3562904) q[62];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
sx q[72];
rz(-pi) q[72];
rz(-1.5708443) q[81];
sx q[81];
rz(-pi) q[81];
rz(-pi/131072) q[82];
sx q[82];
rz(-pi/2) q[82];
cx q[82],q[81];
rz(-pi/2) q[81];
sx q[82];
rz(-pi) q[82];
cx q[82],q[81];
rz(pi/2) q[81];
sx q[82];
cx q[82],q[81];
rz(pi/65536) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[81],q[72];
rz(-pi/2) q[72];
sx q[81];
rz(-pi) q[81];
cx q[81],q[72];
rz(pi/2) q[72];
sx q[81];
cx q[81],q[72];
rz(-pi) q[72];
x q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(-0.001438107) q[60];
cx q[60],q[59];
rz(pi/32768) q[59];
cx q[60],q[59];
rz(-pi/32768) q[59];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[60],q[61];
rz(pi/16384) q[61];
cx q[60],q[61];
cx q[60],q[53];
rz(pi/8192) q[53];
cx q[60],q[53];
rz(-pi/8192) q[53];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[60],q[53];
rz(pi/4096) q[53];
cx q[60],q[53];
rz(-pi/4096) q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[53],q[41];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(-pi/16384) q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
rz(-pi/2048) q[60];
rz(-pi/1024) q[62];
cx q[62],q[61];
rz(pi/1024) q[61];
cx q[62],q[61];
rz(-pi/1024) q[61];
cx q[60],q[61];
rz(pi/2048) q[61];
cx q[60],q[61];
rz(-pi/2048) q[61];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
sx q[63];
rz(-pi) q[63];
cx q[64],q[63];
rz(-pi/2) q[63];
sx q[64];
rz(-pi) q[64];
cx q[64],q[63];
rz(1.5217089) q[63];
sx q[64];
cx q[64],q[63];
x q[63];
rz(3.0434179) q[63];
cx q[63],q[62];
rz(pi/32) q[62];
cx q[63],q[62];
rz(-pi/32) q[62];
rz(-1.6198837) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[64],q[54];
rz(-pi/2) q[54];
sx q[64];
rz(-pi) q[64];
cx q[64],q[54];
rz(1.5462526) q[54];
sx q[64];
cx q[64],q[54];
x q[54];
rz(3.117049) q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[43],q[44];
rz(pi/256) q[44];
cx q[43],q[44];
rz(-pi/256) q[44];
rz(-pi/2) q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(-pi/64) q[63];
cx q[63],q[62];
rz(pi/64) q[62];
cx q[63],q[62];
rz(-pi/64) q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[43],q[42];
rz(pi/128) q[42];
cx q[43],q[42];
rz(-pi/128) q[42];
cx q[62],q[72];
rz(-pi/16) q[64];
cx q[64],q[65];
rz(pi/16) q[65];
cx q[64],q[65];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
rz(pi/8) q[54];
cx q[54],q[45];
rz(pi/8) q[45];
cx q[54],q[45];
rz(-pi/8) q[45];
cx q[54],q[64];
rz(pi/4) q[64];
cx q[54],q[64];
sx q[54];
rz(pi/2) q[54];
rz(-pi/4) q[64];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(-pi/32) q[64];
rz(-pi/16) q[65];
cx q[64],q[65];
rz(pi/32) q[65];
cx q[64],q[65];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
rz(-pi/16) q[54];
cx q[54],q[45];
rz(pi/16) q[45];
cx q[54],q[45];
rz(-pi/16) q[45];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
rz(-pi/32) q[65];
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
rz(-pi/512) q[54];
cx q[54],q[45];
rz(pi/512) q[45];
cx q[54],q[45];
rz(-pi/512) q[45];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(pi/8) q[63];
cx q[63],q[62];
rz(pi/8) q[62];
cx q[63],q[62];
rz(-pi/8) q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[63],q[64];
rz(pi/4) q[64];
cx q[63],q[64];
sx q[63];
rz(pi/2) q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-pi/4) q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
rz(-pi/1024) q[54];
cx q[54],q[45];
rz(pi/1024) q[45];
cx q[54],q[45];
rz(-pi/1024) q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(-pi/256) q[44];
cx q[44],q[43];
rz(pi/256) q[43];
cx q[44],q[43];
rz(-pi/256) q[43];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
rz(-pi/512) q[44];
cx q[44],q[43];
rz(pi/512) q[43];
cx q[44],q[43];
rz(-pi/512) q[43];
rz(-0.14726216) q[64];
cx q[64],q[65];
rz(pi/64) q[65];
cx q[64],q[65];
cx q[64],q[54];
rz(pi/32) q[54];
cx q[64],q[54];
rz(-pi/32) q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-pi/16) q[62];
cx q[62],q[61];
rz(pi/16) q[61];
cx q[62],q[61];
rz(-pi/16) q[61];
cx q[63],q[62];
cx q[62],q[63];
rz(pi/4) q[62];
rz(pi/8) q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
rz(-pi/128) q[64];
rz(-pi/64) q[65];
cx q[64],q[65];
rz(pi/128) q[65];
cx q[64],q[65];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
rz(-pi/64) q[54];
cx q[54],q[45];
rz(pi/64) q[45];
cx q[54],q[45];
rz(-pi/64) q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[63],q[64];
rz(pi/8) q[64];
cx q[63],q[64];
cx q[63],q[62];
rz(-pi/4) q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
sx q[63];
rz(pi/2) q[63];
rz(-pi/8) q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(-pi/32) q[63];
cx q[63],q[62];
rz(pi/32) q[62];
cx q[63],q[62];
rz(-pi/32) q[62];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
rz(-pi/256) q[64];
rz(-pi/128) q[65];
cx q[64],q[65];
rz(pi/256) q[65];
cx q[64],q[65];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(-pi/128) q[45];
cx q[45],q[44];
rz(pi/128) q[44];
cx q[45],q[44];
rz(-pi/128) q[44];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(-pi/16) q[64];
cx q[64],q[54];
rz(pi/16) q[54];
cx q[64],q[54];
rz(-pi/16) q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(pi/32) q[45];
rz(-0.14726216) q[54];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(pi/8) q[62];
cx q[62],q[61];
rz(pi/8) q[61];
cx q[62],q[61];
rz(-pi/8) q[61];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[54],q[64];
cx q[62],q[63];
rz(pi/4) q[63];
cx q[62],q[63];
sx q[62];
rz(pi/2) q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(-pi/4) q[63];
rz(pi/64) q[64];
cx q[54],q[64];
cx q[54],q[45];
rz(-pi/32) q[45];
rz(-pi/64) q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[63],q[64];
cx q[64],q[63];
rz(-3*pi/16) q[63];
cx q[63],q[62];
rz(pi/16) q[62];
cx q[63],q[62];
rz(-pi/16) q[62];
rz(pi/8) q[64];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(pi/4) q[62];
cx q[62],q[61];
rz(pi/4) q[61];
cx q[62],q[61];
rz(-pi/4) q[61];
sx q[62];
rz(pi/2) q[62];
rz(-pi/8) q[64];
rz(-pi/256) q[65];
rz(-1.5708443) q[81];
rz(-1.5708203) q[82];
barrier q[112],q[57],q[2],q[121],q[66],q[11],q[8],q[75],q[82],q[106],q[17],q[72],q[26],q[90],q[35],q[99],q[60],q[108],q[42],q[105],q[50],q[114],q[44],q[4],q[123],q[1],q[68],q[13],q[41],q[10],q[77],q[74],q[19],q[61],q[28],q[92],q[37],q[101],q[34],q[98],q[63],q[107],q[52],q[116],q[54],q[6],q[125],q[3],q[70],q[67],q[12],q[76],q[21],q[85],q[30],q[94],q[27],q[91],q[103],q[36],q[100],q[53],q[109],q[45],q[118],q[59],q[64],q[5],q[124],q[69],q[14],q[78],q[23],q[87],q[20],q[32],q[84],q[96],q[29],q[93],q[38],q[102],q[47],q[111],q[56],q[43],q[120],q[117],q[83],q[7],q[126],q[71],q[16],q[80],q[25],q[89],q[22],q[86],q[31],q[95],q[40],q[104],q[49],q[46],q[113],q[58],q[110],q[55],q[122],q[0],q[119],q[81],q[9],q[73],q[18],q[62],q[15],q[79],q[115],q[24],q[88],q[33],q[97],q[65],q[39],q[51],q[48];
measure q[82] -> c[0];
measure q[81] -> c[1];
measure q[59] -> c[2];
measure q[60] -> c[3];
measure q[41] -> c[4];
measure q[53] -> c[5];
measure q[72] -> c[6];
measure q[42] -> c[7];
measure q[43] -> c[8];
measure q[65] -> c[9];
measure q[44] -> c[10];
measure q[54] -> c[11];
measure q[45] -> c[12];
measure q[63] -> c[13];
measure q[64] -> c[14];
measure q[61] -> c[15];
measure q[62] -> c[16];
