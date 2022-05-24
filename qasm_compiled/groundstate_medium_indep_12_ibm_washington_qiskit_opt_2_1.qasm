OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[12];
rz(-0.30436085) q[45];
rz(1.5852038) q[53];
rz(2.7948524) q[54];
rz(-0.38700345) q[59];
rz(2.5725692) q[60];
rz(0.37197485) q[61];
rz(2.3795938) q[62];
rz(2.2074299) q[63];
rz(-3.9081045267949) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[64],q[63];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[64],q[54];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[63],q[62];
cx q[64],q[54];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[54],q[64];
cx q[63],q[62];
cx q[64],q[54];
cx q[54],q[64];
cx q[54],q[45];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[45];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[62],q[61];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[61],q[60];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[45];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[54],q[64];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[54],q[45];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[64],q[63];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
rz(2.9979641) q[72];
cx q[62],q[72];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[60],q[59];
cx q[60],q[53];
cx q[62],q[72];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[59];
cx q[60],q[53];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[72];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[59];
cx q[60],q[53];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[64],q[63];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
cx q[64],q[54];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[60],q[59];
rz(pi/2) q[59];
sx q[59];
rz(pi/2) q[59];
cx q[60],q[53];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
cx q[63],q[64];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(-2.6065024267949) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
rz(2.3884928) q[82];
cx q[81],q[82];
cx q[81],q[72];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(1.0742371732051) q[81];
sx q[81];
rz(pi/2) q[81];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[72],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
rz(-4.3720696267949) q[72];
sx q[72];
rz(pi/2) q[72];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[62],q[61];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
rz(-1.5544444207949) q[62];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[61],q[62];
cx q[61],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[60];
rz(pi/2) q[59];
sx q[59];
rz(pi/2) q[59];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
rz(-2.9861841267949) q[61];
sx q[61];
rz(pi/2) q[61];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[53];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[82],q[81];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[82],q[81];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
cx q[60],q[59];
rz(-1.1742629167949) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[64],q[63];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[60],q[53];
rz(pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
cx q[60],q[59];
rz(pi/2) q[59];
sx q[59];
rz(pi/2) q[59];
rz(-3.9883408267949) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[72],q[62];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[62],q[72];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[72],q[62];
cx q[62],q[72];
cx q[62],q[61];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
rz(pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
cx q[45],q[44];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[44];
rz(pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[61],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[61],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
rz(-2.3747574867949) q[61];
sx q[61];
rz(pi/2) q[61];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[82],q[81];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[82],q[81];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
cx q[62],q[72];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[63],q[62];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[62];
cx q[61],q[62];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[62],q[61];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
rz(-3.3346303267949) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[44],q[45];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[63],q[62];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[54],q[64];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[60],q[61];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[60];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(0.97258685) q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[64],q[63];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[64],q[63];
cx q[54],q[64];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
cx q[60],q[61];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[60];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[64],q[63];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
rz(-1.341412) q[64];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[54],q[64];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
rz(-3.0980229) q[54];
cx q[45],q[54];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[44],q[45];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
rz(pi/2) q[64];
sx q[64];
rz(1.9458695) q[64];
cx q[54],q[64];
rz(-2.8011713267949) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[64];
rz(-3.9598250267949) q[54];
sx q[54];
rz(pi/2) q[54];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[82],q[81];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[72];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[61];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[82],q[81];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
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
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[43],q[44];
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
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[54];
cx q[53],q[41];
cx q[41],q[53];
cx q[53],q[41];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[64];
rz(-4.1496217267949) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[59],q[60];
cx q[62],q[61];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[59];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[82],q[81];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[82],q[81];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[82],q[81];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[81],q[72];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[60],q[61];
rz(0.805364973205104) q[60];
sx q[60];
rz(pi/2) q[60];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[60];
cx q[60],q[61];
rz(-1.4921953337949) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
cx q[62],q[72];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[64],q[63];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
rz(-3.6421619267949) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[72];
cx q[62],q[72];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[62];
cx q[61],q[62];
rz(-3.0141123267949) q[61];
sx q[61];
rz(pi/2) q[61];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(-0.106561926794897) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[82],q[81];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[62],q[72];
rz(-1.7671772567949) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[81],q[72];
rz(pi/2) q[72];
sx q[72];
rz(-1.4685115) q[72];
sx q[72];
rz(pi/2) q[72];
rz(-0.214479226794897) q[81];
sx q[81];
rz(pi/2) q[81];
rz(-0.102615426794897) q[82];
sx q[82];
rz(pi/2) q[82];
barrier q[38],q[102],q[47],q[111],q[56],q[62],q[120],q[117],q[59],q[7],q[126],q[71],q[16],q[80],q[13],q[25],q[77],q[89],q[22],q[86],q[31],q[95],q[40],q[104],q[49],q[46],q[113],q[110],q[55],q[0],q[119],q[44],q[9],q[73],q[18],q[81],q[15],q[79],q[24],q[88],q[33],q[97],q[41],q[39],q[106],q[51],q[103],q[48],q[115],q[112],q[57],q[2],q[121],q[66],q[11],q[75],q[8],q[61],q[17],q[72],q[26],q[90],q[35],q[99],q[43],q[53],q[108],q[105],q[50],q[114],q[82],q[4],q[123],q[68],q[1],q[65],q[10],q[74],q[19],q[83],q[28],q[92],q[37],q[34],q[101],q[98],q[42],q[107],q[52],q[116],q[64],q[6],q[125],q[58],q[70],q[3],q[122],q[67],q[12],q[76],q[21],q[85],q[96],q[30],q[27],q[94],q[91],q[36],q[100],q[60],q[109],q[54],q[118],q[45],q[63],q[5],q[124],q[69],q[14],q[78],q[87],q[23],q[20],q[32],q[84],q[29],q[93];
measure q[44] -> meas[0];
measure q[45] -> meas[1];
measure q[54] -> meas[2];
measure q[59] -> meas[3];
measure q[60] -> meas[4];
measure q[64] -> meas[5];
measure q[63] -> meas[6];
measure q[61] -> meas[7];
measure q[82] -> meas[8];
measure q[62] -> meas[9];
measure q[81] -> meas[10];
measure q[72] -> meas[11];
