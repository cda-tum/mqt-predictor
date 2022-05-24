OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[9];
rz(pi/2) q[62];
sx q[62];
rz(-pi/2) q[62];
rz(pi/2) q[72];
sx q[72];
rz(-pi/2) q[72];
rz(pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
rz(-pi) q[81];
sx q[81];
rz(3*pi/8) q[81];
sx q[81];
rz(-pi) q[82];
sx q[82];
rz(1.4762895) q[82];
sx q[82];
rz(-pi) q[83];
sx q[83];
rz(1.4822541) q[83];
sx q[83];
rz(-pi) q[84];
sx q[84];
rz(1.6278069) q[84];
sx q[84];
cx q[84],q[83];
rz(-pi) q[83];
sx q[83];
rz(2.2126461) q[83];
sx q[83];
cx q[84],q[83];
rz(-pi) q[85];
sx q[85];
rz(1.4620893) q[85];
sx q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
rz(-pi) q[84];
sx q[84];
rz(2.8561654) q[84];
sx q[84];
cx q[85],q[84];
rz(-pi) q[84];
sx q[84];
rz(3.0100767) q[84];
sx q[84];
cx q[83],q[84];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(-pi) q[84];
sx q[84];
rz(2.5504747) q[84];
sx q[84];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi) q[83];
sx q[83];
rz(3.048471) q[83];
sx q[83];
cx q[82],q[83];
rz(-pi) q[83];
sx q[83];
rz(3.1183126) q[83];
sx q[83];
cx q[84],q[83];
rz(-pi) q[83];
sx q[83];
rz(2.9549245) q[83];
sx q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[85],q[84];
rz(-pi) q[84];
sx q[84];
rz(3.0402317) q[84];
sx q[84];
cx q[83],q[84];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(-pi) q[84];
sx q[84];
rz(3.1315153) q[84];
sx q[84];
cx q[83],q[84];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(-pi) q[84];
sx q[84];
rz(3.0922683) q[84];
sx q[84];
cx q[83],q[84];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
sx q[83];
rz(3.095797) q[83];
sx q[83];
rz(-pi) q[83];
rz(-pi) q[84];
sx q[84];
rz(2.7929039) q[84];
sx q[84];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi) q[83];
sx q[83];
rz(3.095797) q[83];
sx q[83];
cx q[84],q[83];
sx q[83];
rz(3.0500013) q[83];
sx q[83];
rz(-pi) q[83];
cx q[82],q[83];
rz(-pi) q[83];
sx q[83];
rz(3.0500013) q[83];
sx q[83];
cx q[82],q[83];
x q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
sx q[83];
rz(2.95841) q[83];
sx q[83];
rz(-pi) q[83];
cx q[82],q[83];
rz(-pi) q[83];
sx q[83];
rz(2.95841) q[83];
sx q[83];
cx q[82],q[83];
x q[82];
sx q[83];
rz(2.7752273) q[83];
sx q[83];
rz(-pi) q[83];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
cx q[84],q[83];
rz(-pi) q[83];
sx q[83];
rz(2.7752273) q[83];
sx q[83];
cx q[84],q[83];
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
cx q[64],q[63];
x q[63];
cx q[63],q[62];
rz(-pi/4) q[62];
cx q[72],q[62];
rz(pi/4) q[62];
cx q[63],q[62];
rz(-pi/4) q[62];
rz(pi/4) q[63];
cx q[72],q[62];
rz(-pi/4) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[62],q[63];
rz(pi/4) q[62];
rz(-pi/4) q[63];
cx q[62],q[63];
cx q[72],q[81];
rz(-pi/4) q[81];
cx q[82],q[81];
rz(pi/4) q[81];
cx q[72],q[81];
rz(pi/4) q[72];
rz(-pi/4) q[81];
cx q[82],q[81];
rz(3*pi/4) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
rz(-pi/4) q[72];
rz(pi/4) q[81];
cx q[81],q[72];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[82],q[83];
rz(-pi/4) q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[85],q[84];
rz(pi/4) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[82],q[83];
rz(pi/4) q[82];
rz(-pi/4) q[83];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[84],q[83];
rz(3*pi/4) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[83],q[82];
rz(-pi/4) q[82];
rz(pi/4) q[83];
cx q[83],q[82];
rz(pi/2) q[82];
sx q[82];
rz(-pi/2) q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[72],q[81];
rz(-pi/4) q[81];
cx q[82],q[81];
rz(pi/4) q[81];
cx q[72],q[81];
rz(pi/4) q[72];
rz(-pi/4) q[81];
cx q[82],q[81];
rz(pi/4) q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
rz(-pi/4) q[72];
rz(pi/4) q[81];
cx q[81],q[72];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[63],q[62];
rz(-pi/4) q[62];
cx q[72],q[62];
rz(pi/4) q[62];
cx q[63],q[62];
rz(-pi/4) q[62];
rz(pi/4) q[63];
cx q[72],q[62];
rz(pi/4) q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[62],q[63];
rz(pi/4) q[62];
rz(-pi/4) q[63];
cx q[62],q[63];
x q[62];
cx q[64],q[63];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
x q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[84],q[85];
rz(-pi) q[85];
sx q[85];
rz(7*pi/8) q[85];
sx q[85];
cx q[84],q[85];
sx q[85];
rz(7*pi/8) q[85];
sx q[85];
rz(-pi) q[85];
cx q[84],q[85];
sx q[85];
rz(3.1186948) q[85];
sx q[85];
rz(-pi) q[85];
cx q[84],q[85];
sx q[85];
rz(1.5936942) q[85];
sx q[85];
cx q[73],q[85];
rz(-pi/4) q[85];
cx q[84],q[85];
rz(pi/4) q[85];
cx q[73],q[85];
rz(pi/4) q[73];
rz(-pi/4) q[85];
cx q[84],q[85];
rz(3*pi/4) q[85];
sx q[85];
rz(pi/2) q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[73];
rz(-pi/4) q[73];
rz(pi/4) q[85];
cx q[85],q[73];
cx q[85],q[84];
rz(-pi) q[84];
sx q[84];
rz(3.1186948) q[84];
sx q[84];
cx q[85],q[84];
sx q[84];
rz(1.5478985) q[84];
sx q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[73],q[85];
rz(-pi/4) q[85];
cx q[84],q[85];
rz(pi/4) q[85];
cx q[73],q[85];
rz(pi/4) q[73];
rz(-pi/4) q[85];
cx q[84],q[85];
rz(3*pi/4) q[85];
sx q[85];
rz(pi/2) q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[73];
rz(-pi/4) q[73];
rz(pi/4) q[85];
cx q[85],q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[64],q[63];
cx q[63],q[62];
rz(-pi/4) q[62];
cx q[85],q[84];
sx q[84];
rz(3.095797) q[84];
sx q[84];
rz(-pi) q[84];
cx q[85],q[84];
sx q[84];
rz(1.616592) q[84];
sx q[84];
cx q[83],q[84];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(pi/4) q[84];
cx q[83],q[84];
rz(pi/4) q[83];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(3*pi/4) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(pi/4) q[84];
cx q[84],q[83];
cx q[84],q[85];
rz(-pi) q[85];
sx q[85];
rz(3.095797) q[85];
sx q[85];
cx q[84],q[85];
sx q[85];
rz(1.5250007) q[85];
sx q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(pi/4) q[84];
cx q[83],q[84];
rz(pi/4) q[83];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(3*pi/4) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(pi/4) q[84];
cx q[84],q[83];
x q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[72],q[62];
rz(pi/4) q[62];
cx q[63],q[62];
rz(-pi/4) q[62];
rz(pi/4) q[63];
cx q[72],q[62];
rz(3*pi/4) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[62],q[63];
rz(pi/4) q[62];
rz(-pi/4) q[63];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[72],q[81];
rz(-pi/4) q[81];
cx q[84],q[85];
sx q[85];
rz(3.0500013) q[85];
sx q[85];
rz(-pi) q[85];
cx q[84],q[85];
sx q[85];
rz(1.6623877) q[85];
sx q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(pi/4) q[84];
cx q[83],q[84];
rz(pi/4) q[83];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(3*pi/4) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(pi/4) q[84];
cx q[84],q[83];
cx q[84],q[85];
rz(-pi) q[85];
sx q[85];
rz(3.0500013) q[85];
sx q[85];
cx q[84],q[85];
sx q[85];
rz(1.479205) q[85];
sx q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(pi/4) q[84];
cx q[83],q[84];
rz(pi/4) q[83];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(3*pi/4) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(pi/4) q[84];
cx q[84],q[83];
x q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[81];
rz(pi/4) q[81];
cx q[72],q[81];
rz(pi/4) q[72];
rz(-pi/4) q[81];
cx q[82],q[81];
rz(-pi/4) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
rz(-pi/4) q[72];
rz(pi/4) q[81];
cx q[81],q[72];
cx q[84],q[85];
sx q[85];
rz(2.95841) q[85];
sx q[85];
rz(-pi) q[85];
cx q[84],q[85];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
sx q[85];
rz(1.753979) q[85];
sx q[85];
cx q[84],q[85];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi/4) q[85];
cx q[84],q[85];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
rz(pi/4) q[85];
cx q[84],q[85];
rz(pi/4) q[84];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi/4) q[85];
cx q[84],q[85];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(pi/4) q[84];
rz(3*pi/4) q[85];
sx q[85];
rz(pi/2) q[85];
cx q[84],q[85];
rz(-pi) q[85];
sx q[85];
rz(2.95841) q[85];
sx q[85];
cx q[84],q[85];
cx q[83],q[84];
cx q[84],q[83];
sx q[85];
rz(1.3876137) q[85];
sx q[85];
cx q[84],q[85];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi/4) q[85];
cx q[84],q[85];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
rz(pi/4) q[85];
cx q[84],q[85];
rz(pi/4) q[84];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi/4) q[85];
cx q[84],q[85];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(3*pi/4) q[84];
cx q[84],q[83];
sx q[84];
rz(pi/2) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[82],q[83];
rz(-pi/4) q[83];
cx q[84],q[83];
rz(pi/4) q[83];
cx q[82],q[83];
rz(pi/4) q[82];
rz(-pi/4) q[83];
cx q[84],q[83];
rz(pi/4) q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[83],q[82];
rz(-pi/4) q[82];
rz(pi/4) q[83];
cx q[83],q[82];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[72],q[81];
rz(-pi/4) q[81];
cx q[82],q[81];
rz(pi/4) q[81];
cx q[72],q[81];
rz(pi/4) q[72];
rz(-pi/4) q[81];
cx q[82],q[81];
rz(pi/4) q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
rz(-pi/4) q[72];
rz(pi/4) q[81];
cx q[81],q[72];
rz(pi/2) q[72];
sx q[72];
rz(-pi/2) q[72];
cx q[62],q[72];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-pi/4) q[72];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(pi/4) q[72];
cx q[62],q[72];
rz(pi/4) q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-pi/4) q[63];
rz(-pi/4) q[72];
cx q[62],q[72];
rz(pi/4) q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(pi/4) q[72];
cx q[62],q[72];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-pi/4) q[72];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(pi/4) q[72];
cx q[62],q[72];
rz(pi/4) q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-pi/4) q[63];
rz(-pi/4) q[72];
cx q[62],q[72];
rz(pi/4) q[62];
cx q[62],q[63];
rz(-pi/4) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[82];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
rz(-pi/4) q[82];
cx q[81],q[82];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/4) q[82];
cx q[81],q[82];
rz(pi/4) q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(-pi/4) q[72];
rz(-pi/4) q[82];
cx q[81],q[82];
rz(pi/4) q[81];
cx q[81],q[72];
rz(3*pi/4) q[82];
sx q[82];
rz(3*pi/4) q[82];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[82],q[83];
rz(-pi/4) q[83];
cx q[84],q[83];
rz(pi/4) q[83];
cx q[82],q[83];
rz(-pi/4) q[83];
cx q[84],q[83];
rz(3*pi/4) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[83],q[82];
rz(-pi/4) q[82];
rz(pi/2) q[83];
cx q[83],q[82];
rz(pi/2) q[82];
sx q[82];
rz(-pi/2) q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[72],q[81];
rz(-pi/4) q[81];
cx q[82],q[81];
rz(pi/4) q[81];
cx q[72],q[81];
rz(pi/4) q[72];
rz(-pi/4) q[81];
cx q[82],q[81];
rz(pi/4) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[82],q[81];
rz(-pi/4) q[81];
rz(pi/4) q[82];
cx q[82],q[81];
rz(pi/2) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[63],q[62];
rz(-pi/4) q[62];
cx q[72],q[62];
rz(pi/4) q[62];
cx q[63],q[62];
rz(-pi/4) q[62];
rz(pi/4) q[63];
cx q[72],q[62];
rz(pi/4) q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[72],q[62];
rz(-pi/4) q[62];
rz(pi/4) q[72];
cx q[72],q[62];
x q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[64],q[63];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
x q[72];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
x q[82];
rz(3*pi/4) q[85];
sx q[85];
rz(-pi/2) q[85];
cx q[84],q[85];
rz(-pi) q[85];
sx q[85];
rz(2.95841) q[85];
sx q[85];
cx q[84],q[85];
sx q[85];
rz(1.3876137) q[85];
sx q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(pi/4) q[84];
cx q[83],q[84];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(3*pi/4) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(pi/4) q[84];
cx q[84],q[83];
cx q[84],q[85];
sx q[85];
rz(2.95841) q[85];
sx q[85];
rz(-pi) q[85];
cx q[84],q[85];
sx q[85];
rz(1.753979) q[85];
sx q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(pi/4) q[84];
cx q[83],q[84];
rz(pi/4) q[83];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(3*pi/4) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(pi/4) q[84];
cx q[84],q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[84],q[85];
rz(-pi) q[85];
sx q[85];
rz(3.0500013) q[85];
sx q[85];
cx q[84],q[85];
sx q[85];
rz(1.479205) q[85];
sx q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(pi/4) q[84];
cx q[83],q[84];
rz(pi/4) q[83];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(3*pi/4) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(pi/4) q[84];
cx q[84],q[83];
cx q[84],q[85];
sx q[85];
rz(3.0500013) q[85];
sx q[85];
rz(-pi) q[85];
cx q[84],q[85];
sx q[85];
rz(1.6623877) q[85];
sx q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(pi/4) q[84];
cx q[83],q[84];
rz(pi/4) q[83];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(3*pi/4) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(pi/4) q[84];
cx q[84],q[83];
x q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[84],q[85];
rz(-pi) q[85];
sx q[85];
rz(3.095797) q[85];
sx q[85];
cx q[84],q[85];
sx q[85];
rz(1.5250007) q[85];
sx q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(pi/4) q[84];
cx q[83],q[84];
rz(pi/4) q[83];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(3*pi/4) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(pi/4) q[84];
cx q[84],q[83];
cx q[84],q[85];
sx q[85];
rz(3.095797) q[85];
sx q[85];
rz(-pi) q[85];
cx q[84],q[85];
sx q[85];
rz(1.616592) q[85];
sx q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(pi/4) q[84];
cx q[83],q[84];
rz(pi/4) q[83];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(3*pi/4) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(pi/4) q[84];
cx q[84],q[83];
x q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[84],q[85];
rz(-pi) q[85];
sx q[85];
rz(3.1186948) q[85];
sx q[85];
cx q[84],q[85];
sx q[85];
rz(1.5478985) q[85];
sx q[85];
cx q[73],q[85];
rz(-pi/4) q[85];
cx q[84],q[85];
rz(pi/4) q[85];
cx q[73],q[85];
rz(pi/4) q[73];
rz(-pi/4) q[85];
cx q[84],q[85];
rz(3*pi/4) q[85];
sx q[85];
rz(pi/2) q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[73];
rz(-pi/4) q[73];
rz(pi/4) q[85];
cx q[85],q[73];
cx q[85],q[84];
sx q[84];
rz(3.1186948) q[84];
sx q[84];
rz(-pi) q[84];
cx q[85],q[84];
sx q[84];
rz(1.5936942) q[84];
sx q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[73],q[85];
rz(-pi/4) q[85];
cx q[84],q[85];
rz(pi/4) q[85];
cx q[73],q[85];
rz(pi/4) q[73];
rz(-pi/4) q[85];
cx q[84],q[85];
rz(3*pi/4) q[85];
sx q[85];
rz(pi/2) q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[84],q[85];
rz(3*pi/4) q[84];
rz(-pi/4) q[85];
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
cx q[64],q[63];
x q[63];
cx q[63],q[62];
rz(-pi/4) q[62];
cx q[72],q[62];
rz(pi/4) q[62];
cx q[63],q[62];
rz(-pi/4) q[62];
rz(pi/4) q[63];
cx q[72],q[62];
rz(3*pi/4) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[62],q[63];
rz(pi/4) q[62];
rz(-pi/4) q[63];
cx q[62],q[63];
cx q[72],q[81];
rz(-pi/4) q[81];
cx q[82],q[81];
rz(pi/4) q[81];
cx q[72],q[81];
rz(pi/4) q[72];
rz(-pi/4) q[81];
cx q[82],q[81];
rz(-pi/4) q[81];
sx q[81];
rz(pi/2) q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
rz(-pi/4) q[72];
rz(pi/4) q[81];
cx q[81],q[72];
cx q[84],q[85];
sx q[85];
rz(7*pi/8) q[85];
sx q[85];
rz(-pi) q[85];
cx q[84],q[85];
sx q[84];
rz(pi/2) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[82],q[83];
rz(-pi/4) q[83];
cx q[84],q[83];
rz(pi/4) q[83];
cx q[82],q[83];
rz(pi/4) q[82];
rz(-pi/4) q[83];
cx q[84],q[83];
rz(pi/4) q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[83],q[82];
rz(-pi/4) q[82];
rz(pi/4) q[83];
cx q[83],q[82];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[72],q[81];
rz(-pi/4) q[81];
cx q[82],q[81];
rz(pi/4) q[81];
cx q[72],q[81];
rz(pi/4) q[72];
rz(-pi/4) q[81];
cx q[82],q[81];
rz(pi/4) q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
rz(-pi/4) q[72];
rz(pi/4) q[81];
cx q[81],q[72];
rz(pi/2) q[72];
sx q[72];
rz(-pi/2) q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[63],q[62];
rz(-pi/4) q[62];
cx q[72],q[62];
rz(pi/4) q[62];
cx q[63],q[62];
rz(-pi/4) q[62];
rz(pi/4) q[63];
cx q[72],q[62];
rz(pi/4) q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[62],q[63];
rz(pi/4) q[62];
rz(-pi/4) q[63];
cx q[62],q[63];
x q[62];
x q[63];
cx q[64],q[63];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
x q[81];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(-pi) q[85];
sx q[85];
rz(2.3825283) q[85];
sx q[85];
cx q[84],q[85];
sx q[85];
rz(2.7752273) q[85];
sx q[85];
rz(-pi) q[85];
cx q[84],q[85];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
rz(-pi) q[85];
sx q[85];
rz(2.95841) q[85];
sx q[85];
cx q[84],q[85];
sx q[85];
rz(2.95841) q[85];
sx q[85];
rz(-pi) q[85];
cx q[84],q[85];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
rz(-pi) q[85];
sx q[85];
rz(3.0500013) q[85];
sx q[85];
cx q[84],q[85];
sx q[85];
rz(3.0500013) q[85];
sx q[85];
rz(-pi) q[85];
cx q[84],q[85];
rz(-pi) q[85];
sx q[85];
rz(3.095797) q[85];
sx q[85];
cx q[73],q[85];
sx q[85];
rz(3.095797) q[85];
sx q[85];
rz(-pi) q[85];
cx q[73],q[85];
sx q[85];
rz(-5*pi/8) q[85];
sx q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[83],q[84];
sx q[84];
rz(2.7929039) q[84];
sx q[84];
rz(-pi) q[84];
cx q[85],q[84];
sx q[84];
rz(3.0922683) q[84];
sx q[84];
rz(-pi) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[82],q[83];
sx q[83];
rz(3.1315153) q[83];
sx q[83];
rz(-pi) q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[85],q[84];
sx q[84];
rz(3.0402317) q[84];
sx q[84];
rz(-pi) q[84];
cx q[83],q[84];
sx q[84];
rz(2.9549245) q[84];
sx q[84];
rz(-pi) q[84];
cx q[85],q[84];
sx q[84];
rz(3.1183126) q[84];
sx q[84];
rz(-pi) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[82],q[83];
sx q[83];
rz(3.048471) q[83];
sx q[83];
rz(-pi) q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[85],q[84];
sx q[84];
rz(-1.6653033) q[84];
sx q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
cx q[83],q[84];
sx q[84];
rz(2.5504747) q[84];
sx q[84];
rz(-pi) q[84];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[82],q[83];
sx q[83];
rz(3.0100767) q[83];
sx q[83];
rz(-pi) q[83];
cx q[84],q[83];
sx q[83];
rz(2.8561654) q[83];
sx q[83];
rz(-pi) q[83];
cx q[82],q[83];
sx q[83];
rz(-1.6795034) q[83];
sx q[83];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[82];
sx q[82];
rz(2.2126461) q[82];
sx q[82];
rz(-pi) q[82];
cx q[83],q[82];
sx q[82];
rz(-1.6593387) q[82];
sx q[82];
sx q[83];
rz(-1.5137858) q[83];
sx q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-5*pi/4) q[84];
cx q[85],q[73];
cx q[73],q[85];
cx q[84],q[85];
rz(-pi/4) q[85];
cx q[84],q[85];
sx q[84];
rz(3*pi/4) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[82],q[83];
rz(pi/4) q[83];
sx q[83];
rz(pi/2) q[83];
rz(pi/4) q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
rz(pi/4) q[83];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[84],q[83];
rz(-pi/4) q[83];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
rz(pi/4) q[83];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
cx q[84],q[83];
rz(pi/4) q[83];
sx q[83];
rz(3*pi/4) q[83];
cx q[82],q[83];
rz(pi/4) q[83];
sx q[83];
rz(-5*pi/4) q[83];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[83],q[84];
rz(pi/4) q[84];
cx q[83],q[84];
sx q[83];
rz(3*pi/4) q[83];
cx q[82],q[83];
rz(pi/4) q[83];
sx q[83];
rz(3*pi/4) q[83];
rz(-pi/4) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[85],q[84];
rz(-pi/4) q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
rz(pi/4) q[85];
cx q[84],q[85];
rz(-pi/4) q[85];
cx q[73],q[85];
rz(pi/16) q[73];
rz(-3*pi/2) q[85];
sx q[85];
rz(3*pi/4) q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[82],q[83];
rz(3*pi/4) q[83];
sx q[83];
rz(0.057010554) q[83];
sx q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
rz(-pi/16) q[85];
cx q[73],q[85];
rz(pi/16) q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[73],q[85];
rz(-pi/16) q[85];
cx q[85],q[84];
rz(pi/16) q[84];
cx q[85],q[84];
cx q[73],q[85];
rz(-pi/16) q[84];
rz(pi/16) q[85];
cx q[85],q[84];
rz(-pi/16) q[84];
cx q[85],q[84];
rz(pi/16) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[85],q[84];
cx q[73],q[85];
rz(-pi/16) q[84];
cx q[84],q[83];
rz(pi/16) q[83];
cx q[84],q[83];
rz(-pi/16) q[83];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[84];
rz(pi/16) q[84];
cx q[84],q[83];
rz(-pi/16) q[83];
cx q[84],q[83];
rz(pi/16) q[83];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[85],q[84];
rz(-pi/16) q[84];
cx q[84],q[83];
rz(pi/16) q[83];
cx q[84],q[83];
rz(-pi/16) q[83];
sx q[85];
rz(-1.6795034) q[85];
sx q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[84];
rz(pi/16) q[84];
cx q[84],q[83];
rz(-pi/16) q[83];
cx q[84],q[83];
rz(pi/16) q[83];
sx q[83];
rz(-5*pi/8) q[83];
sx q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
sx q[84];
rz(-1.6593387) q[84];
sx q[84];
cx q[83],q[84];
rz(-pi) q[84];
sx q[84];
rz(2.2126461) q[84];
sx q[84];
cx q[83],q[84];
sx q[85];
rz(-1.6653033) q[85];
sx q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[85],q[73];
rz(-pi) q[73];
sx q[73];
rz(2.8561654) q[73];
sx q[73];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
rz(-pi) q[84];
sx q[84];
rz(3.0100767) q[84];
sx q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[73],q[85];
rz(-pi) q[85];
sx q[85];
rz(2.5504747) q[85];
sx q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
cx q[84],q[85];
rz(-pi) q[85];
sx q[85];
rz(3.048471) q[85];
sx q[85];
cx q[73],q[85];
rz(-pi) q[85];
sx q[85];
rz(3.1183126) q[85];
sx q[85];
cx q[84],q[85];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi) q[85];
sx q[85];
rz(2.9549245) q[85];
sx q[85];
cx q[84],q[85];
rz(-pi) q[85];
sx q[85];
rz(3.0402317) q[85];
sx q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[83],q[84];
rz(-pi) q[84];
sx q[84];
rz(3.1315153) q[84];
sx q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[73],q[85];
rz(-pi) q[85];
sx q[85];
rz(3.0922683) q[85];
sx q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[83],q[84];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
sx q[83];
rz(3.095797) q[83];
sx q[83];
rz(-pi) q[83];
rz(-pi) q[84];
sx q[84];
rz(2.7929039) q[84];
sx q[84];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi) q[83];
sx q[83];
rz(3.095797) q[83];
sx q[83];
cx q[84],q[83];
sx q[83];
rz(3.0500013) q[83];
sx q[83];
rz(-pi) q[83];
cx q[82],q[83];
rz(-pi) q[83];
sx q[83];
rz(3.0500013) q[83];
sx q[83];
cx q[82],q[83];
x q[82];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[83],q[82];
x q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[62];
rz(-pi/4) q[62];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[72],q[62];
rz(pi/4) q[62];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[62];
rz(-pi/4) q[62];
rz(pi/4) q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[62];
rz(-pi/4) q[62];
sx q[62];
rz(3*pi/4) q[62];
cx q[62],q[63];
rz(-pi/4) q[63];
rz(pi/4) q[72];
rz(-pi/4) q[81];
cx q[72],q[81];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
sx q[85];
rz(2.95841) q[85];
sx q[85];
rz(-pi) q[85];
cx q[73],q[85];
rz(-pi) q[85];
sx q[85];
rz(2.95841) q[85];
sx q[85];
cx q[73],q[85];
x q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[64],q[63];
rz(pi/4) q[63];
cx q[62],q[63];
rz(-pi/4) q[63];
cx q[64],q[63];
rz(3*pi/4) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[62],q[72];
cx q[64],q[63];
rz(-pi/4) q[63];
rz(pi/4) q[64];
cx q[64],q[63];
cx q[72],q[62];
cx q[62],q[72];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[81],q[82];
rz(-pi/4) q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
sx q[85];
rz(2.7752273) q[85];
sx q[85];
rz(-pi) q[85];
cx q[84],q[85];
rz(-pi) q[85];
sx q[85];
rz(2.7752273) q[85];
sx q[85];
cx q[84],q[85];
cx q[84],q[83];
rz(pi/4) q[83];
cx q[82],q[83];
rz(pi/4) q[82];
rz(-pi/4) q[83];
cx q[84],q[83];
rz(3*pi/4) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[83],q[82];
rz(-pi/4) q[82];
rz(pi/4) q[83];
cx q[83],q[82];
rz(pi/2) q[82];
sx q[82];
rz(-pi/2) q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[63],q[62];
rz(-pi/4) q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[64],q[63];
rz(pi/4) q[63];
cx q[62],q[63];
rz(pi/4) q[62];
rz(-pi/4) q[63];
cx q[64],q[63];
rz(pi/4) q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[63],q[62];
rz(-pi/4) q[62];
rz(pi/4) q[63];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[62],q[72];
x q[63];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[72],q[62];
cx q[62],q[72];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[81],q[72];
rz(-pi/4) q[72];
cx q[62],q[72];
rz(pi/4) q[72];
cx q[81],q[72];
rz(-pi/4) q[72];
cx q[62],q[72];
rz(pi/4) q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
rz(pi/4) q[81];
cx q[72],q[81];
rz(pi/4) q[72];
rz(-pi/4) q[81];
cx q[72],q[81];
x q[72];
cx q[82],q[81];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[84],q[85];
rz(-pi) q[85];
sx q[85];
rz(7*pi/8) q[85];
sx q[85];
cx q[84],q[85];
sx q[85];
rz(7*pi/8) q[85];
sx q[85];
rz(-pi) q[85];
cx q[84],q[85];
sx q[85];
rz(3.1186948) q[85];
sx q[85];
rz(-pi) q[85];
cx q[84],q[85];
sx q[85];
rz(1.5936942) q[85];
sx q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(pi/4) q[84];
cx q[83],q[84];
rz(pi/4) q[83];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(3*pi/4) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(pi/4) q[84];
cx q[84],q[83];
cx q[84],q[85];
rz(-pi) q[85];
sx q[85];
rz(3.1186948) q[85];
sx q[85];
cx q[84],q[85];
sx q[85];
rz(1.5478985) q[85];
sx q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(pi/4) q[84];
cx q[83],q[84];
rz(pi/4) q[83];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(3*pi/4) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(pi/4) q[84];
cx q[84],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[62];
rz(-pi/4) q[62];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[84],q[85];
sx q[85];
rz(3.095797) q[85];
sx q[85];
rz(-pi) q[85];
cx q[84],q[85];
sx q[85];
rz(1.616592) q[85];
sx q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(pi/4) q[84];
cx q[83],q[84];
rz(pi/4) q[83];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(3*pi/4) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(pi/4) q[84];
cx q[84],q[83];
cx q[84],q[85];
rz(-pi) q[85];
sx q[85];
rz(3.095797) q[85];
sx q[85];
cx q[84],q[85];
sx q[85];
rz(1.5250007) q[85];
sx q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[83],q[84];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(pi/4) q[84];
cx q[83],q[84];
rz(pi/4) q[83];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(3*pi/4) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(pi/4) q[84];
cx q[84],q[83];
x q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[72],q[62];
rz(pi/4) q[62];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[62];
rz(-pi/4) q[62];
rz(pi/4) q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[62];
rz(3*pi/4) q[62];
sx q[62];
rz(3*pi/4) q[62];
cx q[62],q[63];
rz(-pi/4) q[63];
rz(pi/4) q[72];
rz(-pi/4) q[81];
cx q[72],q[81];
cx q[84],q[85];
sx q[85];
rz(3.0500013) q[85];
sx q[85];
rz(-pi) q[85];
cx q[84],q[85];
sx q[85];
rz(1.6623877) q[85];
sx q[85];
cx q[73],q[85];
rz(-pi/4) q[85];
cx q[84],q[85];
rz(pi/4) q[85];
cx q[73],q[85];
rz(pi/4) q[73];
rz(-pi/4) q[85];
cx q[84],q[85];
rz(3*pi/4) q[85];
sx q[85];
rz(pi/2) q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[73];
rz(-pi/4) q[73];
rz(pi/4) q[85];
cx q[85],q[73];
cx q[85],q[84];
rz(-pi) q[84];
sx q[84];
rz(3.0500013) q[84];
sx q[84];
cx q[85],q[84];
sx q[84];
rz(1.479205) q[84];
sx q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[73],q[85];
rz(-pi/4) q[85];
cx q[84],q[85];
rz(pi/4) q[85];
cx q[73],q[85];
rz(pi/4) q[73];
rz(-pi/4) q[85];
cx q[84],q[85];
rz(3*pi/4) q[85];
sx q[85];
rz(pi/2) q[85];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[73];
rz(-pi/4) q[73];
rz(pi/4) q[85];
cx q[85],q[73];
x q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[64],q[63];
rz(pi/4) q[63];
cx q[62],q[63];
rz(-pi/4) q[63];
cx q[64],q[63];
rz(-pi/4) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[62],q[72];
cx q[64],q[63];
rz(-pi/4) q[63];
rz(pi/4) q[64];
cx q[64],q[63];
cx q[72],q[62];
cx q[62],q[72];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[85],q[84];
sx q[84];
rz(2.95841) q[84];
sx q[84];
rz(-pi) q[84];
cx q[85],q[84];
sx q[84];
rz(1.753979) q[84];
sx q[84];
cx q[83],q[84];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(pi/4) q[84];
cx q[83],q[84];
rz(pi/4) q[83];
rz(-pi/4) q[84];
cx q[85],q[84];
rz(3*pi/4) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(pi/4) q[84];
cx q[84],q[85];
rz(-pi) q[85];
sx q[85];
rz(2.95841) q[85];
sx q[85];
cx q[84],q[85];
cx q[83],q[84];
cx q[84],q[83];
sx q[85];
rz(1.3876137) q[85];
sx q[85];
cx q[84],q[85];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi/4) q[85];
cx q[84],q[85];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
rz(pi/4) q[85];
cx q[84],q[85];
rz(pi/4) q[84];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
rz(-pi/4) q[85];
cx q[84],q[85];
cx q[84],q[83];
rz(-pi/4) q[83];
rz(3*pi/4) q[84];
cx q[84],q[83];
sx q[84];
rz(pi/2) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[82],q[83];
rz(-pi/4) q[83];
cx q[84],q[83];
rz(pi/4) q[83];
cx q[82],q[83];
rz(pi/4) q[82];
rz(-pi/4) q[83];
cx q[84],q[83];
rz(3*pi/4) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[83],q[82];
rz(-pi/4) q[82];
rz(pi/4) q[83];
cx q[83],q[82];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[63],q[62];
rz(-pi/4) q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[64],q[63];
rz(pi/4) q[63];
cx q[62],q[63];
rz(pi/4) q[62];
rz(-pi/4) q[63];
cx q[64],q[63];
rz(-pi/4) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[63],q[62];
rz(-pi/4) q[62];
rz(pi/4) q[63];
cx q[63],q[62];
rz(pi/2) q[62];
sx q[62];
rz(-pi/2) q[62];
cx q[62],q[72];
x q[63];
cx q[72],q[62];
cx q[62],q[72];
cx q[81],q[72];
rz(-pi/4) q[72];
cx q[62],q[72];
rz(pi/4) q[72];
cx q[81],q[72];
rz(-pi/4) q[72];
cx q[62],q[72];
rz(-pi/4) q[72];
sx q[72];
rz(pi/2) q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
rz(pi/4) q[81];
cx q[72],q[81];
rz(pi/4) q[72];
rz(-pi/4) q[81];
cx q[72],q[81];
x q[72];
x q[81];
cx q[82],q[81];
rz(3*pi/4) q[85];
sx q[85];
rz(pi/2) q[85];
barrier q[0],q[119],q[65],q[9],q[84],q[18],q[82],q[27],q[91],q[24],q[88],q[33],q[97],q[42],q[106],q[51],q[48],q[115],q[60],q[112],q[57],q[124],q[2],q[121],q[73],q[11],q[75],q[20],q[55],q[83],q[17],q[85],q[26],q[90],q[35],q[99],q[44],q[108],q[53],q[50],q[117],q[114],q[59],q[4],q[123],q[68],q[13],q[77],q[10],q[74],q[86],q[19],q[63],q[28],q[92],q[37],q[101],q[46],q[43],q[110],q[107],q[52],q[116],q[61],q[6],q[125],q[70],q[3],q[15],q[67],q[79],q[12],q[76],q[21],q[72],q[30],q[94],q[39],q[36],q[103],q[100],q[45],q[109],q[54],q[118],q[81],q[8],q[64],q[5],q[69],q[14],q[78],q[23],q[87],q[32],q[29],q[96],q[41],q[93],q[38],q[105],q[102],q[47],q[111],q[56],q[1],q[120],q[66],q[62],q[7],q[126],q[71],q[16],q[80],q[25],q[22],q[89],q[34],q[31],q[98],q[95],q[40],q[104],q[49],q[113],q[58],q[122];
measure q[82] -> meas[0];
measure q[72] -> meas[1];
measure q[63] -> meas[2];
measure q[83] -> meas[3];
measure q[85] -> meas[4];
measure q[84] -> meas[5];
measure q[81] -> meas[6];
measure q[62] -> meas[7];
measure q[64] -> meas[8];
