OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg c[10];
rz(-pi/2) q[61];
rx(-pi/2) q[61];
rx(pi) q[63];
rz(0.51695152) q[63];
cz q[63],q[20];
rx(1.3268934) q[20];
cz q[63],q[20];
rx(0.24390293) q[20];
rz(pi/2) q[20];
rx(pi/2) q[20];
cz q[63],q[62];
rx(-0.48780589) q[62];
cz q[63],q[62];
rx(0.48780589) q[62];
cz q[61],q[62];
rx(pi/2) q[61];
rz(pi) q[61];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
cz q[62],q[61];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(pi/2) q[62];
cz q[61],q[62];
cz q[63],q[62];
rx(-0.9756118) q[62];
cz q[63],q[62];
rx(2.5464081) q[62];
rz(pi/2) q[62];
rx(pi/2) q[62];
cz q[63],q[56];
rx(1.1903691) q[56];
cz q[63],q[56];
rx(0.38042723) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(3.9024471) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
cz q[62],q[49];
rx(-0.76085445) q[49];
cz q[62],q[49];
rx(2.3316508) q[49];
rz(pi/2) q[49];
rx(pi/2) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[49],q[62];
rz(-pi/2) q[49];
rx(-pi/2) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[49];
rx(pi/2) q[49];
rz(4.6633016) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[49],q[62];
cz q[49],q[48];
rx(-1.5217089) q[48];
cz q[49],q[48];
rx(1.5217089) q[48];
rx(pi/2) q[49];
rz(pi) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[49];
rx(pi/2) q[49];
rz(pi) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(pi) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[49];
rx(pi/2) q[49];
rz(pi) q[49];
cz q[49],q[48];
rz(-pi/2) q[48];
rx(-pi/2) q[48];
rx(pi/2) q[49];
rz(pi) q[49];
cz q[48],q[49];
rx(pi/2) q[48];
rz(pi) q[48];
rx(pi/2) q[49];
rz(pi) q[49];
cz q[49],q[48];
rx(pi/2) q[48];
rz(pi) q[48];
rx(pi/2) q[62];
rz(pi) q[62];
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
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(3.0434179) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
cz q[56],q[57];
rx(pi/32) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(-pi/32) q[57];
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
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(15*pi/16) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[56],q[57];
rx(1.5462526) q[57];
rz(pi/2) q[57];
rx(pi/2) q[57];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
rz(-pi/2) q[20];
rx(-pi/2) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(15*pi/16) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
cz q[20],q[21];
rx(pi/16) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rz(pi) q[20];
rx(-pi/16) q[21];
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
rz(pi) q[62];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
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
cz q[20],q[21];
rx(pi/2) q[20];
rz(pi) q[20];
rz(-pi/2) q[21];
rx(-pi/2) q[21];
cz q[21],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
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
cz q[61],q[62];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[61];
rx(pi/2) q[61];
rz(7*pi/8) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[61],q[62];
cz q[61],q[60];
rx(pi/8) q[60];
cz q[61],q[60];
rx(-pi/8) q[60];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[61];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[61],q[62];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[61];
rx(pi/2) q[61];
rz(pi) q[61];
cz q[61],q[60];
rz(-pi/2) q[60];
rx(-pi/2) q[60];
rx(pi/2) q[61];
rz(pi) q[61];
cz q[60],q[61];
rx(pi/2) q[60];
rz(pi/2) q[60];
rx(pi/2) q[61];
rz(pi) q[61];
cz q[61],q[60];
rx(1.5646604) q[60];
rz(pi/2) q[60];
rx(pi/2) q[60];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(pi) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[49];
rx(pi/2) q[49];
rz(pi) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(pi) q[49];
cz q[48],q[49];
rx(pi/2) q[48];
rz(pi) q[48];
rx(pi/2) q[49];
rz(pi) q[49];
cz q[49],q[48];
rx(pi/2) q[48];
rz(pi) q[48];
rx(pi/2) q[49];
rz(pi/2) q[49];
cz q[48],q[49];
rx(pi/2) q[48];
rz(pi) q[48];
cz q[11],q[48];
rz(-pi/2) q[11];
rx(-pi/2) q[11];
rx(pi/2) q[48];
rz(pi) q[48];
cz q[48],q[11];
rx(pi/2) q[11];
rz(3*pi/4) q[11];
rx(pi/2) q[48];
rz(pi) q[48];
cz q[11],q[48];
cz q[11],q[10];
rx(pi/4) q[10];
cz q[11],q[10];
rx(pi/4) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
cz q[10],q[21];
rz(-pi/2) q[10];
rx(-pi/2) q[10];
rx(pi/2) q[11];
rz(pi) q[11];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[21],q[10];
rx(pi/2) q[10];
rz(pi) q[10];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rz(pi) q[10];
rx(pi/2) q[48];
rz(pi) q[48];
rx(1.5217089) q[49];
rz(pi/2) q[49];
rx(pi/2) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[61];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[61],q[62];
rx(pi/2) q[61];
rz(pi/2) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[61];
rx(1.4726216) q[61];
rz(pi/2) q[61];
rx(pi/2) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
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
cz q[20],q[21];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[21],q[20];
rx(pi/2) q[20];
rz(pi/2) q[20];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[20],q[21];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[21],q[10];
rx(pi/2) q[10];
rz(pi) q[10];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rz(pi) q[10];
rx(pi/2) q[21];
rz(3.0556897) q[21];
cz q[21],q[10];
rx(pi/2) q[10];
rz(pi) q[10];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
rz(3*pi/4) q[63];
cz q[63],q[62];
rx(3*pi/8) q[62];
rz(pi/2) q[62];
rx(pi/2) q[62];
cz q[63],q[20];
rx(pi/4) q[20];
cz q[63],q[20];
rx(pi/4) q[20];
rz(pi/2) q[20];
rx(pi/2) q[20];
cz q[20],q[63];
rz(-pi/2) q[20];
rx(-pi/2) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi/2) q[20];
cz q[62],q[63];
rx(pi/8) q[63];
cz q[62],q[63];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
rx(-pi/8) q[63];
cz q[56],q[63];
rx(pi/16) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(7*pi/16) q[63];
rz(pi/2) q[63];
rx(pi/2) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi) q[62];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
rz(3*pi/4) q[63];
cz q[63],q[62];
cz q[61],q[62];
rx(pi/32) q[62];
cz q[61],q[62];
rx(-pi/32) q[62];
cz q[49],q[62];
rx(pi/64) q[62];
cz q[49],q[62];
rz(1.4726216) q[49];
rx(1.5217089) q[62];
rz(pi/2) q[62];
rx(pi/2) q[62];
cz q[63],q[20];
rx(pi/4) q[20];
cz q[63],q[20];
rx(-pi/4) q[20];
cz q[62],q[63];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
rz(5*pi/8) q[63];
cz q[63],q[56];
cz q[57],q[56];
rx(pi/128) q[56];
cz q[57],q[56];
rx(1.5462526) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
cz q[63],q[20];
rx(pi/8) q[20];
cz q[63],q[20];
rx(-pi/8) q[20];
cz q[63],q[62];
rx(pi/4) q[62];
cz q[63],q[62];
cz q[56],q[63];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rx(-pi/4) q[62];
cz q[61],q[62];
rz(-pi/2) q[61];
rx(-pi/2) q[61];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
cz q[62],q[61];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(pi/2) q[62];
cz q[61],q[62];
rx(pi/2) q[61];
rz(pi/2) q[61];
rx(7*pi/16) q[62];
rz(pi/2) q[62];
rx(pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rz(-pi/2) q[20];
rx(-pi/2) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi/2) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
cz q[21],q[20];
rx(pi/256) q[20];
cz q[21],q[20];
rx(1.5585245) q[20];
rz(pi/2) q[20];
rx(pi/2) q[20];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[62],q[63];
rx(pi/16) q[63];
cz q[62],q[63];
rz(3*pi/8) q[62];
cz q[62],q[61];
rx(pi/8) q[61];
cz q[62],q[61];
rx(-pi/8) q[61];
rx(-pi/16) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi/2) q[62];
cz q[49],q[62];
rx(pi/32) q[62];
cz q[49],q[62];
rx(-pi/32) q[62];
rx(pi/4) q[63];
rz(pi/2) q[63];
rx(pi/2) q[63];
cz q[63],q[56];
rx(pi/4) q[56];
cz q[63],q[56];
rx(pi/4) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
cz q[56],q[57];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(3.0925053) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[56],q[57];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[62];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(pi) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[49];
rx(pi/2) q[49];
rz(pi) q[49];
rx(pi/2) q[62];
rz(pi/2) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(pi/2) q[49];
rx(7*pi/16) q[62];
rz(pi/2) q[62];
rx(pi/2) q[62];
cz q[62],q[61];
rx(pi/16) q[61];
cz q[62],q[61];
rx(-pi/16) q[61];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[56],q[63];
rx(pi/64) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(-pi/64) q[63];
cz q[20],q[63];
rz(-pi/2) q[20];
rx(-pi/2) q[20];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi/2) q[20];
cz q[21],q[20];
rx(pi/128) q[20];
cz q[21],q[20];
rx(-pi/128) q[20];
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
rz(pi) q[62];
cz q[62],q[61];
rz(-pi/2) q[61];
rx(-pi/2) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[61],q[62];
rx(pi/2) q[61];
rz(pi/2) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[61];
cz q[60],q[61];
rx(pi/512) q[61];
cz q[60],q[61];
rz(1.5585245) q[60];
rx(1.5646604) q[61];
rz(pi/2) q[61];
rx(pi/2) q[61];
rx(pi/2) q[62];
rz(pi/2) q[62];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
rz(3.0434179) q[63];
cz q[63],q[56];
rx(3*pi/8) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
cz q[56],q[57];
rx(pi/8) q[57];
cz q[56],q[57];
rx(-pi/8) q[57];
cz q[63],q[62];
rx(pi/32) q[62];
cz q[63],q[62];
rx(-pi/32) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(15*pi/16) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
cz q[56],q[57];
rx(pi/16) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(-pi/16) q[57];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/4) q[62];
rz(pi/2) q[62];
rx(pi/2) q[62];
cz q[62],q[49];
rx(pi/4) q[49];
cz q[62],q[49];
rx(-pi/4) q[49];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
cz q[63],q[20];
rz(-pi/2) q[20];
rx(-pi/2) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi/2) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
cz q[21],q[20];
rx(pi/64) q[20];
cz q[21],q[20];
rx(-pi/64) q[20];
cz q[21],q[20];
rz(-pi/2) q[20];
rx(-pi/2) q[20];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[21],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rz(pi) q[10];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[21],q[10];
rx(pi/2) q[10];
rz(pi) q[10];
rx(pi/2) q[21];
rz(pi/2) q[21];
cz q[10],q[21];
cz q[10],q[11];
rx(pi/2) q[10];
rz(pi) q[10];
rx(pi/2) q[11];
rz(pi) q[11];
cz q[11],q[10];
rx(pi/2) q[10];
rz(pi/2) q[10];
rx(pi/2) q[11];
rz(pi) q[11];
cz q[10],q[11];
rx(1.5677284) q[21];
rz(pi/2) q[21];
rx(pi/2) q[21];
cz q[48],q[11];
rx(pi/2) q[11];
rz(pi) q[11];
rx(pi/2) q[48];
rz(pi) q[48];
cz q[11],q[48];
rx(pi/2) q[11];
rz(pi) q[11];
rx(pi/2) q[48];
rz(pi) q[48];
cz q[48],q[11];
rx(pi/2) q[11];
rz(pi/2) q[11];
rx(pi/2) q[48];
rz(pi/2) q[48];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[61],q[62];
rz(-pi/2) q[61];
rx(-pi/2) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[61];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[61],q[62];
rx(pi/2) q[61];
rz(pi/2) q[61];
cz q[60],q[61];
rx(pi/256) q[61];
cz q[60],q[61];
rx(-pi/256) q[61];
cz q[60],q[61];
rx(pi/2) q[60];
rz(pi) q[60];
rz(-pi/2) q[61];
rx(-pi/2) q[61];
cz q[61],q[60];
rx(pi/2) q[60];
rz(pi) q[60];
rx(pi/2) q[61];
rz(pi) q[61];
cz q[60],q[61];
rx(pi/2) q[60];
rz(pi/2) q[60];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(1.4726216) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
cz q[56],q[57];
rx(pi/32) q[57];
cz q[56],q[57];
rz(7*pi/16) q[56];
rx(-pi/32) q[57];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(5*pi/8) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
cz q[62],q[49];
rx(pi/8) q[49];
cz q[62],q[49];
rx(3*pi/8) q[49];
rz(pi/2) q[49];
rx(pi/2) q[49];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi/2) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
cz q[21],q[20];
rx(pi/1024) q[20];
cz q[21],q[20];
rx(-pi/1024) q[20];
cz q[21],q[20];
rz(-pi/2) q[20];
rx(-pi/2) q[20];
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
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[62],q[63];
rx(pi/4) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[61];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[61],q[62];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[49],q[62];
rz(-pi/2) q[49];
rx(-pi/2) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[49];
rx(pi/2) q[49];
rz(3.117049) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[49],q[62];
cz q[49],q[48];
rx(pi/128) q[48];
cz q[49],q[48];
rx(-pi/128) q[48];
rx(pi/2) q[49];
rz(pi) q[49];
rx(-pi/4) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi) q[20];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi/2) q[20];
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
rz(pi) q[62];
cz q[62],q[61];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[61],q[62];
rx(pi/2) q[61];
rz(pi/2) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[61];
rx(1.5646604) q[61];
rz(pi/2) q[61];
rx(pi/2) q[61];
cz q[61],q[60];
rx(pi/512) q[60];
cz q[61],q[60];
rx(1.5646604) q[60];
rz(pi/2) q[60];
rx(pi/2) q[60];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[56],q[63];
rx(pi/16) q[63];
cz q[56],q[63];
rx(-pi/16) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[56],q[63];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(3*pi/8) q[63];
rz(pi/2) q[63];
rx(pi/2) q[63];
cz q[63],q[20];
rx(pi/8) q[20];
cz q[63],q[20];
rx(-pi/8) q[20];
rz(pi/4) q[63];
cz q[63],q[62];
rx(pi/4) q[62];
cz q[63],q[62];
rx(-pi/4) q[62];
cz q[61],q[62];
rz(-pi/2) q[61];
rx(-pi/2) q[61];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
cz q[62],q[61];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[61],q[62];
rx(pi/2) q[61];
rz(pi/2) q[61];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[49];
rx(pi/2) q[49];
rz(pi) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(pi/2) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[49];
rx(1.5585245) q[49];
rz(pi/2) q[49];
rx(pi/2) q[49];
cz q[49],q[48];
rx(pi/256) q[48];
cz q[49],q[48];
rx(1.5585245) q[48];
rz(pi/2) q[48];
rx(pi/2) q[48];
rx(pi/2) q[62];
rz(pi) q[62];
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
rz(3.0925053) q[63];
cz q[63],q[62];
cz q[49],q[62];
rz(-pi/2) q[49];
rx(-pi/2) q[49];
rx(pi/2) q[62];
rz(pi) q[62];
cz q[62],q[49];
rx(pi/2) q[49];
rz(pi) q[49];
rx(pi/2) q[62];
rz(pi/2) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(pi/2) q[49];
rx(1.5462526) q[62];
rz(pi/2) q[62];
rx(pi/2) q[62];
cz q[63],q[56];
rx(pi/64) q[56];
cz q[63],q[56];
rx(1.5217089) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(3.0434179) q[56];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[56],q[63];
cz q[56],q[57];
rx(pi/32) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(1.4726216) q[57];
rz(pi/2) q[57];
rx(pi/2) q[57];
cz q[62],q[63];
rx(pi/128) q[63];
cz q[62],q[63];
rx(1.5462526) q[63];
rz(pi/2) q[63];
rx(pi/2) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(15*pi/16) q[63];
cz q[63],q[56];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[57],q[56];
cz q[63],q[20];
rx(pi/16) q[20];
cz q[63],q[20];
rx(-pi/16) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(5*pi/8) q[62];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[62],q[63];
cz q[62],q[61];
rx(pi/8) q[61];
cz q[62],q[61];
rx(-pi/8) q[61];
cz q[62],q[49];
rx(pi/4) q[49];
cz q[62],q[49];
rx(-pi/4) q[49];
rx(1.5217089) q[63];
rz(pi/2) q[63];
rx(pi/2) q[63];
cz q[63],q[56];
rx(pi/64) q[56];
cz q[63],q[56];
rx(1.5217089) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
rz(1.4726216) q[63];
cz q[63],q[20];
rx(pi/32) q[20];
cz q[63],q[20];
rx(1.4726216) q[20];
rz(pi/2) q[20];
rx(pi/2) q[20];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(7*pi/16) q[62];
rz(pi/2) q[62];
rx(pi/2) q[62];
cz q[62],q[61];
rx(pi/16) q[61];
cz q[62],q[61];
rx(7*pi/16) q[61];
rz(pi/2) q[61];
rx(pi/2) q[61];
rz(pi/8) q[62];
cz q[62],q[49];
rx(pi/8) q[49];
cz q[62],q[49];
rx(3*pi/8) q[49];
rz(pi/2) q[49];
rx(pi/2) q[49];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[62],q[63];
rx(pi/4) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(pi/4) q[63];
rz(pi/2) q[63];
rx(pi/2) q[63];
barrier q[29],q[38],q[35],q[44],q[53],q[60],q[7],q[4],q[71],q[68],q[13],q[77],q[22],q[31],q[28],q[40],q[37],q[46],q[55],q[0],q[64],q[48],q[6],q[70],q[15],q[79],q[24],q[33],q[30],q[39],q[20],q[61],q[54],q[66],q[10],q[8],q[72],q[17],q[26],q[23],q[32],q[41],q[50],q[47],q[59],q[57],q[1],q[65],q[62],q[74],q[19],q[16],q[25],q[34],q[43],q[52],q[56],q[58],q[3],q[67],q[12],q[76],q[9],q[49],q[73],q[18],q[27],q[36],q[45],q[42],q[51],q[63],q[5],q[69],q[2],q[14],q[78],q[11],q[75],q[21];
measure q[21] -> c[0];
measure q[60] -> c[1];
measure q[48] -> c[2];
measure q[57] -> c[3];
measure q[56] -> c[4];
measure q[20] -> c[5];
measure q[61] -> c[6];
measure q[49] -> c[7];
measure q[63] -> c[8];
measure q[62] -> c[9];
