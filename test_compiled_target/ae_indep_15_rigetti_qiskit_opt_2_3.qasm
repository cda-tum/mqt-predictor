OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg meas[15];
rz(-pi/2) q[18];
rx(-pi/2) q[18];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
rz(-3*pi/2) q[20];
rx(pi/2) q[20];
rz(3.1323888) q[20];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
rz(-pi/2) q[29];
rx(-pi/2) q[29];
rz(-3*pi/2) q[49];
rx(pi/2) q[49];
rz(3.1410174) q[49];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
rz(-3*pi/2) q[61];
rx(pi/2) q[61];
rz(3.1412092) q[61];
rx(pi/2) q[62];
rz(2.4980915) q[62];
rx(pi/2) q[62];
cz q[49],q[62];
rx(pi/2) q[62];
rz(0.92729522) q[62];
rx(-pi/2) q[62];
cz q[49],q[62];
rx(-pi/2) q[62];
rz(0.92729522) q[62];
rx(pi/2) q[62];
cz q[61],q[62];
rx(pi/2) q[62];
rz(1.8545904) q[62];
rx(-pi/2) q[62];
cz q[61],q[62];
rx(-pi/2) q[62];
rz(1.8545904) q[62];
rx(pi/2) q[62];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[62];
rx(-pi/2) q[62];
rz(2.5740044) q[62];
rx(pi/2) q[62];
cz q[63],q[62];
rx(-pi/2) q[62];
rz(2.1383846) q[62];
rx(-pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(3.1408257) q[62];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[62],q[63];
cz q[56],q[63];
rx(pi/2) q[63];
rz(1.1351764) q[63];
rx(-pi/2) q[63];
cz q[56],q[63];
rx(-pi/2) q[63];
rz(1.1351764) q[63];
rx(pi/2) q[63];
cz q[20],q[63];
rx(pi/2) q[63];
rz(2.2703524) q[63];
rx(-pi/2) q[63];
cz q[20],q[63];
rx(-pi/2) q[63];
rz(2.2703529) q[63];
rx(pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[56];
cz q[19],q[56];
rx(-pi/2) q[56];
rz(1.7424795) q[56];
rx(pi/2) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(3.1293208) q[19];
rx(pi/2) q[56];
rz(1.7424796) q[56];
rx(-pi/2) q[56];
cz q[19],q[56];
rx(pi/2) q[56];
rz(2.7982262) q[56];
rx(-pi/2) q[56];
cz q[19],q[56];
rx(-pi/2) q[56];
rz(2.7982262) q[56];
rx(pi/2) q[56];
cz q[57],q[56];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(3.0925053) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
cz q[70],q[57];
rx(-pi/2) q[57];
rz(0.68673084) q[57];
rx(pi/2) q[57];
cz q[70],q[57];
rx(pi/2) q[57];
rz(0.68673293) q[57];
rx(-pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[57];
rz(1.3734617) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[57];
rz(2.9442622) q[57];
rx(pi/2) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
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
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi/4) q[70];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[70],q[71];
cz q[28],q[71];
rx(-pi/2) q[71];
rz(2.7469333) q[71];
rx(pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[28],q[27];
rx(pi/2) q[27];
rz(pi/32) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[27],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(2.7469317) q[71];
rx(-pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[71];
rz(0.78931862) q[71];
rx(-pi/2) q[71];
cz q[64],q[71];
cz q[64],q[65];
rx(pi/2) q[64];
rz(pi) q[64];
rz(-pi/2) q[65];
rx(-pi/2) q[65];
cz q[65],q[64];
rx(pi/2) q[64];
rz(7*pi/8) q[64];
rx(pi/2) q[65];
rz(pi/2) q[65];
cz q[64],q[65];
rx(7*pi/16) q[65];
rz(pi/2) q[65];
rx(pi/2) q[65];
rx(-pi/2) q[71];
rz(0.78932185) q[71];
rx(pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[71];
rz(1.5786372) q[71];
rx(-pi/2) q[71];
cz q[64],q[71];
rx(-pi/2) q[71];
rz(1.5786437) q[71];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[71];
rz(3.1259108) q[71];
rx(pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[71];
rz(3.1258979) q[71];
rx(-pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[71];
rz(0.03136362) q[71];
rx(-pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
rx(-pi/2) q[71];
rz(0.03136362) q[71];
rx(pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[70],q[71];
rx(pi/4) q[71];
cz q[70],q[71];
rx(-pi/4) q[71];
cz q[64],q[71];
rx(pi/8) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
rx(3*pi/8) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[71];
rz(3*pi/4) q[71];
cz q[71],q[64];
cz q[65],q[64];
rx(pi/16) q[64];
cz q[65],q[64];
rx(-pi/16) q[64];
cz q[27],q[64];
rx(pi/32) q[64];
cz q[27],q[64];
rx(1.4726216) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
rz(pi/8) q[65];
cz q[71],q[70];
rx(pi/4) q[70];
cz q[71],q[70];
cz q[64],q[71];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rx(-pi/4) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
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
rz(pi) q[71];
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
rz(pi/2) q[57];
rx(pi/2) q[70];
rz(3.117049) q[70];
cz q[70],q[57];
cz q[56],q[57];
rx(pi/64) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(-pi/64) q[57];
cz q[70],q[57];
rx(pi/128) q[57];
cz q[70],q[57];
rx(1.5462526) q[57];
rz(pi/2) q[57];
rx(pi/2) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
rz(3.0434179) q[57];
cz q[57],q[56];
cz q[19],q[56];
rx(pi/256) q[56];
cz q[19],q[56];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi/2) q[18];
rx(pi/2) q[19];
rz(3.1354567) q[19];
cz q[19],q[18];
rx(1.5462526) q[18];
rz(pi/2) q[18];
rx(pi/2) q[18];
rx(-pi/256) q[56];
cz q[19],q[56];
rx(pi/512) q[56];
cz q[19],q[56];
rx(1.5646604) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
cz q[56],q[63];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(3.1400587) q[56];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[56],q[63];
cz q[20],q[63];
rx(pi/1024) q[63];
cz q[20],q[63];
rx(-pi/1024) q[63];
cz q[56],q[63];
rx(pi/2048) q[63];
cz q[56],q[63];
rx(-pi/2048) q[63];
cz q[62],q[63];
rx(pi/4096) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rx(1.5700293) q[63];
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
rz(3.1400587) q[63];
cz q[63],q[62];
cz q[61],q[62];
rx(pi/8192) q[62];
cz q[61],q[62];
rx(pi/2) q[61];
rz(pi) q[61];
rx(-pi/8192) q[62];
cz q[49],q[62];
rx(pi/16384) q[62];
cz q[49],q[62];
rx(1.5706046) q[62];
rz(pi/2) q[62];
rx(pi/2) q[62];
cz q[62],q[61];
rx(pi/2) q[61];
rz(pi) q[61];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
cz q[61],q[62];
rx(pi/2) q[61];
rz(pi) q[61];
rx(pi/2) q[62];
rz(3.1408257) q[62];
cz q[62],q[61];
rx(pi/2) q[61];
rz(pi/2) q[61];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi/2) q[64];
cz q[65],q[64];
rx(pi/8) q[64];
cz q[65],q[64];
rx(-pi/8) q[64];
cz q[27],q[64];
rx(pi/16) q[64];
cz q[27],q[64];
rx(-pi/16) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
cz q[65],q[64];
rx(pi/4) q[64];
cz q[65],q[64];
rx(-pi/4) q[64];
cz q[27],q[64];
rx(pi/8) q[64];
cz q[27],q[64];
rx(3*pi/8) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi/2) q[70];
cz q[57],q[70];
rx(pi/32) q[70];
cz q[57],q[70];
rx(-pi/32) q[70];
rx(1.5217089) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[70];
rx(pi/64) q[70];
cz q[71],q[70];
rx(-pi/64) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[56],q[19];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[56];
rz(pi/2) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi/2) q[19];
cz q[18],q[19];
rx(pi/128) q[19];
cz q[18],q[19];
cz q[18],q[29];
rz(-pi/2) q[18];
rx(-pi/2) q[18];
rx(-pi/128) q[19];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[18],q[29];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[28],q[29];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
rx(1.5585245) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
cz q[56],q[19];
rx(pi/256) q[19];
cz q[56],q[19];
rx(-pi/256) q[19];
cz q[20],q[19];
rx(pi/512) q[19];
cz q[20],q[19];
rx(-pi/512) q[19];
cz q[56],q[57];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(3.1385247) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[56],q[57];
cz q[56],q[19];
rx(pi/1024) q[19];
cz q[56],q[19];
rx(-pi/1024) q[19];
cz q[20],q[19];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[19],q[20];
rx(pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[20],q[19];
rx(1.5585245) q[19];
rz(pi/2) q[19];
rx(pi/2) q[19];
rx(pi/2) q[20];
rz(pi/2) q[20];
rx(1.5462526) q[57];
rz(pi/2) q[57];
rx(pi/2) q[57];
cz q[63],q[20];
rx(pi/2048) q[20];
cz q[63],q[20];
rx(-pi/2048) q[20];
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
rx(1.5677284) q[20];
rz(pi/2) q[20];
rx(pi/2) q[20];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[62],q[63];
rx(pi/4096) q[63];
cz q[62],q[63];
rx(-pi/4096) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi) q[62];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[62],q[63];
rx(pi/2) q[62];
rz(pi/2) q[62];
cz q[49],q[62];
rx(pi/8192) q[62];
cz q[49],q[62];
rx(-pi/8192) q[62];
cz q[49],q[62];
rx(pi/2) q[49];
rz(pi) q[49];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
cz q[62],q[49];
rx(pi/2) q[49];
rz(pi/2) q[49];
rx(pi/2) q[62];
rz(pi/2) q[62];
cz q[49],q[62];
rx(1.5700293) q[62];
rz(pi/2) q[62];
rx(pi/2) q[62];
rx(pi/2) q[63];
rz(pi) q[63];
rx(7*pi/16) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[64],q[71];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rx(pi/2) q[64];
rz(3.0434179) q[64];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[64],q[71];
cz q[70],q[71];
rx(pi/16) q[71];
cz q[70],q[71];
rx(-pi/16) q[71];
cz q[64],q[71];
rx(pi/32) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[65],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[64],q[65];
rx(pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[65];
rz(9*pi/16) q[65];
cz q[65],q[64];
cz q[27],q[64];
rx(pi/4) q[64];
cz q[27],q[64];
rx(-pi/4) q[64];
rx(-pi/32) q[71];
cz q[70],q[71];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi/2) q[70];
rx(3*pi/8) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[64];
rx(pi/8) q[64];
cz q[71],q[64];
rx(-pi/8) q[64];
cz q[65],q[64];
rx(pi/16) q[64];
cz q[65],q[64];
rx(-pi/16) q[64];
cz q[71],q[64];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rx(pi/4) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
cz q[64],q[27];
rx(pi/4) q[27];
cz q[64],q[27];
rx(pi/4) q[27];
rz(pi/2) q[27];
rx(pi/2) q[27];
cz q[27],q[64];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
cz q[64],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[64];
rz(pi/2) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[65],q[64];
rx(pi/8) q[64];
cz q[65],q[64];
rx(-pi/8) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi/2) q[28];
rx(1.5217089) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[70];
rx(pi/64) q[70];
cz q[71],q[70];
rx(-pi/64) q[70];
cz q[57],q[70];
rx(pi/128) q[70];
cz q[57],q[70];
rx(-pi/128) q[70];
cz q[57],q[70];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi/2) q[56];
cz q[19],q[56];
rx(pi/256) q[56];
cz q[19],q[56];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
cz q[18],q[29];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi/2) q[29];
cz q[18],q[29];
rx(1.5462526) q[29];
rz(pi/2) q[29];
rx(pi/2) q[29];
rx(-pi/256) q[56];
rx(1.5646604) q[57];
rz(pi/2) q[57];
rx(pi/2) q[57];
cz q[57],q[56];
rx(pi/512) q[56];
cz q[57],q[56];
rx(-pi/512) q[56];
cz q[63],q[56];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(1.5692623) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[20],q[63];
rx(pi/1024) q[63];
cz q[20],q[63];
rz(-pi/2) q[20];
rx(-pi/2) q[20];
cz q[19],q[20];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[20],q[19];
rx(pi/2) q[19];
rz(3.1354567) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[19],q[20];
rx(pi/2) q[20];
rz(pi/2) q[20];
rx(-pi/1024) q[63];
cz q[56],q[63];
rx(pi/2048) q[63];
cz q[56],q[63];
rz(1.5677284) q[56];
rx(-pi/2048) q[63];
cz q[62],q[63];
rx(pi/4096) q[63];
cz q[62],q[63];
rx(-pi/4096) q[63];
cz q[62],q[63];
rz(-pi/2) q[62];
rx(-pi/2) q[62];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[63],q[62];
rx(pi/2) q[62];
rz(pi/2) q[62];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[62],q[63];
rx(1.5692623) q[63];
rz(pi/2) q[63];
rx(pi/2) q[63];
rz(1.276272) q[71];
cz q[71],q[28];
rx(pi/32) q[28];
cz q[71],q[28];
rx(-pi/32) q[28];
cz q[71],q[64];
rx(pi/16) q[64];
cz q[71],q[64];
rx(-pi/16) q[64];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[71];
rz(2.9943305) q[71];
cz q[71],q[70];
rx(3*pi/8) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
cz q[71],q[28];
rx(pi/64) q[28];
cz q[71],q[28];
rx(-pi/64) q[28];
cz q[29],q[28];
rx(pi/128) q[28];
cz q[29],q[28];
rx(-pi/128) q[28];
cz q[71],q[64];
rx(pi/32) q[64];
cz q[71],q[64];
rx(1.4726216) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
cz q[64],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[64],q[27];
rx(pi/2) q[64];
rz(pi/2) q[64];
cz q[65],q[64];
rx(pi/4) q[64];
cz q[65],q[64];
rx(-pi/4) q[64];
cz q[71],q[64];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rx(7*pi/16) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[70],q[71];
rx(pi/8) q[71];
cz q[70],q[71];
rx(-pi/8) q[71];
cz q[64],q[71];
rx(pi/16) q[71];
cz q[64],q[71];
cz q[64],q[65];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[65],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[65];
rz(pi/2) q[65];
cz q[64],q[65];
rx(pi/2) q[64];
rz(pi/2) q[64];
rx(3*pi/8) q[65];
rz(pi/2) q[65];
rx(pi/2) q[65];
rx(-pi/16) q[71];
cz q[70],q[71];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/4) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[64];
rx(pi/4) q[64];
cz q[71],q[64];
rx(-pi/4) q[64];
cz q[65],q[64];
rx(pi/8) q[64];
cz q[65],q[64];
rx(-pi/8) q[64];
rz(-pi/4) q[65];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[71],q[64];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
cz q[65],q[64];
rx(pi/4) q[64];
cz q[65],q[64];
rx(pi/4) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi/2) q[57];
rx(1.5585245) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[28];
rx(pi/256) q[28];
cz q[71],q[28];
rx(-pi/256) q[28];
cz q[29],q[28];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
rz(-pi/2) q[29];
rx(-pi/2) q[29];
cz q[28],q[29];
rx(pi/2) q[28];
rz(pi/2) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[28];
rx(1.5217089) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
cz q[28],q[27];
rx(pi/64) q[27];
cz q[28],q[27];
rx(-pi/64) q[27];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[18],q[29];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[18],q[29];
rx(pi/2) q[18];
rz(pi/2) q[18];
cz q[19],q[18];
rx(pi/512) q[18];
cz q[19],q[18];
rx(-pi/512) q[18];
cz q[19],q[18];
rz(-pi/2) q[18];
rx(-pi/2) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[19],q[18];
rx(pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[18],q[29];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(3.1293208) q[29];
cz q[29],q[18];
cz q[56],q[19];
rx(pi/1024) q[19];
cz q[56],q[19];
rx(-pi/1024) q[19];
cz q[56],q[19];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[56],q[19];
rx(1.5646604) q[19];
rz(pi/2) q[19];
rx(pi/2) q[19];
rx(pi/2) q[56];
rz(pi/2) q[56];
cz q[63],q[56];
rx(pi/2048) q[56];
cz q[63],q[56];
rx(-pi/2048) q[56];
cz q[63],q[56];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[28],q[71];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(3.117049) q[28];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[28],q[71];
cz q[28],q[27];
rx(pi/128) q[27];
cz q[28],q[27];
rx(1.5462526) q[27];
rz(pi/2) q[27];
rx(pi/2) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[27],q[28];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[28],q[27];
rx(pi/2) q[27];
rz(3.0925053) q[27];
rx(pi/2) q[28];
rz(pi/2) q[28];
cz q[27],q[28];
cz q[29],q[28];
rx(pi/256) q[28];
cz q[29],q[28];
rx(1.5585245) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[28],q[29];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[28];
rx(pi/2) q[28];
rz(3.117049) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[28],q[29];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[18],q[29];
rx(pi/2) q[18];
rz(pi/2) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[18];
cz q[19],q[18];
rx(pi/512) q[18];
cz q[19],q[18];
rx(-pi/512) q[18];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[56],q[19];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[56],q[19];
rx(1.5677284) q[19];
rz(pi/2) q[19];
rx(pi/2) q[19];
cz q[19],q[18];
rx(pi/1024) q[18];
cz q[19],q[18];
rx(1.5677284) q[18];
rz(pi/2) q[18];
rx(pi/2) q[18];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[19],q[56];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[56],q[19];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[18],q[19];
rz(-pi/2) q[18];
rx(-pi/2) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[18],q[19];
cz q[18],q[29];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[18];
rx(pi/2) q[18];
rz(pi/2) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[18],q[29];
rx(1.4726216) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[70];
rx(pi/32) q[70];
cz q[71],q[70];
rx(-pi/32) q[70];
cz q[71],q[70];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rx(7*pi/16) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
cz q[70],q[57];
rx(pi/16) q[57];
cz q[70],q[57];
rx(-pi/16) q[57];
rz(3*pi/8) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi/2) q[64];
cz q[27],q[64];
rx(pi/64) q[64];
cz q[27],q[64];
rx(-pi/64) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(pi) q[27];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
cz q[64],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(pi/2) q[27];
cz q[28],q[27];
rx(pi/128) q[27];
cz q[28],q[27];
rx(-pi/128) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[70],q[71];
rx(pi/8) q[71];
cz q[70],q[71];
rx(3*pi/8) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(2.8470683) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[70],q[57];
rx(pi/32) q[57];
cz q[70],q[57];
rx(-pi/32) q[57];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rx(pi/2) q[64];
rz(3*pi/4) q[64];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[64],q[71];
cz q[64],q[65];
rx(pi/4) q[65];
cz q[64],q[65];
rx(pi/2) q[64];
rz(pi) q[64];
rx(-pi/4) q[65];
cz q[70],q[71];
rx(pi/16) q[71];
cz q[70],q[71];
rx(7*pi/16) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi/2) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(3.0925053) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[70],q[57];
rx(pi/64) q[57];
cz q[70],q[57];
rx(-pi/64) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rx(pi/2) q[64];
rz(5*pi/8) q[64];
cz q[64],q[65];
rx(pi/8) q[65];
cz q[64],q[65];
rx(3*pi/8) q[65];
rz(pi/2) q[65];
rx(pi/2) q[65];
rx(pi/2) q[71];
rz(pi/2) q[71];
rx(pi/4) q[71];
cz q[64],q[71];
cz q[65],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rz(-pi/2) q[65];
rx(-pi/2) q[65];
cz q[64],q[65];
rx(pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[65],q[64];
rx(pi/2) q[65];
rz(pi/2) q[65];
rx(pi/4) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi/2) q[70];
rx(pi/8) q[70];
rx(pi/2) q[71];
rz(2.4543693) q[71];
cz q[71],q[28];
rx(pi/32) q[28];
cz q[71],q[28];
rx(1.4726216) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
cz q[28],q[29];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[28];
rx(pi/2) q[28];
rz(3.1293208) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[28],q[29];
cz q[28],q[27];
rx(pi/256) q[27];
cz q[28],q[27];
rx(-pi/256) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[71],q[64];
rx(pi/16) q[64];
cz q[71],q[64];
rx(7*pi/16) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
cz q[71],q[70];
rx(3*pi/8) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
cz q[70],q[57];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rx(pi/2) q[64];
rz(3*pi/4) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
cz q[64],q[65];
rx(pi/4) q[65];
cz q[64],q[65];
rx(pi/4) q[65];
rz(pi/2) q[65];
rx(pi/2) q[65];
cz q[65],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rz(-pi/2) q[65];
rx(-pi/2) q[65];
cz q[64],q[65];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[65],q[64];
rx(pi/2) q[65];
rz(pi/2) q[65];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(2.9697868) q[71];
cz q[71],q[28];
cz q[29],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[28],q[29];
rx(pi/2) q[28];
rz(pi/2) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[71],q[70];
rx(pi/128) q[70];
cz q[71],q[70];
rx(-pi/128) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[71],q[28];
rx(pi/64) q[28];
cz q[71],q[28];
rx(1.5217089) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
cz q[28],q[29];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[28],q[29];
rx(pi/2) q[28];
rz(pi/2) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[71],q[28];
rx(pi/32) q[28];
cz q[71],q[28];
rx(-pi/32) q[28];
cz q[29],q[28];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[28],q[29];
rx(pi/2) q[28];
rz(pi/2) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[28];
rx(pi/2) q[29];
rz(pi) q[29];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(13*pi/16) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
cz q[70],q[57];
rx(pi/16) q[57];
cz q[70],q[57];
rx(7*pi/16) q[57];
rz(pi/2) q[57];
rx(pi/2) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rx(1.5646604) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
cz q[64],q[27];
rx(pi/512) q[27];
cz q[64],q[27];
rx(1.5646604) q[27];
rz(pi/2) q[27];
rx(pi/2) q[27];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[70],q[71];
rx(pi/8) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
rx(-pi/8) q[71];
cz q[64],q[71];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[71],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
rx(pi/2) q[71];
rz(pi) q[71];
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
rx(1.5585245) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
cz q[70],q[57];
rx(pi/256) q[57];
cz q[70],q[57];
rx(-pi/256) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rx(pi/2) q[64];
rz(3*pi/4) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
cz q[64],q[65];
rx(pi/4) q[65];
cz q[64],q[65];
rx(-pi/4) q[65];
cz q[70],q[71];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rx(1.5462526) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[28];
rx(pi/128) q[28];
cz q[71],q[28];
rx(1.5462526) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
cz q[28],q[29];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[28],q[29];
rx(pi/2) q[28];
rz(pi/2) q[28];
rx(pi/2) q[29];
rz(pi/2) q[29];
rz(1.2271846) q[71];
cz q[71],q[28];
rx(pi/64) q[28];
cz q[71],q[28];
rx(1.5217089) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
cz q[71],q[70];
rx(pi/32) q[70];
cz q[71],q[70];
rx(1.4726216) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[70];
rz(pi/2) q[70];
cz q[71],q[70];
rx(pi/16) q[70];
cz q[71],q[70];
rx(7*pi/16) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
cz q[71],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rx(3*pi/8) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
cz q[64],q[65];
rx(pi/8) q[65];
cz q[64],q[65];
rz(pi/4) q[64];
rx(3*pi/8) q[65];
rz(pi/2) q[65];
rx(pi/2) q[65];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[64],q[71];
rx(pi/4) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi/2) q[64];
rx(pi/4) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
barrier q[17],q[26],q[35],q[32],q[41],q[50],q[59],q[4],q[1],q[68],q[62],q[10],q[74],q[28],q[19],q[25],q[37],q[34],q[43],q[52],q[71],q[58],q[3],q[67],q[12],q[76],q[21],q[29],q[30],q[61],q[36],q[45],q[54],q[51],q[60],q[5],q[69],q[14],q[78],q[23],q[57],q[20],q[38],q[47],q[44],q[70],q[53],q[18],q[7],q[49],q[16],q[13],q[77],q[22],q[31],q[40],q[64],q[46],q[55],q[0],q[63],q[8],q[9],q[73],q[6],q[56],q[15],q[79],q[24],q[33],q[42],q[39],q[48],q[27],q[2],q[66],q[11],q[65],q[75],q[72];
measure q[64] -> meas[0];
measure q[71] -> meas[1];
measure q[65] -> meas[2];
measure q[70] -> meas[3];
measure q[57] -> meas[4];
measure q[28] -> meas[5];
measure q[29] -> meas[6];
measure q[56] -> meas[7];
measure q[27] -> meas[8];
measure q[19] -> meas[9];
measure q[63] -> meas[10];
measure q[62] -> meas[11];
measure q[49] -> meas[12];
measure q[61] -> meas[13];
measure q[18] -> meas[14];
