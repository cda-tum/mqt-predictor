OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg meas[14];
rz(-pi/2) q[18];
rx(-pi/2) q[18];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
rz(-pi/2) q[37];
rx(-pi/2) q[37];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
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
rz(-pi/2) q[57];
rx(-pi/2) q[57];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
cz q[20],q[63];
rz(-pi/2) q[20];
rx(-pi/2) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rz(-pi/2) q[65];
rx(-pi/2) q[65];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
rz(pi/2) q[71];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[28],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[27],q[64];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[28];
rz(pi/2) q[28];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[64],q[65];
rx(pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
rx(3*pi/4) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[28];
rx(-pi/4) q[28];
cz q[71],q[28];
rx(pi/4) q[28];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[28],q[71];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(9*pi/8) q[28];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[28],q[71];
cz q[28],q[27];
rx(-pi/8) q[27];
cz q[28],q[27];
rx(5*pi/8) q[27];
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
rz(3.4852044) q[27];
rx(pi/2) q[28];
rz(pi/2) q[28];
cz q[27],q[28];
cz q[27],q[64];
rx(-pi/16) q[64];
cz q[27],q[64];
rx(pi/16) q[64];
rx(3*pi/4) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[28];
rx(-pi/4) q[28];
cz q[71],q[28];
rx(pi/4) q[28];
rz(3.0434179) q[28];
rz(5*pi/8) q[71];
cz q[71],q[64];
rx(-pi/8) q[64];
cz q[71],q[64];
rx(5*pi/8) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
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
rz(pi) q[70];
cz q[70],q[57];
rx(9*pi/16) q[57];
rz(pi/2) q[57];
rx(pi/2) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
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
cz q[28],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[65],q[64];
rx(pi/2) q[65];
rz(pi) q[65];
rx(-pi/4) q[71];
cz q[28],q[71];
rx(pi/4) q[71];
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
cz q[64],q[65];
rx(pi/2) q[64];
rz(pi) q[64];
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
cz q[27],q[64];
rx(-pi/32) q[64];
cz q[27],q[64];
rx(1.6689711) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
rx(3*pi/4) q[65];
rz(pi/2) q[65];
rx(pi/2) q[65];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
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
rz(pi) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi/2) q[64];
cz q[27],q[64];
rx(-pi/64) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[26],q[27];
rz(-pi/2) q[26];
rx(-pi/2) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[27],q[26];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/64) q[64];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
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
cz q[57],q[70];
rx(-pi/16) q[70];
cz q[57],q[70];
rz(1.7426022) q[57];
rx(9*pi/16) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
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
cz q[64],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[27],q[26];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[27];
rz(3.1784082) q[27];
cz q[27],q[26];
rx(pi/2) q[26];
rz(pi) q[26];
cz q[26],q[37];
rx(pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[37];
rz(pi) q[37];
rx(pi/2) q[64];
rz(pi/2) q[64];
cz q[27],q[64];
rx(-pi/128) q[64];
cz q[27],q[64];
cz q[27],q[26];
rx(-pi/256) q[26];
cz q[27],q[26];
rx(1.5830682) q[26];
rz(pi/2) q[26];
rx(pi/2) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[26],q[27];
rz(-pi/2) q[26];
rx(-pi/2) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[27],q[26];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[37],q[26];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[37];
rz(pi) q[37];
cz q[26],q[37];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[37];
rz(pi) q[37];
cz q[37],q[26];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[37];
rz(pi) q[37];
rx(1.59534) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
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
cz q[28],q[71];
rx(pi/2) q[70];
rz(pi/2) q[70];
cz q[57],q[70];
rx(-pi/32) q[70];
cz q[57],q[70];
rx(pi/32) q[70];
rx(-pi/8) q[71];
cz q[28],q[71];
rx(pi/8) q[71];
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
rx(pi/2) q[64];
rz(pi/2) q[64];
cz q[65],q[64];
rx(-pi/4) q[64];
cz q[65],q[64];
rx(3*pi/4) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
cz q[65],q[64];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rz(-pi/2) q[65];
rx(-pi/2) q[65];
cz q[64],q[65];
rx(pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[65];
rz(5*pi/4) q[65];
cz q[65],q[64];
rx(5*pi/8) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
cz q[57],q[70];
rx(-pi/64) q[70];
cz q[57],q[70];
rx(pi/64) q[70];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[28],q[71];
rx(-pi/16) q[71];
cz q[28],q[71];
rx(pi/16) q[71];
cz q[64],q[71];
rx(-pi/8) q[71];
cz q[64],q[71];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[64],q[27];
rx(pi/2) q[27];
rz(3.3379422) q[27];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[27],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/8) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[71],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi/2) q[64];
cz q[65],q[64];
rx(-pi/4) q[64];
cz q[65],q[64];
rx(3*pi/4) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
cz q[65],q[64];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[64],q[65];
rx(pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[65];
rz(5*pi/4) q[65];
cz q[65],q[64];
rx(5*pi/8) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
cz q[57],q[70];
rx(-pi/128) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/128) q[70];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[28],q[71];
rx(-pi/32) q[71];
cz q[28],q[71];
rx(pi/32) q[71];
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
rz(pi/2) q[28];
cz q[27],q[28];
rx(-pi/16) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi) q[27];
rx(9*pi/16) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
cz q[28],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[28];
rz(3.2397674) q[28];
cz q[28],q[27];
cz q[64],q[27];
rx(-pi/8) q[27];
cz q[64],q[27];
rx(5*pi/8) q[27];
rz(pi/2) q[27];
rx(pi/2) q[27];
rz(9*pi/16) q[64];
rx(1.6198837) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[70];
rx(-pi/64) q[70];
cz q[71],q[70];
rx(1.6198837) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
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
cz q[28],q[71];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(3.1538645) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rx(-pi/32) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[29],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rz(-pi/2) q[29];
rx(-pi/2) q[29];
cz q[28],q[29];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[28];
rx(pi/2) q[29];
rz(pi) q[29];
rx(pi/32) q[71];
cz q[64],q[71];
rx(-pi/16) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[27],q[64];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[64],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[64];
rz(pi/2) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[27],q[26];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[26],q[27];
cz q[26],q[37];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[28],q[29];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[28];
rx(pi/2) q[28];
rz(3.19068) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[28],q[29];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
cz q[18],q[19];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[37];
rz(pi) q[37];
cz q[37],q[26];
rx(pi/2) q[26];
rz(3.1477286) q[26];
rx(pi/2) q[37];
rz(pi) q[37];
cz q[26],q[37];
cz q[26],q[27];
rx(-pi/512) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/512) q[27];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[56];
cz q[65],q[64];
rx(-pi/4) q[64];
cz q[65],q[64];
rx(pi/4) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[64],q[65];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[65],q[64];
rx(pi/2) q[64];
rz(9*pi/8) q[64];
rx(pi/2) q[65];
rz(pi/2) q[65];
cz q[64],q[65];
rx(3*pi/4) q[65];
rz(pi/2) q[65];
rx(pi/2) q[65];
rx(pi/16) q[71];
cz q[64],q[71];
rx(-pi/8) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
rx(5*pi/8) q[71];
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
rz(pi) q[71];
cz q[71],q[64];
cz q[65],q[64];
rx(-pi/4) q[64];
cz q[65],q[64];
rx(pi/4) q[64];
cz q[64],q[27];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[64],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[27],q[26];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[27];
rz(3.1446606) q[27];
cz q[27],q[26];
rx(pi/2) q[26];
rz(pi) q[26];
cz q[26],q[37];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[37];
rz(pi) q[37];
cz q[37],q[26];
rx(pi/2) q[26];
rz(3.2397674) q[26];
rx(pi/2) q[37];
rz(pi/2) q[37];
cz q[26],q[37];
rx(3*pi/4) q[37];
rz(pi/2) q[37];
rx(pi/2) q[37];
rz(5*pi/8) q[65];
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
rz(pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[70],q[71];
rx(-pi/256) q[71];
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
rx(1.59534) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
rx(pi/256) q[71];
cz q[70],q[71];
rx(-pi/128) q[71];
cz q[70],q[71];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[70];
rz(3.1477286) q[70];
cz q[70],q[57];
rx(1.5830682) q[57];
rz(pi/2) q[57];
rx(pi/2) q[57];
rx(pi/128) q[71];
cz q[28],q[71];
rx(-pi/64) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
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
cz q[27],q[28];
rx(-pi/1024) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi) q[27];
rx(1.5738643) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
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
rz(pi/2) q[29];
cz q[18],q[29];
rx(1.6198837) q[71];
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
cz q[64],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[64],q[27];
cz q[26],q[27];
rx(-pi/32) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/32) q[27];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[28],q[71];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(3.3379422) q[28];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[28],q[71];
cz q[28],q[27];
rx(-pi/16) q[27];
cz q[28],q[27];
rx(9*pi/16) q[27];
rz(pi/2) q[27];
rx(pi/2) q[27];
cz q[27],q[64];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[64],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[64];
rz(pi/2) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[28],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[28];
rz(3.1431266) q[28];
cz q[28],q[27];
cz q[28],q[29];
rx(-pi/2048) q[29];
cz q[28],q[29];
rx(pi/2) q[28];
rz(pi) q[28];
rx(1.5723303) q[29];
rz(pi/2) q[29];
rx(pi/2) q[29];
cz q[29],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rz(-pi/2) q[29];
rx(-pi/2) q[29];
cz q[28],q[29];
rx(pi/2) q[28];
rz(pi/2) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[18],q[29];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[18];
rx(pi/2) q[18];
rz(3.1423596) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[18],q[29];
cz q[18],q[19];
rx(-pi/4096) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi) q[18];
rx(1.5715633) q[19];
rz(pi/2) q[19];
rx(pi/2) q[19];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi/2) q[18];
rx(pi/2) q[19];
rz(3.1421679) q[19];
cz q[19],q[18];
cz q[19],q[56];
rx(-pi/8192) q[56];
cz q[19],q[56];
rx(1.5711798) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
cz q[56],q[63];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
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
cz q[19],q[56];
rx(-pi/16384) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi) q[19];
rx(1.5709881) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[65],q[64];
rx(-pi/8) q[64];
cz q[65],q[64];
rx(5*pi/8) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
cz q[64],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[64];
rz(3.2888548) q[64];
cz q[64],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[27],q[26];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[27];
rz(3.19068) q[27];
cz q[27],q[26];
cz q[37],q[26];
rx(-pi/4) q[26];
cz q[37],q[26];
rx(3*pi/4) q[26];
rz(pi/2) q[26];
rx(pi/2) q[26];
rz(5*pi/8) q[37];
cz q[70],q[71];
rx(-pi/512) q[71];
cz q[70],q[71];
rx(pi/512) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
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
cz q[57],q[70];
rx(-pi/256) q[70];
cz q[57],q[70];
rz(1.5769322) q[57];
rx(pi/256) q[70];
rx(1.5738643) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[28];
rx(-pi/1024) q[28];
cz q[71],q[28];
rx(1.5738643) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[28],q[71];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[28],q[71];
cz q[28],q[29];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[28];
rx(pi/2) q[28];
rz(3.1661363) q[28];
rx(pi/2) q[29];
rz(pi/2) q[29];
cz q[28],q[29];
rx(1.5723303) q[29];
rz(pi/2) q[29];
rx(pi/2) q[29];
cz q[29],q[18];
rx(-pi/2048) q[18];
cz q[29],q[18];
rx(pi/2048) q[18];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
cz q[57],q[70];
rx(-pi/512) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[56],q[57];
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
rx(1.5738643) q[19];
rz(pi/2) q[19];
rx(pi/2) q[19];
cz q[19],q[18];
rx(-pi/1024) q[18];
cz q[19],q[18];
rx(pi/1024) q[18];
cz q[29],q[18];
rz(-pi/2) q[18];
rx(-pi/2) q[18];
rz(-pi/2) q[29];
rx(-pi/2) q[29];
cz q[18],q[29];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[29],q[18];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi/2) q[18];
rx(pi/2) q[19];
rz(3.1423596) q[19];
cz q[19],q[18];
rx(1.5723303) q[18];
rz(pi/2) q[18];
rx(pi/2) q[18];
rx(pi/2) q[56];
rz(pi) q[56];
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
cz q[19],q[56];
rx(-pi/4096) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi) q[19];
rx(1.5715633) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
cz q[56],q[19];
rx(pi/2) q[19];
rz(pi) q[19];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[56];
rz(3.1419761) q[56];
cz q[56],q[19];
cz q[18],q[19];
rx(-pi/2048) q[19];
cz q[18],q[19];
rz(-pi/2) q[18];
rx(-pi/2) q[18];
rx(1.5723303) q[19];
rz(pi/2) q[19];
rx(pi/2) q[19];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi/2) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[19],q[18];
cz q[56],q[57];
rx(-pi/8192) q[57];
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
rx(1.5715633) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
rx(pi/8192) q[57];
cz q[56],q[57];
rx(-pi/4096) q[57];
cz q[56],q[57];
rx(1.5715633) q[57];
rz(pi/2) q[57];
rx(pi/2) q[57];
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
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi/2) q[63];
rx(pi/512) q[70];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[28],q[71];
rx(-pi/128) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
rx(1.59534) q[71];
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
rz(3.1538645) q[71];
cz q[71],q[28];
cz q[27],q[28];
rx(-pi/64) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi) q[27];
rx(1.6198837) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
cz q[28],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[28];
rz(3.1661363) q[28];
cz q[28],q[27];
cz q[64],q[27];
rx(-pi/32) q[27];
cz q[64],q[27];
rx(pi/32) q[27];
cz q[71],q[70];
rx(-pi/256) q[70];
cz q[71],q[70];
rx(1.5830682) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(3.1477286) q[70];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[70],q[71];
cz q[28],q[71];
rx(-pi/128) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[29],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[28],q[29];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[29];
rz(3.1600004) q[29];
cz q[29],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/128) q[71];
cz q[64],q[71];
rx(-pi/64) q[71];
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
rz(pi) q[65];
cz q[65],q[64];
rx(9*pi/16) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
cz q[64],q[27];
rx(-pi/16) q[27];
cz q[64],q[27];
rx(9*pi/16) q[27];
rz(pi/2) q[27];
rx(pi/2) q[27];
cz q[27],q[26];
rz(-pi/2) q[26];
rx(-pi/2) q[26];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[27];
rz(5*pi/4) q[27];
cz q[27],q[26];
cz q[37],q[26];
rx(-pi/8) q[26];
cz q[37],q[26];
rx(pi/8) q[26];
cz q[27],q[26];
rx(-pi/4) q[26];
cz q[27],q[26];
rx(3*pi/4) q[26];
rz(pi/2) q[26];
rx(pi/2) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[37],q[26];
rz(-pi/2) q[26];
rx(-pi/2) q[26];
rx(pi/2) q[37];
rz(pi) q[37];
cz q[26],q[37];
rx(pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[37];
rz(11*pi/8) q[37];
cz q[37],q[26];
rx(9*pi/16) q[26];
rz(pi/2) q[26];
rx(pi/2) q[26];
rz(1.6689711) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
rx(pi/64) q[71];
cz q[64],q[71];
rx(-pi/32) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
rx(1.6689711) q[71];
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
cz q[64],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[64];
rz(9*pi/8) q[64];
cz q[64],q[27];
cz q[26],q[27];
rx(-pi/16) q[27];
cz q[26],q[27];
rz(-pi/2) q[26];
rx(-pi/2) q[26];
rx(pi/16) q[27];
cz q[64],q[27];
rx(-pi/8) q[27];
cz q[64],q[27];
rx(5*pi/8) q[27];
rz(pi/2) q[27];
rx(pi/2) q[27];
cz q[27],q[26];
rx(pi/2) q[26];
rz(pi) q[26];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[27],q[26];
cz q[37],q[26];
rx(-pi/4) q[26];
cz q[37],q[26];
rx(pi/4) q[26];
cz q[64],q[65];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[65],q[64];
rx(pi/2) q[64];
rz(3.1661363) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[64],q[65];
rx(pi/2) q[65];
rz(pi) q[65];
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
rz(pi) q[28];
cz q[70],q[71];
rx(-pi/512) q[71];
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
rz(pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
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
rz(3.1446606) q[19];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[19],q[56];
cz q[19],q[18];
rx(-pi/1024) q[18];
cz q[19],q[18];
rx(pi/1024) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
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
rx(pi/2) q[56];
rz(pi) q[56];
cz q[56],q[19];
rx(pi/2) q[19];
rz(pi) q[19];
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
rx(pi/2) q[57];
rz(3.1431266) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
cz q[57],q[70];
rx(-pi/2048) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rx(1.5723303) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
rx(1.5769322) q[71];
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
cz q[29],q[28];
rx(-pi/256) q[28];
cz q[29],q[28];
rx(pi/256) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi) q[27];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
cz q[28],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[29],q[18];
rx(-pi/512) q[18];
cz q[29],q[18];
rx(1.5769322) q[18];
rz(pi/2) q[18];
rx(pi/2) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[18],q[29];
rz(-pi/2) q[18];
rx(-pi/2) q[18];
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
cz q[64],q[27];
rx(-pi/128) q[27];
cz q[64],q[27];
rx(pi/128) q[27];
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
rz(pi) q[65];
cz q[65],q[64];
rx(9*pi/16) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(3.19068) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[28],q[71];
cz q[28],q[27];
rx(-pi/64) q[27];
cz q[28],q[27];
rx(pi/64) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi/2) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[28];
rx(1.6689711) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
cz q[28],q[27];
rx(-pi/32) q[27];
cz q[28],q[27];
rx(pi/32) q[27];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
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
cz q[29],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[18],q[29];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(3.1446606) q[29];
cz q[29],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
cz q[64],q[27];
rx(-pi/16) q[27];
cz q[64],q[27];
rx(pi/16) q[27];
cz q[26],q[27];
rz(-pi/2) q[26];
rx(-pi/2) q[26];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
cz q[27],q[26];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[27];
rz(pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(pi/2) q[26];
rx(3*pi/4) q[27];
rz(pi/2) q[27];
rx(pi/2) q[27];
cz q[37],q[26];
rx(-pi/8) q[26];
cz q[37],q[26];
rx(pi/8) q[26];
cz q[27],q[26];
rx(-pi/4) q[26];
cz q[27],q[26];
rx(3*pi/4) q[26];
rz(pi/2) q[26];
rx(pi/2) q[26];
cz q[27],q[28];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[28],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[37],q[26];
rz(-pi/2) q[26];
rx(-pi/2) q[26];
rx(pi/2) q[37];
rz(pi) q[37];
cz q[26],q[37];
rx(pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[37];
rz(pi) q[37];
cz q[37],q[26];
rx(9*pi/16) q[26];
rz(pi/2) q[26];
rx(pi/2) q[26];
rx(pi/2) q[37];
rz(pi) q[37];
cz q[64],q[65];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[65],q[64];
rx(pi/2) q[64];
rz(3.1600004) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[64],q[65];
cz q[64],q[27];
rx(-pi/256) q[27];
cz q[64],q[27];
rx(pi/256) q[27];
rx(pi/2) q[65];
rz(pi) q[65];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rz(-pi/2) q[70];
rx(-pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi/2) q[28];
cz q[29],q[28];
rx(-pi/1024) q[28];
cz q[29],q[28];
rx(1.5738643) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
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
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
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
rx(5*pi/8) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[28],q[71];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(3.1784082) q[28];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[28],q[71];
cz q[28],q[27];
rx(-pi/128) q[27];
cz q[28],q[27];
rx(pi/128) q[27];
cz q[64],q[71];
rx(-pi/512) q[71];
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
rz(pi) q[65];
cz q[65],q[64];
rx(1.6689711) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
rx(pi/512) q[71];
cz q[28],q[71];
rx(-pi/256) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
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
rx(1.6198837) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
cz q[28],q[27];
rx(-pi/64) q[27];
cz q[28],q[27];
rx(pi/64) q[27];
rz(1.59534) q[28];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[64],q[27];
rx(-pi/32) q[27];
cz q[64],q[27];
rx(pi/32) q[27];
cz q[26],q[27];
rx(-pi/16) q[27];
cz q[26],q[27];
rz(-pi/2) q[26];
rx(-pi/2) q[26];
rx(9*pi/16) q[27];
rz(pi/2) q[27];
rx(pi/2) q[27];
rz(1.6198837) q[64];
rx(pi/256) q[71];
cz q[28],q[71];
rx(-pi/128) q[71];
cz q[28],q[71];
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
cz q[28],q[29];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[27],q[28];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[28],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[27],q[28];
cz q[27],q[26];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[27];
rz(3.2397674) q[27];
cz q[27],q[26];
rx(pi/2) q[26];
rz(pi) q[26];
cz q[26],q[37];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[28];
rz(pi) q[28];
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
cz q[29],q[18];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[19],q[18];
cz q[19],q[20];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[20],q[19];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[19],q[20];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[20];
rz(pi/2) q[20];
rx(pi/2) q[29];
rz(pi) q[29];
rx(pi/2) q[37];
rz(pi) q[37];
cz q[37],q[26];
rx(pi/2) q[26];
rz(11*pi/8) q[26];
rx(pi/2) q[37];
rz(pi) q[37];
cz q[26],q[37];
rx(pi/2) q[37];
rz(pi/2) q[37];
rx(pi/128) q[71];
cz q[64],q[71];
rx(-pi/64) q[71];
cz q[64],q[71];
cz q[64],q[65];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[65],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[65];
rz(pi) q[65];
cz q[64],q[65];
rx(pi/2) q[65];
rz(pi/2) q[65];
rx(pi/64) q[71];
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
rz(pi/2) q[28];
cz q[27],q[28];
rx(-pi/32) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/32) q[28];
cz q[70],q[71];
rx(-pi/8) q[71];
cz q[70],q[71];
rz(9*pi/16) q[70];
rx(5*pi/8) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[28];
rz(-pi/2) q[28];
rx(-pi/2) q[28];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[28],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[28],q[27];
cz q[26],q[27];
rx(-pi/4) q[27];
cz q[26],q[27];
rx(pi/4) q[27];
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
cz q[28],q[29];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[70],q[71];
rx(-pi/16) q[71];
cz q[70],q[71];
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
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
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
rz(pi) q[70];
rx(9*pi/16) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[28];
cz q[27],q[28];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[28],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[28];
rz(pi/2) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi/2) q[27];
cz q[26],q[27];
rx(-pi/8) q[27];
cz q[26],q[27];
rx(pi/8) q[27];
rx(3*pi/4) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
cz q[28],q[27];
rx(-pi/4) q[27];
cz q[28],q[27];
rx(pi/4) q[27];
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
rz(pi) q[28];
cz q[27],q[28];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[28],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(pi) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[27],q[26];
rx(pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[27];
rz(pi) q[27];
cz q[26],q[27];
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
cz q[29],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
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
rz(pi/2) q[19];
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
rz(pi/2) q[28];
cz q[29],q[18];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi) q[29];
cz q[18],q[29];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
rz(pi/2) q[29];
cz q[29],q[18];
rx(pi/2) q[18];
rz(pi/2) q[18];
cz q[64],q[27];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[64],q[27];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[71],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[71],q[64];
rx(pi/2) q[64];
rz(pi/2) q[64];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi) q[70];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[71],q[70];
rx(pi/2) q[70];
rz(pi/2) q[70];
barrier q[10],q[77],q[74],q[28],q[19],q[27],q[46],q[43],q[52],q[61],q[6],q[3],q[65],q[15],q[67],q[12],q[79],q[76],q[21],q[30],q[39],q[36],q[45],q[54],q[63],q[8],q[5],q[72],q[69],q[14],q[78],q[23],q[32],q[64],q[41],q[38],q[47],q[71],q[1],q[26],q[62],q[7],q[18],q[16],q[25],q[22],q[34],q[31],q[40],q[49],q[58],q[55],q[0],q[56],q[9],q[73],q[57],q[70],q[24],q[33],q[42],q[51],q[48],q[60],q[20],q[2],q[66],q[11],q[75],q[29],q[17],q[37],q[35],q[44],q[53],q[50],q[59],q[4],q[68],q[13];
measure q[29] -> meas[0];
measure q[28] -> meas[1];
measure q[71] -> meas[2];
measure q[57] -> meas[3];
measure q[64] -> meas[4];
measure q[27] -> meas[5];
measure q[37] -> meas[6];
measure q[20] -> meas[7];
measure q[65] -> meas[8];
measure q[26] -> meas[9];
measure q[56] -> meas[10];
measure q[70] -> meas[11];
measure q[19] -> meas[12];
measure q[18] -> meas[13];
