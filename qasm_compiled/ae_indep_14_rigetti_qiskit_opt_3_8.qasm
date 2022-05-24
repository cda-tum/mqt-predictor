OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg meas[14];
rz(pi/2) q[19];
rx(pi/2) q[19];
rz(pi/2) q[25];
rx(pi/2) q[25];
rz(pi/2) q[26];
rx(pi/2) q[26];
rz(2.129148) q[26];
rz(pi/2) q[27];
rz(-pi/2) q[28];
rz(-pi) q[29];
rx(-pi/2) q[29];
rz(-pi) q[37];
rx(-pi/2) q[37];
rz(-pi/2) q[56];
rx(pi/2) q[56];
rz(3.0504818) q[56];
rz(-pi/2) q[57];
cz q[56],q[57];
rx(pi) q[56];
rx(0.92729522) q[57];
rz(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(pi/2) q[56];
rx(-1.4796855) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(-pi/2) q[56];
rx(pi/2) q[56];
cz q[19],q[56];
rx(-pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[56];
rz(3*pi/2) q[56];
rx(pi/2) q[57];
rz(2.8577985) q[57];
rx(-2.3911695) q[58];
rz(-pi) q[58];
cz q[57],q[58];
rx(pi) q[57];
rx(1.2870023) q[58];
rz(pi/2) q[58];
cz q[57],q[58];
rx(pi/2) q[57];
rz(pi/2) q[57];
rx(-1.2870023) q[57];
cz q[56],q[57];
rx(pi) q[56];
rx(0.56758825) q[57];
cz q[56],q[57];
rz(1.5692623) q[56];
rx(2.5740044) q[57];
rx(pi/2) q[58];
rz(2.3904025) q[58];
rx(2.9787725) q[64];
rz(-pi/2) q[69];
rx(pi/2) q[69];
rz(3.0423832) q[69];
rz(-pi) q[70];
rx(-pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(-pi/2) q[70];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rz(-pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi) q[57];
rx(pi/2) q[70];
cz q[69],q[70];
rx(pi) q[69];
rx(1.1351764) q[70];
rz(pi/2) q[70];
cz q[69],q[70];
rz(-1.0947141) q[69];
rx(pi/2) q[70];
rz(-1.1351764) q[70];
rz(-pi) q[71];
rx(-1.9347757) q[71];
cz q[70],q[71];
rx(pi) q[70];
rx(0.87124027) q[71];
rz(pi) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(pi/2) q[70];
rx(-0.87123975) q[70];
cz q[57],q[70];
rz(pi/2) q[57];
rx(pi) q[57];
rx(1.3991131) q[70];
cz q[57],q[70];
rz(-1.5830682) q[57];
rx(1.7424796) q[70];
rx(-2.7776133) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(-pi/2) q[71];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
rz(3*pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
rz(-pi) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rx(pi/2) q[71];
cz q[28],q[71];
rx(1.2274299) q[28];
rz(pi/2) q[28];
rx(pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(1.9141628) q[28];
rx(-pi/2) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(-pi/2) q[28];
rx(pi/2) q[28];
cz q[27],q[28];
rx(-0.88406549) q[27];
rz(pi) q[27];
rx(pi/2) q[28];
cz q[27],q[28];
rx(2.4548597) q[27];
cz q[26],q[27];
rx(pi) q[26];
rx(1.3734617) q[27];
rz(pi) q[27];
cz q[26],q[27];
rz(-1.0124447) q[26];
rx(-pi/2) q[26];
cz q[26],q[37];
rx(pi/2) q[26];
rx(-1.3734659) q[27];
rx(-pi/2) q[28];
cz q[28],q[29];
rx(pi/2) q[28];
rz(-pi/2) q[29];
rx(pi/2) q[29];
cz q[28],q[29];
rx(-pi/2) q[28];
rx(pi/2) q[29];
cz q[28],q[29];
rx(-pi/2) q[28];
rz(-pi/2) q[28];
rz(-1.7180585) q[29];
rz(-pi/2) q[37];
rx(pi/2) q[37];
cz q[26],q[37];
rx(-pi/2) q[26];
rz(-pi/2) q[26];
rx(pi/2) q[37];
cz q[26],q[37];
rx(-pi) q[26];
cz q[26],q[27];
rz(pi/2) q[26];
rx(pi) q[26];
rx(0.39465931) q[27];
rz(pi) q[27];
cz q[26],q[27];
rx(-pi/2) q[26];
cz q[25],q[26];
rx(pi/2) q[25];
rx(pi/2) q[26];
cz q[25],q[26];
rx(-pi/2) q[25];
rz(pi/2) q[25];
rx(pi/2) q[26];
cz q[25],q[26];
rx(pi/2) q[26];
rz(3*pi/2) q[26];
rx(0.39466095) q[27];
cz q[26],q[27];
rx(pi) q[26];
rx(0.78931862) q[27];
rz(pi/2) q[27];
cz q[26],q[27];
rz(-pi/2) q[26];
rx(pi/2) q[26];
rx(-pi/2) q[27];
rz(0.78932185) q[27];
cz q[27],q[28];
rx(pi) q[27];
rx(1.5629554) q[28];
rz(pi/2) q[28];
cz q[27],q[28];
rx(-pi) q[27];
rz(0.0078473732) q[27];
rx(pi) q[27];
cz q[27],q[64];
rz(pi/2) q[27];
rx(pi) q[27];
rx(-pi/2) q[28];
rz(0.84372238) q[28];
rx(pi/2) q[28];
rz(1.4726216) q[37];
rx(0.01568181) q[64];
rz(pi/2) q[64];
cz q[27],q[64];
rx(-pi) q[27];
rz(-1.5864781) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(pi) q[26];
cz q[25],q[26];
rx(pi/2) q[25];
rz(pi) q[25];
rx(pi/2) q[26];
rz(pi) q[26];
cz q[26],q[25];
rx(pi/2) q[25];
rz(pi/2) q[25];
rx(pi/2) q[26];
rz(pi/2) q[26];
cz q[25],q[26];
rx(1.5685428) q[26];
rz(pi/2) q[26];
rz(-0.32673163) q[27];
rx(pi/2) q[27];
rx(-pi/2) q[64];
rz(0.16282019) q[64];
rx(pi/2) q[64];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(-pi/2) q[71];
rx(pi/2) q[71];
cz q[64],q[71];
rx(-pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(4.1285809) q[64];
rz(-pi/2) q[71];
rx(-pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rx(pi/2) q[71];
cz q[28],q[71];
rx(pi/4) q[28];
rz(pi/2) q[28];
rx(pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(3*pi/4) q[28];
rx(pi/2) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(-pi/2) q[28];
rx(pi/2) q[28];
cz q[27],q[28];
rx(-3*pi/8) q[27];
rz(pi/2) q[27];
rx(pi/2) q[28];
cz q[27],q[28];
rx(-pi/2) q[27];
rz(5*pi/8) q[27];
rx(-pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(-pi/2) q[27];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-7*pi/16) q[26];
rz(pi) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-pi/16) q[26];
rz(-1.3767003) q[27];
rx(-pi) q[27];
rz(-2.4221619) q[28];
cz q[37],q[26];
rx(pi/32) q[26];
cz q[37],q[26];
rx(-pi/32) q[26];
rz(pi/2) q[26];
rx(pi/2) q[26];
rx(-pi/2) q[37];
rz(-1.6291205) q[71];
cz q[28],q[71];
rx(pi/4) q[71];
cz q[28],q[71];
rx(-pi/2) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rx(pi/2) q[28];
cz q[27],q[28];
rx(-pi/2) q[27];
rx(pi/2) q[28];
cz q[27],q[28];
rx(-pi/2) q[28];
rz(-7*pi/8) q[28];
rx(-pi/4) q[71];
cz q[28],q[71];
rx(pi/8) q[71];
cz q[28],q[71];
cz q[28],q[27];
rx(pi/4) q[27];
cz q[28],q[27];
rx(-3*pi/4) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-pi/2) q[26];
rz(3*pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
cz q[26],q[37];
rx(pi/2) q[26];
rx(-pi/2) q[27];
rz(pi) q[27];
rx(-pi/2) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rx(pi/2) q[28];
cz q[27],q[28];
rx(-pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[28];
cz q[27],q[28];
rx(-pi/2) q[27];
rz(pi/2) q[28];
cz q[29],q[28];
rx(pi/64) q[28];
cz q[29],q[28];
rx(-pi/64) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
rx(pi/2) q[37];
cz q[26],q[37];
rx(-pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[37];
cz q[26],q[37];
rx(pi/2) q[26];
cz q[26],q[27];
rx(pi/2) q[26];
rz(-pi/2) q[27];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-pi/2) q[26];
rz(3*pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rz(-9*pi/16) q[27];
rx(-pi/2) q[37];
rz(-pi/2) q[37];
rx(-5*pi/8) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rx(pi/2) q[71];
cz q[28],q[71];
rx(-pi/2) q[28];
rx(pi/2) q[71];
cz q[28],q[71];
cz q[27],q[28];
rx(pi/16) q[28];
cz q[27],q[28];
rx(-pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(-pi/2) q[27];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(9*pi/8) q[26];
cz q[26],q[37];
rz(pi/2) q[27];
rx(-0.42662749) q[27];
rx(-pi/16) q[28];
cz q[29],q[28];
rx(pi/32) q[28];
cz q[29],q[28];
rx(-pi/32) q[28];
rz(-pi/2) q[28];
rx(-pi/2) q[29];
rx(pi/8) q[37];
cz q[26],q[37];
cz q[26],q[27];
rx(pi) q[26];
rx(pi/4) q[27];
rz(pi/2) q[27];
cz q[26],q[27];
rz(pi/2) q[26];
rx(-pi/2) q[26];
rx(-pi/2) q[27];
rz(2.782822) q[27];
rx(-5*pi/8) q[37];
cz q[26],q[37];
rx(pi/2) q[26];
rx(pi/2) q[37];
cz q[26],q[37];
rx(-pi/2) q[26];
rx(pi/2) q[37];
cz q[26],q[37];
rz(pi/2) q[37];
rz(pi/2) q[71];
cz q[64],q[71];
rx(pi/128) q[71];
cz q[64],q[71];
rx(-1.59534) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
cz q[57],q[70];
rx(pi/256) q[70];
cz q[57],q[70];
rx(-pi/256) q[70];
rx(-pi/2) q[71];
rz(-1.5769322) q[71];
cz q[71],q[70];
rx(pi/512) q[70];
cz q[71],q[70];
rx(-pi/512) q[70];
cz q[69],q[70];
rx(pi/1024) q[70];
cz q[69],q[70];
rx(-1.5738643) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
cz q[56],q[57];
rx(pi/2048) q[57];
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
rz(-pi/8192) q[56];
cz q[56],q[19];
rx(1.5677284) q[19];
rz(pi/2) q[19];
rx(pi/2) q[19];
rx(-pi/2048) q[57];
cz q[58],q[57];
rx(pi/4096) q[57];
cz q[58],q[57];
rx(-pi/4096) q[57];
cz q[56],q[57];
rx(pi/8192) q[57];
cz q[56],q[57];
rx(-pi/8192) q[57];
rz(pi/2) q[57];
rx(pi/2) q[57];
rx(pi/2) q[58];
rz(pi) q[58];
rx(-pi/2) q[70];
rz(-3.1661363) q[70];
rx(-pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rx(pi/2) q[71];
cz q[28],q[71];
rx(-pi/2) q[28];
rz(-1.5830682) q[28];
rx(pi/2) q[71];
cz q[28],q[71];
rx(-pi/2) q[71];
rz(-pi/2) q[71];
cz q[64],q[71];
rx(pi/64) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rx(-pi/64) q[71];
cz q[70],q[71];
rx(pi/128) q[71];
cz q[70],q[71];
rx(-pi/128) q[71];
cz q[28],q[71];
rx(pi/256) q[71];
cz q[28],q[71];
cz q[28],q[29];
rx(pi/2) q[28];
rz(-pi/2) q[29];
rx(pi/2) q[29];
cz q[28],q[29];
rx(-pi/2) q[28];
rz(pi/2) q[28];
rx(pi/2) q[29];
cz q[28],q[29];
rx(-pi/2) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(-pi/2) q[28];
rx(pi/2) q[28];
cz q[27],q[28];
rx(-pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(5*pi/16) q[27];
cz q[27],q[26];
rx(pi/16) q[26];
cz q[27],q[26];
rx(-pi/16) q[26];
rz(-pi/2) q[26];
rz(pi/2) q[28];
cz q[27],q[28];
rx(pi/8) q[28];
cz q[27],q[28];
rx(-pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-pi/2) q[26];
rz(-3*pi/4) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
cz q[26],q[37];
rx(-pi/2) q[27];
cz q[27],q[64];
rx(pi/2) q[27];
rx(-pi/8) q[28];
rz(-pi/2) q[29];
rx(-pi/2) q[29];
rx(pi/4) q[37];
cz q[26],q[37];
rx(-pi/4) q[37];
rz(-pi/2) q[64];
rx(pi/2) q[64];
cz q[27],q[64];
rx(-1.4726216) q[27];
rz(pi/2) q[27];
rx(pi/2) q[64];
cz q[27],q[64];
rx(-pi/2) q[27];
rz(-1.3551436) q[27];
cz q[27],q[28];
rx(pi/16) q[28];
cz q[27],q[28];
rx(-pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-pi/2) q[26];
rz(pi/8) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
cz q[26],q[37];
rx(-pi/2) q[27];
rz(-pi/2) q[27];
rx(-pi/16) q[28];
rx(pi/8) q[37];
cz q[26],q[37];
cz q[26],q[27];
rx(pi/4) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rx(-pi/4) q[27];
rz(-pi/2) q[27];
rx(-5*pi/8) q[37];
cz q[26],q[37];
rx(pi/2) q[26];
rx(pi/2) q[37];
cz q[26],q[37];
rx(-pi/2) q[26];
rx(pi/2) q[37];
cz q[26],q[37];
rz(pi/2) q[37];
rz(-3.0434179) q[64];
rx(pi/2) q[64];
rz(-pi/2) q[64];
rx(-1.5830682) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
cz q[69],q[70];
rx(pi/512) q[70];
cz q[69],q[70];
rx(-pi/2) q[69];
rx(-1.5769322) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(-pi/2) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
cz q[19],q[56];
rx(pi/1024) q[56];
cz q[19],q[56];
rz(1.5646604) q[19];
rx(-pi/1024) q[56];
rx(-pi/2) q[57];
cz q[57],q[58];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[58];
rz(pi) q[58];
cz q[58],q[57];
rx(pi/2) q[57];
rz(3.1400587) q[57];
rx(pi/2) q[58];
rz(pi/2) q[58];
cz q[57],q[58];
cz q[57],q[56];
rx(pi/2048) q[56];
cz q[57],q[56];
rx(-pi/2048) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
rx(1.5700293) q[58];
rz(pi/2) q[58];
rx(pi/2) q[58];
rx(-pi/2) q[70];
rz(pi) q[70];
rx(-pi/2) q[71];
rz(-1.7180585) q[71];
cz q[71],q[64];
rx(pi/64) q[64];
cz q[71],q[64];
rx(-pi/64) q[64];
rz(-pi/2) q[64];
cz q[71],q[28];
rx(pi/32) q[28];
cz q[71],q[28];
rx(-pi/32) q[28];
rz(-pi/2) q[28];
cz q[28],q[29];
rx(pi/2) q[28];
rz(-pi/2) q[29];
rx(pi/2) q[29];
cz q[28],q[29];
rx(-pi/2) q[28];
rz(1.3989905) q[28];
rx(pi/2) q[29];
cz q[28],q[29];
rx(-pi/2) q[29];
rz(-pi/2) q[29];
rx(-pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rx(pi/2) q[71];
cz q[64],q[71];
rx(-pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[71];
cz q[64],q[71];
rx(-pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(-pi/2) q[64];
rx(pi/2) q[64];
cz q[27],q[64];
rx(-pi/2) q[27];
rz(-3*pi/16) q[27];
rx(pi/2) q[64];
cz q[27],q[64];
cz q[27],q[26];
rx(pi/16) q[26];
cz q[27],q[26];
rx(-pi/16) q[26];
rz(-pi/2) q[26];
rx(-pi/2) q[64];
rz(-pi/2) q[64];
cz q[27],q[64];
rx(pi/8) q[64];
cz q[27],q[64];
rx(-pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-pi/2) q[26];
rz(-3*pi/4) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
cz q[26],q[37];
rx(-pi/2) q[27];
rz(-pi/2) q[27];
rx(pi/4) q[37];
cz q[26],q[37];
rx(-pi/4) q[37];
rx(-pi/8) q[64];
rz(-pi/2) q[64];
rx(-pi/2) q[71];
rz(-pi/2) q[71];
cz q[28],q[71];
rx(pi/128) q[71];
cz q[28],q[71];
cz q[28],q[29];
rx(pi/64) q[29];
cz q[28],q[29];
cz q[28],q[27];
rx(pi/32) q[27];
cz q[28],q[27];
rx(-pi/32) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
rx(-pi/64) q[29];
rx(-1.59534) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(-pi) q[70];
cz q[69],q[70];
rx(pi/2) q[69];
rx(pi/2) q[70];
cz q[69],q[70];
rx(-1.5585245) q[69];
rz(pi/2) q[69];
rx(pi/2) q[70];
cz q[69],q[70];
rx(-pi/2) q[69];
rz(-pi/256) q[69];
rx(-pi) q[70];
rz(-2.1675638) q[70];
rx(-pi/2) q[71];
rz(pi/2) q[71];
rx(-pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rz(-pi/2) q[71];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
cz q[69],q[70];
rx(pi/2) q[69];
rz(pi) q[69];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[69];
rx(pi/2) q[69];
rz(pi/2) q[69];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[69],q[70];
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
rz(-pi/1024) q[70];
cz q[70],q[57];
rx(-pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
cz q[19],q[56];
rx(pi/512) q[56];
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
rz(1.4102817) q[18];
rx(pi/2) q[19];
rz(pi/2) q[19];
cz q[18],q[19];
rx(-pi/2) q[18];
rx(-pi/2) q[19];
rx(-pi/512) q[56];
rz(-pi/2) q[56];
rz(pi/2) q[57];
cz q[58],q[57];
rx(pi/4096) q[57];
cz q[58],q[57];
rx(1.5700293) q[57];
rz(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rx(-pi/2) q[57];
rz(-pi/2) q[57];
rz(1.3530577) q[58];
rx(-pi/2) q[58];
cz q[70],q[57];
rx(pi/1024) q[57];
cz q[70],q[57];
rx(-pi/1024) q[57];
rz(-pi/2) q[57];
cz q[57],q[58];
rx(pi/2) q[57];
rz(-pi/2) q[58];
rx(pi/2) q[58];
cz q[57],q[58];
rx(-1.5692623) q[57];
rz(pi/2) q[57];
rx(pi/2) q[58];
cz q[57],q[58];
rx(pi/2) q[57];
rz(-1.3637956) q[57];
rz(1.5692623) q[58];
rx(-pi/2) q[71];
cz q[71],q[28];
rx(pi/2) q[28];
rz(pi) q[28];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(pi/2) q[28];
rx(pi/2) q[71];
rz(3.5719248) q[71];
cz q[71],q[28];
rx(1.5462526) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
cz q[28],q[29];
rx(pi/128) q[29];
cz q[28],q[29];
rz(-3.2888548) q[28];
cz q[28],q[27];
rx(pi/64) q[27];
cz q[28],q[27];
rx(-pi/64) q[27];
rz(-pi/2) q[27];
rx(1.5462526) q[29];
rz(-pi) q[29];
cz q[18],q[29];
rx(pi/2) q[18];
rx(pi/2) q[29];
cz q[18],q[29];
rx(-1.5585245) q[18];
rz(pi) q[18];
rx(pi/2) q[29];
cz q[18],q[29];
rx(-pi/256) q[18];
rz(-0.1482428) q[29];
rx(pi/2) q[29];
rx(-pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(-pi/2) q[71];
rx(pi/2) q[71];
cz q[64],q[71];
rx(-7*pi/16) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
cz q[64],q[71];
rx(-2.5149109) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(-pi/2) q[64];
rx(pi/2) q[64];
cz q[27],q[64];
rx(-pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[64];
cz q[27],q[64];
rx(-pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(-pi/2) q[27];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-pi/2) q[26];
rz(5*pi/8) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
cz q[26],q[37];
rx(-pi/2) q[27];
rz(-pi/2) q[27];
rx(-0.42662749) q[27];
rx(pi/8) q[37];
cz q[26],q[37];
cz q[26],q[27];
rx(pi) q[26];
rx(pi/4) q[27];
rz(pi/2) q[27];
cz q[26],q[27];
rz(pi/2) q[26];
rx(-pi/2) q[26];
rx(-pi/2) q[27];
rz(2.782822) q[27];
rx(-5*pi/8) q[37];
cz q[26],q[37];
rx(pi/2) q[26];
rx(pi/2) q[37];
cz q[26],q[37];
rx(-pi/2) q[26];
rx(pi/2) q[37];
cz q[26],q[37];
rz(pi/2) q[37];
rx(-0.42662749) q[37];
rx(-pi/2) q[64];
rz(-pi/2) q[64];
rz(15*pi/16) q[71];
rx(pi/2) q[71];
rz(pi/2) q[71];
cz q[28],q[71];
rx(pi/32) q[71];
cz q[28],q[71];
rx(-pi/2) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(-pi/2) q[28];
rx(pi/2) q[28];
cz q[27],q[28];
rx(-pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(5*pi/16) q[27];
cz q[27],q[26];
rx(pi/16) q[26];
cz q[27],q[26];
rx(-pi/16) q[26];
rz(-pi/2) q[26];
rz(pi/2) q[28];
cz q[27],q[28];
rx(pi/8) q[28];
cz q[27],q[28];
rx(-pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-pi/2) q[26];
rz(3*pi/4) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
cz q[26],q[37];
rx(pi) q[26];
rx(-pi/2) q[27];
rz(-pi/2) q[27];
rx(-pi/8) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
cz q[28],q[29];
rx(pi/2) q[28];
rz(-pi/2) q[29];
rx(pi/2) q[29];
cz q[28],q[29];
rx(-pi/2) q[28];
rz(pi/2) q[28];
rx(pi/2) q[29];
cz q[28],q[29];
rx(pi/2) q[28];
rz(pi) q[28];
rz(-pi/2) q[29];
rx(-pi/2) q[29];
rx(pi/4) q[37];
rz(pi) q[37];
cz q[26],q[37];
rz(pi/2) q[26];
rx(-pi/2) q[26];
rx(-0.35877067) q[37];
rx(-1.6689711) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
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
cz q[28],q[29];
rx(pi/2) q[28];
rz(-pi/2) q[29];
rx(pi/2) q[29];
cz q[28],q[29];
rx(-pi/2) q[28];
rz(3*pi/2) q[28];
rx(pi/2) q[29];
cz q[28],q[29];
rz(3.7061484) q[29];
cz q[29],q[18];
rx(pi/512) q[18];
cz q[29],q[18];
rx(-pi/512) q[18];
rz(pi/2) q[18];
rx(pi/2) q[18];
cz q[18],q[19];
rx(pi/2) q[18];
rx(pi/2) q[19];
cz q[18],q[19];
rx(-pi/2) q[18];
rz(pi/2) q[18];
rx(pi/2) q[19];
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi) q[18];
rx(-pi/2) q[19];
rz(pi) q[19];
cz q[19],q[56];
rx(pi/2) q[19];
rx(pi/2) q[29];
rz(-pi/2) q[56];
rx(pi/2) q[56];
cz q[19],q[56];
rx(-pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[56];
cz q[19],q[56];
rz(pi/2) q[56];
cz q[57],q[56];
rx(pi/1024) q[56];
cz q[57],q[56];
rx(1.5677284) q[56];
rz(pi/2) q[56];
rx(1.5462526) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
cz q[71],q[64];
rx(pi/128) q[64];
cz q[71],q[64];
rx(-pi/128) q[64];
rz(1.5217089) q[71];
cz q[71],q[70];
rx(pi/64) q[70];
cz q[71],q[70];
rx(-pi/64) q[70];
rz(-pi/2) q[70];
cz q[71],q[64];
rz(-pi/2) q[64];
rx(-pi/2) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[71];
rz(-pi/2) q[71];
cz q[71],q[64];
rx(1.4726216) q[64];
rz(pi/2) q[64];
rx(pi/2) q[64];
cz q[64],q[27];
rx(pi/32) q[27];
cz q[64],q[27];
rx(-pi/32) q[27];
rz(pi/2) q[27];
rx(pi/2) q[27];
rz(-3.3379422) q[64];
rx(-pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(-pi/2) q[71];
rx(pi/2) q[71];
cz q[28],q[71];
rx(-pi/2) q[28];
rz(pi/2) q[28];
rx(pi/2) q[71];
cz q[28],q[71];
cz q[28],q[29];
rx(pi/2) q[28];
rz(-pi/2) q[29];
rx(pi/2) q[29];
cz q[28],q[29];
rx(-1.5585245) q[28];
rz(pi/2) q[28];
rx(pi/2) q[29];
cz q[28],q[29];
rx(-pi/2) q[28];
rz(-3.7490999) q[28];
rz(-3.1293208) q[29];
rx(pi/2) q[29];
rz(-pi/252651348286730) q[29];
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
cz q[18],q[19];
rx(pi/2) q[18];
rz(pi) q[18];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[19],q[18];
rx(pi/2) q[18];
rz(pi/2) q[18];
rx(pi/2) q[19];
cz q[18],q[19];
cz q[19],q[56];
rx(pi/2) q[19];
rx(pi/2) q[29];
rz(pi/2) q[29];
rx(pi/2) q[56];
cz q[19],q[56];
rx(-pi/2) q[19];
rx(pi/2) q[56];
cz q[19],q[56];
rx(-pi/2) q[56];
rz(-pi/2) q[56];
cz q[57],q[56];
rx(pi/512) q[56];
cz q[57],q[56];
rx(1.5646604) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
rx(pi/2) q[57];
rz(pi/2) q[71];
cz q[64],q[71];
rx(pi/16) q[71];
cz q[64],q[71];
rx(-pi/2) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(-pi/2) q[64];
rx(pi/2) q[64];
cz q[27],q[64];
rx(-pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[64];
cz q[27],q[64];
rx(-pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(-pi/2) q[27];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(-3*pi/8) q[26];
cz q[26],q[37];
rz(pi/2) q[27];
rx(pi/8) q[37];
cz q[26],q[37];
cz q[26],q[27];
rx(pi/4) q[27];
cz q[26],q[27];
rx(-pi/4) q[27];
rz(pi/2) q[27];
rx(pi/2) q[27];
rx(3*pi/8) q[37];
rz(pi/2) q[37];
cz q[26],q[37];
rx(pi/2) q[26];
rx(pi/2) q[37];
cz q[26],q[37];
rx(-pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[37];
cz q[26],q[37];
rx(pi/2) q[26];
rz(pi/2) q[26];
rx(-pi/2) q[37];
rz(-pi/2) q[37];
rx(-0.42662749) q[37];
rz(pi/2) q[64];
rx(7*pi/16) q[71];
rz(pi/2) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(-pi/2) q[70];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rz(3*pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rz(2.1864358) q[70];
rx(pi/2) q[70];
rx(-pi/2) q[71];
rz(-pi/2) q[71];
cz q[28],q[71];
rx(pi/128) q[71];
cz q[28],q[71];
rx(-pi/2) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(-pi/2) q[28];
rx(pi/2) q[28];
cz q[27],q[28];
rx(-pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(-pi/64) q[27];
cz q[27],q[64];
rz(pi/2) q[28];
rx(pi/64) q[64];
cz q[27],q[64];
rx(-pi/64) q[64];
rx(-1.59534) q[71];
cz q[70],q[71];
rx(pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-1.5585245) q[70];
rz(pi/2) q[70];
rx(pi/2) q[71];
cz q[70],q[71];
rx(-pi/2) q[70];
rz(1.5830682) q[70];
rx(pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(-pi/2) q[70];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi/2) q[57];
rz(pi/2) q[70];
rz(4.0599339) q[71];
cz q[71],q[64];
rx(pi/128) q[64];
cz q[71],q[64];
rx(-1.59534) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rx(pi/2) q[64];
cz q[27],q[64];
rx(-pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(pi) q[27];
rz(pi) q[64];
cz q[71],q[64];
rx(pi/2) q[64];
rz(pi) q[64];
rx(pi/2) q[71];
rz(pi) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rx(pi/2) q[71];
rz(1.4726216) q[71];
cz q[71],q[64];
cz q[71],q[70];
rx(pi/32) q[70];
cz q[71],q[70];
rx(-pi/32) q[70];
rx(-pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rx(pi/2) q[71];
cz q[64],q[71];
rx(-pi/2) q[64];
rz(pi/2) q[64];
rx(pi/2) q[71];
cz q[64],q[71];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[27],q[64];
rx(pi/2) q[27];
rz(pi) q[27];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[64],q[27];
rx(pi/2) q[27];
rz(-11*pi/16) q[27];
rx(pi/2) q[64];
rz(pi) q[64];
cz q[27],q[64];
cz q[27],q[26];
rx(pi/16) q[26];
cz q[27],q[26];
rx(-pi/16) q[26];
rz(pi/2) q[26];
rx(pi/2) q[26];
cz q[27],q[28];
rx(pi/8) q[28];
cz q[27],q[28];
rx(-pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(-pi/2) q[27];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(5*pi/4) q[26];
cz q[26],q[37];
rx(pi) q[26];
rz(pi/2) q[27];
rx(-pi/8) q[28];
rz(pi/2) q[28];
rx(pi/2) q[28];
rx(pi/4) q[37];
rz(pi) q[37];
cz q[26],q[37];
rz(pi/2) q[26];
rx(-pi/2) q[26];
rx(-0.35877067) q[37];
rx(pi/2) q[64];
rz(pi/2) q[64];
rz(-1.6198837) q[71];
cz q[71],q[70];
rx(pi/64) q[70];
cz q[71],q[70];
rx(1.5217089) q[70];
rz(pi/2) q[70];
rx(pi/2) q[70];
rx(-pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(-pi/2) q[71];
rx(pi/2) q[71];
cz q[28],q[71];
rx(-pi/2) q[28];
rz(pi/2) q[28];
rx(pi/2) q[71];
cz q[28],q[71];
rx(pi/2) q[28];
rz(2.8470683) q[28];
cz q[28],q[27];
rx(pi/32) q[27];
cz q[28],q[27];
rx(-pi/32) q[27];
rz(pi/2) q[71];
cz q[28],q[71];
rx(pi/16) q[71];
cz q[28],q[71];
cz q[28],q[27];
rz(-pi/2) q[27];
rx(-pi/2) q[27];
rx(pi/2) q[28];
rz(pi) q[28];
cz q[27],q[28];
rx(pi/2) q[27];
rz(pi/2) q[27];
rx(pi/2) q[28];
rz(pi/2) q[28];
cz q[28],q[27];
rx(-pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rx(-pi/2) q[26];
rz(pi/2) q[26];
rx(pi/2) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(5*pi/8) q[26];
cz q[26],q[37];
rz(pi/2) q[27];
rx(pi/8) q[37];
cz q[26],q[37];
cz q[26],q[27];
rx(pi/4) q[27];
cz q[26],q[27];
rx(pi/2) q[26];
rz(pi/2) q[26];
rx(pi/4) q[27];
rz(pi/2) q[27];
rx(pi/2) q[27];
rx(3*pi/8) q[37];
rz(pi/2) q[37];
rx(pi/2) q[37];
rx(7*pi/16) q[71];
rz(pi/2) q[71];
rx(pi/2) q[71];
barrier q[48],q[25],q[2],q[66],q[63],q[8],q[72],q[17],q[56],q[35],q[32],q[41],q[50],q[59],q[4],q[26],q[1],q[68],q[65],q[10],q[74],q[37],q[64],q[58],q[34],q[43],q[52],q[61],q[27],q[3],q[67],q[12],q[76],q[21],q[29],q[57],q[36],q[45],q[54],q[51],q[60],q[5],q[71],q[14],q[78],q[11],q[23],q[75],q[20],q[18],q[38],q[47],q[44],q[53],q[62],q[7],q[28],q[16],q[13],q[77],q[22],q[31],q[40],q[19],q[49],q[46],q[55],q[0],q[69],q[9],q[73],q[6],q[70],q[15],q[79],q[24],q[33],q[30],q[42],q[39];
measure q[26] -> meas[0];
measure q[27] -> meas[1];
measure q[37] -> meas[2];
measure q[71] -> meas[3];
measure q[28] -> meas[4];
measure q[70] -> meas[5];
measure q[64] -> meas[6];
measure q[57] -> meas[7];
measure q[56] -> meas[8];
measure q[19] -> meas[9];
measure q[58] -> meas[10];
measure q[18] -> meas[11];
measure q[69] -> meas[12];
measure q[25] -> meas[13];
