OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[18];
cx q[3],q[5];
cx q[5],q[3];
cx q[3],q[5];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
rz(pi/2) q[24];
sx q[24];
rz(pi) q[24];
cx q[24],q[25];
sx q[24];
rz(15*pi/16) q[24];
cx q[25],q[22];
cx q[22],q[19];
cx q[19],q[16];
cx q[16],q[14];
cx q[14],q[11];
cx q[24],q[25];
rz(-pi/4) q[25];
cx q[24],q[25];
rz(3*pi/4) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(pi/4) q[22];
cx q[24],q[25];
rz(-pi/8) q[25];
cx q[24],q[25];
rz(pi/8) q[25];
cx q[22],q[25];
rz(-pi/4) q[25];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(pi/8) q[19];
rz(3*pi/4) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[24],q[25];
rz(-pi/16) q[25];
cx q[24],q[25];
rz(pi/16) q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[19],q[22];
rz(-pi/8) q[22];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/16) q[16];
rz(pi/8) q[22];
rz(pi/4) q[25];
cx q[25],q[22];
rz(-pi/4) q[22];
cx q[25],q[22];
rz(3*pi/4) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
rz(3*pi/16) q[24];
rz(pi/32) q[25];
cx q[25],q[22];
rz(-pi/32) q[22];
cx q[25],q[22];
rz(pi/32) q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[16],q[19];
rz(-pi/16) q[19];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(pi/32) q[14];
rz(pi/16) q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(pi/64) q[19];
cx q[19],q[16];
rz(-pi/64) q[16];
cx q[19],q[16];
rz(pi/64) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[11],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[15];
rz(-3*pi/4) q[12];
sx q[12];
cx q[15],q[18];
rz(-pi/4) q[15];
sx q[15];
rz(-pi/2) q[15];
rz(pi/32) q[16];
cx q[16],q[19];
cx q[18],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
rz(0.036815539) q[14];
cx q[14],q[11];
rz(-pi/128) q[11];
cx q[14],q[11];
rz(pi/128) q[11];
rz(-pi/256) q[16];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(0.79153409) q[13];
sx q[13];
rz(-pi/2) q[13];
cx q[13],q[12];
rz(-pi/2) q[12];
sx q[13];
rz(-pi) q[13];
cx q[13],q[12];
rz(1.5646604) q[12];
sx q[13];
cx q[13],q[12];
rz(1.5677284) q[12];
sx q[12];
rz(-pi) q[12];
rz(-0.79153409) q[13];
sx q[13];
rz(-pi) q[13];
rz(0.085902924) q[14];
cx q[14],q[11];
rz(-pi/64) q[11];
cx q[14],q[11];
rz(pi/64) q[11];
cx q[15],q[12];
rz(-pi/2) q[12];
sx q[15];
rz(-pi) q[15];
cx q[15],q[12];
rz(1.5677284) q[12];
sx q[15];
cx q[15],q[12];
rz(1.5677284) q[12];
sx q[12];
rz(-pi) q[12];
rz(-2.3577285) q[15];
sx q[15];
rz(pi/256) q[16];
cx q[14],q[16];
rz(-pi/128) q[16];
cx q[14],q[16];
sx q[14];
rz(-pi/2) q[14];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[14];
rz(-pi) q[14];
cx q[14],q[13];
rz(1.5585245) q[13];
sx q[14];
cx q[14],q[13];
rz(0.79153409) q[13];
sx q[13];
rz(-pi/2) q[13];
cx q[13],q[12];
rz(-pi/2) q[12];
sx q[13];
rz(-pi) q[13];
cx q[13],q[12];
rz(1.5646604) q[12];
sx q[13];
cx q[13],q[12];
rz(1.5677284) q[12];
sx q[12];
rz(-0.79153409) q[13];
sx q[13];
rz(-pi) q[13];
rz(1.5830682) q[14];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(pi/128) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[18],q[15];
rz(pi/2) q[15];
sx q[18];
rz(-pi) q[18];
cx q[18],q[15];
rz(1.5692623) q[15];
sx q[18];
cx q[18],q[15];
rz(-0.78693214) q[15];
sx q[15];
rz(-pi/2) q[15];
cx q[15],q[12];
rz(pi/2) q[12];
sx q[15];
rz(-pi) q[15];
cx q[15],q[12];
rz(1.5677284) q[12];
sx q[15];
cx q[15],q[12];
rz(0.7823302) q[12];
sx q[12];
x q[15];
rz(-pi/4) q[15];
x q[18];
rz(-1.5700293) q[18];
cx q[21],q[23];
cx q[18],q[21];
rz(-pi/4096) q[21];
cx q[18],q[21];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
rz(pi/2048) q[18];
rz(pi/4096) q[21];
cx q[18],q[21];
rz(-pi/2048) q[21];
cx q[18],q[21];
rz(pi/2048) q[21];
cx q[21],q[23];
rz(pi/4) q[22];
cx q[23],q[21];
cx q[21],q[23];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[24],q[25];
rz(-pi/8) q[25];
cx q[24],q[25];
rz(pi/8) q[25];
cx q[22],q[25];
rz(-pi/4) q[25];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(pi/8) q[19];
rz(3*pi/4) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[24],q[25];
rz(-pi/16) q[25];
cx q[24],q[25];
rz(pi/16) q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[19],q[22];
rz(-pi/8) q[22];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/16) q[16];
rz(pi/8) q[22];
rz(pi/4) q[25];
cx q[25],q[22];
rz(-pi/4) q[22];
cx q[25],q[22];
rz(3*pi/4) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[22],q[25];
rz(pi/8) q[24];
cx q[25],q[22];
cx q[22],q[25];
rz(pi/32) q[22];
cx q[22],q[19];
rz(-pi/32) q[19];
cx q[22],q[19];
rz(pi/32) q[19];
cx q[16],q[19];
rz(-pi/16) q[19];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(pi/32) q[14];
rz(pi/16) q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(pi/64) q[19];
cx q[19],q[16];
rz(-pi/64) q[16];
cx q[19],q[16];
rz(pi/64) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(0.085902924) q[11];
rz(pi/32) q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/128) q[16];
cx q[16],q[14];
rz(-pi/128) q[14];
cx q[16],q[14];
rz(pi/128) q[14];
cx q[11],q[14];
rz(-pi/64) q[14];
cx q[11],q[14];
rz(pi/64) q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(pi/256) q[14];
sx q[14];
rz(-pi/2) q[14];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[14];
rz(-pi) q[14];
cx q[14],q[13];
rz(1.5585245) q[13];
sx q[14];
cx q[14],q[13];
rz(pi/512) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[12];
rz(-pi/2) q[12];
sx q[13];
rz(-pi) q[13];
cx q[13],q[12];
rz(1.5646604) q[12];
sx q[13];
cx q[13],q[12];
cx q[12],q[15];
rz(pi/512) q[13];
rz(1.5830682) q[14];
cx q[11],q[14];
rz(-pi/128) q[14];
cx q[11],q[14];
rz(pi/128) q[14];
cx q[15],q[12];
cx q[12],q[15];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[7];
rz(-3*pi/4) q[10];
sx q[10];
rz(0.78597341) q[12];
cx q[12],q[15];
rz(-pi/8192) q[15];
cx q[12],q[15];
sx q[12];
rz(-pi/2) q[12];
cx q[12],q[10];
rz(-pi/2) q[10];
sx q[12];
rz(-pi) q[12];
cx q[12],q[10];
rz(1.5706046) q[10];
sx q[12];
cx q[12],q[10];
rz(2.3560986) q[10];
sx q[10];
rz(-pi/2) q[10];
rz(0.78520642) q[12];
sx q[12];
rz(-pi) q[12];
rz(-2.355811) q[15];
sx q[15];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(0.78616515) q[18];
sx q[18];
rz(-pi/2) q[18];
cx q[18],q[15];
rz(-pi/2) q[15];
sx q[18];
rz(-pi) q[18];
cx q[18],q[15];
rz(1.5700293) q[15];
sx q[18];
cx q[18],q[15];
rz(2.355811) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[12];
rz(-pi/2) q[12];
sx q[15];
rz(-pi) q[15];
cx q[15],q[12];
rz(1.5704128) q[12];
sx q[15];
cx q[15],q[12];
rz(-2.3563862) q[12];
sx q[12];
rz(pi/2) q[12];
x q[15];
rz(-3.1412092) q[15];
rz(2.3554275) q[18];
sx q[18];
rz(-1.5661944) q[21];
cx q[21],q[23];
cx q[22],q[25];
rz(-pi/1024) q[23];
cx q[21],q[23];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[18];
rz(pi/2) q[18];
sx q[21];
rz(-pi) q[21];
cx q[21],q[18];
rz(1.5692623) q[18];
sx q[21];
cx q[21],q[18];
x q[18];
rz(-1.5700293) q[18];
cx q[18],q[15];
rz(-pi/4096) q[15];
cx q[18],q[15];
rz(pi/4096) q[15];
x q[21];
rz(-1.5692623) q[21];
rz(pi/1024) q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
rz(pi/4) q[21];
sx q[21];
rz(-pi/2) q[21];
cx q[25],q[22];
cx q[22],q[25];
rz(pi/4) q[22];
cx q[24],q[25];
rz(-pi/8) q[25];
cx q[24],q[25];
rz(pi/8) q[25];
cx q[22],q[25];
rz(-pi/4) q[25];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(pi/8) q[19];
rz(3*pi/4) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
rz(7*pi/16) q[24];
rz(pi/16) q[25];
cx q[25],q[22];
rz(-pi/16) q[22];
cx q[25],q[22];
rz(pi/16) q[22];
cx q[19],q[22];
rz(-pi/8) q[22];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/16) q[16];
rz(pi/8) q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(pi/32) q[22];
cx q[22],q[19];
rz(-pi/32) q[19];
cx q[22],q[19];
rz(pi/32) q[19];
cx q[16],q[19];
rz(-pi/16) q[19];
cx q[16],q[19];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(pi/32) q[14];
rz(pi/16) q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(pi/64) q[19];
cx q[19],q[16];
rz(-pi/64) q[16];
cx q[19],q[16];
rz(pi/64) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[11],q[14];
rz(pi/64) q[13];
rz(-pi/256) q[14];
cx q[11],q[14];
rz(pi/256) q[14];
rz(pi/32) q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/128) q[16];
cx q[16],q[14];
rz(-pi/128) q[14];
cx q[16],q[14];
rz(pi/128) q[14];
cx q[13],q[14];
rz(-pi/64) q[14];
cx q[13],q[14];
rz(pi/64) q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(pi/128) q[11];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(pi/256) q[14];
sx q[14];
rz(-pi/2) q[14];
cx q[24],q[25];
rz(-pi/4) q[25];
cx q[24],q[25];
rz(3*pi/4) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(pi/4) q[22];
cx q[24],q[25];
rz(-pi/8) q[25];
cx q[24],q[25];
rz(pi/8) q[25];
cx q[22],q[25];
rz(-pi/4) q[25];
cx q[22],q[25];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(3*pi/4) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(pi/4) q[19];
rz(-pi/8) q[22];
sx q[22];
rz(-pi) q[22];
cx q[24],q[25];
rz(-pi/16) q[25];
cx q[24],q[25];
rz(5*pi/16) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[22];
rz(-pi/2) q[22];
sx q[25];
rz(-pi) q[25];
cx q[25],q[22];
rz(3*pi/8) q[22];
sx q[25];
cx q[25],q[22];
rz(-pi/8) q[22];
cx q[19],q[22];
rz(-pi/4) q[22];
cx q[19],q[22];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(3*pi/4) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-pi/4) q[22];
sx q[22];
rz(-pi) q[22];
rz(-pi/4) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(pi/16) q[24];
rz(0.88357293) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[22];
rz(-pi/2) q[22];
sx q[25];
rz(-pi) q[25];
cx q[25],q[22];
rz(1.4726216) q[22];
sx q[25];
cx q[25],q[22];
rz(-pi/4) q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(pi/64) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[19];
rz(-pi/8) q[22];
sx q[22];
rz(-pi) q[22];
rz(-0.68722339) q[25];
cx q[24],q[25];
rz(-pi/16) q[25];
cx q[24],q[25];
rz(5*pi/16) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[22];
rz(-pi/2) q[22];
sx q[25];
rz(-pi) q[25];
cx q[25],q[22];
rz(3*pi/8) q[22];
sx q[25];
cx q[25],q[22];
rz(-pi/8) q[22];
cx q[19],q[22];
rz(-pi/4) q[22];
cx q[19],q[22];
rz(3*pi/4) q[22];
sx q[22];
rz(pi/2) q[22];
rz(-pi/4) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
rz(pi/32) q[25];
cx q[7],q[4];
cx q[4],q[1];
cx q[1],q[2];
sx q[1];
rz(-pi) q[1];
sx q[4];
rz(-pi) q[4];
sx q[7];
rz(-pi) q[7];
cx q[10],q[7];
sx q[10];
rz(-pi) q[10];
rz(pi/2) q[7];
cx q[10],q[7];
sx q[10];
rz(1.5707005) q[7];
cx q[10],q[7];
rz(0.78530229) q[10];
sx q[10];
rz(-pi) q[10];
cx q[12],q[10];
rz(-pi/2) q[10];
sx q[12];
rz(-pi) q[12];
cx q[12],q[10];
rz(1.5706046) q[10];
sx q[12];
cx q[12],q[10];
rz(2.3560986) q[10];
sx q[10];
rz(-pi/2) q[10];
x q[12];
rz(0.78558991) q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(0.78597341) q[12];
cx q[12],q[13];
rz(-pi/8192) q[13];
cx q[12],q[13];
sx q[12];
rz(-pi/2) q[12];
rz(pi/8192) q[13];
rz(-2.3500586) q[18];
sx q[18];
cx q[21],q[18];
rz(-pi/2) q[18];
sx q[21];
rz(-pi) q[21];
cx q[21],q[18];
rz(1.5646604) q[18];
sx q[21];
cx q[21],q[18];
x q[18];
rz(2.3623304) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(-3*pi/4) q[18];
sx q[18];
x q[21];
rz(1.5753983) q[21];
cx q[21],q[23];
rz(-pi/1024) q[23];
cx q[21],q[23];
sx q[21];
rz(-pi/2) q[21];
cx q[21],q[18];
rz(-pi/2) q[18];
sx q[21];
rz(-pi) q[21];
cx q[21],q[18];
rz(1.5692623) q[18];
sx q[21];
cx q[21],q[18];
x q[18];
rz(3*pi/4) q[18];
x q[21];
rz(0.78693214) q[21];
rz(pi/1024) q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(pi/4) q[21];
sx q[21];
rz(-pi/2) q[21];
rz(-1.5708443) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[4];
rz(-pi/2) q[4];
sx q[7];
rz(-pi) q[7];
cx q[7],q[4];
rz(pi/2) q[4];
sx q[7];
cx q[7],q[4];
rz(-pi/131072) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[1];
rz(-pi/2) q[1];
sx q[4];
rz(-pi) q[4];
cx q[4],q[1];
rz(pi/2) q[1];
sx q[4];
cx q[4],q[1];
x q[1];
rz(-3.1415807) q[1];
cx q[1],q[2];
rz(-pi/262144) q[2];
cx q[1],q[2];
rz(pi/262144) q[2];
sx q[2];
rz(-pi/2) q[2];
rz(-1.5707724) q[4];
sx q[4];
rz(-pi) q[4];
rz(-1.5707484) q[7];
sx q[7];
rz(-pi) q[7];
cx q[10],q[7];
sx q[10];
rz(-pi) q[10];
rz(pi/2) q[7];
cx q[10],q[7];
sx q[10];
rz(1.5707005) q[7];
cx q[10],q[7];
rz(0.78530229) q[10];
sx q[10];
rz(-pi) q[10];
cx q[12],q[10];
rz(-pi/2) q[10];
sx q[12];
rz(-pi) q[12];
cx q[12],q[10];
rz(1.5706046) q[10];
sx q[12];
cx q[12],q[10];
rz(2.3560986) q[10];
sx q[10];
rz(-pi/2) q[10];
x q[12];
rz(0.78558991) q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi) q[13];
sx q[13];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[14];
rz(-pi) q[14];
cx q[14],q[13];
rz(1.5585245) q[13];
sx q[14];
cx q[14],q[13];
rz(1.5830682) q[14];
cx q[11],q[14];
rz(-pi/128) q[14];
cx q[11],q[14];
rz(-1.5462526) q[14];
sx q[14];
rz(-pi) q[14];
cx q[15],q[18];
cx q[16],q[14];
rz(-pi/2) q[14];
sx q[16];
rz(-pi) q[16];
cx q[16],q[14];
rz(1.5217089) q[14];
sx q[16];
cx q[16],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(pi/128) q[11];
rz(pi/64) q[16];
cx q[18],q[15];
rz(0.0011504856) q[15];
cx q[15],q[12];
rz(-pi/4096) q[12];
cx q[15],q[12];
rz(pi/4096) q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/8192) q[18];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(0.78558991) q[12];
sx q[12];
rz(-pi/2) q[12];
rz(pi/8192) q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
rz(-2.3500586) q[18];
sx q[18];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(pi/8) q[16];
cx q[21],q[18];
rz(-pi/2) q[18];
sx q[21];
rz(-pi) q[21];
cx q[21],q[18];
rz(1.5646604) q[18];
sx q[21];
cx q[21],q[18];
x q[18];
rz(2.3623304) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
x q[21];
rz(0.78846612) q[21];
cx q[21],q[23];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(-pi/1024) q[23];
cx q[21],q[23];
cx q[18],q[21];
cx q[21],q[18];
rz(0.0023009712) q[18];
rz(-pi/4096) q[21];
rz(pi/1024) q[23];
cx q[25],q[22];
rz(-pi/32) q[22];
cx q[25],q[22];
rz(pi/32) q[22];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(pi/16) q[25];
cx q[25],q[22];
rz(-pi/16) q[22];
cx q[25],q[22];
rz(pi/16) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
rz(-pi/8) q[19];
cx q[16],q[19];
rz(pi/8) q[19];
rz(pi/4) q[22];
cx q[22],q[19];
rz(-pi/4) q[19];
cx q[22],q[19];
rz(3*pi/4) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
rz(pi/64) q[22];
rz(-1.5708443) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[4];
rz(-pi/2) q[4];
sx q[7];
rz(-pi) q[7];
cx q[7],q[4];
rz(pi/2) q[4];
sx q[7];
cx q[7],q[4];
rz(-pi) q[4];
x q[4];
cx q[1],q[4];
cx q[4],q[1];
cx q[1],q[4];
rz(pi/131072) q[1];
sx q[1];
rz(-pi) q[1];
cx q[2],q[1];
rz(-pi/2) q[1];
sx q[2];
rz(-pi) q[2];
cx q[2],q[1];
rz(pi/2) q[1];
sx q[2];
cx q[2],q[1];
x q[1];
rz(-3.1415687) q[1];
cx q[1],q[4];
rz(-pi/2) q[2];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[4],q[1];
cx q[1],q[4];
sx q[4];
rz(-pi) q[4];
cx q[5],q[3];
cx q[3],q[5];
cx q[5],q[3];
rz(-1.5707484) q[7];
sx q[7];
rz(-pi) q[7];
cx q[10],q[7];
sx q[10];
rz(-pi) q[10];
rz(pi/2) q[7];
cx q[10],q[7];
sx q[10];
rz(1.5707005) q[7];
cx q[10],q[7];
rz(0.78530229) q[10];
sx q[10];
rz(-pi) q[10];
cx q[12],q[10];
rz(-pi/2) q[10];
sx q[12];
rz(-pi) q[12];
cx q[12],q[10];
rz(1.5706046) q[10];
sx q[12];
cx q[12],q[10];
rz(2.3560986) q[10];
sx q[10];
rz(pi/2) q[10];
x q[12];
rz(0.78558991) q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[12],q[15];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/256) q[13];
sx q[13];
rz(-pi/2) q[13];
cx q[15],q[12];
cx q[12],q[15];
rz(-pi) q[12];
sx q[12];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[13];
rz(-pi) q[13];
cx q[13],q[12];
rz(1.5585245) q[12];
sx q[13];
cx q[13],q[12];
rz(1.5830682) q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[11],q[14];
rz(-pi/128) q[14];
cx q[11],q[14];
rz(pi/128) q[14];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[18],q[15];
rz(-pi/2048) q[15];
cx q[18],q[15];
rz(pi/2048) q[15];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(0.0005752428) q[12];
cx q[12],q[13];
rz(-pi/8192) q[13];
cx q[12],q[13];
rz(pi/8192) q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-2.3500586) q[15];
sx q[15];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(pi/4096) q[21];
cx q[21],q[23];
cx q[22],q[19];
rz(-pi/64) q[19];
cx q[22],q[19];
rz(pi/64) q[19];
cx q[22],q[25];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(pi/4) q[18];
sx q[18];
rz(-pi/2) q[18];
cx q[18],q[15];
rz(-pi/2) q[15];
sx q[18];
rz(-pi) q[18];
cx q[18],q[15];
rz(1.5646604) q[15];
sx q[18];
cx q[18],q[15];
x q[15];
rz(2.3623304) q[15];
rz(3.1385247) q[18];
sx q[18];
rz(-pi) q[18];
rz(-pi/4) q[21];
sx q[21];
rz(-pi/2) q[21];
cx q[21],q[18];
rz(-pi/2) q[18];
sx q[21];
rz(-pi) q[21];
cx q[21],q[18];
rz(1.5677284) q[18];
sx q[21];
cx q[21],q[18];
x q[18];
rz(-2.3531265) q[18];
x q[21];
rz(-0.78386418) q[21];
cx q[21],q[23];
rz(-pi/2048) q[23];
cx q[21],q[23];
rz(pi/2048) q[23];
cx q[25],q[22];
cx q[22],q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(pi/128) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
rz(-1.4235342) q[22];
cx q[22],q[19];
rz(-pi/32) q[19];
cx q[22],q[19];
rz(pi/32) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
sx q[22];
rz(-pi) q[22];
rz(-1.5708443) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[4];
rz(-pi/2) q[4];
sx q[7];
rz(-pi) q[7];
cx q[7],q[4];
rz(pi/2) q[4];
sx q[7];
cx q[7],q[4];
rz(-pi) q[4];
x q[4];
rz(4.79369e-05) q[7];
sx q[7];
cx q[10],q[7];
sx q[10];
rz(-pi) q[10];
rz(-pi/2) q[7];
cx q[10],q[7];
sx q[10];
rz(1.5707005) q[7];
cx q[10],q[7];
x q[10];
rz(-3.1414968) q[10];
cx q[12],q[10];
rz(-pi/16384) q[10];
cx q[12],q[10];
rz(pi/16384) q[10];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(0.018407769) q[12];
cx q[12],q[15];
rz(-pi/256) q[15];
cx q[12],q[15];
rz(pi/256) q[15];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
rz(-pi/512) q[15];
cx q[12],q[15];
rz(pi/512) q[15];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(0.0011504856) q[13];
cx q[13],q[14];
rz(-pi/4096) q[14];
cx q[13],q[14];
cx q[13],q[12];
rz(-pi/8192) q[12];
cx q[13],q[12];
rz(pi/8192) q[12];
rz(pi/4096) q[14];
rz(pi/1024) q[15];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[15],q[18];
rz(-pi/1024) q[18];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/2048) q[13];
cx q[13],q[14];
rz(-pi/2048) q[14];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/4096) q[12];
cx q[12],q[15];
rz(pi/2048) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(0.29452431) q[14];
cx q[14],q[16];
rz(-pi/4096) q[15];
cx q[12],q[15];
rz(pi/4096) q[15];
rz(-pi/16) q[16];
cx q[14],q[16];
rz(pi/16) q[16];
rz(-1.5677284) q[18];
sx q[18];
rz(-pi) q[18];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(-pi) q[21];
sx q[21];
rz(-pi/2) q[23];
sx q[23];
rz(-pi) q[23];
cx q[24],q[23];
rz(-pi/2) q[23];
sx q[24];
rz(-pi) q[24];
cx q[24],q[23];
rz(1.5462526) q[23];
sx q[24];
cx q[24],q[23];
rz(pi/256) q[23];
sx q[23];
rz(-pi/2) q[23];
cx q[23],q[21];
rz(pi/2) q[21];
sx q[23];
rz(-pi) q[23];
cx q[23],q[21];
rz(1.5585245) q[21];
sx q[23];
cx q[23],q[21];
rz(pi/512) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[18];
rz(-pi/2) q[18];
sx q[21];
rz(-pi) q[21];
cx q[21],q[18];
rz(1.5646604) q[18];
sx q[21];
cx q[21],q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(pi/1024) q[12];
cx q[12],q[13];
rz(-pi/1024) q[13];
cx q[12],q[13];
rz(pi/1024) q[13];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(pi/2048) q[15];
cx q[15],q[18];
rz(-pi/2048) q[18];
cx q[15],q[18];
rz(pi/2048) q[18];
rz(-3.1354567) q[21];
sx q[21];
rz(pi/256) q[23];
sx q[23];
rz(-pi) q[23];
rz(pi/128) q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[22];
rz(-pi/2) q[22];
sx q[25];
rz(-pi) q[25];
cx q[25],q[22];
rz(1.5217089) q[22];
sx q[25];
cx q[25],q[22];
rz(pi/64) q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(pi/32) q[16];
rz(pi/4) q[19];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
rz(pi/128) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[24],q[23];
rz(-pi/2) q[23];
sx q[24];
rz(-pi) q[24];
cx q[24],q[23];
rz(1.5462526) q[23];
sx q[24];
cx q[24],q[23];
rz(pi/256) q[23];
sx q[23];
rz(-pi/2) q[23];
cx q[23],q[21];
rz(pi/2) q[21];
sx q[23];
rz(-pi) q[23];
cx q[23],q[21];
rz(1.5585245) q[21];
sx q[23];
cx q[23],q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(-0.77619428) q[12];
cx q[12],q[13];
rz(-pi/512) q[13];
cx q[12],q[13];
sx q[12];
rz(-pi) q[12];
rz(pi/512) q[13];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
rz(-pi/4) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[12];
rz(pi/2) q[12];
sx q[15];
rz(-pi) q[15];
cx q[15],q[12];
rz(1.5677284) q[12];
sx q[15];
cx q[15],q[12];
x q[12];
rz(-2.3531265) q[12];
x q[15];
rz(-pi/4) q[15];
rz(1.5830682) q[23];
rz(pi/128) q[24];
rz(3*pi/16) q[25];
cx q[25],q[22];
rz(-pi/8) q[22];
cx q[25],q[22];
rz(pi/8) q[22];
cx q[19],q[22];
rz(-pi/4) q[22];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/8) q[16];
rz(3*pi/4) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[25],q[22];
rz(-pi/16) q[22];
cx q[25],q[22];
rz(pi/16) q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[16],q[19];
rz(-pi/8) q[19];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(pi/16) q[14];
rz(pi/64) q[16];
rz(pi/8) q[19];
rz(pi/4) q[22];
cx q[22],q[19];
rz(-pi/4) q[19];
cx q[22],q[19];
rz(3*pi/4) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[16],q[19];
rz(-pi/64) q[19];
cx q[16],q[19];
rz(pi/64) q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(pi/32) q[25];
cx q[25],q[22];
rz(-pi/32) q[22];
cx q[25],q[22];
rz(pi/32) q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[14],q[16];
rz(-pi/16) q[16];
cx q[14],q[16];
rz(pi/16) q[16];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/128) q[22];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[22],q[25];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(pi/64) q[23];
rz(-pi/128) q[25];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(0.018407769) q[14];
cx q[14],q[13];
rz(-pi/256) q[13];
cx q[14],q[13];
rz(pi/256) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[15];
cx q[14],q[13];
rz(-pi/512) q[13];
cx q[14],q[13];
rz(pi/512) q[13];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-pi/2) q[15];
sx q[15];
rz(-pi) q[15];
rz(pi/4) q[19];
rz(pi/128) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[23],q[24];
rz(-pi/64) q[24];
cx q[23],q[24];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(pi/128) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[15];
rz(-pi/2) q[15];
sx q[18];
rz(-pi) q[18];
cx q[18],q[15];
rz(1.5462526) q[15];
sx q[18];
cx q[18],q[15];
rz(pi/256) q[15];
cx q[15],q[12];
rz(-pi/256) q[12];
cx q[15],q[12];
rz(pi/256) q[12];
rz(pi/128) q[18];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(-pi/2) q[23];
sx q[23];
rz(-pi) q[23];
rz(pi/64) q[24];
rz(pi/8) q[25];
cx q[25],q[22];
rz(-pi/8) q[22];
cx q[25],q[22];
rz(pi/8) q[22];
cx q[19],q[22];
rz(-pi/4) q[22];
cx q[19],q[22];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(pi/8) q[16];
rz(3*pi/4) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(pi/32) q[22];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[22],q[25];
rz(pi/16) q[24];
rz(-pi/32) q[25];
cx q[22],q[25];
rz(pi/32) q[25];
cx q[24],q[25];
rz(-pi/16) q[25];
cx q[24],q[25];
rz(pi/16) q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[16],q[19];
rz(-pi/8) q[19];
cx q[16],q[19];
rz(pi/8) q[19];
rz(pi/4) q[22];
cx q[22],q[19];
rz(-pi/4) q[19];
cx q[22],q[19];
rz(3*pi/4) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(pi/4) q[16];
rz(pi/16) q[19];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
rz(pi/64) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[24],q[23];
rz(-pi/2) q[23];
sx q[24];
rz(-pi) q[24];
cx q[24],q[23];
rz(1.5217089) q[23];
sx q[24];
cx q[24],q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(pi/128) q[15];
cx q[15],q[12];
rz(-pi/128) q[12];
cx q[15],q[12];
rz(pi/128) q[12];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(-0.73631078) q[24];
sx q[24];
rz(-pi) q[24];
rz(0.88357293) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[24];
rz(-pi/2) q[24];
sx q[25];
rz(-pi) q[25];
cx q[25],q[24];
rz(1.4726216) q[24];
sx q[25];
cx q[25],q[24];
rz(-pi/4) q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(pi/64) q[15];
cx q[15],q[12];
rz(-pi/64) q[12];
cx q[15],q[12];
rz(pi/64) q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[12],q[15];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
rz(-0.68722339) q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[19],q[22];
rz(-pi/16) q[22];
cx q[19],q[22];
rz(pi/16) q[22];
rz(3*pi/16) q[25];
cx q[25],q[22];
rz(-pi/8) q[22];
cx q[25],q[22];
rz(pi/8) q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[16],q[19];
rz(-pi/4) q[19];
cx q[16],q[19];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(pi/8) q[14];
rz(3*pi/4) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(-pi/4) q[19];
sx q[19];
rz(-pi) q[19];
rz(0.88357293) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[19];
rz(-pi/2) q[19];
sx q[22];
rz(-pi) q[22];
cx q[22],q[19];
rz(1.4726216) q[19];
sx q[22];
cx q[22],q[19];
rz(-pi/4) q[19];
rz(-0.68722339) q[22];
cx q[25],q[22];
rz(-pi/16) q[22];
cx q[25],q[22];
rz(pi/16) q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
rz(-pi/8) q[16];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/8) q[16];
rz(pi/4) q[19];
cx q[19],q[16];
rz(-pi/4) q[16];
cx q[19],q[16];
rz(3*pi/4) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
x q[7];
rz(-pi/2) q[7];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[7];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[1];
cx q[1],q[4];
cx q[4],q[1];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[4];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[5];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
barrier q[14],q[18],q[4],q[22],q[25],q[15],q[19],q[1],q[11],q[0],q[3],q[6],q[13],q[9],q[7],q[23],q[21],q[10],q[12],q[2],q[5],q[8],q[24],q[17],q[20],q[26],q[16];
measure q[12] -> meas[0];
measure q[14] -> meas[1];
measure q[4] -> meas[2];
measure q[18] -> meas[3];
measure q[22] -> meas[4];
measure q[16] -> meas[5];
measure q[23] -> meas[6];
measure q[21] -> meas[7];
measure q[7] -> meas[8];
measure q[13] -> meas[9];
measure q[25] -> meas[10];
measure q[24] -> meas[11];
measure q[8] -> meas[12];
measure q[19] -> meas[13];
measure q[15] -> meas[14];
measure q[1] -> meas[15];
measure q[11] -> meas[16];
measure q[10] -> meas[17];
