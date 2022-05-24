OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[11];
rz(pi/2) q[15];
sx q[15];
rz(pi) q[15];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[11];
cx q[11],q[8];
sx q[15];
rz(7*pi/8) q[15];
cx q[15],q[12];
rz(-pi/4) q[12];
cx q[15],q[12];
rz(3*pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/4) q[13];
cx q[15],q[12];
rz(-pi/8) q[12];
cx q[15],q[12];
rz(pi/8) q[12];
cx q[13],q[12];
rz(-pi/4) q[12];
cx q[13],q[12];
rz(3*pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[15];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/8) q[14];
cx q[15],q[12];
cx q[12],q[15];
rz(pi/16) q[12];
cx q[12],q[13];
rz(-pi/16) q[13];
cx q[12],q[13];
rz(pi/16) q[13];
cx q[14],q[13];
rz(-pi/8) q[13];
cx q[14],q[13];
rz(pi/8) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/32) q[13];
rz(3*pi/8) q[15];
cx q[15],q[12];
rz(-pi/4) q[12];
cx q[15],q[12];
rz(3*pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
rz(pi/32) q[14];
rz(pi/16) q[16];
cx q[16],q[14];
rz(-pi/16) q[14];
cx q[16],q[14];
rz(pi/16) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/4) q[13];
cx q[15],q[12];
rz(-pi/8) q[12];
cx q[15],q[12];
rz(pi/8) q[12];
cx q[13],q[12];
rz(-pi/4) q[12];
cx q[13],q[12];
rz(3*pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(pi/16) q[12];
rz(3*pi/8) q[15];
cx q[8],q[11];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(pi/64) q[11];
cx q[11],q[8];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(-pi/64) q[8];
cx q[11],q[8];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(0.14726216) q[11];
rz(pi/128) q[14];
rz(pi/64) q[8];
cx q[11],q[8];
rz(-pi/32) q[8];
cx q[11],q[8];
rz(pi/32) q[8];
cx q[16],q[19];
cx q[14],q[16];
rz(-pi/128) q[16];
cx q[14],q[16];
rz(pi/128) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[11],q[14];
rz(-pi/64) q[14];
cx q[11],q[14];
rz(pi/64) q[14];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(pi/128) q[14];
rz(pi/256) q[16];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[19],q[22];
cx q[16],q[19];
rz(-pi/256) q[19];
cx q[16],q[19];
rz(pi/256) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
rz(-pi/128) q[16];
cx q[14],q[16];
rz(pi/128) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
rz(-pi/16) q[13];
cx q[12],q[13];
rz(pi/16) q[13];
rz(pi/8) q[14];
cx q[14],q[13];
rz(-pi/8) q[13];
cx q[14],q[13];
rz(pi/8) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/32) q[13];
cx q[15],q[12];
rz(-pi/4) q[12];
cx q[15],q[12];
rz(3*pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/256) q[16];
rz(pi/512) q[19];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(pi/16) q[11];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[11],q[14];
rz(-pi/16) q[14];
cx q[11],q[14];
cx q[11],q[8];
rz(pi/16) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/4) q[13];
rz(0.073631078) q[14];
cx q[15],q[12];
rz(-pi/8) q[12];
cx q[15],q[12];
rz(pi/8) q[12];
cx q[13],q[12];
rz(-pi/4) q[12];
cx q[13],q[12];
rz(3*pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(pi/16) q[12];
rz(1.4726216) q[15];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
rz(-pi/64) q[11];
cx q[14],q[11];
rz(pi/64) q[11];
rz(0.14726216) q[8];
cx q[8],q[11];
rz(-pi/32) q[11];
cx q[8],q[11];
rz(pi/32) q[11];
cx q[22],q[25];
cx q[19],q[22];
rz(-pi/512) q[22];
cx q[19],q[22];
rz(pi/512) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
rz(-pi/256) q[19];
cx q[16],q[19];
rz(pi/256) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
rz(-pi/128) q[16];
cx q[14],q[16];
rz(pi/128) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
rz(-pi/16) q[13];
cx q[12],q[13];
rz(pi/16) q[13];
rz(pi/8) q[14];
cx q[14],q[13];
rz(-pi/8) q[13];
cx q[14],q[13];
rz(pi/8) q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[15],q[12];
rz(-pi/4) q[12];
cx q[15],q[12];
rz(3*pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
rz(0.0046019424) q[22];
cx q[25],q[24];
cx q[22],q[25];
rz(-pi/1024) q[25];
cx q[22],q[25];
rz(pi/1024) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[22],q[25];
rz(-pi/2048) q[25];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/2048) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(0.0092038847) q[25];
cx q[25],q[24];
rz(-pi/512) q[24];
cx q[25],q[24];
rz(pi/512) q[24];
cx q[25],q[22];
rz(-pi/1024) q[22];
cx q[25],q[22];
rz(pi/1024) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(0.018407769) q[22];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[22],q[25];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(-pi/256) q[25];
cx q[22],q[25];
cx q[22],q[19];
rz(-pi/512) q[19];
cx q[22],q[19];
rz(pi/512) q[19];
rz(pi/256) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[8],q[11];
rz(-pi/64) q[11];
cx q[8],q[11];
rz(pi/64) q[11];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(pi/16) q[11];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[11],q[14];
rz(-pi/16) q[14];
cx q[11],q[14];
cx q[11],q[8];
rz(pi/16) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/4) q[13];
cx q[15],q[12];
rz(-pi/8) q[12];
cx q[15],q[12];
rz(pi/8) q[12];
cx q[13],q[12];
rz(-pi/4) q[12];
cx q[13],q[12];
rz(3*pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(0.036815539) q[19];
cx q[19],q[22];
rz(-pi/128) q[22];
cx q[19],q[22];
cx q[19],q[16];
rz(-pi/256) q[16];
cx q[19],q[16];
rz(pi/256) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(0.073631078) q[16];
rz(pi/128) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
rz(-pi/64) q[19];
cx q[16],q[19];
cx q[16],q[14];
rz(-pi/128) q[14];
cx q[16],q[14];
rz(pi/128) q[14];
rz(pi/64) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
rz(0.14726216) q[11];
cx q[11],q[14];
rz(-pi/32) q[14];
cx q[11],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[16];
cx q[15],q[12];
rz(-pi/16) q[12];
cx q[15],q[12];
rz(pi/16) q[12];
cx q[16],q[14];
cx q[14],q[16];
cx q[11],q[14];
rz(-pi/64) q[14];
cx q[11],q[14];
cx q[11],q[8];
rz(pi/64) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/4) q[13];
cx q[15],q[12];
rz(-pi/32) q[12];
cx q[15],q[12];
rz(pi/32) q[12];
rz(3*pi/16) q[16];
cx q[16],q[14];
rz(-pi/8) q[14];
cx q[16],q[14];
rz(pi/8) q[14];
cx q[13],q[14];
rz(-pi/4) q[14];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(3*pi/4) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(pi/8) q[12];
cx q[16],q[14];
rz(-pi/16) q[14];
cx q[16],q[14];
rz(pi/16) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
rz(-pi/8) q[13];
cx q[12],q[13];
cx q[12],q[15];
rz(pi/8) q[13];
cx q[15],q[12];
cx q[12],q[15];
rz(pi/4) q[12];
cx q[12],q[13];
rz(-pi/4) q[13];
cx q[12],q[13];
rz(3*pi/4) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[19],q[22];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[22],q[19];
cx q[19],q[22];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
barrier q[1],q[7],q[4],q[10],q[24],q[19],q[13],q[23],q[15],q[0],q[3],q[6],q[18],q[9],q[11],q[21],q[12],q[14],q[5],q[2],q[8],q[22],q[16],q[17],q[20],q[26],q[25];
measure q[14] -> meas[0];
measure q[15] -> meas[1];
measure q[23] -> meas[2];
measure q[19] -> meas[3];
measure q[13] -> meas[4];
measure q[8] -> meas[5];
measure q[16] -> meas[6];
measure q[22] -> meas[7];
measure q[24] -> meas[8];
measure q[18] -> meas[9];
measure q[11] -> meas[10];
