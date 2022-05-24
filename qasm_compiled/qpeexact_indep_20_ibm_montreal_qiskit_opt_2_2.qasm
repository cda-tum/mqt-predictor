OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[19];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[8],q[9];
cx q[9],q[8];
cx q[8],q[9];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
x q[15];
rz(-1.347668) q[15];
cx q[15],q[12];
rz(1.3464816) q[12];
cx q[15],q[12];
rz(-1.3464816) q[12];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
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
rz(-0.44862946) q[18];
cx q[15],q[18];
rz(0.44862946) q[18];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
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
rz(-0.8972589) q[18];
cx q[15],q[18];
rz(0.8972589) q[18];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[15],q[18];
rz(1.3470748) q[18];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(0.44744302) q[12];
cx q[12],q[13];
rz(-0.44744302) q[13];
cx q[12],q[13];
rz(0.44744302) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-0.018982985) q[13];
cx q[13],q[14];
rz(-0.89488605) q[14];
cx q[13],q[14];
cx q[13],q[12];
rz(1.35182055) q[12];
cx q[13],q[12];
rz(-1.35182055) q[12];
rz(0.89488605) q[14];
rz(-1.3470748) q[18];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[13],q[12];
rz(-0.437951515) q[12];
cx q[13],q[12];
rz(0.437951515) q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(-0.51388355) q[16];
cx q[16],q[19];
rz(-0.87590305) q[19];
cx q[16],q[19];
rz(0.87590305) q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
rz(1.3897866) q[19];
cx q[16],q[19];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(1.0860584) q[14];
cx q[14],q[11];
rz(-0.362019465) q[11];
cx q[14],q[11];
rz(0.362019465) q[11];
rz(-1.3897866) q[19];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[6],q[7];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
rz(-0.72403895) q[13];
cx q[14],q[13];
rz(0.72403895) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[10],q[7];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(1.44807785) q[16];
cx q[16],q[19];
rz(-1.44807785) q[19];
cx q[16],q[19];
rz(1.44807785) q[19];
cx q[7],q[10];
cx q[10],q[7];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(-0.245436925) q[11];
cx q[11],q[8];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[6];
rz(0.245436925) q[8];
cx q[11],q[8];
rz(-0.245436925) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(-0.490873852123405) q[8];
cx q[8],q[5];
rz(0.490873852123405) q[5];
cx q[8],q[5];
cx q[11],q[8];
rz(-0.490873852123405) q[5];
cx q[3],q[5];
cx q[5],q[3];
cx q[3],q[5];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[4],q[1];
cx q[1],q[4];
cx q[4],q[1];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[4];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[7];
rz(-0.68722339) q[10];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[6];
rz(-pi/4096) q[6];
rz(-pi/256) q[7];
cx q[8],q[11];
cx q[11],q[8];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[19];
rz(-3*pi/16) q[22];
cx q[22],q[25];
rz(5*pi/16) q[25];
cx q[22],q[25];
cx q[22],q[19];
rz(-3*pi/8) q[19];
cx q[22],q[19];
rz(3*pi/8) q[19];
rz(-5*pi/16) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[22],q[25];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(pi/4) q[25];
cx q[22],q[25];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(pi/2) q[11];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[20];
rz(-pi/4) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[22],q[25];
cx q[24],q[23];
cx q[23],q[24];
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
rz(pi/4) q[15];
cx q[25],q[22];
cx q[22],q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(-0.0014860439) q[24];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[11],q[8];
rz(-pi/2) q[8];
cx q[11],q[8];
rz(pi/2) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[14],q[16];
cx q[15],q[12];
rz(pi/4) q[12];
cx q[15],q[12];
rz(-pi/4) q[12];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[18];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-0.073631078) q[13];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(-pi/512) q[14];
cx q[16],q[19];
cx q[18],q[15];
rz(pi/8) q[15];
cx q[15],q[12];
rz(pi/8) q[12];
cx q[15],q[12];
rz(-pi/8) q[12];
rz(pi/4) q[18];
cx q[15],q[18];
sx q[15];
rz(pi/2) q[15];
rz(-pi/4) q[18];
cx q[19],q[16];
cx q[16],q[19];
rz(-0.0092038847) q[16];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[20];
rz(-pi/8192) q[19];
rz(-0.0013422332) q[20];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
rz(-pi/16) q[15];
cx q[15],q[12];
rz(pi/16) q[12];
cx q[15],q[12];
rz(-pi/16) q[12];
cx q[10],q[12];
rz(pi/32) q[12];
cx q[10],q[12];
rz(-pi/32) q[12];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(pi/8) q[18];
cx q[18],q[21];
rz(pi/8) q[21];
cx q[18],q[21];
cx q[18],q[15];
rz(pi/4) q[15];
cx q[18],q[15];
rz(-pi/4) q[15];
sx q[18];
rz(pi/2) q[18];
rz(-pi/8) q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[10],q[12];
rz(pi/16) q[12];
cx q[10],q[12];
rz(-pi/16) q[12];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[10],q[12];
rz(pi/8) q[12];
cx q[10],q[12];
rz(-pi/8) q[12];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
rz(-0.3436117) q[21];
cx q[21],q[18];
rz(pi/64) q[18];
cx q[21],q[18];
rz(-pi/64) q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[13],q[12];
rz(pi/128) q[12];
cx q[13],q[12];
rz(-pi/128) q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[21],q[18];
rz(pi/32) q[18];
cx q[21],q[18];
rz(-pi/32) q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
rz(pi/64) q[12];
cx q[13],q[12];
rz(-pi/64) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/32) q[12];
cx q[21],q[18];
rz(pi/16) q[18];
cx q[21],q[18];
rz(-pi/16) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
rz(pi/32) q[15];
cx q[12],q[15];
rz(-pi/32) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(-pi/16) q[15];
rz(pi/4) q[18];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
rz(pi/4) q[21];
cx q[18],q[21];
sx q[18];
rz(pi/2) q[18];
rz(-pi/4) q[21];
rz(pi/8) q[23];
cx q[23],q[21];
rz(pi/8) q[21];
cx q[23],q[21];
rz(-pi/8) q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
rz(pi/16) q[18];
cx q[15],q[18];
rz(-pi/16) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(-pi/8) q[18];
cx q[23],q[21];
rz(pi/4) q[21];
cx q[23],q[21];
rz(-pi/4) q[21];
cx q[18],q[21];
rz(pi/8) q[21];
cx q[18],q[21];
rz(-pi/8) q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(pi/4) q[21];
sx q[23];
rz(pi/2) q[23];
cx q[21],q[23];
rz(pi/4) q[23];
cx q[21],q[23];
sx q[21];
rz(pi/2) q[21];
rz(-pi/4) q[23];
cx q[7],q[10];
rz(pi/256) q[10];
cx q[7],q[10];
rz(-pi/256) q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
rz(pi/512) q[13];
cx q[14],q[13];
rz(-pi/512) q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/256) q[13];
cx q[16],q[14];
rz(pi/1024) q[14];
cx q[16],q[14];
rz(-pi/1024) q[14];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
rz(-0.073631078) q[10];
cx q[10],q[12];
rz(pi/128) q[12];
cx q[10],q[12];
cx q[10],q[7];
rz(-pi/128) q[12];
cx q[13],q[12];
rz(pi/256) q[12];
cx q[13],q[12];
rz(-pi/256) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/64) q[7];
cx q[10],q[7];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-0.073631078) q[10];
rz(-pi/32) q[12];
cx q[12],q[15];
rz(pi/32) q[15];
cx q[12],q[15];
rz(-pi/32) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(-pi/16) q[15];
cx q[15],q[18];
rz(pi/16) q[18];
cx q[15],q[18];
rz(-pi/16) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(pi/8) q[21];
cx q[21],q[23];
rz(pi/8) q[23];
cx q[21],q[23];
cx q[21],q[18];
rz(pi/4) q[18];
cx q[21],q[18];
rz(-pi/4) q[18];
sx q[21];
rz(pi/2) q[21];
rz(-pi/8) q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(-pi/64) q[7];
cx q[10],q[7];
rz(pi/128) q[7];
cx q[10],q[7];
cx q[10],q[12];
rz(pi/64) q[12];
cx q[10],q[12];
rz(-pi/64) q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-pi/32) q[12];
cx q[12],q[15];
rz(pi/32) q[15];
cx q[12],q[15];
rz(-pi/32) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[18],q[15];
cx q[15],q[18];
rz(pi/8) q[15];
rz(-3*pi/16) q[18];
cx q[18],q[21];
rz(pi/16) q[21];
cx q[18],q[21];
cx q[18],q[15];
rz(-pi/8) q[15];
rz(-pi/16) q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
rz(pi/4) q[21];
cx q[21],q[23];
rz(pi/4) q[23];
cx q[21],q[23];
sx q[21];
rz(pi/2) q[21];
rz(-pi/4) q[23];
rz(-pi/128) q[7];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
rz(-0.023009712) q[11];
cx q[11],q[14];
rz(pi/2048) q[14];
cx q[11],q[14];
rz(-pi/2048) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[7];
cx q[16],q[14];
rz(pi/512) q[14];
cx q[16],q[14];
rz(-pi/512) q[14];
cx q[11],q[14];
rz(pi/1024) q[14];
cx q[11],q[14];
rz(-pi/1024) q[14];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/256) q[13];
cx q[13],q[12];
rz(pi/256) q[12];
cx q[13],q[12];
rz(-pi/256) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-0.073631078) q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[11],q[14];
rz(pi/512) q[14];
cx q[11],q[14];
rz(-pi/512) q[14];
cx q[7],q[10];
cx q[10],q[7];
cx q[12],q[10];
rz(pi/128) q[10];
cx q[12],q[10];
rz(-pi/128) q[10];
cx q[12],q[13];
rz(pi/64) q[13];
cx q[12],q[13];
rz(-pi/64) q[13];
cx q[15],q[12];
cx q[12],q[15];
rz(pi/16) q[12];
rz(-0.29452431) q[15];
cx q[15],q[18];
rz(pi/32) q[18];
cx q[15],q[18];
cx q[15],q[12];
rz(-pi/16) q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[11],q[14];
rz(pi/256) q[14];
cx q[11],q[14];
rz(-pi/256) q[14];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-pi/128) q[13];
cx q[13],q[12];
rz(pi/128) q[12];
cx q[13],q[12];
rz(-pi/128) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/64) q[12];
rz(-pi/32) q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
rz(pi/64) q[15];
cx q[12],q[15];
rz(-pi/64) q[15];
cx q[21],q[18];
cx q[18],q[21];
rz(pi/4) q[18];
rz(pi/8) q[21];
cx q[21],q[23];
rz(pi/8) q[23];
cx q[21],q[23];
cx q[21],q[18];
rz(-pi/4) q[18];
sx q[21];
rz(pi/2) q[21];
rz(-pi/8) q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[6],q[7];
rz(pi/4096) q[7];
cx q[6],q[7];
rz(-pi/4096) q[7];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[19],q[16];
rz(pi/8192) q[16];
cx q[19],q[16];
rz(-pi/8192) q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-pi/4096) q[16];
cx q[20],q[19];
rz(pi/16384) q[19];
cx q[20],q[19];
rz(-pi/16384) q[19];
cx q[6],q[7];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-0.0046019424) q[13];
cx q[13],q[14];
rz(pi/2048) q[14];
cx q[13],q[14];
cx q[13],q[12];
rz(pi/1024) q[12];
cx q[13],q[12];
rz(-pi/1024) q[12];
rz(-pi/2048) q[14];
cx q[16],q[14];
rz(pi/4096) q[14];
cx q[16],q[14];
rz(-pi/4096) q[14];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/2048) q[13];
cx q[13],q[12];
rz(pi/2048) q[12];
cx q[13],q[12];
rz(-pi/2048) q[12];
rz(-pi/512) q[14];
cx q[14],q[11];
rz(pi/512) q[11];
cx q[14],q[11];
rz(-pi/512) q[11];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/1024) q[14];
cx q[14],q[11];
rz(pi/1024) q[11];
cx q[14],q[11];
rz(-pi/1024) q[11];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[20],q[19];
rz(pi/8192) q[19];
cx q[20],q[19];
rz(-pi/8192) q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[20],q[19];
rz(pi/4096) q[19];
cx q[20],q[19];
rz(-pi/4096) q[19];
rz(-pi/32) q[7];
cx q[7],q[6];
rz(pi/32) q[6];
cx q[7],q[6];
rz(-pi/32) q[6];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
rz(-pi/16) q[12];
cx q[12],q[15];
rz(pi/16) q[15];
cx q[12],q[15];
rz(-pi/16) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(-pi/8) q[18];
cx q[18],q[21];
rz(pi/8) q[21];
cx q[18],q[21];
rz(-pi/8) q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(pi/4) q[21];
cx q[21],q[23];
rz(pi/4) q[23];
cx q[21],q[23];
sx q[21];
rz(pi/2) q[21];
rz(-pi/4) q[23];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-0.00067111659) q[13];
cx q[13],q[14];
rz(pi/32768) q[14];
cx q[13],q[14];
rz(-pi/32768) q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/16384) q[14];
cx q[13],q[14];
rz(-pi/16384) q[14];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/8192) q[14];
cx q[13],q[14];
rz(-pi/8192) q[14];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(-pi/2048) q[14];
cx q[14],q[11];
rz(pi/2048) q[11];
cx q[14],q[11];
rz(-pi/2048) q[11];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/1024) q[13];
rz(-0.0023009712) q[14];
cx q[14],q[11];
rz(pi/4096) q[11];
cx q[14],q[11];
rz(-pi/4096) q[11];
rz(-0.00017976337) q[20];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(-pi/131072) q[22];
cx q[24],q[25];
rz(pi/65536) q[25];
cx q[24],q[25];
rz(-pi/65536) q[25];
cx q[22],q[25];
rz(pi/131072) q[25];
cx q[22],q[25];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(-pi/131072) q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[20],q[19];
rz(pi/262144) q[19];
cx q[20],q[19];
rz(-pi/262144) q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(-pi/65536) q[22];
cx q[24],q[25];
rz(pi/32768) q[25];
cx q[24],q[25];
rz(-pi/32768) q[25];
cx q[22],q[25];
rz(pi/65536) q[25];
cx q[22],q[25];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(-pi/65536) q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[20],q[19];
rz(pi/131072) q[19];
cx q[20],q[19];
rz(-pi/131072) q[19];
rz(-pi/32768) q[22];
cx q[24],q[25];
rz(pi/16384) q[25];
cx q[24],q[25];
rz(-pi/16384) q[25];
cx q[22],q[25];
rz(pi/32768) q[25];
cx q[22],q[25];
rz(-pi/32768) q[25];
rz(-pi/256) q[7];
cx q[7],q[4];
rz(pi/256) q[4];
cx q[7],q[4];
rz(-pi/256) q[4];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-pi/512) q[10];
rz(-pi/128) q[12];
cx q[12],q[15];
rz(pi/128) q[15];
cx q[12],q[15];
rz(-pi/128) q[15];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[4];
cx q[10],q[7];
rz(pi/512) q[7];
cx q[10],q[7];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
rz(-pi/256) q[12];
cx q[12],q[15];
rz(pi/256) q[15];
cx q[12],q[15];
rz(-pi/256) q[15];
rz(-pi/512) q[7];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[13],q[12];
rz(pi/1024) q[12];
cx q[13],q[12];
rz(-pi/1024) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/512) q[12];
cx q[12],q[15];
cx q[14],q[13];
rz(pi/2048) q[13];
cx q[14],q[13];
rz(-pi/2048) q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-pi/1024) q[13];
rz(pi/512) q[15];
cx q[12],q[15];
rz(-pi/512) q[15];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[20],q[19];
rz(pi/65536) q[19];
cx q[20],q[19];
rz(-pi/65536) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-pi/16384) q[22];
cx q[24],q[25];
rz(pi/8192) q[25];
cx q[24],q[25];
rz(-pi/8192) q[25];
cx q[22],q[25];
rz(pi/16384) q[25];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(-pi/16384) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[20],q[19];
rz(pi/32768) q[19];
cx q[20],q[19];
rz(-pi/32768) q[19];
rz(-pi/8192) q[22];
cx q[24],q[25];
rz(pi/4096) q[25];
cx q[24],q[25];
rz(-pi/4096) q[25];
cx q[22],q[25];
rz(pi/8192) q[25];
cx q[22],q[25];
rz(-pi/8192) q[25];
rz(-0.14726216) q[7];
cx q[7],q[6];
rz(pi/64) q[6];
cx q[7],q[6];
rz(-pi/64) q[6];
cx q[7],q[4];
rz(pi/32) q[4];
cx q[7],q[4];
rz(-pi/32) q[4];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
rz(pi/1024) q[12];
cx q[13],q[12];
rz(-pi/1024) q[12];
rz(-pi/16) q[15];
cx q[15],q[18];
rz(pi/16) q[18];
cx q[15],q[18];
rz(-pi/16) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(pi/8) q[21];
cx q[21],q[23];
rz(pi/8) q[23];
cx q[21],q[23];
cx q[21],q[18];
rz(pi/4) q[18];
cx q[21],q[18];
rz(-pi/4) q[18];
sx q[21];
rz(pi/2) q[21];
rz(-pi/8) q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(-0.073631078) q[7];
cx q[7],q[6];
rz(pi/128) q[6];
cx q[7],q[6];
rz(-pi/128) q[6];
cx q[7],q[4];
rz(pi/64) q[4];
cx q[7],q[4];
cx q[10],q[7];
rz(-pi/64) q[4];
cx q[7],q[10];
cx q[10],q[7];
rz(-pi/32) q[10];
cx q[10],q[12];
rz(pi/32) q[12];
cx q[10],q[12];
rz(-pi/32) q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[18],q[15];
cx q[15],q[18];
rz(pi/8) q[15];
rz(-3*pi/16) q[18];
cx q[18],q[21];
rz(pi/16) q[21];
cx q[18],q[21];
cx q[18],q[15];
rz(-pi/8) q[15];
rz(-pi/16) q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(pi/4) q[21];
cx q[21],q[23];
rz(pi/4) q[23];
cx q[21],q[23];
sx q[21];
rz(pi/2) q[21];
rz(-pi/4) q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
rz(-0.085902924) q[7];
cx q[7],q[6];
rz(pi/256) q[6];
cx q[7],q[6];
rz(-pi/256) q[6];
cx q[7],q[4];
rz(pi/128) q[4];
cx q[7],q[4];
rz(-pi/128) q[4];
cx q[7],q[10];
rz(pi/64) q[10];
cx q[7],q[10];
rz(-pi/64) q[10];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
rz(-0.042951462) q[10];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(-0.29452431) q[15];
cx q[15],q[18];
rz(pi/32) q[18];
cx q[15],q[18];
cx q[15],q[12];
rz(pi/16) q[12];
cx q[15],q[12];
rz(-pi/16) q[12];
rz(-pi/32) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(-pi/8) q[18];
cx q[18],q[21];
rz(pi/8) q[21];
cx q[18],q[21];
rz(-pi/8) q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
rz(pi/4) q[21];
cx q[21],q[23];
rz(pi/4) q[23];
cx q[21],q[23];
sx q[21];
rz(pi/2) q[21];
rz(-pi/4) q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(-pi/2048) q[12];
cx q[12],q[13];
rz(pi/2048) q[13];
cx q[12],q[13];
rz(-pi/2048) q[13];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[6],q[7];
cx q[7],q[6];
cx q[6],q[7];
cx q[10],q[7];
rz(pi/512) q[7];
cx q[10],q[7];
rz(-pi/512) q[7];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[10],q[7];
rz(pi/256) q[7];
cx q[10],q[7];
rz(-pi/256) q[7];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[6];
cx q[10],q[7];
rz(pi/128) q[7];
cx q[10],q[7];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(-0.14726216) q[15];
cx q[15],q[18];
rz(pi/64) q[18];
cx q[15],q[18];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[15],q[12];
rz(-pi/32) q[12];
rz(-pi/64) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(-pi/16) q[18];
cx q[18],q[21];
rz(pi/16) q[21];
cx q[18],q[21];
rz(-pi/16) q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(-pi/8) q[21];
cx q[21],q[23];
rz(pi/8) q[23];
cx q[21],q[23];
rz(-pi/8) q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(pi/4) q[23];
cx q[23],q[24];
rz(pi/4) q[24];
cx q[23],q[24];
sx q[23];
rz(pi/2) q[23];
rz(-pi/4) q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(-pi/128) q[7];
cx q[7],q[10];
cx q[10],q[7];
rz(pi/256) q[10];
rz(-0.021475731) q[7];
cx q[7],q[4];
rz(pi/1024) q[4];
cx q[7],q[4];
rz(-pi/1024) q[4];
cx q[7],q[6];
rz(pi/512) q[6];
cx q[7],q[6];
rz(-pi/512) q[6];
cx q[7],q[10];
rz(-pi/256) q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[7];
rz(-pi/524288) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
rz(-pi/262144) q[11];
cx q[11],q[14];
rz(pi/262144) q[14];
cx q[11],q[14];
rz(-pi/262144) q[14];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(-7.1905349e-05) q[14];
cx q[14],q[16];
rz(pi/131072) q[16];
cx q[14],q[16];
rz(-pi/131072) q[16];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[14],q[16];
rz(pi/65536) q[16];
cx q[14],q[16];
rz(-pi/65536) q[16];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[19];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(-pi/4096) q[14];
cx q[14],q[13];
rz(pi/4096) q[13];
cx q[14],q[13];
rz(-pi/4096) q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-pi/2048) q[10];
cx q[10],q[7];
rz(-0.073631078) q[12];
cx q[12],q[15];
rz(pi/128) q[15];
cx q[12],q[15];
cx q[12],q[13];
rz(pi/64) q[13];
cx q[12],q[13];
rz(-pi/64) q[13];
rz(-pi/128) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(-pi/32) q[15];
cx q[15],q[18];
rz(pi/32) q[18];
cx q[15],q[18];
rz(-pi/32) q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
rz(-pi/16) q[18];
cx q[18],q[21];
rz(pi/16) q[21];
cx q[18],q[21];
rz(-pi/16) q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
rz(-pi/8) q[21];
cx q[21],q[23];
rz(-pi/16384) q[22];
cx q[22],q[25];
rz(pi/8) q[23];
cx q[21],q[23];
rz(-pi/8) q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
rz(pi/4) q[23];
cx q[23],q[24];
rz(pi/4) q[24];
cx q[23],q[24];
sx q[23];
rz(pi/2) q[23];
rz(-pi/4) q[24];
rz(pi/16384) q[25];
cx q[22],q[25];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-pi/8192) q[16];
cx q[16],q[14];
rz(pi/8192) q[14];
cx q[16],q[14];
rz(-pi/8192) q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(-pi/4096) q[14];
rz(-0.0002876214) q[19];
rz(-pi/16384) q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[19],q[22];
rz(pi/32768) q[22];
cx q[19],q[22];
cx q[19],q[16];
rz(pi/16384) q[16];
cx q[19],q[16];
rz(-pi/16384) q[16];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(-pi/8192) q[16];
rz(-pi/32768) q[22];
rz(pi/2048) q[7];
cx q[10],q[7];
rz(-pi/2048) q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
rz(pi/4096) q[13];
cx q[14],q[13];
rz(-pi/4096) q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[16],q[14];
rz(pi/8192) q[14];
cx q[16],q[14];
rz(-pi/8192) q[14];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(-pi/4096) q[14];
rz(-0.021475731) q[7];
cx q[7],q[6];
rz(pi/1024) q[6];
cx q[7],q[6];
rz(-pi/1024) q[6];
cx q[7],q[4];
rz(pi/512) q[4];
cx q[7],q[4];
rz(-pi/512) q[4];
cx q[7],q[10];
rz(pi/256) q[10];
cx q[7],q[10];
rz(-pi/256) q[10];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
rz(-pi/128) q[10];
cx q[10],q[12];
rz(pi/128) q[12];
cx q[10],q[12];
rz(-pi/128) q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-pi/2048) q[10];
rz(-pi/64) q[12];
cx q[12],q[15];
rz(pi/64) q[15];
cx q[12],q[15];
rz(-pi/64) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(-pi/32) q[15];
cx q[15],q[18];
rz(pi/32) q[18];
cx q[15],q[18];
rz(-pi/32) q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
rz(-pi/16) q[18];
cx q[18],q[21];
rz(pi/16) q[21];
cx q[18],q[21];
rz(-pi/16) q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[23],q[21];
cx q[21],q[23];
rz(pi/4) q[21];
rz(pi/8) q[23];
cx q[23],q[24];
rz(pi/8) q[24];
cx q[23],q[24];
cx q[23],q[21];
rz(-pi/4) q[21];
sx q[23];
rz(pi/2) q[23];
rz(-pi/8) q[24];
cx q[6],q[7];
cx q[7],q[6];
cx q[6],q[7];
cx q[10],q[7];
rz(pi/2048) q[7];
cx q[10],q[7];
rz(-pi/2048) q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
rz(pi/4096) q[13];
cx q[14],q[13];
rz(-pi/4096) q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-0.0092038847) q[7];
cx q[7],q[4];
rz(pi/1024) q[4];
cx q[7],q[4];
rz(-pi/1024) q[4];
cx q[7],q[6];
rz(pi/512) q[6];
cx q[7],q[6];
cx q[10],q[7];
rz(-pi/512) q[6];
cx q[7],q[10];
rz(-0.036815539) q[10];
cx q[10],q[12];
rz(pi/256) q[12];
cx q[10],q[12];
rz(-pi/256) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/128) q[7];
cx q[10],q[7];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
rz(-0.0046019424) q[10];
rz(-pi/64) q[12];
cx q[12],q[15];
rz(pi/64) q[15];
cx q[12],q[15];
rz(-pi/64) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(-pi/32) q[15];
cx q[15],q[18];
rz(pi/32) q[18];
cx q[15],q[18];
rz(-pi/32) q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
rz(-pi/16) q[23];
cx q[23],q[24];
rz(pi/16) q[24];
cx q[23],q[24];
cx q[21],q[23];
cx q[23],q[21];
rz(pi/8) q[21];
cx q[21],q[18];
rz(pi/8) q[18];
cx q[21],q[18];
rz(-pi/8) q[18];
rz(pi/4) q[23];
cx q[21],q[23];
sx q[21];
rz(pi/2) q[21];
rz(-pi/4) q[23];
rz(-pi/16) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(-pi/128) q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[4];
cx q[10],q[7];
rz(pi/2048) q[7];
cx q[10],q[7];
rz(-pi/2048) q[7];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[6];
cx q[10],q[7];
rz(pi/1024) q[7];
cx q[10],q[7];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-0.092038847) q[12];
cx q[12],q[13];
rz(pi/512) q[13];
cx q[12],q[13];
rz(-pi/512) q[13];
rz(-pi/1024) q[7];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[12],q[10];
rz(pi/256) q[10];
cx q[12],q[10];
rz(-pi/256) q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[12],q[10];
rz(pi/128) q[10];
cx q[12],q[10];
rz(-pi/128) q[10];
cx q[12],q[15];
rz(pi/64) q[15];
cx q[12],q[15];
rz(-pi/64) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(-pi/32) q[21];
cx q[21],q[23];
rz(pi/32) q[23];
cx q[21],q[23];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
rz(-pi/16) q[18];
cx q[18],q[15];
rz(pi/16) q[15];
cx q[18],q[15];
rz(-pi/16) q[15];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(pi/8) q[21];
rz(-pi/32) q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[21],q[23];
rz(pi/8) q[23];
cx q[21],q[23];
cx q[21],q[18];
rz(pi/4) q[18];
cx q[21],q[18];
rz(-pi/4) q[18];
sx q[21];
rz(pi/2) q[21];
rz(-pi/8) q[23];
barrier q[9],q[4],q[6],q[11],q[3],q[24],q[13],q[21],q[19],q[17],q[25],q[23],q[26],q[1],q[2],q[14],q[16],q[10],q[22],q[15],q[18],q[20],q[0],q[5],q[12],q[7],q[8];
measure q[8] -> c[0];
measure q[11] -> c[1];
measure q[20] -> c[2];
measure q[25] -> c[3];
measure q[22] -> c[4];
measure q[19] -> c[5];
measure q[16] -> c[6];
measure q[14] -> c[7];
measure q[6] -> c[8];
measure q[4] -> c[9];
measure q[13] -> c[10];
measure q[7] -> c[11];
measure q[10] -> c[12];
measure q[12] -> c[13];
measure q[24] -> c[14];
measure q[15] -> c[15];
measure q[23] -> c[16];
measure q[18] -> c[17];
measure q[21] -> c[18];
