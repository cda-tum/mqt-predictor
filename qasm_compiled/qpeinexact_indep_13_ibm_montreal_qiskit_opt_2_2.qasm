OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[12];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
x q[23];
rz(1.34721865) q[23];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[23],q[24];
rz(-1.34721865) q[24];
cx q[23],q[24];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(-0.4471554) q[21];
cx q[21],q[18];
rz(0.4471554) q[18];
cx q[21],q[18];
rz(-0.4471554) q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(0.023009705) q[18];
cx q[18],q[15];
rz(0.8943108) q[15];
cx q[18],q[15];
rz(-0.8943108) q[15];
rz(1.34721865) q[24];
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
rz(-1.35297105) q[21];
cx q[18],q[21];
rz(1.35297105) q[21];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[18],q[21];
rz(0.435650545) q[21];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(-0.8713011) q[12];
cx q[12],q[13];
rz(0.8713011) q[13];
cx q[12],q[13];
cx q[10],q[12];
cx q[12],q[10];
rz(1.0553788) q[10];
cx q[10],q[7];
rz(0.343611695) q[12];
rz(-0.8713011) q[13];
rz(-0.435650545) q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
rz(-1.3989905) q[7];
cx q[10],q[7];
cx q[10],q[12];
rz(-0.343611695) q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-2.0616702) q[13];
cx q[13],q[14];
rz(0.687223392972765) q[14];
cx q[13],q[14];
rz(-0.687223392972767) q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[19];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[22];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[22],q[19];
cx q[19],q[22];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(1.3989905) q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[13],q[12];
rz(7*pi/16) q[12];
cx q[13],q[12];
rz(-7*pi/16) q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(pi/8) q[10];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[4];
cx q[10],q[7];
rz(-pi/8) q[7];
cx q[10],q[7];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(pi/4) q[16];
cx q[16],q[19];
rz(-pi/4) q[19];
cx q[16],q[19];
rz(pi/4) q[19];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-pi/512) q[13];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
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
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(-pi/8) q[21];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(-pi/16) q[23];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(-0.29452431) q[24];
rz(-pi/1024) q[25];
rz(pi/8) q[7];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
rz(pi/4) q[12];
cx q[12],q[15];
rz(pi/4) q[15];
cx q[12],q[15];
sx q[12];
rz(pi/2) q[12];
rz(-pi/4) q[15];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[21],q[18];
rz(pi/8) q[18];
cx q[21],q[18];
rz(-pi/8) q[18];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(pi/4) q[15];
cx q[15],q[12];
rz(pi/4) q[12];
cx q[15],q[12];
rz(-pi/4) q[12];
sx q[15];
rz(pi/2) q[15];
cx q[23],q[21];
rz(pi/16) q[21];
cx q[23],q[21];
rz(-pi/16) q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
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
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(-pi/4) q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
rz(-pi/64) q[21];
cx q[24],q[23];
rz(pi/32) q[23];
cx q[24],q[23];
rz(-pi/32) q[23];
cx q[21],q[23];
rz(pi/64) q[23];
cx q[21],q[23];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
rz(-pi/64) q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(-pi/32) q[21];
cx q[24],q[23];
rz(pi/16) q[23];
cx q[24],q[23];
rz(-pi/16) q[23];
cx q[21],q[23];
rz(pi/32) q[23];
cx q[21],q[23];
rz(-pi/32) q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(-pi/8) q[23];
cx q[23],q[21];
rz(pi/8) q[21];
cx q[23],q[21];
rz(-pi/8) q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(-pi/16) q[24];
cx q[24],q[23];
rz(pi/16) q[23];
cx q[24],q[23];
rz(-pi/16) q[23];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-pi/128) q[12];
cx q[12],q[15];
rz(pi/128) q[15];
cx q[12],q[15];
rz(-pi/128) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
rz(-pi/64) q[15];
cx q[15],q[18];
rz(pi/64) q[18];
cx q[15],q[18];
rz(-pi/64) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(pi/4) q[18];
cx q[18],q[15];
rz(pi/4) q[15];
cx q[18],q[15];
rz(-pi/4) q[15];
sx q[18];
rz(pi/2) q[18];
rz(-pi/32) q[21];
cx q[21],q[23];
rz(pi/32) q[23];
cx q[21],q[23];
rz(-pi/32) q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
rz(pi/8) q[21];
cx q[21],q[18];
rz(pi/8) q[18];
cx q[21],q[18];
rz(-pi/8) q[18];
cx q[21],q[23];
rz(pi/4) q[23];
cx q[21],q[23];
sx q[21];
rz(pi/2) q[21];
rz(-pi/4) q[23];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
rz(-3*pi/16) q[21];
cx q[21],q[18];
rz(pi/16) q[18];
cx q[21],q[18];
rz(-pi/16) q[18];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
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
rz(-0.036815539) q[7];
cx q[7],q[10];
rz(pi/256) q[10];
cx q[7],q[10];
rz(-pi/256) q[10];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[13],q[12];
rz(pi/512) q[12];
cx q[13],q[12];
rz(-pi/512) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/256) q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[25],q[22];
rz(pi/1024) q[22];
cx q[25],q[22];
rz(-pi/1024) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(-pi/4096) q[14];
cx q[7],q[10];
rz(pi/128) q[10];
cx q[7],q[10];
rz(-pi/128) q[10];
cx q[12],q[10];
rz(pi/256) q[10];
cx q[12],q[10];
rz(-pi/256) q[10];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-pi/128) q[10];
rz(-pi/64) q[12];
cx q[12],q[15];
rz(pi/64) q[15];
cx q[12],q[15];
rz(-pi/64) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[10],q[12];
rz(pi/128) q[12];
cx q[10],q[12];
rz(-pi/128) q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[10],q[7];
rz(-pi/64) q[12];
rz(-pi/32) q[15];
cx q[15],q[18];
rz(pi/32) q[18];
cx q[15],q[18];
rz(-pi/32) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
rz(pi/64) q[15];
cx q[12],q[15];
rz(-pi/64) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(-pi/32) q[15];
rz(-pi/16) q[18];
cx q[18],q[21];
rz(pi/16) q[21];
cx q[18],q[21];
rz(-pi/16) q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
rz(pi/32) q[18];
cx q[15],q[18];
rz(-pi/32) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(-pi/16) q[18];
rz(-pi/8) q[21];
cx q[21],q[23];
rz(pi/8) q[23];
cx q[21],q[23];
rz(-pi/8) q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
rz(pi/16) q[21];
cx q[18],q[21];
rz(-pi/16) q[21];
rz(pi/4) q[23];
cx q[23],q[24];
rz(pi/4) q[24];
cx q[23],q[24];
sx q[23];
rz(pi/2) q[23];
rz(-pi/4) q[24];
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
rz(-pi/512) q[18];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
rz(-pi/8) q[23];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[23],q[24];
rz(pi/8) q[24];
cx q[23],q[24];
rz(-pi/8) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(pi/4) q[24];
cx q[24],q[25];
rz(pi/4) q[25];
cx q[24],q[25];
sx q[24];
rz(pi/2) q[24];
rz(-pi/4) q[25];
cx q[7],q[10];
cx q[10],q[7];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[18],q[15];
rz(pi/512) q[15];
cx q[18],q[15];
rz(-pi/512) q[15];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-0.0046019424) q[12];
cx q[12],q[13];
rz(pi/2048) q[13];
cx q[12],q[13];
cx q[12],q[15];
rz(-pi/2048) q[13];
cx q[14],q[13];
rz(pi/4096) q[13];
cx q[14],q[13];
rz(-pi/4096) q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-0.0046019424) q[13];
rz(pi/1024) q[15];
cx q[12],q[15];
rz(-pi/1024) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
rz(pi/2048) q[12];
cx q[13],q[12];
rz(-pi/2048) q[12];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
rz(-pi/256) q[15];
rz(-0.018407769) q[18];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[15],q[12];
rz(pi/256) q[12];
cx q[15],q[12];
rz(-pi/256) q[12];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(-pi/128) q[12];
cx q[18],q[15];
rz(pi/512) q[15];
cx q[18],q[15];
rz(-pi/512) q[15];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[4];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[12],q[10];
rz(pi/128) q[10];
cx q[12],q[10];
rz(-pi/128) q[10];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
rz(-pi/64) q[10];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[13],q[12];
rz(pi/1024) q[12];
cx q[13],q[12];
rz(-pi/1024) q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/512) q[12];
cx q[18],q[15];
rz(pi/256) q[15];
cx q[18],q[15];
rz(-pi/256) q[15];
cx q[12],q[15];
rz(pi/512) q[15];
cx q[12],q[15];
rz(-pi/512) q[15];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[4];
cx q[10],q[7];
rz(pi/64) q[7];
cx q[10],q[7];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
rz(-pi/128) q[10];
rz(-pi/32) q[18];
cx q[18],q[21];
rz(pi/32) q[21];
cx q[18],q[21];
rz(-pi/32) q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(-pi/16) q[21];
cx q[21],q[23];
rz(pi/16) q[23];
cx q[21],q[23];
rz(-pi/16) q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(pi/8) q[24];
cx q[24],q[25];
rz(pi/8) q[25];
cx q[24],q[25];
cx q[24],q[23];
rz(pi/4) q[23];
cx q[24],q[23];
rz(-pi/4) q[23];
sx q[24];
rz(pi/2) q[24];
rz(-pi/8) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(-pi/64) q[7];
cx q[10],q[7];
rz(pi/128) q[7];
cx q[10],q[7];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-pi/256) q[10];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(-pi/64) q[15];
cx q[15],q[18];
rz(pi/64) q[18];
cx q[15],q[18];
rz(-pi/64) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(-pi/32) q[18];
cx q[18],q[21];
rz(pi/32) q[21];
cx q[18],q[21];
rz(-pi/32) q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(-3*pi/16) q[23];
cx q[23],q[24];
rz(pi/16) q[24];
cx q[23],q[24];
cx q[23],q[21];
rz(pi/8) q[21];
cx q[23],q[21];
rz(-pi/8) q[21];
rz(-pi/16) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(pi/4) q[24];
cx q[24],q[25];
rz(pi/4) q[25];
cx q[24],q[25];
sx q[24];
rz(pi/2) q[24];
rz(-pi/4) q[25];
rz(-pi/128) q[7];
cx q[10],q[7];
rz(pi/256) q[7];
cx q[10],q[7];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-pi/128) q[12];
cx q[12],q[15];
rz(pi/128) q[15];
cx q[12],q[15];
rz(-pi/128) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(-pi/64) q[15];
cx q[15],q[18];
rz(pi/64) q[18];
cx q[15],q[18];
rz(-pi/64) q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[21],q[18];
cx q[18],q[21];
rz(pi/16) q[18];
rz(-0.29452431) q[21];
cx q[21],q[23];
rz(pi/32) q[23];
cx q[21],q[23];
cx q[21],q[18];
rz(-pi/16) q[18];
rz(-pi/32) q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[24],q[23];
cx q[23],q[24];
rz(pi/4) q[23];
rz(pi/8) q[24];
cx q[24],q[25];
rz(pi/8) q[25];
cx q[24],q[25];
cx q[24],q[23];
rz(-pi/4) q[23];
sx q[24];
rz(pi/2) q[24];
rz(-pi/8) q[25];
rz(-pi/256) q[7];
barrier q[2],q[5],q[11],q[8],q[18],q[17],q[22],q[20],q[26],q[25],q[1],q[15],q[21],q[16],q[12],q[19],q[24],q[10],q[0],q[3],q[9],q[6],q[23],q[4],q[13],q[7],q[14];
measure q[14] -> c[0];
measure q[4] -> c[1];
measure q[13] -> c[2];
measure q[10] -> c[3];
measure q[7] -> c[4];
measure q[12] -> c[5];
measure q[15] -> c[6];
measure q[21] -> c[7];
measure q[18] -> c[8];
measure q[25] -> c[9];
measure q[23] -> c[10];
measure q[24] -> c[11];
