OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[17];
rz(-3*pi/2) q[3];
sx q[3];
rz(pi) q[3];
rz(-3*pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-3*pi/2) q[7];
sx q[7];
rz(1.5707484) q[7];
rz(-3*pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-3*pi/2) q[9];
sx q[9];
rz(3*pi/4) q[9];
rz(-pi) q[10];
sx q[10];
rz(2.2142974) q[10];
sx q[10];
cx q[7],q[10];
sx q[10];
rz(2.2142974) q[10];
sx q[10];
rz(-pi) q[10];
cx q[7],q[10];
rz(-pi) q[10];
sx q[10];
rz(2.2142974) q[10];
sx q[10];
rz(-3*pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-3*pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-3*pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-3*pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-3*pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[12];
sx q[12];
rz(1.2870023) q[12];
sx q[12];
rz(-pi) q[12];
cx q[15],q[12];
rz(-pi) q[12];
sx q[12];
rz(1.2870023) q[12];
sx q[12];
cx q[10],q[12];
rz(-pi) q[12];
sx q[12];
rz(0.56758825) q[12];
sx q[12];
cx q[10],q[12];
sx q[12];
rz(0.56758825) q[12];
sx q[12];
rz(-pi) q[12];
rz(-3*pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-3*pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
sx q[12];
rz(2.0064163) q[12];
sx q[12];
rz(-pi) q[12];
cx q[15],q[12];
rz(-pi) q[12];
sx q[12];
rz(2.0064163) q[12];
sx q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/8192) q[15];
cx q[16],q[14];
sx q[14];
rz(0.87124027) q[14];
sx q[14];
rz(-pi) q[14];
cx q[16],q[14];
rz(-pi) q[14];
sx q[14];
rz(0.87123975) q[14];
sx q[14];
rz(-3*pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[16],q[14];
rz(-pi) q[14];
sx q[14];
rz(1.3991131) q[14];
sx q[14];
cx q[16],q[14];
sx q[14];
rz(1.3991131) q[14];
sx q[14];
rz(-pi) q[14];
rz(-3*pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[16],q[14];
sx q[14];
rz(0.34336642) q[14];
sx q[14];
rz(-pi) q[14];
cx q[16],q[14];
rz(-pi) q[14];
sx q[14];
rz(0.34336645) q[14];
sx q[14];
rz(-3*pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
rz(-3*pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[16],q[14];
rz(-pi) q[14];
sx q[14];
rz(2.4548618) q[14];
sx q[14];
cx q[16],q[14];
sx q[14];
rz(2.4548597) q[14];
sx q[14];
rz(-pi) q[14];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[22],q[25];
rz(-pi/4096) q[24];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[14];
rz(-pi) q[14];
sx q[14];
rz(1.768131) q[14];
sx q[14];
cx q[16],q[14];
sx q[14];
rz(1.7681268) q[14];
sx q[14];
rz(-pi) q[14];
cx q[13],q[14];
rz(-pi) q[14];
sx q[14];
rz(0.39465931) q[14];
sx q[14];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-0.17180585) q[12];
sx q[14];
rz(0.39466095) q[14];
sx q[14];
rz(-pi) q[14];
cx q[13],q[14];
sx q[14];
rz(2.352274) q[14];
sx q[14];
rz(-pi) q[14];
cx q[13],q[14];
rz(-pi/64) q[13];
rz(-pi) q[14];
sx q[14];
rz(2.3522708) q[14];
sx q[14];
cx q[11],q[14];
sx q[14];
rz(1.5629554) q[14];
sx q[14];
rz(-pi) q[14];
cx q[11],q[14];
rz(-pi) q[14];
sx q[14];
rz(1.562949) q[14];
sx q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(-pi/32) q[14];
rz(-pi/256) q[16];
rz(-pi/512) q[19];
rz(-pi/1024) q[22];
rz(-pi/2048) q[25];
cx q[8],q[11];
rz(-pi) q[11];
sx q[11];
rz(0.01568181) q[11];
sx q[11];
cx q[8],q[11];
sx q[11];
rz(0.015694754) q[11];
sx q[11];
rz(-pi) q[11];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
rz(-pi/16) q[11];
cx q[5],q[8];
sx q[8];
rz(3.110229) q[8];
sx q[8];
rz(-pi) q[8];
cx q[5],q[8];
rz(-pi) q[8];
sx q[8];
rz(3.1102032) q[8];
sx q[8];
cx q[9],q[8];
sx q[8];
rz(3.0786654) q[8];
sx q[8];
rz(-pi) q[8];
cx q[9],q[8];
rz(-pi) q[8];
sx q[8];
rz(3.0788137) q[8];
sx q[8];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[3],q[5];
sx q[5];
rz(3.0157382) q[5];
sx q[5];
rz(-pi) q[5];
cx q[3],q[5];
sx q[3];
rz(pi/2) q[3];
rz(-pi) q[5];
sx q[5];
rz(3.0157382) q[5];
sx q[5];
cx q[3],q[5];
cx q[5],q[3];
cx q[3],q[5];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
rz(pi/8) q[5];
cx q[9],q[8];
rz(pi/4) q[8];
cx q[9],q[8];
rz(-pi/4) q[8];
cx q[5],q[8];
rz(pi/8) q[8];
cx q[5],q[8];
rz(-pi/8) q[8];
cx q[11],q[8];
rz(pi/16) q[8];
cx q[11],q[8];
rz(-pi/16) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
rz(pi/32) q[11];
cx q[14],q[11];
rz(-pi/32) q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(-pi/16) q[11];
cx q[13],q[14];
rz(pi/64) q[14];
cx q[13],q[14];
rz(-pi/64) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
rz(pi/128) q[13];
cx q[12],q[13];
rz(-pi/128) q[13];
rz(-pi/32) q[14];
sx q[9];
rz(pi/2) q[9];
cx q[8],q[9];
cx q[9],q[8];
cx q[8],q[9];
cx q[5],q[8];
rz(pi/4) q[8];
cx q[5],q[8];
sx q[5];
rz(pi/2) q[5];
rz(-pi/4) q[8];
rz(pi/8) q[9];
cx q[9],q[8];
rz(pi/8) q[8];
cx q[9],q[8];
rz(-pi/8) q[8];
cx q[11],q[8];
rz(pi/16) q[8];
cx q[11],q[8];
rz(-pi/16) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
rz(pi/32) q[11];
cx q[14],q[11];
rz(-pi/32) q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(-pi/16) q[11];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
rz(pi/64) q[13];
cx q[12],q[13];
rz(-pi/64) q[13];
cx q[16],q[14];
rz(pi/256) q[14];
cx q[16],q[14];
rz(-pi/256) q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(-pi/128) q[14];
cx q[14],q[13];
rz(pi/128) q[13];
cx q[14],q[13];
rz(-pi/128) q[13];
cx q[19],q[16];
rz(pi/512) q[16];
cx q[19],q[16];
rz(-pi/512) q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-pi/256) q[16];
cx q[22],q[19];
rz(pi/1024) q[19];
cx q[22],q[19];
rz(-pi/1024) q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(-pi/512) q[19];
cx q[25],q[22];
rz(pi/2048) q[22];
cx q[25],q[22];
rz(-pi/2048) q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(-pi/1024) q[22];
cx q[24],q[25];
rz(pi/4096) q[25];
cx q[24],q[25];
rz(-pi/4096) q[25];
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
cx q[15],q[18];
rz(pi/8192) q[18];
cx q[15],q[18];
rz(-pi/8192) q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
rz(-pi/32768) q[18];
rz(-pi/4096) q[21];
rz(-pi/2048) q[25];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
rz(pi/8) q[5];
cx q[9],q[8];
rz(pi/4) q[8];
cx q[9],q[8];
rz(-pi/4) q[8];
cx q[5],q[8];
rz(pi/8) q[8];
cx q[5],q[8];
rz(-pi/8) q[8];
cx q[11],q[8];
rz(pi/16) q[8];
cx q[11],q[8];
rz(-pi/16) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(-pi/64) q[11];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[16],q[14];
rz(pi/256) q[14];
cx q[16],q[14];
rz(-pi/256) q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[11],q[14];
rz(-pi/128) q[13];
rz(pi/64) q[14];
cx q[11],q[14];
rz(-pi/64) q[14];
cx q[13],q[14];
rz(pi/128) q[14];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/16) q[13];
rz(-pi/128) q[14];
cx q[19],q[16];
rz(pi/512) q[16];
cx q[19],q[16];
rz(-pi/512) q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-pi/256) q[16];
cx q[16],q[14];
rz(pi/256) q[14];
cx q[16],q[14];
rz(-pi/256) q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[22],q[19];
rz(pi/1024) q[19];
cx q[22],q[19];
rz(-pi/1024) q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(-pi/512) q[19];
cx q[19],q[16];
rz(pi/512) q[16];
cx q[19],q[16];
rz(-pi/512) q[16];
cx q[25],q[22];
rz(pi/2048) q[22];
cx q[25],q[22];
rz(-pi/2048) q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(-pi/1024) q[19];
cx q[19],q[16];
rz(pi/1024) q[16];
cx q[19],q[16];
rz(-pi/1024) q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(-pi/256) q[19];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[21],q[23];
rz(pi/4096) q[23];
cx q[21],q[23];
rz(-pi/4096) q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(-pi/2048) q[25];
cx q[25],q[22];
rz(pi/2048) q[22];
cx q[25],q[22];
rz(-pi/2048) q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(-pi/1024) q[22];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[5],q[8];
rz(pi/4) q[8];
cx q[5],q[8];
sx q[5];
rz(pi/2) q[5];
rz(-pi/4) q[8];
rz(pi/8) q[9];
cx q[9],q[8];
rz(pi/8) q[8];
cx q[9],q[8];
rz(-pi/8) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[13],q[14];
rz(pi/16) q[14];
cx q[13],q[14];
rz(-pi/16) q[14];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
rz(pi/32) q[5];
cx q[9],q[8];
rz(pi/4) q[8];
cx q[9],q[8];
rz(-pi/4) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[13],q[14];
rz(pi/8) q[14];
cx q[13],q[14];
rz(-pi/8) q[14];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[13],q[14];
rz(pi/4) q[14];
cx q[13],q[14];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-0.3436117) q[13];
rz(-pi/4) q[14];
cx q[5],q[8];
rz(pi/32) q[8];
cx q[5],q[8];
rz(-pi/32) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[13],q[14];
rz(pi/64) q[14];
cx q[13],q[14];
rz(-pi/64) q[14];
cx q[5],q[8];
rz(pi/16) q[8];
cx q[5],q[8];
rz(-pi/16) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[5],q[8];
rz(pi/8) q[8];
cx q[5],q[8];
rz(-pi/8) q[8];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[13],q[14];
rz(pi/16) q[14];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-pi/16384) q[12];
cx q[12],q[15];
rz(-pi/16) q[14];
rz(pi/16384) q[15];
cx q[12],q[15];
rz(-pi/16384) q[15];
cx q[18],q[15];
rz(pi/32768) q[15];
cx q[18],q[15];
rz(-pi/32768) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
rz(-pi/8192) q[15];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[15],q[18];
rz(pi/8192) q[18];
cx q[15],q[18];
rz(-pi/8192) q[18];
rz(-pi/16384) q[21];
cx q[21],q[18];
rz(pi/16384) q[18];
cx q[21],q[18];
rz(-pi/16384) q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
rz(-pi/4096) q[18];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[18],q[21];
rz(pi/4096) q[21];
cx q[18],q[21];
rz(-pi/4096) q[21];
rz(-pi/8192) q[23];
cx q[23],q[21];
rz(pi/8192) q[21];
cx q[23],q[21];
rz(-pi/8192) q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
rz(-pi/2048) q[21];
cx q[7],q[10];
rz(pi/65536) q[10];
cx q[7],q[10];
rz(-pi/65536) q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[7];
rz(-0.17180585) q[9];
cx q[9],q[8];
rz(pi/128) q[8];
cx q[9],q[8];
rz(-pi/128) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(-pi/512) q[14];
cx q[19],q[16];
rz(pi/256) q[16];
cx q[19],q[16];
rz(-pi/256) q[16];
cx q[14],q[16];
rz(pi/512) q[16];
cx q[14],q[16];
rz(-pi/512) q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-0.073631078) q[16];
cx q[22],q[19];
rz(pi/1024) q[19];
cx q[22],q[19];
rz(-pi/1024) q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[21],q[23];
rz(pi/2048) q[23];
cx q[21],q[23];
rz(-pi/2048) q[23];
rz(-0.0023009712) q[24];
cx q[24],q[23];
rz(pi/4096) q[23];
cx q[24],q[23];
rz(-pi/4096) q[23];
cx q[9],q[8];
rz(pi/64) q[8];
cx q[9],q[8];
rz(-pi/64) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(-pi/256) q[11];
cx q[16],q[14];
rz(pi/128) q[14];
cx q[16],q[14];
rz(-pi/128) q[14];
cx q[11],q[14];
rz(pi/256) q[14];
cx q[11],q[14];
rz(-pi/256) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[9],q[8];
rz(pi/32) q[8];
cx q[9],q[8];
rz(-pi/32) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[11],q[8];
cx q[16],q[14];
rz(pi/64) q[14];
cx q[16],q[14];
rz(-pi/64) q[14];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[8],q[11];
cx q[11],q[8];
rz(-pi/128) q[11];
cx q[11],q[14];
rz(pi/128) q[14];
cx q[11],q[14];
rz(-pi/128) q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(-0.018407769) q[14];
cx q[14],q[13];
rz(pi/512) q[13];
cx q[14],q[13];
rz(-pi/512) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-0.0002876214) q[12];
cx q[12],q[15];
rz(pi/32768) q[15];
cx q[12],q[15];
rz(-pi/32768) q[15];
rz(pi/256) q[16];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-pi/128) q[13];
rz(-pi/8) q[14];
rz(-pi/256) q[16];
cx q[16],q[19];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
rz(pi/16384) q[15];
cx q[12],q[15];
rz(-pi/16384) q[15];
cx q[18],q[21];
cx q[19],q[16];
cx q[16],q[19];
rz(-0.68722339) q[16];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(-pi/1024) q[12];
cx q[12],q[10];
rz(pi/1024) q[10];
cx q[12],q[10];
rz(-pi/1024) q[10];
rz(-pi/8192) q[15];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
rz(pi/8192) q[18];
cx q[15],q[18];
rz(-pi/8192) q[18];
cx q[5],q[8];
rz(pi/4) q[8];
cx q[5],q[8];
sx q[5];
rz(pi/2) q[5];
rz(-pi/4) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
rz(pi/8) q[11];
cx q[14],q[11];
rz(-pi/8) q[11];
cx q[8],q[9];
cx q[9],q[8];
cx q[8],q[9];
rz(-pi/16) q[8];
cx q[8],q[11];
rz(pi/16) q[11];
cx q[8],q[11];
rz(-pi/16) q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(pi/4) q[11];
cx q[16],q[14];
rz(pi/32) q[14];
cx q[16],q[14];
rz(-pi/32) q[14];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[5];
cx q[11],q[8];
rz(pi/8) q[5];
rz(pi/4) q[8];
cx q[11],q[8];
sx q[11];
rz(pi/2) q[11];
rz(-pi/4) q[8];
cx q[5],q[8];
rz(pi/8) q[8];
cx q[5],q[8];
rz(-pi/8) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[16],q[14];
rz(pi/16) q[14];
cx q[16],q[14];
rz(-pi/16) q[14];
cx q[5],q[8];
rz(pi/4) q[8];
cx q[5],q[8];
sx q[5];
rz(pi/2) q[5];
rz(-pi/4) q[8];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[16],q[14];
rz(pi/8) q[14];
cx q[16],q[14];
rz(-pi/8) q[14];
rz(-0.3436117) q[9];
cx q[9],q[8];
rz(pi/64) q[8];
cx q[9],q[8];
rz(-pi/64) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[13],q[14];
rz(pi/128) q[14];
cx q[13],q[14];
rz(-pi/128) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/64) q[14];
cx q[9],q[8];
rz(pi/32) q[8];
cx q[9],q[8];
rz(-pi/32) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
rz(pi/64) q[11];
cx q[14],q[11];
rz(-pi/64) q[11];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(-pi/32) q[11];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(pi/4) q[14];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[9],q[8];
rz(pi/16) q[8];
cx q[9],q[8];
rz(-pi/16) q[8];
cx q[11],q[8];
rz(pi/32) q[8];
cx q[11],q[8];
rz(-pi/32) q[8];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
rz(pi/4) q[11];
cx q[14],q[11];
rz(-pi/4) q[11];
sx q[14];
rz(pi/2) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/512) q[14];
cx q[14],q[16];
rz(pi/512) q[16];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-pi/256) q[13];
cx q[13],q[12];
rz(pi/256) q[12];
cx q[13],q[12];
rz(-pi/256) q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(-pi/4096) q[12];
rz(-pi/512) q[16];
cx q[16],q[19];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[19],q[16];
cx q[16],q[19];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[24],q[23];
rz(pi/2048) q[23];
cx q[24],q[23];
rz(-pi/2048) q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
rz(pi/4096) q[15];
cx q[12],q[15];
rz(-pi/4096) q[15];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
rz(-pi/1024) q[22];
cx q[22],q[19];
rz(pi/1024) q[19];
cx q[22],q[19];
rz(-pi/1024) q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
rz(-pi/8) q[8];
cx q[8],q[11];
rz(pi/8) q[11];
cx q[8],q[11];
rz(-pi/8) q[11];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
rz(pi/4) q[11];
cx q[11],q[14];
rz(pi/4) q[14];
cx q[11],q[14];
sx q[11];
rz(pi/2) q[11];
rz(-pi/4) q[14];
rz(-pi/16) q[9];
cx q[9],q[8];
rz(pi/16) q[8];
cx q[9],q[8];
rz(-pi/16) q[8];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[11],q[8];
cx q[8],q[11];
rz(pi/8) q[11];
cx q[11],q[14];
rz(pi/8) q[14];
cx q[11],q[14];
rz(-pi/8) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/128) q[14];
cx q[14],q[16];
rz(pi/128) q[16];
cx q[14],q[16];
rz(-pi/128) q[16];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(pi/4) q[8];
cx q[11],q[8];
sx q[11];
rz(pi/2) q[11];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/512) q[12];
cx q[12],q[10];
rz(pi/512) q[10];
cx q[12],q[10];
rz(-pi/512) q[10];
rz(-pi/4) q[8];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
rz(-0.14726216) q[8];
cx q[8],q[5];
rz(pi/64) q[5];
cx q[8],q[5];
rz(-pi/64) q[5];
cx q[8],q[9];
rz(pi/32) q[9];
cx q[8],q[9];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
rz(pi/16) q[11];
cx q[11],q[14];
rz(pi/16) q[14];
cx q[11],q[14];
cx q[11],q[8];
rz(-pi/16) q[14];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/8) q[8];
cx q[11],q[8];
cx q[11],q[14];
rz(pi/4) q[14];
cx q[11],q[14];
sx q[11];
rz(pi/2) q[11];
rz(-pi/4) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/256) q[13];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/256) q[14];
cx q[13],q[14];
rz(-pi/256) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[11],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[14],q[11];
cx q[11],q[14];
rz(-0.073631078) q[11];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-pi/2048) q[19];
cx q[19],q[22];
rz(pi/2048) q[22];
cx q[19],q[22];
rz(-pi/2048) q[22];
rz(-pi/8) q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[5];
cx q[11],q[8];
rz(pi/128) q[8];
cx q[11],q[8];
rz(-pi/128) q[8];
rz(-pi/32) q[9];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[11],q[8];
rz(pi/64) q[8];
cx q[11],q[8];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(-pi/32) q[14];
cx q[14],q[16];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(-3*pi/16) q[11];
rz(-pi/32) q[16];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-0.0092038847) q[13];
cx q[13],q[12];
rz(pi/1024) q[12];
cx q[13],q[12];
rz(-pi/1024) q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[13],q[12];
rz(pi/512) q[12];
cx q[13],q[12];
rz(-pi/512) q[12];
rz(-pi/64) q[8];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[11],q[8];
rz(pi/16) q[8];
cx q[11],q[8];
cx q[11],q[14];
rz(pi/8) q[14];
cx q[11],q[14];
rz(-pi/8) q[14];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(pi/4) q[14];
cx q[14],q[16];
rz(pi/4) q[16];
cx q[14],q[16];
sx q[14];
rz(pi/2) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(-0.036815539) q[11];
rz(-pi/4) q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-pi/16) q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[8],q[9];
cx q[11],q[8];
rz(pi/256) q[8];
cx q[11],q[8];
rz(-pi/256) q[8];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[11],q[8];
rz(pi/128) q[8];
cx q[11],q[8];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(pi/64) q[14];
cx q[14],q[16];
rz(pi/64) q[16];
cx q[14],q[16];
rz(-pi/64) q[16];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(-pi/128) q[8];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
rz(pi/32) q[11];
cx q[14],q[11];
rz(-pi/32) q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
rz(pi/16) q[11];
cx q[14],q[11];
rz(-pi/16) q[11];
cx q[14],q[16];
rz(pi/8) q[16];
cx q[14],q[16];
cx q[14],q[13];
rz(pi/4) q[13];
cx q[14],q[13];
rz(-pi/4) q[13];
sx q[14];
rz(pi/2) q[14];
rz(-pi/8) q[16];
barrier q[2],q[21],q[15],q[18],q[10],q[20],q[17],q[25],q[26],q[1],q[4],q[14],q[22],q[3],q[8],q[19],q[5],q[9],q[0],q[6],q[7],q[23],q[16],q[11],q[13],q[24],q[12];
measure q[14] -> meas[0];
measure q[13] -> meas[1];
measure q[16] -> meas[2];
measure q[11] -> meas[3];
measure q[8] -> meas[4];
measure q[19] -> meas[5];
measure q[9] -> meas[6];
measure q[5] -> meas[7];
measure q[12] -> meas[8];
measure q[10] -> meas[9];
measure q[22] -> meas[10];
measure q[15] -> meas[11];
measure q[18] -> meas[12];
measure q[21] -> meas[13];
measure q[23] -> meas[14];
measure q[7] -> meas[15];
measure q[3] -> meas[16];
