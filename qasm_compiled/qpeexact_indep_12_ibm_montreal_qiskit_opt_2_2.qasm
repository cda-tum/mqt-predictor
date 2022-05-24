OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[11];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
x q[23];
rz(-1.34683515) q[23];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[23],q[24];
rz(1.34683515) q[24];
cx q[23],q[24];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(-0.00613591) q[21];
cx q[21],q[18];
rz(-0.44792239) q[18];
cx q[21],q[18];
rz(0.44792239) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[21],q[18];
rz(-0.8958448) q[18];
cx q[21],q[18];
rz(0.8958448) q[18];
rz(-1.34683515) q[24];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[22],q[25];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[21],q[23];
rz(1.3499031) q[23];
cx q[21],q[23];
rz(-1.3499031) q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(0.441786465) q[23];
cx q[23],q[24];
rz(-0.441786465) q[24];
cx q[23],q[24];
rz(0.441786465) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(0.883572933822129) q[24];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[24],q[25];
rz(-0.88357293382213) q[25];
cx q[24],q[25];
rz(0.883572933822129) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
rz(-5*pi/16) q[25];
cx q[25],q[22];
rz(7*pi/16) q[22];
cx q[25],q[22];
rz(-7*pi/16) q[22];
rz(pi/2) q[26];
sx q[26];
rz(pi/2) q[26];
cx q[25],q[26];
rz(-pi/8) q[26];
cx q[25],q[26];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(3*pi/4) q[14];
cx q[14],q[13];
rz(-pi/4) q[13];
cx q[14],q[13];
rz(pi/4) q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[12],q[15];
cx q[14],q[13];
rz(-pi/2) q[13];
cx q[14],q[13];
rz(pi/2) q[13];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/1024) q[12];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
rz(-pi/512) q[15];
cx q[18],q[21];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(-pi/2048) q[16];
rz(pi/4) q[19];
cx q[19],q[22];
cx q[21],q[18];
cx q[18],q[21];
rz(-0.3436117) q[18];
rz(-pi/8) q[21];
rz(pi/4) q[22];
cx q[19],q[22];
sx q[19];
rz(pi/2) q[19];
rz(-pi/4) q[22];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(pi/8) q[26];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[21],q[23];
rz(pi/8) q[23];
cx q[21],q[23];
rz(-pi/8) q[23];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(-pi/32) q[23];
rz(-pi/16) q[25];
cx q[25],q[24];
rz(pi/16) q[24];
cx q[25],q[24];
rz(-pi/16) q[24];
cx q[23],q[24];
rz(pi/32) q[24];
cx q[23],q[24];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(-pi/32) q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
rz(pi/64) q[21];
cx q[18],q[21];
rz(-pi/64) q[21];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(pi/4) q[25];
cx q[25],q[22];
rz(pi/4) q[22];
cx q[25],q[22];
rz(-pi/4) q[22];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(pi/4) q[24];
rz(pi/8) q[25];
cx q[25],q[22];
rz(pi/8) q[22];
cx q[25],q[22];
rz(-pi/8) q[22];
cx q[25],q[24];
rz(-pi/4) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
sx q[25];
rz(pi/2) q[25];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(-pi/128) q[23];
cx q[23],q[21];
rz(pi/128) q[21];
cx q[23],q[21];
rz(-pi/128) q[21];
rz(pi/16) q[25];
cx q[25],q[22];
rz(pi/16) q[22];
cx q[25],q[22];
rz(-pi/16) q[22];
cx q[25],q[24];
rz(pi/8) q[24];
cx q[25],q[24];
rz(-pi/8) q[24];
cx q[25],q[26];
rz(pi/4) q[26];
cx q[25],q[26];
sx q[25];
rz(pi/2) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(-0.036815539) q[22];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[18],q[21];
rz(pi/32) q[21];
cx q[18],q[21];
rz(-pi/32) q[21];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(-pi/64) q[23];
cx q[23],q[21];
rz(pi/64) q[21];
cx q[23],q[21];
rz(-pi/64) q[21];
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
rz(pi/16) q[21];
cx q[18],q[21];
rz(-pi/16) q[21];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(-pi/32) q[23];
cx q[23],q[21];
rz(pi/32) q[21];
cx q[23],q[21];
rz(-pi/32) q[21];
rz(pi/256) q[25];
cx q[22],q[25];
rz(-pi/256) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[22],q[25];
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
rz(pi/512) q[18];
cx q[15],q[18];
rz(-pi/512) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
rz(pi/1024) q[15];
cx q[12],q[15];
rz(-pi/1024) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[16],q[14];
rz(pi/2048) q[14];
cx q[16],q[14];
rz(-pi/2048) q[14];
rz(pi/128) q[25];
cx q[22],q[25];
rz(-pi/128) q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(-pi/64) q[24];
cx q[24],q[23];
rz(pi/64) q[23];
cx q[24],q[23];
rz(-pi/64) q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(-pi/8) q[23];
rz(-pi/4) q[26];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[23],q[24];
rz(pi/8) q[24];
cx q[23],q[24];
rz(-pi/8) q[24];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
rz(-pi/16) q[25];
cx q[25],q[24];
rz(pi/16) q[24];
cx q[25],q[24];
rz(-pi/16) q[24];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
rz(-pi/32) q[25];
cx q[25],q[24];
rz(pi/32) q[24];
cx q[25],q[24];
cx q[22],q[25];
rz(-pi/32) q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
rz(-0.085902924) q[23];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[23],q[24];
rz(pi/256) q[24];
cx q[23],q[24];
cx q[23],q[21];
rz(pi/128) q[21];
cx q[23],q[21];
rz(-pi/128) q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[23],q[21];
rz(pi/64) q[21];
cx q[23],q[21];
rz(-pi/64) q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(-0.018407769) q[21];
rz(-pi/256) q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[21],q[23];
rz(pi/512) q[23];
cx q[21],q[23];
cx q[21],q[18];
rz(pi/256) q[18];
cx q[21],q[18];
rz(-pi/256) q[18];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
rz(-pi/128) q[18];
cx q[18],q[15];
rz(pi/128) q[15];
cx q[18],q[15];
rz(-pi/128) q[15];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/64) q[13];
rz(-pi/512) q[23];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
rz(pi/4) q[22];
cx q[22],q[19];
rz(pi/4) q[19];
cx q[22],q[19];
rz(-pi/4) q[19];
sx q[22];
rz(pi/2) q[22];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
rz(pi/8) q[22];
cx q[22],q[19];
rz(pi/8) q[19];
cx q[22],q[19];
rz(-pi/8) q[19];
cx q[22],q[25];
rz(pi/4) q[25];
cx q[22],q[25];
sx q[22];
rz(pi/2) q[22];
rz(-pi/4) q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(-pi/16) q[22];
cx q[22],q[19];
rz(pi/16) q[19];
cx q[22],q[19];
rz(-pi/16) q[19];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
rz(-pi/32) q[22];
cx q[22],q[19];
rz(pi/32) q[19];
cx q[22],q[19];
rz(-pi/32) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[13],q[14];
rz(pi/64) q[14];
cx q[13],q[14];
rz(-pi/64) q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/8) q[25];
cx q[25],q[26];
rz(pi/8) q[26];
cx q[25],q[26];
cx q[25],q[24];
rz(pi/4) q[24];
cx q[25],q[24];
rz(-pi/4) q[24];
sx q[25];
rz(pi/2) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(pi/16) q[25];
rz(-pi/8) q[26];
cx q[25],q[26];
rz(pi/16) q[26];
cx q[25],q[26];
cx q[25],q[24];
rz(pi/8) q[24];
cx q[25],q[24];
rz(-pi/8) q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[25],q[22];
rz(pi/4) q[22];
cx q[25],q[22];
rz(-pi/4) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-0.046019424) q[25];
cx q[25],q[24];
rz(pi/1024) q[24];
cx q[25],q[24];
rz(-pi/1024) q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[25],q[24];
rz(pi/512) q[24];
cx q[25],q[24];
rz(-pi/512) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[25],q[24];
rz(pi/256) q[24];
cx q[25],q[24];
rz(-pi/256) q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[25],q[22];
rz(pi/128) q[22];
cx q[25],q[22];
rz(-pi/128) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(-0.29452431) q[25];
rz(-pi/16) q[26];
cx q[25],q[26];
rz(pi/32) q[26];
cx q[25],q[26];
cx q[25],q[24];
rz(pi/16) q[24];
cx q[25],q[24];
rz(-pi/16) q[24];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(pi/8) q[16];
cx q[16],q[14];
rz(pi/8) q[14];
cx q[16],q[14];
rz(-pi/8) q[14];
cx q[16],q[19];
rz(pi/4) q[19];
cx q[16],q[19];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(-pi/4) q[19];
rz(-0.14726216) q[25];
rz(-pi/32) q[26];
cx q[25],q[26];
rz(pi/64) q[26];
cx q[25],q[26];
cx q[25],q[24];
rz(pi/32) q[24];
cx q[25],q[24];
rz(-pi/32) q[24];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[22];
cx q[19],q[22];
cx q[22],q[19];
rz(pi/16) q[19];
cx q[19],q[16];
rz(pi/16) q[16];
cx q[19],q[16];
rz(-pi/16) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(pi/8) q[22];
cx q[19],q[22];
cx q[19],q[16];
rz(pi/4) q[16];
cx q[19],q[16];
rz(-pi/4) q[16];
sx q[19];
rz(pi/2) q[19];
rz(-pi/8) q[22];
rz(-pi/64) q[26];
barrier q[21],q[25],q[18],q[13],q[2],q[5],q[11],q[8],q[12],q[17],q[15],q[20],q[14],q[4],q[1],q[7],q[10],q[19],q[22],q[24],q[26],q[23],q[0],q[3],q[9],q[6],q[16];
measure q[13] -> c[0];
measure q[18] -> c[1];
measure q[21] -> c[2];
measure q[23] -> c[3];
measure q[25] -> c[4];
measure q[26] -> c[5];
measure q[24] -> c[6];
measure q[14] -> c[7];
measure q[22] -> c[8];
measure q[16] -> c[9];
measure q[19] -> c[10];
