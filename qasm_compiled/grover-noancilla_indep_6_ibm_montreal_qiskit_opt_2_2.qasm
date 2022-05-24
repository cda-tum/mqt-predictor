OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[6];
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
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[18];
sx q[18];
rz(1.6689711) q[18];
cx q[18],q[15];
rz(-pi/32) q[15];
cx q[18],q[15];
rz(pi/32) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[18],q[15];
rz(-pi/32) q[15];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[15],q[12];
rz(-pi/32) q[12];
cx q[18],q[15];
rz(pi/32) q[15];
cx q[15],q[12];
rz(-pi/32) q[12];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[15],q[12];
rz(-pi/32) q[12];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[12],q[13];
cx q[12],q[15];
rz(pi/32) q[13];
cx q[15],q[12];
cx q[12],q[15];
cx q[18],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(-pi/32) q[12];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
cx q[12],q[15];
rz(-pi/32) q[13];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/32) q[14];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
cx q[12],q[13];
rz(pi/32) q[13];
rz(-pi/32) q[14];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/32) q[14];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
cx q[12],q[13];
rz(pi/32) q[13];
rz(-pi/32) q[14];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/32) q[14];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
cx q[12],q[13];
rz(pi/32) q[13];
rz(-pi/32) q[14];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/32) q[14];
cx q[14],q[16];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/32) q[14];
rz(pi/32) q[16];
cx q[14],q[16];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(pi/32) q[16];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/32) q[14];
cx q[14],q[16];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[15],q[18];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/32) q[14];
rz(pi/32) q[16];
cx q[14],q[16];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/32) q[16];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[14],q[16];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/32) q[14];
rz(pi/32) q[16];
cx q[14],q[16];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/32) q[16];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[14],q[16];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
rz(-pi/2) q[12];
sx q[12];
rz(-3*pi/4) q[12];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/32) q[14];
rz(pi/32) q[16];
cx q[14],q[16];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[12],q[13];
rz(-pi/4) q[13];
cx q[12],q[13];
sx q[12];
rz(3*pi/4) q[12];
rz(pi/4) q[13];
rz(-1.4726216) q[14];
cx q[15],q[12];
rz(pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[15],q[18];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
rz(pi/4) q[12];
rz(pi/32) q[16];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
rz(-pi/4) q[12];
cx q[13],q[12];
rz(pi/4) q[12];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[15],q[12];
rz(pi/4) q[12];
sx q[12];
rz(3*pi/4) q[12];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[12];
rz(pi/4) q[12];
sx q[12];
rz(-5*pi/4) q[12];
cx q[12],q[13];
rz(pi/4) q[13];
cx q[12],q[13];
sx q[12];
rz(3*pi/4) q[12];
rz(-pi/4) q[13];
cx q[15],q[12];
rz(pi/4) q[12];
sx q[12];
rz(3*pi/4) q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[18],q[15];
rz(-pi/4) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
rz(pi/4) q[13];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[13];
rz(-pi/4) q[13];
cx q[14],q[13];
rz(-3*pi/2) q[13];
sx q[13];
rz(3*pi/4) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[15];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/16) q[13];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
rz(-pi/16) q[12];
cx q[13],q[12];
rz(pi/16) q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-pi/16) q[13];
cx q[13],q[12];
rz(pi/16) q[12];
cx q[13],q[12];
rz(-pi/16) q[12];
cx q[14],q[13];
rz(pi/16) q[13];
cx q[13],q[12];
rz(-pi/16) q[12];
cx q[13],q[12];
rz(pi/16) q[12];
cx q[18],q[15];
rz(3*pi/4) q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[13],q[12];
rz(-pi/16) q[12];
cx q[12],q[15];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/16) q[15];
cx q[12],q[15];
cx q[13],q[12];
rz(pi/16) q[12];
rz(-pi/16) q[15];
cx q[12],q[15];
rz(-pi/16) q[15];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
rz(-pi/16) q[13];
rz(-3*pi/2) q[14];
sx q[14];
rz(-pi/2) q[14];
cx q[14],q[16];
rz(pi/16) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
rz(pi/16) q[12];
cx q[13],q[12];
rz(-pi/16) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[15],q[12];
rz(-4.5160394) q[12];
cx q[12],q[13];
rz(-pi/16) q[13];
cx q[12],q[13];
sx q[12];
rz(-pi/2) q[12];
rz(9*pi/16) q[13];
sx q[13];
rz(-1.4726216) q[13];
rz(-3*pi/2) q[15];
sx q[15];
rz(-pi/2) q[15];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/32) q[14];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/32) q[14];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[15];
rz(-pi/32) q[13];
rz(pi/32) q[14];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/32) q[14];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
cx q[12],q[13];
rz(pi/32) q[13];
rz(-pi/32) q[14];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/32) q[14];
cx q[14],q[16];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[15];
rz(-pi/32) q[14];
cx q[15],q[12];
cx q[12],q[15];
rz(pi/32) q[16];
cx q[14],q[16];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
rz(-pi/32) q[14];
rz(pi/32) q[16];
cx q[14],q[16];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[12],q[13];
rz(-pi/32) q[13];
rz(pi/32) q[14];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(-pi/32) q[12];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[14],q[13];
rz(-pi/32) q[13];
cx q[13],q[12];
rz(pi/32) q[12];
cx q[13],q[12];
rz(-pi/32) q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[16];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[16],q[14];
cx q[14],q[16];
cx q[14],q[13];
rz(-pi/32) q[13];
cx q[13],q[12];
rz(pi/32) q[12];
cx q[13],q[12];
rz(-pi/32) q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
rz(-pi/32) q[13];
cx q[13],q[12];
rz(pi/32) q[12];
cx q[13],q[12];
rz(-pi/32) q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[12];
rz(-pi/32) q[12];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[12],q[13];
rz(-pi/32) q[13];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[14],q[13];
rz(-pi/32) q[13];
cx q[13],q[12];
rz(pi/32) q[12];
cx q[13],q[12];
rz(-pi/32) q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/32) q[12];
cx q[12],q[15];
rz(-pi/32) q[15];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/32) q[15];
cx q[16],q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/32) q[12];
cx q[12],q[15];
rz(pi/32) q[15];
cx q[12],q[15];
cx q[13],q[12];
rz(pi/32) q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-pi/32) q[15];
cx q[12],q[15];
rz(-pi/32) q[15];
cx q[12],q[15];
cx q[13],q[12];
rz(-pi/32) q[12];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-5*pi/4) q[14];
rz(pi/32) q[15];
cx q[12],q[15];
rz(pi/32) q[15];
cx q[12],q[15];
cx q[13],q[12];
rz(-1.4726216) q[12];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[13];
rz(-pi/4) q[13];
cx q[14],q[13];
rz(pi/4) q[13];
sx q[14];
rz(3*pi/4) q[14];
rz(-pi/32) q[15];
cx q[12],q[15];
rz(-pi/32) q[15];
cx q[12],q[15];
sx q[12];
rz(pi/2) q[12];
rz(pi/32) q[15];
rz(-pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[14];
rz(pi/4) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
rz(pi/4) q[13];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[13];
rz(-pi/4) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[15],q[12];
rz(pi/4) q[12];
cx q[13],q[12];
rz(pi/4) q[12];
sx q[12];
rz(3*pi/4) q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
rz(pi/4) q[13];
sx q[13];
rz(-5*pi/4) q[13];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/4) q[14];
cx q[13],q[14];
sx q[13];
rz(3*pi/4) q[13];
rz(-pi/4) q[14];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
rz(pi/4) q[13];
sx q[13];
rz(3*pi/4) q[13];
cx q[12],q[13];
rz(-pi/4) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[15],q[12];
rz(pi/4) q[12];
cx q[13],q[12];
rz(-pi/4) q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[14],q[16];
cx q[15],q[12];
rz(-3*pi/2) q[12];
sx q[12];
rz(3*pi/4) q[12];
cx q[13],q[12];
rz(3*pi/4) q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(pi/16) q[12];
cx q[15],q[18];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
rz(-pi/16) q[13];
cx q[12],q[13];
rz(pi/16) q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(-pi/16) q[14];
cx q[14],q[13];
rz(pi/16) q[13];
cx q[14],q[13];
rz(-pi/16) q[13];
cx q[16],q[14];
rz(pi/16) q[14];
cx q[14],q[13];
rz(-pi/16) q[13];
cx q[14],q[13];
rz(pi/16) q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
rz(-pi/16) q[13];
cx q[13],q[12];
rz(pi/16) q[12];
cx q[13],q[12];
rz(-pi/16) q[12];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[14],q[13];
rz(pi/16) q[13];
cx q[13],q[12];
rz(-pi/16) q[12];
cx q[13],q[12];
rz(pi/16) q[12];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
rz(-pi/16) q[13];
cx q[13],q[12];
rz(pi/16) q[12];
cx q[13],q[12];
rz(-pi/16) q[12];
rz(-3*pi/2) q[14];
sx q[14];
rz(-pi/2) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[16],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-4.5160394) q[13];
cx q[13],q[12];
rz(-pi/16) q[12];
cx q[13],q[12];
rz(9*pi/16) q[12];
sx q[12];
rz(-1.4726216) q[12];
sx q[13];
rz(-pi/2) q[13];
rz(-3*pi/2) q[16];
sx q[16];
rz(-pi/2) q[16];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
rz(-pi/32) q[15];
cx q[12],q[15];
rz(pi/32) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
rz(-pi/32) q[15];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[15],q[12];
rz(-pi/32) q[12];
cx q[18],q[15];
rz(pi/32) q[15];
cx q[15],q[12];
rz(-pi/32) q[12];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[15],q[12];
rz(-pi/32) q[12];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[12],q[13];
cx q[12],q[15];
rz(pi/32) q[13];
cx q[15],q[12];
cx q[12],q[15];
cx q[18],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(-pi/32) q[12];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
cx q[12],q[15];
rz(-pi/32) q[13];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/32) q[14];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
cx q[12],q[13];
rz(pi/32) q[13];
rz(-pi/32) q[14];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/32) q[14];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
cx q[12],q[13];
rz(pi/32) q[13];
rz(-pi/32) q[14];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/32) q[14];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
cx q[12],q[13];
rz(pi/32) q[13];
rz(-pi/32) q[14];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/32) q[14];
cx q[14],q[16];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/32) q[14];
rz(pi/32) q[16];
cx q[14],q[16];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(pi/32) q[16];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/32) q[14];
cx q[14],q[16];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[15],q[18];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/32) q[14];
rz(pi/32) q[16];
cx q[14],q[16];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/32) q[16];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[14],q[16];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/32) q[14];
rz(pi/32) q[16];
cx q[14],q[16];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/32) q[16];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[14],q[16];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
rz(-pi/2) q[12];
sx q[12];
rz(-3*pi/4) q[12];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/32) q[14];
rz(pi/32) q[16];
cx q[14],q[16];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[12],q[13];
rz(-pi/4) q[13];
cx q[12],q[13];
sx q[12];
rz(3*pi/4) q[12];
rz(pi/4) q[13];
rz(-1.4726216) q[14];
cx q[15],q[12];
rz(pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[15],q[18];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
rz(pi/4) q[12];
rz(pi/32) q[16];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
rz(-pi/4) q[12];
cx q[13],q[12];
rz(pi/4) q[12];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[15],q[12];
rz(pi/4) q[12];
sx q[12];
rz(3*pi/4) q[12];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[12];
rz(pi/4) q[12];
sx q[12];
rz(-5*pi/4) q[12];
cx q[12],q[13];
rz(pi/4) q[13];
cx q[12],q[13];
sx q[12];
rz(3*pi/4) q[12];
rz(-pi/4) q[13];
cx q[15],q[12];
rz(pi/4) q[12];
sx q[12];
rz(3*pi/4) q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[18],q[15];
rz(-pi/4) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
rz(pi/4) q[13];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[13];
rz(-pi/4) q[13];
cx q[14],q[13];
rz(-3*pi/2) q[13];
sx q[13];
rz(3*pi/4) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[15];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/16) q[13];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
rz(-pi/16) q[12];
cx q[13],q[12];
rz(pi/16) q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-pi/16) q[13];
cx q[13],q[12];
rz(pi/16) q[12];
cx q[13],q[12];
rz(-pi/16) q[12];
cx q[14],q[13];
rz(pi/16) q[13];
cx q[13],q[12];
rz(-pi/16) q[12];
cx q[13],q[12];
rz(pi/16) q[12];
cx q[18],q[15];
rz(3*pi/4) q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[13],q[12];
rz(-pi/16) q[12];
cx q[12],q[15];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/16) q[15];
cx q[12],q[15];
cx q[13],q[12];
rz(pi/16) q[12];
rz(-pi/16) q[15];
cx q[12],q[15];
rz(-pi/16) q[15];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
rz(-pi/16) q[13];
rz(-3*pi/2) q[14];
sx q[14];
rz(-pi/2) q[14];
cx q[14],q[16];
rz(pi/16) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
rz(pi/16) q[12];
cx q[13],q[12];
rz(-pi/16) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[15],q[12];
rz(-4.5160394) q[12];
cx q[12],q[13];
rz(-pi/16) q[13];
cx q[12],q[13];
sx q[12];
rz(-pi/2) q[12];
rz(9*pi/16) q[13];
sx q[13];
rz(-1.4726216) q[13];
rz(-3*pi/2) q[15];
sx q[15];
rz(-pi/2) q[15];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/32) q[14];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/32) q[14];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[15];
rz(-pi/32) q[13];
rz(pi/32) q[14];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[15],q[12];
cx q[12],q[15];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/32) q[14];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
cx q[12],q[13];
rz(pi/32) q[13];
rz(-pi/32) q[14];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/32) q[14];
cx q[14],q[16];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[15];
rz(-pi/32) q[14];
cx q[15],q[12];
cx q[12],q[15];
rz(pi/32) q[16];
cx q[14],q[16];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/32) q[14];
rz(-pi/32) q[16];
cx q[14],q[16];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
rz(-pi/32) q[14];
rz(pi/32) q[16];
cx q[14],q[16];
rz(pi/32) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(-pi/32) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[12],q[13];
rz(-pi/32) q[13];
rz(pi/32) q[14];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[12],q[13];
rz(-pi/32) q[13];
cx q[12],q[13];
rz(pi/32) q[13];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-pi/32) q[15];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[15],q[12];
rz(-pi/32) q[12];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[18],q[15];
rz(pi/32) q[15];
cx q[15],q[12];
rz(-pi/32) q[12];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
rz(-pi/32) q[12];
cx q[12],q[15];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/32) q[15];
cx q[12],q[15];
rz(-pi/32) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[18],q[15];
rz(pi/32) q[15];
cx q[15],q[12];
rz(-pi/32) q[12];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
rz(-pi/32) q[12];
cx q[12],q[15];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[14],q[16];
rz(pi/32) q[15];
cx q[12],q[15];
rz(-pi/32) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[16],q[14];
cx q[14],q[16];
cx q[18],q[15];
rz(pi/32) q[15];
cx q[15],q[12];
rz(-pi/32) q[12];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
rz(-pi/32) q[12];
cx q[12],q[15];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/32) q[15];
cx q[12],q[15];
rz(-pi/32) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[18],q[15];
rz(pi/32) q[15];
cx q[15],q[12];
rz(-pi/32) q[12];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
rz(-pi/32) q[12];
cx q[12],q[15];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/32) q[15];
cx q[12],q[15];
rz(-pi/32) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[18],q[15];
rz(pi/32) q[15];
cx q[15],q[12];
rz(-pi/32) q[12];
cx q[15],q[12];
rz(pi/32) q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
rz(-pi/32) q[12];
cx q[12],q[15];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/32) q[15];
cx q[12],q[15];
rz(-pi/32) q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(pi/32) q[15];
cx q[15],q[18];
rz(-pi/32) q[18];
cx q[15],q[18];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[13],q[12];
cx q[12],q[15];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[15],q[12];
cx q[12],q[15];
rz(-pi/32) q[15];
rz(pi/32) q[18];
cx q[15],q[18];
rz(pi/32) q[18];
cx q[15],q[18];
cx q[12],q[15];
rz(pi/32) q[15];
rz(-pi/32) q[18];
cx q[15],q[18];
rz(-pi/32) q[18];
cx q[15],q[18];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[13],q[12];
cx q[12],q[15];
rz(-pi/2) q[13];
sx q[13];
rz(-3*pi/4) q[13];
cx q[15],q[12];
cx q[12],q[15];
rz(-pi/32) q[15];
rz(pi/32) q[18];
cx q[15],q[18];
rz(pi/32) q[18];
cx q[15],q[18];
cx q[12],q[15];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[13],q[12];
rz(-pi/4) q[12];
cx q[13],q[12];
rz(pi/4) q[12];
sx q[13];
rz(3*pi/4) q[13];
cx q[14],q[13];
rz(pi/4) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[16];
rz(-1.4726216) q[15];
cx q[16],q[14];
cx q[14],q[16];
rz(-pi/32) q[18];
cx q[15],q[18];
rz(-pi/32) q[18];
cx q[15],q[18];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[13];
rz(pi/4) q[13];
cx q[14],q[13];
rz(-pi/4) q[13];
cx q[12],q[13];
cx q[12],q[15];
rz(pi/4) q[13];
cx q[14],q[13];
rz(pi/4) q[13];
sx q[13];
rz(3*pi/4) q[13];
cx q[15],q[12];
cx q[12],q[15];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
rz(pi/4) q[13];
sx q[13];
rz(-5*pi/4) q[13];
cx q[13],q[12];
rz(pi/4) q[12];
cx q[13],q[12];
rz(-pi/4) q[12];
sx q[13];
rz(3*pi/4) q[13];
cx q[14],q[13];
rz(pi/4) q[13];
sx q[13];
rz(3*pi/4) q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[16],q[14];
rz(-pi/4) q[14];
cx q[13],q[14];
rz(pi/4) q[14];
cx q[16],q[14];
rz(-pi/4) q[14];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/16) q[12];
cx q[12],q[15];
rz(-3*pi/2) q[14];
sx q[14];
rz(3*pi/4) q[14];
cx q[13],q[14];
rz(3*pi/4) q[14];
cx q[14],q[16];
rz(-pi/16) q[15];
cx q[12],q[15];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/16) q[15];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/16) q[12];
cx q[12],q[15];
rz(pi/16) q[15];
cx q[12],q[15];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[14],q[13];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/16) q[12];
rz(-pi/16) q[13];
rz(-pi/16) q[15];
cx q[12],q[15];
rz(-pi/16) q[15];
cx q[12],q[15];
rz(pi/16) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
rz(pi/16) q[12];
cx q[13],q[12];
rz(-pi/16) q[12];
cx q[14],q[13];
rz(pi/16) q[13];
cx q[13],q[12];
rz(-pi/16) q[12];
cx q[13],q[12];
rz(pi/16) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[15],q[12];
rz(-pi/16) q[12];
cx q[12],q[13];
rz(pi/16) q[13];
cx q[12],q[13];
rz(-pi/16) q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
rz(-4.5160394) q[13];
cx q[13],q[12];
rz(-pi/16) q[12];
cx q[13],q[12];
rz(9*pi/16) q[12];
sx q[12];
rz(-pi/2) q[12];
sx q[13];
rz(-pi/2) q[13];
rz(-3*pi/2) q[14];
sx q[14];
rz(-pi/2) q[14];
rz(-3*pi/2) q[15];
sx q[15];
rz(-pi/2) q[15];
rz(pi/32) q[18];
barrier q[14],q[22],q[19],q[25],q[2],q[8],q[5],q[11],q[15],q[20],q[17],q[23],q[26],q[3],q[0],q[6],q[9],q[18],q[16],q[12],q[21],q[1],q[24],q[4],q[10],q[7],q[13];
measure q[14] -> meas[0];
measure q[15] -> meas[1];
measure q[13] -> meas[2];
measure q[16] -> meas[3];
measure q[12] -> meas[4];
measure q[18] -> meas[5];
