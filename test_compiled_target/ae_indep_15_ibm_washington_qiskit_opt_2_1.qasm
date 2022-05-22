OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[15];
rz(-3*pi/2) q[28];
sx q[28];
rz(pi/2) q[28];
rz(-3*pi/2) q[35];
sx q[35];
rz(pi/2) q[35];
rz(-3*pi/2) q[44];
sx q[44];
rz(0.83448555) q[44];
rz(-3*pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
rz(-3*pi/2) q[46];
sx q[46];
rz(pi/2) q[46];
rz(-3*pi/2) q[47];
sx q[47];
rz(pi/2) q[47];
rz(-3*pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
rz(-3*pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
rz(-3*pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
rz(-3*pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(-3*pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
rz(-3*pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
rz(-pi) q[65];
sx q[65];
rz(2.2142974) q[65];
sx q[65];
rz(-3*pi/2) q[66];
sx q[66];
rz(pi/2) q[66];
cx q[66],q[65];
sx q[65];
rz(2.2142974) q[65];
sx q[65];
rz(-pi) q[65];
cx q[66],q[65];
rz(-pi) q[65];
sx q[65];
rz(2.2142974) q[65];
sx q[65];
rz(-3*pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[66],q[65];
sx q[65];
rz(1.2870023) q[65];
sx q[65];
rz(-pi) q[65];
cx q[66],q[65];
rz(-pi) q[65];
sx q[65];
rz(1.2870023) q[65];
sx q[65];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
rz(-pi) q[64];
sx q[64];
rz(0.56758825) q[64];
sx q[64];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(-0.023776702) q[60];
sx q[64];
rz(0.56758825) q[64];
sx q[64];
rz(-pi) q[64];
cx q[63],q[64];
sx q[64];
rz(2.0064163) q[64];
sx q[64];
rz(-pi) q[64];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
rz(-pi/2048) q[61];
rz(-pi) q[64];
sx q[64];
rz(2.0064163) q[64];
sx q[64];
cx q[63],q[64];
sx q[64];
rz(0.87124027) q[64];
sx q[64];
rz(-pi) q[64];
cx q[63],q[64];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-pi/1024) q[62];
rz(-pi) q[64];
sx q[64];
rz(0.87123975) q[64];
sx q[64];
cx q[54],q[64];
rz(-pi) q[64];
sx q[64];
rz(1.3991131) q[64];
sx q[64];
cx q[54],q[64];
sx q[64];
rz(1.3991131) q[64];
sx q[64];
rz(-pi) q[64];
cx q[63],q[64];
sx q[64];
rz(0.34336642) q[64];
sx q[64];
rz(-pi) q[64];
cx q[63],q[64];
rz(-pi/256) q[63];
rz(-pi) q[64];
sx q[64];
rz(0.34336645) q[64];
sx q[64];
cx q[65],q[64];
rz(-pi) q[64];
sx q[64];
rz(2.4548618) q[64];
sx q[64];
cx q[65],q[64];
sx q[64];
rz(2.4548597) q[64];
sx q[64];
rz(-pi) q[64];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[44],q[45];
rz(-pi) q[45];
sx q[45];
rz(1.768131) q[45];
sx q[45];
cx q[44],q[45];
sx q[45];
rz(1.7681268) q[45];
sx q[45];
rz(-pi) q[45];
cx q[54],q[45];
rz(-pi) q[45];
sx q[45];
rz(0.39465931) q[45];
sx q[45];
cx q[54],q[45];
sx q[45];
rz(0.39466095) q[45];
sx q[45];
rz(-pi) q[45];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[47],q[46];
sx q[46];
rz(2.352274) q[46];
sx q[46];
rz(-pi) q[46];
cx q[47],q[46];
rz(-pi) q[46];
sx q[46];
rz(2.3522708) q[46];
sx q[46];
cx q[45],q[46];
sx q[46];
rz(1.5629554) q[46];
sx q[46];
rz(-pi) q[46];
cx q[45],q[46];
rz(-pi/8) q[45];
rz(-pi) q[46];
sx q[46];
rz(1.562949) q[46];
sx q[46];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[28],q[35];
rz(-pi) q[35];
sx q[35];
rz(0.01568181) q[35];
sx q[35];
cx q[28],q[35];
sx q[35];
rz(0.015694754) q[35];
sx q[35];
rz(-pi) q[35];
cx q[47],q[35];
sx q[35];
rz(3.110229) q[35];
sx q[35];
rz(-pi) q[35];
cx q[47],q[35];
rz(-pi) q[35];
sx q[35];
rz(3.110229) q[35];
sx q[35];
cx q[28],q[35];
cx q[35],q[28];
cx q[28],q[35];
rz(pi/4) q[35];
rz(pi/2) q[47];
sx q[47];
rz(pi/2) q[47];
cx q[35],q[47];
rz(pi/4) q[47];
cx q[35],q[47];
sx q[35];
rz(pi/2) q[35];
rz(-pi/4) q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[45],q[46];
rz(pi/8) q[46];
cx q[45],q[46];
rz(-pi/8) q[46];
rz(-pi/16) q[47];
cx q[47],q[46];
rz(pi/16) q[46];
cx q[47],q[46];
cx q[35],q[47];
rz(-pi/16) q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
rz(pi/4) q[46];
cx q[47],q[35];
cx q[35],q[47];
rz(pi/8) q[35];
cx q[46],q[47];
rz(pi/4) q[47];
cx q[46],q[47];
sx q[46];
rz(pi/2) q[46];
rz(-pi/4) q[47];
cx q[35],q[47];
rz(pi/8) q[47];
cx q[35],q[47];
rz(-pi/8) q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[35],q[47];
rz(pi/4) q[47];
cx q[35],q[47];
sx q[35];
rz(pi/2) q[35];
rz(-pi/4) q[47];
rz(-pi/32) q[54];
cx q[54],q[45];
rz(pi/32) q[45];
cx q[54],q[45];
rz(-pi/32) q[45];
cx q[44],q[45];
rz(pi/64) q[45];
cx q[44],q[45];
rz(-pi/64) q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(-pi/16) q[45];
cx q[45],q[46];
rz(pi/16) q[46];
cx q[45],q[46];
rz(-pi/16) q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[44],q[45];
rz(pi/32) q[45];
cx q[44],q[45];
rz(-pi/32) q[45];
rz(-pi/8) q[46];
cx q[46],q[47];
rz(pi/8) q[47];
cx q[46],q[47];
rz(-pi/8) q[47];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
rz(pi/4) q[47];
cx q[47],q[35];
rz(pi/4) q[35];
cx q[47],q[35];
rz(-pi/4) q[35];
sx q[47];
rz(pi/2) q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[47],q[35];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
rz(-pi/512) q[54];
rz(-0.76085447) q[65];
cx q[65],q[64];
rz(pi/128) q[64];
cx q[65],q[64];
rz(-pi/128) q[64];
cx q[63],q[64];
rz(pi/256) q[64];
cx q[63],q[64];
rz(-pi/256) q[64];
cx q[54],q[64];
rz(pi/512) q[64];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(-pi/512) q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[54],q[64];
cx q[62],q[63];
rz(pi/1024) q[63];
cx q[62],q[63];
rz(-pi/1024) q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
rz(pi/2048) q[62];
cx q[61],q[62];
rz(-pi/2048) q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[60],q[61];
rz(pi/4096) q[61];
cx q[60],q[61];
rz(-pi/4096) q[61];
rz(-pi/1024) q[62];
rz(-pi/512) q[63];
cx q[64],q[54];
cx q[54],q[64];
rz(-pi/128) q[54];
cx q[65],q[64];
rz(pi/64) q[64];
cx q[65],q[64];
rz(-pi/64) q[64];
cx q[54],q[64];
rz(pi/128) q[64];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[44],q[45];
rz(pi/16) q[45];
cx q[44],q[45];
rz(-pi/16) q[45];
rz(-pi/256) q[54];
rz(-pi/128) q[64];
cx q[54],q[64];
rz(pi/256) q[64];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
rz(-pi/256) q[64];
cx q[63],q[64];
rz(pi/512) q[64];
cx q[63],q[64];
rz(-pi/512) q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[54],q[64];
cx q[62],q[63];
rz(pi/1024) q[63];
cx q[62],q[63];
rz(-pi/1024) q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[60],q[61];
rz(pi/2048) q[61];
cx q[60],q[61];
rz(-pi/2048) q[61];
rz(-pi/512) q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[44],q[45];
rz(pi/8) q[45];
cx q[44],q[45];
rz(-pi/8) q[45];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[35],q[47];
rz(-pi/128) q[46];
cx q[47],q[35];
cx q[35],q[47];
rz(-0.77312632) q[35];
rz(-pi/64) q[54];
cx q[65],q[64];
rz(pi/32) q[64];
cx q[65],q[64];
rz(-pi/32) q[64];
cx q[54],q[64];
rz(pi/64) q[64];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(-pi/64) q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[46],q[45];
rz(pi/128) q[45];
cx q[46],q[45];
rz(-pi/128) q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[35],q[47];
rz(pi/256) q[47];
cx q[35],q[47];
rz(-pi/256) q[47];
rz(-pi/32) q[54];
cx q[65],q[64];
rz(pi/16) q[64];
cx q[65],q[64];
rz(-pi/16) q[64];
cx q[54],q[64];
rz(pi/32) q[64];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(pi/4) q[45];
cx q[45],q[46];
rz(pi/4) q[46];
cx q[45],q[46];
sx q[45];
rz(pi/2) q[45];
rz(-pi/4) q[46];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
rz(-pi/64) q[54];
rz(-pi/32) q[64];
cx q[54],q[64];
rz(pi/64) q[64];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
rz(-pi/64) q[64];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
rz(-pi/16) q[45];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[35],q[47];
rz(pi/128) q[47];
cx q[35],q[47];
rz(-pi/128) q[47];
cx q[65],q[64];
rz(pi/8) q[64];
cx q[65],q[64];
rz(-pi/8) q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[45],q[54];
rz(pi/16) q[54];
cx q[45],q[54];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(pi/8) q[44];
rz(-pi/32) q[45];
rz(-pi/16) q[54];
cx q[45],q[54];
rz(pi/32) q[54];
cx q[45],q[54];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
rz(-pi/32) q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[35],q[47];
rz(-pi/16) q[46];
rz(pi/64) q[47];
cx q[35],q[47];
rz(-pi/64) q[47];
cx q[63],q[64];
rz(pi/512) q[64];
cx q[63],q[64];
rz(-pi/512) q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
rz(pi/1024) q[61];
cx q[60],q[61];
rz(-pi/1024) q[61];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
rz(pi/4) q[64];
cx q[64],q[54];
rz(pi/4) q[54];
cx q[64],q[54];
rz(-pi/4) q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[44],q[45];
rz(pi/8) q[45];
cx q[44],q[45];
rz(-pi/8) q[45];
cx q[46],q[45];
rz(pi/16) q[45];
cx q[46],q[45];
rz(-pi/16) q[45];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[35],q[47];
rz(-pi/8) q[46];
rz(pi/32) q[47];
cx q[35],q[47];
rz(-pi/32) q[47];
sx q[64];
rz(pi/2) q[64];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
rz(-pi/8192) q[64];
cx q[64],q[63];
rz(pi/8192) q[63];
cx q[64],q[63];
rz(-pi/8192) q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(-pi/4096) q[63];
cx q[63],q[62];
rz(pi/4096) q[62];
cx q[63],q[62];
rz(-pi/4096) q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-pi/2048) q[62];
cx q[62],q[61];
rz(pi/2048) q[61];
cx q[62],q[61];
rz(-pi/2048) q[61];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[44],q[45];
rz(pi/4) q[45];
cx q[44],q[45];
sx q[44];
rz(pi/2) q[44];
rz(-pi/4) q[45];
cx q[46],q[45];
rz(pi/8) q[45];
cx q[46],q[45];
rz(-pi/8) q[45];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
rz(pi/4) q[45];
cx q[45],q[44];
rz(pi/4) q[44];
cx q[45],q[44];
rz(-pi/4) q[44];
sx q[45];
rz(pi/2) q[45];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[35],q[47];
rz(pi/16) q[47];
cx q[35],q[47];
rz(-pi/16) q[47];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
rz(-pi/256) q[65];
cx q[65],q[64];
rz(pi/256) q[64];
cx q[65],q[64];
rz(-pi/256) q[64];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
rz(pi/512) q[61];
cx q[60],q[61];
rz(-pi/512) q[61];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-pi/1024) q[62];
cx q[62],q[61];
rz(pi/1024) q[61];
cx q[62],q[61];
rz(-pi/1024) q[61];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
rz(-pi/128) q[64];
cx q[64],q[54];
rz(pi/128) q[54];
cx q[64],q[54];
rz(-pi/128) q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(-pi/64) q[45];
cx q[45],q[46];
rz(pi/64) q[46];
cx q[45],q[46];
rz(-pi/64) q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
rz(-pi/32) q[46];
cx q[46],q[47];
rz(pi/32) q[47];
cx q[46],q[47];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
rz(-pi/32) q[47];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[35],q[47];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
rz(-pi/16) q[46];
rz(pi/8) q[47];
cx q[35],q[47];
rz(-pi/8) q[47];
cx q[46],q[47];
rz(pi/16) q[47];
cx q[46],q[47];
rz(-pi/16) q[47];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
rz(pi/4) q[47];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[60],q[61];
rz(pi/256) q[61];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
rz(-0.073631078) q[43];
cx q[43],q[44];
rz(pi/128) q[44];
cx q[43],q[44];
rz(-pi/128) q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[43],q[44];
rz(pi/64) q[44];
cx q[43],q[44];
rz(-pi/64) q[44];
rz(-pi/256) q[61];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-pi/512) q[62];
cx q[62],q[61];
rz(pi/512) q[61];
cx q[62],q[61];
rz(-pi/512) q[61];
rz(-pi/16384) q[73];
cx q[73],q[66];
rz(pi/16384) q[66];
cx q[73],q[66];
rz(-pi/16384) q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
rz(-pi/8192) q[66];
cx q[66],q[65];
rz(pi/8192) q[65];
cx q[66],q[65];
rz(-pi/8192) q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
rz(-pi/4096) q[65];
cx q[65],q[64];
rz(pi/4096) q[64];
cx q[65],q[64];
rz(-pi/4096) q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
rz(-pi/2048) q[64];
cx q[64],q[63];
rz(pi/2048) q[63];
cx q[64],q[63];
rz(-pi/2048) q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[54],q[64];
rz(-0.0092038847) q[63];
cx q[63],q[62];
rz(pi/1024) q[62];
cx q[63],q[62];
rz(-pi/1024) q[62];
cx q[64],q[54];
cx q[54],q[64];
rz(-pi/256) q[54];
cx q[54],q[45];
rz(pi/256) q[45];
cx q[54],q[45];
rz(-pi/256) q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(-pi/128) q[45];
cx q[45],q[44];
rz(pi/128) q[44];
cx q[45],q[44];
rz(-pi/128) q[44];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
rz(-0.29452431) q[44];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[44],q[45];
rz(pi/32) q[45];
cx q[44],q[45];
rz(-pi/32) q[45];
rz(-pi/64) q[46];
cx q[46],q[45];
rz(pi/64) q[45];
cx q[46],q[45];
rz(-pi/64) q[45];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[47],q[46];
rz(pi/4) q[46];
cx q[47],q[46];
rz(-pi/4) q[46];
sx q[47];
rz(pi/2) q[47];
cx q[47],q[35];
cx q[35],q[47];
rz(pi/4) q[35];
rz(pi/8) q[47];
cx q[47],q[46];
rz(pi/8) q[46];
cx q[47],q[46];
rz(-pi/8) q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[44],q[45];
rz(pi/16) q[45];
cx q[44],q[45];
rz(-pi/16) q[45];
rz(-pi/32) q[46];
cx q[46],q[45];
rz(pi/32) q[45];
cx q[46],q[45];
rz(-pi/32) q[45];
cx q[47],q[35];
rz(-pi/4) q[35];
sx q[47];
rz(pi/2) q[47];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
rz(-pi/16) q[47];
cx q[63],q[64];
rz(pi/512) q[64];
cx q[63],q[64];
rz(-pi/512) q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(-pi/256) q[44];
cx q[44],q[43];
rz(pi/256) q[43];
cx q[44],q[43];
rz(-pi/256) q[43];
rz(-pi/8) q[45];
cx q[45],q[46];
rz(pi/8) q[46];
cx q[45],q[46];
rz(-pi/8) q[46];
cx q[47],q[46];
rz(pi/16) q[46];
cx q[47],q[46];
rz(-pi/16) q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(pi/32) q[44];
rz(-0.17180585) q[45];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
rz(pi/4) q[47];
cx q[47],q[35];
rz(pi/4) q[35];
cx q[47],q[35];
rz(-pi/4) q[35];
sx q[47];
rz(pi/2) q[47];
cx q[47],q[46];
cx q[46],q[47];
rz(pi/4) q[46];
rz(pi/8) q[47];
cx q[47],q[35];
rz(pi/8) q[35];
cx q[47],q[35];
rz(-pi/8) q[35];
cx q[47],q[46];
rz(-pi/4) q[46];
sx q[47];
rz(pi/2) q[47];
cx q[35],q[47];
cx q[47],q[35];
cx q[35],q[47];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
rz(pi/128) q[54];
cx q[45],q[54];
rz(-pi/128) q[54];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[45],q[54];
rz(pi/64) q[54];
cx q[45],q[54];
cx q[45],q[44];
rz(-pi/32) q[44];
cx q[46],q[45];
cx q[45],q[46];
rz(pi/8) q[45];
rz(-3*pi/16) q[46];
cx q[46],q[47];
rz(pi/16) q[47];
cx q[46],q[47];
cx q[46],q[45];
rz(-pi/8) q[45];
rz(-pi/16) q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
rz(pi/4) q[47];
cx q[47],q[35];
rz(pi/4) q[35];
cx q[47],q[35];
rz(-pi/4) q[35];
sx q[47];
rz(pi/2) q[47];
rz(-pi/64) q[54];
barrier q[18],q[82],q[15],q[79],q[24],q[88],q[33],q[97],q[41],q[39],q[106],q[51],q[48],q[115],q[112],q[57],q[2],q[121],q[47],q[11],q[75],q[8],q[72],q[17],q[81],q[26],q[90],q[73],q[99],q[63],q[53],q[108],q[105],q[50],q[114],q[59],q[4],q[123],q[68],q[1],q[13],q[28],q[77],q[10],q[74],q[19],q[83],q[66],q[92],q[37],q[34],q[101],q[98],q[42],q[107],q[52],q[116],q[46],q[6],q[125],q[70],q[3],q[67],q[12],q[76],q[21],q[85],q[30],q[27],q[94],q[91],q[36],q[103],q[100],q[62],q[109],q[54],q[118],q[64],q[44],q[5],q[124],q[69],q[14],q[78],q[23],q[20],q[87],q[32],q[84],q[29],q[96],q[93],q[38],q[102],q[61],q[111],q[56],q[120],q[60],q[117],q[45],q[7],q[126],q[71],q[16],q[80],q[25],q[22],q[89],q[86],q[31],q[95],q[40],q[104],q[49],q[113],q[65],q[58],q[110],q[122],q[55],q[0],q[119],q[43],q[9],q[35];
measure q[47] -> meas[0];
measure q[35] -> meas[1];
measure q[45] -> meas[2];
measure q[46] -> meas[3];
measure q[44] -> meas[4];
measure q[54] -> meas[5];
measure q[64] -> meas[6];
measure q[43] -> meas[7];
measure q[63] -> meas[8];
measure q[62] -> meas[9];
measure q[61] -> meas[10];
measure q[65] -> meas[11];
measure q[66] -> meas[12];
measure q[73] -> meas[13];
measure q[28] -> meas[14];
