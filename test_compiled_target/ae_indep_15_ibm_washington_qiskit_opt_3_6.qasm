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
rz(-3*pi/2) q[43];
sx q[43];
rz(pi/2) q[43];
rz(-3*pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
rz(-pi) q[45];
sx q[45];
rz(-pi/2) q[45];
rz(-3*pi/2) q[46];
sx q[46];
rz(pi/2) q[46];
rz(-3*pi/2) q[47];
sx q[47];
rz(pi/2) q[47];
rz(-3*pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
rz(-3*pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
rz(-3*pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(-3*pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
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
rz(-0.011504856) q[61];
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
rz(3.1400587) q[62];
sx q[62];
rz(-pi) q[64];
sx q[64];
rz(2.0064163) q[64];
sx q[64];
cx q[65],q[64];
sx q[64];
rz(0.87124027) q[64];
sx q[64];
rz(-pi) q[64];
cx q[65],q[64];
rz(-pi) q[64];
sx q[64];
rz(0.87123975) q[64];
sx q[64];
cx q[63],q[64];
rz(-pi) q[64];
sx q[64];
rz(1.3991131) q[64];
sx q[64];
cx q[63],q[64];
rz(-pi/512) q[63];
sx q[64];
rz(1.3991131) q[64];
sx q[64];
rz(-pi) q[64];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
rz(pi/2) q[54];
sx q[54];
cx q[54],q[45];
rz(-pi/2) q[45];
sx q[54];
rz(-pi) q[54];
cx q[54],q[45];
rz(1.2274299) q[45];
sx q[54];
cx q[54],q[45];
rz(-1.2274299) q[45];
sx q[45];
cx q[44],q[45];
rz(-pi) q[45];
sx q[45];
rz(2.4548618) q[45];
sx q[45];
cx q[44],q[45];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
rz(-0.17180585) q[43];
sx q[45];
rz(2.4548597) q[45];
sx q[45];
rz(-pi) q[45];
rz(pi/2) q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[54],q[45];
rz(-pi) q[45];
sx q[45];
rz(1.768131) q[45];
sx q[45];
cx q[54],q[45];
sx q[45];
rz(1.7681268) q[45];
sx q[45];
rz(-pi) q[45];
cx q[44],q[45];
rz(-pi) q[45];
sx q[45];
rz(0.39465931) q[45];
sx q[45];
cx q[44],q[45];
rz(-pi/32) q[44];
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
cx q[44],q[45];
rz(pi/32) q[45];
cx q[44],q[45];
rz(-pi/32) q[45];
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
rz(-pi/4) q[46];
sx q[46];
rz(-pi/2) q[46];
rz(pi/4) q[47];
cx q[35],q[47];
sx q[35];
rz(pi/2) q[35];
rz(-pi/4) q[47];
sx q[47];
rz(pi/2) q[47];
rz(-0.14726216) q[54];
cx q[54],q[45];
rz(pi/64) q[45];
cx q[54],q[45];
rz(-pi/64) q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[43],q[44];
rz(pi/128) q[44];
cx q[43],q[44];
rz(-pi/128) q[44];
rz(3*pi/16) q[45];
sx q[45];
rz(-pi) q[45];
cx q[46],q[45];
rz(-pi/2) q[45];
sx q[46];
rz(-pi) q[46];
cx q[46],q[45];
rz(7*pi/16) q[45];
sx q[46];
cx q[46],q[45];
x q[45];
rz(-13*pi/16) q[45];
rz(pi/8) q[46];
sx q[46];
cx q[47],q[46];
rz(pi/2) q[46];
sx q[47];
rz(-pi) q[47];
cx q[47],q[46];
rz(3*pi/8) q[46];
sx q[47];
cx q[47],q[46];
rz(-0.33078919) q[46];
sx q[46];
rz(pi/2) q[46];
rz(3*pi/4) q[47];
cx q[47],q[35];
rz(pi/4) q[35];
cx q[47],q[35];
rz(-pi/4) q[35];
sx q[47];
rz(pi/2) q[47];
cx q[54],q[45];
rz(pi/32) q[45];
cx q[54],q[45];
rz(-pi/32) q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[43],q[44];
rz(pi/64) q[44];
cx q[43],q[44];
rz(-pi/64) q[44];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(-0.91983781) q[45];
sx q[45];
rz(-pi) q[45];
cx q[46],q[45];
rz(-pi/2) q[45];
sx q[46];
rz(-pi) q[46];
cx q[46],q[45];
rz(7*pi/16) q[45];
sx q[46];
cx q[46],q[45];
x q[45];
rz(2.2217548) q[45];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[43],q[44];
rz(pi/32) q[44];
cx q[43],q[44];
rz(-pi/32) q[44];
sx q[44];
rz(-pi) q[44];
sx q[45];
rz(-pi) q[45];
rz(-0.84730806) q[46];
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
rz(-pi/2) q[35];
sx q[35];
rz(-pi) q[35];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
sx q[47];
rz(pi/2) q[47];
sx q[54];
rz(-pi) q[54];
rz(-pi/256) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[64],q[54];
rz(-pi/2) q[54];
sx q[64];
rz(-pi) q[64];
cx q[64],q[54];
rz(1.5585245) q[54];
sx q[64];
cx q[64],q[54];
rz(pi/128) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[54],q[45];
rz(-pi/2) q[45];
sx q[54];
rz(-pi) q[54];
cx q[54],q[45];
rz(1.5462526) q[45];
sx q[54];
cx q[54],q[45];
rz(pi/64) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[45],q[44];
rz(-pi/2) q[44];
sx q[45];
rz(-pi) q[45];
cx q[45],q[44];
rz(1.5217089) q[44];
sx q[45];
cx q[45],q[44];
rz(-pi) q[44];
x q[44];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
rz(-0.29452431) q[43];
rz(-0.91983781) q[44];
sx q[44];
rz(-pi) q[44];
rz(-1.6198837) q[45];
sx q[45];
rz(-pi) q[45];
rz(-1.59534) q[54];
sx q[54];
rz(-pi) q[54];
rz(-1.5830682) q[64];
cx q[63],q[64];
rz(pi/512) q[64];
cx q[63],q[64];
rz(-pi/512) q[64];
rz(-0.021475731) q[65];
cx q[65],q[64];
rz(pi/1024) q[64];
cx q[65],q[64];
rz(-pi/1024) q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(-pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[63],q[62];
rz(-pi/2) q[62];
sx q[63];
rz(-pi) q[63];
cx q[63],q[62];
rz(1.5692623) q[62];
sx q[63];
cx q[63],q[62];
rz(1.5692623) q[62];
cx q[61],q[62];
rz(pi/4096) q[62];
cx q[61],q[62];
rz(-pi/4096) q[62];
rz(-1.5677284) q[63];
sx q[63];
rz(-pi) q[63];
rz(-pi/256) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[64],q[54];
rz(-pi/2) q[54];
sx q[64];
rz(-pi) q[64];
cx q[64],q[54];
rz(1.5585245) q[54];
sx q[64];
cx q[64],q[54];
rz(pi/128) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[54],q[45];
rz(-pi/2) q[45];
sx q[54];
rz(-pi) q[54];
cx q[54],q[45];
rz(1.5462526) q[45];
sx q[54];
cx q[54],q[45];
rz(-pi) q[45];
x q[45];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
rz(0.72348827) q[45];
sx q[45];
rz(-pi/2) q[45];
cx q[45],q[44];
rz(-pi/2) q[44];
sx q[45];
rz(-pi) q[45];
cx q[45],q[44];
rz(7*pi/16) q[44];
sx q[45];
cx q[45],q[44];
x q[44];
rz(2.2217548) q[44];
cx q[43],q[44];
rz(pi/32) q[44];
cx q[43],q[44];
rz(-pi/32) q[44];
rz(-0.84730806) q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
rz(-pi/64) q[45];
cx q[45],q[44];
rz(pi/64) q[44];
cx q[45],q[44];
rz(-pi/64) q[44];
rz(7*pi/8) q[46];
sx q[46];
cx q[47],q[46];
rz(pi/2) q[46];
sx q[47];
rz(-pi) q[47];
cx q[47],q[46];
rz(3*pi/8) q[46];
sx q[47];
cx q[47],q[46];
x q[46];
rz(7*pi/8) q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[43],q[44];
rz(pi/16) q[44];
cx q[43],q[44];
rz(-pi/16) q[44];
rz(-pi/32) q[46];
rz(3*pi/4) q[47];
sx q[47];
rz(-pi/2) q[47];
cx q[47],q[35];
rz(pi/2) q[35];
sx q[47];
rz(-pi) q[47];
cx q[47],q[35];
rz(pi/4) q[35];
sx q[47];
cx q[47],q[35];
sx q[35];
rz(pi/2) q[35];
x q[47];
rz(-pi/4) q[47];
rz(-1.59534) q[54];
rz(-1.5830682) q[64];
cx q[65],q[64];
rz(pi/512) q[64];
cx q[65],q[64];
rz(-1.5769322) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[64],q[63];
rz(-pi/2) q[63];
sx q[64];
rz(-pi) q[64];
cx q[64],q[63];
rz(1.5677284) q[63];
sx q[64];
cx q[64],q[63];
rz(1.5677284) q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
rz(pi/2048) q[62];
cx q[61],q[62];
rz(-pi/2048) q[62];
x q[64];
rz(pi/2) q[64];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
rz(-pi/512) q[54];
cx q[65],q[64];
rz(pi/256) q[64];
cx q[65],q[64];
rz(-pi/256) q[64];
cx q[54],q[64];
rz(pi/512) q[64];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(-pi/256) q[45];
rz(-pi/512) q[64];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
rz(pi/1024) q[62];
cx q[61],q[62];
rz(-pi/1024) q[62];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
rz(-pi/128) q[64];
cx q[64],q[54];
rz(pi/128) q[54];
cx q[64],q[54];
rz(-pi/128) q[54];
cx q[45],q[54];
rz(pi/256) q[54];
cx q[45],q[54];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(-pi/128) q[44];
cx q[46],q[45];
rz(pi/32) q[45];
cx q[46],q[45];
rz(-pi/32) q[45];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
rz(-pi/16) q[47];
rz(-pi/256) q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
rz(-pi/64) q[54];
cx q[54],q[45];
rz(pi/64) q[45];
cx q[54],q[45];
rz(-pi/64) q[45];
cx q[44],q[45];
rz(pi/128) q[45];
cx q[44],q[45];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
rz(-0.3436117) q[43];
rz(-pi/128) q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(0.68722339) q[44];
sx q[44];
rz(-pi) q[44];
rz(-pi/8) q[45];
cx q[45],q[46];
rz(pi/8) q[46];
cx q[45],q[46];
rz(-pi/8) q[46];
cx q[47],q[46];
rz(pi/16) q[46];
cx q[47],q[46];
cx q[35],q[47];
rz(-pi/16) q[46];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
rz(-pi/4) q[45];
sx q[45];
rz(-pi/2) q[45];
cx q[45],q[44];
rz(-pi/2) q[44];
sx q[45];
rz(-pi) q[45];
cx q[45],q[44];
rz(1.4726216) q[44];
sx q[45];
cx q[45],q[44];
x q[44];
rz(-2.4543693) q[44];
cx q[43],q[44];
rz(pi/64) q[44];
cx q[43],q[44];
rz(-pi/64) q[44];
rz(-9*pi/16) q[45];
sx q[45];
rz(-pi) q[45];
rz(pi/4) q[46];
cx q[47],q[35];
cx q[35],q[47];
rz(-3*pi/8) q[35];
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
rz(-pi/4) q[46];
sx q[46];
rz(-pi/2) q[46];
cx q[46],q[45];
rz(-pi/2) q[45];
sx q[46];
rz(-pi) q[46];
cx q[46],q[45];
rz(7*pi/16) q[45];
sx q[46];
cx q[46],q[45];
x q[45];
rz(-13*pi/16) q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[43],q[44];
rz(pi/32) q[44];
cx q[43],q[44];
rz(-pi/32) q[44];
rz(pi/8) q[46];
sx q[46];
rz(pi/4) q[47];
cx q[35],q[47];
x q[35];
rz(pi/2) q[35];
rz(-pi/4) q[47];
sx q[47];
rz(pi/2) q[47];
cx q[47],q[46];
rz(pi/2) q[46];
sx q[47];
rz(-pi) q[47];
cx q[47],q[46];
rz(3*pi/8) q[46];
sx q[47];
cx q[47],q[46];
x q[46];
rz(7*pi/8) q[46];
rz(3*pi/4) q[47];
sx q[47];
rz(-pi/2) q[47];
cx q[47],q[35];
rz(pi/2) q[35];
sx q[47];
rz(-pi) q[47];
cx q[47],q[35];
rz(pi/4) q[35];
sx q[47];
cx q[47],q[35];
sx q[35];
rz(pi/2) q[35];
x q[47];
rz(-pi/4) q[47];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
rz(pi/512) q[62];
cx q[61],q[62];
rz(-pi/512) q[62];
rz(-pi) q[63];
sx q[63];
rz(pi/4) q[64];
sx q[64];
rz(-pi) q[64];
rz(pi/4) q[65];
sx q[65];
rz(-pi) q[65];
rz(-0.78578166) q[66];
sx q[66];
rz(-pi/2) q[66];
cx q[66],q[65];
rz(-pi/2) q[65];
sx q[66];
rz(-pi) q[66];
cx q[66],q[65];
rz(1.5704128) q[65];
sx q[66];
cx q[66],q[65];
rz(pi/4096) q[65];
sx q[65];
rz(pi/2) q[65];
cx q[65],q[64];
rz(-pi/2) q[64];
sx q[65];
rz(-pi) q[65];
cx q[65],q[64];
rz(1.5700293) q[64];
sx q[65];
cx q[65],q[64];
rz(0.78693214) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[64],q[63];
rz(-pi/2) q[63];
sx q[64];
rz(-pi) q[64];
cx q[64],q[63];
rz(1.5692623) q[63];
sx q[64];
cx q[64],q[63];
rz(1.5677284) q[63];
cx q[63],q[62];
rz(pi/1024) q[62];
cx q[63],q[62];
rz(-pi/1024) q[62];
rz(-0.78386418) q[64];
sx q[64];
rz(-pi) q[64];
rz(-1.5715633) q[65];
sx q[65];
rz(-pi) q[65];
rz(-2.356578) q[66];
sx q[66];
rz(-pi) q[66];
rz(-pi/16384) q[73];
sx q[73];
rz(-pi/2) q[73];
cx q[73],q[66];
rz(-pi/2) q[66];
sx q[73];
rz(-pi) q[73];
cx q[73],q[66];
rz(1.5706046) q[66];
sx q[73];
cx q[73],q[66];
rz(0.78578166) q[66];
sx q[66];
rz(pi/2) q[66];
cx q[66],q[65];
rz(-pi/2) q[65];
sx q[66];
rz(-pi) q[66];
cx q[66],q[65];
rz(1.5704128) q[65];
sx q[66];
cx q[66],q[65];
rz(-1.5700293) q[65];
sx q[65];
rz(-pi/2) q[65];
cx q[65],q[64];
rz(pi/2) q[64];
sx q[65];
rz(-pi) q[65];
cx q[65],q[64];
rz(1.5700293) q[64];
sx q[65];
cx q[65],q[64];
x q[64];
rz(3*pi/4) q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[54],q[64];
rz(-pi/2048) q[63];
cx q[63],q[62];
rz(pi/2048) q[62];
cx q[63],q[62];
rz(-pi/2048) q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(0.7823302) q[62];
sx q[62];
rz(-pi) q[62];
rz(-pi/256) q[63];
cx q[64],q[54];
cx q[54],q[64];
rz(-pi/512) q[54];
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
rz(-pi/256) q[45];
rz(-pi/512) q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(-pi/4) q[63];
sx q[63];
rz(-pi/2) q[63];
cx q[63],q[62];
rz(-pi/2) q[62];
sx q[63];
rz(-pi) q[63];
cx q[63],q[62];
rz(1.5677284) q[62];
sx q[63];
cx q[63],q[62];
x q[62];
rz(-2.3592625) q[62];
rz(-pi/512) q[63];
sx q[63];
rz(-pi/128) q[64];
cx q[64],q[54];
rz(pi/128) q[54];
cx q[64],q[54];
rz(-pi/128) q[54];
cx q[45],q[54];
rz(pi/256) q[54];
cx q[45],q[54];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[43],q[44];
rz(pi/16) q[44];
cx q[43],q[44];
rz(-pi/16) q[44];
rz(-pi/128) q[46];
rz(-pi/256) q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
rz(-pi/64) q[54];
cx q[54],q[45];
rz(pi/64) q[45];
cx q[54],q[45];
rz(-pi/64) q[45];
cx q[46],q[45];
rz(pi/128) q[45];
cx q[46],q[45];
rz(-pi/128) q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(-pi/32) q[45];
cx q[45],q[44];
rz(pi/32) q[44];
cx q[45],q[44];
rz(-pi/32) q[44];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
rz(-pi/64) q[45];
cx q[45],q[44];
rz(pi/64) q[44];
cx q[45],q[44];
rz(-pi/64) q[44];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
rz(-pi/16) q[47];
rz(3*pi/4) q[54];
sx q[54];
rz(pi/4) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[64],q[63];
rz(pi/2) q[63];
sx q[64];
rz(-pi) q[64];
cx q[64],q[63];
rz(1.5646604) q[63];
sx q[64];
cx q[64],q[63];
x q[63];
rz(2.3500586) q[63];
rz(1.5585245) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[64],q[54];
rz(-pi/2) q[54];
sx q[64];
rz(-pi) q[64];
cx q[64],q[54];
rz(1.5585245) q[54];
sx q[64];
cx q[64],q[54];
rz(pi/4) q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
rz(-pi/128) q[45];
cx q[45],q[44];
rz(pi/128) q[44];
cx q[45],q[44];
rz(-pi/128) q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
rz(-pi/8) q[44];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[44],q[45];
rz(pi/8) q[45];
cx q[44],q[45];
rz(-pi/8) q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[47],q[46];
rz(pi/16) q[46];
cx q[47],q[46];
rz(-pi/16) q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
rz(-pi/64) q[46];
cx q[47],q[35];
cx q[35],q[47];
cx q[47],q[35];
rz(pi/8) q[35];
rz(-0.29452431) q[54];
cx q[54],q[45];
rz(pi/32) q[45];
cx q[54],q[45];
rz(-pi/32) q[45];
cx q[46],q[45];
rz(pi/64) q[45];
cx q[46],q[45];
rz(-pi/64) q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
rz(pi/4) q[45];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[45],q[46];
rz(pi/4) q[46];
cx q[45],q[46];
sx q[45];
rz(pi/2) q[45];
rz(-pi/4) q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[35],q[47];
rz(pi/8) q[47];
cx q[35],q[47];
rz(-pi/8) q[47];
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
rz(-pi/32) q[46];
rz(pi/4) q[47];
cx q[35],q[47];
sx q[35];
rz(pi/2) q[35];
rz(-pi/4) q[47];
cx q[54],q[45];
rz(pi/16) q[45];
cx q[54],q[45];
rz(-pi/16) q[45];
cx q[46],q[45];
rz(pi/32) q[45];
cx q[46],q[45];
rz(-pi/32) q[45];
cx q[45],q[54];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
rz(-pi/16) q[47];
cx q[54],q[45];
cx q[45],q[54];
rz(-pi/8) q[45];
cx q[45],q[46];
rz(pi/8) q[46];
cx q[45],q[46];
rz(-pi/8) q[46];
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
x q[64];
rz(2.3439226) q[64];
rz(2.3554275) q[65];
rz(-2.356578) q[66];
rz(-1.5709881) q[73];
barrier q[18],q[82],q[15],q[79],q[24],q[88],q[33],q[97],q[42],q[39],q[106],q[51],q[48],q[115],q[112],q[57],q[2],q[121],q[35],q[11],q[75],q[8],q[72],q[17],q[81],q[26],q[90],q[73],q[99],q[64],q[41],q[108],q[105],q[50],q[114],q[59],q[4],q[123],q[68],q[1],q[13],q[28],q[77],q[10],q[74],q[19],q[83],q[66],q[92],q[37],q[34],q[101],q[98],q[62],q[107],q[52],q[116],q[44],q[6],q[125],q[70],q[3],q[67],q[12],q[76],q[21],q[85],q[30],q[27],q[94],q[91],q[36],q[103],q[100],q[43],q[109],q[63],q[118],q[46],q[60],q[5],q[124],q[69],q[14],q[78],q[23],q[20],q[87],q[32],q[84],q[29],q[96],q[93],q[38],q[102],q[61],q[111],q[56],q[120],q[53],q[117],q[45],q[7],q[126],q[71],q[16],q[80],q[25],q[22],q[89],q[86],q[31],q[95],q[40],q[104],q[49],q[113],q[65],q[58],q[110],q[122],q[55],q[0],q[119],q[54],q[9],q[47];
measure q[35] -> meas[0];
measure q[47] -> meas[1];
measure q[46] -> meas[2];
measure q[45] -> meas[3];
measure q[54] -> meas[4];
measure q[44] -> meas[5];
measure q[43] -> meas[6];
measure q[64] -> meas[7];
measure q[63] -> meas[8];
measure q[62] -> meas[9];
measure q[61] -> meas[10];
measure q[65] -> meas[11];
measure q[66] -> meas[12];
measure q[73] -> meas[13];
measure q[28] -> meas[14];
