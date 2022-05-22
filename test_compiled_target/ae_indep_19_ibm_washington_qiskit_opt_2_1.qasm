OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[19];
rz(-3*pi/2) q[41];
sx q[41];
rz(pi/2) q[41];
rz(-3*pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
rz(-3*pi/2) q[59];
sx q[59];
rz(pi/2) q[59];
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
rz(-3*pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
rz(-3*pi/2) q[65];
sx q[65];
rz(pi/2) q[65];
rz(-3*pi/2) q[66];
sx q[66];
rz(pi/2) q[66];
rz(-3*pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
rz(-3*pi/2) q[85];
sx q[85];
rz(pi/2) q[85];
rz(-3*pi/2) q[86];
sx q[86];
rz(pi/2) q[86];
rz(-3*pi/2) q[87];
sx q[87];
rz(pi/2) q[87];
rz(-3*pi/2) q[88];
sx q[88];
rz(pi/2) q[88];
rz(-3*pi/2) q[89];
sx q[89];
rz(pi/2) q[89];
rz(-pi) q[93];
sx q[93];
rz(2.2142974) q[93];
sx q[93];
rz(-3*pi/2) q[105];
sx q[105];
rz(pi/2) q[105];
rz(-3*pi/2) q[106];
sx q[106];
rz(pi/2) q[106];
cx q[106],q[93];
sx q[93];
rz(2.2142974) q[93];
sx q[93];
rz(-pi) q[93];
cx q[106],q[93];
cx q[105],q[106];
cx q[106],q[105];
cx q[105],q[106];
rz(-pi) q[93];
sx q[93];
rz(2.2142974) q[93];
sx q[93];
cx q[106],q[93];
sx q[93];
rz(1.2870023) q[93];
sx q[93];
rz(-pi) q[93];
cx q[106],q[93];
rz(-pi/131072) q[106];
rz(-pi) q[93];
sx q[93];
rz(1.2870023) q[93];
sx q[93];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[88],q[87];
rz(-pi) q[87];
sx q[87];
rz(0.56758825) q[87];
sx q[87];
cx q[88],q[87];
sx q[87];
rz(0.56758825) q[87];
sx q[87];
rz(-pi) q[87];
cx q[88],q[89];
cx q[89],q[88];
cx q[88],q[89];
cx q[88],q[87];
sx q[87];
rz(2.0064163) q[87];
sx q[87];
rz(-pi) q[87];
cx q[88],q[87];
rz(-pi) q[87];
sx q[87];
rz(2.0064163) q[87];
sx q[87];
rz(-pi/32768) q[88];
cx q[93],q[87];
sx q[87];
rz(0.87124027) q[87];
sx q[87];
rz(-pi) q[87];
cx q[93],q[87];
rz(-pi) q[87];
sx q[87];
rz(0.87123975) q[87];
sx q[87];
cx q[86],q[87];
rz(-pi) q[87];
sx q[87];
rz(1.3991131) q[87];
sx q[87];
cx q[86],q[87];
sx q[87];
rz(1.3991131) q[87];
sx q[87];
rz(-pi) q[87];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[85],q[86];
sx q[86];
rz(0.34336642) q[86];
sx q[86];
rz(-pi) q[86];
cx q[85],q[86];
rz(-pi) q[86];
sx q[86];
rz(0.34336645) q[86];
sx q[86];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[73],q[85];
rz(-pi) q[85];
sx q[85];
rz(2.4548618) q[85];
sx q[85];
cx q[73],q[85];
sx q[85];
rz(2.4548597) q[85];
sx q[85];
rz(-pi) q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
rz(-pi) q[73];
sx q[73];
rz(1.768131) q[73];
sx q[73];
cx q[66],q[73];
sx q[73];
rz(1.7681268) q[73];
sx q[73];
rz(-pi) q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[65],q[66];
rz(-pi) q[66];
sx q[66];
rz(0.39465931) q[66];
sx q[66];
cx q[65],q[66];
sx q[66];
rz(0.39466095) q[66];
sx q[66];
rz(-pi) q[66];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
sx q[65];
rz(2.352274) q[65];
sx q[65];
rz(-pi) q[65];
cx q[64],q[65];
rz(-pi) q[65];
sx q[65];
rz(2.3522708) q[65];
sx q[65];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
sx q[64];
rz(1.5629554) q[64];
sx q[64];
rz(-pi) q[64];
cx q[63],q[64];
rz(-pi) q[64];
sx q[64];
rz(1.562949) q[64];
sx q[64];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
rz(-pi) q[62];
sx q[62];
rz(0.01568181) q[62];
sx q[62];
cx q[61],q[62];
sx q[62];
rz(0.015694754) q[62];
sx q[62];
rz(-pi) q[62];
cx q[63],q[62];
sx q[62];
rz(3.110229) q[62];
sx q[62];
rz(-pi) q[62];
cx q[63],q[62];
rz(-pi) q[62];
sx q[62];
rz(3.1102032) q[62];
sx q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
sx q[60];
rz(3.0786654) q[60];
sx q[60];
rz(-pi) q[60];
cx q[53],q[60];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
rz(-pi/16) q[41];
rz(-pi) q[60];
sx q[60];
rz(3.0788137) q[60];
sx q[60];
cx q[61],q[60];
sx q[60];
rz(3.0157382) q[60];
sx q[60];
rz(-pi) q[60];
cx q[61],q[60];
rz(-pi) q[60];
sx q[60];
rz(3.0160347) q[60];
sx q[60];
cx q[59],q[60];
sx q[60];
rz(2.8908837) q[60];
sx q[60];
rz(-pi) q[60];
cx q[59],q[60];
rz(-pi) q[60];
sx q[60];
rz(2.8904767) q[60];
sx q[60];
cx q[53],q[60];
sx q[60];
rz(2.6381747) q[60];
sx q[60];
rz(-pi) q[60];
cx q[53],q[60];
rz(pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
rz(-pi) q[60];
sx q[60];
rz(2.6381747) q[60];
sx q[60];
cx q[59],q[60];
cx q[60],q[59];
cx q[59],q[60];
rz(pi/4) q[60];
cx q[60],q[53];
rz(pi/4) q[53];
cx q[60],q[53];
rz(-pi/4) q[53];
sx q[60];
rz(pi/2) q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(pi/8) q[60];
cx q[60],q[53];
rz(pi/8) q[53];
cx q[60],q[53];
rz(-pi/8) q[53];
cx q[41],q[53];
rz(pi/16) q[53];
cx q[41],q[53];
rz(-pi/16) q[53];
rz(pi/4) q[61];
cx q[60],q[61];
sx q[60];
rz(pi/2) q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
rz(pi/8) q[53];
rz(-pi/4) q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
rz(pi/8) q[60];
cx q[53],q[60];
cx q[53],q[41];
rz(pi/4) q[41];
cx q[53],q[41];
rz(-pi/4) q[41];
sx q[53];
rz(pi/2) q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[53],q[41];
rz(-pi/8) q[60];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(-pi/64) q[61];
rz(-pi/32) q[63];
cx q[63],q[62];
rz(pi/32) q[62];
cx q[63],q[62];
rz(-pi/32) q[62];
cx q[61],q[62];
rz(pi/64) q[62];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
rz(-pi/32) q[60];
rz(-pi/64) q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-pi/16) q[62];
cx q[62],q[61];
rz(pi/16) q[61];
cx q[62],q[61];
rz(-pi/16) q[61];
cx q[60],q[61];
rz(pi/32) q[61];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(-pi/16) q[53];
rz(-pi/32) q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(-pi/8) q[61];
cx q[61],q[60];
rz(pi/8) q[60];
cx q[61],q[60];
rz(-pi/8) q[60];
cx q[53],q[60];
rz(pi/16) q[60];
cx q[53],q[60];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
rz(pi/8) q[41];
rz(-pi/16) q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
rz(pi/4) q[60];
cx q[60],q[53];
rz(pi/4) q[53];
cx q[60],q[53];
rz(-pi/4) q[53];
cx q[41],q[53];
rz(pi/8) q[53];
cx q[41],q[53];
rz(-pi/8) q[53];
sx q[60];
rz(pi/2) q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[41],q[53];
rz(pi/4) q[53];
cx q[41],q[53];
sx q[41];
rz(pi/2) q[41];
rz(-pi/4) q[53];
rz(-pi/128) q[64];
cx q[64],q[63];
rz(pi/128) q[63];
cx q[64],q[63];
rz(-pi/128) q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(-pi/64) q[63];
cx q[63],q[62];
rz(pi/64) q[62];
cx q[63],q[62];
rz(-pi/64) q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-pi/32) q[62];
cx q[62],q[61];
rz(pi/32) q[61];
cx q[62],q[61];
rz(-pi/32) q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(-pi/16) q[61];
cx q[61],q[60];
rz(pi/16) q[60];
cx q[61],q[60];
rz(-pi/16) q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
rz(-pi/8) q[60];
cx q[60],q[53];
rz(pi/8) q[53];
cx q[60],q[53];
rz(-pi/8) q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(pi/4) q[53];
cx q[53],q[41];
rz(pi/4) q[41];
cx q[53],q[41];
rz(-pi/4) q[41];
sx q[53];
rz(pi/2) q[53];
rz(-pi/256) q[65];
cx q[65],q[64];
rz(pi/256) q[64];
cx q[65],q[64];
rz(-pi/256) q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
rz(-pi/128) q[64];
cx q[64],q[63];
rz(pi/128) q[63];
cx q[64],q[63];
rz(-pi/128) q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(-pi/64) q[63];
cx q[63],q[62];
rz(pi/64) q[62];
cx q[63],q[62];
rz(-pi/64) q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-pi/32) q[62];
cx q[62],q[61];
rz(pi/32) q[61];
cx q[62],q[61];
rz(-pi/32) q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(-pi/16) q[61];
cx q[61],q[60];
rz(pi/16) q[60];
cx q[61],q[60];
rz(-pi/16) q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
rz(pi/8) q[53];
cx q[53],q[41];
rz(pi/8) q[41];
cx q[53],q[41];
rz(-pi/8) q[41];
rz(pi/4) q[60];
cx q[53],q[60];
sx q[53];
rz(pi/2) q[53];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
rz(-pi/4) q[60];
rz(-pi/512) q[66];
cx q[66],q[65];
rz(pi/512) q[65];
cx q[66],q[65];
rz(-pi/512) q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
rz(-pi/256) q[65];
cx q[65],q[64];
rz(pi/256) q[64];
cx q[65],q[64];
rz(-pi/256) q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
rz(-pi/128) q[64];
cx q[64],q[63];
rz(pi/128) q[63];
cx q[64],q[63];
rz(-pi/128) q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(-pi/64) q[63];
cx q[63],q[62];
rz(pi/64) q[62];
cx q[63],q[62];
rz(-pi/64) q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-pi/32) q[62];
cx q[62],q[61];
rz(pi/32) q[61];
cx q[62],q[61];
rz(-pi/32) q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
rz(-3*pi/16) q[60];
cx q[60],q[53];
rz(pi/16) q[53];
cx q[60],q[53];
rz(-pi/16) q[53];
rz(pi/8) q[61];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(pi/4) q[53];
cx q[53],q[41];
rz(pi/4) q[41];
cx q[53],q[41];
rz(-pi/4) q[41];
sx q[53];
rz(pi/2) q[53];
rz(-pi/8) q[61];
rz(-pi/1024) q[73];
cx q[73],q[66];
rz(pi/1024) q[66];
cx q[73],q[66];
rz(-pi/1024) q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
rz(-pi/512) q[66];
cx q[66],q[65];
rz(pi/512) q[65];
cx q[66],q[65];
rz(-pi/512) q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
rz(-pi/256) q[65];
cx q[65],q[64];
rz(pi/256) q[64];
cx q[65],q[64];
rz(-pi/256) q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
rz(-pi/128) q[64];
cx q[64],q[63];
rz(pi/128) q[63];
cx q[64],q[63];
rz(-pi/128) q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(-pi/64) q[63];
cx q[63],q[62];
rz(pi/64) q[62];
cx q[63],q[62];
rz(-pi/64) q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
rz(-0.29452431) q[61];
cx q[61],q[60];
rz(pi/32) q[60];
cx q[61],q[60];
rz(-pi/32) q[60];
rz(pi/16) q[62];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
rz(pi/8) q[53];
cx q[53],q[41];
rz(pi/8) q[41];
cx q[53],q[41];
rz(-pi/8) q[41];
rz(pi/4) q[60];
cx q[53],q[60];
sx q[53];
rz(pi/2) q[53];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
rz(-pi/4) q[60];
rz(-pi/16) q[62];
rz(-pi/2048) q[85];
cx q[85],q[73];
rz(pi/2048) q[73];
cx q[85],q[73];
rz(-pi/2048) q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
rz(-pi/1024) q[73];
cx q[73],q[66];
rz(pi/1024) q[66];
cx q[73],q[66];
rz(-pi/1024) q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
rz(-pi/512) q[66];
cx q[66],q[65];
rz(pi/512) q[65];
cx q[66],q[65];
rz(-pi/512) q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
rz(-pi/256) q[65];
cx q[65],q[64];
rz(pi/256) q[64];
cx q[65],q[64];
rz(-pi/256) q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
rz(-pi/128) q[64];
cx q[64],q[63];
rz(pi/128) q[63];
cx q[64],q[63];
rz(-pi/128) q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
rz(-0.14726216) q[62];
cx q[62],q[61];
rz(pi/64) q[61];
cx q[62],q[61];
rz(-pi/64) q[61];
rz(pi/32) q[63];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
rz(-3*pi/16) q[60];
cx q[60],q[53];
rz(pi/16) q[53];
cx q[60],q[53];
rz(-pi/16) q[53];
rz(pi/8) q[61];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(pi/4) q[53];
cx q[53],q[41];
rz(pi/4) q[41];
cx q[53],q[41];
rz(-pi/4) q[41];
sx q[53];
rz(pi/2) q[53];
rz(-pi/8) q[61];
rz(-pi/32) q[63];
rz(-pi/4096) q[86];
cx q[86],q[85];
rz(pi/4096) q[85];
cx q[86],q[85];
rz(-pi/4096) q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
rz(-pi/2048) q[85];
cx q[85],q[73];
rz(pi/2048) q[73];
cx q[85],q[73];
rz(-pi/2048) q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
rz(-pi/1024) q[73];
cx q[73],q[66];
rz(pi/1024) q[66];
cx q[73],q[66];
rz(-pi/1024) q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
rz(-pi/512) q[66];
cx q[66],q[65];
rz(pi/512) q[65];
cx q[66],q[65];
rz(-pi/512) q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
rz(-pi/256) q[65];
cx q[65],q[64];
rz(pi/256) q[64];
cx q[65],q[64];
rz(-pi/256) q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
rz(-0.073631078) q[63];
cx q[63],q[62];
rz(pi/128) q[62];
cx q[63],q[62];
rz(-pi/128) q[62];
rz(pi/64) q[64];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
rz(-0.29452431) q[61];
cx q[61],q[60];
rz(pi/32) q[60];
cx q[61],q[60];
rz(-pi/32) q[60];
rz(pi/16) q[62];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
rz(pi/8) q[53];
cx q[53],q[41];
rz(pi/8) q[41];
cx q[53],q[41];
rz(-pi/8) q[41];
rz(pi/4) q[60];
cx q[53],q[60];
sx q[53];
rz(pi/2) q[53];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
rz(-pi/4) q[60];
rz(-pi/16) q[62];
rz(-pi/64) q[64];
rz(-pi/8192) q[87];
cx q[87],q[86];
rz(pi/8192) q[86];
cx q[87],q[86];
rz(-pi/8192) q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
rz(-pi/4096) q[86];
cx q[86],q[85];
rz(pi/4096) q[85];
cx q[86],q[85];
rz(-pi/4096) q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
rz(-pi/2048) q[85];
cx q[85],q[73];
rz(pi/2048) q[73];
cx q[85],q[73];
rz(-pi/2048) q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
rz(-pi/1024) q[73];
cx q[73],q[66];
rz(pi/1024) q[66];
cx q[73],q[66];
rz(-pi/1024) q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
rz(-pi/512) q[66];
cx q[66],q[65];
rz(pi/512) q[65];
cx q[66],q[65];
rz(-pi/512) q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
rz(-0.036815539) q[64];
cx q[64],q[63];
rz(pi/256) q[63];
cx q[64],q[63];
rz(-pi/256) q[63];
rz(pi/128) q[65];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
rz(-0.14726216) q[62];
cx q[62],q[61];
rz(pi/64) q[61];
cx q[62],q[61];
rz(-pi/64) q[61];
rz(pi/32) q[63];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
rz(-3*pi/16) q[60];
cx q[60],q[53];
rz(pi/16) q[53];
cx q[60],q[53];
rz(-pi/16) q[53];
rz(pi/8) q[61];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(pi/4) q[53];
cx q[53],q[41];
rz(pi/4) q[41];
cx q[53],q[41];
rz(-pi/4) q[41];
sx q[53];
rz(pi/2) q[53];
rz(-pi/8) q[61];
rz(-pi/32) q[63];
rz(-pi/128) q[65];
rz(-pi/16384) q[93];
cx q[93],q[87];
rz(pi/16384) q[87];
cx q[93],q[87];
rz(-pi/16384) q[87];
cx q[88],q[87];
rz(pi/32768) q[87];
cx q[88],q[87];
rz(-pi/32768) q[87];
cx q[89],q[88];
cx q[88],q[89];
cx q[89],q[88];
rz(-pi/65536) q[88];
cx q[88],q[87];
rz(pi/65536) q[87];
cx q[88],q[87];
rz(-pi/65536) q[87];
cx q[87],q[93];
rz(-0.0005752428) q[89];
cx q[93],q[87];
cx q[87],q[93];
cx q[106],q[93];
rz(-pi/8192) q[87];
cx q[87],q[86];
rz(pi/8192) q[86];
cx q[87],q[86];
rz(-pi/8192) q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
rz(-pi/4096) q[86];
cx q[86],q[85];
rz(pi/4096) q[85];
cx q[86],q[85];
rz(-pi/4096) q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
rz(-pi/2048) q[85];
cx q[85],q[73];
rz(pi/2048) q[73];
cx q[85],q[73];
rz(-pi/2048) q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
rz(-pi/1024) q[73];
cx q[73],q[66];
rz(pi/1024) q[66];
cx q[73],q[66];
rz(-pi/1024) q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[65],q[66];
cx q[66],q[65];
rz(-0.018407769) q[65];
cx q[65],q[64];
rz(pi/512) q[64];
cx q[65],q[64];
rz(-pi/512) q[64];
rz(pi/256) q[66];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
rz(-0.073631078) q[63];
cx q[63],q[62];
rz(pi/128) q[62];
cx q[63],q[62];
rz(-pi/128) q[62];
rz(pi/64) q[64];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
rz(-0.29452431) q[61];
cx q[61],q[60];
rz(pi/32) q[60];
cx q[61],q[60];
rz(-pi/32) q[60];
rz(pi/16) q[62];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
rz(pi/8) q[53];
cx q[53],q[41];
rz(pi/8) q[41];
cx q[53],q[41];
rz(-pi/8) q[41];
rz(pi/4) q[60];
cx q[53],q[60];
sx q[53];
rz(pi/2) q[53];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
rz(-pi/4) q[60];
rz(-pi/16) q[62];
rz(-pi/64) q[64];
rz(-pi/256) q[66];
cx q[87],q[88];
cx q[88],q[87];
cx q[87],q[88];
rz(-pi/32768) q[87];
cx q[89],q[88];
rz(pi/16384) q[88];
cx q[89],q[88];
rz(-pi/16384) q[88];
cx q[87],q[88];
rz(pi/32768) q[88];
cx q[87],q[88];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
rz(-pi/16384) q[86];
rz(-pi/32768) q[88];
cx q[88],q[87];
cx q[87],q[88];
cx q[88],q[87];
cx q[89],q[88];
rz(pi/8192) q[88];
cx q[89],q[88];
rz(-pi/8192) q[88];
rz(pi/131072) q[93];
cx q[106],q[93];
cx q[106],q[105];
cx q[105],q[106];
cx q[106],q[105];
rz(-pi/65536) q[105];
rz(-pi/262144) q[106];
rz(-pi/131072) q[93];
cx q[106],q[93];
rz(pi/262144) q[93];
cx q[106],q[93];
rz(-pi/262144) q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[88];
cx q[88],q[87];
cx q[87],q[88];
cx q[86],q[87];
rz(pi/16384) q[87];
cx q[86],q[87];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
rz(-pi/8192) q[85];
rz(-pi/16384) q[87];
cx q[88],q[89];
cx q[89],q[88];
cx q[88],q[89];
cx q[93],q[106];
cx q[106],q[93];
cx q[93],q[106];
cx q[105],q[106];
rz(pi/65536) q[106];
cx q[105],q[106];
rz(-pi/65536) q[106];
rz(-pi/131072) q[93];
cx q[93],q[106];
rz(pi/131072) q[106];
cx q[93],q[106];
rz(-pi/131072) q[106];
cx q[106],q[105];
cx q[105],q[106];
cx q[106],q[105];
rz(-pi/32768) q[106];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[106],q[93];
rz(-pi/65536) q[87];
rz(pi/32768) q[93];
cx q[106],q[93];
rz(-pi/32768) q[93];
cx q[87],q[93];
rz(pi/65536) q[93];
cx q[87],q[93];
cx q[88],q[87];
cx q[87],q[88];
cx q[88],q[87];
rz(-pi/4096) q[87];
cx q[87],q[86];
rz(pi/4096) q[86];
cx q[87],q[86];
rz(-pi/4096) q[86];
cx q[85],q[86];
rz(pi/8192) q[86];
cx q[85],q[86];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
rz(-pi/4096) q[73];
rz(-pi/8192) q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
rz(-pi/2048) q[86];
cx q[86],q[85];
rz(pi/2048) q[85];
cx q[86],q[85];
rz(-pi/2048) q[85];
cx q[73],q[85];
rz(pi/4096) q[85];
cx q[73],q[85];
rz(-pi/4096) q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
rz(-0.0092038847) q[66];
cx q[66],q[65];
rz(pi/1024) q[65];
cx q[66],q[65];
rz(-pi/1024) q[65];
rz(pi/512) q[73];
cx q[66],q[73];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
rz(-0.036815539) q[64];
cx q[64],q[63];
rz(pi/256) q[63];
cx q[64],q[63];
rz(-pi/256) q[63];
rz(pi/128) q[65];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
rz(-0.14726216) q[62];
cx q[62],q[61];
rz(pi/64) q[61];
cx q[62],q[61];
rz(-pi/64) q[61];
rz(pi/32) q[63];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
rz(-3*pi/16) q[60];
cx q[60],q[53];
rz(pi/16) q[53];
cx q[60],q[53];
rz(-pi/16) q[53];
rz(pi/8) q[61];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(pi/4) q[53];
cx q[53],q[41];
rz(pi/4) q[41];
cx q[53],q[41];
rz(-pi/4) q[41];
sx q[53];
rz(pi/2) q[53];
rz(-pi/8) q[61];
rz(-pi/32) q[63];
rz(-pi/128) q[65];
rz(-pi/512) q[73];
cx q[73],q[85];
cx q[85],q[73];
rz(-0.0046019424) q[73];
cx q[73],q[66];
rz(pi/2048) q[66];
cx q[73],q[66];
rz(-pi/2048) q[66];
rz(pi/1024) q[85];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[65],q[66];
cx q[66],q[65];
rz(-0.018407769) q[65];
cx q[65],q[64];
rz(pi/512) q[64];
cx q[65],q[64];
rz(-pi/512) q[64];
rz(pi/256) q[66];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
rz(-0.073631078) q[63];
cx q[63],q[62];
rz(pi/128) q[62];
cx q[63],q[62];
rz(-pi/128) q[62];
rz(pi/64) q[64];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
rz(-0.29452431) q[61];
cx q[61],q[60];
rz(pi/32) q[60];
cx q[61],q[60];
rz(-pi/32) q[60];
rz(pi/16) q[62];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
rz(pi/8) q[53];
cx q[53],q[41];
rz(pi/8) q[41];
cx q[53],q[41];
rz(-pi/8) q[41];
cx q[53],q[60];
rz(pi/4) q[60];
cx q[53],q[60];
sx q[53];
rz(pi/2) q[53];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
rz(-pi/4) q[60];
rz(-pi/16) q[62];
rz(-pi/64) q[64];
rz(-pi/256) q[66];
rz(-pi/1024) q[85];
rz(-0.0002876214) q[88];
rz(-pi/65536) q[93];
cx q[93],q[106];
cx q[106],q[93];
cx q[93],q[106];
rz(-pi/16384) q[93];
cx q[93],q[87];
rz(pi/16384) q[87];
cx q[93],q[87];
rz(-pi/16384) q[87];
cx q[88],q[87];
rz(pi/32768) q[87];
cx q[88],q[87];
rz(-pi/32768) q[87];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
rz(-pi/8192) q[87];
cx q[87],q[86];
rz(pi/8192) q[86];
cx q[87],q[86];
rz(-pi/8192) q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[85],q[86];
cx q[86],q[85];
rz(-0.0023009712) q[85];
cx q[85],q[73];
rz(pi/4096) q[73];
cx q[85],q[73];
rz(-pi/4096) q[73];
rz(pi/2048) q[86];
cx q[85],q[86];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
rz(-0.0092038847) q[66];
cx q[66],q[65];
rz(pi/1024) q[65];
cx q[66],q[65];
rz(-pi/1024) q[65];
rz(pi/512) q[73];
cx q[66],q[73];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
rz(-0.036815539) q[64];
cx q[64],q[63];
rz(pi/256) q[63];
cx q[64],q[63];
rz(-pi/256) q[63];
rz(pi/128) q[65];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
rz(-0.14726216) q[62];
cx q[62],q[61];
rz(pi/64) q[61];
cx q[62],q[61];
rz(-pi/64) q[61];
rz(pi/32) q[63];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
rz(-3*pi/16) q[60];
cx q[60],q[53];
rz(pi/16) q[53];
cx q[60],q[53];
rz(-pi/16) q[53];
rz(pi/8) q[61];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(pi/4) q[53];
cx q[53],q[41];
rz(pi/4) q[41];
cx q[53],q[41];
rz(-pi/4) q[41];
sx q[53];
rz(pi/2) q[53];
rz(-pi/8) q[61];
rz(-pi/32) q[63];
rz(-pi/128) q[65];
rz(-pi/512) q[73];
rz(-pi/2048) q[86];
cx q[88],q[87];
rz(pi/16384) q[87];
cx q[88],q[87];
rz(-pi/16384) q[87];
cx q[87],q[88];
cx q[88],q[87];
cx q[87],q[88];
cx q[86],q[87];
cx q[87],q[86];
rz(-0.0011504856) q[86];
cx q[86],q[85];
rz(pi/8192) q[85];
cx q[86],q[85];
rz(-pi/8192) q[85];
rz(pi/4096) q[87];
cx q[86],q[87];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[73],q[85];
cx q[85],q[73];
rz(-0.0046019424) q[73];
cx q[73],q[66];
rz(pi/2048) q[66];
cx q[73],q[66];
rz(-pi/2048) q[66];
rz(pi/1024) q[85];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[65],q[66];
cx q[66],q[65];
rz(-0.018407769) q[65];
cx q[65],q[64];
rz(pi/512) q[64];
cx q[65],q[64];
rz(-pi/512) q[64];
rz(pi/256) q[66];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
rz(-0.073631078) q[63];
cx q[63],q[62];
rz(pi/128) q[62];
cx q[63],q[62];
rz(-pi/128) q[62];
rz(pi/64) q[64];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
rz(-0.29452431) q[61];
cx q[61],q[60];
rz(pi/32) q[60];
cx q[61],q[60];
rz(-pi/32) q[60];
rz(pi/16) q[62];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
rz(pi/8) q[53];
cx q[53],q[41];
rz(pi/8) q[41];
cx q[53],q[41];
rz(-pi/8) q[41];
rz(pi/4) q[60];
cx q[53],q[60];
sx q[53];
rz(pi/2) q[53];
rz(-pi/4) q[60];
rz(-pi/16) q[62];
rz(-pi/64) q[64];
rz(-pi/256) q[66];
rz(-pi/1024) q[85];
rz(-pi/4096) q[87];
barrier q[18],q[82],q[15],q[79],q[24],q[41],q[33],q[97],q[42],q[39],q[53],q[51],q[48],q[115],q[112],q[57],q[2],q[121],q[65],q[11],q[75],q[8],q[72],q[17],q[81],q[26],q[90],q[35],q[99],q[44],q[89],q[108],q[60],q[50],q[114],q[105],q[4],q[123],q[68],q[1],q[13],q[85],q[77],q[10],q[74],q[19],q[83],q[28],q[92],q[37],q[34],q[101],q[98],q[43],q[107],q[52],q[116],q[86],q[6],q[125],q[70],q[3],q[67],q[12],q[76],q[21],q[63],q[30],q[27],q[94],q[91],q[36],q[103],q[100],q[45],q[109],q[54],q[118],q[87],q[106],q[5],q[124],q[69],q[14],q[78],q[23],q[20],q[61],q[32],q[84],q[29],q[96],q[59],q[38],q[102],q[47],q[111],q[56],q[120],q[93],q[117],q[88],q[7],q[126],q[71],q[16],q[80],q[25],q[22],q[62],q[64],q[31],q[95],q[40],q[104],q[49],q[113],q[46],q[58],q[110],q[122],q[55],q[0],q[119],q[73],q[9],q[66];
measure q[53] -> meas[0];
measure q[60] -> meas[1];
measure q[41] -> meas[2];
measure q[62] -> meas[3];
measure q[61] -> meas[4];
measure q[64] -> meas[5];
measure q[63] -> meas[6];
measure q[66] -> meas[7];
measure q[65] -> meas[8];
measure q[85] -> meas[9];
measure q[73] -> meas[10];
measure q[87] -> meas[11];
measure q[86] -> meas[12];
measure q[88] -> meas[13];
measure q[93] -> meas[14];
measure q[106] -> meas[15];
measure q[105] -> meas[16];
measure q[89] -> meas[17];
measure q[59] -> meas[18];
