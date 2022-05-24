OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[18];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
sx q[13];
rz(-pi/2) q[13];
sx q[14];
rz(-pi/2) q[14];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
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
x q[25];
rz(-0.89785214) q[25];
cx q[25],q[22];
rz(1.3464816) q[22];
cx q[25],q[22];
rz(-1.3464816) q[22];
cx q[25],q[24];
rz(-0.44862946) q[24];
cx q[25],q[24];
cx q[22],q[25];
rz(0.44862946) q[24];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-2.1094068) q[16];
sx q[16];
rz(-pi/2) q[16];
cx q[16],q[14];
rz(pi/2) q[14];
sx q[16];
rz(-pi) q[16];
cx q[16],q[14];
rz(0.67353743) q[14];
sx q[16];
cx q[16],q[14];
rz(2.2736944) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[14];
rz(-pi) q[14];
cx q[14],q[13];
rz(0.22372153) q[13];
sx q[14];
cx q[14],q[13];
x q[13];
rz(2.8040851) q[13];
cx q[13],q[12];
rz(-0.44744302) q[12];
cx q[13],q[12];
rz(0.44744302) q[12];
rz(0.22372153) q[14];
rz(2.4680552) q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
rz(-0.89488605) q[14];
cx q[13],q[14];
rz(0.89488605) q[14];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[20];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
rz(1.35182055) q[14];
cx q[13],q[14];
rz(-1.35182055) q[14];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[13],q[14];
rz(-0.437951515) q[14];
cx q[13],q[14];
rz(0.437951515) q[14];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[13],q[14];
rz(-0.87590305) q[14];
cx q[13],q[14];
rz(0.87590305) q[14];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[3],q[5];
cx q[5],q[3];
cx q[3],q[5];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[13],q[14];
rz(1.3897866) q[14];
cx q[13],q[14];
rz(-1.3897866) q[14];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[13],q[14];
rz(-0.362019465) q[14];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(0.72403895) q[12];
cx q[12],q[10];
rz(-0.72403895) q[10];
cx q[12],q[10];
rz(0.72403895) q[10];
rz(0.362019465) q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(0.71176707) q[15];
cx q[15],q[18];
rz(-1.44807785) q[18];
cx q[15],q[18];
rz(1.44807785) q[18];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[15],q[12];
rz(0.245436925) q[12];
cx q[15],q[12];
rz(-0.245436925) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[15],q[12];
rz(0.490873852123405) q[12];
cx q[15],q[12];
rz(-0.490873852123405) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[15];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[14],q[16];
cx q[15],q[12];
cx q[12],q[15];
rz(-5*pi/16) q[12];
cx q[16],q[14];
cx q[14],q[16];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[20];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(pi/16) q[20];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[1],q[4];
cx q[4],q[1];
cx q[1],q[4];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[12],q[10];
rz(5*pi/16) q[10];
cx q[12],q[10];
rz(-5*pi/16) q[10];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-pi/1024) q[10];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(0.78424768) q[15];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-0.68722339) q[13];
rz(pi/8) q[14];
rz(pi/8) q[18];
cx q[18],q[21];
rz(-3*pi/8) q[21];
cx q[18],q[21];
rz(3*pi/8) q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
rz(pi/4) q[21];
cx q[18],q[21];
rz(-pi/4) q[21];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
x q[24];
rz(pi/2) q[24];
rz(pi/4) q[25];
sx q[25];
rz(-pi/2) q[25];
cx q[25],q[24];
rz(pi/2) q[24];
sx q[25];
rz(-pi) q[25];
cx q[25],q[24];
rz(pi/4) q[24];
sx q[25];
cx q[25],q[24];
sx q[24];
rz(pi/2) q[24];
x q[25];
rz(-pi/4) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
rz(pi/8) q[16];
cx q[14],q[16];
rz(-pi/8) q[16];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
rz(pi/4) q[16];
cx q[14],q[16];
sx q[14];
rz(pi/2) q[14];
rz(-pi/4) q[16];
cx q[20],q[19];
rz(pi/16) q[19];
cx q[20],q[19];
rz(-pi/16) q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/32) q[14];
cx q[13],q[14];
rz(-pi/32) q[14];
cx q[20],q[19];
rz(pi/8) q[19];
cx q[20],q[19];
rz(-pi/8) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/16) q[14];
cx q[13],q[14];
rz(-pi/16) q[14];
cx q[20],q[19];
rz(pi/4) q[19];
cx q[20],q[19];
rz(-pi/4) q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[13],q[14];
rz(pi/8) q[14];
cx q[13],q[14];
rz(-pi/8) q[14];
sx q[20];
rz(pi/2) q[20];
rz(-0.3436117) q[22];
cx q[22],q[19];
rz(pi/64) q[19];
cx q[22],q[19];
rz(-pi/64) q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[22],q[19];
rz(pi/32) q[19];
cx q[22],q[19];
rz(-pi/32) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[22],q[19];
rz(pi/16) q[19];
cx q[22],q[19];
rz(-pi/16) q[19];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(-0.0002876214) q[23];
rz(-pi/262144) q[24];
sx q[24];
rz(-pi/2) q[24];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[3];
cx q[3],q[5];
cx q[5],q[3];
rz(2.3500586) q[3];
sx q[3];
rz(-pi/128) q[5];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[20];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
rz(-pi/256) q[11];
cx q[5],q[8];
rz(pi/128) q[8];
cx q[5],q[8];
rz(-pi/128) q[8];
cx q[11],q[8];
rz(pi/256) q[8];
cx q[11],q[8];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(-pi/128) q[14];
rz(-pi/256) q[8];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
rz(pi/4) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[3];
rz(pi/2) q[3];
sx q[5];
rz(-pi) q[5];
cx q[5],q[3];
rz(1.5646604) q[3];
sx q[5];
cx q[5],q[3];
x q[3];
rz(2.3500586) q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[4],q[1];
cx q[1],q[4];
cx q[4],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
rz(2.2702916) q[5];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[4];
cx q[10],q[7];
cx q[4],q[1];
cx q[1],q[4];
cx q[4],q[1];
rz(-0.0023009712) q[4];
rz(pi/1024) q[7];
cx q[10],q[7];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
rz(-pi/1024) q[7];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[7];
rz(-pi/64) q[8];
cx q[8],q[11];
rz(pi/64) q[11];
cx q[8],q[11];
rz(-pi/64) q[11];
cx q[14],q[11];
rz(pi/128) q[11];
cx q[14],q[11];
rz(-pi/128) q[11];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(-pi/64) q[16];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
rz(-pi/32) q[11];
cx q[11],q[14];
rz(pi/32) q[14];
cx q[11],q[14];
rz(-pi/32) q[14];
cx q[16],q[14];
rz(pi/64) q[14];
cx q[16],q[14];
rz(-pi/64) q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/16) q[13];
rz(pi/4) q[14];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
rz(pi/4) q[16];
cx q[14],q[16];
sx q[14];
rz(pi/2) q[14];
rz(-pi/4) q[16];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
rz(pi/8) q[19];
cx q[19],q[16];
rz(pi/8) q[16];
cx q[19],q[16];
rz(-pi/8) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/16) q[14];
cx q[13],q[14];
rz(-pi/16) q[14];
cx q[19],q[16];
rz(pi/4) q[16];
cx q[19],q[16];
rz(-pi/4) q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[13],q[14];
rz(pi/8) q[14];
cx q[13],q[14];
rz(-pi/8) q[14];
sx q[19];
rz(pi/2) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
rz(pi/4) q[14];
cx q[13],q[14];
sx q[13];
rz(pi/2) q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(2.3500586) q[13];
sx q[13];
rz(-pi/4) q[14];
rz(-0.68722339) q[22];
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
cx q[22],q[19];
rz(pi/16) q[19];
cx q[22],q[19];
rz(-pi/16) q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(-pi/4) q[16];
sx q[16];
rz(-pi/2) q[16];
cx q[22],q[19];
rz(pi/8) q[19];
cx q[22],q[19];
rz(-pi/8) q[19];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[19];
cx q[5],q[8];
rz(pi/256) q[8];
cx q[5],q[8];
rz(-pi/256) q[8];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(pi/4) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[14];
rz(-pi) q[14];
cx q[14],q[13];
rz(1.5646604) q[13];
sx q[14];
cx q[14],q[13];
x q[13];
rz(2.3500586) q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(2.319379) q[14];
cx q[5],q[8];
rz(pi/128) q[8];
cx q[5],q[8];
rz(-pi/128) q[8];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
rz(pi/256) q[11];
cx q[14],q[11];
rz(-pi/256) q[11];
cx q[5],q[8];
rz(pi/64) q[8];
cx q[5],q[8];
rz(-pi/64) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
rz(pi/128) q[11];
cx q[14],q[11];
rz(-pi/128) q[11];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(-pi/64) q[11];
rz(0.68722339) q[14];
sx q[14];
rz(-pi) q[14];
cx q[16],q[14];
rz(-pi/2) q[14];
sx q[16];
rz(-pi) q[16];
cx q[16],q[14];
rz(1.4726216) q[14];
sx q[16];
cx q[16],q[14];
x q[14];
rz(-2.4543693) q[14];
cx q[11],q[14];
rz(pi/64) q[14];
cx q[11],q[14];
rz(-pi/64) q[14];
rz(-3*pi/4) q[16];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-0.0046019424) q[13];
cx q[13],q[12];
rz(pi/2048) q[12];
cx q[13],q[12];
rz(-pi/2048) q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[13],q[12];
rz(pi/1024) q[12];
cx q[13],q[12];
rz(-pi/1024) q[12];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(-pi/16) q[19];
cx q[19],q[20];
rz(pi/16) q[20];
cx q[19],q[20];
cx q[19],q[22];
rz(-pi/16) q[20];
cx q[22],q[19];
cx q[19],q[22];
rz(pi/4) q[19];
cx q[19],q[16];
rz(pi/4) q[16];
cx q[19],q[16];
rz(-pi/4) q[16];
sx q[19];
rz(pi/2) q[19];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
rz(pi/8) q[19];
cx q[19],q[16];
rz(pi/8) q[16];
cx q[19],q[16];
rz(-pi/8) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(-0.018407769) q[11];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-0.29452431) q[16];
cx q[19],q[22];
rz(pi/4) q[22];
cx q[19],q[22];
sx q[19];
rz(pi/2) q[19];
cx q[19],q[20];
cx q[20],q[19];
cx q[19],q[20];
cx q[16],q[19];
rz(pi/32) q[19];
cx q[16],q[19];
cx q[16],q[14];
rz(pi/16) q[14];
cx q[16],q[14];
rz(-pi/16) q[14];
rz(-pi/32) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/8) q[19];
rz(-pi/4) q[22];
cx q[19],q[22];
rz(pi/8) q[22];
cx q[19],q[22];
cx q[19],q[20];
rz(pi/4) q[20];
cx q[19],q[20];
sx q[19];
rz(pi/2) q[19];
rz(-pi/4) q[20];
rz(-pi/8) q[22];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[4],q[7];
rz(pi/4096) q[7];
cx q[4],q[7];
rz(-pi/4096) q[7];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[15],q[12];
rz(pi/8192) q[12];
cx q[15],q[12];
rz(-pi/8192) q[12];
sx q[15];
rz(pi/2) q[15];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[4],q[7];
rz(pi/2048) q[7];
cx q[4],q[7];
cx q[1],q[4];
cx q[4],q[1];
cx q[1],q[4];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
rz(0.7823302) q[3];
sx q[3];
rz(-pi) q[3];
rz(-pi/2048) q[7];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
rz(3*pi/4) q[12];
sx q[12];
cx q[15],q[12];
rz(pi/2) q[12];
sx q[15];
rz(-pi) q[15];
cx q[15],q[12];
rz(1.5700293) q[12];
sx q[15];
cx q[15],q[12];
x q[12];
rz(3*pi/4) q[12];
rz(2.3554275) q[15];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[4];
rz(-pi/16384) q[7];
cx q[7],q[10];
rz(pi/16384) q[10];
cx q[7],q[10];
rz(-pi/16384) q[10];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[23],q[21];
rz(pi/32768) q[21];
cx q[23],q[21];
rz(-pi/32768) q[21];
sx q[23];
rz(-pi/2) q[23];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
rz(-pi/8192) q[10];
cx q[10],q[12];
rz(pi/8192) q[12];
cx q[10],q[12];
rz(-pi/8192) q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(-pi/131072) q[18];
sx q[18];
rz(-pi/2) q[18];
sx q[21];
rz(-pi) q[21];
cx q[23],q[21];
rz(-pi/2) q[21];
sx q[23];
rz(-pi) q[23];
cx q[23],q[21];
rz(1.5706046) q[21];
sx q[23];
cx q[23],q[21];
rz(-pi) q[21];
x q[21];
rz(-1.5709881) q[23];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[7];
rz(-0.78693214) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-pi/65536) q[12];
cx q[12],q[15];
rz(pi/65536) q[15];
cx q[12],q[15];
rz(-pi/65536) q[15];
sx q[15];
rz(-pi) q[15];
cx q[18],q[15];
rz(-pi/2) q[15];
sx q[18];
rz(-pi) q[18];
cx q[18],q[15];
rz(pi/2) q[15];
sx q[18];
cx q[18],q[15];
rz(-pi) q[15];
x q[15];
rz(-1.5708203) q[18];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(-pi/8192) q[12];
sx q[12];
rz(-pi/2) q[12];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(-pi/32768) q[18];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
rz(pi/32768) q[21];
cx q[18],q[21];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
rz(-pi/65536) q[18];
rz(-pi/32768) q[21];
cx q[18],q[21];
rz(pi/65536) q[21];
cx q[18],q[21];
rz(-pi/65536) q[21];
sx q[21];
rz(-pi) q[21];
sx q[23];
rz(-pi) q[23];
cx q[24],q[23];
rz(-pi/2) q[23];
sx q[24];
rz(-pi) q[24];
cx q[24],q[23];
rz(pi/2) q[23];
sx q[24];
cx q[24],q[23];
rz(2.396845e-05) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[23],q[21];
rz(-pi/2) q[21];
sx q[23];
rz(-pi) q[23];
cx q[23],q[21];
rz(pi/2) q[21];
sx q[23];
cx q[23],q[21];
rz(-pi) q[21];
x q[21];
rz(-1.5708203) q[23];
rz(-1.5708083) q[24];
rz(-0.78616515) q[7];
sx q[7];
rz(-pi/2) q[7];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[5];
cx q[11],q[8];
rz(pi/512) q[8];
cx q[11],q[8];
rz(-pi/512) q[8];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[11],q[8];
rz(-pi/4) q[5];
sx q[5];
rz(-pi/2) q[5];
cx q[5],q[3];
rz(-pi/2) q[3];
sx q[5];
rz(-pi) q[5];
cx q[5],q[3];
rz(1.5677284) q[3];
sx q[5];
cx q[5],q[3];
x q[3];
rz(-2.3592625) q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
rz(3*pi/4) q[1];
sx q[1];
cx q[4],q[1];
rz(-pi/2) q[1];
sx q[4];
rz(-pi) q[4];
cx q[4],q[1];
rz(1.5692623) q[1];
sx q[4];
cx q[4],q[1];
rz(pi/4) q[1];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
rz(3.1385247) q[3];
sx q[3];
rz(pi/2048) q[4];
sx q[4];
rz(-2.3623304) q[5];
sx q[5];
rz(-pi) q[5];
cx q[7],q[4];
rz(-pi/2) q[4];
sx q[7];
rz(-pi) q[7];
cx q[7],q[4];
rz(1.5700293) q[4];
sx q[7];
cx q[7],q[4];
x q[4];
rz(-2.3577285) q[4];
rz(-2.3569615) q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
sx q[10];
rz(-pi) q[10];
cx q[12],q[10];
rz(-pi/2) q[10];
sx q[12];
rz(-pi) q[12];
cx q[12],q[10];
rz(1.5704128) q[10];
sx q[12];
cx q[12],q[10];
rz(-pi) q[10];
x q[10];
rz(-1.5711798) q[12];
rz(pi/256) q[8];
cx q[11],q[8];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(-0.17180585) q[14];
cx q[14],q[13];
rz(pi/128) q[13];
cx q[14],q[13];
rz(-pi/128) q[13];
cx q[14],q[16];
rz(pi/64) q[16];
cx q[14],q[16];
cx q[14],q[11];
rz(pi/32) q[11];
cx q[14],q[11];
rz(-pi/32) q[11];
rz(-pi/64) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/16) q[19];
cx q[19],q[22];
rz(pi/16) q[22];
cx q[19],q[22];
cx q[19],q[20];
rz(pi/8) q[20];
cx q[19],q[20];
cx q[19],q[16];
rz(pi/4) q[16];
cx q[19],q[16];
rz(-pi/4) q[16];
sx q[19];
rz(pi/2) q[19];
rz(-pi/8) q[20];
rz(-pi/16) q[22];
rz(-pi/256) q[8];
sx q[8];
rz(-pi/2) q[8];
cx q[8],q[5];
rz(-pi/2) q[5];
sx q[8];
rz(-pi) q[8];
cx q[8],q[5];
rz(1.5646604) q[5];
sx q[8];
cx q[8],q[5];
rz(1.5769322) q[5];
sx q[5];
rz(-pi/2) q[5];
cx q[5],q[3];
rz(-pi/2) q[3];
sx q[5];
rz(-pi) q[5];
cx q[5],q[3];
rz(1.5677284) q[3];
sx q[5];
cx q[5],q[3];
rz(1.5677284) q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[4],q[1];
rz(pi/2048) q[1];
cx q[4],q[1];
rz(-pi/2048) q[1];
cx q[4],q[7];
x q[5];
rz(pi/2) q[5];
cx q[7],q[4];
cx q[4],q[7];
cx q[1],q[4];
cx q[4],q[1];
cx q[1],q[4];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(3.1385247) q[12];
sx q[12];
rz(2.3538935) q[7];
cx q[7],q[4];
rz(pi/4096) q[4];
cx q[7],q[4];
rz(-pi/4096) q[4];
sx q[7];
rz(-pi/2) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(-0.085902924) q[14];
cx q[14],q[13];
rz(pi/256) q[13];
cx q[14],q[13];
rz(-pi/256) q[13];
cx q[14],q[11];
rz(pi/128) q[11];
cx q[14],q[11];
rz(-pi/128) q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
rz(pi/64) q[11];
cx q[14],q[11];
rz(-pi/64) q[11];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(-0.29452431) q[19];
cx q[19],q[22];
rz(pi/32) q[22];
cx q[19],q[22];
cx q[19],q[20];
rz(pi/16) q[20];
cx q[19],q[20];
cx q[16],q[19];
cx q[19],q[16];
rz(pi/8) q[16];
cx q[16],q[14];
rz(pi/8) q[14];
cx q[16],q[14];
rz(-pi/8) q[14];
rz(pi/4) q[19];
cx q[16],q[19];
sx q[16];
rz(pi/2) q[16];
rz(-pi/4) q[19];
rz(-pi/16) q[20];
rz(-pi/32) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
sx q[16];
rz(-pi/2) q[16];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[5];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(-0.018407769) q[14];
cx q[14],q[13];
rz(pi/512) q[13];
cx q[14],q[13];
rz(-1.5769322) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[12];
rz(-pi/2) q[12];
sx q[13];
rz(-pi) q[13];
cx q[13],q[12];
rz(1.5677284) q[12];
sx q[13];
cx q[13],q[12];
rz(1.5677284) q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
rz(-pi/4) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[7];
sx q[10];
rz(-pi) q[10];
x q[13];
rz(pi/2) q[13];
sx q[14];
rz(-pi/2) q[14];
cx q[5],q[8];
rz(-pi/2) q[7];
cx q[10],q[7];
sx q[10];
rz(1.5692623) q[7];
cx q[10],q[7];
x q[10];
rz(3*pi/4) q[10];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
sx q[10];
rz(-pi) q[10];
rz(3.1385247) q[12];
sx q[12];
rz(0.78386418) q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[4];
rz(pi/4) q[7];
sx q[7];
rz(-pi) q[7];
cx q[8],q[5];
cx q[5],q[8];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
sx q[11];
rz(-pi) q[11];
cx q[14],q[11];
rz(-pi/2) q[11];
sx q[14];
rz(-pi) q[14];
cx q[14],q[11];
rz(1.5585245) q[11];
sx q[14];
cx q[14],q[11];
x q[11];
rz(3.117049) q[11];
rz(-1.5830682) q[14];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[5];
cx q[11],q[8];
rz(pi/128) q[8];
cx q[11],q[8];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
sx q[11];
rz(-pi) q[11];
rz(-pi/64) q[14];
sx q[14];
rz(-pi) q[14];
cx q[16],q[14];
rz(-pi/2) q[14];
sx q[16];
rz(-pi) q[16];
cx q[16],q[14];
rz(1.5217089) q[14];
sx q[16];
cx q[16],q[14];
x q[14];
rz(3.0925053) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/512) q[14];
sx q[14];
rz(-pi/2) q[14];
cx q[14],q[11];
rz(-pi/2) q[11];
sx q[14];
rz(-pi) q[14];
cx q[14],q[11];
rz(1.5646604) q[11];
sx q[14];
cx q[14],q[11];
rz(-pi) q[11];
x q[11];
rz(-1.5769322) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[12];
rz(-pi/2) q[12];
sx q[13];
rz(-pi) q[13];
cx q[13],q[12];
rz(1.5677284) q[12];
sx q[13];
cx q[13],q[12];
rz(1.5677284) q[12];
x q[13];
rz(pi/2) q[13];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(-pi/16384) q[12];
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
rz(0.78578166) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[7];
sx q[10];
rz(-pi) q[10];
rz(-0.78558991) q[12];
sx q[12];
rz(-pi) q[12];
cx q[15],q[18];
rz(-pi/2) q[16];
cx q[18],q[15];
cx q[15],q[18];
rz(-0.78549404) q[15];
sx q[15];
rz(-pi/2) q[15];
cx q[15],q[12];
rz(-pi/2) q[12];
sx q[15];
rz(-pi) q[15];
cx q[15],q[12];
rz(1.5707005) q[12];
sx q[15];
cx q[15],q[12];
rz(-0.78520642) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-2.3562904) q[15];
sx q[15];
rz(-pi) q[15];
cx q[18],q[21];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
rz(-pi/32) q[19];
cx q[19],q[20];
rz(pi/32) q[20];
cx q[19],q[20];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(13*pi/16) q[16];
rz(-pi/32) q[20];
cx q[21],q[18];
cx q[18],q[21];
rz(-pi/65536) q[18];
sx q[18];
rz(-pi/2) q[18];
cx q[18],q[15];
rz(-pi/2) q[15];
sx q[18];
rz(-pi) q[18];
cx q[18],q[15];
rz(pi/2) q[15];
sx q[18];
cx q[18],q[15];
x q[15];
rz(3.1414968) q[15];
rz(-1.5708443) q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
sx q[19];
rz(pi/2) q[19];
rz(-pi/2) q[7];
cx q[10],q[7];
sx q[10];
rz(1.5704128) q[7];
cx q[10],q[7];
rz(-2.356578) q[10];
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
rz(-pi) q[10];
x q[10];
rz(-1.5709881) q[12];
cx q[15],q[12];
rz(pi/32768) q[12];
cx q[15],q[12];
rz(-pi/32768) q[12];
x q[7];
rz(-2.3569615) q[7];
cx q[7],q[4];
rz(pi/4096) q[4];
cx q[7],q[4];
rz(-pi/4096) q[4];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(-pi/2048) q[15];
cx q[15],q[18];
rz(pi/2048) q[18];
cx q[15],q[18];
rz(-pi/2048) q[18];
rz(-pi/8192) q[7];
cx q[7],q[4];
rz(pi/8192) q[4];
cx q[7],q[4];
cx q[10],q[7];
rz(-pi/8192) q[4];
cx q[7],q[10];
cx q[10],q[7];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-pi/16384) q[10];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(-pi/4096) q[15];
cx q[15],q[18];
rz(pi/4096) q[18];
cx q[15],q[18];
rz(-pi/4096) q[18];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[10],q[7];
rz(pi/16384) q[7];
cx q[10],q[7];
rz(-pi/16384) q[7];
rz(-pi/128) q[8];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
sx q[11];
rz(-pi/2) q[11];
cx q[16],q[14];
rz(pi/16) q[14];
cx q[16],q[14];
rz(-pi/16) q[14];
sx q[16];
cx q[19],q[16];
rz(pi/2) q[16];
sx q[19];
rz(-pi) q[19];
cx q[19],q[16];
rz(3*pi/8) q[16];
sx q[19];
cx q[19],q[16];
x q[16];
rz(7*pi/8) q[16];
rz(3*pi/4) q[19];
cx q[19],q[22];
rz(pi/4) q[22];
cx q[19],q[22];
sx q[19];
rz(pi/2) q[19];
cx q[20],q[19];
cx q[19],q[20];
cx q[20],q[19];
rz(-pi/4) q[22];
rz(-0.036815539) q[8];
cx q[8],q[5];
rz(pi/256) q[5];
cx q[8],q[5];
rz(-pi/256) q[5];
sx q[8];
rz(-pi) q[8];
cx q[11],q[8];
sx q[11];
rz(-pi) q[11];
rz(-pi/2) q[8];
cx q[11],q[8];
sx q[11];
rz(1.5462526) q[8];
cx q[11],q[8];
rz(-pi/2) q[11];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(-0.018407769) q[11];
rz(-0.3436117) q[16];
cx q[16],q[19];
rz(pi/64) q[19];
cx q[16],q[19];
cx q[16],q[14];
rz(pi/32) q[14];
cx q[16],q[14];
rz(-pi/32) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[16],q[14];
rz(pi/16) q[14];
cx q[16],q[14];
rz(-pi/16) q[14];
rz(-pi/64) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/8) q[19];
cx q[19],q[22];
rz(pi/8) q[22];
cx q[19],q[22];
cx q[19],q[20];
rz(pi/4) q[20];
cx q[19],q[20];
sx q[19];
rz(pi/2) q[19];
rz(-pi/4) q[20];
rz(-pi/8) q[22];
x q[8];
rz(3.117049) q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[5];
cx q[11],q[8];
rz(pi/512) q[8];
cx q[11],q[8];
rz(-pi/512) q[8];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[11],q[8];
rz(pi/256) q[8];
cx q[11],q[8];
cx q[14],q[11];
cx q[11],q[14];
rz(pi/32) q[11];
rz(-0.17180585) q[14];
cx q[14],q[16];
rz(pi/128) q[16];
cx q[14],q[16];
cx q[14],q[13];
rz(pi/64) q[13];
cx q[14],q[13];
rz(-pi/64) q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[14],q[11];
rz(-pi/32) q[11];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(-pi/8192) q[15];
cx q[15],q[18];
rz(-pi/128) q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(2.3546605) q[13];
sx q[13];
rz(-1.5738643) q[14];
sx q[14];
rz(-pi/2) q[14];
rz(pi/8192) q[18];
cx q[15],q[18];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
rz(0.78463117) q[12];
sx q[12];
rz(-pi) q[12];
rz(-pi/8192) q[18];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/4) q[16];
rz(pi/16) q[19];
cx q[19],q[22];
rz(pi/16) q[22];
cx q[19],q[22];
cx q[19],q[20];
rz(pi/8) q[20];
cx q[19],q[20];
cx q[19],q[16];
rz(-pi/4) q[16];
sx q[19];
rz(pi/2) q[19];
rz(-pi/8) q[20];
rz(-pi/16) q[22];
rz(-pi/256) q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[5];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
rz(-pi/2) q[11];
sx q[11];
rz(-pi) q[11];
cx q[14],q[11];
rz(pi/2) q[11];
sx q[14];
rz(-pi) q[14];
cx q[14],q[11];
rz(1.5677284) q[11];
sx q[14];
cx q[14],q[11];
rz(1.5646604) q[11];
rz(-2.3531265) q[14];
sx q[14];
rz(-pi/2) q[14];
cx q[14],q[13];
rz(-pi/2) q[13];
sx q[14];
rz(-pi) q[14];
cx q[14],q[13];
rz(1.5692623) q[13];
sx q[14];
cx q[14],q[13];
rz(-pi/2048) q[13];
sx q[13];
rz(-pi/2) q[13];
cx q[13],q[12];
rz(-pi/2) q[12];
sx q[13];
rz(-pi) q[13];
cx q[13],q[12];
rz(1.5700293) q[12];
sx q[13];
cx q[13],q[12];
x q[12];
rz(-2.3569615) q[12];
rz(-3*pi/4) q[13];
x q[14];
rz(3*pi/4) q[14];
cx q[15],q[12];
cx q[12],q[15];
cx q[15],q[12];
sx q[12];
rz(-pi) q[12];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[5];
cx q[11],q[8];
rz(pi/512) q[8];
cx q[11],q[8];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(-pi/1024) q[11];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-pi/256) q[13];
sx q[13];
rz(-pi/2) q[13];
cx q[13],q[12];
rz(-pi/2) q[12];
sx q[13];
rz(-pi) q[13];
cx q[13],q[12];
rz(1.5585245) q[12];
sx q[13];
cx q[13],q[12];
x q[12];
rz(3.117049) q[12];
cx q[12],q[10];
rz(pi/128) q[10];
cx q[12],q[10];
rz(-pi/128) q[10];
rz(-1.5830682) q[13];
sx q[13];
rz(-pi) q[13];
rz(-pi/512) q[8];
cx q[11],q[8];
rz(pi/1024) q[8];
cx q[11],q[8];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(3.1369907) q[11];
rz(-pi/512) q[14];
sx q[14];
rz(-pi/2) q[14];
cx q[14],q[13];
rz(-pi/2) q[13];
sx q[14];
rz(-pi) q[14];
cx q[14],q[13];
rz(1.5646604) q[13];
sx q[14];
cx q[14],q[13];
rz(-pi) q[13];
x q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/256) q[12];
cx q[12],q[10];
rz(pi/256) q[10];
cx q[12],q[10];
rz(-pi/256) q[10];
rz(3.1354567) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/1024) q[8];
cx q[11],q[8];
rz(pi/2048) q[8];
cx q[11],q[8];
sx q[11];
cx q[14],q[11];
rz(-pi/2) q[11];
sx q[14];
rz(-pi) q[14];
cx q[14],q[11];
rz(1.5677284) q[11];
sx q[14];
cx q[14],q[11];
rz(1.5677284) q[11];
x q[14];
rz(pi/2) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(-pi/512) q[12];
cx q[12],q[10];
rz(pi/512) q[10];
cx q[12],q[10];
rz(-pi/512) q[10];
rz(-pi/64) q[14];
rz(-pi/2048) q[8];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
rz(pi/64) q[11];
cx q[14],q[11];
rz(-pi/64) q[11];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/128) q[14];
cx q[14],q[11];
rz(pi/128) q[11];
cx q[14],q[11];
rz(-pi/128) q[11];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(-pi/256) q[14];
cx q[14],q[11];
rz(pi/256) q[11];
cx q[14],q[11];
rz(-pi/256) q[11];
rz(-0.29452431) q[19];
cx q[19],q[22];
rz(pi/32) q[22];
cx q[19],q[22];
cx q[19],q[20];
rz(pi/16) q[20];
cx q[19],q[20];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/8) q[13];
cx q[13],q[12];
rz(pi/8) q[12];
cx q[13],q[12];
rz(-pi/8) q[12];
cx q[13],q[14];
rz(pi/4) q[14];
cx q[13],q[14];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi/4) q[14];
rz(-0.14726216) q[19];
rz(-pi/16) q[20];
rz(-pi/32) q[22];
cx q[19],q[22];
rz(pi/64) q[22];
cx q[19],q[22];
cx q[19],q[20];
rz(pi/32) q[20];
cx q[19],q[20];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[14],q[16];
cx q[16],q[14];
rz(-3*pi/16) q[14];
cx q[14],q[13];
rz(pi/16) q[13];
cx q[14],q[13];
rz(-pi/16) q[13];
rz(pi/8) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/4) q[13];
cx q[13],q[12];
rz(pi/4) q[12];
cx q[13],q[12];
rz(-pi/4) q[12];
sx q[13];
rz(pi/2) q[13];
rz(-pi/8) q[16];
rz(-0.073631078) q[19];
rz(-pi/32) q[20];
rz(-pi/64) q[22];
cx q[19],q[22];
rz(pi/128) q[22];
cx q[19],q[22];
cx q[19],q[20];
rz(pi/64) q[20];
cx q[19],q[20];
cx q[16],q[19];
cx q[19],q[16];
rz(-0.29452431) q[16];
cx q[16],q[14];
rz(pi/32) q[14];
cx q[16],q[14];
rz(-pi/32) q[14];
rz(pi/16) q[19];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/8) q[13];
cx q[13],q[12];
rz(pi/8) q[12];
cx q[13],q[12];
rz(-pi/8) q[12];
rz(pi/4) q[14];
cx q[13],q[14];
sx q[13];
rz(pi/2) q[13];
rz(-pi/4) q[14];
rz(-pi/16) q[19];
rz(-pi/64) q[20];
rz(-pi/128) q[22];
barrier q[3],q[10],q[5],q[8],q[21],q[17],q[13],q[15],q[26],q[12],q[2],q[16],q[22],q[18],q[4],q[25],q[24],q[1],q[0],q[11],q[9],q[6],q[7],q[20],q[19],q[14],q[23];
measure q[24] -> c[0];
measure q[23] -> c[1];
measure q[21] -> c[2];
measure q[4] -> c[3];
measure q[7] -> c[4];
measure q[18] -> c[5];
measure q[15] -> c[6];
measure q[5] -> c[7];
measure q[8] -> c[8];
measure q[10] -> c[9];
measure q[11] -> c[10];
measure q[22] -> c[11];
measure q[20] -> c[12];
measure q[16] -> c[13];
measure q[19] -> c[14];
measure q[12] -> c[15];
measure q[14] -> c[16];
measure q[13] -> c[17];
