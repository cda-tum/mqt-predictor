OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[4];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[12];
sx q[12];
rz(0.0021580268) q[12];
cx q[12],q[10];
rz(9.8849116) q[10];
cx q[12],q[10];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[12],q[13];
rz(9.8845542) q[13];
cx q[12],q[13];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[12],q[15];
rz(9.8847922) q[15];
cx q[12],q[15];
sx q[12];
rz(5.96623295358979) q[12];
sx q[12];
rz(7*pi/2) q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[13];
rz(9.8845862) q[13];
cx q[12],q[13];
cx q[12],q[15];
rz(9.8848894) q[15];
cx q[12],q[15];
rz(-1.5742057) q[12];
sx q[12];
rz(5.96623295358979) q[12];
sx q[12];
rz(7*pi/2) q[12];
cx q[10],q[12];
rz(2.0986142) q[12];
cx q[10],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[12],q[15];
rz(9.8847845) q[15];
cx q[12],q[15];
rz(-1.5842812) q[12];
sx q[12];
rz(5.96623295358979) q[12];
sx q[12];
rz(7*pi/2) q[12];
cx q[10],q[12];
rz(2.0985384) q[12];
cx q[10],q[12];
cx q[13],q[12];
rz(2.0985451) q[12];
cx q[13],q[12];
rz(-1.5772288) q[15];
sx q[15];
rz(5.96623295358979) q[15];
sx q[15];
rz(7*pi/2) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[10],q[12];
rz(2.0985889) q[12];
cx q[10],q[12];
rz(1.5712545) q[10];
sx q[10];
rz(6.26703835358979) q[10];
sx q[10];
rz(5*pi/2) q[10];
cx q[13],q[12];
rz(2.0986095) q[12];
cx q[13],q[12];
rz(1.5700725) q[13];
sx q[13];
rz(6.26703835358979) q[13];
sx q[13];
rz(5*pi/2) q[13];
cx q[15],q[12];
rz(2.0985872) q[12];
cx q[15],q[12];
rz(1.5694307) q[12];
sx q[12];
rz(6.26703835358979) q[12];
sx q[12];
rz(5*pi/2) q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[13];
rz(6.3498524) q[13];
cx q[12],q[13];
rz(1.5679334) q[15];
sx q[15];
rz(6.26703835358979) q[15];
sx q[15];
rz(5*pi/2) q[15];
cx q[12],q[15];
rz(6.3496228) q[15];
cx q[12],q[15];
cx q[12],q[10];
rz(6.3497757) q[10];
cx q[12],q[10];
rz(1.5721826) q[12];
sx q[12];
rz(3.47801267358979) q[12];
sx q[12];
rz(5*pi/2) q[12];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[13],q[12];
rz(6.3496434) q[12];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[13],q[12];
rz(6.3498381) q[12];
cx q[13],q[12];
cx q[10],q[12];
rz(6.3497708) q[12];
cx q[10],q[12];
rz(1.5621339) q[10];
sx q[10];
rz(3.47801267358979) q[10];
sx q[10];
rz(5*pi/2) q[10];
rz(1.5666643) q[12];
sx q[12];
rz(3.47801267358979) q[12];
sx q[12];
rz(5*pi/2) q[12];
rz(1.5686062) q[13];
sx q[13];
rz(3.47801267358979) q[13];
sx q[13];
rz(5*pi/2) q[13];
barrier q[15],q[18],q[12],q[21],q[24],q[4],q[1],q[7],q[10],q[13],q[16],q[19],q[25],q[22],q[2],q[5],q[11],q[8],q[14],q[17],q[23],q[20],q[26],q[0],q[6],q[3],q[9];
measure q[12] -> meas[0];
measure q[10] -> meas[1];
measure q[13] -> meas[2];
measure q[15] -> meas[3];
