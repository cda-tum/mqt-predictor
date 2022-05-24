OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[4];
sx q[12];
rz(0.0064574529) q[12];
sx q[12];
rz(-pi) q[15];
sx q[15];
rz(2.3929282) q[15];
sx q[15];
rz(-pi) q[18];
sx q[18];
rz(2.6142168) q[18];
sx q[18];
cx q[18],q[15];
rz(-pi) q[21];
sx q[21];
rz(2.9669986) q[21];
sx q[21];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(1.9138092) q[15];
sx q[15];
rz(-pi) q[15];
cx q[12],q[15];
sx q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[15];
cx q[12],q[15];
rz(-pi/2) q[12];
sx q[12];
rz(-2.7985798) q[12];
sx q[12];
rz(-1.9509085) q[12];
rz(-pi/2) q[15];
sx q[15];
cx q[18],q[21];
rz(-1.5728355) q[18];
sx q[18];
rz(-pi) q[18];
cx q[15],q[18];
sx q[15];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[18];
cx q[15],q[18];
rz(2.6857491) q[15];
sx q[15];
rz(-1.5698986) q[15];
sx q[15];
rz(-3.1397617) q[15];
cx q[12],q[15];
sx q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[15];
cx q[12],q[15];
rz(-pi) q[12];
sx q[12];
rz(-pi) q[12];
rz(-pi) q[15];
sx q[15];
rz(-pi/2) q[15];
rz(pi/2) q[18];
sx q[18];
rz(-pi) q[18];
cx q[21],q[18];
rz(-pi) q[18];
sx q[18];
rz(2.8271351) q[18];
sx q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
cx q[12],q[15];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
sx q[21];
rz(-2.6569536) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[18],q[21];
sx q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[21];
cx q[18],q[21];
rz(-pi) q[18];
sx q[18];
rz(-pi/2) q[18];
cx q[15],q[18];
sx q[15];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[18];
cx q[15],q[18];
rz(-pi) q[15];
sx q[15];
cx q[15],q[12];
rz(-pi) q[12];
sx q[12];
rz(2.2268576) q[12];
sx q[12];
sx q[15];
rz(0.23400615) q[15];
sx q[15];
rz(2.4633906) q[18];
sx q[18];
rz(3.0585417) q[21];
sx q[21];
cx q[21],q[18];
rz(pi/2) q[18];
sx q[18];
rz(-pi) q[18];
cx q[15],q[18];
sx q[15];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[18];
cx q[15],q[18];
rz(pi/2) q[15];
sx q[15];
cx q[12],q[15];
cx q[15],q[12];
rz(-pi) q[12];
sx q[12];
rz(2.6876093) q[12];
sx q[12];
rz(-pi) q[15];
x q[15];
rz(pi/2) q[18];
sx q[18];
cx q[18],q[21];
cx q[21],q[18];
rz(1.5692793) q[18];
sx q[18];
rz(-pi) q[18];
cx q[15],q[18];
sx q[15];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[18];
cx q[15],q[18];
rz(-2.5437107) q[15];
sx q[15];
rz(-1.5716503) q[15];
sx q[15];
rz(-1.5695425) q[15];
rz(pi/2) q[18];
sx q[18];
rz(-pi) q[18];
cx q[21],q[18];
rz(-pi) q[18];
sx q[18];
rz(2.2402546) q[18];
sx q[18];
rz(-pi) q[21];
sx q[21];
rz(2.9202491) q[21];
sx q[21];
barrier q[2],q[25],q[5],q[8],q[14],q[11],q[17],q[20],q[26],q[23],q[0],q[3],q[6],q[18],q[9],q[12],q[21],q[15],q[24],q[1],q[7],q[4],q[10],q[13],q[19],q[16],q[22];
measure q[15] -> meas[0];
measure q[12] -> meas[1];
measure q[21] -> meas[2];
measure q[18] -> meas[3];
