OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[7];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[14];
sx q[14];
rz(4.3291427) q[14];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[12];
rz(-5.18559484832958) q[12];
cx q[15],q[12];
cx q[10],q[12];
rz(-5.18559484832958) q[12];
cx q[10],q[12];
rz(pi/2) q[12];
sx q[12];
rz(-5.13152680581451) q[12];
sx q[12];
rz(5*pi/2) q[12];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[14],q[16];
rz(-5.18559484832958) q[16];
cx q[14],q[16];
sx q[14];
rz(-pi) q[14];
rz(-2.2842368) q[16];
sx q[16];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[15];
rz(-5.18559484832958) q[15];
cx q[18],q[15];
rz(pi/2) q[15];
sx q[15];
rz(-5.13152680581451) q[15];
sx q[15];
rz(5*pi/2) q[15];
cx q[15],q[12];
rz(-0.992428798165926) q[12];
cx q[15],q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
rz(-5.18559484832958) q[12];
cx q[15],q[12];
rz(pi/2) q[12];
sx q[12];
rz(-5.13152680581451) q[12];
sx q[12];
rz(5*pi/2) q[12];
cx q[12],q[10];
rz(-0.992428798165926) q[10];
cx q[12],q[10];
rz(pi/2) q[10];
sx q[10];
rz(6.47159240889839) q[10];
sx q[10];
rz(5*pi/2) q[10];
rz(-pi/2) q[15];
sx q[15];
rz(-1.1516585) q[15];
sx q[15];
cx q[15],q[18];
rz(-0.992428798165926) q[18];
cx q[15],q[18];
cx q[15],q[12];
rz(-0.992428798165926) q[12];
cx q[15],q[12];
rz(pi/2) q[12];
sx q[12];
rz(6.47159240889839) q[12];
sx q[12];
rz(5*pi/2) q[12];
sx q[15];
rz(6.47159240889839) q[15];
sx q[15];
rz(5*pi/2) q[15];
rz(pi/2) q[18];
sx q[18];
rz(6.47159240889839) q[18];
sx q[18];
rz(5*pi/2) q[18];
sx q[19];
cx q[19],q[16];
rz(-pi/2) q[16];
sx q[19];
rz(-pi) q[19];
cx q[19],q[16];
rz(0.47320587) q[16];
sx q[19];
cx q[19],q[16];
rz(1.0584763) q[16];
sx q[16];
rz(-pi/2) q[16];
cx q[16],q[14];
rz(-pi/2) q[14];
sx q[16];
rz(-pi) q[16];
cx q[16],q[14];
rz(0.47320587) q[14];
sx q[16];
cx q[16],q[14];
rz(2.6292726) q[14];
sx q[14];
rz(-1.9899342) q[14];
sx q[14];
rz(pi/2) q[14];
rz(0.38324625) q[16];
sx q[16];
rz(-1.1516585) q[16];
sx q[16];
rz(-pi/2) q[16];
rz(0.85735589) q[19];
sx q[19];
rz(-1.9899342) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[16],q[19];
rz(-0.992428798165926) q[19];
cx q[16],q[19];
cx q[16],q[14];
cx q[14],q[16];
rz(-0.992428798165926) q[14];
cx q[16],q[19];
rz(-0.992428798165926) q[19];
cx q[16],q[19];
cx q[16],q[14];
rz(pi/2) q[14];
sx q[14];
rz(6.47159240889839) q[14];
sx q[14];
rz(5*pi/2) q[14];
rz(pi/2) q[16];
sx q[16];
rz(6.47159240889839) q[16];
sx q[16];
rz(5*pi/2) q[16];
rz(pi/2) q[19];
sx q[19];
rz(6.47159240889839) q[19];
sx q[19];
rz(5*pi/2) q[19];
barrier q[15],q[24],q[21],q[1],q[7],q[4],q[12],q[13],q[16],q[19],q[22],q[25],q[5],q[2],q[8],q[11],q[17],q[14],q[20],q[26],q[0],q[23],q[3],q[6],q[10],q[9],q[18];
measure q[15] -> meas[0];
measure q[16] -> meas[1];
measure q[14] -> meas[2];
measure q[12] -> meas[3];
measure q[19] -> meas[4];
measure q[18] -> meas[5];
measure q[10] -> meas[6];
