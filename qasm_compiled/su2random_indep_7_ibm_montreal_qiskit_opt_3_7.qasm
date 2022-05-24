OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[7];
rz(-2.3610477) q[10];
sx q[10];
rz(-0.69885236) q[10];
sx q[10];
rz(1.1799467) q[12];
sx q[12];
rz(0.50314683) q[12];
sx q[12];
rz(0.35637806) q[13];
sx q[13];
rz(0.45153049) q[13];
sx q[13];
sx q[14];
rz(-2.195071) q[14];
sx q[14];
rz(-3.098849) q[14];
sx q[16];
rz(-2.2934021) q[16];
sx q[16];
rz(-2.8012634) q[16];
sx q[19];
rz(-3.0938139) q[19];
sx q[19];
rz(-2.6721259) q[19];
cx q[19],q[16];
sx q[22];
rz(-2.3914498) q[22];
sx q[22];
rz(-2.8508359) q[22];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[14],q[16];
cx q[16],q[14];
rz(-1.2522728) q[14];
sx q[14];
cx q[13],q[14];
sx q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[14];
cx q[13],q[14];
rz(-pi/2) q[13];
sx q[13];
rz(3.1369601) q[13];
sx q[13];
cx q[12],q[13];
sx q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[13];
cx q[12],q[13];
rz(-pi/2) q[12];
sx q[12];
rz(1.8532091) q[12];
sx q[12];
cx q[10],q[12];
sx q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[12];
cx q[10],q[12];
rz(-0.12915998) q[10];
sx q[10];
rz(-1.3832568) q[10];
sx q[10];
rz(2.8723564) q[10];
sx q[12];
rz(1.0035253) q[12];
sx q[12];
rz(pi/2) q[12];
sx q[13];
rz(2.4344661) q[13];
sx q[13];
rz(-pi/2) q[13];
sx q[14];
rz(1.7318287) q[14];
sx q[14];
rz(-pi/2) q[14];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
rz(-pi) q[16];
sx q[16];
cx q[14],q[16];
sx q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[16];
cx q[14],q[16];
rz(-pi/2) q[14];
cx q[13],q[14];
sx q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[14];
cx q[13],q[14];
rz(pi/2) q[13];
sx q[13];
rz(1.9282962) q[13];
sx q[13];
cx q[12],q[13];
sx q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[13];
cx q[12],q[13];
rz(0.014319454) q[12];
sx q[12];
rz(-1.5654475) q[12];
sx q[12];
rz(-1.3984355) q[12];
cx q[10],q[12];
rz(-pi) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[13];
sx q[13];
rz(-pi) q[13];
rz(pi/2) q[14];
sx q[14];
x q[14];
rz(pi/2) q[16];
sx q[16];
cx q[19],q[22];
cx q[22],q[19];
rz(-pi) q[19];
sx q[19];
cx q[16],q[19];
sx q[16];
rz(-pi/2) q[16];
sx q[16];
rz(pi/2) q[19];
cx q[16],q[19];
rz(-pi/2) q[16];
cx q[14],q[16];
sx q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[16];
cx q[14],q[16];
rz(-pi/2) q[14];
sx q[14];
rz(1.5683251) q[14];
sx q[14];
cx q[13],q[14];
sx q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[14];
cx q[13],q[14];
rz(2.2415557) q[13];
sx q[13];
rz(-1.5727321) q[13];
sx q[13];
rz(-2.6899532) q[13];
cx q[12],q[13];
sx q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[13];
cx q[12],q[13];
rz(-pi) q[12];
sx q[12];
rz(-pi) q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(-pi) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi) q[13];
sx q[13];
rz(-pi/2) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[14];
sx q[14];
rz(-pi) q[14];
rz(pi/2) q[16];
sx q[16];
x q[16];
rz(-pi/2) q[19];
sx q[19];
rz(-pi/2) q[19];
cx q[19],q[22];
sx q[19];
rz(-pi) q[19];
cx q[16],q[19];
sx q[16];
rz(-pi/2) q[16];
sx q[16];
rz(pi/2) q[19];
cx q[16],q[19];
rz(pi/2) q[16];
sx q[16];
cx q[16],q[14];
rz(-pi) q[14];
x q[14];
sx q[16];
rz(-2.1591853) q[16];
sx q[16];
rz(-2.7475685) q[16];
rz(pi/2) q[19];
sx q[19];
x q[19];
rz(pi/2) q[22];
sx q[22];
rz(-pi) q[22];
cx q[19],q[22];
sx q[19];
rz(-pi/2) q[19];
sx q[19];
rz(pi/2) q[22];
cx q[19],q[22];
rz(pi/2) q[19];
sx q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(-pi) q[16];
sx q[16];
rz(-pi) q[16];
cx q[14],q[16];
sx q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[16];
cx q[14],q[16];
sx q[14];
rz(0.7505634) q[14];
sx q[14];
rz(0.56563488) q[14];
rz(pi/2) q[16];
sx q[16];
rz(-pi) q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(pi/2) q[14];
cx q[13],q[14];
sx q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[14];
cx q[13],q[14];
rz(-pi) q[13];
sx q[13];
rz(-pi/2) q[13];
cx q[12],q[13];
sx q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[13];
cx q[12],q[13];
rz(-pi) q[12];
sx q[12];
rz(-pi/2) q[12];
cx q[10],q[12];
sx q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[12];
cx q[10],q[12];
rz(-pi) q[10];
sx q[10];
rz(-pi/2) q[12];
sx q[12];
rz(-pi) q[12];
rz(-pi/2) q[13];
sx q[13];
rz(-pi) q[13];
rz(-pi/2) q[14];
sx q[14];
rz(-pi) q[14];
rz(pi/2) q[16];
cx q[14],q[16];
sx q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[16];
cx q[14],q[16];
rz(-pi) q[14];
sx q[14];
rz(-pi/2) q[14];
cx q[13],q[14];
sx q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[14];
cx q[13],q[14];
rz(-pi) q[13];
sx q[13];
rz(-pi/2) q[13];
cx q[12],q[13];
sx q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[13];
cx q[12],q[13];
rz(-pi) q[12];
sx q[12];
rz(-pi) q[12];
cx q[10],q[12];
sx q[10];
rz(pi/2) q[10];
rz(-pi/2) q[13];
sx q[13];
rz(-pi) q[13];
rz(-pi/2) q[14];
sx q[14];
rz(-pi) q[14];
rz(-pi/2) q[16];
sx q[16];
rz(-pi) q[16];
rz(-pi) q[19];
x q[19];
rz(pi/2) q[22];
sx q[22];
rz(-2.1412993) q[22];
sx q[22];
cx q[19],q[22];
sx q[19];
rz(-pi/2) q[19];
sx q[19];
rz(pi/2) q[22];
cx q[19],q[22];
rz(0.078339964) q[19];
sx q[19];
rz(-1.6209718) q[19];
sx q[19];
rz(1.2687537) q[19];
cx q[16],q[19];
sx q[16];
rz(-pi/2) q[16];
sx q[16];
rz(pi/2) q[19];
cx q[16],q[19];
rz(-pi) q[16];
sx q[16];
rz(-pi/2) q[16];
cx q[14],q[16];
sx q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[16];
cx q[14],q[16];
rz(-pi) q[14];
sx q[14];
rz(-pi/2) q[14];
cx q[13],q[14];
sx q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[14];
cx q[13],q[14];
rz(-pi) q[13];
sx q[13];
rz(-pi) q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/2) q[12];
cx q[10],q[12];
sx q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[12];
cx q[10],q[12];
rz(-pi) q[10];
sx q[10];
rz(-pi) q[10];
sx q[12];
rz(-3.0778954) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi) q[14];
sx q[14];
rz(-pi/2) q[14];
rz(-pi) q[16];
sx q[16];
rz(-pi/2) q[16];
rz(-pi) q[19];
sx q[19];
rz(-pi/2) q[19];
rz(-1.8015935) q[22];
sx q[22];
rz(0.31822379) q[22];
cx q[19],q[22];
sx q[19];
rz(-2.4378277) q[19];
sx q[19];
rz(-2.9187683) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[22];
sx q[19];
rz(-2.1776651) q[19];
sx q[19];
rz(-2.4128858) q[19];
cx q[16],q[19];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[16],q[19];
sx q[16];
rz(-2.4508864) q[16];
sx q[16];
rz(-2.4850569) q[16];
cx q[14],q[16];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/2) q[13];
cx q[12],q[13];
sx q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[13];
cx q[12],q[13];
sx q[12];
cx q[10],q[12];
sx q[10];
rz(0.43175804) q[10];
sx q[10];
rz(2.2604624) q[10];
sx q[12];
rz(-2.8554254) q[12];
sx q[12];
rz(-2.6928765) q[12];
rz(-1.85614) q[13];
sx q[13];
rz(-1.5887486) q[13];
sx q[13];
rz(-2.1138851) q[13];
sx q[14];
rz(-2.7486122) q[14];
sx q[14];
rz(-2.2535712) q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(pi/2) q[14];
sx q[14];
rz(-pi) q[14];
cx q[13],q[14];
sx q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[14];
cx q[13],q[14];
rz(pi/2) q[13];
sx q[13];
cx q[12],q[13];
cx q[13],q[12];
sx q[12];
rz(-pi) q[12];
cx q[10],q[12];
sx q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[12];
cx q[10],q[12];
sx q[10];
rz(1.8043373) q[10];
sx q[10];
rz(-2.3496058) q[10];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[14];
sx q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[22],q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[14],q[16];
cx q[16],q[14];
x q[16];
rz(-pi) q[19];
x q[19];
sx q[22];
cx q[19],q[22];
sx q[19];
rz(-pi/2) q[19];
sx q[19];
rz(pi/2) q[22];
cx q[19],q[22];
rz(-pi) q[19];
sx q[19];
rz(-pi/2) q[19];
cx q[16],q[19];
sx q[16];
rz(-pi/2) q[16];
sx q[16];
rz(pi/2) q[19];
cx q[16],q[19];
rz(-pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[13],q[14];
cx q[14],q[13];
rz(1.5484068) q[13];
sx q[13];
cx q[12],q[13];
sx q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[13];
cx q[12],q[13];
rz(0.79100166) q[12];
sx q[12];
rz(-1.5867179) q[12];
sx q[12];
rz(-0.97695691) q[12];
rz(pi/2) q[13];
sx q[13];
rz(-pi) q[13];
cx q[14],q[16];
cx q[16],q[14];
rz(-1.5806623) q[14];
sx q[14];
rz(-pi) q[14];
cx q[13],q[14];
sx q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[14];
cx q[13],q[14];
rz(2.9190744) q[13];
sx q[13];
rz(-1.568619) q[13];
sx q[13];
rz(1.9002847) q[13];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[19];
sx q[19];
sx q[22];
rz(-pi/2) q[22];
cx q[19],q[22];
sx q[19];
rz(-pi/2) q[19];
sx q[19];
rz(pi/2) q[22];
cx q[19],q[22];
rz(pi/2) q[19];
sx q[19];
cx q[16],q[19];
cx q[19],q[16];
rz(-1.8014307) q[16];
sx q[16];
cx q[14],q[16];
sx q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[16];
cx q[14],q[16];
rz(-0.79624543) q[14];
sx q[14];
rz(-1.4045171) q[14];
sx q[14];
rz(2.1252274) q[14];
rz(-pi/2) q[16];
sx q[16];
rz(-pi) q[16];
rz(pi/2) q[22];
sx q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[19],q[16];
sx q[19];
rz(pi/2) q[19];
rz(-pi/2) q[22];
cx q[19],q[22];
sx q[19];
rz(-pi/2) q[19];
sx q[19];
rz(pi/2) q[22];
cx q[19],q[22];
rz(-pi) q[19];
sx q[19];
cx q[16],q[19];
sx q[16];
rz(-3.1338583) q[16];
sx q[16];
rz(-2.9072391) q[16];
sx q[19];
rz(-2.8933932) q[19];
sx q[19];
rz(-2.5643262) q[19];
rz(-pi/2) q[22];
sx q[22];
rz(-1.8857311) q[22];
sx q[22];
rz(0.23774313) q[22];
barrier q[22],q[20],q[17],q[23],q[3],q[26],q[0],q[6],q[9],q[15],q[16],q[18],q[21],q[24],q[1],q[4],q[7],q[14],q[19],q[12],q[13],q[10],q[25],q[2],q[8],q[5],q[11];
measure q[10] -> meas[0];
measure q[12] -> meas[1];
measure q[13] -> meas[2];
measure q[14] -> meas[3];
measure q[22] -> meas[4];
measure q[16] -> meas[5];
measure q[19] -> meas[6];
