OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[7];
sx q[10];
rz(-2.2726946) q[10];
sx q[10];
rz(-2.5726828) q[10];
sx q[12];
rz(-2.1762562) q[12];
sx q[12];
rz(-2.1974382) q[12];
sx q[13];
rz(-2.6640766) q[13];
sx q[13];
rz(-2.8222765) q[13];
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
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[12],q[10];
sx q[12];
rz(-2.9143106) q[12];
sx q[12];
rz(-2.8023615) q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[13],q[12];
sx q[13];
rz(-3.1263069) q[13];
sx q[13];
rz(-2.6117703) q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[10],q[12];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[14],q[13];
sx q[14];
rz(-2.2415542) q[14];
sx q[14];
rz(-2.6884172) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[19],q[22];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[16],q[14];
sx q[16];
rz(-2.1591853) q[16];
sx q[16];
rz(-2.7475685) q[16];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[16],q[14];
sx q[16];
rz(-2.3213597) q[16];
sx q[16];
rz(-2.5759578) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[10],q[12];
cx q[22],q[19];
sx q[19];
rz(-2.9107955) q[19];
sx q[19];
rz(-2.8233689) q[19];
sx q[22];
rz(-3.0485895) q[22];
sx q[22];
rz(-2.4413751) q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
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
cx q[12],q[13];
sx q[12];
rz(-2.8557002) q[12];
sx q[12];
rz(-2.6596222) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[10],q[12];
sx q[10];
rz(-2.7098346) q[10];
sx q[10];
rz(-2.2604624) q[10];
sx q[12];
rz(-2.8554254) q[12];
sx q[12];
rz(-2.6928765) q[12];
sx q[14];
rz(-2.7486122) q[14];
sx q[14];
rz(-2.2535712) q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[10];
sx q[12];
rz(-2.9080516) q[12];
sx q[12];
rz(-2.3496058) q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[22],q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
sx q[13];
rz(-2.3504657) q[13];
sx q[13];
rz(-2.5634954) q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[13];
sx q[14];
rz(-2.9190639) q[14];
sx q[14];
rz(-2.821727) q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[19],q[22];
cx q[22],q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[14];
sx q[16];
rz(-2.3319383) q[16];
sx q[16];
rz(-2.7477289) q[16];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[16];
sx q[19];
rz(-2.8266579) q[19];
sx q[19];
rz(-2.9038495) q[19];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[16],q[19];
sx q[16];
rz(-3.1338583) q[16];
sx q[16];
rz(-2.9072391) q[16];
sx q[19];
rz(-2.8933932) q[19];
sx q[19];
rz(-2.5643262) q[19];
barrier q[22],q[20],q[17],q[23],q[3],q[26],q[0],q[6],q[9],q[15],q[16],q[18],q[21],q[24],q[1],q[4],q[7],q[14],q[19],q[12],q[13],q[10],q[25],q[2],q[8],q[5],q[11];
measure q[10] -> meas[0];
measure q[12] -> meas[1];
measure q[13] -> meas[2];
measure q[14] -> meas[3];
measure q[22] -> meas[4];
measure q[16] -> meas[5];
measure q[19] -> meas[6];
