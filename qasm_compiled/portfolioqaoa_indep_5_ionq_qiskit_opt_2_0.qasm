OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg meas[5];
ry(pi/2) q[0];
rx(pi/2) q[0];
ry(pi/2) q[1];
rx(pi/2) q[1];
ry(pi/2) q[2];
rx(pi/2) q[2];
ry(pi/2) q[3];
rx(pi/2) q[3];
rx(-14.502839) q[4];
rxx(pi/2) q[4],q[3];
rz(-20.06105) q[3];
rxx(pi/2) q[4],q[3];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rxx(pi/2) q[4],q[2];
rz(-20.061047) q[2];
rxx(pi/2) q[4],q[2];
rx(-pi) q[2];
rxx(pi/2) q[3],q[2];
rz(-20.060795) q[2];
rx(-5*pi/2) q[3];
rxx(pi/2) q[3],q[2];
rx(-pi/2) q[2];
ry(pi/2) q[2];
rxx(pi/2) q[4],q[1];
rz(-20.060977) q[1];
rxx(pi/2) q[4],q[1];
rx(-pi) q[1];
rxx(pi/2) q[3],q[1];
rz(-20.060922) q[1];
rxx(pi/2) q[3],q[1];
rx(-pi) q[1];
rxx(pi/2) q[2],q[1];
rz(-20.060949) q[1];
rx(-3*pi/2) q[2];
rxx(pi/2) q[2],q[1];
rx(-pi/2) q[1];
ry(pi/2) q[1];
rxx(pi/2) q[4],q[0];
rz(-20.060977) q[0];
rxx(pi/2) q[4],q[0];
rx(-pi) q[0];
rxx(pi/2) q[3],q[0];
rz(-20.060356) q[0];
rxx(pi/2) q[3],q[0];
rx(-pi) q[0];
rxx(pi/2) q[2],q[0];
rz(-20.061069) q[0];
rxx(pi/2) q[2],q[0];
rx(-pi) q[0];
rxx(pi/2) q[1],q[0];
rz(-20.061192) q[0];
rx(-pi/2) q[1];
rxx(pi/2) q[1],q[0];
ry(-1.1804617) q[0];
rx(-1.9750989) q[0];
rz(2.0009741) q[1];
ry(1.2323702) q[1];
rz(2.9904199) q[1];
rz(1.9979978) q[2];
ry(1.2514514) q[2];
rz(2.9996344) q[2];
rz(2.0018391) q[3];
ry(1.2270653) q[3];
rz(2.9878341) q[3];
rz(-1.1664938) q[4];
rxx(pi/2) q[4],q[3];
rx(-pi/2) q[3];
rz(-11.55624) q[3];
rx(-13.580171) q[4];
rxx(pi/2) q[4],q[3];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rxx(pi/2) q[4],q[2];
rx(-pi/2) q[2];
rz(-11.556238) q[2];
rxx(pi/2) q[4],q[2];
rx(-pi) q[2];
rxx(pi/2) q[3],q[2];
rz(-11.556094) q[2];
rx(-5*pi/2) q[3];
rxx(pi/2) q[3],q[2];
rx(-pi/2) q[2];
ry(pi/2) q[2];
rxx(pi/2) q[4],q[1];
rx(-pi/2) q[1];
rz(-11.556198) q[1];
rxx(pi/2) q[4],q[1];
rx(-pi) q[1];
rxx(pi/2) q[3],q[1];
rz(-11.556166) q[1];
rxx(pi/2) q[3],q[1];
rx(-pi) q[1];
rxx(pi/2) q[2],q[1];
rz(-11.556182) q[1];
rx(-3*pi/2) q[2];
rxx(pi/2) q[2],q[1];
rx(-pi/2) q[1];
ry(pi/2) q[1];
rxx(pi/2) q[4],q[0];
rz(-11.556198) q[0];
rxx(pi/2) q[4],q[0];
rx(-pi) q[0];
rxx(pi/2) q[3],q[0];
rz(-11.555841) q[0];
rxx(pi/2) q[3],q[0];
rx(-pi) q[0];
rxx(pi/2) q[2],q[0];
rz(-11.556251) q[0];
rxx(pi/2) q[2],q[0];
rx(-pi) q[0];
rxx(pi/2) q[1],q[0];
rz(-11.556322) q[0];
rx(-pi/2) q[1];
rxx(pi/2) q[1],q[0];
ry(1.0280066) q[0];
rx(-1.558797) q[0];
rz(-1.5566797) q[1];
ry(1.0160296) q[1];
rz(-0.0074362332) q[1];
rz(-1.5565721) q[2];
ry(1.0039455) q[2];
rz(-0.0076384557) q[2];
rz(-1.556709) q[3];
ry(1.0193921) q[3];
rz(-0.0073805024) q[3];
rz(-1.5827957) q[4];
rxx(pi/2) q[4],q[3];
rx(-pi/2) q[3];
rz(25.798471) q[3];
rx(-7*pi/2) q[4];
rxx(pi/2) q[4],q[3];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rxx(pi/2) q[4],q[2];
rx(-pi/2) q[2];
rz(25.798466) q[2];
rxx(pi/2) q[4],q[2];
rx(-pi) q[2];
rxx(pi/2) q[3],q[2];
rz(25.798143) q[2];
rx(-5*pi/2) q[3];
rxx(pi/2) q[3],q[2];
rx(-pi/2) q[2];
ry(pi/2) q[2];
rxx(pi/2) q[4],q[1];
rx(-pi/2) q[1];
rz(25.798377) q[1];
rxx(pi/2) q[4],q[1];
rx(-pi) q[1];
rxx(pi/2) q[3],q[1];
rz(25.798305) q[1];
rxx(pi/2) q[3],q[1];
rx(-pi) q[1];
rxx(pi/2) q[2],q[1];
rz(25.79834) q[1];
rx(-3*pi/2) q[2];
rxx(pi/2) q[2],q[1];
rx(-pi/2) q[1];
ry(pi/2) q[1];
rxx(pi/2) q[4],q[0];
rz(25.798376) q[0];
rxx(pi/2) q[4],q[0];
rx(-pi) q[0];
rxx(pi/2) q[3],q[0];
rz(25.797578) q[0];
rxx(pi/2) q[3],q[0];
rx(-pi) q[0];
rxx(pi/2) q[2],q[0];
rz(25.798494) q[0];
rxx(pi/2) q[2],q[0];
rx(-pi) q[0];
rxx(pi/2) q[1],q[0];
rz(25.798652) q[0];
rx(-pi/2) q[1];
rxx(pi/2) q[1],q[0];
ry(0.62582139) q[0];
rx(-0.65206353) q[0];
rz(-2.4696305) q[1];
ry(0.88734528) q[1];
rz(0.67098156) q[1];
rz(-2.4528436) q[2];
ry(0.90431851) q[2];
rz(0.64411882) q[2];
rz(-2.4744473) q[3];
ry(0.88268589) q[3];
rz(0.67858764) q[3];
rz(-2.466535) q[4];
ry(0.89038788) q[4];
rz(0.66607034) q[4];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
