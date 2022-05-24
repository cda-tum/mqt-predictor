OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg meas[7];
ry(pi/2) q[0];
rx(pi/2) q[0];
ry(pi/2) q[1];
rx(pi/2) q[1];
ry(pi/2) q[2];
rx(pi/2) q[2];
ry(pi/2) q[3];
rx(pi/2) q[3];
ry(pi/2) q[4];
rx(pi/2) q[4];
ry(pi/2) q[5];
rx(pi/2) q[5];
rx(-18.62357) q[6];
rxx(pi/2) q[6],q[5];
rz(9.1929231) q[5];
rxx(pi/2) q[6],q[5];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rxx(pi/2) q[6],q[4];
rz(9.1928846) q[4];
rxx(pi/2) q[6],q[4];
rx(-pi) q[4];
rxx(pi/2) q[5],q[4];
rz(9.1927934) q[4];
rx(-9*pi/2) q[5];
rxx(pi/2) q[5],q[4];
rx(-pi/2) q[4];
ry(pi/2) q[4];
rxx(pi/2) q[6],q[3];
rz(9.1930459) q[3];
rxx(pi/2) q[6],q[3];
rx(-pi) q[3];
rxx(pi/2) q[5],q[3];
rz(9.1928897) q[3];
rxx(pi/2) q[5],q[3];
rx(-pi) q[3];
rxx(pi/2) q[4],q[3];
rz(9.1925335) q[3];
rx(-7*pi/2) q[4];
rxx(pi/2) q[4],q[3];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rxx(pi/2) q[6],q[2];
rz(9.1929401) q[2];
rxx(pi/2) q[6],q[2];
rx(-pi) q[2];
rxx(pi/2) q[5],q[2];
rz(9.1928854) q[2];
rxx(pi/2) q[5],q[2];
rx(-pi) q[2];
rxx(pi/2) q[4],q[2];
rz(9.1929502) q[2];
rxx(pi/2) q[4],q[2];
rx(-pi) q[2];
rxx(pi/2) q[3],q[2];
rz(9.1929089) q[2];
rx(-5*pi/2) q[3];
rxx(pi/2) q[3],q[2];
rx(-pi/2) q[2];
ry(pi/2) q[2];
rxx(pi/2) q[6],q[1];
rz(9.1947137) q[1];
rxx(pi/2) q[6],q[1];
rx(-pi) q[1];
rxx(pi/2) q[5],q[1];
rz(9.193103) q[1];
rxx(pi/2) q[5],q[1];
rx(-pi) q[1];
rxx(pi/2) q[4],q[1];
rz(9.1944014) q[1];
rxx(pi/2) q[4],q[1];
rx(-pi) q[1];
rxx(pi/2) q[3],q[1];
rz(9.1897608) q[1];
rxx(pi/2) q[3],q[1];
rx(-pi) q[1];
rxx(pi/2) q[2],q[1];
rz(9.1943551) q[1];
rx(-3*pi/2) q[2];
rxx(pi/2) q[2],q[1];
rx(-pi/2) q[1];
ry(pi/2) q[1];
rxx(pi/2) q[6],q[0];
rz(9.1929263) q[0];
rxx(pi/2) q[6],q[0];
rx(-pi) q[0];
rxx(pi/2) q[5],q[0];
rz(9.1928953) q[0];
rxx(pi/2) q[5],q[0];
rx(-pi) q[0];
rxx(pi/2) q[4],q[0];
rz(9.1928782) q[0];
rxx(pi/2) q[4],q[0];
rx(-pi) q[0];
rxx(pi/2) q[3],q[0];
rz(9.1927684) q[0];
rxx(pi/2) q[3],q[0];
rx(-pi) q[0];
rxx(pi/2) q[2],q[0];
rz(9.1929154) q[0];
rxx(pi/2) q[2],q[0];
rx(-pi) q[0];
rxx(pi/2) q[1],q[0];
rz(9.1916959) q[0];
rx(-pi/2) q[1];
rxx(pi/2) q[1],q[0];
ry(2.9133436) q[0];
rx(-2.1077057) q[0];
rz(-2.9674787) q[1];
ry(2.5955203) q[1];
rz(-1.3678028) q[1];
rz(-2.7737986) q[2];
ry(2.561453) q[2];
rz(-1.1390756) q[2];
rz(-2.7981866) q[3];
ry(2.5673511) q[3];
rz(-1.1681775) q[3];
rz(-2.7799233) q[4];
ry(2.562983) q[4];
rz(-1.1463947) q[4];
rz(-2.775413) q[5];
ry(2.5618595) q[5];
rz(-1.1410055) q[5];
rz(-1.033887) q[6];
rxx(pi/2) q[6],q[5];
rx(-pi/2) q[5];
rz(23.724605) q[5];
rx(-17.456567) q[6];
rxx(pi/2) q[6],q[5];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rxx(pi/2) q[6],q[4];
rx(-pi/2) q[4];
rz(23.724505) q[4];
rxx(pi/2) q[6],q[4];
rx(-pi) q[4];
rxx(pi/2) q[5],q[4];
rz(23.72427) q[4];
rx(-9*pi/2) q[5];
rxx(pi/2) q[5],q[4];
rx(-pi/2) q[4];
ry(pi/2) q[4];
rxx(pi/2) q[6],q[3];
rx(-pi/2) q[3];
rz(23.724921) q[3];
rxx(pi/2) q[6],q[3];
rx(-pi) q[3];
rxx(pi/2) q[5],q[3];
rz(23.724518) q[3];
rxx(pi/2) q[5],q[3];
rx(-pi) q[3];
rxx(pi/2) q[4],q[3];
rz(23.723599) q[3];
rx(-7*pi/2) q[4];
rxx(pi/2) q[4],q[3];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rxx(pi/2) q[6],q[2];
rx(-pi/2) q[2];
rz(23.724648) q[2];
rxx(pi/2) q[6],q[2];
rx(-pi) q[2];
rxx(pi/2) q[5],q[2];
rz(23.724507) q[2];
rxx(pi/2) q[5],q[2];
rx(-pi) q[2];
rxx(pi/2) q[4],q[2];
rz(23.724674) q[2];
rxx(pi/2) q[4],q[2];
rx(-pi) q[2];
rxx(pi/2) q[3],q[2];
rz(23.724568) q[2];
rx(-5*pi/2) q[3];
rxx(pi/2) q[3],q[2];
rx(-pi/2) q[2];
ry(pi/2) q[2];
rxx(pi/2) q[6],q[1];
rx(-pi/2) q[1];
rz(23.729226) q[1];
rxx(pi/2) q[6],q[1];
rx(-pi) q[1];
rxx(pi/2) q[5],q[1];
rz(23.725069) q[1];
rxx(pi/2) q[5],q[1];
rx(-pi) q[1];
rxx(pi/2) q[4],q[1];
rz(23.72842) q[1];
rxx(pi/2) q[4],q[1];
rx(-pi) q[1];
rxx(pi/2) q[3],q[1];
rz(23.716443) q[1];
rxx(pi/2) q[3],q[1];
rx(-pi) q[1];
rxx(pi/2) q[2],q[1];
rz(23.7283) q[1];
rx(-3*pi/2) q[2];
rxx(pi/2) q[2],q[1];
rx(-pi/2) q[1];
ry(pi/2) q[1];
rxx(pi/2) q[6],q[0];
rz(23.724613) q[0];
rxx(pi/2) q[6],q[0];
rx(-pi) q[0];
rxx(pi/2) q[5],q[0];
rz(23.724533) q[0];
rxx(pi/2) q[5],q[0];
rx(-pi) q[0];
rxx(pi/2) q[4],q[0];
rz(23.724489) q[0];
rxx(pi/2) q[4],q[0];
rx(-pi) q[0];
rxx(pi/2) q[3],q[0];
rz(23.724205) q[0];
rxx(pi/2) q[3],q[0];
rx(-pi) q[0];
rxx(pi/2) q[2],q[0];
rz(23.724585) q[0];
rxx(pi/2) q[2],q[0];
rx(-pi) q[0];
rxx(pi/2) q[1],q[0];
rz(23.721438) q[0];
rx(-pi/2) q[1];
rxx(pi/2) q[1],q[0];
ry(-1.3988309) q[0];
rx(-3.0090051) q[0];
rz(3.0244745) q[1];
ry(1.5085022) q[1];
rz(2.6549257) q[1];
rz(3.0107591) q[2];
ry(1.5492398) q[2];
rz(2.9792256) q[2];
rz(3.011817) q[3];
ry(1.5435586) q[3];
rz(2.935867) q[3];
rz(3.0110036) q[4];
ry(1.5477969) q[4];
rz(2.9682474) q[4];
rz(3.0108221) q[5];
ry(1.5488584) q[5];
rz(2.976326) q[5];
rz(-0.13258753) q[6];
rxx(pi/2) q[6],q[5];
rx(-pi/2) q[5];
rz(-14.439748) q[5];
rx(-11*pi/2) q[6];
rxx(pi/2) q[6],q[5];
rx(-pi/2) q[5];
ry(pi/2) q[5];
rxx(pi/2) q[6],q[4];
rx(-pi/2) q[4];
rz(-14.439687) q[4];
rxx(pi/2) q[6],q[4];
rx(-pi) q[4];
rxx(pi/2) q[5],q[4];
rz(-14.439544) q[4];
rx(-9*pi/2) q[5];
rxx(pi/2) q[5],q[4];
rx(-pi/2) q[4];
ry(pi/2) q[4];
rxx(pi/2) q[6],q[3];
rx(-pi/2) q[3];
rz(-14.43994) q[3];
rxx(pi/2) q[6],q[3];
rx(-pi) q[3];
rxx(pi/2) q[5],q[3];
rz(-14.439695) q[3];
rxx(pi/2) q[5],q[3];
rx(-pi) q[3];
rxx(pi/2) q[4],q[3];
rz(-14.439136) q[3];
rx(-7*pi/2) q[4];
rxx(pi/2) q[4],q[3];
rx(-pi/2) q[3];
ry(pi/2) q[3];
rxx(pi/2) q[6],q[2];
rx(-pi/2) q[2];
rz(-14.439774) q[2];
rxx(pi/2) q[6],q[2];
rx(-pi) q[2];
rxx(pi/2) q[5],q[2];
rz(-14.439688) q[2];
rxx(pi/2) q[5],q[2];
rx(-pi) q[2];
rxx(pi/2) q[4],q[2];
rz(-14.43979) q[2];
rxx(pi/2) q[4],q[2];
rx(-pi) q[2];
rxx(pi/2) q[3],q[2];
rz(-14.439725) q[2];
rx(-5*pi/2) q[3];
rxx(pi/2) q[3],q[2];
rx(-pi/2) q[2];
ry(pi/2) q[2];
rxx(pi/2) q[6],q[1];
rx(-pi/2) q[1];
rz(-14.44256) q[1];
rxx(pi/2) q[6],q[1];
rx(-pi) q[1];
rxx(pi/2) q[5],q[1];
rz(-14.44003) q[1];
rxx(pi/2) q[5],q[1];
rx(-pi) q[1];
rxx(pi/2) q[4],q[1];
rz(-14.44207) q[1];
rxx(pi/2) q[4],q[1];
rx(-pi) q[1];
rxx(pi/2) q[3],q[1];
rz(-14.434781) q[1];
rxx(pi/2) q[3],q[1];
rx(-pi) q[1];
rxx(pi/2) q[2],q[1];
rz(-14.441997) q[1];
rx(-3*pi/2) q[2];
rxx(pi/2) q[2],q[1];
rx(-pi/2) q[1];
ry(pi/2) q[1];
rxx(pi/2) q[6],q[0];
rz(-14.439753) q[0];
rxx(pi/2) q[6],q[0];
rx(-pi) q[0];
rxx(pi/2) q[5],q[0];
rz(-14.439704) q[0];
rxx(pi/2) q[5],q[0];
rx(-pi) q[0];
rxx(pi/2) q[4],q[0];
rz(-14.439677) q[0];
rxx(pi/2) q[4],q[0];
rx(-pi) q[0];
rxx(pi/2) q[3],q[0];
rz(-14.439505) q[0];
rxx(pi/2) q[3],q[0];
rx(-pi) q[0];
rxx(pi/2) q[2],q[0];
rz(-14.439736) q[0];
rxx(pi/2) q[2],q[0];
rx(-pi) q[0];
rxx(pi/2) q[1],q[0];
rz(-14.43782) q[0];
rx(-pi/2) q[1];
rxx(pi/2) q[1],q[0];
ry(-1.8790409) q[0];
rx(-0.38624683) q[0];
rz(2.0051597) q[1];
ry(2.0327943) q[1];
rz(-2.9376833) q[1];
rz(1.9736146) q[2];
ry(1.8510865) q[2];
rz(-3.0242549) q[2];
rz(1.9767677) q[3];
ry(1.8755482) q[3];
rz(-3.0133216) q[3];
rz(1.9743845) q[4];
ry(1.8572838) q[4];
rz(-3.0215014) q[4];
rz(1.9738161) q[5];
ry(1.8527236) q[5];
rz(-3.0235286) q[5];
rz(1.9745935) q[6];
ry(1.8589394) q[6];
rz(-3.020764) q[6];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
