// Benchmark was created by MQT Bench on 2022-04-07
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 0.1.0
// Qiskit version: {'qiskit-terra': '0.19.2', 'qiskit-aer': '0.10.3', 'qiskit-ignis': '0.7.0', 'qiskit-ibmq-provider': '0.18.3', 'qiskit-aqua': '0.9.5', 'qiskit': '0.34.2', 'qiskit-nature': '0.3.1', 'qiskit-finance': '0.3.1', 'qiskit-optimization': '0.3.1', 'qiskit-machine-learning': '0.3.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg meas[7];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
cx q[6],q[5];
rz(9.1929231) q[5];
cx q[6],q[5];
cx q[6],q[4];
rz(9.1928846) q[4];
cx q[6],q[4];
cx q[5],q[4];
rz(9.1927934) q[4];
cx q[5],q[4];
cx q[6],q[3];
rz(9.1930459) q[3];
cx q[6],q[3];
cx q[5],q[3];
rz(9.1928897) q[3];
cx q[5],q[3];
cx q[4],q[3];
rz(9.1925335) q[3];
cx q[4],q[3];
cx q[6],q[2];
rz(9.1929401) q[2];
cx q[6],q[2];
cx q[5],q[2];
rz(9.1928854) q[2];
cx q[5],q[2];
cx q[4],q[2];
rz(9.1929502) q[2];
cx q[4],q[2];
cx q[3],q[2];
rz(9.1929089) q[2];
cx q[3],q[2];
cx q[6],q[1];
rz(9.1947137) q[1];
cx q[6],q[1];
cx q[5],q[1];
rz(9.193103) q[1];
cx q[5],q[1];
cx q[4],q[1];
rz(9.1944014) q[1];
cx q[4],q[1];
cx q[3],q[1];
rz(9.1897608) q[1];
cx q[3],q[1];
cx q[2],q[1];
rz(9.1943551) q[1];
cx q[2],q[1];
cx q[6],q[0];
rz(9.1929263) q[0];
cx q[6],q[0];
cx q[5],q[0];
rz(9.1928953) q[0];
cx q[5],q[0];
cx q[4],q[0];
rz(9.1928782) q[0];
cx q[4],q[0];
cx q[3],q[0];
rz(9.1927684) q[0];
cx q[3],q[0];
cx q[2],q[0];
rz(9.1929154) q[0];
cx q[2],q[0];
cx q[1],q[0];
rz(9.1916959) q[0];
cx q[1],q[0];
u3(1.033887,-pi/2,-1.3425473) q[0];
u3(1.033887,-pi/2,-1.465905) q[1];
u3(1.033887,-pi/2,-1.3393753) q[2];
u3(1.033887,-pi/2,-1.3563148) q[3];
u3(1.033887,-pi/2,-1.3436647) q[4];
u3(1.033887,-pi/2,-1.3405083) q[5];
u3(1.033887,-pi/2,-1.3448109) q[6];
cx q[6],q[5];
rz(23.724605) q[5];
cx q[6],q[5];
cx q[6],q[4];
rz(23.724505) q[4];
cx q[6],q[4];
cx q[5],q[4];
rz(23.72427) q[4];
cx q[5],q[4];
cx q[6],q[3];
rz(23.724921) q[3];
cx q[6],q[3];
cx q[5],q[3];
rz(23.724518) q[3];
cx q[5],q[3];
cx q[4],q[3];
rz(23.723599) q[3];
cx q[4],q[3];
cx q[6],q[2];
rz(23.724648) q[2];
cx q[6],q[2];
cx q[5],q[2];
rz(23.724507) q[2];
cx q[5],q[2];
cx q[4],q[2];
rz(23.724674) q[2];
cx q[4],q[2];
cx q[3],q[2];
rz(23.724568) q[2];
cx q[3],q[2];
cx q[6],q[1];
rz(23.729226) q[1];
cx q[6],q[1];
cx q[5],q[1];
rz(23.725069) q[1];
cx q[5],q[1];
cx q[4],q[1];
rz(23.72842) q[1];
cx q[4],q[1];
cx q[3],q[1];
rz(23.716443) q[1];
cx q[3],q[1];
cx q[2],q[1];
rz(23.7283) q[1];
cx q[2],q[1];
cx q[6],q[0];
rz(23.724613) q[0];
cx q[6],q[0];
cx q[5],q[0];
rz(23.724533) q[0];
cx q[5],q[0];
cx q[4],q[0];
rz(23.724489) q[0];
cx q[4],q[0];
cx q[3],q[0];
rz(23.724205) q[0];
cx q[3],q[0];
cx q[2],q[0];
rz(23.724585) q[0];
cx q[2],q[0];
cx q[1],q[0];
rz(23.721438) q[0];
cx q[1],q[0];
u3(0.13258753,-pi/2,2.9696272) q[0];
u3(0.13258753,-pi/2,2.6512725) q[1];
u3(0.13258753,-pi/2,2.9778134) q[2];
u3(0.13258753,-pi/2,2.934097) q[3];
u3(0.13258753,-pi/2,2.9667435) q[4];
u3(0.13258753,-pi/2,2.9748895) q[5];
u3(0.13258753,-pi/2,2.9637856) q[6];
cx q[6],q[5];
rz(-14.439748) q[5];
cx q[6],q[5];
cx q[6],q[4];
rz(-14.439687) q[4];
cx q[6],q[4];
cx q[5],q[4];
rz(-14.439544) q[4];
cx q[5],q[4];
cx q[6],q[3];
rz(-14.43994) q[3];
cx q[6],q[3];
cx q[5],q[3];
rz(-14.439695) q[3];
cx q[5],q[3];
cx q[4],q[3];
rz(-14.439136) q[3];
cx q[4],q[3];
cx q[6],q[2];
rz(-14.439774) q[2];
cx q[6],q[2];
cx q[5],q[2];
rz(-14.439688) q[2];
cx q[5],q[2];
cx q[4],q[2];
rz(-14.43979) q[2];
cx q[4],q[2];
cx q[3],q[2];
rz(-14.439725) q[2];
cx q[3],q[2];
cx q[6],q[1];
rz(-14.44256) q[1];
cx q[6],q[1];
cx q[5],q[1];
rz(-14.44003) q[1];
cx q[5],q[1];
cx q[4],q[1];
rz(-14.44207) q[1];
cx q[4],q[1];
cx q[3],q[1];
rz(-14.434781) q[1];
cx q[3],q[1];
cx q[2],q[1];
rz(-14.441997) q[1];
cx q[2],q[1];
cx q[6],q[0];
rz(-14.439753) q[0];
cx q[6],q[0];
cx q[5],q[0];
rz(-14.439704) q[0];
cx q[5],q[0];
cx q[4],q[0];
rz(-14.439677) q[0];
cx q[4],q[0];
cx q[3],q[0];
rz(-14.439505) q[0];
cx q[3],q[0];
cx q[2],q[0];
rz(-14.439736) q[0];
cx q[2],q[0];
cx q[1],q[0];
rz(-14.43782) q[0];
cx q[1],q[0];
u3(1.1845495,-pi/2,-2.8333481) q[0];
u3(1.1845495,-pi/2,-2.6395846) q[1];
u3(1.1845495,-pi/2,-2.8383305) q[2];
u3(1.1845495,-pi/2,-2.8117229) q[3];
u3(1.1845495,-pi/2,-2.8315929) q[4];
u3(1.1845495,-pi/2,-2.8365509) q[5];
u3(1.1845495,-pi/2,-2.8297926) q[6];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
