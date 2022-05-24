// Benchmark was created by MQT Bench on 2022-04-07
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 0.1.0
// Qiskit version: {'qiskit-terra': '0.19.2', 'qiskit-aer': '0.10.3', 'qiskit-ignis': '0.7.0', 'qiskit-ibmq-provider': '0.18.3', 'qiskit-aqua': '0.9.5', 'qiskit': '0.34.2', 'qiskit-nature': '0.3.1', 'qiskit-finance': '0.3.1', 'qiskit-optimization': '0.3.1', 'qiskit-machine-learning': '0.3.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg meas[5];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
cx q[4],q[3];
rz(-20.06105) q[3];
cx q[4],q[3];
cx q[4],q[2];
rz(-20.061047) q[2];
cx q[4],q[2];
cx q[3],q[2];
rz(-20.060795) q[2];
cx q[3],q[2];
cx q[4],q[1];
rz(-20.060977) q[1];
cx q[4],q[1];
cx q[3],q[1];
rz(-20.060922) q[1];
cx q[3],q[1];
cx q[2],q[1];
rz(-20.060949) q[1];
cx q[2],q[1];
cx q[4],q[0];
rz(-20.060977) q[0];
cx q[4],q[0];
cx q[3],q[0];
rz(-20.060356) q[0];
cx q[3],q[0];
cx q[2],q[0];
rz(-20.061069) q[0];
cx q[2],q[0];
cx q[1],q[0];
rz(-20.061192) q[0];
cx q[1],q[0];
u3(1.1664938,-pi/2,2.751258) q[0];
u3(1.1664938,-pi/2,2.7721269) q[1];
u3(1.1664938,-pi/2,2.7931064) q[2];
u3(1.1664938,-pi/2,2.7662891) q[3];
u3(1.1664938,-pi/2,2.7759201) q[4];
cx q[4],q[3];
rz(-11.55624) q[3];
cx q[4],q[3];
cx q[4],q[2];
rz(-11.556238) q[2];
cx q[4],q[2];
cx q[3],q[2];
rz(-11.556094) q[2];
cx q[3],q[2];
cx q[4],q[1];
rz(-11.556198) q[1];
cx q[4],q[1];
cx q[3],q[1];
rz(-11.556166) q[1];
cx q[3],q[1];
cx q[2],q[1];
rz(-11.556182) q[1];
cx q[2],q[1];
cx q[4],q[0];
rz(-11.556198) q[0];
cx q[4],q[0];
cx q[3],q[0];
rz(-11.555841) q[0];
cx q[3],q[0];
cx q[2],q[0];
rz(-11.556251) q[0];
cx q[2],q[0];
cx q[1],q[0];
rz(-11.556322) q[0];
cx q[1],q[0];
u3(1.5827957,-pi/2,0.5427897) q[0];
u3(1.5827957,-pi/2,0.55481134) q[1];
u3(1.5827957,-pi/2,0.56689664) q[2];
u3(1.5827957,-pi/2,0.55144846) q[3];
u3(1.5827957,-pi/2,0.55699642) q[4];
cx q[4],q[3];
rz(25.798471) q[3];
cx q[4],q[3];
cx q[4],q[2];
rz(25.798466) q[2];
cx q[4],q[2];
cx q[3],q[2];
rz(25.798143) q[2];
cx q[3],q[2];
cx q[4],q[1];
rz(25.798377) q[1];
cx q[4],q[1];
cx q[3],q[1];
rz(25.798305) q[1];
cx q[3],q[1];
cx q[2],q[1];
rz(25.79834) q[1];
cx q[2],q[1];
cx q[4],q[0];
rz(25.798376) q[0];
cx q[4],q[0];
cx q[3],q[0];
rz(25.797578) q[0];
cx q[3],q[0];
cx q[2],q[0];
rz(25.798494) q[0];
cx q[2],q[0];
cx q[1],q[0];
rz(25.798652) q[0];
cx q[1],q[0];
u3(0.9187328,-pi/2,0.94497494) q[0];
u3(0.9187328,-pi/2,0.91813751) q[1];
u3(0.9187328,-pi/2,0.89115796) q[2];
u3(0.9187328,-pi/2,0.9256449) q[3];
u3(0.9187328,-pi/2,0.91325947) q[4];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
