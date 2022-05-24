// Benchmark was created by MQT Bench on 2022-04-10
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 0.1.0
// Qiskit version: {'qiskit-terra': '0.19.2', 'qiskit-aer': '0.10.3', 'qiskit-ignis': '0.7.0', 'qiskit-ibmq-provider': '0.18.3', 'qiskit-aqua': '0.9.5', 'qiskit': '0.34.2', 'qiskit-nature': '0.3.1', 'qiskit-finance': '0.3.1', 'qiskit-optimization': '0.3.1', 'qiskit-machine-learning': '0.3.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg meas[7];
u3(0.04777873,0.46946675,0) q[0];
u3(0.84819056,0.34032923,0) q[1];
cx q[0],q[1];
u3(0.75014289,0.29075675,0) q[2];
cx q[0],q[2];
cx q[1],q[2];
u3(0.47751604,0.31931611,0) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
u3(0.94652167,0.0427437,0) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
u3(0.96533641,0.9441545,0) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
u3(0.86889803,0.56890983,0) q[6];
cx q[0],q[6];
u3(0.22728201,0.33923113,0) q[0];
cx q[1],q[6];
u3(0.015285765,0.52982234,0) q[1];
cx q[0],q[1];
cx q[2],q[6];
u3(0.90003849,0.45317549,0) q[2];
cx q[0],q[2];
cx q[1],q[2];
cx q[3],q[6];
u3(0.98240737,0.39402417,0) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
cx q[4],q[6];
u3(0.82023293,0.56563488,0) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
cx q[5],q[6];
u3(0.093003109,0.70021759,0) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
u3(0.23079715,0.31822379,0) q[6];
cx q[0],q[6];
u3(0.70376497,0.22282437,0) q[0];
cx q[1],q[6];
u3(0.9639276,0.72870686,0) q[1];
cx q[0],q[1];
cx q[2],q[6];
u3(0.69070628,0.6565358,0) q[2];
cx q[0],q[2];
cx q[1],q[2];
cx q[3],q[6];
u3(0.28589245,0.48197048,0) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
cx q[4],q[6];
u3(0.39298041,0.88802145,0) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
cx q[5],q[6];
u3(0.43175804,0.88113025,0) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
u3(0.28616723,0.44871615,0) q[6];
cx q[0],q[6];
u3(0.23354101,0.79198687,0) q[0];
cx q[1],q[6];
u3(0.79112699,0.57809723,0) q[1];
cx q[2],q[6];
u3(0.22252874,0.31986564,0) q[2];
cx q[3],q[6];
u3(0.8096544,0.39386374,0) q[3];
cx q[4],q[6];
u3(0.31493475,0.23774313,0) q[4];
cx q[5],q[6];
u3(0.0077343394,0.23435356,0) q[5];
u3(0.24819946,0.57726647,0) q[6];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
