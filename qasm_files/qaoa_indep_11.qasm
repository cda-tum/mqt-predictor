// Benchmark was created by MQT Bench on 2022-04-07
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 0.1.0
// Qiskit version: {'qiskit-terra': '0.19.2', 'qiskit-aer': '0.10.3', 'qiskit-ignis': '0.7.0', 'qiskit-ibmq-provider': '0.18.3', 'qiskit-aqua': '0.9.5', 'qiskit': '0.34.2', 'qiskit-nature': '0.3.1', 'qiskit-finance': '0.3.1', 'qiskit-optimization': '0.3.1', 'qiskit-machine-learning': '0.3.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
creg meas[11];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
rzz(-3.3953569143505) q[7],q[9];
rzz(-3.3953569143505) q[5],q[9];
rzz(-3.3953569143505) q[2],q[5];
rx(8.35680073596923) q[5];
rzz(-3.3953569143505) q[7],q[8];
rzz(-3.3953569143505) q[0],q[8];
rx(8.35680073596923) q[7];
rx(8.35680073596923) q[8];
rx(8.35680073596923) q[9];
rzz(5.99188362813693) q[7],q[9];
rzz(5.99188362813693) q[5],q[9];
rzz(5.99188362813693) q[7],q[8];
rx(1.97047238016489) q[7];
rx(1.97047238016489) q[9];
h q[10];
rzz(-3.3953569143505) q[6],q[10];
rzz(-3.3953569143505) q[3],q[6];
rzz(-3.3953569143505) q[0],q[3];
rx(8.35680073596923) q[0];
rzz(5.99188362813693) q[0],q[8];
rx(8.35680073596923) q[3];
rzz(-3.3953569143505) q[4],q[10];
rzz(-3.3953569143505) q[1],q[4];
rzz(-3.3953569143505) q[1],q[2];
rx(8.35680073596923) q[1];
rx(8.35680073596923) q[10];
rx(8.35680073596923) q[2];
rzz(5.99188362813693) q[2],q[5];
rx(8.35680073596923) q[4];
rx(1.97047238016489) q[5];
rx(8.35680073596923) q[6];
rzz(5.99188362813693) q[6],q[10];
rzz(5.99188362813693) q[3],q[6];
rzz(5.99188362813693) q[0],q[3];
rx(1.97047238016489) q[0];
rx(1.97047238016489) q[3];
rzz(5.99188362813693) q[4],q[10];
rzz(5.99188362813693) q[1],q[4];
rzz(5.99188362813693) q[1],q[2];
rx(1.97047238016489) q[1];
rx(1.97047238016489) q[10];
rx(1.97047238016489) q[2];
rx(1.97047238016489) q[4];
rx(1.97047238016489) q[6];
rx(1.97047238016489) q[8];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q[10] -> meas[10];
