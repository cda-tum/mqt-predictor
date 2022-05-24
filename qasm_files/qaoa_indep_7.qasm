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
rzz(-5.18559484832958) q[2],q[4];
rzz(-5.18559484832958) q[1],q[4];
rzz(-5.18559484832958) q[1],q[2];
rx(-8.2731194594043) q[1];
rx(-8.2731194594043) q[2];
rx(-8.2731194594043) q[4];
rzz(-0.992428798165926) q[2],q[4];
rzz(-0.992428798165926) q[1],q[4];
rzz(-0.992428798165926) q[1],q[2];
rx(3.3299997553086) q[1];
rx(3.3299997553086) q[2];
rx(3.3299997553086) q[4];
h q[5];
h q[6];
rzz(-5.18559484832958) q[5],q[6];
rzz(-5.18559484832958) q[0],q[5];
rzz(-5.18559484832958) q[3],q[6];
rzz(-5.18559484832958) q[0],q[3];
rx(-8.2731194594043) q[0];
rx(-8.2731194594043) q[3];
rx(-8.2731194594043) q[5];
rx(-8.2731194594043) q[6];
rzz(-0.992428798165926) q[5],q[6];
rzz(-0.992428798165926) q[0],q[5];
rzz(-0.992428798165926) q[3],q[6];
rzz(-0.992428798165926) q[0],q[3];
rx(3.3299997553086) q[0];
rx(3.3299997553086) q[3];
rx(3.3299997553086) q[5];
rx(3.3299997553086) q[6];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
