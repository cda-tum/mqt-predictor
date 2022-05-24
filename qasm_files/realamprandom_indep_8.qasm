// Benchmark was created by MQT Bench on 2022-04-08
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 0.1.0
// Qiskit version: {'qiskit-terra': '0.19.2', 'qiskit-aer': '0.10.3', 'qiskit-ignis': '0.7.0', 'qiskit-ibmq-provider': '0.18.3', 'qiskit-aqua': '0.9.5', 'qiskit': '0.34.2', 'qiskit-nature': '0.3.1', 'qiskit-finance': '0.3.1', 'qiskit-optimization': '0.3.1', 'qiskit-machine-learning': '0.3.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg meas[8];
ry(0.80625732807157) q[0];
ry(0.669459318727495) q[1];
cx q[0],q[1];
ry(0.188487561185058) q[2];
cx q[0],q[2];
cx q[1],q[2];
ry(0.459038472967673) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
ry(0.22776733302296) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
ry(0.92133541418249) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
ry(0.227418742934309) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
ry(0.797260045476456) q[7];
cx q[0],q[7];
ry(0.99484154186132) q[0];
cx q[1],q[7];
ry(0.745084625383115) q[1];
cx q[0],q[1];
cx q[2],q[7];
ry(0.450906537463562) q[2];
cx q[0],q[2];
cx q[1],q[2];
cx q[3],q[7];
ry(0.0656909939371442) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
cx q[4],q[7];
ry(0.593415262068914) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
cx q[5],q[7];
ry(0.835776584340396) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
cx q[6],q[7];
ry(0.38936141267058) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
ry(0.109791693811586) q[7];
cx q[0],q[7];
ry(0.768351613970013) q[0];
cx q[1],q[7];
ry(0.160316610623928) q[1];
cx q[0],q[1];
cx q[2],q[7];
ry(0.695385002807251) q[2];
cx q[0],q[2];
cx q[1],q[2];
cx q[3],q[7];
ry(0.123126841306782) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
cx q[4],q[7];
ry(0.199336753693642) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
cx q[5],q[7];
ry(0.668711065715869) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
cx q[6],q[7];
ry(0.76146079653793) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
ry(0.859548374325492) q[7];
cx q[0],q[7];
ry(0.56091471918761) q[0];
cx q[1],q[7];
ry(0.581903804255912) q[1];
cx q[2],q[7];
ry(0.665524088514027) q[2];
cx q[3],q[7];
ry(0.896561129581808) q[3];
cx q[4],q[7];
ry(0.114409643883595) q[4];
cx q[5],q[7];
ry(0.698189614386364) q[5];
cx q[6],q[7];
ry(0.0736934630067582) q[6];
ry(0.487245967246332) q[7];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
