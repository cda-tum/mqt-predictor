// Benchmark was created by MQT Bench on 2022-04-07
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 0.1.0
// Qiskit version: {'qiskit-terra': '0.19.2', 'qiskit-aer': '0.10.3', 'qiskit-ignis': '0.7.0', 'qiskit-ibmq-provider': '0.18.3', 'qiskit-aqua': '0.9.5', 'qiskit': '0.34.2', 'qiskit-nature': '0.3.1', 'qiskit-finance': '0.3.1', 'qiskit-optimization': '0.3.1', 'qiskit-machine-learning': '0.3.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg meas[12];
ry(-1.96866746224441) q[0];
ry(4.65115628738698) q[1];
cz q[0],q[1];
ry(3.28577993421041) q[2];
cz q[0],q[2];
cz q[1],q[2];
ry(-3.31639896068433) q[3];
cz q[0],q[3];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.93050265470951) q[4];
cz q[0],q[4];
cz q[1],q[4];
cz q[2],q[4];
cz q[3],q[4];
ry(-2.49848891263555) q[5];
cz q[0],q[5];
cz q[1],q[5];
cz q[2],q[5];
cz q[3],q[5];
cz q[4],q[5];
ry(-4.52023662319203) q[6];
cz q[0],q[6];
cz q[1],q[6];
cz q[2],q[6];
cz q[3],q[6];
cz q[4],q[6];
cz q[5],q[6];
ry(5.15637121357692) q[7];
cz q[0],q[7];
cz q[1],q[7];
cz q[2],q[7];
cz q[3],q[7];
cz q[4],q[7];
cz q[5],q[7];
cz q[6],q[7];
ry(1.26282798573359) q[8];
cz q[0],q[8];
cz q[1],q[8];
cz q[2],q[8];
cz q[3],q[8];
cz q[4],q[8];
cz q[5],q[8];
cz q[6],q[8];
cz q[7],q[8];
ry(-1.84009456962673) q[9];
cz q[0],q[9];
cz q[1],q[9];
cz q[2],q[9];
cz q[3],q[9];
cz q[4],q[9];
cz q[5],q[9];
cz q[6],q[9];
cz q[7],q[9];
cz q[8],q[9];
ry(2.91317337828844) q[10];
cz q[0],q[10];
cz q[1],q[10];
cz q[2],q[10];
cz q[3],q[10];
cz q[4],q[10];
cz q[5],q[10];
cz q[6],q[10];
cz q[7],q[10];
cz q[8],q[10];
cz q[9],q[10];
ry(0.034747637018091) q[11];
cz q[0],q[11];
ry(-6.17686511187443) q[0];
cz q[1],q[11];
ry(2.15330407240643) q[1];
cz q[0],q[1];
cz q[2],q[11];
ry(3.84903893235342) q[2];
cz q[0],q[2];
cz q[1],q[2];
cz q[3],q[11];
ry(5.57243197408426) q[3];
cz q[0],q[3];
cz q[1],q[3];
cz q[2],q[3];
cz q[4],q[11];
ry(-5.85434021406228) q[4];
cz q[0],q[4];
cz q[1],q[4];
cz q[2],q[4];
cz q[3],q[4];
cz q[5],q[11];
ry(-3.03983146450222) q[5];
cz q[0],q[5];
cz q[1],q[5];
cz q[2],q[5];
cz q[3],q[5];
cz q[4],q[5];
cz q[6],q[11];
ry(5.39130395940649) q[6];
cz q[0],q[6];
cz q[1],q[6];
cz q[2],q[6];
cz q[3],q[6];
cz q[4],q[6];
cz q[5],q[6];
cz q[7],q[11];
ry(-4.52501928058865) q[7];
cz q[0],q[7];
cz q[1],q[7];
cz q[2],q[7];
cz q[3],q[7];
cz q[4],q[7];
cz q[5],q[7];
cz q[6],q[7];
cz q[8],q[11];
ry(-3.88410455650076) q[8];
cz q[0],q[8];
cz q[1],q[8];
cz q[2],q[8];
cz q[3],q[8];
cz q[4],q[8];
cz q[5],q[8];
cz q[6],q[8];
cz q[7],q[8];
cz q[9],q[11];
cz q[10],q[11];
ry(2.56810577945972) q[10];
ry(-2.61570424124593) q[11];
ry(-3.80669991340623) q[9];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
ry(-3.66146472899752) q[0];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
ry(5.17624341724964) q[1];
cz q[0],q[1];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
ry(-5.59585145763147) q[2];
cz q[0],q[2];
cz q[1],q[2];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
ry(-3.87639894909308) q[3];
cz q[0],q[3];
cz q[1],q[3];
cz q[2],q[3];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
ry(-4.18615688006732) q[4];
cz q[0],q[4];
cz q[1],q[4];
cz q[2],q[4];
cz q[3],q[4];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
ry(-1.17828025722629) q[5];
cz q[0],q[5];
cz q[1],q[5];
cz q[2],q[5];
cz q[3],q[5];
cz q[4],q[5];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
ry(0.525506691859232) q[6];
cz q[0],q[6];
cz q[1],q[6];
cz q[2],q[6];
cz q[3],q[6];
cz q[4],q[6];
cz q[5],q[6];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
ry(4.39097725037306) q[7];
cz q[0],q[7];
cz q[1],q[7];
cz q[2],q[7];
cz q[3],q[7];
cz q[4],q[7];
cz q[5],q[7];
cz q[6],q[7];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
ry(5.90229988356867) q[8];
cz q[0],q[8];
cz q[1],q[8];
cz q[2],q[8];
cz q[3],q[8];
cz q[4],q[8];
cz q[5],q[8];
cz q[6],q[8];
cz q[7],q[8];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-2.69052852737617) q[10];
ry(5.59420801483987) q[11];
ry(2.59013793201032) q[9];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
ry(2.98566026407053) q[0];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
ry(4.21410289684894) q[1];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
ry(-1.43659416549886) q[2];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
ry(1.51342195798125) q[3];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
ry(-3.07040397660211) q[4];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
ry(3.46107336204716) q[5];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
ry(1.82936933498298) q[6];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
ry(0.225910635166347) q[7];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
ry(6.15168407713806) q[8];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.00485886677007) q[10];
ry(2.88322714925328) q[11];
ry(-2.5773661368614) q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11];
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
measure q[11] -> meas[11];
