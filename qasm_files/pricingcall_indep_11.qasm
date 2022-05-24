// Benchmark was created by MQT Bench on 2022-04-07
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 0.1.0
// Qiskit version: {'qiskit-terra': '0.19.2', 'qiskit-aer': '0.10.3', 'qiskit-ignis': '0.7.0', 'qiskit-ibmq-provider': '0.18.3', 'qiskit-aqua': '0.9.5', 'qiskit': '0.34.2', 'qiskit-nature': '0.3.1', 'qiskit-finance': '0.3.1', 'qiskit-optimization': '0.3.1', 'qiskit-machine-learning': '0.3.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
creg meas[11];
ry(1.6198683) q[0];
ry(1.6522303) q[1];
ry(1.668654) q[2];
ry(1.6513284) q[3];
ry(1.5137916) q[4];
cx q[4],q[3];
ry(0.8865441) q[3];
cx q[4],q[3];
cx q[3],q[2];
ry(0.27136875) q[2];
cx q[4],q[2];
ry(0.12296747) q[2];
cx q[3],q[2];
ry(0.55611414) q[2];
cx q[4],q[2];
cx q[2],q[1];
ry(0.085977103) q[1];
cx q[3],q[1];
ry(0.021285711) q[1];
cx q[2],q[1];
ry(0.17170629) q[1];
cx q[4],q[1];
ry(0.090209643) q[1];
cx q[2],q[1];
ry(0.0092159449) q[1];
cx q[3],q[1];
ry(0.044235404) q[1];
cx q[2],q[1];
ry(0.32183425) q[1];
cx q[4],q[1];
cx q[1],q[0];
ry(0.024426126) q[0];
cx q[2],q[0];
ry(0.0038978569) q[0];
cx q[1],q[0];
ry(0.048554281) q[0];
cx q[3],q[0];
ry(0.015244745) q[0];
cx q[1],q[0];
ry(0.0012686783) q[0];
cx q[2],q[0];
ry(0.0076534546) q[0];
cx q[1],q[0];
ry(0.094541741) q[0];
cx q[4],q[0];
ry(0.05275865) q[0];
cx q[1],q[0];
ry(0.0046560511) q[0];
cx q[2],q[0];
ry(0.00078021278) q[0];
cx q[1],q[0];
ry(0.0092972183) q[0];
cx q[3],q[0];
ry(0.027179311) q[0];
cx q[1],q[0];
ry(0.0023524332) q[0];
cx q[2],q[0];
ry(0.013674426) q[0];
cx q[1],q[0];
ry(0.17090792) q[0];
cx q[4],q[0];
ry(3*pi/8) q[5];
cry(0) q[0],q[5];
cry(0) q[1],q[5];
x q[1];
cry(0) q[2],q[5];
cry(0) q[3],q[5];
cry(0) q[4],q[5];
x q[4];
x q[6];
x q[7];
x q[8];
ccx q[1],q[7],q[8];
ccx q[2],q[8],q[9];
ccx q[3],q[9],q[10];
x q[10];
ccx q[4],q[10],q[6];
x q[10];
ccx q[3],q[9],q[10];
ccx q[2],q[8],q[9];
x q[4];
cx q[6],q[5];
u(0.29425236,0,0) q[5];
cx q[6],q[5];
u3(0.29425236,-pi,-pi) q[5];
cx q[6],q[5];
u(-0.011079862,0,0) q[5];
cx q[6],q[5];
u(0.011079862,0,0) q[5];
ccx q[6],q[0],q[5];
cx q[6],q[5];
u(0.011079862,0,0) q[5];
cx q[6],q[5];
u(-0.011079862,0,0) q[5];
ccx q[6],q[0],q[5];
cx q[6],q[5];
u(-0.022159724,0,0) q[5];
cx q[6],q[5];
u(0.022159724,0,0) q[5];
x q[8];
ccx q[1],q[7],q[8];
x q[1];
ccx q[6],q[1],q[5];
cx q[6],q[5];
u(0.022159724,0,0) q[5];
cx q[6],q[5];
u(-0.022159724,0,0) q[5];
ccx q[6],q[1],q[5];
x q[1];
ccx q[1],q[7],q[8];
cx q[6],q[5];
u(-0.044319448,0,0) q[5];
cx q[6],q[5];
u(0.044319448,0,0) q[5];
ccx q[6],q[2],q[5];
cx q[6],q[5];
u(0.044319448,0,0) q[5];
cx q[6],q[5];
u(-0.044319448,0,0) q[5];
ccx q[6],q[2],q[5];
cx q[6],q[5];
u(-0.088638896,0,0) q[5];
cx q[6],q[5];
u(0.088638896,0,0) q[5];
ccx q[6],q[3],q[5];
cx q[6],q[5];
u(0.088638896,0,0) q[5];
cx q[6],q[5];
u(-0.088638896,0,0) q[5];
ccx q[6],q[3],q[5];
cx q[6],q[5];
u(-0.17727779,0,0) q[5];
cx q[6],q[5];
u(0.17727779,0,0) q[5];
ccx q[6],q[4],q[5];
cx q[6],q[5];
u(0.17727779,0,0) q[5];
cx q[6],q[5];
u(-0.17727779,0,0) q[5];
ccx q[6],q[4],q[5];
x q[4];
u1(-pi) q[5];
x q[8];
ccx q[2],q[8],q[9];
ccx q[3],q[9],q[10];
x q[10];
ccx q[4],q[10],q[6];
x q[10];
ccx q[3],q[9],q[10];
ccx q[2],q[8],q[9];
ccx q[1],q[7],q[8];
ccx q[1],q[7],q[8];
ccx q[2],q[8],q[9];
ccx q[3],q[9],q[10];
x q[10];
ccx q[4],q[10],q[6];
x q[10];
ccx q[3],q[9],q[10];
ccx q[2],q[8],q[9];
x q[4];
cx q[6],q[5];
u(0.17727779,0,0) q[5];
cx q[6],q[5];
u(-0.17727779,0,0) q[5];
ccx q[6],q[4],q[5];
cx q[6],q[5];
u(-0.17727779,0,0) q[5];
cx q[6],q[5];
u(0.17727779,0,0) q[5];
ccx q[6],q[4],q[5];
x q[4];
cx q[6],q[5];
u(0.088638896,0,0) q[5];
cx q[6],q[5];
u(-0.088638896,0,0) q[5];
ccx q[6],q[3],q[5];
cx q[6],q[5];
u(-0.088638896,0,0) q[5];
cx q[6],q[5];
u(0.088638896,0,0) q[5];
ccx q[6],q[3],q[5];
cx q[6],q[5];
u(0.044319448,0,0) q[5];
cx q[6],q[5];
u(-0.044319448,0,0) q[5];
ccx q[6],q[2],q[5];
cx q[6],q[5];
u(-0.044319448,0,0) q[5];
cx q[6],q[5];
u(0.044319448,0,0) q[5];
ccx q[6],q[2],q[5];
cx q[6],q[5];
u(0.022159724,0,0) q[5];
cx q[6],q[5];
u(-0.022159724,0,0) q[5];
x q[8];
ccx q[1],q[7],q[8];
x q[1];
ccx q[6],q[1],q[5];
cx q[6],q[5];
u(-0.022159724,0,0) q[5];
cx q[6],q[5];
u(0.022159724,0,0) q[5];
ccx q[6],q[1],q[5];
x q[1];
ccx q[1],q[7],q[8];
cx q[6],q[5];
u(0.011079862,0,0) q[5];
cx q[6],q[5];
u(-0.011079862,0,0) q[5];
ccx q[6],q[0],q[5];
cx q[6],q[5];
u(-0.011079862,0,0) q[5];
cx q[6],q[5];
u(0.011079862,0,0) q[5];
ccx q[6],q[0],q[5];
cx q[6],q[5];
u(-0.29425236,0,0) q[5];
cx q[6],q[5];
u(0.29425236,0,0) q[5];
x q[8];
ccx q[2],q[8],q[9];
ccx q[3],q[9],q[10];
x q[10];
ccx q[4],q[10],q[6];
x q[10];
ccx q[3],q[9],q[10];
ccx q[2],q[8],q[9];
ccx q[1],q[7],q[8];
x q[1];
x q[4];
cry(0) q[4],q[5];
cry(0) q[3],q[5];
cry(0) q[2],q[5];
cry(0) q[1],q[5];
cry(0) q[0],q[5];
cx q[4],q[0];
ry(-0.17090792) q[0];
cx q[1],q[0];
ry(-0.013674426) q[0];
cx q[2],q[0];
ry(-0.0023524332) q[0];
cx q[1],q[0];
ry(-0.027179311) q[0];
cx q[3],q[0];
ry(-0.0092972183) q[0];
cx q[1],q[0];
ry(-0.00078021278) q[0];
cx q[2],q[0];
ry(-0.0046560511) q[0];
cx q[1],q[0];
ry(-0.05275865) q[0];
cx q[4],q[0];
ry(-0.094541741) q[0];
cx q[1],q[0];
ry(-0.0076534546) q[0];
cx q[2],q[0];
ry(-0.0012686783) q[0];
cx q[1],q[0];
ry(-0.015244745) q[0];
cx q[3],q[0];
ry(-0.048554281) q[0];
cx q[1],q[0];
ry(-0.0038978569) q[0];
cx q[2],q[0];
ry(-0.024426126) q[0];
cx q[1],q[0];
u3(1.5217244,-pi,0) q[0];
cx q[4],q[1];
ry(-0.32183425) q[1];
cx q[2],q[1];
ry(-0.044235404) q[1];
cx q[3],q[1];
ry(-0.0092159449) q[1];
cx q[2],q[1];
ry(-0.090209643) q[1];
cx q[4],q[1];
ry(-0.17170629) q[1];
cx q[2],q[1];
ry(-0.021285711) q[1];
cx q[3],q[1];
ry(-0.085977103) q[1];
cx q[2],q[1];
u3(1.4893624,-pi,0) q[1];
cx q[4],q[2];
ry(-0.55611414) q[2];
cx q[3],q[2];
ry(-0.12296747) q[2];
cx q[4],q[2];
ry(-0.27136875) q[2];
cx q[3],q[2];
u3(1.4729386,-pi,0) q[2];
cx q[4],q[3];
ry(-0.8865441) q[3];
cx q[4],q[3];
u3(1.4902642,-pi,0) q[3];
u3(1.6278011,-pi,0) q[4];
u3(5*pi/8,-pi,0) q[5];
cu1(pi/16) q[4],q[5];
cx q[4],q[3];
cu1(-pi/16) q[3],q[5];
cx q[4],q[3];
cu1(pi/16) q[3],q[5];
cx q[3],q[2];
cu1(-pi/16) q[2],q[5];
cx q[4],q[2];
cu1(pi/16) q[2],q[5];
cx q[3],q[2];
cu1(-pi/16) q[2],q[5];
cx q[4],q[2];
cu1(pi/16) q[2],q[5];
cx q[2],q[1];
cu1(-pi/16) q[1],q[5];
cx q[4],q[1];
cu1(pi/16) q[1],q[5];
cx q[3],q[1];
cu1(-pi/16) q[1],q[5];
cx q[4],q[1];
cu1(pi/16) q[1],q[5];
cx q[2],q[1];
cu1(-pi/16) q[1],q[5];
cx q[4],q[1];
cu1(pi/16) q[1],q[5];
cx q[3],q[1];
cu1(-pi/16) q[1],q[5];
cx q[4],q[1];
cu1(pi/16) q[1],q[5];
cx q[1],q[0];
cu1(-pi/16) q[0],q[5];
cx q[4],q[0];
cu1(pi/16) q[0],q[5];
cx q[3],q[0];
cu1(-pi/16) q[0],q[5];
cx q[4],q[0];
cu1(pi/16) q[0],q[5];
cx q[2],q[0];
cu1(-pi/16) q[0],q[5];
cx q[4],q[0];
cu1(pi/16) q[0],q[5];
cx q[3],q[0];
cu1(-pi/16) q[0],q[5];
cx q[4],q[0];
cu1(pi/16) q[0],q[5];
cx q[1],q[0];
cu1(-pi/16) q[0],q[5];
u3(1.4893624,-pi,0) q[1];
cx q[4],q[0];
cu1(pi/16) q[0],q[5];
cx q[3],q[0];
cu1(-pi/16) q[0],q[5];
cx q[4],q[0];
cu1(pi/16) q[0],q[5];
cx q[2],q[0];
cu1(-pi/16) q[0],q[5];
u3(1.4729386,-pi,0) q[2];
cx q[4],q[0];
cu1(pi/16) q[0],q[5];
cx q[3],q[0];
cu1(-pi/16) q[0],q[5];
u3(1.4902642,-pi,0) q[3];
cx q[4],q[0];
cu1(pi/16) q[0],q[5];
u3(1.5217244,-pi,0) q[0];
u3(1.6278011,-pi,0) q[4];
cx q[4],q[3];
ry(0.8865441) q[3];
cx q[4],q[3];
cx q[3],q[2];
ry(0.27136875) q[2];
cx q[4],q[2];
ry(0.12296747) q[2];
cx q[3],q[2];
ry(0.55611414) q[2];
cx q[4],q[2];
cx q[2],q[1];
ry(0.085977103) q[1];
cx q[3],q[1];
ry(0.021285711) q[1];
cx q[2],q[1];
ry(0.17170629) q[1];
cx q[4],q[1];
ry(0.090209643) q[1];
cx q[2],q[1];
ry(0.0092159449) q[1];
cx q[3],q[1];
ry(0.044235404) q[1];
cx q[2],q[1];
ry(0.32183425) q[1];
cx q[4],q[1];
cx q[1],q[0];
ry(0.024426126) q[0];
cx q[2],q[0];
ry(0.0038978569) q[0];
cx q[1],q[0];
ry(0.048554281) q[0];
cx q[3],q[0];
ry(0.015244745) q[0];
cx q[1],q[0];
ry(0.0012686783) q[0];
cx q[2],q[0];
ry(0.0076534546) q[0];
cx q[1],q[0];
ry(0.094541741) q[0];
cx q[4],q[0];
ry(0.05275865) q[0];
cx q[1],q[0];
ry(0.0046560511) q[0];
cx q[2],q[0];
ry(0.00078021278) q[0];
cx q[1],q[0];
ry(0.0092972183) q[0];
cx q[3],q[0];
ry(0.027179311) q[0];
cx q[1],q[0];
ry(0.0023524332) q[0];
cx q[2],q[0];
ry(0.013674426) q[0];
cx q[1],q[0];
ry(0.17090792) q[0];
cx q[4],q[0];
u3(5*pi/8,-pi,0) q[5];
cry(0) q[0],q[5];
cry(0) q[1],q[5];
x q[1];
ccx q[1],q[7],q[8];
cry(0) q[2],q[5];
ccx q[2],q[8],q[9];
cry(0) q[3],q[5];
ccx q[3],q[9],q[10];
x q[10];
cry(0) q[4],q[5];
x q[4];
ccx q[4],q[10],q[6];
x q[10];
ccx q[3],q[9],q[10];
ccx q[2],q[8],q[9];
x q[4];
cx q[6],q[5];
u(0.29425236,0,0) q[5];
cx q[6],q[5];
u3(0.29425236,-pi,-pi) q[5];
cx q[6],q[5];
u(-0.011079862,0,0) q[5];
cx q[6],q[5];
u(0.011079862,0,0) q[5];
ccx q[6],q[0],q[5];
cx q[6],q[5];
u(0.011079862,0,0) q[5];
cx q[6],q[5];
u(-0.011079862,0,0) q[5];
ccx q[6],q[0],q[5];
cx q[6],q[5];
u(-0.022159724,0,0) q[5];
cx q[6],q[5];
u(0.022159724,0,0) q[5];
x q[8];
ccx q[1],q[7],q[8];
x q[1];
ccx q[6],q[1],q[5];
cx q[6],q[5];
u(0.022159724,0,0) q[5];
cx q[6],q[5];
u(-0.022159724,0,0) q[5];
ccx q[6],q[1],q[5];
x q[1];
ccx q[1],q[7],q[8];
cx q[6],q[5];
u(-0.044319448,0,0) q[5];
cx q[6],q[5];
u(0.044319448,0,0) q[5];
ccx q[6],q[2],q[5];
cx q[6],q[5];
u(0.044319448,0,0) q[5];
cx q[6],q[5];
u(-0.044319448,0,0) q[5];
ccx q[6],q[2],q[5];
cx q[6],q[5];
u(-0.088638896,0,0) q[5];
cx q[6],q[5];
u(0.088638896,0,0) q[5];
ccx q[6],q[3],q[5];
cx q[6],q[5];
u(0.088638896,0,0) q[5];
cx q[6],q[5];
u(-0.088638896,0,0) q[5];
ccx q[6],q[3],q[5];
cx q[6],q[5];
u(-0.17727779,0,0) q[5];
cx q[6],q[5];
u(0.17727779,0,0) q[5];
ccx q[6],q[4],q[5];
cx q[6],q[5];
u(0.17727779,0,0) q[5];
cx q[6],q[5];
u(-0.17727779,0,0) q[5];
ccx q[6],q[4],q[5];
x q[4];
x q[8];
ccx q[2],q[8],q[9];
ccx q[3],q[9],q[10];
x q[10];
ccx q[4],q[10],q[6];
x q[10];
ccx q[3],q[9],q[10];
ccx q[2],q[8],q[9];
ccx q[1],q[7],q[8];
x q[1];
x q[4];
x q[6];
x q[7];
x q[8];
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
