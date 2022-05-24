// Benchmark was created by MQT Bench on 2022-04-07
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 0.1.0
// Qiskit version: {'qiskit-terra': '0.19.2', 'qiskit-aer': '0.10.3', 'qiskit-ignis': '0.7.0', 'qiskit-ibmq-provider': '0.18.3', 'qiskit-aqua': '0.9.5', 'qiskit': '0.34.2', 'qiskit-nature': '0.3.1', 'qiskit-finance': '0.3.1', 'qiskit-optimization': '0.3.1', 'qiskit-machine-learning': '0.3.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg meas[8];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
cx q[7],q[6];
rz(-26.327951) q[6];
cx q[7],q[6];
cx q[7],q[5];
rz(-26.328108) q[5];
cx q[7],q[5];
cx q[6],q[5];
rz(-26.334041) q[5];
cx q[6],q[5];
cx q[7],q[4];
rz(-26.328393) q[4];
cx q[7],q[4];
cx q[6],q[4];
rz(-26.328329) q[4];
cx q[6],q[4];
cx q[5],q[4];
rz(-26.327992) q[4];
cx q[5],q[4];
cx q[7],q[3];
rz(-26.328085) q[3];
cx q[7],q[3];
cx q[6],q[3];
rz(-26.328344) q[3];
cx q[6],q[3];
cx q[5],q[3];
rz(-26.328394) q[3];
cx q[5],q[3];
cx q[4],q[3];
rz(-26.328127) q[3];
cx q[4],q[3];
cx q[7],q[2];
rz(-26.328207) q[2];
cx q[7],q[2];
cx q[6],q[2];
rz(-26.328357) q[2];
cx q[6],q[2];
cx q[5],q[2];
rz(-26.327484) q[2];
cx q[5],q[2];
cx q[4],q[2];
rz(-26.328232) q[2];
cx q[4],q[2];
cx q[3],q[2];
rz(-26.328348) q[2];
cx q[3],q[2];
cx q[7],q[1];
rz(-26.328218) q[1];
cx q[7],q[1];
cx q[6],q[1];
rz(-26.328368) q[1];
cx q[6],q[1];
cx q[5],q[1];
rz(-26.32549) q[1];
cx q[5],q[1];
cx q[4],q[1];
rz(-26.328259) q[1];
cx q[4],q[1];
cx q[3],q[1];
rz(-26.328275) q[1];
cx q[3],q[1];
cx q[2],q[1];
rz(-26.328446) q[1];
cx q[2],q[1];
cx q[7],q[0];
rz(-26.328071) q[0];
cx q[7],q[0];
cx q[6],q[0];
rz(-26.327225) q[0];
cx q[6],q[0];
cx q[5],q[0];
rz(-26.326264) q[0];
cx q[5],q[0];
cx q[4],q[0];
rz(-26.328235) q[0];
cx q[4],q[0];
cx q[3],q[0];
rz(-26.328163) q[0];
cx q[3],q[0];
cx q[2],q[0];
rz(-26.32795) q[0];
cx q[2],q[0];
cx q[1],q[0];
rz(-26.328318) q[0];
cx q[1],q[0];
u3(2.7120514,-pi/2,1.6012787) q[0];
u3(2.7120514,-pi/2,1.5859849) q[1];
u3(2.7120514,-pi/2,1.5617692) q[2];
u3(2.7120514,-pi/2,1.571688) q[3];
u3(2.7120514,-pi/2,1.5810567) q[4];
u3(2.7120514,-pi/2,2.1140951) q[5];
u3(2.7120514,-pi/2,1.5680835) q[6];
u3(2.7120514,-pi/2,1.5630443) q[7];
cx q[7],q[6];
rz(46.784733) q[6];
cx q[7],q[6];
cx q[7],q[5];
rz(46.785013) q[5];
cx q[7],q[5];
cx q[6],q[5];
rz(46.795556) q[5];
cx q[6],q[5];
cx q[7],q[4];
rz(46.785519) q[4];
cx q[7],q[4];
cx q[6],q[4];
rz(46.785406) q[4];
cx q[6],q[4];
cx q[5],q[4];
rz(46.784808) q[4];
cx q[5],q[4];
cx q[7],q[3];
rz(46.784972) q[3];
cx q[7],q[3];
cx q[6],q[3];
rz(46.785432) q[3];
cx q[6],q[3];
cx q[5],q[3];
rz(46.785521) q[3];
cx q[5],q[3];
cx q[4],q[3];
rz(46.785047) q[3];
cx q[4],q[3];
cx q[7],q[2];
rz(46.78519) q[2];
cx q[7],q[2];
cx q[6],q[2];
rz(46.785456) q[2];
cx q[6],q[2];
cx q[5],q[2];
rz(46.783905) q[2];
cx q[5],q[2];
cx q[4],q[2];
rz(46.785234) q[2];
cx q[4],q[2];
cx q[3],q[2];
rz(46.78544) q[2];
cx q[3],q[2];
cx q[7],q[1];
rz(46.785208) q[1];
cx q[7],q[1];
cx q[6],q[1];
rz(46.785474) q[1];
cx q[6],q[1];
cx q[5],q[1];
rz(46.780362) q[1];
cx q[5],q[1];
cx q[4],q[1];
rz(46.785281) q[1];
cx q[4],q[1];
cx q[3],q[1];
rz(46.785311) q[1];
cx q[3],q[1];
cx q[2],q[1];
rz(46.785614) q[1];
cx q[2],q[1];
cx q[7],q[0];
rz(46.784947) q[0];
cx q[7],q[0];
cx q[6],q[0];
rz(46.783445) q[0];
cx q[6],q[0];
cx q[5],q[0];
rz(46.781736) q[0];
cx q[5],q[0];
cx q[4],q[0];
rz(46.785238) q[0];
cx q[4],q[0];
cx q[3],q[0];
rz(46.78511) q[0];
cx q[3],q[0];
cx q[2],q[0];
rz(46.784732) q[0];
cx q[2],q[0];
cx q[1],q[0];
rz(46.785386) q[0];
cx q[1],q[0];
u3(0.68916299,-pi/2,1.5166291) q[0];
u3(0.68916299,-pi/2,1.5438062) q[1];
u3(0.68916299,-pi/2,1.5868376) q[2];
u3(0.68916299,-pi/2,1.5692118) q[3];
u3(0.68916299,-pi/2,1.5525637) q[4];
u3(0.68916299,-pi/2,0.60535519) q[5];
u3(0.68916299,-pi/2,1.5756171) q[6];
u3(0.68916299,-pi/2,1.5845716) q[7];
cx q[7],q[6];
rz(-17.615657) q[6];
cx q[7],q[6];
cx q[7],q[5];
rz(-17.615763) q[5];
cx q[7],q[5];
cx q[6],q[5];
rz(-17.619732) q[5];
cx q[6],q[5];
cx q[7],q[4];
rz(-17.615953) q[4];
cx q[7],q[4];
cx q[6],q[4];
rz(-17.615911) q[4];
cx q[6],q[4];
cx q[5],q[4];
rz(-17.615685) q[4];
cx q[5],q[4];
cx q[7],q[3];
rz(-17.615747) q[3];
cx q[7],q[3];
cx q[6],q[3];
rz(-17.615921) q[3];
cx q[6],q[3];
cx q[5],q[3];
rz(-17.615954) q[3];
cx q[5],q[3];
cx q[4],q[3];
rz(-17.615776) q[3];
cx q[4],q[3];
cx q[7],q[2];
rz(-17.615829) q[2];
cx q[7],q[2];
cx q[6],q[2];
rz(-17.615929) q[2];
cx q[6],q[2];
cx q[5],q[2];
rz(-17.615346) q[2];
cx q[5],q[2];
cx q[4],q[2];
rz(-17.615846) q[2];
cx q[4],q[2];
cx q[3],q[2];
rz(-17.615923) q[2];
cx q[3],q[2];
cx q[7],q[1];
rz(-17.615836) q[1];
cx q[7],q[1];
cx q[6],q[1];
rz(-17.615936) q[1];
cx q[6],q[1];
cx q[5],q[1];
rz(-17.614011) q[1];
cx q[5],q[1];
cx q[4],q[1];
rz(-17.615864) q[1];
cx q[4],q[1];
cx q[3],q[1];
rz(-17.615875) q[1];
cx q[3],q[1];
cx q[2],q[1];
rz(-17.615989) q[1];
cx q[2],q[1];
cx q[7],q[0];
rz(-17.615738) q[0];
cx q[7],q[0];
cx q[6],q[0];
rz(-17.615172) q[0];
cx q[6],q[0];
cx q[5],q[0];
rz(-17.614529) q[0];
cx q[5],q[0];
cx q[4],q[0];
rz(-17.615848) q[0];
cx q[4],q[0];
cx q[3],q[0];
rz(-17.615799) q[0];
cx q[3],q[0];
cx q[2],q[0];
rz(-17.615657) q[0];
cx q[2],q[0];
cx q[1],q[0];
rz(-17.615903) q[0];
cx q[1],q[0];
u3(0.72883034,pi/2,-1.550401) q[0];
u3(0.72883034,pi/2,-1.5606339) q[1];
u3(0.72883034,pi/2,-1.5768363) q[2];
u3(0.72883034,pi/2,-1.5701997) q[3];
u3(0.72883034,pi/2,-1.5639313) q[4];
u3(0.72883034,pi/2,-1.2072829) q[5];
u3(0.72883034,pi/2,-1.5726115) q[6];
u3(0.72883034,pi/2,-1.5759831) q[7];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
