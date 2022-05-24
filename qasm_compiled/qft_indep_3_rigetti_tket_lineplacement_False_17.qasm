OPENQASM 2.0;
include "qelib1.inc";

qreg node[3];
creg c[3];
creg meas[3];
rz(0.5*pi) node[0];
rz(3.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[0];
rx(3.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(3.8750000000000004*pi) node[1];
rz(0.5*pi) node[2];
cz node[1],node[2];
rz(0.5*pi) node[2];
rx(0.5*pi) node[2];
rz(0.5*pi) node[2];
rz(3.75*pi) node[2];
rz(0.5*pi) node[2];
rx(0.5*pi) node[2];
rz(0.5*pi) node[2];
cz node[1],node[2];
cz node[1],node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rx(0.5*pi) node[2];
rx(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.75*pi) node[2];
rz(3.8750000000000004*pi) node[0];
rx(2.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.75*pi) node[2];
rx(0.5*pi) node[0];
rz(0.5*pi) node[0];
cz node[1],node[0];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.1250000000000001*pi) node[0];
cz node[2],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[1],node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[2],node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[1],node[0];
rz(0.5*pi) node[0];
rx(0.5*pi) node[0];
rz(0.5*pi) node[0];
rz(3.75*pi) node[0];
rz(0.5*pi) node[0];
rx(0.5*pi) node[0];
rz(0.5*pi) node[0];
cz node[1],node[0];
rz(0.5*pi) node[0];
rx(0.5*pi) node[0];
rz(0.5*pi) node[0];
rz(0.75*pi) node[0];
rx(2.5*pi) node[0];
rz(0.5*pi) node[0];
barrier node[2],node[1],node[0];
measure node[2] -> meas[0];
measure node[1] -> meas[1];
measure node[0] -> meas[2];
