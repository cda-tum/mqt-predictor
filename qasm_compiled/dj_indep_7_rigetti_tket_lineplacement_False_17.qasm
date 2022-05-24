OPENQASM 2.0;
include "qelib1.inc";

qreg node[15];
creg c[6];
rz(3.5*pi) node[0];
rz(0.5*pi) node[1];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
rz(3.5*pi) node[7];
rz(3.5*pi) node[13];
rz(3.5*pi) node[14];
rx(1.5*pi) node[0];
rx(1.5*pi) node[1];
rx(1.5*pi) node[2];
rx(1.5*pi) node[3];
rx(1.5*pi) node[7];
rx(1.5*pi) node[13];
rx(1.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[0],node[1];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
cz node[7],node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[0];
cz node[2],node[1];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
cz node[0],node[7];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[3],node[2];
rz(0.5*pi) node[7];
cz node[7],node[0];
cz node[14],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[14];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[2],node[3];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
cz node[0],node[1];
cz node[3],node[2];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[2],node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
cz node[13],node[2];
rz(0.5*pi) node[2];
rx(0.5*pi) node[2];
rz(0.5*pi) node[2];
cz node[2],node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
cz node[13],node[2];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[13];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[2];
cz node[2],node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
barrier node[7],node[3],node[14],node[0],node[2],node[13],node[1];
measure node[7] -> c[0];
measure node[3] -> c[1];
measure node[14] -> c[2];
measure node[0] -> c[3];
measure node[2] -> c[4];
measure node[13] -> c[5];
