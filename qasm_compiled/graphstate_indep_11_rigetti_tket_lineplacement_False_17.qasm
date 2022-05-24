OPENQASM 2.0;
include "qelib1.inc";

qreg node[16];
creg meas[11];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[4];
rx(0.5*pi) node[5];
rx(0.5*pi) node[6];
rx(0.5*pi) node[7];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[0],node[1];
cz node[3],node[2];
cz node[5],node[6];
cz node[0],node[7];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[5],node[4];
rz(0.5*pi) node[6];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[4];
rx(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[4];
rz(0.5*pi) node[6];
rx(0.5*pi) node[7];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[4];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[4];
rx(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[4];
rz(0.5*pi) node[6];
rx(0.5*pi) node[7];
cz node[1],node[14];
cz node[2],node[13];
rz(0.5*pi) node[4];
rz(0.5*pi) node[7];
cz node[3],node[4];
cz node[6],node[7];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[4];
rz(0.5*pi) node[7];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[4];
rx(0.5*pi) node[7];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[4];
rz(0.5*pi) node[7];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[4];
rz(0.5*pi) node[7];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[4];
rx(0.5*pi) node[7];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[4];
rz(0.5*pi) node[7];
cz node[14],node[15];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[15],node[14];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[14],node[15];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[15],node[14];
rz(0.5*pi) node[14];
rx(0.5*pi) node[14];
rz(0.5*pi) node[14];
rz(0.5*pi) node[14];
rx(0.5*pi) node[14];
rz(0.5*pi) node[14];
cz node[13],node[14];
rz(0.5*pi) node[14];
rx(0.5*pi) node[14];
rz(0.5*pi) node[14];
rz(0.5*pi) node[14];
rx(0.5*pi) node[14];
rz(0.5*pi) node[14];
barrier node[0],node[1],node[5],node[6],node[3],node[2],node[15],node[4],node[7],node[13],node[14];
measure node[0] -> meas[0];
measure node[1] -> meas[1];
measure node[5] -> meas[2];
measure node[6] -> meas[3];
measure node[3] -> meas[4];
measure node[2] -> meas[5];
measure node[15] -> meas[6];
measure node[4] -> meas[7];
measure node[7] -> meas[8];
measure node[13] -> meas[9];
measure node[14] -> meas[10];
