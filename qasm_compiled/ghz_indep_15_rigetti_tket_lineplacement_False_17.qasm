OPENQASM 2.0;
include "qelib1.inc";

qreg node[56];
creg meas[15];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[54];
rz(0.5*pi) node[55];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[4];
rx(0.5*pi) node[5];
rx(0.5*pi) node[6];
rx(0.5*pi) node[7];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[40];
rx(0.5*pi) node[41];
rx(0.5*pi) node[54];
rx(0.5*pi) node[55];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[54];
rz(0.5*pi) node[55];
cz node[5],node[4];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rx(0.5*pi) node[4];
rx(0.5*pi) node[5];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
cz node[4],node[3];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rx(0.5*pi) node[3];
rx(0.5*pi) node[4];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
cz node[3],node[2];
cz node[47],node[4];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[47];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[4];
rx(0.5*pi) node[47];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[47];
cz node[2],node[1];
cz node[4],node[47];
rz(0.5*pi) node[1];
rz(0.5*pi) node[4];
rz(0.5*pi) node[47];
rx(0.5*pi) node[1];
rx(0.5*pi) node[4];
rx(0.5*pi) node[47];
rz(0.5*pi) node[1];
rz(0.5*pi) node[4];
rz(0.5*pi) node[47];
cz node[1],node[0];
cz node[47],node[4];
rz(0.5*pi) node[0];
rz(0.5*pi) node[4];
rx(0.5*pi) node[0];
rx(0.5*pi) node[4];
rz(0.5*pi) node[0];
rz(0.5*pi) node[4];
cz node[0],node[7];
rz(0.5*pi) node[4];
rx(0.5*pi) node[4];
rz(0.5*pi) node[7];
rz(0.5*pi) node[4];
rx(0.5*pi) node[7];
rz(0.5*pi) node[7];
cz node[7],node[6];
rz(0.5*pi) node[6];
rx(0.5*pi) node[6];
rz(0.5*pi) node[6];
cz node[6],node[5];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rx(0.5*pi) node[5];
rx(0.5*pi) node[6];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
cz node[5],node[6];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rx(0.5*pi) node[5];
rx(0.5*pi) node[6];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
cz node[6],node[5];
rz(0.5*pi) node[5];
rx(0.5*pi) node[5];
rz(0.5*pi) node[5];
cz node[5],node[4];
rz(0.5*pi) node[4];
rx(0.5*pi) node[4];
rz(0.5*pi) node[4];
cz node[4],node[3];
rz(0.5*pi) node[3];
rx(0.5*pi) node[3];
rz(0.5*pi) node[3];
cz node[3],node[40];
rz(0.5*pi) node[3];
rz(0.5*pi) node[40];
rx(0.5*pi) node[3];
rx(0.5*pi) node[40];
rz(0.5*pi) node[3];
rz(0.5*pi) node[40];
cz node[4],node[3];
rz(0.5*pi) node[40];
rz(0.5*pi) node[3];
rx(0.5*pi) node[40];
rx(0.5*pi) node[3];
rz(0.5*pi) node[40];
rz(0.5*pi) node[3];
cz node[3],node[40];
rz(0.5*pi) node[40];
rx(0.5*pi) node[40];
rz(0.5*pi) node[40];
cz node[40],node[41];
rz(0.5*pi) node[41];
rx(0.5*pi) node[41];
rz(0.5*pi) node[41];
cz node[41],node[54];
rz(0.5*pi) node[54];
rx(0.5*pi) node[54];
rz(0.5*pi) node[54];
cz node[54],node[55];
rz(0.5*pi) node[55];
rx(0.5*pi) node[55];
rz(0.5*pi) node[55];
cz node[55],node[12];
rz(0.5*pi) node[12];
rx(0.5*pi) node[12];
rz(0.5*pi) node[12];
cz node[12],node[13];
rz(0.5*pi) node[13];
rx(0.5*pi) node[13];
rz(0.5*pi) node[13];
barrier node[13],node[12],node[55],node[54],node[41],node[40],node[4],node[5],node[7],node[0],node[1],node[2],node[3],node[47],node[6];
measure node[13] -> meas[0];
measure node[12] -> meas[1];
measure node[55] -> meas[2];
measure node[54] -> meas[3];
measure node[41] -> meas[4];
measure node[40] -> meas[5];
measure node[4] -> meas[6];
measure node[5] -> meas[7];
measure node[7] -> meas[8];
measure node[0] -> meas[9];
measure node[1] -> meas[10];
measure node[2] -> meas[11];
measure node[3] -> meas[12];
measure node[47] -> meas[13];
measure node[6] -> meas[14];
