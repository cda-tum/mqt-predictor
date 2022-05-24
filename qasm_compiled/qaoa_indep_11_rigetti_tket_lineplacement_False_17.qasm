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
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
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
rx(0.5*pi) node[8];
rx(0.5*pi) node[9];
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
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[5];
rz(0.5*pi) node[7];
rz(0.5*pi) node[15];
rx(0.5*pi) node[5];
rx(0.5*pi) node[7];
rx(0.5*pi) node[15];
rz(0.5*pi) node[5];
rz(0.5*pi) node[7];
rz(0.5*pi) node[15];
cz node[6],node[5];
cz node[8],node[15];
rz(0.5*pi) node[5];
rz(0.5*pi) node[15];
rx(0.5*pi) node[5];
rx(0.5*pi) node[15];
rz(0.5*pi) node[5];
rz(0.5*pi) node[15];
rz(2.9192243270397458*pi) node[5];
rz(2.9192243270397458*pi) node[15];
rz(0.5*pi) node[5];
rz(0.5*pi) node[15];
rx(0.5*pi) node[5];
rx(0.5*pi) node[15];
rz(0.5*pi) node[5];
rz(0.5*pi) node[15];
cz node[6],node[5];
cz node[8],node[15];
rz(0.5*pi) node[5];
cz node[6],node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[15];
rx(0.5*pi) node[5];
rz(0.5*pi) node[7];
rx(0.5*pi) node[8];
rx(0.5*pi) node[15];
rz(0.5*pi) node[5];
rx(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[15];
rz(0.5*pi) node[5];
rz(0.5*pi) node[7];
cz node[9],node[8];
rz(0.5*pi) node[15];
rx(0.5*pi) node[5];
rz(2.9192243270397458*pi) node[7];
rz(0.5*pi) node[8];
rx(0.5*pi) node[15];
rz(0.5*pi) node[5];
rz(0.5*pi) node[7];
rx(0.5*pi) node[8];
rz(0.5*pi) node[15];
cz node[4],node[5];
rx(0.5*pi) node[7];
rz(0.5*pi) node[8];
cz node[14],node[15];
rz(0.5*pi) node[5];
rz(0.5*pi) node[7];
rz(2.9192243270397458*pi) node[8];
rz(0.5*pi) node[15];
rx(0.5*pi) node[5];
cz node[6],node[7];
rz(0.5*pi) node[8];
rx(0.5*pi) node[15];
rz(0.5*pi) node[5];
rx(2.660052291126984*pi) node[6];
rz(0.5*pi) node[7];
rx(0.5*pi) node[8];
rz(0.5*pi) node[15];
rz(2.9192243270397458*pi) node[5];
rx(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(2.9192243270397458*pi) node[15];
rz(0.5*pi) node[5];
rz(0.5*pi) node[7];
cz node[9],node[8];
rz(0.5*pi) node[15];
rx(0.5*pi) node[5];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rx(0.5*pi) node[15];
rz(0.5*pi) node[5];
rx(0.5*pi) node[7];
rx(0.5*pi) node[8];
rz(0.5*pi) node[15];
cz node[4],node[5];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
cz node[14],node[15];
cz node[0],node[7];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rx(0.660052291126984*pi) node[8];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[4];
rx(0.5*pi) node[5];
rz(0.5*pi) node[7];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rx(0.5*pi) node[7];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[1],node[14];
cz node[3],node[4];
rx(2.660052291126984*pi) node[5];
rz(0.5*pi) node[7];
rx(2.660052291126984*pi) node[15];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(2.9192243270397458*pi) node[7];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[4];
rx(0.5*pi) node[5];
rz(0.5*pi) node[7];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rx(0.5*pi) node[7];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(2.9192243270397458*pi) node[4];
cz node[6],node[5];
rz(0.5*pi) node[7];
cz node[8],node[15];
rz(2.9192243270397458*pi) node[14];
cz node[0],node[7];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[4];
rx(0.5*pi) node[5];
rz(0.5*pi) node[7];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rx(0.5*pi) node[7];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[1],node[14];
cz node[3],node[4];
rz(1.907275795698785*pi) node[5];
rz(0.5*pi) node[7];
rz(1.907275795698785*pi) node[15];
cz node[3],node[2];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rx(2.660052291126984*pi) node[7];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[4];
rx(0.5*pi) node[5];
rz(0.5*pi) node[7];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rx(0.5*pi) node[7];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.660052291126984*pi) node[4];
cz node[6],node[5];
rz(0.5*pi) node[7];
cz node[8],node[15];
rx(0.660052291126984*pi) node[14];
cz node[2],node[3];
rz(0.5*pi) node[5];
cz node[6],node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[5];
rz(0.5*pi) node[7];
rx(0.5*pi) node[8];
rx(0.5*pi) node[15];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[5];
rx(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[5];
rz(0.5*pi) node[7];
cz node[9],node[8];
rz(0.5*pi) node[15];
cz node[3],node[2];
rx(0.5*pi) node[5];
rz(1.907275795698785*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rx(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[5];
rz(0.5*pi) node[7];
rx(0.5*pi) node[8];
rx(0.5*pi) node[9];
rz(0.5*pi) node[15];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
cz node[4],node[5];
rx(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
cz node[14],node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[5];
rz(0.5*pi) node[7];
cz node[8],node[9];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rx(0.5*pi) node[5];
cz node[6],node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rx(0.5*pi) node[15];
rx(0.5*pi) node[2];
rz(0.5*pi) node[5];
rx(0.6272208390585893*pi) node[6];
rz(0.5*pi) node[7];
rx(0.5*pi) node[8];
rx(0.5*pi) node[9];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(1.907275795698785*pi) node[5];
rx(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(1.907275795698785*pi) node[15];
cz node[1],node[2];
rz(0.5*pi) node[5];
rz(0.5*pi) node[7];
cz node[9],node[8];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rx(0.5*pi) node[5];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rx(0.5*pi) node[15];
rx(0.5*pi) node[2];
rz(0.5*pi) node[5];
rx(0.5*pi) node[7];
rx(0.5*pi) node[8];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
cz node[4],node[5];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
cz node[14],node[15];
rz(2.9192243270397458*pi) node[2];
cz node[4],node[3];
rz(0.5*pi) node[5];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rx(0.5*pi) node[5];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rx(2.627220839058589*pi) node[5];
rx(2.627220839058589*pi) node[15];
cz node[1],node[2];
cz node[3],node[4];
rz(0.5*pi) node[15];
rx(2.660052291126984*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rx(0.5*pi) node[15];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[4];
rz(0.5*pi) node[15];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
cz node[8],node[15];
rz(0.5*pi) node[1];
rx(0.660052291126984*pi) node[2];
cz node[4],node[3];
rz(0.5*pi) node[8];
rz(0.5*pi) node[15];
rz(0.5*pi) node[3];
rx(0.5*pi) node[8];
rx(0.5*pi) node[15];
rx(0.5*pi) node[3];
rz(0.5*pi) node[8];
rz(0.5*pi) node[15];
rz(0.5*pi) node[3];
cz node[15],node[8];
rz(0.5*pi) node[3];
rz(0.5*pi) node[8];
rz(0.5*pi) node[15];
rx(0.5*pi) node[3];
rx(0.5*pi) node[8];
rx(0.5*pi) node[15];
rz(0.5*pi) node[3];
rz(0.5*pi) node[8];
rz(0.5*pi) node[15];
cz node[2],node[3];
cz node[8],node[15];
rz(0.5*pi) node[3];
rz(0.5*pi) node[8];
rz(0.5*pi) node[15];
rx(0.5*pi) node[3];
rx(0.5*pi) node[8];
rx(0.5*pi) node[15];
rz(0.5*pi) node[3];
rz(0.5*pi) node[8];
rz(0.5*pi) node[15];
rz(1.907275795698785*pi) node[3];
cz node[9],node[8];
cz node[15],node[14];
rz(0.5*pi) node[3];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[3];
rx(0.5*pi) node[8];
rx(0.5*pi) node[9];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[3];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[2],node[3];
cz node[8],node[9];
cz node[14],node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[8];
rx(0.5*pi) node[9];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(2.627220839058589*pi) node[3];
cz node[9],node[8];
cz node[15],node[14];
rz(0.5*pi) node[8];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[8];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[8];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[14],node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
cz node[1],node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
cz node[14],node[1];
rz(0.5*pi) node[1];
cz node[14],node[15];
rx(0.5*pi) node[1];
rz(0.5*pi) node[15];
rz(0.5*pi) node[1];
rx(0.5*pi) node[15];
rz(0.5*pi) node[1];
rz(0.5*pi) node[15];
rx(0.5*pi) node[1];
rz(1.907275795698785*pi) node[15];
rz(0.5*pi) node[1];
rz(0.5*pi) node[15];
cz node[0],node[1];
rx(0.5*pi) node[15];
rz(0.5*pi) node[1];
rz(0.5*pi) node[15];
rx(0.5*pi) node[1];
cz node[14],node[15];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(2.9192243270397458*pi) node[1];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[1];
rx(2.627220839058589*pi) node[15];
rz(0.5*pi) node[1];
rz(0.5*pi) node[15];
cz node[0],node[1];
rx(0.5*pi) node[15];
rx(2.660052291126984*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[15];
cz node[0],node[7];
rx(0.5*pi) node[1];
cz node[8],node[15];
rz(0.5*pi) node[1];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[15];
rx(0.660052291126984*pi) node[1];
rx(0.5*pi) node[7];
rx(0.5*pi) node[8];
rx(0.5*pi) node[15];
cz node[1],node[14];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[15];
rz(0.5*pi) node[1];
rz(1.907275795698785*pi) node[7];
cz node[15],node[8];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rx(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[1];
rx(0.5*pi) node[7];
rx(0.5*pi) node[8];
rz(0.5*pi) node[14];
rx(0.5*pi) node[15];
cz node[14],node[1];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[15];
cz node[0],node[7];
rz(0.5*pi) node[1];
cz node[8],node[15];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[7];
rx(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[1];
rx(0.5*pi) node[7];
rz(0.5*pi) node[14];
rx(0.5*pi) node[15];
cz node[1],node[14];
rz(0.5*pi) node[7];
rz(0.5*pi) node[15];
cz node[1],node[2];
rx(2.627220839058589*pi) node[7];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
cz node[14],node[15];
rz(1.907275795698785*pi) node[2];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rx(0.5*pi) node[15];
rx(0.5*pi) node[2];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(1.907275795698785*pi) node[15];
cz node[1],node[2];
rz(0.5*pi) node[15];
rx(0.6272208390585893*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[15];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[15];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[14],node[15];
rz(0.5*pi) node[1];
rx(2.627220839058589*pi) node[2];
rz(0.5*pi) node[15];
cz node[14],node[1];
rx(0.5*pi) node[15];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[1];
rx(0.5*pi) node[14];
rx(2.627220839058589*pi) node[15];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
cz node[1],node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
cz node[14],node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[0],node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
rz(1.907275795698785*pi) node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[0],node[1];
rx(0.6272208390585893*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
rx(2.627220839058589*pi) node[1];
barrier node[0],node[14],node[2],node[1],node[8],node[3],node[15],node[6],node[7],node[5],node[9];
measure node[0] -> meas[0];
measure node[14] -> meas[1];
measure node[2] -> meas[2];
measure node[1] -> meas[3];
measure node[8] -> meas[4];
measure node[3] -> meas[5];
measure node[15] -> meas[6];
measure node[6] -> meas[7];
measure node[7] -> meas[8];
measure node[5] -> meas[9];
measure node[9] -> meas[10];
