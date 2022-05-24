OPENQASM 2.0;
include "qelib1.inc";

qreg node[15];
creg meas[6];
rz(3.5*pi) node[0];
rz(3.5*pi) node[1];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
rz(3.5*pi) node[7];
rz(0.5*pi) node[14];
rx(0.13484851412163087*pi) node[0];
rx(2.285477739984519*pi) node[1];
rx(2.207984417483903*pi) node[2];
rx(2.1943555143565368*pi) node[3];
rx(2.0165642645356163*pi) node[7];
rx(1.933697049882744*pi) node[14];
rz(1.0*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[7];
rz(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[7];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[7];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[7];
rz(0.5*pi) node[14];
cz node[0],node[1];
cz node[0],node[7];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[7];
rz(0.5*pi) node[1];
rx(0.5*pi) node[7];
rz(0.5*pi) node[1];
rz(0.5*pi) node[7];
rz(0.5*pi) node[1];
rz(0.5*pi) node[7];
rx(0.5*pi) node[1];
rx(0.5*pi) node[7];
rz(0.5*pi) node[1];
rz(0.5*pi) node[7];
cz node[0],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[1],node[0];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[0],node[1];
cz node[0],node[7];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[7];
rz(0.5*pi) node[1];
rx(0.5*pi) node[7];
cz node[1],node[2];
rz(0.5*pi) node[7];
cz node[1],node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
cz node[2],node[1];
rz(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
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
cz node[2],node[3];
rx(0.5*pi) node[1];
rx(0.21096923571710519*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[1];
rz(1.0*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[0],node[1];
cz node[3],node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[1];
cz node[2],node[3];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
cz node[0],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[3],node[2];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
cz node[1],node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[0],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[7],node[0];
cz node[1],node[14];
rz(0.5*pi) node[0];
cz node[1],node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[0];
rx(2.1960867118103113*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
rx(0.5*pi) node[0];
rx(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
cz node[7],node[0];
cz node[1],node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
cz node[0],node[7];
cz node[2],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
cz node[7],node[0];
cz node[1],node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[14],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
cz node[3],node[2];
rz(0.5*pi) node[14];
cz node[1],node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
cz node[14],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
cz node[3],node[2];
rz(0.5*pi) node[14];
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
rz(0.5*pi) node[1];
cz node[3],node[2];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
cz node[0],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[1],node[0];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[0],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[7],node[0];
cz node[1],node[14];
rz(0.5*pi) node[0];
rx(2.0394470168872116*pi) node[1];
rz(0.5*pi) node[14];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
cz node[2],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[7],node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
cz node[1],node[2];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
cz node[0],node[7];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
cz node[2],node[1];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
cz node[7],node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[0];
cz node[1],node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[14],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
cz node[3],node[2];
rz(0.5*pi) node[14];
cz node[1],node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
cz node[14],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
cz node[3],node[2];
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
cz node[2],node[3];
rx(2.2448764517450837*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[1];
cz node[3],node[2];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
cz node[0],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[1],node[0];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[0],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[7],node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(1.0*pi) node[7];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(1.8860030469285858*pi) node[7];
rz(0.5*pi) node[0];
cz node[14],node[1];
rz(0.5*pi) node[7];
rz(3.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.01388746350891493*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[7],node[0];
cz node[2],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
cz node[0],node[7];
cz node[1],node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
cz node[7],node[0];
cz node[2],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
rz(0.5*pi) node[0];
cz node[1],node[2];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[14],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
cz node[3],node[2];
rz(0.5*pi) node[14];
cz node[1],node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
cz node[14],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
cz node[3],node[2];
cz node[1],node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[2],node[3];
cz node[0],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[3],node[2];
cz node[1],node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[0],node[1];
cz node[0],node[7];
rz(0.5*pi) node[1];
rx(0.14347847868233934*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[7];
rz(1.0*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[7],node[0];
cz node[14],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[7];
cz node[0],node[7];
rz(0.5*pi) node[1];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
cz node[2],node[1];
rz(0.5*pi) node[7];
cz node[7],node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
cz node[1],node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[2],node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[1],node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[14],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
cz node[3],node[2];
rz(0.5*pi) node[14];
cz node[1],node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
cz node[14],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
cz node[3],node[2];
cz node[1],node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[0];
rx(2.1680117750793086*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[2],node[3];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
cz node[0],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[3],node[2];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
cz node[1],node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[0],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[7],node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[0];
cz node[14],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(2.030563669714277*pi) node[14];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[14];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[7],node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
cz node[2],node[1];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(2.265162878296814*pi) node[2];
rz(0.5*pi) node[7];
cz node[0],node[7];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
rx(0.5*pi) node[2];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
cz node[7],node[0];
cz node[1],node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[2],node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[1],node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[14],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
cz node[3],node[2];
rz(0.5*pi) node[14];
cz node[1],node[14];
rz(0.5*pi) node[2];
rz(1.0*pi) node[3];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(1.9303470103259401*pi) node[3];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(3.5*pi) node[2];
rz(0.5*pi) node[14];
cz node[14],node[1];
rx(0.1701719422815904*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
cz node[3],node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
cz node[0],node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[1];
cz node[2],node[3];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[1];
cz node[3],node[2];
cz node[0],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[1],node[0];
rx(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[0],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[7],node[0];
cz node[1],node[14];
rz(0.5*pi) node[0];
cz node[1],node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[0];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[0];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
rx(0.5*pi) node[0];
rx(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
cz node[7],node[0];
cz node[1],node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
cz node[0],node[7];
cz node[2],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
cz node[7],node[0];
cz node[1],node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[14],node[1];
cz node[2],node[3];
rz(0.5*pi) node[1];
rx(0.10017002479477449*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[14];
cz node[1],node[14];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
cz node[3],node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[14];
cz node[14],node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
cz node[2],node[3];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[14];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[1];
cz node[3],node[2];
cz node[0],node[1];
rz(0.5*pi) node[2];
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
cz node[0],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[1],node[0];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[0],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[7],node[0];
cz node[1],node[14];
rz(0.5*pi) node[0];
cz node[1],node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[0];
rx(0.22777323624811102*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[0];
cz node[14],node[1];
rz(0.5*pi) node[2];
cz node[7],node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[7];
rx(0.5*pi) node[14];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[7];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
cz node[1],node[14];
rz(0.5*pi) node[7];
cz node[0],node[7];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[7];
rx(0.5*pi) node[14];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[7];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
cz node[14],node[1];
rz(0.5*pi) node[7];
cz node[7],node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[0],node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[0],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[1],node[0];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[0],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[7],node[0];
cz node[1],node[2];
rz(0.5*pi) node[0];
rx(0.04997134334957564*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
cz node[2],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
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
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[0],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[1],node[0];
rz(0.5*pi) node[0];
rx(0.5*pi) node[0];
rz(0.5*pi) node[0];
rz(0.5*pi) node[0];
rx(0.5*pi) node[0];
rz(0.5*pi) node[0];
cz node[7],node[0];
rz(0.5*pi) node[0];
rx(0.2551915956010156*pi) node[7];
rx(0.5*pi) node[0];
rz(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[0];
rx(0.5*pi) node[0];
rz(0.5*pi) node[0];
cz node[1],node[0];
rz(0.5*pi) node[0];
rx(0.20297532515999317*pi) node[1];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(3.5*pi) node[0];
rx(2.2267767690837967*pi) node[0];
rz(0.5*pi) node[0];
barrier node[3],node[14],node[2],node[7],node[1],node[0];
measure node[3] -> meas[0];
measure node[14] -> meas[1];
measure node[2] -> meas[2];
measure node[7] -> meas[3];
measure node[1] -> meas[4];
measure node[0] -> meas[5];
