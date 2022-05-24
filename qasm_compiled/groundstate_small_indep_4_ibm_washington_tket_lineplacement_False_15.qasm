OPENQASM 2.0;
include "qelib1.inc";

qreg node[15];
creg meas[4];
rz(3.8654418931375574*pi) node[0];
rz(3.295769378545352*pi) node[1];
rz(3.2868820996780244*pi) node[2];
rz(3.4598248996871765*pi) node[14];
sx node[0];
sx node[1];
sx node[2];
sx node[14];
rz(3.5*pi) node[0];
rz(1.5*pi) node[1];
rz(1.5*pi) node[2];
rz(1.5*pi) node[14];
sx node[0];
sx node[1];
sx node[2];
sx node[14];
rz(0.7470783715094114*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
cx node[0],node[1];
cx node[0],node[14];
sx node[1];
rz(2.5*pi) node[1];
sx node[1];
rz(3.37553913040158*pi) node[1];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[0],node[14];
cx node[1],node[2];
sx node[1];
sx node[14];
rz(3.5*pi) node[1];
rz(2.5*pi) node[14];
sx node[1];
sx node[14];
rz(2.996369189754164*pi) node[1];
rz(1.9542707190122357*pi) node[14];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[0],node[1];
sx node[0];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[14],node[0];
cx node[2],node[1];
rz(3.5*pi) node[0];
sx node[1];
sx node[14];
sx node[0];
rz(2.5*pi) node[1];
rz(0.17825953754694357*pi) node[0];
sx node[1];
sx node[0];
rz(2.6200856598527547*pi) node[1];
rz(1.0*pi) node[0];
cx node[2],node[1];
cx node[14],node[0];
cx node[1],node[2];
cx node[0],node[14];
cx node[2],node[1];
cx node[14],node[0];
cx node[1],node[0];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[0],node[14];
cx node[2],node[1];
rz(0.5*pi) node[0];
sx node[1];
sx node[0];
rz(2.5*pi) node[1];
rz(3.5*pi) node[0];
sx node[1];
sx node[0];
rz(3.212450156208937*pi) node[1];
rz(1.0*pi) node[0];
cx node[2],node[1];
cx node[14],node[0];
cx node[0],node[14];
cx node[14],node[0];
cx node[1],node[0];
cx node[2],node[1];
cx node[1],node[0];
rz(0.5*pi) node[2];
cx node[1],node[0];
sx node[2];
rz(1.5*pi) node[0];
rz(0.5*pi) node[1];
rz(3.5*pi) node[2];
sx node[0];
sx node[1];
sx node[2];
rz(1.2086295536934637*pi) node[0];
rz(3.5*pi) node[1];
rz(1.0*pi) node[2];
sx node[0];
sx node[1];
rz(1.5*pi) node[0];
rz(1.0*pi) node[1];
barrier node[14],node[2],node[1],node[0];
measure node[14] -> meas[0];
measure node[2] -> meas[1];
measure node[1] -> meas[2];
measure node[0] -> meas[3];
