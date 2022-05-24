OPENQASM 2.0;
include "qelib1.inc";

qreg node[13];
creg meas[10];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[7];
x node[8];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[7];
sx node[10];
sx node[12];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[7];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[7];
sx node[10];
sx node[12];
rz(0.36613976630153944*pi) node[0];
rz(0.37662413000870665*pi) node[1];
rz(0.38497327099935286*pi) node[2];
rz(0.39182654651086735*pi) node[3];
rz(0.35241637668553194*pi) node[4];
rz(0.3975836264363418*pi) node[5];
rz(0.33333333333333337*pi) node[7];
rz(0.3040867245816834*pi) node[10];
rz(0.25*pi) node[12];
cx node[8],node[5];
rz(1.6024163735636583*pi) node[5];
sx node[5];
rz(0.5*pi) node[5];
sx node[5];
rz(1.5*pi) node[5];
cx node[5],node[3];
rz(1.6081734534891328*pi) node[3];
cx node[5],node[8];
sx node[3];
rz(0.5*pi) node[3];
sx node[3];
rz(1.5*pi) node[3];
cx node[3],node[2];
rz(1.615026729000647*pi) node[2];
cx node[3],node[5];
sx node[2];
rz(0.5*pi) node[2];
sx node[2];
rz(1.5*pi) node[2];
cx node[2],node[1];
rz(1.6233758699912932*pi) node[1];
cx node[2],node[3];
sx node[1];
rz(0.5*pi) node[1];
sx node[1];
rz(1.5*pi) node[1];
cx node[1],node[0];
rz(1.6338602336984605*pi) node[0];
cx node[1],node[2];
sx node[0];
rz(0.5*pi) node[0];
sx node[0];
rz(1.5*pi) node[0];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[1],node[4];
cx node[1],node[0];
rz(1.647583623314468*pi) node[4];
sx node[4];
rz(0.5*pi) node[4];
sx node[4];
rz(1.5*pi) node[4];
cx node[4],node[7];
cx node[4],node[1];
rz(1.6666666666666665*pi) node[7];
sx node[7];
rz(0.5*pi) node[7];
sx node[7];
rz(1.5*pi) node[7];
cx node[7],node[10];
cx node[7],node[4];
rz(1.6959132754183164*pi) node[10];
sx node[10];
rz(0.5*pi) node[10];
sx node[10];
rz(1.5*pi) node[10];
cx node[10],node[12];
cx node[10],node[7];
rz(1.75*pi) node[12];
sx node[12];
rz(0.5*pi) node[12];
sx node[12];
rz(1.5*pi) node[12];
cx node[12],node[10];
barrier node[12],node[10],node[7],node[4],node[1],node[0],node[2],node[3],node[5],node[8];
measure node[12] -> meas[0];
measure node[10] -> meas[1];
measure node[7] -> meas[2];
measure node[4] -> meas[3];
measure node[1] -> meas[4];
measure node[0] -> meas[5];
measure node[2] -> meas[6];
measure node[3] -> meas[7];
measure node[5] -> meas[8];
measure node[8] -> meas[9];
