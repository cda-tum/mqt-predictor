OPENQASM 2.0;
include "qelib1.inc";

qreg node[23];
creg meas[12];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
x node[5];
rz(0.5*pi) node[14];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[14];
sx node[18];
sx node[19];
sx node[20];
sx node[21];
sx node[22];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[14];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[14];
sx node[18];
sx node[19];
sx node[20];
sx node[21];
sx node[22];
rz(0.38497327099935286*pi) node[0];
rz(0.39182654651086735*pi) node[1];
rz(0.3975836264363418*pi) node[2];
rz(0.4025088989672406*pi) node[3];
rz(0.40678526496416534*pi) node[4];
rz(0.37662413000870665*pi) node[14];
rz(0.36613976630153944*pi) node[18];
rz(0.35241637668553194*pi) node[19];
rz(0.33333333333333337*pi) node[20];
rz(0.3040867245816834*pi) node[21];
rz(0.25*pi) node[22];
cx node[5],node[4];
rz(1.5932147350358346*pi) node[4];
sx node[4];
rz(0.5*pi) node[4];
sx node[4];
rz(1.5*pi) node[4];
cx node[4],node[3];
rz(1.5974911010327593*pi) node[3];
cx node[4],node[5];
sx node[3];
rz(0.5*pi) node[3];
sx node[3];
rz(1.5*pi) node[3];
cx node[3],node[2];
rz(1.6024163735636583*pi) node[2];
cx node[3],node[4];
sx node[2];
rz(0.5*pi) node[2];
sx node[2];
rz(1.5*pi) node[2];
cx node[2],node[1];
rz(1.6081734534891328*pi) node[1];
cx node[2],node[3];
sx node[1];
rz(0.5*pi) node[1];
sx node[1];
rz(1.5*pi) node[1];
cx node[1],node[0];
rz(1.615026729000647*pi) node[0];
cx node[1],node[2];
sx node[0];
rz(0.5*pi) node[0];
sx node[0];
rz(1.5*pi) node[0];
cx node[0],node[14];
cx node[0],node[1];
rz(1.6233758699912932*pi) node[14];
sx node[14];
rz(0.5*pi) node[14];
sx node[14];
rz(1.5*pi) node[14];
cx node[14],node[18];
cx node[14],node[0];
rz(1.6338602336984605*pi) node[18];
sx node[18];
rz(0.5*pi) node[18];
sx node[18];
rz(1.5*pi) node[18];
cx node[18],node[19];
cx node[18],node[14];
rz(1.647583623314468*pi) node[19];
sx node[19];
rz(0.5*pi) node[19];
sx node[19];
rz(1.5*pi) node[19];
cx node[19],node[20];
cx node[19],node[18];
rz(1.6666666666666665*pi) node[20];
sx node[20];
rz(0.5*pi) node[20];
sx node[20];
rz(1.5*pi) node[20];
cx node[20],node[21];
cx node[20],node[19];
rz(1.6959132754183164*pi) node[21];
sx node[21];
rz(0.5*pi) node[21];
sx node[21];
rz(1.5*pi) node[21];
cx node[21],node[22];
cx node[21],node[20];
rz(1.75*pi) node[22];
sx node[22];
rz(0.5*pi) node[22];
sx node[22];
rz(1.5*pi) node[22];
cx node[22],node[21];
barrier node[22],node[21],node[20],node[19],node[18],node[14],node[0],node[1],node[2],node[3],node[4],node[5];
measure node[22] -> meas[0];
measure node[21] -> meas[1];
measure node[20] -> meas[2];
measure node[19] -> meas[3];
measure node[18] -> meas[4];
measure node[14] -> meas[5];
measure node[0] -> meas[6];
measure node[1] -> meas[7];
measure node[2] -> meas[8];
measure node[3] -> meas[9];
measure node[4] -> meas[10];
measure node[5] -> meas[11];
