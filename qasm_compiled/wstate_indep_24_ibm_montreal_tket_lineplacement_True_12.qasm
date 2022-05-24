OPENQASM 2.0;
include "qelib1.inc";

qreg node[26];
creg meas[24];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
x node[15];
rz(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[7];
sx node[8];
sx node[9];
sx node[10];
sx node[11];
sx node[12];
sx node[13];
sx node[14];
sx node[16];
sx node[18];
sx node[19];
sx node[20];
sx node[21];
sx node[22];
sx node[23];
sx node[24];
sx node[25];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[7];
sx node[8];
sx node[9];
sx node[10];
sx node[11];
sx node[12];
sx node[13];
sx node[14];
sx node[16];
sx node[18];
sx node[19];
sx node[20];
sx node[21];
sx node[22];
sx node[23];
sx node[24];
sx node[25];
rz(0.39182654651086735*pi) node[0];
rz(0.3975836264363418*pi) node[1];
rz(0.38497327099935286*pi) node[2];
rz(0.37662413000870665*pi) node[3];
rz(0.4025088989672406*pi) node[4];
rz(0.36613976630153944*pi) node[5];
rz(0.40678526496416534*pi) node[7];
rz(0.35241637668553194*pi) node[8];
rz(0.3040867245816834*pi) node[9];
rz(0.41054380443824645*pi) node[10];
rz(0.33333333333333337*pi) node[11];
rz(0.41388134725686065*pi) node[12];
rz(0.41687100920086506*pi) node[13];
rz(0.41956938576802205*pi) node[14];
rz(0.4220208811874552*pi) node[16];
rz(0.43456537830899256*pi) node[18];
rz(0.4242609870114735*pi) node[19];
rz(0.25*pi) node[20];
rz(0.43313718551166325*pi) node[21];
rz(0.4263184784537881*pi) node[22];
rz(0.4316111760863093*pi) node[23];
rz(0.42997563622911983*pi) node[24];
rz(0.4282168467839995*pi) node[25];
cx node[15],node[18];
rz(1.5654346216910078*pi) node[18];
sx node[18];
rz(0.5*pi) node[18];
sx node[18];
rz(1.5*pi) node[18];
cx node[18],node[21];
cx node[18],node[15];
rz(1.566862814488337*pi) node[21];
sx node[21];
rz(0.5*pi) node[21];
sx node[21];
rz(1.5*pi) node[21];
cx node[21],node[23];
cx node[21],node[18];
rz(1.5683888239136907*pi) node[23];
sx node[23];
rz(0.5*pi) node[23];
sx node[23];
rz(1.5*pi) node[23];
cx node[23],node[24];
cx node[23],node[21];
rz(1.5700243637708802*pi) node[24];
sx node[24];
rz(0.5*pi) node[24];
sx node[24];
rz(1.5*pi) node[24];
cx node[24],node[25];
cx node[24],node[23];
rz(1.571783153216*pi) node[25];
sx node[25];
rz(0.5*pi) node[25];
sx node[25];
rz(1.5*pi) node[25];
cx node[25],node[22];
rz(1.5736815215462117*pi) node[22];
cx node[25],node[24];
sx node[22];
rz(0.5*pi) node[22];
sx node[22];
rz(1.5*pi) node[22];
cx node[22],node[19];
rz(1.5757390129885265*pi) node[19];
cx node[22],node[25];
sx node[19];
rz(0.5*pi) node[19];
sx node[19];
rz(1.5*pi) node[19];
cx node[19],node[16];
rz(1.577979118812545*pi) node[16];
cx node[19],node[22];
sx node[16];
rz(0.5*pi) node[16];
sx node[16];
rz(1.5*pi) node[16];
cx node[16],node[14];
rz(1.580430614231978*pi) node[14];
cx node[16],node[19];
sx node[14];
cx node[20],node[19];
rz(0.5*pi) node[14];
cx node[19],node[20];
sx node[14];
cx node[20],node[19];
rz(1.5*pi) node[14];
cx node[14],node[13];
rz(1.583128990799135*pi) node[13];
cx node[14],node[16];
sx node[13];
cx node[19],node[16];
rz(0.5*pi) node[13];
cx node[16],node[19];
sx node[13];
cx node[19],node[16];
rz(1.5*pi) node[13];
cx node[13],node[12];
rz(1.5861186527431395*pi) node[12];
cx node[13],node[14];
sx node[12];
cx node[16],node[14];
rz(0.5*pi) node[12];
cx node[14],node[16];
sx node[12];
cx node[16],node[14];
rz(1.5*pi) node[12];
cx node[12],node[10];
rz(1.5894561955617534*pi) node[10];
cx node[12],node[13];
sx node[10];
rz(0.5*pi) node[10];
sx node[10];
rz(1.5*pi) node[10];
cx node[10],node[7];
rz(1.5932147350358346*pi) node[7];
cx node[10],node[12];
sx node[7];
rz(0.5*pi) node[7];
sx node[7];
rz(1.5*pi) node[7];
cx node[7],node[4];
rz(1.5974911010327593*pi) node[4];
cx node[7],node[10];
sx node[4];
rz(0.5*pi) node[4];
sx node[4];
rz(1.5*pi) node[4];
cx node[4],node[1];
rz(1.6024163735636583*pi) node[1];
cx node[4],node[7];
sx node[1];
rz(0.5*pi) node[1];
sx node[1];
rz(1.5*pi) node[1];
cx node[1],node[0];
rz(1.6081734534891328*pi) node[0];
cx node[1],node[4];
sx node[0];
rz(0.5*pi) node[0];
sx node[0];
rz(1.5*pi) node[0];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[1],node[2];
cx node[1],node[0];
rz(1.615026729000647*pi) node[2];
sx node[2];
rz(0.5*pi) node[2];
sx node[2];
rz(1.5*pi) node[2];
cx node[2],node[3];
cx node[2],node[1];
rz(1.6233758699912932*pi) node[3];
sx node[3];
rz(0.5*pi) node[3];
sx node[3];
rz(1.5*pi) node[3];
cx node[3],node[5];
cx node[3],node[2];
rz(1.6338602336984605*pi) node[5];
sx node[5];
rz(0.5*pi) node[5];
sx node[5];
rz(1.5*pi) node[5];
cx node[5],node[8];
cx node[5],node[3];
rz(1.647583623314468*pi) node[8];
sx node[8];
rz(0.5*pi) node[8];
sx node[8];
rz(1.5*pi) node[8];
cx node[8],node[11];
cx node[8],node[5];
rz(1.6666666666666665*pi) node[11];
sx node[11];
rz(0.5*pi) node[11];
sx node[11];
rz(1.5*pi) node[11];
cx node[11],node[8];
cx node[8],node[11];
cx node[11],node[8];
cx node[8],node[9];
cx node[8],node[11];
rz(1.6959132754183164*pi) node[9];
sx node[9];
cx node[14],node[11];
rz(0.5*pi) node[9];
cx node[11],node[14];
sx node[9];
cx node[14],node[11];
rz(1.5*pi) node[9];
cx node[9],node[8];
cx node[8],node[9];
cx node[9],node[8];
cx node[8],node[11];
cx node[8],node[9];
rz(1.75*pi) node[11];
sx node[11];
rz(0.5*pi) node[11];
sx node[11];
rz(1.5*pi) node[11];
cx node[11],node[8];
barrier node[11],node[8],node[9],node[14],node[5],node[3],node[2],node[1],node[0],node[4],node[7],node[10],node[12],node[13],node[16],node[19],node[20],node[22],node[25],node[24],node[23],node[21],node[18],node[15];
measure node[11] -> meas[0];
measure node[8] -> meas[1];
measure node[9] -> meas[2];
measure node[14] -> meas[3];
measure node[5] -> meas[4];
measure node[3] -> meas[5];
measure node[2] -> meas[6];
measure node[1] -> meas[7];
measure node[0] -> meas[8];
measure node[4] -> meas[9];
measure node[7] -> meas[10];
measure node[10] -> meas[11];
measure node[12] -> meas[12];
measure node[13] -> meas[13];
measure node[16] -> meas[14];
measure node[19] -> meas[15];
measure node[20] -> meas[16];
measure node[22] -> meas[17];
measure node[25] -> meas[18];
measure node[24] -> meas[19];
measure node[23] -> meas[20];
measure node[21] -> meas[21];
measure node[18] -> meas[22];
measure node[15] -> meas[23];
