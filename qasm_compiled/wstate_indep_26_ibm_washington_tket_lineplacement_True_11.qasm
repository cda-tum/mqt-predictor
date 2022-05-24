OPENQASM 2.0;
include "qelib1.inc";

qreg node[127];
creg meas[26];
rz(0.5*pi) node[24];
rz(0.5*pi) node[34];
rz(0.5*pi) node[43];
rz(0.5*pi) node[44];
rz(0.5*pi) node[45];
rz(0.5*pi) node[54];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[73];
rz(0.5*pi) node[85];
rz(0.5*pi) node[86];
rz(0.5*pi) node[87];
rz(0.5*pi) node[93];
rz(0.5*pi) node[104];
rz(0.5*pi) node[105];
rz(0.5*pi) node[106];
x node[107];
rz(0.5*pi) node[108];
rz(0.5*pi) node[111];
rz(0.5*pi) node[112];
rz(0.5*pi) node[122];
rz(0.5*pi) node[123];
rz(0.5*pi) node[124];
rz(0.5*pi) node[125];
rz(0.5*pi) node[126];
sx node[24];
sx node[34];
sx node[43];
sx node[44];
sx node[45];
sx node[54];
sx node[64];
sx node[65];
sx node[66];
sx node[73];
sx node[85];
sx node[86];
sx node[87];
sx node[93];
sx node[104];
sx node[105];
sx node[106];
sx node[108];
sx node[111];
sx node[112];
sx node[122];
sx node[123];
sx node[124];
sx node[125];
sx node[126];
rz(0.5*pi) node[24];
rz(0.5*pi) node[34];
rz(0.5*pi) node[43];
rz(0.5*pi) node[44];
rz(0.5*pi) node[45];
rz(0.5*pi) node[54];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[73];
rz(0.5*pi) node[85];
rz(0.5*pi) node[86];
rz(0.5*pi) node[87];
rz(0.5*pi) node[93];
rz(0.5*pi) node[104];
rz(0.5*pi) node[105];
rz(0.5*pi) node[106];
rz(0.5*pi) node[108];
rz(0.5*pi) node[111];
rz(0.5*pi) node[112];
rz(0.5*pi) node[122];
rz(0.5*pi) node[123];
rz(0.5*pi) node[124];
rz(0.5*pi) node[125];
rz(0.5*pi) node[126];
sx node[24];
sx node[34];
sx node[43];
sx node[44];
sx node[45];
sx node[54];
sx node[64];
sx node[65];
sx node[66];
sx node[73];
sx node[85];
sx node[86];
sx node[87];
sx node[93];
sx node[104];
sx node[105];
sx node[106];
sx node[108];
sx node[111];
sx node[112];
sx node[122];
sx node[123];
sx node[124];
sx node[125];
sx node[126];
rz(0.25*pi) node[24];
rz(0.3040867245816834*pi) node[34];
rz(0.33333333333333337*pi) node[43];
rz(0.35241637668553194*pi) node[44];
rz(0.36613976630153944*pi) node[45];
rz(0.37662413000870665*pi) node[54];
rz(0.38497327099935286*pi) node[64];
rz(0.39182654651086735*pi) node[65];
rz(0.3975836264363418*pi) node[66];
rz(0.4025088989672406*pi) node[73];
rz(0.40678526496416534*pi) node[85];
rz(0.41054380443824645*pi) node[86];
rz(0.41388134725686065*pi) node[87];
rz(0.41687100920086506*pi) node[93];
rz(0.4242609870114735*pi) node[104];
rz(0.4220208811874552*pi) node[105];
rz(0.41956938576802205*pi) node[106];
rz(0.4371670523327271*pi) node[108];
rz(0.4263184784537881*pi) node[111];
rz(0.43590578123971224*pi) node[112];
rz(0.4282168467839995*pi) node[122];
rz(0.42997563622911983*pi) node[123];
rz(0.4316111760863093*pi) node[124];
rz(0.43313718551166325*pi) node[125];
rz(0.43456537830899256*pi) node[126];
cx node[107],node[108];
rz(1.5628329476672729*pi) node[108];
sx node[108];
rz(0.5*pi) node[108];
sx node[108];
rz(1.5*pi) node[108];
cx node[108],node[112];
cx node[108],node[107];
rz(1.5640942187602875*pi) node[112];
sx node[112];
rz(0.5*pi) node[112];
sx node[112];
rz(1.5*pi) node[112];
cx node[112],node[126];
cx node[112],node[108];
rz(1.5654346216910078*pi) node[126];
sx node[126];
rz(0.5*pi) node[126];
sx node[126];
rz(1.5*pi) node[126];
cx node[126],node[125];
cx node[126],node[112];
rz(1.566862814488337*pi) node[125];
sx node[125];
rz(0.5*pi) node[125];
sx node[125];
rz(1.5*pi) node[125];
cx node[125],node[124];
rz(1.5683888239136907*pi) node[124];
cx node[125],node[126];
sx node[124];
rz(0.5*pi) node[124];
sx node[124];
rz(1.5*pi) node[124];
cx node[124],node[123];
rz(1.5700243637708802*pi) node[123];
cx node[124],node[125];
sx node[123];
rz(0.5*pi) node[123];
sx node[123];
rz(1.5*pi) node[123];
cx node[123],node[122];
rz(1.571783153216*pi) node[122];
cx node[123],node[124];
sx node[122];
rz(0.5*pi) node[122];
sx node[122];
rz(1.5*pi) node[122];
cx node[122],node[111];
rz(1.5736815215462117*pi) node[111];
cx node[122],node[123];
sx node[111];
rz(0.5*pi) node[111];
sx node[111];
rz(1.5*pi) node[111];
cx node[111],node[104];
rz(1.5757390129885265*pi) node[104];
cx node[111],node[122];
sx node[104];
rz(0.5*pi) node[104];
sx node[104];
rz(1.5*pi) node[104];
cx node[104],node[105];
cx node[104],node[111];
rz(1.577979118812545*pi) node[105];
sx node[105];
rz(0.5*pi) node[105];
sx node[105];
rz(1.5*pi) node[105];
cx node[105],node[106];
cx node[105],node[104];
rz(1.580430614231978*pi) node[106];
sx node[106];
rz(0.5*pi) node[106];
sx node[106];
rz(1.5*pi) node[106];
cx node[106],node[93];
rz(1.583128990799135*pi) node[93];
cx node[106],node[105];
sx node[93];
rz(0.5*pi) node[93];
sx node[93];
rz(1.5*pi) node[93];
cx node[93],node[87];
rz(1.5861186527431395*pi) node[87];
cx node[93],node[106];
sx node[87];
rz(0.5*pi) node[87];
sx node[87];
rz(1.5*pi) node[87];
cx node[87],node[86];
rz(1.5894561955617534*pi) node[86];
cx node[87],node[93];
sx node[86];
rz(0.5*pi) node[86];
sx node[86];
rz(1.5*pi) node[86];
cx node[86],node[85];
rz(1.5932147350358346*pi) node[85];
cx node[86],node[87];
sx node[85];
rz(0.5*pi) node[85];
sx node[85];
rz(1.5*pi) node[85];
cx node[85],node[73];
rz(1.5974911010327593*pi) node[73];
cx node[85],node[86];
sx node[73];
rz(0.5*pi) node[73];
sx node[73];
rz(1.5*pi) node[73];
cx node[73],node[66];
rz(1.6024163735636583*pi) node[66];
cx node[73],node[85];
sx node[66];
rz(0.5*pi) node[66];
sx node[66];
rz(1.5*pi) node[66];
cx node[66],node[65];
rz(1.6081734534891328*pi) node[65];
cx node[66],node[73];
sx node[65];
rz(0.5*pi) node[65];
sx node[65];
rz(1.5*pi) node[65];
cx node[65],node[64];
rz(1.615026729000647*pi) node[64];
cx node[65],node[66];
sx node[64];
rz(0.5*pi) node[64];
sx node[64];
rz(1.5*pi) node[64];
cx node[64],node[54];
rz(1.6233758699912932*pi) node[54];
cx node[64],node[65];
sx node[54];
rz(0.5*pi) node[54];
sx node[54];
rz(1.5*pi) node[54];
cx node[54],node[45];
rz(1.6338602336984605*pi) node[45];
cx node[54],node[64];
sx node[45];
rz(0.5*pi) node[45];
sx node[45];
rz(1.5*pi) node[45];
cx node[45],node[44];
rz(1.647583623314468*pi) node[44];
cx node[45],node[54];
sx node[44];
rz(0.5*pi) node[44];
sx node[44];
rz(1.5*pi) node[44];
cx node[44],node[43];
rz(1.6666666666666665*pi) node[43];
cx node[44],node[45];
sx node[43];
rz(0.5*pi) node[43];
sx node[43];
rz(1.5*pi) node[43];
cx node[43],node[34];
rz(1.6959132754183164*pi) node[34];
cx node[43],node[44];
sx node[34];
rz(0.5*pi) node[34];
sx node[34];
rz(1.5*pi) node[34];
cx node[34],node[24];
rz(1.75*pi) node[24];
cx node[34],node[43];
sx node[24];
rz(0.5*pi) node[24];
sx node[24];
rz(1.5*pi) node[24];
cx node[24],node[34];
barrier node[24],node[34],node[43],node[44],node[45],node[54],node[64],node[65],node[66],node[73],node[85],node[86],node[87],node[93],node[106],node[105],node[104],node[111],node[122],node[123],node[124],node[125],node[126],node[112],node[108],node[107];
measure node[24] -> meas[0];
measure node[34] -> meas[1];
measure node[43] -> meas[2];
measure node[44] -> meas[3];
measure node[45] -> meas[4];
measure node[54] -> meas[5];
measure node[64] -> meas[6];
measure node[65] -> meas[7];
measure node[66] -> meas[8];
measure node[73] -> meas[9];
measure node[85] -> meas[10];
measure node[86] -> meas[11];
measure node[87] -> meas[12];
measure node[93] -> meas[13];
measure node[106] -> meas[14];
measure node[105] -> meas[15];
measure node[104] -> meas[16];
measure node[111] -> meas[17];
measure node[122] -> meas[18];
measure node[123] -> meas[19];
measure node[124] -> meas[20];
measure node[125] -> meas[21];
measure node[126] -> meas[22];
measure node[112] -> meas[23];
measure node[108] -> meas[24];
measure node[107] -> meas[25];
