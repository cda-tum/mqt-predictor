OPENQASM 2.0;
include "qelib1.inc";

qreg node[52];
creg meas[46];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
x node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[28];
rz(0.5*pi) node[29];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[32];
rz(0.5*pi) node[33];
rz(0.5*pi) node[34];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[42];
rz(0.5*pi) node[43];
rz(0.5*pi) node[44];
rz(0.5*pi) node[45];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[50];
rz(0.5*pi) node[51];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[6];
sx node[7];
sx node[8];
sx node[14];
sx node[15];
sx node[16];
sx node[17];
sx node[18];
sx node[19];
sx node[20];
sx node[21];
sx node[22];
sx node[23];
sx node[24];
sx node[25];
sx node[26];
sx node[27];
sx node[28];
sx node[29];
sx node[30];
sx node[31];
sx node[32];
sx node[33];
sx node[34];
sx node[35];
sx node[36];
sx node[38];
sx node[39];
sx node[40];
sx node[41];
sx node[42];
sx node[43];
sx node[44];
sx node[45];
sx node[46];
sx node[47];
sx node[48];
sx node[49];
sx node[50];
sx node[51];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[28];
rz(0.5*pi) node[29];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[32];
rz(0.5*pi) node[33];
rz(0.5*pi) node[34];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[42];
rz(0.5*pi) node[43];
rz(0.5*pi) node[44];
rz(0.5*pi) node[45];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[50];
rz(0.5*pi) node[51];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[6];
sx node[7];
sx node[8];
sx node[14];
sx node[15];
sx node[16];
sx node[17];
sx node[18];
sx node[19];
sx node[20];
sx node[21];
sx node[22];
sx node[23];
sx node[24];
sx node[25];
sx node[26];
sx node[27];
sx node[28];
sx node[29];
sx node[30];
sx node[31];
sx node[32];
sx node[33];
sx node[34];
sx node[35];
sx node[36];
sx node[38];
sx node[39];
sx node[40];
sx node[41];
sx node[42];
sx node[43];
sx node[44];
sx node[45];
sx node[46];
sx node[47];
sx node[48];
sx node[49];
sx node[50];
sx node[51];
rz(0.4506866917905885*pi) node[0];
rz(0.4512680529667147*pi) node[1];
rz(0.4518293287890226*pi) node[2];
rz(0.4523716333421135*pi) node[3];
rz(0.45289604887960166*pi) node[4];
rz(0.43835664003937325*pi) node[6];
rz(0.43948113337429473*pi) node[7];
rz(0.44054626191544266*pi) node[8];
rz(0.450084003852088*pi) node[14];
rz(0.44593623504918156*pi) node[15];
rz(0.4415570231280308*pi) node[16];
rz(0.4282168467839995*pi) node[17];
rz(0.44945868408068*pi) node[18];
rz(0.44880923641989934*pi) node[19];
rz(0.44813410115130337*pi) node[20];
rz(0.44743152757051863*pi) node[21];
rz(0.44669960581822754*pi) node[22];
rz(0.44513909160121146*pi) node[23];
rz(0.4443056289952281*pi) node[24];
rz(0.4434329506112663*pi) node[25];
rz(0.44251793701244235*pi) node[26];
rz(0.4371670523327271*pi) node[27];
rz(0.43590578123971224*pi) node[28];
rz(0.43313718551166325*pi) node[29];
rz(0.42997563622911983*pi) node[30];
rz(0.4263184784537881*pi) node[31];
rz(0.4242609870114735*pi) node[32];
rz(0.3040867245816834*pi) node[33];
rz(0.43456537830899256*pi) node[34];
rz(0.4316111760863093*pi) node[35];
rz(0.4220208811874552*pi) node[36];
rz(0.25*pi) node[38];
rz(0.33333333333333337*pi) node[39];
rz(0.35241637668553194*pi) node[40];
rz(0.36613976630153944*pi) node[41];
rz(0.37662413000870665*pi) node[42];
rz(0.38497327099935286*pi) node[43];
rz(0.39182654651086735*pi) node[44];
rz(0.3975836264363418*pi) node[45];
rz(0.4025088989672406*pi) node[46];
rz(0.40678526496416534*pi) node[47];
rz(0.41054380443824645*pi) node[48];
rz(0.41388134725686065*pi) node[49];
rz(0.41687100920086506*pi) node[50];
rz(0.41956938576802205*pi) node[51];
cx node[5],node[4];
rz(1.5471039511203986*pi) node[4];
sx node[4];
rz(0.5*pi) node[4];
sx node[4];
rz(1.5*pi) node[4];
cx node[4],node[3];
rz(1.5476283666578863*pi) node[3];
cx node[4],node[5];
sx node[3];
rz(0.5*pi) node[3];
sx node[3];
rz(1.5*pi) node[3];
cx node[3],node[2];
rz(1.5481706712109777*pi) node[2];
cx node[3],node[4];
sx node[2];
rz(0.5*pi) node[2];
sx node[2];
rz(1.5*pi) node[2];
cx node[2],node[1];
rz(1.5487319470332856*pi) node[1];
cx node[2],node[3];
sx node[1];
rz(0.5*pi) node[1];
sx node[1];
rz(1.5*pi) node[1];
cx node[1],node[0];
rz(1.5493133082094115*pi) node[0];
cx node[1],node[2];
sx node[0];
rz(0.5*pi) node[0];
sx node[0];
rz(1.5*pi) node[0];
cx node[0],node[14];
cx node[0],node[1];
rz(1.5499159961479119*pi) node[14];
sx node[14];
rz(0.5*pi) node[14];
sx node[14];
rz(1.5*pi) node[14];
cx node[14],node[18];
cx node[14],node[0];
rz(1.55054131591932*pi) node[18];
sx node[18];
rz(0.5*pi) node[18];
sx node[18];
rz(1.5*pi) node[18];
cx node[18],node[19];
cx node[18],node[14];
rz(1.5511907635801008*pi) node[19];
sx node[19];
rz(0.5*pi) node[19];
sx node[19];
rz(1.5*pi) node[19];
cx node[19],node[20];
cx node[19],node[18];
rz(1.5518658988486966*pi) node[20];
sx node[20];
rz(0.5*pi) node[20];
sx node[20];
rz(1.5*pi) node[20];
cx node[20],node[21];
cx node[20],node[19];
rz(1.5525684724294817*pi) node[21];
sx node[21];
rz(0.5*pi) node[21];
sx node[21];
rz(1.5*pi) node[21];
cx node[21],node[22];
cx node[21],node[20];
rz(1.5533003941817725*pi) node[22];
sx node[22];
rz(0.5*pi) node[22];
sx node[22];
rz(1.5*pi) node[22];
cx node[22],node[15];
rz(1.5540637649508184*pi) node[15];
cx node[22],node[21];
sx node[15];
rz(0.5*pi) node[15];
sx node[15];
rz(1.5*pi) node[15];
cx node[15],node[22];
cx node[22],node[15];
cx node[15],node[22];
cx node[22],node[23];
cx node[22],node[15];
rz(1.5548609083987888*pi) node[23];
sx node[23];
rz(0.5*pi) node[23];
sx node[23];
rz(1.5*pi) node[23];
cx node[23],node[24];
cx node[23],node[22];
rz(1.5556943710047721*pi) node[24];
sx node[24];
rz(0.5*pi) node[24];
sx node[24];
rz(1.5*pi) node[24];
cx node[24],node[25];
cx node[24],node[23];
rz(1.5565670493887336*pi) node[25];
sx node[25];
rz(0.5*pi) node[25];
sx node[25];
rz(1.5*pi) node[25];
cx node[25],node[26];
cx node[25],node[24];
rz(1.5574820629875576*pi) node[26];
cx node[34],node[24];
sx node[26];
cx node[24],node[34];
rz(0.5*pi) node[26];
cx node[34],node[24];
sx node[26];
rz(1.5*pi) node[26];
cx node[26],node[16];
rz(1.558442976871969*pi) node[16];
cx node[26],node[25];
sx node[16];
cx node[24],node[25];
rz(0.5*pi) node[16];
cx node[25],node[24];
sx node[16];
cx node[24],node[25];
rz(1.5*pi) node[16];
cx node[16],node[8];
rz(1.5594537380845572*pi) node[8];
cx node[16],node[26];
sx node[8];
cx node[27],node[26];
rz(0.5*pi) node[8];
cx node[26],node[27];
sx node[8];
cx node[27],node[26];
rz(1.5*pi) node[8];
cx node[28],node[27];
cx node[8],node[7];
cx node[27],node[28];
rz(1.5605188666257055*pi) node[7];
cx node[8],node[16];
cx node[28],node[27];
sx node[7];
cx node[26],node[16];
cx node[29],node[28];
rz(0.5*pi) node[7];
cx node[16],node[26];
cx node[28],node[29];
sx node[7];
cx node[26],node[16];
cx node[29],node[28];
rz(1.5*pi) node[7];
cx node[27],node[26];
cx node[30],node[29];
cx node[7],node[6];
cx node[26],node[27];
cx node[29],node[30];
rz(1.561643359960627*pi) node[6];
cx node[7],node[8];
cx node[27],node[26];
cx node[30],node[29];
sx node[6];
cx node[16],node[8];
cx node[17],node[30];
cx node[28],node[27];
rz(0.5*pi) node[6];
cx node[8],node[16];
cx node[30],node[17];
cx node[27],node[28];
sx node[6];
cx node[16],node[8];
cx node[17],node[30];
cx node[28],node[27];
rz(1.5*pi) node[6];
cx node[26],node[16];
cx node[35],node[28];
cx node[6],node[7];
cx node[16],node[26];
cx node[28],node[35];
cx node[7],node[6];
cx node[26],node[16];
cx node[35],node[28];
cx node[6],node[7];
cx node[25],node[26];
cx node[7],node[8];
cx node[26],node[25];
cx node[7],node[6];
rz(1.5628329476672729*pi) node[8];
cx node[25],node[26];
sx node[8];
rz(0.5*pi) node[8];
sx node[8];
rz(1.5*pi) node[8];
cx node[8],node[16];
cx node[8],node[7];
rz(1.5640942187602875*pi) node[16];
sx node[16];
rz(0.5*pi) node[16];
sx node[16];
rz(1.5*pi) node[16];
cx node[16],node[26];
cx node[16],node[8];
rz(1.5654346216910078*pi) node[26];
sx node[26];
rz(0.5*pi) node[26];
sx node[26];
rz(1.5*pi) node[26];
cx node[26],node[27];
cx node[26],node[16];
rz(1.566862814488337*pi) node[27];
sx node[27];
rz(0.5*pi) node[27];
sx node[27];
rz(1.5*pi) node[27];
cx node[27],node[28];
cx node[27],node[26];
rz(1.5683888239136907*pi) node[28];
sx node[28];
rz(0.5*pi) node[28];
sx node[28];
rz(1.5*pi) node[28];
cx node[28],node[29];
cx node[28],node[27];
rz(1.5700243637708802*pi) node[29];
sx node[29];
rz(0.5*pi) node[29];
sx node[29];
rz(1.5*pi) node[29];
cx node[29],node[30];
cx node[29],node[28];
rz(1.571783153216*pi) node[30];
sx node[30];
rz(0.5*pi) node[30];
sx node[30];
rz(1.5*pi) node[30];
cx node[30],node[31];
cx node[30],node[29];
rz(1.5736815215462117*pi) node[31];
sx node[31];
rz(0.5*pi) node[31];
sx node[31];
rz(1.5*pi) node[31];
cx node[31],node[32];
cx node[31],node[30];
rz(1.5757390129885265*pi) node[32];
sx node[32];
rz(0.5*pi) node[32];
sx node[32];
rz(1.5*pi) node[32];
cx node[32],node[36];
cx node[32],node[31];
rz(1.577979118812545*pi) node[36];
sx node[36];
rz(0.5*pi) node[36];
sx node[36];
rz(1.5*pi) node[36];
cx node[36],node[51];
cx node[36],node[32];
rz(1.580430614231978*pi) node[51];
sx node[51];
rz(0.5*pi) node[51];
sx node[51];
rz(1.5*pi) node[51];
cx node[51],node[50];
cx node[51],node[36];
rz(1.583128990799135*pi) node[50];
sx node[50];
rz(0.5*pi) node[50];
sx node[50];
rz(1.5*pi) node[50];
cx node[50],node[49];
rz(1.5861186527431395*pi) node[49];
cx node[50],node[51];
sx node[49];
rz(0.5*pi) node[49];
sx node[49];
rz(1.5*pi) node[49];
cx node[49],node[48];
rz(1.5894561955617534*pi) node[48];
cx node[49],node[50];
sx node[48];
rz(0.5*pi) node[48];
sx node[48];
rz(1.5*pi) node[48];
cx node[48],node[47];
rz(1.5932147350358346*pi) node[47];
cx node[48],node[49];
sx node[47];
rz(0.5*pi) node[47];
sx node[47];
rz(1.5*pi) node[47];
cx node[47],node[46];
rz(1.5974911010327593*pi) node[46];
cx node[47],node[48];
sx node[46];
rz(0.5*pi) node[46];
sx node[46];
rz(1.5*pi) node[46];
cx node[46],node[45];
rz(1.6024163735636583*pi) node[45];
cx node[46],node[47];
sx node[45];
rz(0.5*pi) node[45];
sx node[45];
rz(1.5*pi) node[45];
cx node[45],node[44];
rz(1.6081734534891328*pi) node[44];
cx node[45],node[46];
sx node[44];
rz(0.5*pi) node[44];
sx node[44];
rz(1.5*pi) node[44];
cx node[44],node[43];
rz(1.615026729000647*pi) node[43];
cx node[44],node[45];
sx node[43];
rz(0.5*pi) node[43];
sx node[43];
rz(1.5*pi) node[43];
cx node[43],node[42];
rz(1.6233758699912932*pi) node[42];
cx node[43],node[44];
sx node[42];
rz(0.5*pi) node[42];
sx node[42];
rz(1.5*pi) node[42];
cx node[42],node[41];
rz(1.6338602336984605*pi) node[41];
cx node[42],node[43];
sx node[41];
rz(0.5*pi) node[41];
sx node[41];
rz(1.5*pi) node[41];
cx node[41],node[40];
rz(1.647583623314468*pi) node[40];
cx node[41],node[42];
sx node[40];
rz(0.5*pi) node[40];
sx node[40];
rz(1.5*pi) node[40];
cx node[40],node[39];
rz(1.6666666666666665*pi) node[39];
cx node[40],node[41];
sx node[39];
rz(0.5*pi) node[39];
sx node[39];
rz(1.5*pi) node[39];
cx node[39],node[33];
rz(1.6959132754183164*pi) node[33];
cx node[39],node[40];
sx node[33];
rz(0.5*pi) node[33];
sx node[33];
rz(1.5*pi) node[33];
cx node[33],node[39];
cx node[39],node[33];
cx node[33],node[39];
cx node[39],node[38];
cx node[39],node[33];
rz(1.75*pi) node[38];
sx node[38];
rz(0.5*pi) node[38];
sx node[38];
rz(1.5*pi) node[38];
cx node[38],node[39];
barrier node[38],node[39],node[33],node[40],node[41],node[42],node[43],node[44],node[45],node[46],node[47],node[48],node[49],node[50],node[51],node[36],node[32],node[31],node[30],node[29],node[28],node[27],node[26],node[16],node[8],node[7],node[6],node[25],node[35],node[17],node[24],node[34],node[23],node[22],node[15],node[21],node[20],node[19],node[18],node[14],node[0],node[1],node[2],node[3],node[4],node[5];
measure node[38] -> meas[0];
measure node[39] -> meas[1];
measure node[33] -> meas[2];
measure node[40] -> meas[3];
measure node[41] -> meas[4];
measure node[42] -> meas[5];
measure node[43] -> meas[6];
measure node[44] -> meas[7];
measure node[45] -> meas[8];
measure node[46] -> meas[9];
measure node[47] -> meas[10];
measure node[48] -> meas[11];
measure node[49] -> meas[12];
measure node[50] -> meas[13];
measure node[51] -> meas[14];
measure node[36] -> meas[15];
measure node[32] -> meas[16];
measure node[31] -> meas[17];
measure node[30] -> meas[18];
measure node[29] -> meas[19];
measure node[28] -> meas[20];
measure node[27] -> meas[21];
measure node[26] -> meas[22];
measure node[16] -> meas[23];
measure node[8] -> meas[24];
measure node[7] -> meas[25];
measure node[6] -> meas[26];
measure node[25] -> meas[27];
measure node[35] -> meas[28];
measure node[17] -> meas[29];
measure node[24] -> meas[30];
measure node[34] -> meas[31];
measure node[23] -> meas[32];
measure node[22] -> meas[33];
measure node[15] -> meas[34];
measure node[21] -> meas[35];
measure node[20] -> meas[36];
measure node[19] -> meas[37];
measure node[18] -> meas[38];
measure node[14] -> meas[39];
measure node[0] -> meas[40];
measure node[1] -> meas[41];
measure node[2] -> meas[42];
measure node[3] -> meas[43];
measure node[4] -> meas[44];
measure node[5] -> meas[45];
