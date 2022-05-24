OPENQASM 2.0;
include "qelib1.inc";

qreg node[27];
creg c[10];
x node[11];
x node[13];
x node[14];
sx node[16];
sx node[19];
sx node[20];
sx node[22];
sx node[24];
rz(0.5712890647268734*pi) node[25];
sx node[26];
sx node[13];
sx node[14];
rz(3.5*pi) node[16];
rz(3.5*pi) node[19];
rz(3.5*pi) node[20];
rz(3.5*pi) node[22];
rz(3.5*pi) node[24];
rz(3.5*pi) node[26];
sx node[16];
sx node[19];
sx node[20];
sx node[22];
sx node[24];
sx node[26];
rz(1.0*pi) node[16];
rz(1.0*pi) node[19];
rz(1.0*pi) node[20];
rz(1.0*pi) node[22];
rz(1.0*pi) node[24];
rz(1.0*pi) node[26];
cx node[25],node[24];
rz(0.9287109433048286*pi) node[24];
x node[24];
rz(0.5*pi) node[24];
cx node[25],node[24];
cx node[25],node[22];
rz(3.572265619195171*pi) node[24];
rz(0.3574218750199285*pi) node[22];
x node[22];
rz(0.5*pi) node[22];
cx node[25],node[22];
rz(2.144531249980071*pi) node[22];
cx node[25],node[26];
rz(0.21484374367365922*pi) node[26];
x node[26];
rz(0.5*pi) node[26];
cx node[25],node[26];
cx node[25],node[22];
rz(2.2890625063263403*pi) node[26];
cx node[22],node[25];
cx node[25],node[22];
cx node[22],node[19];
cx node[26],node[25];
rz(0.9296875021201463*pi) node[19];
cx node[25],node[26];
x node[19];
cx node[26],node[25];
rz(0.5*pi) node[19];
cx node[22],node[19];
rz(1.5781249978798533*pi) node[19];
cx node[22],node[19];
cx node[19],node[22];
cx node[22],node[19];
cx node[19],node[16];
rz(0.35937500060831074*pi) node[16];
x node[16];
rz(0.5*pi) node[16];
cx node[19],node[16];
rz(2.1406249993916893*pi) node[16];
cx node[19],node[20];
rz(0.21874999999999978*pi) node[20];
x node[20];
rz(0.5*pi) node[20];
cx node[19],node[20];
rz(0.5*pi) node[19];
rz(3.78125*pi) node[20];
sx node[19];
rz(3.5*pi) node[19];
sx node[19];
rz(1.0*pi) node[19];
cx node[19],node[16];
cx node[16],node[19];
cx node[19],node[16];
cx node[16],node[14];
cx node[20],node[19];
rz(1.5*pi) node[16];
rz(0.5312499999999996*pi) node[20];
sx node[16];
cx node[20],node[19];
rz(3.4375*pi) node[16];
rz(0.015624999999999556*pi) node[19];
sx node[16];
cx node[20],node[19];
rz(2.5*pi) node[16];
cx node[19],node[20];
cx node[16],node[14];
cx node[20],node[19];
sx node[14];
sx node[16];
rz(2.5*pi) node[14];
sx node[14];
rz(2.124999999999999*pi) node[14];
cx node[16],node[14];
cx node[14],node[16];
cx node[16],node[14];
cx node[14],node[11];
rz(0.5*pi) node[14];
sx node[14];
rz(2.875*pi) node[14];
sx node[14];
rz(2.5*pi) node[14];
cx node[14],node[11];
sx node[11];
sx node[14];
rz(2.5*pi) node[11];
cx node[14],node[13];
sx node[11];
rz(2.5*pi) node[14];
rz(1.25*pi) node[11];
sx node[14];
cx node[8],node[11];
rz(1.75*pi) node[14];
cx node[11],node[8];
sx node[14];
cx node[8],node[11];
rz(1.0*pi) node[14];
cx node[14],node[13];
sx node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[13];
sx node[14];
sx node[13];
rz(3.5*pi) node[14];
sx node[14];
rz(1.0*pi) node[14];
cx node[14],node[11];
cx node[13],node[14];
cx node[14],node[13];
cx node[13],node[14];
cx node[14],node[11];
rz(0.25*pi) node[11];
cx node[14],node[11];
rz(3.75*pi) node[11];
rz(0.5*pi) node[14];
cx node[8],node[11];
sx node[14];
rz(0.125*pi) node[11];
rz(3.5*pi) node[14];
cx node[8],node[11];
sx node[14];
rz(3.875*pi) node[11];
rz(1.0*pi) node[14];
cx node[14],node[11];
cx node[11],node[14];
cx node[14],node[11];
cx node[8],node[11];
cx node[16],node[14];
rz(0.25*pi) node[11];
rz(0.0625*pi) node[14];
cx node[8],node[11];
cx node[16],node[14];
rz(0.5*pi) node[8];
rz(3.75*pi) node[11];
rz(3.9375*pi) node[14];
sx node[8];
cx node[16],node[14];
rz(3.5*pi) node[8];
cx node[14],node[16];
sx node[8];
cx node[16],node[14];
rz(1.0*pi) node[8];
cx node[14],node[11];
cx node[19],node[16];
rz(0.125*pi) node[11];
rz(0.03125*pi) node[16];
cx node[14],node[11];
cx node[19],node[16];
rz(3.875*pi) node[11];
rz(3.96875*pi) node[16];
cx node[14],node[11];
cx node[19],node[16];
cx node[11],node[14];
cx node[16],node[19];
cx node[14],node[11];
cx node[19],node[16];
cx node[11],node[8];
cx node[16],node[14];
cx node[20],node[19];
rz(0.25*pi) node[8];
rz(0.0625*pi) node[14];
rz(0.015625*pi) node[19];
cx node[11],node[8];
cx node[16],node[14];
cx node[20],node[19];
rz(3.75*pi) node[8];
rz(0.5*pi) node[11];
rz(3.9375*pi) node[14];
rz(3.984375*pi) node[19];
sx node[11];
cx node[16],node[14];
cx node[22],node[19];
rz(3.5*pi) node[11];
cx node[14],node[16];
rz(0.0078125*pi) node[19];
sx node[11];
cx node[16],node[14];
cx node[22],node[19];
rz(1.0*pi) node[11];
rz(3.9921875*pi) node[19];
cx node[8],node[11];
cx node[19],node[22];
cx node[11],node[8];
cx node[22],node[19];
cx node[8],node[11];
cx node[19],node[22];
cx node[14],node[11];
cx node[20],node[19];
cx node[25],node[22];
rz(0.125*pi) node[11];
cx node[19],node[20];
rz(0.00390625*pi) node[22];
cx node[14],node[11];
cx node[20],node[19];
cx node[25],node[22];
rz(3.875*pi) node[11];
cx node[19],node[16];
rz(3.99609375*pi) node[22];
cx node[14],node[11];
rz(0.03125*pi) node[16];
cx node[25],node[22];
cx node[11],node[14];
cx node[19],node[16];
cx node[22],node[25];
cx node[14],node[11];
rz(3.96875*pi) node[16];
cx node[25],node[22];
cx node[11],node[8];
cx node[19],node[16];
cx node[26],node[25];
rz(0.25*pi) node[8];
cx node[16],node[19];
rz(0.001953125*pi) node[25];
cx node[11],node[8];
cx node[19],node[16];
cx node[26],node[25];
rz(3.75*pi) node[8];
rz(0.5*pi) node[11];
cx node[16],node[14];
cx node[20],node[19];
rz(3.998046875*pi) node[25];
sx node[11];
rz(0.0625*pi) node[14];
rz(0.015625*pi) node[19];
cx node[24],node[25];
rz(3.5*pi) node[11];
cx node[16],node[14];
cx node[20],node[19];
rz(0.0009765625*pi) node[25];
sx node[11];
rz(3.9375*pi) node[14];
rz(3.984375*pi) node[19];
cx node[24],node[25];
rz(1.0*pi) node[11];
cx node[16],node[14];
cx node[22],node[19];
rz(3.9990234375*pi) node[25];
cx node[8],node[11];
cx node[14],node[16];
rz(0.0078125*pi) node[19];
cx node[26],node[25];
cx node[11],node[8];
cx node[16],node[14];
cx node[22],node[19];
cx node[25],node[26];
cx node[8],node[11];
rz(3.9921875*pi) node[19];
cx node[26],node[25];
cx node[14],node[11];
cx node[19],node[22];
rz(0.125*pi) node[11];
cx node[22],node[19];
cx node[14],node[11];
cx node[19],node[22];
rz(3.875*pi) node[11];
cx node[20],node[19];
cx node[25],node[22];
cx node[14],node[11];
cx node[19],node[20];
rz(0.00390625*pi) node[22];
cx node[11],node[14];
cx node[20],node[19];
cx node[25],node[22];
cx node[14],node[11];
cx node[19],node[16];
rz(3.99609375*pi) node[22];
cx node[11],node[8];
rz(0.03125*pi) node[16];
cx node[25],node[22];
rz(0.25*pi) node[8];
cx node[19],node[16];
cx node[22],node[25];
cx node[11],node[8];
rz(3.96875*pi) node[16];
cx node[25],node[22];
rz(3.75*pi) node[8];
rz(0.5*pi) node[11];
cx node[19],node[16];
cx node[24],node[25];
sx node[11];
cx node[16],node[19];
rz(0.001953125*pi) node[25];
rz(3.5*pi) node[11];
cx node[19],node[16];
cx node[24],node[25];
sx node[11];
cx node[16],node[14];
cx node[20],node[19];
rz(3.998046875*pi) node[25];
rz(1.0*pi) node[11];
rz(0.0625*pi) node[14];
rz(0.015625*pi) node[19];
cx node[24],node[25];
cx node[8],node[11];
cx node[16],node[14];
cx node[20],node[19];
cx node[25],node[24];
cx node[11],node[8];
rz(3.9375*pi) node[14];
rz(3.984375*pi) node[19];
cx node[24],node[25];
cx node[8],node[11];
cx node[16],node[14];
cx node[22],node[19];
cx node[14],node[16];
rz(0.0078125*pi) node[19];
cx node[16],node[14];
cx node[22],node[19];
cx node[14],node[11];
rz(3.9921875*pi) node[19];
rz(0.125*pi) node[11];
cx node[19],node[22];
cx node[14],node[11];
cx node[22],node[19];
rz(3.875*pi) node[11];
cx node[19],node[22];
cx node[14],node[11];
cx node[20],node[19];
cx node[25],node[22];
cx node[11],node[14];
cx node[19],node[20];
rz(0.00390625*pi) node[22];
cx node[14],node[11];
cx node[20],node[19];
cx node[25],node[22];
cx node[11],node[8];
cx node[19],node[16];
rz(3.99609375*pi) node[22];
rz(0.25*pi) node[8];
rz(0.03125*pi) node[16];
cx node[25],node[22];
cx node[11],node[8];
cx node[19],node[16];
cx node[22],node[25];
rz(3.75*pi) node[8];
rz(0.5*pi) node[11];
rz(3.96875*pi) node[16];
cx node[25],node[22];
sx node[11];
cx node[19],node[16];
rz(3.5*pi) node[11];
cx node[16],node[19];
sx node[11];
cx node[19],node[16];
rz(1.0*pi) node[11];
cx node[16],node[14];
cx node[20],node[19];
cx node[8],node[11];
rz(0.0625*pi) node[14];
rz(0.015625*pi) node[19];
cx node[11],node[8];
cx node[16],node[14];
cx node[20],node[19];
cx node[8],node[11];
rz(3.9375*pi) node[14];
rz(3.984375*pi) node[19];
cx node[16],node[14];
cx node[22],node[19];
cx node[14],node[16];
rz(0.0078125*pi) node[19];
cx node[16],node[14];
cx node[22],node[19];
cx node[14],node[11];
rz(3.9921875*pi) node[19];
rz(0.125*pi) node[11];
cx node[20],node[19];
cx node[14],node[11];
cx node[19],node[20];
rz(3.875*pi) node[11];
cx node[20],node[19];
cx node[14],node[11];
cx node[19],node[16];
cx node[11],node[14];
rz(0.03125*pi) node[16];
cx node[14],node[11];
cx node[19],node[16];
cx node[11],node[8];
rz(3.96875*pi) node[16];
rz(0.25*pi) node[8];
cx node[19],node[16];
cx node[11],node[8];
cx node[16],node[19];
rz(3.75*pi) node[8];
rz(0.5*pi) node[11];
cx node[19],node[16];
sx node[11];
cx node[16],node[14];
cx node[22],node[19];
rz(3.5*pi) node[11];
rz(0.0625*pi) node[14];
rz(0.015625*pi) node[19];
sx node[11];
cx node[16],node[14];
cx node[22],node[19];
rz(1.0*pi) node[11];
rz(3.9375*pi) node[14];
rz(3.984375*pi) node[19];
cx node[8],node[11];
cx node[16],node[14];
cx node[22],node[19];
cx node[11],node[8];
cx node[14],node[16];
cx node[19],node[22];
cx node[8],node[11];
cx node[16],node[14];
cx node[22],node[19];
cx node[14],node[11];
cx node[19],node[16];
rz(0.125*pi) node[11];
rz(0.03125*pi) node[16];
cx node[14],node[11];
cx node[19],node[16];
rz(3.875*pi) node[11];
rz(3.96875*pi) node[16];
cx node[14],node[11];
cx node[19],node[16];
cx node[11],node[14];
cx node[16],node[19];
cx node[14],node[11];
cx node[19],node[16];
cx node[11],node[8];
cx node[16],node[14];
rz(0.25*pi) node[8];
rz(0.0625*pi) node[14];
cx node[11],node[8];
cx node[16],node[14];
rz(3.75*pi) node[8];
rz(0.5*pi) node[11];
rz(3.9375*pi) node[14];
sx node[11];
cx node[16],node[14];
rz(3.5*pi) node[11];
cx node[14],node[16];
sx node[11];
cx node[16],node[14];
rz(1.0*pi) node[11];
cx node[14],node[11];
cx node[11],node[14];
cx node[14],node[11];
cx node[11],node[8];
rz(0.125*pi) node[8];
cx node[11],node[8];
rz(3.875*pi) node[8];
cx node[11],node[14];
rz(0.25*pi) node[14];
cx node[11],node[14];
rz(0.5*pi) node[11];
rz(3.75*pi) node[14];
sx node[11];
rz(3.5*pi) node[11];
sx node[11];
rz(1.0*pi) node[11];
barrier node[26],node[24],node[25],node[20],node[22],node[19],node[16],node[8],node[14],node[11],node[13];
measure node[26] -> c[0];
measure node[24] -> c[1];
measure node[25] -> c[2];
measure node[20] -> c[3];
measure node[22] -> c[4];
measure node[19] -> c[5];
measure node[16] -> c[6];
measure node[8] -> c[7];
measure node[14] -> c[8];
measure node[11] -> c[9];
