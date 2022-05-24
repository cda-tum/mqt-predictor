OPENQASM 2.0;
include "qelib1.inc";

qreg node[126];
creg c[19];
sx node[61];
sx node[62];
sx node[72];
sx node[80];
sx node[81];
sx node[82];
sx node[83];
sx node[84];
sx node[92];
sx node[101];
sx node[102];
sx node[103];
sx node[104];
sx node[105];
sx node[111];
sx node[121];
sx node[122];
sx node[123];
sx node[124];
sx node[125];
rz(0.5*pi) node[61];
rz(0.5*pi) node[62];
rz(0.5*pi) node[72];
rz(0.5*pi) node[80];
rz(0.5*pi) node[81];
rz(0.5*pi) node[82];
rz(0.5*pi) node[83];
rz(0.5*pi) node[84];
rz(0.5*pi) node[92];
rz(0.5*pi) node[101];
rz(0.5*pi) node[102];
rz(0.5*pi) node[103];
rz(0.5*pi) node[104];
rz(0.5*pi) node[105];
rz(0.5*pi) node[111];
rz(0.5*pi) node[121];
rz(0.5*pi) node[122];
rz(0.5*pi) node[123];
rz(3.5*pi) node[124];
rz(0.5*pi) node[125];
sx node[61];
sx node[62];
sx node[72];
sx node[80];
sx node[81];
sx node[82];
sx node[83];
sx node[84];
sx node[92];
sx node[101];
sx node[102];
sx node[103];
sx node[104];
sx node[105];
sx node[111];
sx node[121];
sx node[122];
sx node[123];
sx node[124];
sx node[125];
rz(0.5*pi) node[61];
rz(0.5*pi) node[62];
rz(0.5*pi) node[72];
rz(0.5*pi) node[80];
rz(0.5*pi) node[81];
rz(0.5*pi) node[82];
rz(0.5*pi) node[83];
rz(0.5*pi) node[84];
rz(0.5*pi) node[92];
rz(0.5*pi) node[101];
rz(0.5*pi) node[102];
rz(0.5*pi) node[103];
rz(0.5*pi) node[104];
rz(0.5*pi) node[105];
rz(0.5*pi) node[111];
rz(0.5*pi) node[121];
rz(0.5*pi) node[122];
rz(0.5*pi) node[123];
rz(0.5*pi) node[125];
cx node[123],node[124];
rz(0.5*pi) node[123];
cx node[125],node[124];
sx node[123];
rz(0.5*pi) node[125];
rz(3.5*pi) node[123];
sx node[125];
sx node[123];
rz(3.5*pi) node[125];
rz(1.0*pi) node[123];
sx node[125];
cx node[124],node[123];
rz(1.0*pi) node[125];
cx node[123],node[124];
cx node[124],node[123];
cx node[122],node[123];
rz(0.5*pi) node[122];
sx node[122];
rz(3.5*pi) node[122];
sx node[122];
rz(1.0*pi) node[122];
cx node[123],node[122];
cx node[122],node[123];
cx node[123],node[122];
cx node[111],node[122];
rz(0.5*pi) node[111];
cx node[121],node[122];
sx node[111];
rz(0.5*pi) node[121];
rz(3.5*pi) node[111];
sx node[121];
sx node[111];
rz(3.5*pi) node[121];
rz(1.0*pi) node[111];
sx node[121];
cx node[122],node[111];
rz(1.0*pi) node[121];
cx node[111],node[122];
cx node[122],node[111];
cx node[104],node[111];
rz(0.5*pi) node[104];
sx node[104];
rz(3.5*pi) node[104];
sx node[104];
rz(1.0*pi) node[104];
cx node[111],node[104];
cx node[104],node[111];
cx node[111],node[104];
cx node[103],node[104];
rz(0.5*pi) node[103];
cx node[105],node[104];
sx node[103];
rz(0.5*pi) node[105];
rz(3.5*pi) node[103];
sx node[105];
sx node[103];
rz(3.5*pi) node[105];
rz(1.0*pi) node[103];
sx node[105];
cx node[104],node[103];
rz(1.0*pi) node[105];
cx node[103],node[104];
cx node[104],node[103];
cx node[102],node[103];
rz(0.5*pi) node[102];
sx node[102];
rz(3.5*pi) node[102];
sx node[102];
rz(1.0*pi) node[102];
cx node[103],node[102];
cx node[102],node[103];
cx node[103],node[102];
cx node[92],node[102];
rz(0.5*pi) node[92];
cx node[101],node[102];
sx node[92];
rz(0.5*pi) node[101];
rz(3.5*pi) node[92];
sx node[101];
sx node[92];
rz(3.5*pi) node[101];
rz(1.0*pi) node[92];
sx node[101];
cx node[102],node[92];
rz(1.0*pi) node[101];
cx node[92],node[102];
cx node[102],node[92];
cx node[83],node[92];
rz(0.5*pi) node[83];
sx node[83];
rz(3.5*pi) node[83];
sx node[83];
rz(1.0*pi) node[83];
cx node[92],node[83];
cx node[83],node[92];
cx node[92],node[83];
cx node[82],node[83];
rz(0.5*pi) node[82];
cx node[84],node[83];
sx node[82];
rz(0.5*pi) node[84];
rz(3.5*pi) node[82];
sx node[84];
sx node[82];
rz(3.5*pi) node[84];
rz(1.0*pi) node[82];
sx node[84];
cx node[83],node[82];
rz(1.0*pi) node[84];
cx node[82],node[83];
cx node[83],node[82];
cx node[81],node[82];
rz(0.5*pi) node[81];
sx node[81];
rz(3.5*pi) node[81];
sx node[81];
rz(1.0*pi) node[81];
cx node[82],node[81];
cx node[81],node[82];
cx node[82],node[81];
cx node[72],node[81];
rz(0.5*pi) node[72];
cx node[80],node[81];
sx node[72];
rz(0.5*pi) node[80];
rz(3.5*pi) node[72];
sx node[80];
sx node[72];
rz(3.5*pi) node[80];
rz(1.0*pi) node[72];
sx node[80];
cx node[81],node[72];
rz(1.0*pi) node[80];
cx node[72],node[81];
cx node[81],node[72];
cx node[62],node[72];
rz(0.5*pi) node[62];
sx node[62];
rz(3.5*pi) node[62];
sx node[62];
rz(1.0*pi) node[62];
cx node[72],node[62];
cx node[62],node[72];
cx node[72],node[62];
cx node[61],node[62];
rz(0.5*pi) node[61];
sx node[61];
rz(3.5*pi) node[61];
sx node[61];
rz(1.0*pi) node[61];
barrier node[124],node[125],node[123],node[122],node[121],node[111],node[104],node[105],node[103],node[102],node[101],node[92],node[83],node[84],node[82],node[81],node[80],node[72],node[61],node[62];
measure node[124] -> c[0];
measure node[125] -> c[1];
measure node[123] -> c[2];
measure node[122] -> c[3];
measure node[121] -> c[4];
measure node[111] -> c[5];
measure node[104] -> c[6];
measure node[105] -> c[7];
measure node[103] -> c[8];
measure node[102] -> c[9];
measure node[101] -> c[10];
measure node[92] -> c[11];
measure node[83] -> c[12];
measure node[84] -> c[13];
measure node[82] -> c[14];
measure node[81] -> c[15];
measure node[80] -> c[16];
measure node[72] -> c[17];
measure node[61] -> c[18];
