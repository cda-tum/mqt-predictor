OPENQASM 2.0;
include "qelib1.inc";

qreg node[127];
creg meas[25];
sx node[86];
sx node[87];
sx node[92];
sx node[93];
sx node[100];
sx node[101];
sx node[102];
sx node[103];
sx node[104];
sx node[105];
sx node[106];
sx node[107];
sx node[108];
sx node[110];
sx node[111];
sx node[112];
sx node[118];
sx node[119];
sx node[120];
sx node[121];
sx node[122];
sx node[123];
sx node[124];
sx node[125];
sx node[126];
rz(0.5*pi) node[86];
rz(0.5*pi) node[92];
rz(0.5*pi) node[100];
rz(0.5*pi) node[107];
rz(0.5*pi) node[112];
rz(0.5*pi) node[122];
rz(0.5*pi) node[124];
sx node[86];
sx node[92];
sx node[100];
sx node[107];
sx node[112];
sx node[122];
sx node[124];
cx node[86],node[87];
cx node[92],node[102];
cx node[100],node[110];
cx node[108],node[107];
cx node[122],node[111];
cx node[126],node[112];
cx node[124],node[125];
sx node[87];
sx node[102];
cx node[107],node[108];
sx node[110];
sx node[111];
cx node[112],node[126];
cx node[122],node[121];
cx node[124],node[123];
sx node[125];
rz(2.5*pi) node[87];
rz(2.5*pi) node[102];
cx node[108],node[107];
rz(2.5*pi) node[110];
rz(2.5*pi) node[111];
cx node[126],node[112];
sx node[121];
rz(2.5*pi) node[125];
sx node[87];
sx node[102];
cx node[107],node[106];
sx node[110];
sx node[111];
rz(2.5*pi) node[121];
sx node[125];
rz(1.5*pi) node[87];
rz(1.5*pi) node[102];
cx node[106],node[107];
rz(1.5*pi) node[110];
rz(1.5*pi) node[111];
sx node[121];
rz(1.5*pi) node[125];
cx node[87],node[93];
cx node[102],node[101];
cx node[111],node[104];
cx node[107],node[106];
cx node[110],node[118];
rz(1.5*pi) node[121];
cx node[126],node[125];
cx node[86],node[87];
cx node[92],node[102];
cx node[100],node[101];
cx node[106],node[105];
sx node[118];
cx node[121],node[120];
cx node[125],node[126];
cx node[87],node[86];
cx node[102],node[92];
sx node[101];
cx node[105],node[106];
rz(2.5*pi) node[118];
cx node[126],node[125];
cx node[86],node[87];
cx node[92],node[102];
rz(2.5*pi) node[101];
cx node[106],node[105];
sx node[118];
cx node[125],node[124];
cx node[87],node[93];
sx node[101];
cx node[105],node[104];
rz(1.5*pi) node[118];
cx node[124],node[125];
cx node[93],node[87];
rz(1.5*pi) node[101];
cx node[104],node[105];
cx node[118],node[119];
cx node[125],node[124];
cx node[87],node[93];
cx node[105],node[104];
sx node[119];
cx node[124],node[123];
cx node[93],node[106];
cx node[104],node[103];
rz(2.5*pi) node[119];
sx node[123];
cx node[106],node[93];
cx node[103],node[104];
sx node[119];
rz(2.5*pi) node[123];
cx node[93],node[106];
cx node[104],node[103];
rz(1.5*pi) node[119];
sx node[123];
cx node[87],node[93];
cx node[102],node[103];
cx node[106],node[107];
cx node[119],node[120];
rz(1.5*pi) node[123];
cx node[93],node[87];
cx node[107],node[106];
sx node[120];
cx node[124],node[123];
cx node[87],node[93];
cx node[106],node[107];
rz(2.5*pi) node[120];
cx node[123],node[124];
cx node[108],node[107];
sx node[120];
cx node[124],node[123];
cx node[107],node[108];
rz(1.5*pi) node[120];
cx node[123],node[122];
cx node[108],node[107];
cx node[122],node[123];
cx node[107],node[106];
cx node[108],node[112];
cx node[123],node[122];
cx node[106],node[107];
cx node[122],node[111];
cx node[126],node[112];
cx node[107],node[106];
cx node[111],node[122];
sx node[112];
cx node[106],node[105];
cx node[122],node[111];
rz(2.5*pi) node[112];
cx node[111],node[104];
cx node[105],node[106];
sx node[112];
sx node[104];
cx node[106],node[105];
rz(1.5*pi) node[112];
cx node[93],node[106];
rz(2.5*pi) node[104];
cx node[106],node[93];
sx node[104];
cx node[93],node[106];
rz(1.5*pi) node[104];
cx node[105],node[104];
cx node[104],node[103];
cx node[105],node[104];
cx node[104],node[103];
cx node[106],node[105];
sx node[103];
cx node[105],node[106];
rz(2.5*pi) node[103];
cx node[106],node[105];
sx node[103];
cx node[104],node[105];
cx node[106],node[107];
rz(1.5*pi) node[103];
sx node[105];
sx node[107];
rz(2.5*pi) node[105];
rz(2.5*pi) node[107];
sx node[105];
sx node[107];
rz(1.5*pi) node[105];
rz(1.5*pi) node[107];
cx node[107],node[106];
cx node[106],node[107];
cx node[107],node[106];
cx node[106],node[93];
cx node[93],node[87];
cx node[106],node[93];
cx node[93],node[87];
sx node[87];
rz(2.5*pi) node[87];
sx node[87];
rz(1.5*pi) node[87];
cx node[87],node[93];
sx node[93];
rz(2.5*pi) node[93];
sx node[93];
rz(1.5*pi) node[93];
barrier node[108],node[86],node[102],node[92],node[125],node[126],node[123],node[122],node[121],node[107],node[103],node[100],node[110],node[111],node[124],node[112],node[118],node[101],node[106],node[87],node[119],node[93],node[120],node[104],node[105];
measure node[108] -> meas[0];
measure node[86] -> meas[1];
measure node[102] -> meas[2];
measure node[92] -> meas[3];
measure node[125] -> meas[4];
measure node[126] -> meas[5];
measure node[123] -> meas[6];
measure node[122] -> meas[7];
measure node[121] -> meas[8];
measure node[107] -> meas[9];
measure node[103] -> meas[10];
measure node[100] -> meas[11];
measure node[110] -> meas[12];
measure node[111] -> meas[13];
measure node[124] -> meas[14];
measure node[112] -> meas[15];
measure node[118] -> meas[16];
measure node[101] -> meas[17];
measure node[106] -> meas[18];
measure node[87] -> meas[19];
measure node[119] -> meas[20];
measure node[93] -> meas[21];
measure node[120] -> meas[22];
measure node[104] -> meas[23];
measure node[105] -> meas[24];
