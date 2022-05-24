OPENQASM 2.0;
include "qelib1.inc";

qreg node[80];
creg meas[34];
rz(2.5*pi) node[0];
rz(2.5*pi) node[1];
rz(2.5*pi) node[2];
rz(2.5*pi) node[3];
rz(2.5*pi) node[4];
rz(2.5*pi) node[5];
rz(2.5*pi) node[6];
rz(2.5*pi) node[7];
rz(2.5*pi) node[10];
rz(2.5*pi) node[11];
rz(2.5*pi) node[12];
rz(2.5*pi) node[13];
rz(2.5*pi) node[19];
rz(2.5*pi) node[20];
rz(2.5*pi) node[21];
rz(2.5*pi) node[40];
rz(2.5*pi) node[46];
rz(2.5*pi) node[47];
rz(2.5*pi) node[56];
rz(2.5*pi) node[57];
rz(2.5*pi) node[65];
rz(2.5*pi) node[66];
rz(2.5*pi) node[67];
rz(2.5*pi) node[68];
rz(2.5*pi) node[69];
rz(2.5*pi) node[70];
rz(2.5*pi) node[72];
rz(2.5*pi) node[73];
rz(2.5*pi) node[74];
rz(2.5*pi) node[75];
rz(2.5*pi) node[76];
rz(2.5*pi) node[77];
rz(2.5*pi) node[78];
rz(0.5*pi) node[79];
rx(2.897583626436342*pi) node[0];
rx(2.9025088989672407*pi) node[1];
rx(2.9067852649641654*pi) node[2];
rx(2.852416376685532*pi) node[3];
rx(2.8661397663015395*pi) node[4];
rx(2.8766241300087065*pi) node[5];
rx(2.884973270999353*pi) node[6];
rx(2.8918265465108672*pi) node[7];
rx(2.919569385768022*pi) node[10];
rx(2.9168710092008645*pi) node[11];
rx(2.9138813472568605*pi) node[12];
rx(2.9105438044382463*pi) node[13];
rx(2.9263184784537883*pi) node[19];
rx(2.9242609870114737*pi) node[20];
rx(2.9220208811874553*pi) node[21];
rx(2.833333333333333*pi) node[40];
rx(2.75*pi) node[46];
rx(2.8040867245816834*pi) node[47];
rx(2.9282168467839997*pi) node[56];
rx(2.92997563622912*pi) node[57];
rx(2.9383566400393732*pi) node[65];
rx(2.937167052332727*pi) node[66];
rx(2.9359057812397125*pi) node[67];
rx(2.9345653783089922*pi) node[68];
rx(2.9331371855116632*pi) node[69];
rx(2.9316111760863093*pi) node[70];
rx(2.9451390916012112*pi) node[72];
rx(2.944305628995228*pi) node[73];
rx(2.9434329506112666*pi) node[74];
rx(2.9425179370124424*pi) node[75];
rx(2.9415570231280306*pi) node[76];
rx(2.9405462619154425*pi) node[77];
rx(2.9394811333742945*pi) node[78];
rx(1.0*pi) node[79];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[40];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[68];
rz(0.5*pi) node[69];
rz(0.5*pi) node[70];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[40];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[68];
rz(0.5*pi) node[69];
rz(0.5*pi) node[70];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[4];
rx(0.5*pi) node[5];
rx(0.5*pi) node[6];
rx(0.5*pi) node[7];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[19];
rx(0.5*pi) node[20];
rx(0.5*pi) node[21];
rx(0.5*pi) node[40];
rx(0.5*pi) node[46];
rx(0.5*pi) node[47];
rx(0.5*pi) node[56];
rx(0.5*pi) node[57];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
rx(0.5*pi) node[68];
rx(0.5*pi) node[69];
rx(0.5*pi) node[70];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rx(0.5*pi) node[74];
rx(0.5*pi) node[75];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[40];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[68];
rz(0.5*pi) node[69];
rz(0.5*pi) node[70];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
cz node[79],node[72];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(2.5*pi) node[72];
rx(0.9451390916012115*pi) node[72];
rz(0.5*pi) node[72];
cz node[72],node[73];
cz node[72],node[79];
rz(0.5*pi) node[73];
rz(0.5*pi) node[72];
rx(0.5*pi) node[73];
rz(0.5*pi) node[79];
rx(0.5*pi) node[72];
rz(0.5*pi) node[73];
rx(0.5*pi) node[79];
rz(0.5*pi) node[72];
rz(2.5*pi) node[73];
rz(0.5*pi) node[79];
rx(0.9443056289952279*pi) node[73];
rz(0.5*pi) node[73];
cz node[73],node[74];
cz node[73],node[72];
rz(0.5*pi) node[74];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rx(0.5*pi) node[74];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(2.5*pi) node[74];
rx(0.9434329506112664*pi) node[74];
rz(0.5*pi) node[74];
cz node[74],node[75];
cz node[74],node[73];
rz(0.5*pi) node[75];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rx(0.5*pi) node[75];
rx(0.5*pi) node[73];
rx(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(2.5*pi) node[75];
rx(0.9425179370124424*pi) node[75];
rz(0.5*pi) node[75];
cz node[75],node[76];
cz node[75],node[74];
rz(0.5*pi) node[76];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rx(0.5*pi) node[76];
rx(0.5*pi) node[74];
rx(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(2.5*pi) node[76];
rx(0.9415570231280308*pi) node[76];
rz(0.5*pi) node[76];
cz node[76],node[77];
cz node[76],node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rx(0.5*pi) node[77];
rx(0.5*pi) node[75];
rx(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(2.5*pi) node[77];
rx(0.9405462619154428*pi) node[77];
rz(0.5*pi) node[77];
cz node[77],node[78];
cz node[77],node[76];
rz(0.5*pi) node[78];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(2.5*pi) node[78];
rx(0.9394811333742946*pi) node[78];
rz(0.5*pi) node[78];
cz node[78],node[65];
rz(0.5*pi) node[65];
cz node[78],node[77];
rx(0.5*pi) node[65];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[65];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(2.5*pi) node[65];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rx(0.9383566400393731*pi) node[65];
rz(0.5*pi) node[65];
cz node[65],node[66];
cz node[65],node[78];
rz(0.5*pi) node[66];
rz(0.5*pi) node[65];
rx(0.5*pi) node[66];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rz(0.5*pi) node[66];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
rz(2.5*pi) node[66];
rz(0.5*pi) node[78];
rx(0.9371670523327271*pi) node[66];
rz(0.5*pi) node[66];
cz node[66],node[67];
cz node[66],node[65];
rz(0.5*pi) node[67];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rx(0.5*pi) node[67];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(2.5*pi) node[67];
rx(0.9359057812397125*pi) node[67];
rz(0.5*pi) node[67];
cz node[67],node[68];
cz node[67],node[66];
rz(0.5*pi) node[68];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rx(0.5*pi) node[68];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
rz(0.5*pi) node[68];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(2.5*pi) node[68];
rx(0.9345653783089924*pi) node[68];
rz(0.5*pi) node[68];
cz node[68],node[69];
cz node[68],node[67];
rz(0.5*pi) node[69];
rz(0.5*pi) node[67];
rz(0.5*pi) node[68];
rx(0.5*pi) node[69];
rx(0.5*pi) node[67];
rx(0.5*pi) node[68];
rz(0.5*pi) node[69];
rz(0.5*pi) node[67];
rz(0.5*pi) node[68];
rz(2.5*pi) node[69];
rx(0.933137185511663*pi) node[69];
rz(0.5*pi) node[69];
cz node[69],node[70];
cz node[69],node[68];
rz(0.5*pi) node[70];
rz(0.5*pi) node[68];
rz(0.5*pi) node[69];
rx(0.5*pi) node[70];
rx(0.5*pi) node[68];
rx(0.5*pi) node[69];
rz(0.5*pi) node[70];
rz(0.5*pi) node[68];
rz(0.5*pi) node[69];
rz(2.5*pi) node[70];
rx(0.9316111760863093*pi) node[70];
rz(0.5*pi) node[70];
cz node[70],node[57];
rz(0.5*pi) node[57];
cz node[70],node[69];
rx(0.5*pi) node[57];
rz(0.5*pi) node[69];
rz(0.5*pi) node[70];
rz(0.5*pi) node[57];
rx(0.5*pi) node[69];
rx(0.5*pi) node[70];
rz(2.5*pi) node[57];
rz(0.5*pi) node[69];
rz(0.5*pi) node[70];
rx(0.9299756362291198*pi) node[57];
rz(0.5*pi) node[57];
cz node[57],node[56];
rz(0.5*pi) node[56];
cz node[57],node[70];
rx(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[70];
rz(0.5*pi) node[56];
rx(0.5*pi) node[57];
rx(0.5*pi) node[70];
rz(2.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[70];
rx(0.9282168467839998*pi) node[56];
rz(0.5*pi) node[56];
cz node[56],node[19];
rz(0.5*pi) node[19];
cz node[56],node[57];
rx(0.5*pi) node[19];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[19];
rx(0.5*pi) node[56];
rx(0.5*pi) node[57];
rz(2.5*pi) node[19];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rx(0.9263184784537883*pi) node[19];
rz(0.5*pi) node[19];
cz node[19],node[20];
cz node[19],node[56];
rz(0.5*pi) node[20];
rz(0.5*pi) node[19];
rx(0.5*pi) node[20];
rz(0.5*pi) node[56];
rx(0.5*pi) node[19];
rz(0.5*pi) node[20];
rx(0.5*pi) node[56];
rz(0.5*pi) node[19];
rz(2.5*pi) node[20];
rz(0.5*pi) node[56];
rx(0.9242609870114735*pi) node[20];
rz(0.5*pi) node[20];
cz node[20],node[21];
cz node[20],node[19];
rz(0.5*pi) node[21];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rx(0.5*pi) node[21];
rx(0.5*pi) node[19];
rx(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(2.5*pi) node[21];
rx(0.9220208811874551*pi) node[21];
rz(0.5*pi) node[21];
cz node[21],node[10];
rz(0.5*pi) node[10];
cz node[21],node[20];
rx(0.5*pi) node[10];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[10];
rx(0.5*pi) node[20];
rx(0.5*pi) node[21];
rz(2.5*pi) node[10];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rx(0.919569385768022*pi) node[10];
rz(0.5*pi) node[10];
cz node[10],node[11];
cz node[10],node[21];
rz(0.5*pi) node[11];
rz(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[21];
rx(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[21];
rz(0.5*pi) node[10];
rz(2.5*pi) node[11];
rz(0.5*pi) node[21];
rx(0.9168710092008648*pi) node[11];
rz(0.5*pi) node[11];
cz node[11],node[12];
cz node[11],node[10];
rz(0.5*pi) node[12];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[12];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(2.5*pi) node[12];
rx(0.9138813472568608*pi) node[12];
rz(0.5*pi) node[12];
cz node[12],node[13];
cz node[12],node[11];
rz(0.5*pi) node[13];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[11];
rx(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(2.5*pi) node[13];
rx(0.9105438044382466*pi) node[13];
rz(0.5*pi) node[13];
cz node[13],node[2];
rz(0.5*pi) node[2];
cz node[13],node[12];
rx(0.5*pi) node[2];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[2];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rz(2.5*pi) node[2];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rx(0.9067852649641656*pi) node[2];
rz(0.5*pi) node[2];
cz node[2],node[1];
rz(0.5*pi) node[1];
cz node[2],node[13];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[13];
rz(2.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rx(0.9025088989672407*pi) node[1];
rz(0.5*pi) node[1];
cz node[1],node[0];
rz(0.5*pi) node[0];
cz node[1],node[2];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(2.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.8975836264363417*pi) node[0];
rz(0.5*pi) node[0];
cz node[0],node[7];
cz node[0],node[1];
rz(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[7];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(2.5*pi) node[7];
rx(0.8918265465108672*pi) node[7];
rz(0.5*pi) node[7];
cz node[7],node[6];
cz node[7],node[0];
rz(0.5*pi) node[6];
rz(0.5*pi) node[0];
rx(0.5*pi) node[6];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
rz(0.5*pi) node[6];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(2.5*pi) node[6];
rz(0.5*pi) node[7];
rx(0.884973270999353*pi) node[6];
rz(0.5*pi) node[6];
cz node[6],node[5];
rz(0.5*pi) node[5];
cz node[6],node[7];
rx(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[5];
rx(0.5*pi) node[6];
rx(0.5*pi) node[7];
rz(2.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rx(0.8766241300087068*pi) node[5];
rz(0.5*pi) node[5];
cz node[5],node[4];
rz(0.5*pi) node[4];
cz node[5],node[6];
rx(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[4];
rx(0.5*pi) node[5];
rx(0.5*pi) node[6];
rz(2.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rx(0.8661397663015394*pi) node[4];
rz(0.5*pi) node[4];
cz node[4],node[3];
rz(0.5*pi) node[3];
cz node[4],node[5];
rx(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[3];
rx(0.5*pi) node[4];
rx(0.5*pi) node[5];
rz(2.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rx(0.8524163766855318*pi) node[3];
rz(0.5*pi) node[3];
cz node[3],node[40];
cz node[3],node[4];
rz(0.5*pi) node[40];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rx(0.5*pi) node[40];
rx(0.5*pi) node[3];
rx(0.5*pi) node[4];
rz(0.5*pi) node[40];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(2.5*pi) node[40];
rx(0.8333333333333341*pi) node[40];
rz(0.5*pi) node[40];
cz node[40],node[47];
cz node[40],node[3];
rz(0.5*pi) node[47];
rz(0.5*pi) node[3];
rz(0.5*pi) node[40];
rx(0.5*pi) node[47];
rx(0.5*pi) node[3];
rx(0.5*pi) node[40];
rz(0.5*pi) node[47];
rz(0.5*pi) node[3];
rz(0.5*pi) node[40];
rz(2.5*pi) node[47];
rx(0.8040867245816836*pi) node[47];
rz(0.5*pi) node[47];
cz node[47],node[46];
cz node[47],node[40];
rz(0.5*pi) node[46];
rz(0.5*pi) node[40];
rx(0.5*pi) node[46];
rz(0.5*pi) node[47];
rx(0.5*pi) node[40];
rz(0.5*pi) node[46];
rx(0.5*pi) node[47];
rz(0.5*pi) node[40];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
rx(0.75*pi) node[46];
rz(0.5*pi) node[46];
cz node[46],node[47];
rz(0.5*pi) node[47];
rx(0.5*pi) node[47];
rz(0.5*pi) node[47];
barrier node[46],node[47],node[40],node[3],node[4],node[5],node[6],node[7],node[0],node[1],node[2],node[13],node[12],node[11],node[10],node[21],node[20],node[19],node[56],node[57],node[70],node[69],node[68],node[67],node[66],node[65],node[78],node[77],node[76],node[75],node[74],node[73],node[72],node[79];
measure node[46] -> meas[0];
measure node[47] -> meas[1];
measure node[40] -> meas[2];
measure node[3] -> meas[3];
measure node[4] -> meas[4];
measure node[5] -> meas[5];
measure node[6] -> meas[6];
measure node[7] -> meas[7];
measure node[0] -> meas[8];
measure node[1] -> meas[9];
measure node[2] -> meas[10];
measure node[13] -> meas[11];
measure node[12] -> meas[12];
measure node[11] -> meas[13];
measure node[10] -> meas[14];
measure node[21] -> meas[15];
measure node[20] -> meas[16];
measure node[19] -> meas[17];
measure node[56] -> meas[18];
measure node[57] -> meas[19];
measure node[70] -> meas[20];
measure node[69] -> meas[21];
measure node[68] -> meas[22];
measure node[67] -> meas[23];
measure node[66] -> meas[24];
measure node[65] -> meas[25];
measure node[78] -> meas[26];
measure node[77] -> meas[27];
measure node[76] -> meas[28];
measure node[75] -> meas[29];
measure node[74] -> meas[30];
measure node[73] -> meas[31];
measure node[72] -> meas[32];
measure node[79] -> meas[33];
