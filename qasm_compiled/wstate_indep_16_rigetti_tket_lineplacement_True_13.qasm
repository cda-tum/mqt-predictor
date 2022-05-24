OPENQASM 2.0;
include "qelib1.inc";

qreg node[80];
creg meas[16];
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
rx(2.75*pi) node[56];
rx(2.8040867245816834*pi) node[57];
rx(2.8918265465108672*pi) node[65];
rx(2.884973270999353*pi) node[66];
rx(2.8766241300087065*pi) node[67];
rx(2.8661397663015395*pi) node[68];
rx(2.852416376685532*pi) node[69];
rx(2.833333333333333*pi) node[70];
rx(2.919569385768022*pi) node[72];
rx(2.9168710092008645*pi) node[73];
rx(2.9138813472568605*pi) node[74];
rx(2.9105438044382463*pi) node[75];
rx(2.9067852649641654*pi) node[76];
rx(2.9025088989672407*pi) node[77];
rx(2.897583626436342*pi) node[78];
rx(1.0*pi) node[79];
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
rx(0.919569385768022*pi) node[72];
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
rx(0.9168710092008648*pi) node[73];
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
rx(0.9138813472568608*pi) node[74];
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
rx(0.9105438044382466*pi) node[75];
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
rx(0.9067852649641656*pi) node[76];
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
rx(0.9025088989672407*pi) node[77];
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
rx(0.8975836264363417*pi) node[78];
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
rx(0.8918265465108672*pi) node[65];
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
rx(0.884973270999353*pi) node[66];
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
rx(0.8766241300087068*pi) node[67];
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
rx(0.8661397663015394*pi) node[68];
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
rx(0.8524163766855318*pi) node[69];
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
rx(0.8333333333333341*pi) node[70];
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
rx(0.8040867245816836*pi) node[57];
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
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[70];
rx(0.75*pi) node[56];
rz(0.5*pi) node[56];
cz node[56],node[57];
rz(0.5*pi) node[57];
rx(0.5*pi) node[57];
rz(0.5*pi) node[57];
barrier node[56],node[57],node[70],node[69],node[68],node[67],node[66],node[65],node[78],node[77],node[76],node[75],node[74],node[73],node[72],node[79];
measure node[56] -> meas[0];
measure node[57] -> meas[1];
measure node[70] -> meas[2];
measure node[69] -> meas[3];
measure node[68] -> meas[4];
measure node[67] -> meas[5];
measure node[66] -> meas[6];
measure node[65] -> meas[7];
measure node[78] -> meas[8];
measure node[77] -> meas[9];
measure node[76] -> meas[10];
measure node[75] -> meas[11];
measure node[74] -> meas[12];
measure node[73] -> meas[13];
measure node[72] -> meas[14];
measure node[79] -> meas[15];
