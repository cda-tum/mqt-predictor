OPENQASM 2.0;
include "qelib1.inc";

qreg node[80];
creg c[8];
rz(3.5*pi) node[27];
rz(3.5*pi) node[28];
rz(3.5*pi) node[64];
rz(3.5*pi) node[65];
rz(3.5*pi) node[66];
rz(3.5*pi) node[71];
rz(3.5*pi) node[77];
rz(3.5*pi) node[78];
rz(3.5*pi) node[79];
rx(1.5*pi) node[27];
rx(1.5*pi) node[28];
rx(1.5*pi) node[64];
rx(1.5*pi) node[65];
rx(1.5*pi) node[66];
rx(1.5*pi) node[71];
rx(1.5*pi) node[77];
rx(1.5*pi) node[78];
rx(1.5*pi) node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[79],node[78];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[78],node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[79],node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[77],node[78];
rx(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[79],node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[79];
rx(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[65],node[78];
rx(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rz(0.5*pi) node[65];
cz node[78],node[65];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
cz node[65],node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
cz node[78],node[65];
rz(0.5*pi) node[65];
rx(0.5*pi) node[65];
rz(0.5*pi) node[65];
rz(0.5*pi) node[65];
rx(0.5*pi) node[65];
rz(0.5*pi) node[65];
cz node[64],node[65];
rx(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[64];
rx(0.5*pi) node[65];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rx(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[64];
rx(0.5*pi) node[65];
rz(0.5*pi) node[65];
cz node[66],node[65];
rz(0.5*pi) node[65];
rx(0.5*pi) node[66];
rx(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[65];
cz node[65],node[64];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
cz node[64],node[65];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
cz node[65],node[64];
rz(0.5*pi) node[64];
rx(0.5*pi) node[64];
rz(0.5*pi) node[64];
rz(0.5*pi) node[64];
rx(0.5*pi) node[64];
rz(0.5*pi) node[64];
cz node[27],node[64];
rx(0.5*pi) node[27];
rz(0.5*pi) node[64];
rz(0.5*pi) node[27];
rx(0.5*pi) node[64];
rz(0.5*pi) node[27];
rz(0.5*pi) node[64];
rx(0.5*pi) node[27];
rz(0.5*pi) node[64];
rz(0.5*pi) node[27];
rx(0.5*pi) node[64];
cz node[28],node[27];
rz(0.5*pi) node[64];
rz(0.5*pi) node[27];
cz node[71],node[64];
rx(0.5*pi) node[27];
rz(0.5*pi) node[64];
rx(0.5*pi) node[71];
rz(0.5*pi) node[27];
rx(0.5*pi) node[64];
rz(0.5*pi) node[71];
rz(0.5*pi) node[64];
rz(0.5*pi) node[64];
rx(0.5*pi) node[64];
rz(0.5*pi) node[64];
cz node[27],node[64];
rz(0.5*pi) node[27];
rz(0.5*pi) node[64];
rx(0.5*pi) node[27];
rx(0.5*pi) node[64];
rz(0.5*pi) node[27];
rz(0.5*pi) node[64];
cz node[28],node[27];
rz(0.5*pi) node[64];
rz(0.5*pi) node[27];
rx(0.5*pi) node[28];
rx(0.5*pi) node[64];
rx(0.5*pi) node[27];
rz(0.5*pi) node[28];
rz(0.5*pi) node[64];
rz(0.5*pi) node[27];
cz node[27],node[64];
rz(0.5*pi) node[64];
rx(0.5*pi) node[64];
rz(0.5*pi) node[64];
barrier node[77],node[79],node[78],node[65],node[66],node[27],node[71],node[28],node[64];
measure node[77] -> c[0];
measure node[79] -> c[1];
measure node[78] -> c[2];
measure node[65] -> c[3];
measure node[66] -> c[4];
measure node[27] -> c[5];
measure node[71] -> c[6];
measure node[28] -> c[7];
