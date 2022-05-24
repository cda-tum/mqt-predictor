OPENQASM 2.0;
include "qelib1.inc";

qreg node[80];
creg meas[15];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[26];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rx(0.5*pi) node[74];
rx(0.5*pi) node[75];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[73];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
rx(0.5*pi) node[37];
rx(0.5*pi) node[65];
rx(0.5*pi) node[73];
rx(0.5*pi) node[75];
rx(0.5*pi) node[77];
rx(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[73];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
cz node[72],node[73];
cz node[74],node[75];
cz node[76],node[77];
cz node[78],node[79];
rz(0.5*pi) node[73];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
rx(0.5*pi) node[73];
rx(0.5*pi) node[75];
rx(0.5*pi) node[77];
rx(0.5*pi) node[79];
rz(0.5*pi) node[73];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
rz(1.0806471402905597*pi) node[73];
rz(1.0806471402905597*pi) node[75];
rz(1.0806471402905597*pi) node[77];
rz(1.0806471402905597*pi) node[79];
rz(0.5*pi) node[73];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
rx(0.5*pi) node[73];
rx(0.5*pi) node[75];
rx(0.5*pi) node[77];
rx(0.5*pi) node[79];
rz(0.5*pi) node[73];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
cz node[72],node[73];
cz node[74],node[75];
cz node[76],node[77];
cz node[78],node[79];
cz node[78],node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rx(0.5*pi) node[74];
rx(0.5*pi) node[75];
rx(0.5*pi) node[77];
rz(0.5*pi) node[78];
rx(0.5*pi) node[79];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[65];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[65],node[78];
rx(0.5*pi) node[75];
rx(0.5*pi) node[77];
rx(0.5*pi) node[79];
rz(0.5*pi) node[65];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[36],node[79];
rx(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[78],node[65];
rx(0.5*pi) node[79];
rz(0.5*pi) node[65];
cz node[78],node[77];
rz(0.5*pi) node[79];
rx(0.5*pi) node[65];
rz(0.5*pi) node[77];
rz(1.0806471402905597*pi) node[79];
rz(0.5*pi) node[65];
rx(0.5*pi) node[77];
rz(0.5*pi) node[79];
rz(0.5*pi) node[65];
rz(0.5*pi) node[77];
rx(0.5*pi) node[79];
rx(0.5*pi) node[65];
rz(1.0806471402905597*pi) node[77];
rz(0.5*pi) node[79];
cz node[36],node[79];
rz(0.5*pi) node[65];
rz(0.5*pi) node[77];
cz node[36],node[37];
cz node[64],node[65];
rx(0.5*pi) node[77];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[77];
rx(0.5*pi) node[79];
rx(0.5*pi) node[37];
rx(0.5*pi) node[65];
cz node[78],node[77];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rx(2.5638330033927854*pi) node[79];
rz(1.0806471402905597*pi) node[37];
rz(1.0806471402905597*pi) node[65];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rx(0.5*pi) node[37];
rx(0.5*pi) node[65];
rx(2.5638330033927854*pi) node[77];
cz node[79],node[78];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[36],node[37];
cz node[64],node[65];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rx(2.563833003392786*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[37];
rx(0.5*pi) node[65];
cz node[78],node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rx(0.5638330033927857*pi) node[65];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rx(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rx(0.5*pi) node[65];
cz node[79],node[78];
cz node[26],node[37];
rz(0.5*pi) node[65];
cz node[79],node[72];
rz(0.5*pi) node[78];
rz(0.5*pi) node[37];
cz node[64],node[65];
rz(0.5*pi) node[72];
rx(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
rz(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[37];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(1.0806471402905597*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
cz node[72],node[79];
rz(0.5*pi) node[37];
cz node[65],node[64];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rx(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rz(0.5*pi) node[37];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
cz node[26],node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
cz node[79],node[72];
rz(0.5*pi) node[37];
cz node[64],node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rx(0.5*pi) node[37];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rz(0.5*pi) node[37];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rx(2.5638330033927854*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
rx(0.5*pi) node[37];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
cz node[73],node[72];
cz node[26],node[37];
cz node[78],node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rz(0.5*pi) node[78];
rx(0.5*pi) node[26];
rx(0.5*pi) node[37];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rx(0.5*pi) node[78];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
cz node[72],node[73];
rz(0.5*pi) node[78];
cz node[37],node[26];
cz node[65],node[78];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rz(0.5*pi) node[78];
rx(0.5*pi) node[26];
rx(0.5*pi) node[37];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rx(0.5*pi) node[78];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
cz node[73],node[72];
rz(0.5*pi) node[78];
cz node[26],node[37];
cz node[78],node[65];
rz(0.5*pi) node[72];
cz node[73],node[74];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
rz(0.5*pi) node[74];
cz node[78],node[77];
rx(0.5*pi) node[26];
rx(0.5*pi) node[37];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rx(0.5*pi) node[74];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[74];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
rz(1.0806471402905597*pi) node[74];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rx(0.5*pi) node[37];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[74];
cz node[77],node[78];
cz node[35],node[72];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rx(0.5*pi) node[74];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
cz node[64],node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[74];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
cz node[73],node[74];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rx(2.563833003392786*pi) node[73];
rz(0.5*pi) node[74];
cz node[78],node[77];
rz(0.5*pi) node[65];
rz(1.0806471402905597*pi) node[72];
rx(0.5*pi) node[74];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(1.5276028477869952*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[74];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
rx(0.5638330033927857*pi) node[74];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[77];
cz node[35],node[72];
rz(0.5*pi) node[65];
rx(0.5*pi) node[77];
cz node[64],node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[77];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
cz node[76],node[77];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rx(2.5638330033927854*pi) node[72];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
cz node[65],node[78];
rz(0.5*pi) node[72];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
cz node[77],node[76];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
cz node[73],node[72];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rz(0.5*pi) node[78];
cz node[78],node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
cz node[76],node[77];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
cz node[76],node[75];
rz(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
cz node[72],node[73];
rz(0.5*pi) node[75];
rx(0.5*pi) node[77];
rz(0.5*pi) node[78];
cz node[65],node[78];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rx(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(1.0806471402905597*pi) node[75];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
cz node[73],node[72];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
cz node[66],node[77];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rx(0.5*pi) node[75];
cz node[78],node[79];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
cz node[76],node[75];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[75];
rx(2.563833003392786*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[75];
rz(1.0806471402905597*pi) node[77];
cz node[79],node[78];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(2.5638330033927854*pi) node[75];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[66],node[77];
rx(0.5*pi) node[75];
cz node[78],node[79];
rz(0.5*pi) node[66];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[66];
cz node[74],node[75];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[66];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[75];
rx(0.5638330033927857*pi) node[77];
rz(0.5*pi) node[79];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rx(0.5*pi) node[79];
rz(1.5276028477869952*pi) node[75];
rx(0.5*pi) node[77];
rz(0.5*pi) node[79];
cz node[36],node[79];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rx(0.5*pi) node[75];
cz node[76],node[77];
rz(0.5*pi) node[79];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rx(0.5*pi) node[79];
cz node[74],node[75];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rz(0.5*pi) node[79];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(1.5276028477869952*pi) node[79];
rx(0.5*pi) node[74];
rx(0.5*pi) node[75];
cz node[77],node[76];
rz(0.5*pi) node[79];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rx(0.5*pi) node[79];
rz(0.5*pi) node[75];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rz(0.5*pi) node[79];
cz node[36],node[79];
rx(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
cz node[36],node[37];
rz(0.5*pi) node[75];
cz node[76],node[77];
rz(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[77];
rx(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[77];
rz(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[77];
rx(1.2961325361261486*pi) node[79];
cz node[37],node[36];
cz node[77],node[66];
rz(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[66];
rz(0.5*pi) node[77];
rx(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[66];
rx(0.5*pi) node[77];
rz(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[66];
rz(0.5*pi) node[77];
cz node[36],node[37];
cz node[66],node[77];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[66];
rz(0.5*pi) node[77];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[66];
rx(0.5*pi) node[77];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[66];
rz(0.5*pi) node[77];
cz node[37],node[26];
cz node[35],node[36];
cz node[77],node[66];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[66];
cz node[77],node[78];
rx(0.5*pi) node[26];
rx(0.5*pi) node[36];
rx(0.5*pi) node[66];
rz(0.5*pi) node[78];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[66];
rx(0.5*pi) node[78];
rz(1.5276028477869952*pi) node[26];
rz(1.0806471402905597*pi) node[36];
cz node[66],node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(1.0806471402905597*pi) node[78];
rx(0.5*pi) node[26];
rx(0.5*pi) node[36];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rz(0.5*pi) node[78];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rx(0.5*pi) node[78];
cz node[37],node[26];
cz node[35],node[36];
cz node[65],node[66];
rz(0.5*pi) node[78];
rz(0.5*pi) node[26];
rx(2.563833003392786*pi) node[35];
rz(0.5*pi) node[36];
rx(3.2961325361261484*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
cz node[77],node[78];
rx(0.5*pi) node[26];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rx(2.563833003392786*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5638330033927857*pi) node[36];
rz(0.5*pi) node[37];
cz node[66],node[65];
rx(0.5*pi) node[77];
rz(0.5*pi) node[78];
cz node[26],node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[77];
rx(0.5638330033927857*pi) node[78];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
cz node[76],node[77];
cz node[78],node[79];
rx(0.5*pi) node[26];
rx(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
cz node[65],node[64];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
cz node[37],node[26];
rz(0.5*pi) node[64];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rx(0.5*pi) node[64];
cz node[77],node[76];
cz node[79],node[78];
rx(0.5*pi) node[26];
rx(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(1.5276028477869952*pi) node[64];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
cz node[26],node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rx(0.5*pi) node[64];
cz node[76],node[77];
cz node[78],node[79];
rx(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
cz node[65],node[64];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
cz node[65],node[66];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[37];
rx(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rx(0.5*pi) node[79];
cz node[36],node[37];
rx(1.2961325361261486*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
cz node[66],node[65];
cz node[72],node[79];
rx(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rz(1.5276028477869952*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
cz node[65],node[66];
cz node[79],node[72];
rx(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
cz node[36],node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[66];
cz node[72],node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[66];
cz node[72],node[73];
rz(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[66];
rz(0.5*pi) node[73];
rx(0.5*pi) node[79];
rx(1.2961325361261486*pi) node[37];
cz node[77],node[66];
rx(0.5*pi) node[73];
rz(0.5*pi) node[79];
rz(0.5*pi) node[66];
rz(0.5*pi) node[73];
rz(0.5*pi) node[77];
rx(0.5*pi) node[66];
rz(1.5276028477869952*pi) node[73];
rx(0.5*pi) node[77];
rz(0.5*pi) node[66];
rz(0.5*pi) node[73];
rz(0.5*pi) node[77];
cz node[66],node[77];
rx(0.5*pi) node[73];
rz(0.5*pi) node[66];
rz(0.5*pi) node[73];
rz(0.5*pi) node[77];
rx(0.5*pi) node[66];
cz node[72],node[73];
rx(0.5*pi) node[77];
rz(0.5*pi) node[66];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[77];
cz node[77],node[66];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rz(0.5*pi) node[66];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[77];
rx(0.5*pi) node[66];
rx(0.5*pi) node[77];
rz(0.5*pi) node[66];
rz(0.5*pi) node[77];
cz node[66],node[65];
cz node[76],node[77];
rz(0.5*pi) node[65];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rx(0.5*pi) node[65];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rz(0.5*pi) node[65];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(1.5276028477869952*pi) node[65];
cz node[77],node[76];
rz(0.5*pi) node[65];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rx(0.5*pi) node[65];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rz(0.5*pi) node[65];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
cz node[66],node[65];
cz node[76],node[77];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
cz node[76],node[75];
rz(0.5*pi) node[77];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rz(0.5*pi) node[75];
rx(0.5*pi) node[77];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rx(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[65];
cz node[77],node[66];
rz(0.5*pi) node[75];
rx(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(1.5276028477869952*pi) node[75];
rz(0.5*pi) node[65];
rx(0.5*pi) node[66];
rz(0.5*pi) node[75];
rz(0.5*pi) node[66];
rx(0.5*pi) node[75];
rz(1.5276028477869952*pi) node[66];
rz(0.5*pi) node[75];
rz(0.5*pi) node[66];
cz node[76],node[75];
rx(0.5*pi) node[66];
rz(0.5*pi) node[75];
rx(3.2961325361261484*pi) node[76];
rz(0.5*pi) node[66];
rx(0.5*pi) node[75];
cz node[77],node[66];
rz(0.5*pi) node[75];
rz(0.5*pi) node[66];
rx(1.2961325361261486*pi) node[75];
cz node[77],node[78];
rx(0.5*pi) node[66];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[66];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(1.2961325361261486*pi) node[66];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
cz node[78],node[77];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
cz node[77],node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
rz(0.5*pi) node[78];
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
cz node[79],node[72];
rz(0.5*pi) node[78];
rz(0.5*pi) node[72];
rx(0.5*pi) node[78];
rx(0.5*pi) node[72];
rz(0.5*pi) node[78];
cz node[78],node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[65];
rz(1.5276028477869952*pi) node[72];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
rz(1.5276028477869952*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[65];
cz node[79],node[72];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rx(3.2961325361261484*pi) node[79];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
rz(0.5*pi) node[79];
cz node[78],node[65];
rz(0.5*pi) node[72];
rx(0.5*pi) node[79];
rz(0.5*pi) node[65];
rx(1.2961325361261486*pi) node[72];
rz(0.5*pi) node[79];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
cz node[78],node[79];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(1.2961325361261486*pi) node[65];
rz(0.5*pi) node[72];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[79],node[78];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[78],node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
cz node[79],node[72];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
cz node[72],node[79];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
cz node[79],node[72];
rz(0.5*pi) node[72];
rx(0.5*pi) node[72];
rz(0.5*pi) node[72];
rz(0.5*pi) node[72];
rx(0.5*pi) node[72];
rz(0.5*pi) node[72];
cz node[73],node[72];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
cz node[72],node[73];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
cz node[73],node[72];
rz(0.5*pi) node[72];
cz node[73],node[74];
rx(0.5*pi) node[72];
rz(0.5*pi) node[74];
rz(0.5*pi) node[72];
rx(0.5*pi) node[74];
rz(0.5*pi) node[72];
rz(0.5*pi) node[74];
rx(0.5*pi) node[72];
rz(1.5276028477869952*pi) node[74];
rz(0.5*pi) node[72];
rz(0.5*pi) node[74];
cz node[35],node[72];
rx(0.5*pi) node[74];
rz(0.5*pi) node[72];
rz(0.5*pi) node[74];
rx(0.5*pi) node[72];
cz node[73],node[74];
rz(0.5*pi) node[72];
rx(3.2961325361261484*pi) node[73];
rz(0.5*pi) node[74];
rz(1.5276028477869952*pi) node[72];
rx(0.5*pi) node[74];
rz(0.5*pi) node[72];
rz(0.5*pi) node[74];
rx(0.5*pi) node[72];
rx(1.2961325361261486*pi) node[74];
rz(0.5*pi) node[72];
cz node[35],node[72];
cz node[35],node[36];
rz(0.5*pi) node[72];
rz(0.5*pi) node[36];
rx(0.5*pi) node[72];
rx(0.5*pi) node[36];
rz(0.5*pi) node[72];
rz(0.5*pi) node[36];
rx(1.2961325361261486*pi) node[72];
rz(1.5276028477869952*pi) node[36];
rz(0.5*pi) node[36];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
cz node[35],node[36];
rx(3.2961325361261484*pi) node[35];
rz(0.5*pi) node[36];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
rx(1.2961325361261486*pi) node[36];
barrier node[78],node[35],node[73],node[79],node[76],node[36],node[26],node[37],node[74],node[72],node[75],node[66],node[64],node[65],node[77];
measure node[78] -> meas[0];
measure node[35] -> meas[1];
measure node[73] -> meas[2];
measure node[79] -> meas[3];
measure node[76] -> meas[4];
measure node[36] -> meas[5];
measure node[26] -> meas[6];
measure node[37] -> meas[7];
measure node[74] -> meas[8];
measure node[72] -> meas[9];
measure node[75] -> meas[10];
measure node[66] -> meas[11];
measure node[64] -> meas[12];
measure node[65] -> meas[13];
measure node[77] -> meas[14];
