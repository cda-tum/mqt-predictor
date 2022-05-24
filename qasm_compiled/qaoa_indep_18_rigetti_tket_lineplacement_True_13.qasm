OPENQASM 2.0;
include "qelib1.inc";

qreg node[80];
creg meas[18];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[26];
rx(0.5*pi) node[27];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rx(0.5*pi) node[74];
rx(0.5*pi) node[75];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[64];
rz(0.5*pi) node[66];
rz(0.5*pi) node[73];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
rx(0.5*pi) node[64];
rx(0.5*pi) node[66];
rx(0.5*pi) node[73];
rx(0.5*pi) node[75];
rx(0.5*pi) node[77];
rx(0.5*pi) node[79];
rz(0.5*pi) node[64];
rz(0.5*pi) node[66];
rz(0.5*pi) node[73];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
cz node[65],node[66];
cz node[72],node[73];
cz node[74],node[75];
cz node[76],node[77];
cz node[78],node[79];
rz(0.5*pi) node[66];
rz(0.5*pi) node[73];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
rx(0.5*pi) node[66];
rx(0.5*pi) node[73];
rx(0.5*pi) node[75];
rx(0.5*pi) node[77];
rx(0.5*pi) node[79];
rz(0.5*pi) node[66];
rz(0.5*pi) node[73];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
rz(0.8477895035480554*pi) node[66];
rz(0.8477895035480554*pi) node[73];
rz(0.8477895035480554*pi) node[75];
rz(0.8477895035480554*pi) node[77];
rz(0.8477895035480554*pi) node[79];
rz(0.5*pi) node[66];
rz(0.5*pi) node[73];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
rx(0.5*pi) node[66];
rx(0.5*pi) node[73];
rx(0.5*pi) node[75];
rx(0.5*pi) node[77];
rx(0.5*pi) node[79];
rz(0.5*pi) node[66];
rz(0.5*pi) node[73];
rz(0.5*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
cz node[65],node[66];
cz node[72],node[73];
cz node[74],node[75];
cz node[76],node[77];
cz node[78],node[79];
cz node[65],node[64];
rz(0.5*pi) node[66];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rx(0.5*pi) node[66];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rx(0.5*pi) node[74];
rx(0.5*pi) node[75];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rx(0.5*pi) node[79];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
cz node[35],node[72];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
cz node[73],node[74];
cz node[75],node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
cz node[64],node[65];
rx(0.5*pi) node[66];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rx(0.5*pi) node[77];
rx(0.5*pi) node[79];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rx(0.5*pi) node[74];
rx(0.5*pi) node[75];
rx(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
cz node[36],node[79];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
cz node[67],node[66];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rz(0.5*pi) node[36];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.8477895035480554*pi) node[72];
cz node[74],node[73];
cz node[76],node[75];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
cz node[65],node[64];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rx(0.5*pi) node[74];
rx(0.5*pi) node[75];
rx(0.5*pi) node[76];
rz(0.5*pi) node[79];
cz node[79],node[36];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
cz node[66],node[67];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[75];
rz(0.5*pi) node[76];
cz node[35],node[72];
rz(0.5*pi) node[36];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
cz node[73],node[74];
cz node[75],node[76];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rz(0.5*pi) node[64];
cz node[78],node[65];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[76];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rx(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rx(0.5*pi) node[74];
rx(0.5*pi) node[76];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[36],node[79];
rz(0.5*pi) node[64];
rx(0.5*pi) node[65];
cz node[67],node[66];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[76];
rx(0.5*pi) node[78];
cz node[27],node[64];
rz(0.5*pi) node[36];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rx(1.3770510094858368*pi) node[72];
rz(0.5*pi) node[74];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[27];
rx(0.5*pi) node[36];
rz(0.5*pi) node[64];
cz node[65],node[78];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
cz node[72],node[73];
rx(0.5*pi) node[74];
rx(0.5*pi) node[79];
rx(0.5*pi) node[27];
rz(0.5*pi) node[36];
rx(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[27];
cz node[37],node[36];
rz(0.5*pi) node[64];
rx(0.5*pi) node[65];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
cz node[75],node[74];
rx(0.5*pi) node[78];
cz node[64],node[27];
rz(0.5*pi) node[36];
rz(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[78];
rz(0.5*pi) node[27];
rx(0.5*pi) node[36];
rz(0.5*pi) node[64];
cz node[78],node[65];
cz node[73],node[72];
rx(0.5*pi) node[74];
rx(0.5*pi) node[27];
rz(0.5*pi) node[36];
rx(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
cz node[78],node[77];
rz(0.5*pi) node[27];
rz(0.8477895035480554*pi) node[36];
rz(0.5*pi) node[64];
rx(0.5*pi) node[65];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rz(0.8477895035480554*pi) node[74];
rz(0.5*pi) node[77];
cz node[27],node[64];
rz(0.5*pi) node[36];
rz(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rx(0.5*pi) node[77];
rz(0.5*pi) node[27];
rx(0.5*pi) node[36];
rz(0.5*pi) node[64];
cz node[72],node[73];
rx(0.5*pi) node[74];
rz(0.5*pi) node[77];
rx(0.5*pi) node[27];
rz(0.5*pi) node[36];
rx(0.5*pi) node[64];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.8477895035480554*pi) node[77];
rz(0.5*pi) node[27];
cz node[37],node[36];
rz(0.5*pi) node[64];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
cz node[75],node[74];
rz(0.5*pi) node[77];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rx(3.377051009485837*pi) node[75];
rx(0.5*pi) node[77];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[64];
cz node[79],node[72];
rx(0.5*pi) node[74];
rz(0.5*pi) node[77];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[72];
rz(0.5*pi) node[74];
cz node[78],node[77];
cz node[26],node[37];
rx(3.377051009485837*pi) node[36];
cz node[65],node[64];
rx(0.5*pi) node[72];
rx(3.377051009485837*pi) node[74];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[72];
rz(0.5*pi) node[74];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[26];
rx(0.5*pi) node[37];
rx(0.5*pi) node[64];
rz(0.8477895035480554*pi) node[72];
rx(0.5*pi) node[74];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[72];
rz(0.5*pi) node[74];
rx(3.377051009485837*pi) node[77];
cz node[37],node[26];
rz(0.8477895035480554*pi) node[64];
rx(0.5*pi) node[72];
cz node[73],node[74];
rz(0.5*pi) node[77];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[72];
rz(0.5*pi) node[74];
rx(0.5*pi) node[77];
rx(0.5*pi) node[26];
rx(0.5*pi) node[37];
rx(0.5*pi) node[64];
cz node[79],node[72];
rx(0.5*pi) node[74];
rz(0.5*pi) node[77];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[72];
rz(0.5*pi) node[74];
cz node[76],node[77];
rz(0.5*pi) node[79];
cz node[26],node[37];
cz node[65],node[64];
rx(0.5*pi) node[72];
rz(3.9090454112978845*pi) node[74];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rx(0.5*pi) node[79];
cz node[26],node[27];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rx(3.377051009485837*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[74];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rz(0.5*pi) node[79];
rz(0.5*pi) node[27];
cz node[36],node[79];
rx(0.5*pi) node[37];
rx(0.5*pi) node[64];
cz node[65],node[78];
rx(1.3770510094858368*pi) node[72];
rx(0.5*pi) node[74];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rx(0.5*pi) node[27];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[74];
cz node[77],node[76];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[27];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rx(0.5*pi) node[65];
cz node[73],node[74];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.8477895035480554*pi) node[27];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[27];
cz node[79],node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
cz node[78],node[65];
rx(0.5*pi) node[73];
rx(0.5*pi) node[74];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rx(0.5*pi) node[27];
rz(0.5*pi) node[36];
rz(0.5*pi) node[65];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
cz node[76],node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[27];
rx(0.5*pi) node[36];
rx(0.5*pi) node[65];
rz(0.5*pi) node[74];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
cz node[26],node[27];
rz(0.5*pi) node[36];
rz(0.5*pi) node[65];
rx(0.5*pi) node[74];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(3.377051009485837*pi) node[26];
rz(0.5*pi) node[27];
cz node[36],node[79];
cz node[65],node[78];
rz(0.5*pi) node[74];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rx(0.5*pi) node[27];
cz node[36],node[37];
cz node[65],node[64];
cz node[75],node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[27];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[76];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rx(1.3770510094858368*pi) node[27];
rx(0.5*pi) node[37];
rx(0.5*pi) node[64];
rx(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
cz node[66],node[77];
rz(0.5*pi) node[76];
rz(0.5*pi) node[79];
rz(0.8477895035480554*pi) node[37];
rz(0.8477895035480554*pi) node[64];
rz(3.9090454112978845*pi) node[76];
rz(0.5*pi) node[77];
rx(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[76];
rx(0.5*pi) node[77];
rz(0.5*pi) node[79];
rx(0.5*pi) node[37];
rx(0.5*pi) node[64];
rx(0.5*pi) node[76];
rz(0.5*pi) node[77];
cz node[78],node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[76];
rz(0.8477895035480554*pi) node[77];
rz(0.5*pi) node[79];
cz node[36],node[37];
cz node[65],node[64];
cz node[75],node[76];
rz(0.5*pi) node[77];
rx(0.5*pi) node[79];
rx(3.377051009485837*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rx(3.377051009485837*pi) node[65];
cz node[75],node[74];
rz(0.5*pi) node[76];
rx(0.5*pi) node[77];
rz(0.5*pi) node[79];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[74];
rx(0.5*pi) node[76];
rz(0.5*pi) node[77];
rz(3.9090454112978845*pi) node[79];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rx(0.5*pi) node[65];
cz node[66],node[77];
rx(0.5*pi) node[74];
rz(0.5*pi) node[76];
rz(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(3.377051009485837*pi) node[64];
rz(0.5*pi) node[65];
cz node[66],node[67];
rz(0.5*pi) node[74];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
rx(0.5*pi) node[79];
rx(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[67];
rz(3.9090454112978845*pi) node[74];
rx(0.5*pi) node[76];
rx(0.5*pi) node[77];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rx(0.5*pi) node[64];
rx(0.5*pi) node[67];
rz(0.5*pi) node[74];
rz(0.5*pi) node[76];
rz(0.5*pi) node[77];
cz node[78],node[79];
cz node[38],node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[67];
rx(0.5*pi) node[74];
rx(3.377051009485837*pi) node[77];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.8477895035480554*pi) node[67];
rz(0.5*pi) node[74];
rz(0.5*pi) node[77];
rx(0.5*pi) node[79];
rx(0.5*pi) node[37];
rz(0.5*pi) node[67];
cz node[75],node[74];
rx(0.5*pi) node[77];
rz(0.5*pi) node[79];
cz node[79],node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[67];
rz(0.5*pi) node[74];
rx(1.0056883931475766*pi) node[75];
rz(0.5*pi) node[77];
rz(0.5*pi) node[36];
rz(0.8477895035480554*pi) node[37];
rz(0.5*pi) node[67];
rx(0.5*pi) node[74];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[66],node[67];
rz(0.5*pi) node[74];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(3.377051009485837*pi) node[66];
rz(0.5*pi) node[67];
rx(3.0056883931475773*pi) node[74];
rz(0.5*pi) node[79];
cz node[36],node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[66];
rx(0.5*pi) node[67];
rz(0.5*pi) node[36];
cz node[38],node[37];
rx(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[66];
rx(3.377051009485837*pi) node[67];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
cz node[67],node[66];
rz(0.5*pi) node[79];
cz node[79],node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[36];
rx(3.377051009485837*pi) node[37];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
cz node[66],node[67];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
cz node[72],node[79];
cz node[38],node[37];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
cz node[67],node[66];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[66];
cz node[79],node[72];
cz node[37],node[38];
rx(0.5*pi) node[66];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[66];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
cz node[66],node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
cz node[72],node[79];
cz node[38],node[37];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rx(0.5*pi) node[37];
cz node[65],node[66];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[37];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rx(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[37];
cz node[66],node[65];
cz node[36],node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[65];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[65],node[64];
cz node[37],node[36];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[64],node[65];
cz node[36],node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[65],node[64];
cz node[35],node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
cz node[26],node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[64];
cz node[78],node[65];
rz(0.8477895035480554*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[64];
rx(0.5*pi) node[65];
cz node[27],node[64];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rz(0.5*pi) node[36];
rz(3.9090454112978845*pi) node[37];
rz(0.5*pi) node[64];
rz(3.9090454112978845*pi) node[65];
cz node[35],node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[64];
rz(0.5*pi) node[65];
rx(3.377051009485837*pi) node[35];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[64];
rx(0.5*pi) node[65];
cz node[35],node[72];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(3.9090454112978845*pi) node[64];
rz(0.5*pi) node[65];
cz node[26],node[37];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[64];
cz node[78],node[65];
rz(0.5*pi) node[72];
rx(0.5*pi) node[35];
rx(1.3770510094858368*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[64];
rz(0.5*pi) node[65];
rx(0.5*pi) node[72];
rx(1.0056883931475766*pi) node[78];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[64];
rx(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[78];
cz node[27],node[64];
cz node[72],node[35];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[27];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rx(3.0056883931475773*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[78];
rx(0.5*pi) node[27];
rx(0.5*pi) node[35];
rz(0.5*pi) node[37];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rx(0.5*pi) node[72];
cz node[79],node[78];
rz(0.5*pi) node[27];
rz(0.5*pi) node[35];
rx(0.5*pi) node[37];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[72];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[26],node[27];
cz node[35],node[72];
rz(0.5*pi) node[37];
rz(0.5*pi) node[64];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[27];
cz node[35],node[36];
cz node[38],node[37];
rx(0.5*pi) node[64];
rz(0.5*pi) node[72];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[27];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[64];
rx(0.5*pi) node[72];
cz node[78],node[79];
rz(0.5*pi) node[27];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(3.9090454112978845*pi) node[27];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
cz node[72],node[73];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[27];
cz node[36],node[35];
cz node[37],node[38];
rz(0.5*pi) node[73];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[27];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rx(0.5*pi) node[73];
cz node[79],node[78];
rz(0.5*pi) node[27];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rz(0.5*pi) node[73];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[26],node[27];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(3.9090454112978845*pi) node[73];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rx(1.0056883931475766*pi) node[26];
rz(0.5*pi) node[27];
cz node[35],node[36];
cz node[38],node[37];
rz(0.5*pi) node[73];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[27];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[73];
cz node[78],node[77];
rz(0.5*pi) node[27];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[73];
rz(0.5*pi) node[77];
rx(3.0056883931475773*pi) node[27];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[72],node[73];
rx(0.5*pi) node[77];
rz(0.5*pi) node[37];
rz(0.5*pi) node[73];
rz(0.5*pi) node[77];
rx(0.5*pi) node[37];
rx(0.5*pi) node[73];
rz(3.9090454112978845*pi) node[77];
rz(0.5*pi) node[37];
rz(0.5*pi) node[73];
rz(0.5*pi) node[77];
rx(3.0056883931475773*pi) node[73];
rx(0.5*pi) node[77];
rz(0.5*pi) node[77];
cz node[78],node[77];
rz(0.5*pi) node[77];
cz node[78],node[79];
rx(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[77];
cz node[79],node[78];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[66],node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[66];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[66];
rx(0.5*pi) node[77];
cz node[78],node[79];
rz(0.5*pi) node[66];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
cz node[77],node[66];
rx(0.5*pi) node[79];
rz(0.5*pi) node[66];
rz(0.5*pi) node[77];
rz(0.5*pi) node[79];
rx(0.5*pi) node[66];
rx(0.5*pi) node[77];
rz(0.5*pi) node[79];
rz(0.5*pi) node[66];
rz(0.5*pi) node[77];
rx(0.5*pi) node[79];
cz node[66],node[77];
rz(0.5*pi) node[79];
cz node[36],node[79];
rz(0.5*pi) node[66];
rz(0.5*pi) node[77];
rx(0.5*pi) node[66];
rx(0.5*pi) node[77];
rz(0.5*pi) node[79];
rz(0.5*pi) node[66];
rz(0.5*pi) node[77];
rx(0.5*pi) node[79];
cz node[67],node[66];
cz node[77],node[76];
rz(0.5*pi) node[79];
rz(0.5*pi) node[66];
rz(0.5*pi) node[76];
rz(3.9090454112978845*pi) node[79];
rx(0.5*pi) node[66];
rx(0.5*pi) node[76];
rz(0.5*pi) node[79];
rz(0.5*pi) node[66];
rz(0.5*pi) node[76];
rx(0.5*pi) node[79];
rz(3.9090454112978845*pi) node[66];
rz(3.9090454112978845*pi) node[76];
rz(0.5*pi) node[79];
cz node[36],node[79];
rz(0.5*pi) node[66];
rz(0.5*pi) node[76];
cz node[36],node[37];
rx(0.5*pi) node[66];
rx(0.5*pi) node[76];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[66];
rz(0.5*pi) node[76];
rx(0.5*pi) node[79];
rx(0.5*pi) node[37];
cz node[67],node[66];
cz node[77],node[76];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[66];
rz(0.5*pi) node[76];
rx(3.0056883931475773*pi) node[79];
rz(3.9090454112978845*pi) node[37];
rx(0.5*pi) node[66];
rx(0.5*pi) node[76];
rz(0.5*pi) node[37];
rz(0.5*pi) node[66];
rz(0.5*pi) node[76];
rx(0.5*pi) node[37];
rx(3.0056883931475773*pi) node[66];
rx(3.0056883931475773*pi) node[76];
rz(0.5*pi) node[37];
rz(0.5*pi) node[66];
cz node[36],node[37];
rx(0.5*pi) node[66];
rx(1.0056883931475766*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[66];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
cz node[67],node[66];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[36];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
cz node[37],node[36];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[66],node[67];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[66];
rx(0.5*pi) node[67];
cz node[36],node[37];
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[67],node[66];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[66];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[66];
cz node[37],node[36];
rz(0.5*pi) node[66];
rz(0.5*pi) node[36];
cz node[66],node[65];
rx(0.5*pi) node[36];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[36];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rz(0.5*pi) node[36];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rx(0.5*pi) node[36];
cz node[65],node[66];
rz(0.5*pi) node[36];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
cz node[35],node[36];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rz(0.5*pi) node[36];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rx(0.5*pi) node[36];
cz node[66],node[65];
rz(0.5*pi) node[36];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(3.9090454112978845*pi) node[36];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rz(0.5*pi) node[36];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rx(0.5*pi) node[36];
cz node[65],node[64];
cz node[77],node[66];
rz(0.5*pi) node[36];
rz(0.5*pi) node[64];
rz(0.5*pi) node[66];
cz node[35],node[36];
rx(0.5*pi) node[64];
rx(0.5*pi) node[66];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[64];
rz(0.5*pi) node[66];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rz(3.9090454112978845*pi) node[64];
rz(3.9090454112978845*pi) node[66];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[64];
rz(0.5*pi) node[66];
cz node[72],node[35];
rx(3.0056883931475773*pi) node[36];
rx(0.5*pi) node[64];
rx(0.5*pi) node[66];
rz(0.5*pi) node[35];
rz(0.5*pi) node[64];
rz(0.5*pi) node[66];
rx(0.5*pi) node[35];
cz node[65],node[64];
cz node[77],node[66];
rz(0.5*pi) node[35];
rz(0.5*pi) node[64];
rx(1.0056883931475766*pi) node[65];
rz(0.5*pi) node[66];
rx(1.0056883931475766*pi) node[77];
rz(3.9090454112978845*pi) node[35];
rx(0.5*pi) node[64];
rx(0.5*pi) node[66];
rz(0.5*pi) node[35];
rz(0.5*pi) node[64];
rz(0.5*pi) node[66];
rx(0.5*pi) node[35];
rx(3.0056883931475773*pi) node[64];
rx(3.0056883931475773*pi) node[66];
rz(0.5*pi) node[35];
cz node[72],node[35];
rz(0.5*pi) node[35];
rx(1.0056883931475766*pi) node[72];
rx(0.5*pi) node[35];
rz(0.5*pi) node[35];
rx(3.0056883931475773*pi) node[35];
barrier node[72],node[65],node[77],node[35],node[37],node[75],node[36],node[26],node[73],node[27],node[64],node[79],node[78],node[66],node[74],node[67],node[76],node[38];
measure node[72] -> meas[0];
measure node[65] -> meas[1];
measure node[77] -> meas[2];
measure node[35] -> meas[3];
measure node[37] -> meas[4];
measure node[75] -> meas[5];
measure node[36] -> meas[6];
measure node[26] -> meas[7];
measure node[73] -> meas[8];
measure node[27] -> meas[9];
measure node[64] -> meas[10];
measure node[79] -> meas[11];
measure node[78] -> meas[12];
measure node[66] -> meas[13];
measure node[74] -> meas[14];
measure node[67] -> meas[15];
measure node[76] -> meas[16];
measure node[38] -> meas[17];
