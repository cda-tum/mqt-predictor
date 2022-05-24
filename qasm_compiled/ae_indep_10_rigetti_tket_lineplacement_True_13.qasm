OPENQASM 2.0;
include "qelib1.inc";

qreg node[80];
creg meas[10];
rz(3.5*pi) node[25];
rz(3.5*pi) node[26];
rz(3.5*pi) node[27];
rz(3.5*pi) node[35];
rz(3.5*pi) node[36];
rz(3.5*pi) node[37];
rz(3.5*pi) node[38];
rz(3.5*pi) node[72];
rz(3.5*pi) node[78];
rz(3.5*pi) node[79];
rx(1.5*pi) node[25];
rx(3.5*pi) node[26];
rx(3.5*pi) node[27];
rx(3.5*pi) node[35];
rx(3.5*pi) node[36];
rx(3.5*pi) node[37];
rx(3.5*pi) node[38];
rx(3.5*pi) node[72];
rx(3.5*pi) node[78];
rx(2.2951672359369732*pi) node[79];
rz(3.5624999999999996*pi) node[26];
rz(3.75*pi) node[27];
rz(3.5312499999999996*pi) node[35];
rz(3.5039062499999996*pi) node[36];
rz(3.5156249999999996*pi) node[37];
rz(3.6249999999999996*pi) node[38];
rz(3.5078124999999996*pi) node[72];
rz(3.5019531249999996*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
cz node[78],node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
rz(3.5*pi) node[79];
rx(3.7048327640630268*pi) node[79];
rz(0.5*pi) node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
cz node[78],node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
rz(3.5*pi) node[79];
rx(0.29516723593697364*pi) node[79];
rz(0.5*pi) node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
cz node[36],node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
rz(3.5*pi) node[79];
rx(3.409665540858449*pi) node[79];
rz(0.5*pi) node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
cz node[36],node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
rz(3.5*pi) node[79];
rx(0.5903344591415508*pi) node[79];
rz(0.5*pi) node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
cz node[72],node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
rz(3.5*pi) node[79];
rx(2.8193310498859097*pi) node[79];
rz(0.5*pi) node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
cz node[72],node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
rz(3.5*pi) node[79];
rx(1.1806689523994232*pi) node[79];
rz(0.5*pi) node[79];
cz node[79],node[36];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
cz node[36],node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
cz node[79],node[36];
rz(0.5*pi) node[36];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
rz(0.5*pi) node[36];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
cz node[37],node[36];
rz(0.5*pi) node[36];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
rz(3.5*pi) node[36];
rx(1.638662131602808*pi) node[36];
rz(0.5*pi) node[36];
rz(0.5*pi) node[36];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
cz node[37],node[36];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(3.5*pi) node[36];
rx(0.36133787068252504*pi) node[36];
rz(0.5*pi) node[36];
rz(0.5*pi) node[36];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
cz node[35],node[36];
rz(0.5*pi) node[36];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
rz(3.5*pi) node[36];
rx(3.2773243905295706*pi) node[36];
rz(0.5*pi) node[36];
rz(0.5*pi) node[36];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
cz node[35],node[36];
rz(0.5*pi) node[36];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
rz(3.5*pi) node[36];
rx(0.7226757731960389*pi) node[36];
rz(0.5*pi) node[36];
cz node[36],node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[37],node[36];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[36],node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[35],node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[26],node[37];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
cz node[36],node[35];
rz(0.5*pi) node[37];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(3.5*pi) node[37];
cz node[35],node[36];
rx(2.5546484627492543*pi) node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[37];
cz node[26],node[37];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rx(0.5*pi) node[26];
rx(0.5*pi) node[37];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(3.5*pi) node[37];
rx(1.445351516846423*pi) node[37];
rz(0.5*pi) node[37];
rz(0.5*pi) node[37];
rx(0.5*pi) node[37];
rz(0.5*pi) node[37];
cz node[38],node[37];
rz(0.5*pi) node[37];
rx(0.5*pi) node[37];
rz(0.5*pi) node[37];
rz(3.5*pi) node[37];
rx(1.1092969254985086*pi) node[37];
rz(0.5*pi) node[37];
rz(0.5*pi) node[37];
rx(0.5*pi) node[37];
rz(0.5*pi) node[37];
cz node[38],node[37];
rz(0.5*pi) node[37];
rx(0.5*pi) node[37];
rz(0.5*pi) node[37];
rz(3.5*pi) node[37];
rx(0.8907030632385018*pi) node[37];
rz(0.5*pi) node[37];
cz node[37],node[26];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rx(0.5*pi) node[26];
rx(0.5*pi) node[37];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
cz node[26],node[37];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
rx(0.5*pi) node[26];
rx(0.5*pi) node[37];
rz(0.5*pi) node[26];
rz(0.5*pi) node[37];
cz node[37],node[26];
rz(0.5*pi) node[26];
rx(0.5*pi) node[26];
rz(0.5*pi) node[26];
rz(0.5*pi) node[26];
rx(0.5*pi) node[26];
rz(0.5*pi) node[26];
cz node[27],node[26];
rz(0.5*pi) node[26];
rx(0.5*pi) node[26];
rz(0.5*pi) node[26];
rz(3.5*pi) node[26];
rx(2.2185932143772433*pi) node[26];
rz(0.5*pi) node[26];
rz(0.5*pi) node[26];
rx(0.5*pi) node[26];
rz(0.5*pi) node[26];
cz node[27],node[26];
rz(0.5*pi) node[26];
rx(0.5*pi) node[26];
rz(0.5*pi) node[26];
rz(3.5*pi) node[26];
rx(1.781406119213039*pi) node[26];
rz(0.5*pi) node[26];
rz(0.5*pi) node[26];
rx(0.5*pi) node[26];
rz(0.5*pi) node[26];
cz node[25],node[26];
rz(0.5*pi) node[26];
rx(0.5*pi) node[26];
rz(0.5*pi) node[26];
rz(3.5*pi) node[26];
rx(0.43718642875448716*pi) node[26];
rz(0.5*pi) node[26];
rz(0.5*pi) node[26];
rx(0.5*pi) node[26];
rz(0.5*pi) node[26];
cz node[25],node[26];
rx(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[25];
rx(0.5*pi) node[26];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rx(0.5*pi) node[25];
rz(3.5*pi) node[26];
rz(0.5*pi) node[25];
rx(1.5628135712455133*pi) node[26];
rz(0.5*pi) node[26];
rz(0.5*pi) node[26];
rx(0.5*pi) node[26];
rz(0.5*pi) node[26];
cz node[27],node[26];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rx(0.5*pi) node[26];
rx(0.5*pi) node[27];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
cz node[26],node[27];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rx(0.5*pi) node[26];
rx(0.5*pi) node[27];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
cz node[27],node[26];
rz(0.5*pi) node[26];
rx(0.5*pi) node[26];
rz(0.5*pi) node[26];
cz node[26],node[25];
rz(0.5*pi) node[25];
rx(0.5*pi) node[25];
rz(0.5*pi) node[25];
rz(0.25*pi) node[25];
rz(0.5*pi) node[25];
rx(0.5*pi) node[25];
rz(0.5*pi) node[25];
cz node[26],node[25];
rz(0.5*pi) node[25];
rx(0.5*pi) node[26];
rx(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(3.75*pi) node[25];
rx(0.5*pi) node[26];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rx(0.5*pi) node[25];
rz(0.5*pi) node[25];
cz node[38],node[25];
rz(0.5*pi) node[25];
rx(0.5*pi) node[25];
rz(0.5*pi) node[25];
rz(0.1250000000000001*pi) node[25];
rz(0.5*pi) node[25];
rx(0.5*pi) node[25];
rz(0.5*pi) node[25];
cz node[38],node[25];
rz(0.5*pi) node[25];
rz(0.5*pi) node[38];
rx(0.5*pi) node[25];
rx(0.5*pi) node[38];
rz(0.5*pi) node[25];
rz(0.5*pi) node[38];
rz(3.8750000000000004*pi) node[25];
cz node[25],node[38];
rz(0.5*pi) node[25];
rz(0.5*pi) node[38];
rx(0.5*pi) node[25];
rx(0.5*pi) node[38];
rz(0.5*pi) node[25];
rz(0.5*pi) node[38];
cz node[38],node[25];
rz(0.5*pi) node[25];
rz(0.5*pi) node[38];
rx(0.5*pi) node[25];
rx(0.5*pi) node[38];
rz(0.5*pi) node[25];
rz(0.5*pi) node[38];
cz node[25],node[38];
cz node[25],node[26];
rz(0.5*pi) node[38];
rz(0.5*pi) node[26];
rx(0.5*pi) node[38];
rx(0.5*pi) node[26];
rz(0.5*pi) node[38];
rz(0.5*pi) node[26];
rz(0.5*pi) node[38];
rz(0.25*pi) node[26];
rx(0.5*pi) node[38];
rz(0.5*pi) node[26];
rz(0.5*pi) node[38];
rx(0.5*pi) node[26];
cz node[37],node[38];
rz(0.5*pi) node[26];
rz(0.5*pi) node[38];
cz node[25],node[26];
rx(0.5*pi) node[38];
rx(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[38];
rz(0.5*pi) node[25];
rx(0.5*pi) node[26];
rz(0.06250000000000044*pi) node[38];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[38];
rx(0.5*pi) node[25];
rz(3.75*pi) node[26];
rx(0.5*pi) node[38];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[38];
rx(0.5*pi) node[26];
cz node[37],node[38];
rz(0.5*pi) node[26];
rz(0.5*pi) node[38];
cz node[37],node[26];
rx(0.5*pi) node[38];
rz(0.5*pi) node[26];
rz(0.5*pi) node[38];
rx(0.5*pi) node[26];
rz(3.9375*pi) node[38];
rz(0.5*pi) node[26];
rz(0.5*pi) node[38];
rz(0.1250000000000001*pi) node[26];
rx(0.5*pi) node[38];
rz(0.5*pi) node[26];
rz(0.5*pi) node[38];
rx(0.5*pi) node[26];
rz(0.5*pi) node[26];
cz node[37],node[26];
rz(0.5*pi) node[26];
cz node[37],node[38];
rx(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[26];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rz(3.8750000000000004*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[26];
cz node[38],node[37];
rx(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[26];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
cz node[37],node[38];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
cz node[38],node[25];
cz node[36],node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[37];
rx(0.5*pi) node[25];
rx(0.5*pi) node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[37];
rz(0.25*pi) node[25];
rz(0.031250000000000555*pi) node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[37];
rx(0.5*pi) node[25];
rx(0.5*pi) node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[37];
cz node[38],node[25];
cz node[36],node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[25];
rx(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[25];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(3.75*pi) node[25];
rz(3.96875*pi) node[37];
rx(0.5*pi) node[38];
rz(0.5*pi) node[25];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rx(0.5*pi) node[25];
rx(0.5*pi) node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[37];
cz node[36],node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[37],node[36];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[36],node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[37],node[26];
cz node[35],node[36];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rx(0.5*pi) node[26];
rx(0.5*pi) node[36];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.06250000000000044*pi) node[26];
rz(0.01562500000000011*pi) node[36];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rx(0.5*pi) node[26];
rx(0.5*pi) node[36];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
cz node[37],node[26];
cz node[35],node[36];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rx(0.5*pi) node[26];
rx(0.5*pi) node[36];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(3.9375*pi) node[26];
rz(3.984375*pi) node[36];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rx(0.5*pi) node[26];
rx(0.5*pi) node[36];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
cz node[37],node[26];
cz node[35],node[36];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[26];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[26],node[37];
cz node[36],node[35];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[26];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[37],node[26];
cz node[35],node[36];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[26];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[26],node[25];
cz node[72],node[35];
cz node[36],node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[35];
rz(0.5*pi) node[37];
rx(0.5*pi) node[25];
rx(0.5*pi) node[35];
rx(0.5*pi) node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[35];
rz(0.5*pi) node[37];
rz(0.1250000000000001*pi) node[25];
rz(0.007812500000000444*pi) node[35];
rz(0.031250000000000555*pi) node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[35];
rz(0.5*pi) node[37];
rx(0.5*pi) node[25];
rx(0.5*pi) node[35];
rx(0.5*pi) node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[35];
rz(0.5*pi) node[37];
cz node[26],node[25];
cz node[72],node[35];
cz node[36],node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[37];
rz(0.5*pi) node[72];
rx(0.5*pi) node[25];
rx(0.5*pi) node[26];
rx(0.5*pi) node[35];
rx(0.5*pi) node[37];
rx(0.5*pi) node[72];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[37];
rz(0.5*pi) node[72];
rz(3.8750000000000004*pi) node[25];
rz(3.9921875000000004*pi) node[35];
rz(3.96875*pi) node[37];
cz node[25],node[26];
cz node[35],node[72];
rz(0.5*pi) node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rx(0.5*pi) node[37];
rz(0.5*pi) node[72];
rx(0.5*pi) node[25];
rx(0.5*pi) node[26];
rx(0.5*pi) node[35];
rz(0.5*pi) node[37];
rx(0.5*pi) node[72];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
cz node[36],node[37];
rz(0.5*pi) node[72];
cz node[26],node[25];
cz node[72],node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[72];
rx(0.5*pi) node[25];
rx(0.5*pi) node[26];
rx(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[72];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
cz node[37],node[36];
rz(0.5*pi) node[72];
cz node[25],node[26];
cz node[35],node[72];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[25],node[38];
rz(0.5*pi) node[26];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[72];
rx(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rx(0.5*pi) node[72];
rz(0.5*pi) node[26];
cz node[36],node[37];
rx(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rx(0.5*pi) node[26];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.25*pi) node[38];
rx(0.5*pi) node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
cz node[37],node[26];
cz node[35],node[36];
rx(0.5*pi) node[38];
cz node[79],node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
cz node[25],node[38];
rx(0.5*pi) node[26];
rx(0.5*pi) node[36];
rx(0.5*pi) node[72];
rx(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[25];
rz(0.06250000000000044*pi) node[26];
rz(0.01562500000000011*pi) node[36];
rx(0.5*pi) node[38];
rz(0.003906250000000555*pi) node[72];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rx(0.5*pi) node[25];
rx(0.5*pi) node[26];
rx(0.5*pi) node[36];
rz(3.75*pi) node[38];
rx(0.5*pi) node[72];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
cz node[37],node[26];
cz node[35],node[36];
rx(0.5*pi) node[38];
cz node[79],node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rx(0.5*pi) node[26];
rx(0.5*pi) node[36];
cz node[37],node[38];
rx(0.5*pi) node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(3.9375*pi) node[26];
rz(3.984375*pi) node[36];
rx(0.5*pi) node[38];
rz(3.99609375*pi) node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rx(0.5*pi) node[26];
rx(0.5*pi) node[36];
rz(0.1250000000000001*pi) node[38];
rx(0.5*pi) node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
cz node[79],node[36];
rx(0.5*pi) node[38];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rx(0.5*pi) node[36];
cz node[37],node[38];
cz node[37],node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[26];
rz(0.007812500000000444*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[26];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[26];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(3.8750000000000004*pi) node[38];
cz node[26],node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[26];
cz node[79],node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[26];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[26];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[37],node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[26];
rz(3.9921875000000004*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[26];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[26];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[26],node[25];
rz(0.5*pi) node[36];
rz(0.5*pi) node[25];
cz node[35],node[36];
rx(0.5*pi) node[25];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[25];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rz(0.25*pi) node[25];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[25];
cz node[36],node[35];
rx(0.5*pi) node[25];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[25];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
cz node[26],node[25];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[25];
rx(0.5*pi) node[26];
cz node[35],node[36];
rx(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rx(0.5*pi) node[36];
rz(3.75*pi) node[25];
rx(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
cz node[36],node[37];
rx(0.5*pi) node[25];
rz(0.5*pi) node[37];
rz(0.5*pi) node[25];
rx(0.5*pi) node[37];
rz(0.5*pi) node[37];
rz(0.031250000000000555*pi) node[37];
rz(0.5*pi) node[37];
rx(0.5*pi) node[37];
rz(0.5*pi) node[37];
cz node[36],node[37];
rz(0.5*pi) node[37];
rx(0.5*pi) node[37];
rz(0.5*pi) node[37];
rz(3.96875*pi) node[37];
rz(0.5*pi) node[37];
rx(0.5*pi) node[37];
rz(0.5*pi) node[37];
cz node[36],node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[37],node[36];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[36],node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[79],node[36];
cz node[37],node[38];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rx(0.5*pi) node[36];
rx(0.5*pi) node[38];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.01562500000000011*pi) node[36];
rz(0.06250000000000044*pi) node[38];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rx(0.5*pi) node[36];
rx(0.5*pi) node[38];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
cz node[79],node[36];
cz node[37],node[38];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rx(0.5*pi) node[36];
rx(0.5*pi) node[38];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(3.984375*pi) node[36];
rz(3.9375*pi) node[38];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rx(0.5*pi) node[36];
rx(0.5*pi) node[38];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
cz node[79],node[36];
cz node[37],node[38];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[79];
cz node[36],node[79];
cz node[38],node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[79];
cz node[79],node[36];
cz node[37],node[38];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[79];
cz node[38],node[25];
cz node[36],node[37];
cz node[78],node[79];
rz(0.5*pi) node[25];
rz(0.5*pi) node[37];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[25];
rx(0.5*pi) node[37];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[25];
rz(0.5*pi) node[37];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.1250000000000001*pi) node[25];
rz(0.031250000000000555*pi) node[37];
cz node[79],node[78];
rz(0.5*pi) node[25];
rz(0.5*pi) node[37];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[25];
rx(0.5*pi) node[37];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[25];
rz(0.5*pi) node[37];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[38],node[25];
cz node[36],node[37];
cz node[78],node[79];
rz(0.5*pi) node[25];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[25];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[25];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(3.8750000000000004*pi) node[25];
rz(3.96875*pi) node[37];
cz node[79],node[72];
cz node[25],node[38];
rz(0.5*pi) node[37];
rz(0.5*pi) node[72];
rz(0.5*pi) node[25];
rx(0.5*pi) node[37];
rz(0.5*pi) node[38];
rx(0.5*pi) node[72];
rx(0.5*pi) node[25];
rz(0.5*pi) node[37];
rx(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[25];
cz node[36],node[37];
rz(0.5*pi) node[38];
rz(0.001953125000000333*pi) node[72];
cz node[38],node[25];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[72];
rz(0.5*pi) node[25];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[38];
rx(0.5*pi) node[72];
rx(0.5*pi) node[25];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[25];
cz node[37],node[36];
rz(0.5*pi) node[38];
cz node[79],node[72];
cz node[25],node[38];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[72];
cz node[25],node[26];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[38];
rx(0.5*pi) node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[38];
rz(0.5*pi) node[72];
rx(0.5*pi) node[26];
cz node[36],node[37];
rz(0.5*pi) node[38];
rz(1.998046875*pi) node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.25*pi) node[26];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rx(0.5*pi) node[26];
cz node[35],node[72];
cz node[37],node[38];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
cz node[25],node[26];
rx(0.5*pi) node[35];
rx(0.5*pi) node[38];
rx(0.5*pi) node[72];
rx(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[25];
rx(0.5*pi) node[26];
cz node[72],node[35];
rz(0.06250000000000044*pi) node[38];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rx(0.5*pi) node[25];
rz(3.75*pi) node[26];
rx(0.5*pi) node[35];
rx(0.5*pi) node[38];
rx(0.5*pi) node[72];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rx(0.5*pi) node[26];
cz node[35],node[72];
cz node[37],node[38];
rz(0.5*pi) node[26];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
cz node[37],node[26];
rx(0.5*pi) node[38];
rx(0.5*pi) node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rx(0.5*pi) node[26];
rz(3.9375*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[38];
rx(0.5*pi) node[72];
rz(0.1250000000000001*pi) node[26];
rx(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[38];
cz node[79],node[72];
rx(0.5*pi) node[26];
rz(0.5*pi) node[72];
rz(0.5*pi) node[26];
rx(0.5*pi) node[72];
cz node[37],node[26];
rz(0.5*pi) node[72];
rz(0.5*pi) node[26];
cz node[37],node[38];
rz(0.003906250000000555*pi) node[72];
rx(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[26];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[72];
rz(3.8750000000000004*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[26];
cz node[38],node[37];
cz node[79],node[72];
rx(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
cz node[79],node[78];
rz(0.5*pi) node[26];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[72];
rz(0.5*pi) node[78];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rx(0.5*pi) node[78];
cz node[37],node[38];
rz(1.99609375*pi) node[72];
rz(0.5*pi) node[78];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.007812500000000444*pi) node[78];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rz(0.5*pi) node[78];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rx(0.5*pi) node[78];
cz node[38],node[25];
rz(0.5*pi) node[78];
rz(0.5*pi) node[25];
cz node[79],node[78];
rx(0.5*pi) node[25];
cz node[79],node[36];
rz(0.5*pi) node[78];
rz(0.5*pi) node[25];
rz(0.5*pi) node[36];
rx(0.5*pi) node[78];
rz(0.25*pi) node[25];
rx(0.5*pi) node[36];
rz(0.5*pi) node[78];
rz(0.5*pi) node[25];
rz(0.5*pi) node[36];
rz(1.9921875*pi) node[78];
rx(0.5*pi) node[25];
rz(0.01562500000000011*pi) node[36];
rz(0.5*pi) node[25];
rz(0.5*pi) node[36];
cz node[38],node[25];
rx(0.5*pi) node[36];
rz(0.5*pi) node[25];
rz(0.5*pi) node[36];
rx(0.5*pi) node[38];
rx(0.5*pi) node[25];
cz node[79],node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[25];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(3.75*pi) node[25];
rx(0.5*pi) node[36];
rx(0.5*pi) node[38];
rz(0.5*pi) node[25];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rx(0.5*pi) node[25];
rz(1.984375*pi) node[36];
rz(0.5*pi) node[25];
rz(0.5*pi) node[36];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
cz node[79],node[36];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
cz node[36],node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
cz node[79],node[36];
rz(0.5*pi) node[36];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
cz node[36],node[37];
rz(0.5*pi) node[37];
rx(0.5*pi) node[37];
rz(0.5*pi) node[37];
rz(0.031250000000000555*pi) node[37];
rz(0.5*pi) node[37];
rx(0.5*pi) node[37];
rz(0.5*pi) node[37];
cz node[36],node[37];
rz(0.5*pi) node[37];
rx(0.5*pi) node[37];
rz(0.5*pi) node[37];
rz(1.96875*pi) node[37];
rz(0.5*pi) node[37];
rx(0.5*pi) node[37];
rz(0.5*pi) node[37];
cz node[36],node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[37],node[36];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[36],node[37];
rz(0.5*pi) node[37];
rx(0.5*pi) node[37];
rz(0.5*pi) node[37];
cz node[37],node[26];
rz(0.5*pi) node[26];
rx(0.5*pi) node[26];
rz(0.5*pi) node[26];
rz(0.06250000000000044*pi) node[26];
rz(0.5*pi) node[26];
rx(0.5*pi) node[26];
rz(0.5*pi) node[26];
cz node[37],node[26];
rz(0.5*pi) node[26];
cz node[37],node[38];
rx(0.5*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[26];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rz(1.9375*pi) node[26];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
cz node[38],node[37];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
cz node[37],node[38];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
cz node[38],node[25];
rz(0.5*pi) node[25];
rx(0.5*pi) node[25];
rz(0.5*pi) node[25];
rz(0.1250000000000001*pi) node[25];
rz(0.5*pi) node[25];
rx(0.5*pi) node[25];
rz(0.5*pi) node[25];
cz node[38],node[25];
rz(0.5*pi) node[25];
cz node[38],node[37];
rx(0.5*pi) node[25];
rz(0.5*pi) node[37];
rz(0.5*pi) node[25];
rx(0.5*pi) node[37];
rz(1.875*pi) node[25];
rz(0.5*pi) node[37];
rz(0.25*pi) node[37];
rz(0.5*pi) node[37];
rx(0.5*pi) node[37];
rz(0.5*pi) node[37];
cz node[38],node[37];
rz(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[37];
rz(1.75*pi) node[37];
barrier node[38],node[37],node[25],node[26],node[36],node[79],node[78],node[72],node[35],node[27];
measure node[38] -> meas[0];
measure node[37] -> meas[1];
measure node[25] -> meas[2];
measure node[26] -> meas[3];
measure node[36] -> meas[4];
measure node[79] -> meas[5];
measure node[78] -> meas[6];
measure node[72] -> meas[7];
measure node[35] -> meas[8];
measure node[27] -> meas[9];
