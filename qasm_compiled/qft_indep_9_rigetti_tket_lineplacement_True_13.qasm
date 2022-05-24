OPENQASM 2.0;
include "qelib1.inc";

qreg node[80];
creg c[9];
creg meas[9];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[78];
rz(3.5*pi) node[79];
rx(0.5*pi) node[26];
rx(0.5*pi) node[27];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[72];
rx(0.5*pi) node[78];
rx(3.5*pi) node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[78];
rz(3.9980468750000004*pi) node[79];
cz node[79],node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
rz(3.75*pi) node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[79],node[78];
cz node[79],node[36];
rz(0.5*pi) node[78];
rz(0.5*pi) node[36];
rx(0.5*pi) node[78];
rx(0.5*pi) node[36];
rz(0.5*pi) node[78];
rz(0.5*pi) node[36];
rz(0.75*pi) node[78];
rz(3.8750000000000004*pi) node[36];
rx(2.5*pi) node[78];
rz(0.5*pi) node[36];
rz(2.9960937500000004*pi) node[78];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
cz node[79],node[36];
rz(0.5*pi) node[36];
cz node[79],node[72];
rx(0.5*pi) node[36];
rz(0.5*pi) node[72];
rz(0.5*pi) node[36];
rx(0.5*pi) node[72];
rz(0.1250000000000001*pi) node[36];
rz(0.5*pi) node[72];
rz(3.9375*pi) node[72];
rz(0.5*pi) node[72];
rx(0.5*pi) node[72];
rz(0.5*pi) node[72];
cz node[79],node[72];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
cz node[36],node[79];
rz(0.06250000000000044*pi) node[72];
rz(0.5*pi) node[36];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
cz node[79],node[36];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
cz node[36],node[79];
cz node[36],node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rx(0.5*pi) node[79];
rx(0.5*pi) node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[79];
rz(3.96875*pi) node[37];
rx(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[79];
rx(0.5*pi) node[37];
cz node[78],node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[79];
cz node[36],node[37];
rx(0.5*pi) node[79];
cz node[36],node[35];
rz(0.5*pi) node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rx(0.5*pi) node[37];
rz(3.75*pi) node[79];
rx(0.5*pi) node[35];
rz(0.5*pi) node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.031250000000000555*pi) node[37];
rx(0.5*pi) node[79];
rz(3.984375*pi) node[35];
rz(0.5*pi) node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rx(0.5*pi) node[37];
cz node[78],node[79];
rx(0.5*pi) node[35];
rz(0.5*pi) node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rx(0.5*pi) node[79];
cz node[36],node[35];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
cz node[36],node[37];
rz(0.75*pi) node[79];
rx(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(2.5*pi) node[79];
rz(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(2.9921875000000004*pi) node[79];
rz(0.01562500000000011*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
cz node[37],node[36];
rx(0.5*pi) node[79];
rx(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
cz node[78],node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[36],node[37];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
cz node[79],node[78];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[37],node[26];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[26];
cz node[78],node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[79];
rz(3.9921875000000004*pi) node[26];
rx(0.5*pi) node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[79];
rx(0.5*pi) node[26];
cz node[79],node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[72];
cz node[37],node[26];
rx(0.5*pi) node[72];
rz(0.5*pi) node[26];
cz node[37],node[38];
rz(0.5*pi) node[72];
rx(0.5*pi) node[26];
rz(0.5*pi) node[38];
rz(3.8750000000000004*pi) node[72];
rz(0.5*pi) node[26];
rx(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.007812500000000444*pi) node[26];
rz(0.5*pi) node[38];
rx(0.5*pi) node[72];
rz(3.99609375*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[38];
cz node[79],node[72];
cz node[79],node[36];
rx(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rx(0.5*pi) node[72];
rx(0.5*pi) node[36];
cz node[37],node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.1250000000000001*pi) node[72];
rz(3.9375*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rx(0.5*pi) node[72];
cz node[26],node[37];
rx(0.5*pi) node[36];
rz(0.003906250000000555*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rx(0.5*pi) node[26];
cz node[79],node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
cz node[79],node[72];
cz node[37],node[26];
rx(0.5*pi) node[36];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rx(0.5*pi) node[26];
rz(0.06250000000000044*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[72],node[79];
cz node[26],node[37];
rx(0.5*pi) node[36];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
cz node[26],node[27];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rz(0.5*pi) node[27];
rx(0.5*pi) node[37];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rx(0.5*pi) node[27];
rz(0.5*pi) node[37];
cz node[79],node[72];
rz(0.5*pi) node[27];
rz(0.5*pi) node[37];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(3.9980468750000004*pi) node[27];
rx(0.5*pi) node[37];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rz(0.5*pi) node[27];
rz(0.5*pi) node[37];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rx(0.5*pi) node[27];
cz node[72],node[35];
cz node[78],node[79];
rz(0.5*pi) node[27];
rz(0.5*pi) node[35];
rz(0.5*pi) node[79];
cz node[26],node[27];
rx(0.5*pi) node[35];
rx(0.5*pi) node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[35];
rz(0.5*pi) node[79];
rx(0.5*pi) node[26];
rx(0.5*pi) node[27];
rz(3.96875*pi) node[35];
rz(3.75*pi) node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[35];
rz(0.5*pi) node[79];
rz(0.001953125000000333*pi) node[27];
rx(0.5*pi) node[35];
rx(0.5*pi) node[79];
cz node[27],node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
cz node[72],node[35];
cz node[78],node[79];
rx(0.5*pi) node[26];
rx(0.5*pi) node[27];
rz(0.5*pi) node[35];
rz(0.5*pi) node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rx(0.5*pi) node[35];
rx(0.5*pi) node[79];
cz node[26],node[27];
rz(0.5*pi) node[35];
rz(0.5*pi) node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.031250000000000555*pi) node[35];
rz(0.75*pi) node[79];
rx(0.5*pi) node[26];
rx(0.5*pi) node[27];
rz(0.5*pi) node[35];
rx(2.5*pi) node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rx(0.5*pi) node[35];
rz(2.9843750000000004*pi) node[79];
cz node[27],node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[79];
rz(0.5*pi) node[26];
cz node[72],node[35];
rx(0.5*pi) node[79];
rx(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.5*pi) node[26];
rx(0.5*pi) node[35];
rx(0.5*pi) node[72];
cz node[78],node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[72];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[26];
cz node[35],node[72];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[72];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[35];
rx(0.5*pi) node[72];
cz node[79],node[78];
rz(0.5*pi) node[35];
rz(0.5*pi) node[72];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[72],node[35];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.5*pi) node[72];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[35];
rx(0.5*pi) node[72];
cz node[78],node[79];
rz(0.5*pi) node[35];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
cz node[79],node[36];
rz(0.5*pi) node[36];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
rz(3.8750000000000004*pi) node[36];
rz(0.5*pi) node[36];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
cz node[79],node[36];
rz(0.5*pi) node[36];
cz node[79],node[72];
rx(0.5*pi) node[36];
rz(0.5*pi) node[72];
rz(0.5*pi) node[36];
rx(0.5*pi) node[72];
rz(0.1250000000000001*pi) node[36];
rz(0.5*pi) node[72];
rz(3.9375*pi) node[72];
rz(0.5*pi) node[72];
rx(0.5*pi) node[72];
rz(0.5*pi) node[72];
cz node[79],node[72];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
cz node[36],node[79];
rz(0.06250000000000044*pi) node[72];
rz(0.5*pi) node[36];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[72];
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
cz node[35],node[36];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rx(0.5*pi) node[79];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
cz node[78],node[79];
cz node[36],node[35];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rx(0.5*pi) node[79];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(3.75*pi) node[79];
cz node[35],node[36];
rz(0.5*pi) node[79];
rz(0.5*pi) node[36];
rx(0.5*pi) node[79];
rx(0.5*pi) node[36];
rz(0.5*pi) node[79];
rz(0.5*pi) node[36];
cz node[78],node[79];
cz node[36],node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rx(0.5*pi) node[79];
rx(0.5*pi) node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.75*pi) node[79];
rz(3.984375*pi) node[37];
rx(2.5*pi) node[79];
rz(0.5*pi) node[37];
rz(0.9687500000000004*pi) node[79];
rx(0.5*pi) node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[37];
rx(0.5*pi) node[79];
cz node[36],node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
cz node[78],node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.01562500000000011*pi) node[37];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[37],node[36];
cz node[79],node[78];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[36],node[37];
cz node[78],node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[79];
cz node[37],node[36];
cz node[79],node[72];
rz(0.5*pi) node[36];
cz node[37],node[38];
rz(0.5*pi) node[72];
rx(0.5*pi) node[36];
rz(0.5*pi) node[38];
rx(0.5*pi) node[72];
rz(0.5*pi) node[36];
rx(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(3.8750000000000004*pi) node[72];
rx(0.5*pi) node[36];
rz(3.9921875000000004*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rx(0.5*pi) node[72];
cz node[35],node[36];
rx(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
cz node[79],node[72];
rx(0.5*pi) node[36];
cz node[37],node[38];
rz(0.5*pi) node[72];
cz node[37],node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rx(0.5*pi) node[72];
rz(0.5*pi) node[26];
rz(3.96875*pi) node[36];
rx(0.5*pi) node[38];
rz(0.5*pi) node[72];
rx(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[38];
rz(0.1250000000000001*pi) node[72];
rz(0.5*pi) node[26];
rx(0.5*pi) node[36];
rz(0.007812500000000444*pi) node[38];
rz(0.5*pi) node[72];
rz(3.99609375*pi) node[26];
rz(0.5*pi) node[36];
rx(0.5*pi) node[72];
rz(0.5*pi) node[26];
cz node[35],node[36];
rz(0.5*pi) node[72];
rx(0.5*pi) node[26];
rz(0.5*pi) node[36];
cz node[79],node[72];
rz(0.5*pi) node[26];
rx(0.5*pi) node[36];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
cz node[37],node[26];
rz(0.5*pi) node[36];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rz(0.5*pi) node[26];
rz(0.031250000000000555*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rx(0.5*pi) node[26];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
cz node[72],node[79];
rz(0.5*pi) node[26];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.003906250000000555*pi) node[26];
rz(0.5*pi) node[36];
cz node[38],node[37];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rz(0.5*pi) node[26];
cz node[35],node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rx(0.5*pi) node[26];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
cz node[79],node[72];
rz(0.5*pi) node[26];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
cz node[37],node[38];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
cz node[36],node[35];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
cz node[78],node[79];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
cz node[38],node[37];
rx(0.5*pi) node[79];
cz node[35],node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(3.75*pi) node[79];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[79];
cz node[72],node[35];
rx(0.5*pi) node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.5*pi) node[37];
cz node[78],node[79];
rx(0.5*pi) node[35];
cz node[36],node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.5*pi) node[37];
rx(0.5*pi) node[79];
rz(3.9375*pi) node[35];
rx(0.5*pi) node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.5*pi) node[37];
rz(0.75*pi) node[79];
rx(0.5*pi) node[35];
rz(3.984375*pi) node[37];
rx(2.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.5*pi) node[37];
rz(2.9375000000000004*pi) node[79];
cz node[72],node[35];
rx(0.5*pi) node[37];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.5*pi) node[37];
rz(0.5*pi) node[72];
rx(0.5*pi) node[79];
rx(0.5*pi) node[35];
cz node[36],node[37];
rx(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.5*pi) node[37];
rz(0.5*pi) node[72];
cz node[78],node[79];
rz(0.06250000000000044*pi) node[35];
rx(0.5*pi) node[37];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[35],node[72];
rz(0.5*pi) node[37];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[35];
rz(0.01562500000000011*pi) node[37];
rz(0.5*pi) node[72];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[35];
rz(0.5*pi) node[37];
rx(0.5*pi) node[72];
cz node[79],node[78];
rz(0.5*pi) node[35];
rx(0.5*pi) node[37];
rz(0.5*pi) node[72];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[72],node[35];
rz(0.5*pi) node[37];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[35];
cz node[36],node[37];
rz(0.5*pi) node[72];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[72];
cz node[78],node[79];
rz(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
cz node[35],node[72];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[79];
cz node[37],node[36];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[72];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[72];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[72];
cz node[36],node[37];
rx(0.5*pi) node[72];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[72];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
cz node[79],node[72];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[72];
cz node[37],node[26];
cz node[35],node[36];
rx(0.5*pi) node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[72];
rx(0.5*pi) node[26];
rx(0.5*pi) node[36];
rz(3.8750000000000004*pi) node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[72];
rz(3.9921875000000004*pi) node[26];
rz(3.96875*pi) node[36];
rx(0.5*pi) node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[72];
rx(0.5*pi) node[26];
rx(0.5*pi) node[36];
cz node[79],node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[72];
cz node[37],node[26];
cz node[35],node[36];
rx(0.5*pi) node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[72];
rx(0.5*pi) node[26];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.1250000000000001*pi) node[72];
rz(0.5*pi) node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[72];
rz(0.007812500000000444*pi) node[26];
rz(0.031250000000000555*pi) node[36];
rx(0.5*pi) node[72];
cz node[26],node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[72];
rz(0.5*pi) node[26];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[26];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[26];
cz node[79],node[36];
rz(0.5*pi) node[37];
cz node[37],node[26];
rz(0.5*pi) node[36];
rz(0.5*pi) node[26];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[26];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[26];
rz(3.9375*pi) node[36];
rz(0.5*pi) node[37];
cz node[26],node[37];
rz(0.5*pi) node[36];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
cz node[79],node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[79];
rz(0.06250000000000044*pi) node[36];
cz node[78],node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
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
rx(0.5*pi) node[72];
rz(0.5*pi) node[72];
rz(3.75*pi) node[72];
rz(0.5*pi) node[72];
rx(0.5*pi) node[72];
rz(0.5*pi) node[72];
cz node[79],node[72];
cz node[79],node[36];
rz(0.5*pi) node[72];
rz(0.5*pi) node[36];
rx(0.5*pi) node[72];
rx(0.5*pi) node[36];
rz(0.5*pi) node[72];
rz(0.5*pi) node[36];
rz(0.75*pi) node[72];
rz(3.8750000000000004*pi) node[36];
rx(2.5*pi) node[72];
rz(0.5*pi) node[36];
rz(2.8750000000000004*pi) node[72];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
cz node[79],node[36];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
rz(0.1250000000000001*pi) node[36];
rz(0.5*pi) node[36];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
cz node[35],node[36];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
cz node[36],node[35];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
cz node[35],node[36];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
cz node[72],node[35];
cz node[36],node[37];
rz(0.5*pi) node[35];
rz(0.5*pi) node[37];
rx(0.5*pi) node[35];
rx(0.5*pi) node[37];
rz(0.5*pi) node[35];
rz(0.5*pi) node[37];
rz(3.75*pi) node[35];
rz(3.984375*pi) node[37];
rz(0.5*pi) node[35];
rz(0.5*pi) node[37];
rx(0.5*pi) node[35];
rx(0.5*pi) node[37];
rz(0.5*pi) node[35];
rz(0.5*pi) node[37];
cz node[72],node[35];
cz node[36],node[37];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.75*pi) node[35];
rz(0.01562500000000011*pi) node[37];
rx(2.5*pi) node[35];
cz node[37],node[36];
rz(2.75*pi) node[35];
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
cz node[37],node[36];
rz(0.5*pi) node[36];
rx(0.5*pi) node[36];
rz(0.5*pi) node[36];
cz node[36],node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
cz node[79],node[36];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(0.5*pi) node[79];
cz node[36],node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
cz node[78],node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
rz(3.96875*pi) node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
cz node[78],node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
rz(0.031250000000000555*pi) node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
cz node[36],node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
rz(3.9375*pi) node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
cz node[36],node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
rz(0.06250000000000044*pi) node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
cz node[72],node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
rz(3.8750000000000004*pi) node[79];
rz(0.5*pi) node[79];
rx(0.5*pi) node[79];
rz(0.5*pi) node[79];
cz node[72],node[79];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rx(0.5*pi) node[72];
rx(0.5*pi) node[79];
rz(0.5*pi) node[72];
rz(0.5*pi) node[79];
rz(0.1250000000000001*pi) node[79];
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
cz node[35],node[72];
rz(0.5*pi) node[72];
rx(0.5*pi) node[72];
rz(0.5*pi) node[72];
rz(3.75*pi) node[72];
rz(0.5*pi) node[72];
rx(0.5*pi) node[72];
rz(0.5*pi) node[72];
cz node[35],node[72];
rz(0.5*pi) node[72];
rx(0.5*pi) node[72];
rz(0.5*pi) node[72];
rz(0.75*pi) node[72];
rx(2.5*pi) node[72];
rz(0.5*pi) node[72];
barrier node[27],node[38],node[26],node[37],node[78],node[36],node[79],node[35],node[72];
measure node[27] -> meas[0];
measure node[38] -> meas[1];
measure node[26] -> meas[2];
measure node[37] -> meas[3];
measure node[78] -> meas[4];
measure node[36] -> meas[5];
measure node[79] -> meas[6];
measure node[35] -> meas[7];
measure node[72] -> meas[8];
