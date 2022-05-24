OPENQASM 2.0;
include "qelib1.inc";

qreg node[16];
creg c[8];
rz(0.5*pi) node[0];
rz(2.5703125027284566*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(1.5*pi) node[15];
rx(0.5*pi) node[0];
rx(1.0*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[7];
rx(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[0];
rx(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
cz node[1],node[0];
rz(0.5*pi) node[0];
rx(0.5*pi) node[0];
rz(0.5*pi) node[0];
rz(0.4296875021201464*pi) node[0];
rz(0.5*pi) node[0];
rx(0.5*pi) node[0];
rz(0.5*pi) node[0];
cz node[1],node[0];
rz(0.5*pi) node[0];
cz node[1],node[2];
rx(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rx(0.5*pi) node[2];
rz(3.574218747879854*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(3.8593750006083116*pi) node[2];
rx(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
rx(0.5*pi) node[2];
cz node[7],node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
cz node[1],node[2];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
cz node[1],node[14];
rz(0.5*pi) node[2];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rx(0.5*pi) node[2];
rz(0.5*pi) node[7];
rz(0.5*pi) node[14];
cz node[0],node[7];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(2.1484374993916884*pi) node[2];
rz(0.5*pi) node[7];
rz(0.5*pi) node[14];
rx(0.5*pi) node[0];
rz(0.5*pi) node[2];
rx(0.5*pi) node[7];
rz(3.7187500000000004*pi) node[14];
rz(0.5*pi) node[0];
rx(0.5*pi) node[2];
rz(0.5*pi) node[7];
rz(0.5*pi) node[14];
cz node[7],node[0];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[0];
cz node[3],node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[0];
cz node[1],node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[0];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[14];
rx(0.5*pi) node[0];
cz node[2],node[3];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(2.296875*pi) node[14];
cz node[1],node[0];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[14];
rx(0.5*pi) node[0];
cz node[3],node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
cz node[13],node[14];
rz(0.4375000000000001*pi) node[0];
rx(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rx(0.5*pi) node[2];
cz node[14],node[13];
cz node[1],node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
cz node[1],node[2];
cz node[13],node[14];
rz(1.59375*pi) node[0];
rx(3.875000000000001*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[2];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[2];
cz node[12],node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[2];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[2];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
cz node[1],node[2];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[13],node[12];
cz node[1],node[14];
rx(0.5*pi) node[2];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rx(3.75*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(3.5*pi) node[2];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(3.5*pi) node[2];
cz node[12],node[13];
rz(0.5*pi) node[14];
rz(1.6874999999999998*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
cz node[1],node[14];
rz(0.5*pi) node[13];
rx(0.5*pi) node[1];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[14];
rz(3.5*pi) node[14];
rx(3.5*pi) node[14];
rz(1.375*pi) node[14];
rz(0.5*pi) node[14];
rx(0.5*pi) node[14];
rz(0.5*pi) node[14];
cz node[15],node[14];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[14],node[15];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[15],node[14];
rz(0.5*pi) node[14];
rx(0.5*pi) node[14];
rz(0.5*pi) node[14];
rz(0.5*pi) node[14];
rx(0.5*pi) node[14];
rz(0.5*pi) node[14];
cz node[1],node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
cz node[0],node[1];
rz(3.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(3.5*pi) node[14];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.25*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[14],node[13];
cz node[1],node[0];
rz(0.5*pi) node[13];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[13];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[13];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.25*pi) node[13];
cz node[0],node[1];
rz(0.5*pi) node[13];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[13];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[13];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[14],node[13];
cz node[7],node[0];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[7];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[0];
rx(0.5*pi) node[7];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[7];
rz(3.75*pi) node[13];
rx(0.5*pi) node[14];
cz node[0],node[7];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[7];
rx(0.5*pi) node[13];
cz node[15],node[14];
rx(0.5*pi) node[0];
rx(0.5*pi) node[7];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[0];
rz(0.5*pi) node[7];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
cz node[7],node[0];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[0];
cz node[14],node[15];
rx(0.5*pi) node[0];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[0];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[15],node[14];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[14],node[13];
rz(0.5*pi) node[13];
rx(0.5*pi) node[13];
rz(0.5*pi) node[13];
rz(0.1250000000000001*pi) node[13];
rz(0.5*pi) node[13];
rx(0.5*pi) node[13];
rz(0.5*pi) node[13];
cz node[14],node[13];
rz(0.5*pi) node[13];
cz node[14],node[15];
rx(0.5*pi) node[13];
rz(0.5*pi) node[15];
rz(0.5*pi) node[13];
rx(0.5*pi) node[15];
rz(3.8750000000000004*pi) node[13];
rz(0.5*pi) node[15];
rz(0.5*pi) node[13];
rz(0.25*pi) node[15];
rx(0.5*pi) node[13];
rz(0.5*pi) node[15];
rz(0.5*pi) node[13];
rx(0.5*pi) node[15];
cz node[2],node[13];
rz(0.5*pi) node[15];
rz(0.5*pi) node[13];
cz node[14],node[15];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.06250000000000044*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(3.75*pi) node[15];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[13];
cz node[15],node[14];
cz node[2],node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rx(0.5*pi) node[2];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
cz node[14],node[15];
rz(3.9375*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[13],node[2];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[2];
rx(0.5*pi) node[13];
cz node[15],node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[2],node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[2];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
cz node[13],node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[2];
cz node[13],node[14];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[2];
rz(0.1250000000000001*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
cz node[1],node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[2];
cz node[13],node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(0.031250000000000555*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[2];
rz(3.8750000000000004*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
cz node[1],node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
cz node[1],node[14];
rx(0.5*pi) node[2];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(3.96875*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[2];
rz(0.06250000000000044*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[14];
rz(0.5*pi) node[14];
cz node[1],node[14];
rz(0.5*pi) node[14];
rx(0.5*pi) node[14];
rz(0.5*pi) node[14];
rz(3.9375*pi) node[14];
rz(0.5*pi) node[14];
rx(0.5*pi) node[14];
rz(0.5*pi) node[14];
cz node[13],node[14];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
cz node[14],node[13];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
cz node[13],node[14];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
cz node[12],node[13];
cz node[14],node[15];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[15];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[15];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[15];
cz node[13],node[12];
rz(0.25*pi) node[15];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[15];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[15];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[15];
cz node[12],node[13];
cz node[14],node[15];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[13],node[2];
rx(0.5*pi) node[14];
rz(3.75*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[1],node[14];
rx(0.5*pi) node[2];
rx(0.5*pi) node[15];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[1];
rz(0.01562500000000011*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
cz node[14],node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
cz node[13],node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[13],node[12];
rz(0.5*pi) node[14];
cz node[1],node[14];
rx(0.5*pi) node[2];
rz(0.5*pi) node[12];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[12];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(3.984375*pi) node[2];
rz(0.5*pi) node[12];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.031250000000000555*pi) node[12];
rz(0.5*pi) node[14];
rx(0.5*pi) node[2];
rz(0.5*pi) node[12];
cz node[14],node[15];
rz(0.5*pi) node[2];
rx(0.5*pi) node[12];
rz(0.5*pi) node[15];
cz node[3],node[2];
rz(0.5*pi) node[12];
rx(0.5*pi) node[15];
rz(0.5*pi) node[2];
cz node[13],node[12];
rz(0.5*pi) node[15];
rx(0.5*pi) node[2];
rz(0.5*pi) node[12];
rz(0.1250000000000001*pi) node[15];
rz(0.5*pi) node[2];
rx(0.5*pi) node[12];
rz(0.5*pi) node[15];
rz(0.007812500000000444*pi) node[2];
rz(0.5*pi) node[12];
rx(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(3.96875*pi) node[12];
rz(0.5*pi) node[15];
rx(0.5*pi) node[2];
cz node[14],node[15];
cz node[14],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[15];
rz(0.5*pi) node[1];
cz node[3],node[2];
rx(0.5*pi) node[15];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[15];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(3.8750000000000004*pi) node[15];
rz(0.25*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(3.9921875000000004*pi) node[2];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[14],node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rz(3.75*pi) node[1];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
cz node[15],node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[2],node[1];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
cz node[14],node[15];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[1],node[2];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
cz node[15],node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[2],node[1];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
cz node[13],node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
cz node[0],node[1];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.06250000000000044*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rz(0.003906250000000555*pi) node[1];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
cz node[13],node[14];
rz(0.5*pi) node[1];
cz node[13],node[2];
rz(0.5*pi) node[14];
cz node[0],node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(3.9375*pi) node[14];
rz(0.5*pi) node[1];
rz(0.1250000000000001*pi) node[2];
rz(0.5*pi) node[14];
rz(1.99609375*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
cz node[13],node[2];
cz node[0],node[1];
rz(0.5*pi) node[2];
cz node[13],node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(3.8750000000000004*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
cz node[1],node[0];
rz(0.5*pi) node[2];
cz node[14],node[13];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[3],node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
cz node[0],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
cz node[13],node[14];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
cz node[2],node[3];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
cz node[12],node[13];
cz node[14],node[15];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[15];
cz node[3],node[2];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
cz node[13],node[12];
rz(0.25*pi) node[15];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[15];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[15];
cz node[12],node[13];
cz node[14],node[15];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(3.75*pi) node[15];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[13];
cz node[15],node[14];
cz node[2],node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[13];
cz node[14],node[15];
rz(0.01562500000000011*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[13];
cz node[15],node[14];
cz node[2],node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rx(0.5*pi) node[2];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(3.984375*pi) node[13];
rx(0.5*pi) node[14];
cz node[13],node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rx(0.5*pi) node[2];
rx(0.5*pi) node[13];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
cz node[2],node[13];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rx(0.5*pi) node[2];
rx(0.5*pi) node[13];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
cz node[13],node[2];
rz(0.5*pi) node[2];
cz node[13],node[12];
rx(0.5*pi) node[2];
rz(0.5*pi) node[12];
rz(0.5*pi) node[2];
rx(0.5*pi) node[12];
rz(0.5*pi) node[2];
rz(0.5*pi) node[12];
rx(0.5*pi) node[2];
rz(0.031250000000000555*pi) node[12];
rz(0.5*pi) node[2];
rz(0.5*pi) node[12];
cz node[1],node[2];
rx(0.5*pi) node[12];
rz(0.5*pi) node[2];
rz(0.5*pi) node[12];
rx(0.5*pi) node[2];
cz node[13],node[12];
rz(0.5*pi) node[2];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.007812500000000444*pi) node[2];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rz(0.5*pi) node[2];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rx(0.5*pi) node[2];
rz(3.96875*pi) node[12];
rz(0.5*pi) node[2];
rz(0.5*pi) node[12];
cz node[1],node[2];
rx(0.5*pi) node[12];
rz(0.5*pi) node[2];
rz(0.5*pi) node[12];
rx(0.5*pi) node[2];
rz(0.5*pi) node[2];
rz(1.9921875*pi) node[2];
rz(0.5*pi) node[2];
rx(0.5*pi) node[2];
rz(0.5*pi) node[2];
cz node[1],node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[2],node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[1],node[2];
rz(0.5*pi) node[2];
rx(0.5*pi) node[2];
rz(0.5*pi) node[2];
cz node[2],node[13];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rx(0.5*pi) node[2];
rx(0.5*pi) node[13];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
cz node[13],node[2];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rx(0.5*pi) node[2];
rx(0.5*pi) node[13];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
cz node[2],node[13];
cz node[2],node[3];
rz(0.5*pi) node[13];
rz(0.5*pi) node[3];
rx(0.5*pi) node[13];
rx(0.5*pi) node[3];
rz(0.5*pi) node[13];
rz(0.5*pi) node[3];
cz node[13],node[12];
rz(0.06250000000000044*pi) node[3];
rz(0.5*pi) node[12];
rz(0.5*pi) node[3];
rx(0.5*pi) node[12];
rx(0.5*pi) node[3];
rz(0.5*pi) node[12];
rz(0.5*pi) node[3];
rz(0.01562500000000011*pi) node[12];
cz node[2],node[3];
rz(0.5*pi) node[12];
rz(0.5*pi) node[3];
rx(0.5*pi) node[12];
rx(0.5*pi) node[3];
rz(0.5*pi) node[12];
rz(0.5*pi) node[3];
cz node[13],node[12];
rz(3.9375*pi) node[3];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[3];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[3];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
cz node[2],node[13];
rz(0.5*pi) node[3];
rz(1.984375*pi) node[12];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rx(0.5*pi) node[2];
rx(0.5*pi) node[13];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
cz node[13],node[2];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rx(0.5*pi) node[2];
rx(0.5*pi) node[13];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
cz node[2],node[13];
cz node[2],node[3];
rz(0.5*pi) node[13];
rz(0.5*pi) node[3];
rx(0.5*pi) node[13];
rx(0.5*pi) node[3];
rz(0.5*pi) node[13];
rz(0.5*pi) node[3];
cz node[13],node[14];
rz(0.031250000000000555*pi) node[3];
rz(0.5*pi) node[14];
rz(0.5*pi) node[3];
rx(0.5*pi) node[14];
rx(0.5*pi) node[3];
rz(0.5*pi) node[14];
rz(0.5*pi) node[3];
rz(0.1250000000000001*pi) node[14];
cz node[2],node[3];
rz(0.5*pi) node[14];
rz(0.5*pi) node[3];
rx(0.5*pi) node[14];
rx(0.5*pi) node[3];
rz(0.5*pi) node[14];
rz(0.5*pi) node[3];
cz node[13],node[14];
rz(1.96875*pi) node[3];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(3.8750000000000004*pi) node[14];
cz node[14],node[13];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
cz node[13],node[14];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
cz node[14],node[13];
rz(0.5*pi) node[13];
cz node[14],node[15];
rx(0.5*pi) node[13];
rz(0.5*pi) node[15];
rz(0.5*pi) node[13];
rx(0.5*pi) node[15];
rz(0.5*pi) node[13];
rz(0.5*pi) node[15];
rx(0.5*pi) node[13];
rz(0.25*pi) node[15];
rz(0.5*pi) node[13];
rz(0.5*pi) node[15];
cz node[2],node[13];
rx(0.5*pi) node[15];
rz(0.5*pi) node[13];
rz(0.5*pi) node[15];
rx(0.5*pi) node[13];
cz node[14],node[15];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.06250000000000044*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(3.75*pi) node[15];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
cz node[2],node[13];
rx(0.5*pi) node[15];
rz(0.5*pi) node[13];
rz(0.5*pi) node[15];
rx(0.5*pi) node[13];
rz(0.5*pi) node[13];
rz(1.9375*pi) node[13];
rz(0.5*pi) node[13];
rx(0.5*pi) node[13];
rz(0.5*pi) node[13];
cz node[2],node[13];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rx(0.5*pi) node[2];
rx(0.5*pi) node[13];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
cz node[13],node[2];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rx(0.5*pi) node[2];
rx(0.5*pi) node[13];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
cz node[2],node[13];
rz(0.5*pi) node[13];
rx(0.5*pi) node[13];
rz(0.5*pi) node[13];
cz node[13],node[14];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
cz node[14],node[13];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
cz node[13],node[14];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
cz node[14],node[15];
rz(0.5*pi) node[15];
rx(0.5*pi) node[15];
rz(0.5*pi) node[15];
rz(0.1250000000000001*pi) node[15];
rz(0.5*pi) node[15];
rx(0.5*pi) node[15];
rz(0.5*pi) node[15];
cz node[14],node[15];
cz node[14],node[13];
rz(0.5*pi) node[15];
rz(0.5*pi) node[13];
rx(0.5*pi) node[15];
rx(0.5*pi) node[13];
rz(0.5*pi) node[15];
rz(0.5*pi) node[13];
rz(1.875*pi) node[15];
rz(0.25*pi) node[13];
rz(0.5*pi) node[13];
rx(0.5*pi) node[13];
rz(0.5*pi) node[13];
cz node[14],node[13];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[13];
rz(1.75*pi) node[13];
barrier node[0],node[1],node[12],node[3],node[2],node[15],node[13],node[14],node[7];
measure node[0] -> c[0];
measure node[1] -> c[1];
measure node[12] -> c[2];
measure node[3] -> c[3];
measure node[2] -> c[4];
measure node[15] -> c[5];
measure node[13] -> c[6];
measure node[14] -> c[7];
