OPENQASM 2.0;
include "qelib1.inc";

qreg node[26];
creg c[9];
sx node[11];
x node[13];
sx node[14];
sx node[16];
sx node[19];
sx node[20];
rz(2.5703125027284566*pi) node[22];
x node[23];
sx node[24];
sx node[25];
rz(0.5*pi) node[11];
sx node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
x node[22];
sx node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
sx node[11];
sx node[14];
sx node[16];
sx node[19];
sx node[20];
rz(0.5*pi) node[22];
sx node[24];
sx node[25];
cx node[22],node[25];
rz(1.0*pi) node[24];
cx node[23],node[24];
rz(0.42968750212014617*pi) node[25];
cx node[22],node[25];
rz(0.5*pi) node[23];
cx node[22],node[19];
sx node[23];
rz(3.5722656228798533*pi) node[25];
rz(3.8593750006083107*pi) node[19];
rz(2.75*pi) node[23];
cx node[22],node[19];
sx node[23];
rz(2.144531249391689*pi) node[19];
rz(1.0*pi) node[23];
cx node[22],node[19];
cx node[23],node[24];
cx node[19],node[22];
sx node[23];
sx node[24];
cx node[22],node[19];
rz(3.25*pi) node[23];
rz(3.5*pi) node[24];
cx node[19],node[16];
sx node[23];
sx node[24];
rz(3.71875*pi) node[16];
rz(0.5*pi) node[23];
rz(0.75*pi) node[24];
cx node[19],node[16];
cx node[23],node[21];
cx node[24],node[25];
rz(2.2890624999999996*pi) node[16];
cx node[19],node[20];
cx node[21],node[23];
cx node[25],node[24];
rz(0.4375000000000002*pi) node[20];
cx node[23],node[21];
cx node[24],node[25];
cx node[21],node[18];
cx node[19],node[20];
cx node[25],node[22];
cx node[19],node[16];
cx node[18],node[21];
rz(1.5781249999999996*pi) node[20];
cx node[22],node[25];
cx node[16],node[19];
cx node[21],node[18];
cx node[25],node[22];
cx node[18],node[15];
cx node[19],node[16];
cx node[16],node[14];
cx node[15],node[18];
cx node[22],node[19];
rz(3.875*pi) node[14];
cx node[18],node[15];
cx node[19],node[22];
cx node[15],node[12];
cx node[16],node[14];
cx node[22],node[19];
cx node[12],node[15];
rz(0.15624999999999956*pi) node[14];
cx node[15],node[12];
cx node[16],node[14];
cx node[14],node[16];
cx node[16],node[14];
cx node[14],node[11];
cx node[19],node[16];
rz(3.75*pi) node[11];
cx node[16],node[19];
cx node[14],node[11];
cx node[19],node[16];
rz(0.3125*pi) node[11];
cx node[14],node[13];
sx node[13];
cx node[16],node[14];
rz(2.5*pi) node[13];
cx node[14],node[16];
sx node[13];
cx node[16],node[14];
rz(0.6249999999999993*pi) node[13];
cx node[19],node[16];
cx node[13],node[14];
cx node[16],node[19];
rz(0.125*pi) node[14];
cx node[19],node[16];
cx node[13],node[14];
cx node[20],node[19];
cx node[13],node[12];
rz(3.875*pi) node[14];
cx node[19],node[20];
cx node[11],node[14];
rz(0.25*pi) node[12];
cx node[20],node[19];
cx node[13],node[12];
rz(0.0625*pi) node[14];
cx node[11],node[14];
rz(3.75*pi) node[12];
rz(0.5*pi) node[13];
sx node[13];
rz(3.9375*pi) node[14];
rz(3.5*pi) node[13];
cx node[16],node[14];
sx node[13];
rz(0.03125*pi) node[14];
rz(1.0*pi) node[13];
cx node[16],node[14];
cx node[12],node[13];
rz(3.96875*pi) node[14];
cx node[13],node[12];
cx node[14],node[16];
cx node[12],node[13];
cx node[16],node[14];
cx node[14],node[16];
cx node[11],node[14];
cx node[19],node[16];
cx node[14],node[11];
rz(0.015625*pi) node[16];
cx node[11],node[14];
cx node[19],node[16];
cx node[14],node[13];
rz(3.984375*pi) node[16];
rz(0.125*pi) node[13];
cx node[19],node[16];
cx node[14],node[13];
cx node[16],node[19];
rz(3.875*pi) node[13];
cx node[19],node[16];
cx node[13],node[14];
cx node[22],node[19];
cx node[14],node[13];
rz(0.0078125*pi) node[19];
cx node[13],node[14];
cx node[22],node[19];
cx node[11],node[14];
cx node[13],node[12];
rz(3.9921875*pi) node[19];
rz(0.25*pi) node[12];
rz(0.0625*pi) node[14];
cx node[22],node[19];
cx node[11],node[14];
cx node[13],node[12];
cx node[19],node[22];
rz(3.75*pi) node[12];
rz(0.5*pi) node[13];
rz(3.9375*pi) node[14];
cx node[22],node[19];
sx node[13];
cx node[16],node[14];
cx node[25],node[22];
rz(3.5*pi) node[13];
rz(0.03125*pi) node[14];
rz(0.00390625*pi) node[22];
sx node[13];
cx node[16],node[14];
cx node[25],node[22];
rz(1.0*pi) node[13];
rz(3.96875*pi) node[14];
rz(3.99609375*pi) node[22];
cx node[12],node[13];
cx node[14],node[16];
cx node[25],node[22];
cx node[13],node[12];
cx node[16],node[14];
cx node[22],node[25];
cx node[12],node[13];
cx node[14],node[16];
cx node[25],node[22];
cx node[11],node[14];
cx node[19],node[16];
cx node[24],node[25];
cx node[14],node[11];
rz(0.015625*pi) node[16];
rz(0.001953125*pi) node[25];
cx node[11],node[14];
cx node[19],node[16];
cx node[24],node[25];
cx node[14],node[13];
rz(3.984375*pi) node[16];
rz(3.998046875*pi) node[25];
rz(0.125*pi) node[13];
cx node[16],node[19];
cx node[24],node[25];
cx node[14],node[13];
cx node[19],node[16];
cx node[25],node[24];
rz(3.875*pi) node[13];
cx node[16],node[19];
cx node[24],node[25];
cx node[13],node[14];
cx node[22],node[19];
cx node[14],node[13];
rz(0.0078125*pi) node[19];
cx node[13],node[14];
cx node[22],node[19];
cx node[11],node[14];
cx node[13],node[12];
rz(3.9921875*pi) node[19];
rz(0.25*pi) node[12];
rz(0.0625*pi) node[14];
cx node[19],node[22];
cx node[11],node[14];
cx node[13],node[12];
cx node[22],node[19];
rz(3.75*pi) node[12];
rz(0.5*pi) node[13];
rz(3.9375*pi) node[14];
cx node[19],node[22];
sx node[13];
cx node[16],node[14];
cx node[25],node[22];
rz(3.5*pi) node[13];
rz(0.03125*pi) node[14];
rz(0.00390625*pi) node[22];
sx node[13];
cx node[16],node[14];
cx node[25],node[22];
rz(1.0*pi) node[13];
rz(3.96875*pi) node[14];
rz(3.99609375*pi) node[22];
cx node[12],node[13];
cx node[14],node[16];
cx node[25],node[22];
cx node[13],node[12];
cx node[16],node[14];
cx node[22],node[25];
cx node[12],node[13];
cx node[14],node[16];
cx node[25],node[22];
cx node[11],node[14];
cx node[19],node[16];
cx node[14],node[11];
rz(0.015625*pi) node[16];
cx node[11],node[14];
cx node[19],node[16];
cx node[14],node[13];
rz(3.984375*pi) node[16];
rz(0.125*pi) node[13];
cx node[16],node[19];
cx node[14],node[13];
cx node[19],node[16];
rz(3.875*pi) node[13];
cx node[16],node[19];
cx node[13],node[14];
cx node[22],node[19];
cx node[14],node[13];
rz(0.0078125*pi) node[19];
cx node[13],node[14];
cx node[22],node[19];
cx node[11],node[14];
cx node[13],node[12];
rz(3.9921875*pi) node[19];
rz(0.25*pi) node[12];
rz(0.0625*pi) node[14];
cx node[22],node[19];
cx node[11],node[14];
cx node[13],node[12];
cx node[19],node[22];
rz(3.75*pi) node[12];
rz(0.5*pi) node[13];
rz(3.9375*pi) node[14];
cx node[22],node[19];
sx node[13];
cx node[16],node[14];
rz(3.5*pi) node[13];
rz(0.03125*pi) node[14];
sx node[13];
cx node[16],node[14];
rz(1.0*pi) node[13];
rz(3.96875*pi) node[14];
cx node[12],node[13];
cx node[14],node[16];
cx node[13],node[12];
cx node[16],node[14];
cx node[12],node[13];
cx node[14],node[16];
cx node[11],node[14];
cx node[19],node[16];
cx node[14],node[11];
rz(0.015625*pi) node[16];
cx node[11],node[14];
cx node[19],node[16];
cx node[14],node[13];
rz(3.984375*pi) node[16];
rz(0.125*pi) node[13];
cx node[19],node[16];
cx node[14],node[13];
cx node[16],node[19];
rz(3.875*pi) node[13];
cx node[19],node[16];
cx node[13],node[14];
cx node[14],node[13];
cx node[13],node[14];
cx node[11],node[14];
cx node[13],node[12];
rz(0.25*pi) node[12];
rz(0.0625*pi) node[14];
cx node[11],node[14];
cx node[13],node[12];
rz(3.75*pi) node[12];
rz(0.5*pi) node[13];
rz(3.9375*pi) node[14];
sx node[13];
cx node[16],node[14];
rz(3.5*pi) node[13];
rz(0.03125*pi) node[14];
sx node[13];
cx node[16],node[14];
rz(1.0*pi) node[13];
rz(3.96875*pi) node[14];
cx node[11],node[14];
cx node[12],node[13];
cx node[14],node[11];
cx node[13],node[12];
cx node[11],node[14];
cx node[12],node[13];
cx node[14],node[13];
rz(0.125*pi) node[13];
cx node[14],node[13];
rz(3.875*pi) node[13];
cx node[14],node[13];
cx node[13],node[14];
cx node[14],node[13];
cx node[13],node[12];
cx node[16],node[14];
rz(0.25*pi) node[12];
rz(0.0625*pi) node[14];
cx node[13],node[12];
cx node[16],node[14];
rz(3.75*pi) node[12];
rz(0.5*pi) node[13];
rz(3.9375*pi) node[14];
sx node[13];
cx node[16],node[14];
rz(3.5*pi) node[13];
cx node[14],node[16];
sx node[13];
cx node[16],node[14];
rz(1.0*pi) node[13];
cx node[14],node[13];
cx node[13],node[14];
cx node[14],node[13];
cx node[13],node[12];
rz(0.125*pi) node[12];
cx node[13],node[12];
rz(3.875*pi) node[12];
cx node[13],node[14];
rz(0.25*pi) node[14];
cx node[13],node[14];
rz(0.5*pi) node[13];
rz(3.75*pi) node[14];
sx node[13];
rz(3.5*pi) node[13];
sx node[13];
rz(1.0*pi) node[13];
barrier node[24],node[25],node[22],node[19],node[11],node[16],node[12],node[14],node[13],node[20];
measure node[24] -> c[0];
measure node[25] -> c[1];
measure node[22] -> c[2];
measure node[19] -> c[3];
measure node[11] -> c[4];
measure node[16] -> c[5];
measure node[12] -> c[6];
measure node[14] -> c[7];
measure node[13] -> c[8];
