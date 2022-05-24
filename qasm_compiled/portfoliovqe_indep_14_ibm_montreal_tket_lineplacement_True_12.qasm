OPENQASM 2.0;
include "qelib1.inc";

qreg node[26];
creg meas[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[4];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[15];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[21];
sx node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
sx node[1];
sx node[4];
sx node[6];
sx node[7];
sx node[10];
sx node[12];
sx node[13];
sx node[15];
sx node[17];
sx node[18];
sx node[21];
rz(2.92197402005774*pi) node[23];
sx node[24];
sx node[25];
rz(0.5*pi) node[1];
rz(0.5*pi) node[4];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(2.5*pi) node[10];
rz(2.5*pi) node[12];
rz(0.5*pi) node[13];
rz(2.5*pi) node[15];
rz(2.5*pi) node[17];
rz(2.5*pi) node[18];
rz(0.5*pi) node[21];
sx node[23];
rz(2.5*pi) node[24];
rz(2.5*pi) node[25];
sx node[1];
sx node[4];
sx node[6];
sx node[7];
sx node[10];
sx node[12];
sx node[13];
sx node[15];
sx node[17];
sx node[18];
sx node[21];
rz(1.0*pi) node[23];
sx node[24];
sx node[25];
rz(3.970429351823287*pi) node[1];
rz(0.6417301873330498*pi) node[4];
rz(0.9462238172341481*pi) node[6];
rz(1.269827711821554*pi) node[7];
rz(1.1726344757001774*pi) node[10];
rz(0.5239919652520937*pi) node[12];
rz(0.008162827989969257*pi) node[13];
rz(0.2246134785142273*pi) node[15];
rz(1.4417848502906216*pi) node[17];
rz(0.024477102080700197*pi) node[18];
rz(1.1323264544082132*pi) node[21];
rz(0.38210924100167865*pi) node[24];
rz(3.978875657858146*pi) node[25];
cx node[23],node[24];
sx node[24];
rz(2.5*pi) node[24];
sx node[24];
rz(1.5*pi) node[24];
cx node[25],node[24];
cx node[24],node[25];
cx node[25],node[24];
cx node[23],node[24];
cx node[23],node[21];
cx node[25],node[24];
cx node[23],node[21];
sx node[24];
cx node[21],node[23];
rz(2.5*pi) node[24];
cx node[23],node[21];
sx node[24];
cx node[21],node[18];
rz(1.5*pi) node[24];
cx node[21],node[18];
cx node[25],node[24];
cx node[18],node[21];
cx node[24],node[25];
cx node[21],node[18];
cx node[25],node[24];
cx node[18],node[15];
cx node[24],node[23];
cx node[18],node[17];
cx node[24],node[23];
cx node[18],node[15];
cx node[23],node[24];
cx node[15],node[18];
cx node[24],node[23];
cx node[18],node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[15],node[12];
cx node[23],node[21];
sx node[24];
cx node[15],node[12];
cx node[21],node[23];
rz(2.5*pi) node[24];
cx node[12],node[15];
cx node[23],node[21];
sx node[24];
cx node[15],node[12];
cx node[21],node[18];
rz(1.5*pi) node[24];
cx node[12],node[10];
cx node[21],node[18];
cx node[25],node[24];
cx node[12],node[13];
cx node[18],node[21];
cx node[24],node[25];
cx node[12],node[10];
cx node[21],node[18];
cx node[25],node[24];
cx node[10],node[12];
cx node[18],node[17];
cx node[24],node[23];
cx node[12],node[10];
cx node[18],node[15];
cx node[24],node[23];
cx node[10],node[7];
cx node[18],node[15];
cx node[23],node[24];
cx node[10],node[7];
cx node[15],node[18];
cx node[24],node[23];
cx node[7],node[10];
cx node[18],node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[10],node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
cx node[7],node[4];
cx node[15],node[12];
cx node[18],node[17];
cx node[21],node[23];
rz(2.5*pi) node[24];
cx node[7],node[6];
cx node[12],node[15];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
cx node[7],node[4];
cx node[15],node[12];
cx node[21],node[18];
rz(1.5*pi) node[24];
cx node[4],node[1];
cx node[12],node[13];
cx node[21],node[18];
cx node[25],node[24];
cx node[7],node[4];
cx node[12],node[10];
cx node[18],node[21];
cx node[24],node[25];
cx node[4],node[1];
sx node[7];
cx node[12],node[10];
cx node[21],node[18];
cx node[25],node[24];
rz(3.804883686263117*pi) node[7];
cx node[10],node[12];
cx node[18],node[17];
cx node[24],node[23];
sx node[7];
cx node[12],node[10];
cx node[18],node[15];
cx node[23],node[24];
rz(1.0*pi) node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[24],node[23];
cx node[4],node[7];
cx node[12],node[13];
cx node[15],node[18];
cx node[23],node[24];
cx node[7],node[4];
cx node[13],node[12];
cx node[18],node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[4],node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
cx node[1],node[4];
cx node[10],node[7];
cx node[12],node[15];
cx node[18],node[17];
cx node[21],node[23];
rz(2.5*pi) node[24];
cx node[4],node[1];
cx node[10],node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
cx node[1],node[4];
cx node[7],node[10];
cx node[12],node[15];
cx node[21],node[18];
rz(1.5*pi) node[24];
cx node[10],node[7];
cx node[12],node[13];
cx node[21],node[18];
cx node[25],node[24];
cx node[7],node[6];
cx node[12],node[10];
cx node[18],node[21];
cx node[24],node[25];
cx node[7],node[4];
cx node[12],node[10];
cx node[21],node[18];
cx node[25],node[24];
rz(0.5*pi) node[7];
cx node[10],node[12];
cx node[18],node[17];
cx node[24],node[23];
sx node[7];
cx node[12],node[10];
cx node[18],node[15];
cx node[24],node[23];
rz(2.5*pi) node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[23],node[24];
sx node[7];
cx node[12],node[13];
cx node[15],node[18];
cx node[24],node[23];
rz(0.3429792499196991*pi) node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[7],node[4];
cx node[15],node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
cx node[4],node[7];
cx node[12],node[15];
cx node[18],node[17];
cx node[21],node[23];
rz(2.5*pi) node[24];
cx node[7],node[4];
cx node[15],node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
cx node[1],node[4];
cx node[6],node[7];
cx node[12],node[15];
cx node[21],node[18];
rz(1.5*pi) node[24];
sx node[4];
cx node[7],node[6];
cx node[12],node[13];
cx node[21],node[18];
cx node[25],node[24];
rz(2.5*pi) node[4];
cx node[6],node[7];
cx node[18],node[21];
cx node[24],node[25];
sx node[4];
cx node[10],node[7];
cx node[21],node[18];
cx node[25],node[24];
rz(1.5*pi) node[4];
cx node[10],node[7];
cx node[18],node[17];
cx node[24],node[23];
cx node[1],node[4];
cx node[7],node[10];
cx node[18],node[15];
cx node[23],node[24];
cx node[4],node[1];
cx node[10],node[7];
cx node[18],node[15];
cx node[24],node[23];
cx node[1],node[4];
cx node[7],node[6];
cx node[12],node[10];
cx node[15],node[18];
cx node[23],node[24];
rz(0.5*pi) node[7];
cx node[12],node[10];
cx node[18],node[15];
cx node[23],node[21];
cx node[25],node[24];
sx node[7];
cx node[10],node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
rz(2.5*pi) node[7];
cx node[12],node[10];
cx node[18],node[17];
cx node[21],node[23];
rz(2.5*pi) node[24];
sx node[7];
cx node[13],node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
rz(1.4716990840513113*pi) node[7];
cx node[12],node[13];
cx node[21],node[18];
rz(1.5*pi) node[24];
cx node[4],node[7];
cx node[13],node[12];
cx node[21],node[18];
cx node[25],node[24];
cx node[7],node[4];
cx node[15],node[12];
cx node[18],node[21];
cx node[24],node[25];
cx node[4],node[7];
cx node[12],node[15];
cx node[21],node[18];
cx node[25],node[24];
cx node[7],node[4];
cx node[15],node[12];
cx node[18],node[17];
cx node[24],node[23];
cx node[1],node[4];
cx node[6],node[7];
cx node[12],node[15];
cx node[23],node[24];
sx node[4];
cx node[7],node[6];
cx node[12],node[13];
cx node[18],node[15];
cx node[24],node[23];
rz(2.5*pi) node[4];
cx node[6],node[7];
cx node[18],node[15];
cx node[23],node[24];
sx node[4];
cx node[10],node[7];
cx node[15],node[18];
cx node[23],node[21];
cx node[25],node[24];
rz(1.5*pi) node[4];
rz(0.5*pi) node[10];
cx node[18],node[15];
cx node[23],node[21];
sx node[24];
cx node[1],node[4];
sx node[10];
cx node[17],node[18];
cx node[21],node[23];
rz(2.5*pi) node[24];
cx node[4],node[1];
rz(2.5*pi) node[10];
cx node[18],node[17];
cx node[23],node[21];
sx node[24];
cx node[1],node[4];
sx node[10];
cx node[17],node[18];
rz(1.5*pi) node[24];
rz(0.8818209568663498*pi) node[10];
cx node[21],node[18];
cx node[25],node[24];
cx node[10],node[7];
cx node[21],node[18];
cx node[24],node[25];
cx node[7],node[10];
cx node[18],node[21];
cx node[25],node[24];
cx node[10],node[7];
cx node[21],node[18];
cx node[24],node[23];
cx node[6],node[7];
cx node[12],node[10];
cx node[18],node[17];
cx node[23],node[24];
cx node[4],node[7];
rz(0.5*pi) node[12];
cx node[24],node[23];
cx node[7],node[4];
sx node[12];
cx node[23],node[24];
cx node[4],node[7];
rz(0.5*pi) node[12];
cx node[23],node[21];
cx node[25],node[24];
cx node[7],node[4];
sx node[12];
cx node[23],node[21];
sx node[24];
cx node[1],node[4];
cx node[6],node[7];
rz(3.511200008439274*pi) node[12];
cx node[21],node[23];
rz(2.5*pi) node[24];
sx node[4];
cx node[7],node[6];
cx node[12],node[10];
cx node[23],node[21];
sx node[24];
rz(2.5*pi) node[4];
cx node[6],node[7];
cx node[10],node[12];
rz(1.5*pi) node[24];
sx node[4];
cx node[12],node[10];
cx node[25],node[24];
rz(1.5*pi) node[4];
cx node[7],node[10];
cx node[13],node[12];
cx node[24],node[25];
cx node[1],node[4];
cx node[10],node[7];
cx node[12],node[13];
cx node[25],node[24];
cx node[4],node[1];
cx node[7],node[10];
cx node[13],node[12];
cx node[24],node[23];
cx node[1],node[4];
cx node[10],node[7];
cx node[15],node[12];
cx node[23],node[24];
cx node[6],node[7];
cx node[12],node[15];
cx node[24],node[23];
cx node[4],node[7];
cx node[15],node[12];
cx node[23],node[24];
cx node[7],node[4];
cx node[12],node[15];
cx node[25],node[24];
cx node[4],node[7];
cx node[12],node[13];
cx node[18],node[15];
sx node[24];
cx node[7],node[4];
rz(0.5*pi) node[12];
cx node[18],node[15];
rz(2.5*pi) node[24];
cx node[1],node[4];
cx node[6],node[7];
sx node[12];
cx node[15],node[18];
sx node[24];
sx node[4];
cx node[7],node[6];
rz(2.5*pi) node[12];
cx node[18],node[15];
rz(1.5*pi) node[24];
rz(2.5*pi) node[4];
cx node[6],node[7];
sx node[12];
cx node[17],node[18];
cx node[25],node[24];
sx node[4];
rz(0.30493464935688686*pi) node[12];
cx node[18],node[17];
cx node[24],node[25];
rz(1.5*pi) node[4];
cx node[10],node[12];
cx node[17],node[18];
cx node[25],node[24];
cx node[1],node[4];
cx node[12],node[10];
cx node[21],node[18];
cx node[4],node[1];
cx node[10],node[12];
cx node[21],node[18];
cx node[1],node[4];
cx node[12],node[10];
cx node[18],node[21];
cx node[7],node[10];
cx node[13],node[12];
cx node[21],node[18];
cx node[10],node[7];
cx node[12],node[13];
cx node[18],node[17];
cx node[23],node[21];
cx node[7],node[10];
cx node[13],node[12];
cx node[23],node[21];
cx node[10],node[7];
cx node[15],node[12];
cx node[21],node[23];
cx node[6],node[7];
rz(0.5*pi) node[15];
cx node[23],node[21];
cx node[4],node[7];
sx node[15];
cx node[24],node[23];
cx node[7],node[4];
rz(0.5*pi) node[15];
cx node[23],node[24];
cx node[4],node[7];
sx node[15];
cx node[24],node[23];
cx node[7],node[4];
rz(3.562143482245661*pi) node[15];
cx node[23],node[24];
cx node[1],node[4];
cx node[6],node[7];
cx node[12],node[15];
cx node[25],node[24];
sx node[4];
cx node[7],node[6];
cx node[15],node[12];
sx node[24];
rz(2.5*pi) node[4];
cx node[6],node[7];
cx node[12],node[15];
rz(2.5*pi) node[24];
sx node[4];
cx node[13],node[12];
cx node[18],node[15];
sx node[24];
rz(1.5*pi) node[4];
cx node[10],node[12];
rz(0.5*pi) node[18];
rz(1.5*pi) node[24];
cx node[1],node[4];
cx node[12],node[10];
sx node[18];
cx node[25],node[24];
cx node[4],node[1];
cx node[10],node[12];
rz(2.5*pi) node[18];
cx node[24],node[25];
cx node[1],node[4];
cx node[12],node[10];
sx node[18];
cx node[25],node[24];
cx node[7],node[10];
cx node[13],node[12];
rz(1.0442658640063625*pi) node[18];
cx node[10],node[7];
cx node[12],node[13];
cx node[18],node[15];
cx node[7],node[10];
cx node[13],node[12];
cx node[15],node[18];
cx node[10],node[7];
cx node[18],node[15];
cx node[6],node[7];
cx node[12],node[15];
cx node[17],node[18];
cx node[4],node[7];
cx node[12],node[15];
cx node[18],node[17];
cx node[7],node[4];
cx node[15],node[12];
cx node[17],node[18];
cx node[4],node[7];
cx node[12],node[15];
cx node[21],node[18];
cx node[7],node[4];
cx node[13],node[12];
cx node[21],node[18];
cx node[1],node[4];
cx node[6],node[7];
cx node[10],node[12];
cx node[18],node[21];
sx node[4];
cx node[7],node[6];
cx node[12],node[10];
cx node[21],node[18];
rz(2.5*pi) node[4];
cx node[6],node[7];
cx node[10],node[12];
cx node[18],node[17];
cx node[23],node[21];
sx node[4];
cx node[12],node[10];
rz(0.5*pi) node[18];
cx node[23],node[21];
rz(1.5*pi) node[4];
cx node[7],node[10];
cx node[13],node[12];
sx node[18];
cx node[21],node[23];
cx node[1],node[4];
cx node[10],node[7];
cx node[12],node[13];
rz(0.5*pi) node[18];
cx node[23],node[21];
cx node[4],node[1];
cx node[7],node[10];
cx node[13],node[12];
sx node[18];
cx node[24],node[23];
cx node[1],node[4];
cx node[10],node[7];
rz(0.3135146767820023*pi) node[18];
cx node[23],node[24];
cx node[6],node[7];
cx node[15],node[18];
cx node[24],node[23];
cx node[4],node[7];
cx node[18],node[15];
cx node[23],node[24];
cx node[7],node[4];
cx node[15],node[18];
cx node[25],node[24];
cx node[4],node[7];
cx node[18],node[15];
sx node[24];
cx node[7],node[4];
cx node[12],node[15];
cx node[17],node[18];
rz(2.5*pi) node[24];
cx node[1],node[4];
cx node[6],node[7];
cx node[12],node[15];
cx node[18],node[17];
sx node[24];
sx node[4];
cx node[7],node[6];
cx node[15],node[12];
cx node[17],node[18];
rz(1.5*pi) node[24];
rz(2.5*pi) node[4];
cx node[6],node[7];
cx node[12],node[15];
cx node[21],node[18];
cx node[25],node[24];
sx node[4];
cx node[13],node[12];
rz(0.5*pi) node[21];
cx node[24],node[25];
rz(1.5*pi) node[4];
cx node[10],node[12];
sx node[21];
cx node[25],node[24];
cx node[1],node[4];
cx node[12],node[10];
rz(2.5*pi) node[21];
cx node[4],node[1];
cx node[10],node[12];
sx node[21];
cx node[1],node[4];
cx node[12],node[10];
rz(0.3445262543886769*pi) node[21];
cx node[7],node[10];
cx node[13],node[12];
cx node[21],node[18];
cx node[10],node[7];
cx node[12],node[13];
cx node[18],node[21];
cx node[7],node[10];
cx node[13],node[12];
cx node[21],node[18];
cx node[10],node[7];
cx node[17],node[18];
cx node[23],node[21];
cx node[6],node[7];
cx node[15],node[18];
rz(0.5*pi) node[23];
cx node[4],node[7];
cx node[18],node[15];
sx node[23];
cx node[7],node[4];
cx node[15],node[18];
rz(2.5*pi) node[23];
cx node[4],node[7];
cx node[18],node[15];
sx node[23];
cx node[7],node[4];
cx node[12],node[15];
cx node[17],node[18];
rz(1.4007557562775501*pi) node[23];
cx node[1],node[4];
cx node[6],node[7];
cx node[12],node[15];
cx node[18],node[17];
cx node[23],node[21];
sx node[4];
cx node[7],node[6];
cx node[15],node[12];
cx node[17],node[18];
cx node[21],node[23];
rz(2.5*pi) node[4];
cx node[6],node[7];
cx node[12],node[15];
cx node[23],node[21];
sx node[4];
cx node[13],node[12];
cx node[18],node[21];
cx node[24],node[23];
rz(1.5*pi) node[4];
cx node[10],node[12];
cx node[21],node[18];
rz(0.5*pi) node[24];
cx node[1],node[4];
cx node[12],node[10];
cx node[18],node[21];
sx node[24];
cx node[4],node[1];
cx node[10],node[12];
cx node[21],node[18];
rz(0.5*pi) node[24];
cx node[1],node[4];
cx node[12],node[10];
cx node[17],node[18];
sx node[24];
cx node[7],node[10];
cx node[13],node[12];
cx node[15],node[18];
rz(0.22203966901883332*pi) node[24];
cx node[10],node[7];
cx node[12],node[13];
cx node[18],node[15];
cx node[23],node[24];
cx node[7],node[10];
cx node[13],node[12];
cx node[15],node[18];
cx node[24],node[23];
cx node[10],node[7];
cx node[18],node[15];
cx node[23],node[24];
cx node[6],node[7];
cx node[12],node[15];
cx node[17],node[18];
cx node[21],node[23];
cx node[25],node[24];
cx node[4],node[7];
cx node[12],node[15];
cx node[18],node[17];
cx node[23],node[21];
rz(1.672977947539851*pi) node[24];
rz(0.5*pi) node[25];
cx node[7],node[4];
cx node[15],node[12];
cx node[17],node[18];
cx node[21],node[23];
sx node[25];
cx node[4],node[7];
cx node[12],node[15];
cx node[23],node[21];
rz(2.5*pi) node[25];
cx node[7],node[4];
cx node[13],node[12];
cx node[18],node[21];
sx node[25];
cx node[1],node[4];
cx node[6],node[7];
cx node[10],node[12];
cx node[21],node[18];
rz(0.18047439028949197*pi) node[25];
sx node[4];
cx node[7],node[6];
cx node[12],node[10];
cx node[18],node[21];
cx node[25],node[24];
rz(2.5*pi) node[4];
cx node[6],node[7];
cx node[10],node[12];
cx node[21],node[18];
cx node[24],node[25];
sx node[4];
cx node[12],node[10];
cx node[17],node[18];
cx node[25],node[24];
rz(1.5*pi) node[4];
cx node[7],node[10];
cx node[13],node[12];
cx node[15],node[18];
cx node[23],node[24];
cx node[1],node[4];
cx node[10],node[7];
cx node[12],node[13];
cx node[18],node[15];
cx node[23],node[24];
cx node[4],node[1];
cx node[7],node[10];
cx node[13],node[12];
cx node[15],node[18];
cx node[24],node[23];
cx node[1],node[4];
cx node[10],node[7];
cx node[18],node[15];
cx node[23],node[24];
cx node[6],node[7];
cx node[12],node[15];
cx node[17],node[18];
cx node[21],node[23];
cx node[24],node[25];
cx node[4],node[7];
cx node[12],node[15];
cx node[18],node[17];
cx node[23],node[21];
sx node[24];
cx node[7],node[4];
cx node[15],node[12];
cx node[17],node[18];
cx node[21],node[23];
rz(3.7497449420310875*pi) node[24];
cx node[4],node[7];
cx node[12],node[15];
cx node[23],node[21];
sx node[24];
cx node[7],node[4];
cx node[13],node[12];
cx node[18],node[21];
rz(1.0*pi) node[24];
cx node[1],node[4];
cx node[6],node[7];
cx node[10],node[12];
cx node[21],node[18];
cx node[25],node[24];
sx node[4];
cx node[7],node[6];
cx node[12],node[10];
cx node[18],node[21];
cx node[24],node[25];
rz(2.5*pi) node[4];
cx node[6],node[7];
cx node[10],node[12];
cx node[21],node[18];
cx node[25],node[24];
sx node[4];
cx node[12],node[10];
cx node[17],node[18];
cx node[23],node[24];
rz(1.5*pi) node[4];
cx node[7],node[10];
cx node[13],node[12];
cx node[15],node[18];
rz(0.5*pi) node[23];
cx node[1],node[4];
cx node[10],node[7];
cx node[12],node[13];
cx node[18],node[15];
sx node[23];
cx node[4],node[1];
cx node[7],node[10];
cx node[13],node[12];
cx node[15],node[18];
rz(0.5*pi) node[23];
cx node[1],node[4];
cx node[10],node[7];
cx node[18],node[15];
sx node[23];
cx node[6],node[7];
cx node[12],node[15];
cx node[17],node[18];
rz(0.2533662570890324*pi) node[23];
cx node[4],node[7];
cx node[12],node[15];
cx node[18],node[17];
cx node[23],node[24];
cx node[7],node[4];
cx node[15],node[12];
cx node[17],node[18];
cx node[24],node[23];
cx node[4],node[7];
cx node[12],node[15];
cx node[23],node[24];
cx node[7],node[4];
cx node[13],node[12];
cx node[21],node[23];
cx node[25],node[24];
cx node[1],node[4];
cx node[6],node[7];
cx node[10],node[12];
rz(0.5*pi) node[21];
sx node[24];
sx node[4];
cx node[7],node[6];
cx node[12],node[10];
sx node[21];
rz(2.5*pi) node[24];
rz(2.5*pi) node[4];
cx node[6],node[7];
cx node[10],node[12];
rz(2.5*pi) node[21];
sx node[24];
sx node[4];
cx node[12],node[10];
sx node[21];
rz(1.5*pi) node[24];
rz(1.5*pi) node[4];
cx node[7],node[10];
cx node[13],node[12];
rz(3.6276651752584756*pi) node[21];
cx node[25],node[24];
cx node[1],node[4];
cx node[10],node[7];
cx node[12],node[13];
cx node[23],node[21];
cx node[24],node[25];
cx node[4],node[1];
cx node[7],node[10];
cx node[13],node[12];
cx node[21],node[23];
cx node[25],node[24];
cx node[1],node[4];
cx node[10],node[7];
cx node[23],node[21];
cx node[6],node[7];
cx node[18],node[21];
cx node[24],node[23];
cx node[4],node[7];
rz(0.5*pi) node[18];
cx node[23],node[24];
cx node[7],node[4];
sx node[18];
cx node[24],node[23];
cx node[4],node[7];
rz(0.5*pi) node[18];
cx node[23],node[24];
cx node[7],node[4];
sx node[18];
cx node[25],node[24];
cx node[1],node[4];
cx node[6],node[7];
rz(0.7038716832026509*pi) node[18];
sx node[24];
sx node[4];
cx node[7],node[6];
cx node[21],node[18];
rz(2.5*pi) node[24];
rz(2.5*pi) node[4];
cx node[6],node[7];
cx node[18],node[21];
sx node[24];
sx node[4];
cx node[21],node[18];
rz(1.5*pi) node[24];
rz(1.5*pi) node[4];
cx node[17],node[18];
cx node[23],node[21];
cx node[25],node[24];
cx node[1],node[4];
cx node[15],node[18];
rz(0.5*pi) node[17];
cx node[23],node[21];
cx node[24],node[25];
cx node[4],node[1];
rz(0.5*pi) node[15];
sx node[17];
cx node[21],node[23];
cx node[25],node[24];
cx node[1],node[4];
sx node[15];
rz(2.5*pi) node[17];
cx node[23],node[21];
rz(2.5*pi) node[15];
sx node[17];
cx node[24],node[23];
sx node[15];
rz(3.7830877727242305*pi) node[17];
cx node[23],node[24];
rz(1.2057293031972232*pi) node[15];
cx node[24],node[23];
cx node[18],node[15];
cx node[23],node[24];
cx node[15],node[18];
cx node[25],node[24];
cx node[18],node[15];
sx node[24];
cx node[12],node[15];
cx node[17],node[18];
rz(2.5*pi) node[24];
rz(0.5*pi) node[12];
cx node[18],node[17];
sx node[24];
sx node[12];
cx node[17],node[18];
rz(1.5*pi) node[24];
rz(0.5*pi) node[12];
cx node[21],node[18];
cx node[25],node[24];
sx node[12];
cx node[21],node[18];
cx node[24],node[25];
rz(3.7841247839435783*pi) node[12];
cx node[18],node[21];
cx node[25],node[24];
cx node[12],node[15];
cx node[21],node[18];
cx node[15],node[12];
cx node[18],node[17];
cx node[23],node[21];
cx node[12],node[15];
cx node[23],node[21];
cx node[13],node[12];
cx node[18],node[15];
cx node[21],node[23];
cx node[10],node[12];
rz(0.5*pi) node[13];
cx node[18],node[15];
cx node[23],node[21];
rz(0.5*pi) node[10];
sx node[13];
cx node[15],node[18];
cx node[24],node[23];
sx node[10];
rz(0.5*pi) node[13];
cx node[18],node[15];
cx node[23],node[24];
rz(0.5*pi) node[10];
sx node[13];
cx node[17],node[18];
cx node[24],node[23];
sx node[10];
rz(1.3932107124147406*pi) node[13];
cx node[18],node[17];
cx node[23],node[24];
rz(0.06622862583262401*pi) node[10];
cx node[17],node[18];
cx node[25],node[24];
cx node[12],node[10];
cx node[21],node[18];
sx node[24];
cx node[10],node[12];
cx node[21],node[18];
rz(2.5*pi) node[24];
cx node[12],node[10];
cx node[18],node[21];
sx node[24];
cx node[7],node[10];
cx node[13],node[12];
cx node[21],node[18];
rz(1.5*pi) node[24];
rz(0.5*pi) node[7];
cx node[12],node[13];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
sx node[7];
cx node[13],node[12];
cx node[23],node[21];
cx node[24],node[25];
rz(2.5*pi) node[7];
cx node[15],node[12];
cx node[21],node[23];
cx node[25],node[24];
sx node[7];
cx node[12],node[15];
cx node[23],node[21];
rz(0.2010865453887557*pi) node[7];
cx node[15],node[12];
cx node[24],node[23];
cx node[10],node[7];
cx node[12],node[15];
cx node[23],node[24];
cx node[7],node[10];
cx node[12],node[13];
cx node[18],node[15];
cx node[24],node[23];
cx node[10],node[7];
cx node[18],node[15];
cx node[23],node[24];
cx node[6],node[7];
cx node[12],node[10];
cx node[15],node[18];
cx node[25],node[24];
cx node[4],node[7];
rz(0.5*pi) node[6];
cx node[12],node[10];
cx node[18],node[15];
sx node[24];
rz(0.5*pi) node[4];
sx node[6];
cx node[10],node[12];
cx node[17],node[18];
rz(2.5*pi) node[24];
sx node[4];
rz(2.5*pi) node[6];
cx node[12],node[10];
cx node[18],node[17];
sx node[24];
rz(2.5*pi) node[4];
sx node[6];
cx node[13],node[12];
cx node[17],node[18];
rz(1.5*pi) node[24];
sx node[4];
rz(3.9443494957775664*pi) node[6];
cx node[12],node[13];
cx node[21],node[18];
cx node[25],node[24];
rz(0.03568270698984177*pi) node[4];
cx node[13],node[12];
cx node[21],node[18];
cx node[24],node[25];
cx node[7],node[4];
cx node[15],node[12];
cx node[18],node[21];
cx node[25],node[24];
cx node[4],node[7];
cx node[12],node[15];
cx node[21],node[18];
cx node[7],node[4];
cx node[15],node[12];
cx node[18],node[17];
cx node[23],node[21];
cx node[1],node[4];
cx node[6],node[7];
cx node[12],node[15];
cx node[23],node[21];
rz(0.5*pi) node[1];
rz(3.9470908578721486*pi) node[4];
cx node[7],node[6];
cx node[12],node[13];
cx node[18],node[15];
cx node[21],node[23];
sx node[1];
cx node[6],node[7];
cx node[18],node[15];
cx node[23],node[21];
rz(2.5*pi) node[1];
cx node[10],node[7];
cx node[15],node[18];
cx node[24],node[23];
sx node[1];
cx node[10],node[7];
cx node[18],node[15];
cx node[23],node[24];
rz(0.894298489544948*pi) node[1];
cx node[7],node[10];
cx node[17],node[18];
cx node[24],node[23];
cx node[1],node[4];
cx node[10],node[7];
cx node[18],node[17];
cx node[23],node[24];
cx node[4],node[1];
cx node[7],node[6];
cx node[12],node[10];
cx node[17],node[18];
cx node[25],node[24];
cx node[1],node[4];
cx node[12],node[10];
cx node[21],node[18];
sx node[24];
cx node[7],node[4];
cx node[10],node[12];
cx node[21],node[18];
rz(2.5*pi) node[24];
cx node[7],node[4];
cx node[12],node[10];
cx node[18],node[21];
sx node[24];
cx node[4],node[7];
cx node[13],node[12];
cx node[21],node[18];
rz(1.5*pi) node[24];
cx node[7],node[4];
cx node[12],node[13];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
cx node[4],node[1];
cx node[6],node[7];
cx node[13],node[12];
cx node[23],node[21];
cx node[24],node[25];
sx node[4];
cx node[7],node[6];
cx node[15],node[12];
cx node[21],node[23];
cx node[25],node[24];
rz(3.9688469481094915*pi) node[4];
cx node[6],node[7];
cx node[12],node[15];
cx node[23],node[21];
sx node[4];
cx node[10],node[7];
cx node[15],node[12];
cx node[24],node[23];
rz(1.0*pi) node[4];
cx node[10],node[7];
cx node[12],node[15];
cx node[23],node[24];
cx node[1],node[4];
cx node[7],node[10];
cx node[12],node[13];
cx node[18],node[15];
cx node[24],node[23];
cx node[4],node[1];
cx node[10],node[7];
cx node[18],node[15];
cx node[23],node[24];
cx node[1],node[4];
cx node[7],node[6];
cx node[12],node[10];
cx node[15],node[18];
cx node[25],node[24];
cx node[7],node[4];
cx node[12],node[10];
cx node[18],node[15];
sx node[24];
sx node[7];
cx node[10],node[12];
cx node[17],node[18];
rz(2.5*pi) node[24];
rz(3.1910336539930855*pi) node[7];
cx node[12],node[10];
cx node[18],node[17];
sx node[24];
sx node[7];
cx node[13],node[12];
cx node[17],node[18];
rz(1.5*pi) node[24];
rz(1.0*pi) node[7];
cx node[12],node[13];
cx node[21],node[18];
cx node[25],node[24];
cx node[6],node[7];
cx node[13],node[12];
cx node[21],node[18];
cx node[24],node[25];
cx node[7],node[6];
cx node[15],node[12];
cx node[18],node[21];
cx node[25],node[24];
cx node[6],node[7];
cx node[12],node[15];
cx node[21],node[18];
cx node[10],node[7];
cx node[15],node[12];
cx node[18],node[17];
cx node[23],node[21];
cx node[10],node[7];
cx node[12],node[15];
cx node[23],node[21];
cx node[7],node[10];
cx node[12],node[13];
cx node[18],node[15];
cx node[21],node[23];
cx node[10],node[7];
cx node[18],node[15];
cx node[23],node[21];
cx node[7],node[4];
cx node[12],node[10];
cx node[15],node[18];
cx node[24],node[23];
sx node[7];
cx node[12],node[10];
cx node[18],node[15];
cx node[23],node[24];
rz(0.778952095467909*pi) node[7];
cx node[10],node[12];
cx node[17],node[18];
cx node[24],node[23];
sx node[7];
cx node[12],node[10];
cx node[18],node[17];
cx node[23],node[24];
rz(1.0*pi) node[7];
cx node[13],node[12];
cx node[17],node[18];
cx node[25],node[24];
cx node[4],node[7];
cx node[12],node[13];
cx node[21],node[18];
sx node[24];
cx node[7],node[4];
cx node[13],node[12];
cx node[21],node[18];
rz(2.5*pi) node[24];
cx node[4],node[7];
cx node[15],node[12];
cx node[18],node[21];
sx node[24];
cx node[10],node[7];
cx node[12],node[15];
cx node[21],node[18];
rz(1.5*pi) node[24];
sx node[10];
cx node[15],node[12];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
rz(2.465637528310913*pi) node[10];
cx node[12],node[15];
cx node[23],node[21];
cx node[24],node[25];
sx node[10];
cx node[12],node[13];
cx node[18],node[15];
cx node[21],node[23];
cx node[25],node[24];
rz(1.0*pi) node[10];
cx node[18],node[15];
cx node[23],node[21];
cx node[7],node[10];
cx node[15],node[18];
cx node[24],node[23];
cx node[10],node[7];
cx node[18],node[15];
cx node[23],node[24];
cx node[7],node[10];
cx node[17],node[18];
cx node[24],node[23];
cx node[12],node[10];
cx node[18],node[17];
cx node[23],node[24];
sx node[12];
cx node[17],node[18];
cx node[25],node[24];
rz(2.4657258302053173*pi) node[12];
cx node[21],node[18];
sx node[24];
sx node[12];
cx node[21],node[18];
rz(2.5*pi) node[24];
rz(1.0*pi) node[12];
cx node[18],node[21];
sx node[24];
cx node[13],node[12];
cx node[21],node[18];
rz(1.5*pi) node[24];
cx node[12],node[13];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
cx node[13],node[12];
cx node[23],node[21];
cx node[24],node[25];
cx node[15],node[12];
cx node[21],node[23];
cx node[25],node[24];
cx node[15],node[12];
cx node[23],node[21];
cx node[12],node[15];
cx node[24],node[23];
cx node[15],node[12];
cx node[23],node[24];
cx node[12],node[10];
cx node[18],node[15];
cx node[24],node[23];
sx node[12];
cx node[18],node[15];
cx node[23],node[24];
rz(2.088441838446564*pi) node[12];
cx node[15],node[18];
cx node[25],node[24];
sx node[12];
cx node[18],node[15];
sx node[24];
rz(1.0*pi) node[12];
cx node[17],node[18];
rz(2.5*pi) node[24];
cx node[10],node[12];
cx node[18],node[17];
sx node[24];
cx node[12],node[10];
cx node[17],node[18];
rz(1.5*pi) node[24];
cx node[10],node[12];
cx node[21],node[18];
cx node[25],node[24];
cx node[15],node[12];
cx node[21],node[18];
cx node[24],node[25];
sx node[15];
cx node[18],node[21];
cx node[25],node[24];
rz(2.0030430055293698*pi) node[15];
cx node[21],node[18];
sx node[15];
cx node[18],node[17];
cx node[23],node[21];
rz(1.0*pi) node[15];
cx node[23],node[21];
cx node[12],node[15];
cx node[21],node[23];
cx node[15],node[12];
cx node[23],node[21];
cx node[12],node[15];
cx node[24],node[23];
cx node[18],node[15];
cx node[23],node[24];
sx node[18];
cx node[24],node[23];
rz(0.9138663786362524*pi) node[18];
cx node[23],node[24];
sx node[18];
cx node[25],node[24];
rz(1.0*pi) node[18];
sx node[24];
cx node[17],node[18];
rz(2.5*pi) node[24];
cx node[18],node[17];
sx node[24];
cx node[17],node[18];
rz(1.5*pi) node[24];
cx node[21],node[18];
cx node[25],node[24];
cx node[21],node[18];
cx node[24],node[25];
cx node[18],node[21];
cx node[25],node[24];
cx node[21],node[18];
cx node[18],node[15];
cx node[23],node[21];
sx node[18];
cx node[23],node[21];
rz(2.3455124512711167*pi) node[18];
cx node[21],node[23];
sx node[18];
cx node[23],node[21];
rz(1.0*pi) node[18];
cx node[24],node[23];
cx node[15],node[18];
cx node[25],node[24];
cx node[18],node[15];
cx node[24],node[23];
cx node[15],node[18];
cx node[25],node[24];
cx node[21],node[18];
cx node[24],node[23];
sx node[21];
sx node[23];
rz(1.890695573195842*pi) node[21];
rz(2.5*pi) node[23];
sx node[21];
sx node[23];
rz(1.0*pi) node[21];
rz(1.5*pi) node[23];
cx node[18],node[21];
cx node[21],node[18];
cx node[18],node[21];
cx node[21],node[23];
cx node[23],node[21];
cx node[21],node[23];
cx node[24],node[23];
sx node[24];
rz(0.38864712124747447*pi) node[24];
sx node[24];
rz(1.0*pi) node[24];
cx node[25],node[24];
cx node[24],node[25];
cx node[25],node[24];
cx node[24],node[23];
cx node[21],node[23];
sx node[24];
sx node[21];
rz(1.3593987397244551*pi) node[23];
rz(1.2465889258217304*pi) node[24];
rz(3.101057860769994*pi) node[21];
sx node[23];
sx node[24];
sx node[21];
rz(2.5*pi) node[23];
rz(1.0*pi) node[24];
rz(1.0*pi) node[21];
sx node[23];
rz(1.5*pi) node[23];
barrier node[1],node[6],node[4],node[7],node[13],node[10],node[12],node[17],node[15],node[18],node[25],node[24],node[21],node[23];
measure node[1] -> meas[0];
measure node[6] -> meas[1];
measure node[4] -> meas[2];
measure node[7] -> meas[3];
measure node[13] -> meas[4];
measure node[10] -> meas[5];
measure node[12] -> meas[6];
measure node[17] -> meas[7];
measure node[15] -> meas[8];
measure node[18] -> meas[9];
measure node[25] -> meas[10];
measure node[24] -> meas[11];
measure node[21] -> meas[12];
measure node[23] -> meas[13];
