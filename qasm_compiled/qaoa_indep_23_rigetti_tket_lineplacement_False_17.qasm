OPENQASM 2.0;
include "qelib1.inc";

qreg node[64];
creg meas[23];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[48];
rz(0.5*pi) node[55];
rz(0.5*pi) node[63];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[4];
rx(0.5*pi) node[8];
rx(0.5*pi) node[9];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rx(0.5*pi) node[16];
rx(0.5*pi) node[17];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rx(0.5*pi) node[20];
rx(0.5*pi) node[21];
rx(0.5*pi) node[22];
rx(0.5*pi) node[23];
rx(0.5*pi) node[48];
rx(0.5*pi) node[55];
rx(0.5*pi) node[63];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[48];
rz(0.5*pi) node[55];
rz(0.5*pi) node[63];
rz(0.5*pi) node[2];
rz(0.5*pi) node[10];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[20];
rz(0.5*pi) node[22];
rx(0.5*pi) node[2];
rx(0.5*pi) node[10];
rx(0.5*pi) node[14];
rx(0.5*pi) node[16];
rx(0.5*pi) node[18];
rx(0.5*pi) node[20];
rx(0.5*pi) node[22];
rz(0.5*pi) node[2];
rz(0.5*pi) node[10];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[20];
rz(0.5*pi) node[22];
cz node[1],node[2];
cz node[11],node[10];
cz node[13],node[14];
cz node[23],node[16];
rz(0.5*pi) node[2];
rz(0.5*pi) node[10];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rx(0.5*pi) node[2];
rx(0.5*pi) node[10];
rx(0.5*pi) node[14];
rx(0.5*pi) node[16];
rz(0.5*pi) node[2];
rz(0.5*pi) node[10];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rz(1.4968079523298883*pi) node[2];
rz(1.4968079523298883*pi) node[10];
rz(1.4968079523298883*pi) node[14];
rz(1.4968079523298883*pi) node[16];
rz(0.5*pi) node[2];
rz(0.5*pi) node[10];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rx(0.5*pi) node[2];
rx(0.5*pi) node[10];
rx(0.5*pi) node[14];
rx(0.5*pi) node[16];
rz(0.5*pi) node[2];
rz(0.5*pi) node[10];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
cz node[1],node[2];
cz node[11],node[10];
cz node[13],node[14];
cz node[23],node[16];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
cz node[23],node[22];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[16];
rz(0.5*pi) node[22];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rx(0.5*pi) node[22];
cz node[0],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[10];
cz node[12],node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rz(0.5*pi) node[22];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[10];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[16];
rz(1.4968079523298883*pi) node[22];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[10];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rz(0.5*pi) node[22];
rz(0.5*pi) node[1];
cz node[3],node[2];
cz node[9],node[10];
rz(0.5*pi) node[13];
cz node[15],node[14];
cz node[17],node[16];
rx(0.5*pi) node[22];
rz(1.4968079523298883*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[10];
rz(1.4968079523298883*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rz(0.5*pi) node[22];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[10];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[16];
cz node[23],node[22];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[10];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rz(0.5*pi) node[22];
rx(3.5151603506619153*pi) node[23];
rz(0.5*pi) node[1];
rz(1.4968079523298883*pi) node[2];
rz(1.4968079523298883*pi) node[10];
rz(0.5*pi) node[13];
rz(1.4968079523298883*pi) node[14];
rz(1.4968079523298883*pi) node[16];
rx(0.5*pi) node[22];
cz node[0],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[10];
cz node[12],node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rz(0.5*pi) node[22];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[10];
cz node[12],node[11];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[16];
rz(0.5*pi) node[22];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rx(0.5*pi) node[22];
rz(0.5*pi) node[1];
cz node[3],node[2];
cz node[9],node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[13];
cz node[15],node[14];
cz node[17],node[16];
rz(0.5*pi) node[22];
rx(1.5151603506619156*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
cz node[9],node[22];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(1.5151603506619156*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
cz node[17],node[18];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[10];
rz(1.4968079523298883*pi) node[11];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rx(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[22];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rx(0.5*pi) node[18];
rx(0.5*pi) node[22];
rx(3.5151603506619153*pi) node[2];
cz node[4],node[3];
cz node[8],node[15];
rx(3.5151603506619153*pi) node[10];
rx(0.5*pi) node[11];
rx(3.5151603506619153*pi) node[14];
rx(3.5151603506619153*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[22];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rz(1.4968079523298883*pi) node[18];
rz(1.4968079523298883*pi) node[22];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[10];
cz node[12],node[11];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rx(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[22];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(3.5151603506619153*pi) node[12];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rx(0.5*pi) node[18];
rx(0.5*pi) node[22];
cz node[1],node[2];
rz(1.4968079523298883*pi) node[3];
rx(0.5*pi) node[11];
cz node[13],node[14];
rz(1.4968079523298883*pi) node[15];
cz node[23],node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[22];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
cz node[9],node[22];
rz(0.5*pi) node[11];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
cz node[17],node[18];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(3.5151603506619153*pi) node[9];
rx(1.5151603506619156*pi) node[11];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rx(0.5*pi) node[16];
rx(3.5151603506619153*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[22];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
cz node[11],node[10];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rx(0.5*pi) node[18];
rx(0.5*pi) node[22];
rz(0.20108998815487666*pi) node[2];
cz node[4],node[3];
cz node[8],node[15];
rz(0.5*pi) node[10];
rz(0.20108998815487666*pi) node[14];
rz(0.20108998815487666*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[22];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[10];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rz(0.5*pi) node[18];
rx(3.5151603506619153*pi) node[22];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[10];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rx(0.5*pi) node[16];
rx(0.5*pi) node[18];
rz(0.5*pi) node[22];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.20108998815487666*pi) node[10];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rz(0.5*pi) node[18];
rx(0.5*pi) node[22];
cz node[1],node[2];
rx(1.5151603506619156*pi) node[3];
rz(0.5*pi) node[10];
cz node[13],node[14];
rx(1.5151603506619156*pi) node[15];
cz node[23],node[16];
cz node[19],node[18];
rz(0.5*pi) node[22];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[10];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rz(0.5*pi) node[18];
cz node[23],node[22];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[10];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[16];
rx(0.5*pi) node[18];
rz(0.5*pi) node[22];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[11],node[10];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rz(0.5*pi) node[18];
rx(0.5*pi) node[22];
cz node[0],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
cz node[12],node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rz(1.4968079523298883*pi) node[18];
rz(0.5*pi) node[22];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(0.20108998815487666*pi) node[22];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rx(0.5*pi) node[18];
rz(0.5*pi) node[22];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[3],node[2];
rz(0.5*pi) node[10];
rz(0.5*pi) node[13];
cz node[15],node[14];
cz node[17],node[16];
rz(0.5*pi) node[18];
rx(0.5*pi) node[22];
cz node[1],node[0];
rz(0.5*pi) node[2];
rx(0.5*pi) node[10];
rz(0.20108998815487666*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
cz node[19],node[18];
rz(0.5*pi) node[22];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[10];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[16];
rz(0.5*pi) node[18];
cz node[19],node[20];
cz node[23],node[22];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[9],node[10];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rx(0.5*pi) node[18];
rz(0.5*pi) node[20];
rz(0.5*pi) node[22];
rx(0.11743560232400796*pi) node[23];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.20108998815487666*pi) node[2];
rz(0.5*pi) node[10];
rz(0.5*pi) node[13];
rz(0.20108998815487666*pi) node[14];
rz(0.20108998815487666*pi) node[16];
rz(0.5*pi) node[18];
rx(0.5*pi) node[20];
rx(0.5*pi) node[22];
cz node[0],node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[10];
cz node[12],node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rx(3.5151603506619153*pi) node[18];
rz(0.5*pi) node[20];
rz(0.5*pi) node[22];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[10];
cz node[12],node[11];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(1.4968079523298883*pi) node[20];
rz(0.5*pi) node[22];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.20108998815487666*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
rx(0.5*pi) node[18];
rz(0.5*pi) node[20];
rx(0.5*pi) node[22];
rz(0.5*pi) node[1];
cz node[3],node[2];
rz(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[13];
cz node[15],node[14];
cz node[17],node[16];
rz(0.5*pi) node[18];
rx(0.5*pi) node[20];
rz(0.5*pi) node[22];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(2.117435602324008*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
cz node[17],node[18];
rz(0.5*pi) node[20];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[10];
rz(0.20108998815487666*pi) node[11];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rx(0.5*pi) node[16];
rz(0.5*pi) node[18];
cz node[19],node[20];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
cz node[9],node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rx(0.5*pi) node[18];
rx(3.5151603506619153*pi) node[19];
rz(0.5*pi) node[20];
rx(2.117435602324008*pi) node[2];
cz node[4],node[3];
cz node[9],node[22];
rz(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[13];
rx(2.117435602324008*pi) node[14];
rx(2.117435602324008*pi) node[16];
rz(0.5*pi) node[18];
rx(0.5*pi) node[20];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rx(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[14];
rz(0.20108998815487666*pi) node[18];
rz(0.5*pi) node[20];
rz(0.5*pi) node[22];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[4];
rz(0.5*pi) node[10];
cz node[12],node[11];
rx(0.5*pi) node[14];
rz(0.5*pi) node[18];
rz(0.5*pi) node[20];
rx(0.5*pi) node[22];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rx(2.117435602324008*pi) node[10];
rz(0.5*pi) node[11];
rx(0.11743560232400796*pi) node[12];
rz(0.5*pi) node[14];
rx(0.5*pi) node[18];
rx(0.5*pi) node[20];
rz(0.5*pi) node[22];
cz node[1],node[14];
cz node[3],node[4];
rz(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[18];
rz(0.5*pi) node[20];
rz(0.20108998815487666*pi) node[22];
rz(0.5*pi) node[1];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rx(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[12];
rz(0.5*pi) node[14];
cz node[17],node[18];
cz node[63],node[20];
rz(0.5*pi) node[22];
rx(0.5*pi) node[1];
rx(0.5*pi) node[3];
rx(0.5*pi) node[4];
rz(0.5*pi) node[10];
rx(2.117435602324008*pi) node[11];
rz(0.5*pi) node[12];
rx(0.5*pi) node[14];
rx(0.11743560232400796*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[20];
rx(0.5*pi) node[22];
rz(0.5*pi) node[1];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[11];
rz(0.5*pi) node[14];
rx(0.5*pi) node[18];
rx(0.5*pi) node[20];
rz(0.5*pi) node[22];
cz node[14],node[1];
cz node[4],node[3];
cz node[9],node[22];
rx(0.5*pi) node[11];
rz(0.5*pi) node[18];
rz(0.5*pi) node[20];
rz(0.5*pi) node[1];
rz(0.5*pi) node[3];
rx(0.11743560232400796*pi) node[9];
rz(0.5*pi) node[11];
rz(0.5*pi) node[14];
rz(0.5*pi) node[18];
rz(1.4968079523298883*pi) node[20];
rz(0.5*pi) node[22];
rx(0.5*pi) node[1];
rx(0.5*pi) node[3];
rz(0.5*pi) node[9];
rx(0.5*pi) node[14];
rx(0.5*pi) node[18];
rz(0.5*pi) node[20];
rx(0.5*pi) node[22];
rz(0.5*pi) node[1];
rz(0.5*pi) node[3];
rx(0.5*pi) node[9];
rz(0.5*pi) node[14];
rz(0.5*pi) node[18];
rx(0.5*pi) node[20];
rz(0.5*pi) node[22];
cz node[1],node[14];
cz node[3],node[2];
rz(0.5*pi) node[9];
cz node[19],node[18];
rz(0.5*pi) node[20];
rx(2.117435602324008*pi) node[22];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[14];
rz(0.5*pi) node[18];
cz node[63],node[20];
rz(0.5*pi) node[22];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[14];
rx(0.5*pi) node[18];
rz(0.5*pi) node[20];
rx(0.5*pi) node[22];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[14];
rz(0.5*pi) node[18];
rx(0.5*pi) node[20];
rz(0.5*pi) node[22];
cz node[0],node[1];
cz node[2],node[3];
rz(0.20108998815487666*pi) node[18];
rz(0.5*pi) node[20];
cz node[21],node[22];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[18];
rx(3.5151603506619153*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[18];
rz(0.5*pi) node[20];
rx(0.5*pi) node[21];
rx(0.5*pi) node[22];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[18];
rx(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
cz node[1],node[0];
cz node[3],node[2];
cz node[19],node[18];
rz(0.5*pi) node[20];
cz node[22],node[21];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[18];
cz node[19],node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[18];
rz(0.5*pi) node[20];
rx(0.5*pi) node[21];
rx(0.5*pi) node[22];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[18];
rx(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
cz node[0],node[1];
cz node[2],node[13];
cz node[4],node[3];
rx(2.117435602324008*pi) node[18];
rz(0.5*pi) node[20];
cz node[21],node[22];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[13];
rz(0.20108998815487666*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[4];
rx(0.5*pi) node[13];
rz(0.5*pi) node[20];
rx(0.5*pi) node[21];
rx(0.5*pi) node[22];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[13];
rx(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[1];
cz node[13],node[2];
cz node[3],node[4];
cz node[22],node[9];
rz(0.5*pi) node[20];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[9];
rz(0.5*pi) node[13];
cz node[19],node[20];
rz(0.5*pi) node[22];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[4];
rx(0.5*pi) node[9];
rx(0.5*pi) node[13];
rx(0.11743560232400796*pi) node[19];
rz(0.5*pi) node[20];
rx(0.5*pi) node[22];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[9];
rz(0.5*pi) node[13];
rx(0.5*pi) node[20];
rz(0.5*pi) node[22];
cz node[2],node[13];
cz node[4],node[3];
cz node[9],node[22];
rz(0.5*pi) node[20];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[9];
rz(0.5*pi) node[13];
rz(0.5*pi) node[20];
rz(0.5*pi) node[22];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[9];
rx(0.5*pi) node[13];
rx(0.5*pi) node[20];
rx(0.5*pi) node[22];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[9];
rz(0.5*pi) node[13];
rz(0.5*pi) node[20];
rz(0.5*pi) node[22];
cz node[3],node[2];
cz node[22],node[9];
cz node[13],node[12];
cz node[63],node[20];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[9];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[20];
rz(0.5*pi) node[63];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[9];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[20];
rx(0.5*pi) node[63];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[9];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[20];
rz(0.5*pi) node[63];
cz node[2],node[3];
rz(0.5*pi) node[9];
cz node[12],node[13];
cz node[20],node[63];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[9];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[20];
rz(0.5*pi) node[63];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[9];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[20];
rx(0.5*pi) node[63];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
cz node[8],node[9];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[20];
rz(0.5*pi) node[63];
cz node[3],node[2];
rz(0.5*pi) node[9];
cz node[13],node[12];
cz node[63],node[20];
rz(0.5*pi) node[2];
rx(0.5*pi) node[9];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[20];
rx(0.5*pi) node[2];
rz(0.5*pi) node[9];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[20];
rz(0.5*pi) node[2];
rz(1.4968079523298883*pi) node[9];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[20];
cz node[2],node[13];
rz(0.5*pi) node[9];
cz node[12],node[11];
cz node[20],node[21];
rz(0.5*pi) node[2];
rx(0.5*pi) node[9];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rx(0.5*pi) node[2];
rz(0.5*pi) node[9];
rx(0.5*pi) node[11];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[20];
rx(0.5*pi) node[21];
rz(0.5*pi) node[2];
cz node[8],node[9];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
cz node[13],node[2];
rx(3.5151603506619153*pi) node[8];
rz(0.5*pi) node[9];
cz node[11],node[12];
cz node[21],node[20];
rz(0.5*pi) node[2];
cz node[8],node[15];
rx(0.5*pi) node[9];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rx(0.5*pi) node[2];
rz(0.5*pi) node[9];
rx(0.5*pi) node[11];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rz(0.5*pi) node[15];
rx(0.5*pi) node[20];
rx(0.5*pi) node[21];
rz(0.5*pi) node[2];
cz node[9],node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rx(0.5*pi) node[15];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
cz node[2],node[13];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
cz node[12],node[11];
rz(0.5*pi) node[15];
cz node[20],node[21];
rx(0.5*pi) node[9];
rx(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.20108998815487666*pi) node[15];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rz(0.5*pi) node[15];
rx(0.5*pi) node[20];
rx(0.5*pi) node[21];
cz node[10],node[9];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rx(0.5*pi) node[15];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[15];
cz node[63],node[20];
cz node[8],node[15];
rx(0.5*pi) node[9];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[20];
rz(0.5*pi) node[63];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[15];
rx(0.5*pi) node[20];
rx(0.5*pi) node[63];
cz node[9],node[10];
rx(0.5*pi) node[15];
rz(0.5*pi) node[20];
rz(0.5*pi) node[63];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[15];
cz node[20],node[63];
rx(0.5*pi) node[9];
rx(0.5*pi) node[10];
rx(2.117435602324008*pi) node[15];
rz(0.5*pi) node[20];
rz(0.5*pi) node[63];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rx(0.5*pi) node[20];
rx(0.5*pi) node[63];
cz node[8],node[9];
cz node[10],node[11];
rz(0.5*pi) node[20];
rz(0.5*pi) node[63];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
cz node[63],node[20];
rx(0.5*pi) node[8];
rx(0.5*pi) node[9];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[20];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[20];
cz node[9],node[8];
cz node[11],node[10];
rz(0.5*pi) node[20];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[20];
rx(0.5*pi) node[8];
rx(0.5*pi) node[9];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(0.5*pi) node[20];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[20];
cz node[8],node[9];
cz node[10],node[11];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[9];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
cz node[21],node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[10];
cz node[48],node[11];
rz(1.4968079523298883*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[10];
rz(1.4968079523298883*pi) node[11];
cz node[21],node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(3.5151603506619153*pi) node[21];
rx(0.5*pi) node[10];
rz(0.5*pi) node[11];
cz node[21],node[20];
rz(0.5*pi) node[10];
cz node[48],node[11];
rz(0.5*pi) node[20];
rx(1.5151603506619156*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[20];
cz node[48],node[55];
rx(0.5*pi) node[11];
rz(0.5*pi) node[20];
rz(0.5*pi) node[48];
rz(0.5*pi) node[55];
rz(0.5*pi) node[11];
rz(0.20108998815487666*pi) node[20];
rx(0.5*pi) node[48];
rx(0.5*pi) node[55];
rx(3.5151603506619153*pi) node[11];
rz(0.5*pi) node[20];
rz(0.5*pi) node[48];
rz(0.5*pi) node[55];
rz(0.5*pi) node[11];
rx(0.5*pi) node[20];
cz node[55],node[48];
rx(0.5*pi) node[11];
rz(0.5*pi) node[20];
rz(0.5*pi) node[48];
rz(0.5*pi) node[55];
rz(0.5*pi) node[11];
cz node[21],node[20];
rx(0.5*pi) node[48];
rx(0.5*pi) node[55];
cz node[10],node[11];
rz(0.5*pi) node[20];
rz(0.5*pi) node[48];
rz(0.5*pi) node[55];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[20];
cz node[48],node[55];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[20];
rz(0.5*pi) node[55];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(2.117435602324008*pi) node[20];
rx(0.5*pi) node[55];
cz node[11],node[10];
rz(0.5*pi) node[55];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
cz node[55],node[12];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[55];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[12];
rx(0.5*pi) node[55];
cz node[10],node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[55];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
cz node[12],node[55];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[55];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[12];
rx(0.5*pi) node[55];
cz node[9],node[10];
rz(0.5*pi) node[12];
rz(0.5*pi) node[55];
rz(0.5*pi) node[10];
cz node[55],node[12];
rx(0.5*pi) node[10];
rz(0.5*pi) node[12];
rz(0.5*pi) node[10];
rx(0.5*pi) node[12];
rz(0.20108998815487666*pi) node[10];
rz(0.5*pi) node[12];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
rx(0.5*pi) node[10];
rx(0.5*pi) node[12];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
cz node[9],node[10];
cz node[13],node[12];
rx(0.11743560232400796*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rx(0.5*pi) node[10];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
cz node[12],node[13];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
cz node[13],node[12];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[12];
cz node[14],node[13];
rx(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[12];
rx(0.5*pi) node[13];
cz node[11],node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[12];
rz(1.4968079523298883*pi) node[13];
rx(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[12];
rx(0.5*pi) node[13];
rz(0.20108998815487666*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[12];
cz node[14],node[13];
rx(0.5*pi) node[12];
rz(0.5*pi) node[13];
rx(3.5151603506619153*pi) node[14];
cz node[14],node[1];
rz(0.5*pi) node[12];
rx(0.5*pi) node[13];
rz(0.5*pi) node[1];
cz node[11],node[12];
rz(0.5*pi) node[13];
rx(0.5*pi) node[1];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rx(1.5151603506619156*pi) node[13];
rz(0.5*pi) node[1];
rx(0.5*pi) node[11];
rx(0.5*pi) node[12];
rz(0.20108998815487666*pi) node[1];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[1];
cz node[10],node[11];
rx(2.117435602324008*pi) node[12];
rx(0.5*pi) node[1];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[1];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(0.5*pi) node[12];
cz node[14],node[1];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[1];
cz node[11],node[10];
rx(0.5*pi) node[1];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[1];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(2.117435602324008*pi) node[1];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
cz node[10],node[11];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
cz node[21],node[10];
cz node[11],node[12];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(0.5*pi) node[12];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.20108998815487666*pi) node[10];
cz node[12],node[11];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(0.5*pi) node[12];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
cz node[21],node[10];
cz node[11],node[12];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
rx(0.11743560232400796*pi) node[21];
rx(0.5*pi) node[10];
rx(0.5*pi) node[12];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
rx(2.117435602324008*pi) node[10];
rz(0.5*pi) node[12];
rx(0.5*pi) node[12];
rz(0.5*pi) node[12];
cz node[13],node[12];
rz(0.5*pi) node[12];
rx(0.5*pi) node[12];
rz(0.5*pi) node[12];
rz(0.20108998815487666*pi) node[12];
rz(0.5*pi) node[12];
rx(0.5*pi) node[12];
rz(0.5*pi) node[12];
cz node[13],node[12];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rx(2.117435602324008*pi) node[12];
cz node[14],node[13];
rz(0.5*pi) node[13];
rx(0.5*pi) node[13];
rz(0.5*pi) node[13];
rz(0.20108998815487666*pi) node[13];
rz(0.5*pi) node[13];
rx(0.5*pi) node[13];
rz(0.5*pi) node[13];
cz node[14],node[13];
rz(0.5*pi) node[13];
rx(0.11743560232400796*pi) node[14];
rx(0.5*pi) node[13];
rz(0.5*pi) node[13];
rx(2.117435602324008*pi) node[13];
barrier node[14],node[21],node[19],node[13],node[9],node[22],node[10],node[11],node[12],node[15],node[20],node[17],node[1],node[18],node[4],node[23],node[2],node[55],node[3],node[63],node[8],node[0],node[16];
measure node[14] -> meas[0];
measure node[21] -> meas[1];
measure node[19] -> meas[2];
measure node[13] -> meas[3];
measure node[9] -> meas[4];
measure node[22] -> meas[5];
measure node[10] -> meas[6];
measure node[11] -> meas[7];
measure node[12] -> meas[8];
measure node[15] -> meas[9];
measure node[20] -> meas[10];
measure node[17] -> meas[11];
measure node[1] -> meas[12];
measure node[18] -> meas[13];
measure node[4] -> meas[14];
measure node[23] -> meas[15];
measure node[2] -> meas[16];
measure node[55] -> meas[17];
measure node[3] -> meas[18];
measure node[63] -> meas[19];
measure node[8] -> meas[20];
measure node[0] -> meas[21];
measure node[16] -> meas[22];
