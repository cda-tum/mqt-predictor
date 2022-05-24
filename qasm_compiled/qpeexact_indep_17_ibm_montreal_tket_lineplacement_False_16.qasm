OPENQASM 2.0;
include "qelib1.inc";

qreg node[19];
creg c[16];
sx node[0];
rz(0.5713958797050214*pi) node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
sx node[7];
rz(1.5*pi) node[8];
rz(3.5*pi) node[9];
sx node[10];
rz(3.5*pi) node[11];
sx node[12];
sx node[13];
sx node[14];
rz(0.5*pi) node[15];
rz(3.5*pi) node[0];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
rz(3.5*pi) node[6];
rz(3.5*pi) node[7];
sx node[8];
x node[9];
rz(3.5*pi) node[10];
x node[11];
sx node[15];
sx node[0];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
sx node[7];
rz(3.5*pi) node[8];
rz(0.5*pi) node[9];
sx node[10];
rz(0.5*pi) node[11];
rz(3.25*pi) node[15];
rz(1.0*pi) node[0];
rz(1.0*pi) node[2];
rz(1.0*pi) node[3];
rz(1.0*pi) node[4];
rz(1.0*pi) node[5];
rz(1.0*pi) node[6];
rz(1.0*pi) node[7];
sx node[8];
rz(1.0*pi) node[10];
sx node[15];
cx node[1],node[0];
rz(0.5*pi) node[8];
rz(0.5*pi) node[15];
rz(0.9286041185070255*pi) node[0];
x node[0];
rz(0.5*pi) node[0];
cx node[1],node[0];
rz(3.5714111402820365*pi) node[0];
cx node[1],node[2];
rz(0.35720825248066224*pi) node[2];
x node[2];
rz(0.5*pi) node[2];
cx node[1],node[2];
cx node[1],node[4];
rz(2.1428222650974624*pi) node[2];
cx node[3],node[2];
rz(0.21441650814442337*pi) node[4];
cx node[2],node[3];
x node[4];
cx node[3],node[2];
rz(0.5*pi) node[4];
cx node[1],node[4];
cx node[5],node[3];
cx node[1],node[2];
cx node[3],node[5];
rz(2.285644527011826*pi) node[4];
rz(0.9288330151461801*pi) node[2];
cx node[5],node[3];
cx node[7],node[4];
x node[2];
cx node[4],node[7];
cx node[8],node[5];
rz(0.5*pi) node[2];
cx node[7],node[4];
cx node[5],node[8];
cx node[1],node[2];
cx node[8],node[5];
cx node[10],node[7];
cx node[1],node[4];
rz(1.5712890551663192*pi) node[2];
cx node[7],node[10];
cx node[11],node[8];
cx node[3],node[2];
rz(0.3576660155195326*pi) node[4];
cx node[10],node[7];
cx node[8],node[11];
cx node[2],node[3];
x node[4];
cx node[11],node[8];
cx node[12],node[10];
cx node[3],node[2];
rz(0.5*pi) node[4];
cx node[10],node[12];
cx node[1],node[4];
cx node[5],node[3];
cx node[12],node[10];
cx node[1],node[2];
cx node[3],node[5];
rz(2.142578125105467*pi) node[4];
cx node[13],node[12];
rz(0.21533203103906517*pi) node[2];
cx node[5],node[3];
cx node[7],node[4];
cx node[12],node[13];
x node[2];
cx node[4],node[7];
cx node[8],node[5];
cx node[13],node[12];
rz(0.5*pi) node[2];
cx node[7],node[4];
cx node[5],node[8];
cx node[14],node[13];
cx node[1],node[2];
cx node[8],node[5];
cx node[6],node[7];
cx node[13],node[14];
cx node[1],node[4];
rz(2.2851562502109344*pi) node[2];
cx node[7],node[6];
cx node[9],node[8];
cx node[14],node[13];
cx node[3],node[2];
rz(0.9306640609354637*pi) node[4];
cx node[6],node[7];
cx node[8],node[9];
cx node[2],node[3];
x node[4];
cx node[9],node[8];
cx node[3],node[2];
rz(0.5*pi) node[4];
cx node[1],node[4];
cx node[5],node[3];
cx node[3],node[5];
rz(1.5703125015645356*pi) node[4];
cx node[5],node[3];
cx node[7],node[4];
cx node[4],node[7];
cx node[8],node[5];
cx node[7],node[4];
cx node[5],node[8];
cx node[1],node[4];
cx node[8],node[5];
cx node[10],node[7];
rz(0.36132812460514363*pi) node[4];
cx node[7],node[10];
x node[4];
cx node[10],node[7];
rz(0.5*pi) node[4];
cx node[12],node[10];
cx node[1],node[4];
cx node[10],node[12];
rz(0.5*pi) node[1];
rz(2.140625000394856*pi) node[4];
cx node[12],node[10];
sx node[1];
cx node[7],node[4];
cx node[15],node[12];
rz(3.5*pi) node[1];
cx node[4],node[7];
cx node[12],node[15];
sx node[1];
cx node[7],node[4];
cx node[15],node[12];
rz(1.0*pi) node[1];
cx node[10],node[7];
cx node[18],node[15];
cx node[1],node[2];
cx node[7],node[10];
cx node[15],node[18];
rz(1.5*pi) node[1];
cx node[10],node[7];
cx node[18],node[15];
sx node[1];
cx node[12],node[10];
rz(3.2773437539728114*pi) node[1];
cx node[10],node[12];
sx node[1];
cx node[12],node[10];
rz(1.5*pi) node[1];
cx node[13],node[12];
cx node[1],node[2];
cx node[12],node[13];
cx node[1],node[4];
sx node[2];
cx node[13],node[12];
rz(1.5*pi) node[1];
rz(3.5*pi) node[2];
sx node[1];
sx node[2];
rz(3.4453125068272046*pi) node[1];
rz(3.781250003972811*pi) node[2];
sx node[1];
cx node[3],node[2];
rz(1.5*pi) node[1];
cx node[2],node[3];
cx node[1],node[4];
cx node[3],node[2];
sx node[1];
cx node[5],node[3];
sx node[4];
cx node[1],node[2];
cx node[3],node[5];
rz(2.5*pi) node[4];
rz(1.5*pi) node[1];
cx node[5],node[3];
sx node[4];
sx node[1];
rz(1.062499993172795*pi) node[4];
cx node[5],node[8];
rz(0.10937499952686958*pi) node[1];
cx node[7],node[4];
cx node[8],node[5];
sx node[1];
cx node[4],node[7];
cx node[5],node[8];
rz(1.5*pi) node[1];
cx node[7],node[4];
cx node[1],node[2];
cx node[10],node[7];
sx node[2];
cx node[7],node[10];
rz(3.5*pi) node[2];
cx node[10],node[7];
sx node[2];
cx node[12],node[10];
rz(1.1249999995268687*pi) node[2];
cx node[10],node[12];
cx node[3],node[2];
cx node[12],node[10];
cx node[2],node[3];
cx node[15],node[12];
cx node[3],node[2];
cx node[12],node[15];
cx node[1],node[2];
cx node[15],node[12];
rz(1.5*pi) node[1];
sx node[1];
rz(0.21874999999999933*pi) node[1];
sx node[1];
rz(0.5*pi) node[1];
cx node[1],node[2];
sx node[1];
sx node[2];
cx node[1],node[4];
rz(3.5*pi) node[2];
rz(0.5*pi) node[1];
sx node[2];
sx node[1];
rz(1.25*pi) node[2];
rz(2.5625*pi) node[1];
sx node[1];
rz(1.0*pi) node[1];
cx node[1],node[4];
sx node[1];
sx node[4];
rz(0.75*pi) node[1];
rz(0.5*pi) node[4];
sx node[1];
sx node[4];
rz(1.5*pi) node[1];
cx node[7],node[4];
cx node[4],node[7];
cx node[7],node[4];
cx node[1],node[4];
cx node[10],node[7];
rz(0.5*pi) node[1];
cx node[7],node[10];
sx node[1];
cx node[10],node[7];
rz(2.875*pi) node[1];
cx node[12],node[10];
sx node[1];
cx node[10],node[12];
rz(1.0*pi) node[1];
cx node[12],node[10];
cx node[1],node[4];
sx node[1];
sx node[4];
rz(0.75*pi) node[1];
rz(2.5*pi) node[4];
sx node[1];
sx node[4];
rz(2.5*pi) node[1];
rz(1.75*pi) node[4];
cx node[7],node[4];
cx node[4],node[7];
cx node[7],node[4];
cx node[1],node[4];
cx node[10],node[7];
rz(1.5*pi) node[1];
cx node[7],node[10];
sx node[1];
cx node[10],node[7];
rz(3.25*pi) node[1];
sx node[1];
rz(1.0*pi) node[1];
cx node[1],node[4];
rz(0.5*pi) node[1];
sx node[4];
sx node[1];
rz(2.5*pi) node[4];
rz(3.5*pi) node[1];
sx node[4];
sx node[1];
rz(1.5*pi) node[4];
rz(1.0*pi) node[1];
cx node[1],node[4];
cx node[4],node[7];
cx node[1],node[4];
cx node[2],node[1];
cx node[4],node[7];
cx node[1],node[2];
cx node[4],node[7];
cx node[2],node[1];
rz(0.25*pi) node[7];
cx node[3],node[2];
cx node[4],node[7];
cx node[2],node[3];
rz(0.5*pi) node[4];
rz(3.75*pi) node[7];
cx node[3],node[2];
sx node[4];
cx node[10],node[7];
rz(3.5*pi) node[4];
rz(0.125*pi) node[7];
sx node[4];
cx node[10],node[7];
rz(1.0*pi) node[4];
rz(3.875*pi) node[7];
cx node[10],node[7];
cx node[7],node[10];
cx node[10],node[7];
cx node[7],node[4];
cx node[12],node[10];
rz(0.25*pi) node[4];
rz(0.0625*pi) node[10];
cx node[7],node[4];
cx node[12],node[10];
rz(3.75*pi) node[4];
rz(0.5*pi) node[7];
rz(3.9375*pi) node[10];
sx node[7];
rz(3.5*pi) node[7];
sx node[7];
rz(1.0*pi) node[7];
cx node[10],node[7];
cx node[7],node[10];
cx node[10],node[7];
cx node[7],node[4];
cx node[12],node[10];
cx node[4],node[7];
cx node[10],node[12];
cx node[7],node[4];
cx node[12],node[10];
cx node[1],node[4];
cx node[10],node[7];
rz(0.03125*pi) node[4];
rz(0.125*pi) node[7];
cx node[1],node[4];
cx node[10],node[7];
rz(3.96875*pi) node[4];
rz(3.875*pi) node[7];
cx node[10],node[12];
cx node[1],node[4];
rz(0.25*pi) node[12];
cx node[4],node[1];
cx node[10],node[12];
cx node[1],node[4];
rz(0.5*pi) node[10];
rz(3.75*pi) node[12];
cx node[2],node[1];
cx node[4],node[7];
sx node[10];
rz(0.015625*pi) node[1];
rz(0.0625*pi) node[7];
rz(3.5*pi) node[10];
cx node[2],node[1];
cx node[4],node[7];
sx node[10];
rz(3.984375*pi) node[1];
rz(3.9375*pi) node[7];
rz(1.0*pi) node[10];
cx node[1],node[4];
cx node[12],node[10];
cx node[4],node[1];
cx node[10],node[12];
cx node[1],node[4];
cx node[12],node[10];
cx node[7],node[4];
cx node[4],node[7];
cx node[7],node[4];
cx node[1],node[4];
cx node[10],node[7];
cx node[4],node[1];
cx node[7],node[10];
cx node[1],node[4];
cx node[10],node[7];
cx node[2],node[1];
cx node[4],node[7];
cx node[12],node[10];
rz(0.03125*pi) node[1];
rz(0.125*pi) node[7];
cx node[10],node[12];
cx node[2],node[1];
cx node[4],node[7];
cx node[12],node[10];
rz(3.96875*pi) node[1];
rz(3.875*pi) node[7];
cx node[15],node[12];
cx node[7],node[4];
rz(0.0078125*pi) node[12];
cx node[4],node[7];
cx node[15],node[12];
cx node[7],node[4];
rz(3.9921875*pi) node[12];
cx node[1],node[4];
cx node[7],node[10];
cx node[12],node[13];
cx node[4],node[1];
rz(0.25*pi) node[10];
cx node[13],node[12];
cx node[1],node[4];
cx node[7],node[10];
cx node[12],node[13];
cx node[2],node[1];
rz(0.5*pi) node[7];
rz(3.75*pi) node[10];
cx node[15],node[12];
cx node[13],node[14];
rz(0.0625*pi) node[1];
sx node[7];
cx node[12],node[15];
cx node[14],node[13];
cx node[2],node[1];
rz(3.5*pi) node[7];
cx node[15],node[12];
cx node[13],node[14];
rz(3.9375*pi) node[1];
sx node[7];
cx node[14],node[11];
rz(1.0*pi) node[7];
cx node[11],node[14];
cx node[10],node[7];
cx node[14],node[11];
cx node[7],node[10];
cx node[8],node[11];
cx node[10],node[7];
rz(0.00390625*pi) node[11];
cx node[7],node[4];
cx node[8],node[11];
cx node[4],node[7];
rz(3.99609375*pi) node[11];
cx node[7],node[4];
cx node[11],node[14];
cx node[4],node[1];
cx node[10],node[7];
cx node[14],node[11];
cx node[1],node[4];
cx node[7],node[10];
cx node[11],node[14];
cx node[4],node[1];
cx node[10],node[7];
cx node[8],node[11];
cx node[14],node[13];
cx node[2],node[1];
cx node[7],node[4];
cx node[11],node[8];
cx node[12],node[10];
cx node[13],node[14];
rz(0.125*pi) node[1];
cx node[4],node[7];
cx node[8],node[11];
rz(0.015625*pi) node[10];
cx node[14],node[13];
cx node[2],node[1];
cx node[7],node[4];
cx node[5],node[8];
cx node[12],node[10];
cx node[11],node[14];
rz(3.875*pi) node[1];
cx node[8],node[5];
rz(3.984375*pi) node[10];
cx node[14],node[11];
cx node[1],node[4];
cx node[5],node[8];
cx node[12],node[10];
cx node[11],node[14];
cx node[4],node[1];
cx node[8],node[11];
cx node[10],node[12];
cx node[1],node[4];
cx node[11],node[8];
cx node[12],node[10];
cx node[2],node[1];
cx node[10],node[7];
cx node[8],node[11];
cx node[12],node[13];
rz(0.25*pi) node[1];
rz(0.03125*pi) node[7];
cx node[9],node[8];
cx node[13],node[12];
cx node[2],node[1];
cx node[10],node[7];
cx node[8],node[9];
cx node[12],node[13];
rz(3.75*pi) node[1];
rz(0.5*pi) node[2];
rz(3.96875*pi) node[7];
cx node[9],node[8];
cx node[15],node[12];
cx node[14],node[13];
sx node[2];
cx node[10],node[7];
rz(0.001953125*pi) node[12];
rz(0.0078125*pi) node[13];
rz(3.5*pi) node[2];
cx node[7],node[10];
cx node[15],node[12];
cx node[14],node[13];
sx node[2];
cx node[10],node[7];
rz(3.998046875*pi) node[12];
rz(3.9921875*pi) node[13];
rz(1.0*pi) node[2];
cx node[7],node[4];
cx node[12],node[15];
rz(0.0625*pi) node[4];
cx node[15],node[12];
cx node[7],node[4];
cx node[12],node[15];
rz(3.9375*pi) node[4];
cx node[12],node[13];
cx node[18],node[15];
cx node[7],node[4];
rz(0.00390625*pi) node[13];
rz(0.0009765625*pi) node[15];
cx node[4],node[7];
cx node[12],node[13];
cx node[18],node[15];
cx node[7],node[4];
rz(3.99609375*pi) node[13];
rz(3.9990234375*pi) node[15];
cx node[4],node[1];
cx node[6],node[7];
cx node[15],node[12];
rz(0.125*pi) node[1];
cx node[7],node[6];
cx node[12],node[15];
cx node[4],node[1];
cx node[6],node[7];
cx node[15],node[12];
rz(3.875*pi) node[1];
cx node[7],node[10];
cx node[12],node[13];
cx node[4],node[1];
cx node[10],node[7];
cx node[13],node[12];
cx node[1],node[4];
cx node[7],node[10];
cx node[12],node[13];
cx node[4],node[1];
cx node[12],node[15];
cx node[13],node[14];
cx node[1],node[2];
cx node[15],node[12];
cx node[14],node[13];
rz(0.25*pi) node[2];
cx node[12],node[15];
cx node[13],node[14];
cx node[1],node[2];
cx node[11],node[14];
cx node[18],node[15];
rz(0.5*pi) node[1];
rz(3.75*pi) node[2];
rz(0.00048828125*pi) node[14];
rz(0.001953125*pi) node[15];
sx node[1];
cx node[11],node[14];
cx node[18],node[15];
rz(3.5*pi) node[1];
rz(3.99951171875*pi) node[14];
rz(3.998046875*pi) node[15];
sx node[1];
cx node[15],node[12];
rz(1.0*pi) node[1];
cx node[12],node[15];
cx node[0],node[1];
cx node[15],node[12];
cx node[1],node[0];
cx node[13],node[12];
cx node[0],node[1];
cx node[12],node[13];
cx node[1],node[2];
cx node[13],node[12];
cx node[2],node[1];
cx node[12],node[10];
cx node[13],node[14];
cx node[1],node[2];
cx node[10],node[12];
cx node[14],node[13];
cx node[12],node[10];
cx node[13],node[14];
cx node[10],node[7];
cx node[11],node[14];
cx node[12],node[13];
rz(0.015625*pi) node[7];
rz(0.000244140625*pi) node[13];
rz(0.0009765625*pi) node[14];
cx node[10],node[7];
cx node[11],node[14];
cx node[12],node[13];
rz(3.984375*pi) node[7];
rz(3.999755859375*pi) node[13];
rz(3.9990234375*pi) node[14];
cx node[10],node[7];
cx node[13],node[14];
cx node[7],node[10];
cx node[14],node[13];
cx node[10],node[7];
cx node[13],node[14];
cx node[7],node[6];
cx node[14],node[11];
cx node[12],node[13];
rz(0.03125*pi) node[6];
cx node[11],node[14];
rz(0.00048828125*pi) node[13];
cx node[7],node[6];
cx node[14],node[11];
cx node[12],node[13];
cx node[7],node[4];
rz(3.96875*pi) node[6];
cx node[8],node[11];
cx node[15],node[12];
rz(3.99951171875*pi) node[13];
rz(0.0625*pi) node[4];
rz(0.0001220703125*pi) node[11];
cx node[12],node[15];
cx node[13],node[14];
cx node[7],node[4];
cx node[8],node[11];
cx node[15],node[12];
cx node[14],node[13];
rz(3.9375*pi) node[4];
cx node[12],node[10];
rz(3.9998779296875*pi) node[11];
cx node[13],node[14];
cx node[18],node[15];
cx node[7],node[4];
cx node[8],node[11];
rz(0.0078125*pi) node[10];
cx node[15],node[18];
cx node[4],node[7];
cx node[11],node[8];
cx node[12],node[10];
cx node[18],node[15];
cx node[7],node[4];
cx node[8],node[11];
rz(3.9921875*pi) node[10];
cx node[4],node[1];
cx node[6],node[7];
cx node[9],node[8];
cx node[12],node[10];
cx node[11],node[14];
rz(0.125*pi) node[1];
cx node[7],node[6];
rz(6.103515625e-05*pi) node[8];
cx node[10],node[12];
rz(0.000244140625*pi) node[14];
cx node[4],node[1];
cx node[6],node[7];
cx node[9],node[8];
cx node[12],node[10];
cx node[11],node[14];
rz(3.875*pi) node[1];
cx node[10],node[7];
rz(3.99993896484375*pi) node[8];
cx node[15],node[12];
rz(3.999755859375*pi) node[14];
cx node[4],node[1];
cx node[5],node[8];
rz(0.015625*pi) node[7];
cx node[14],node[11];
rz(0.00390625*pi) node[12];
cx node[1],node[4];
cx node[10],node[7];
rz(3.0517578125e-05*pi) node[8];
cx node[11],node[14];
cx node[15],node[12];
cx node[4],node[1];
cx node[5],node[8];
rz(3.984375*pi) node[7];
cx node[14],node[11];
rz(3.99609375*pi) node[12];
cx node[1],node[0];
cx node[10],node[7];
rz(3.999969482421875*pi) node[8];
cx node[13],node[12];
rz(0.25*pi) node[0];
cx node[8],node[5];
cx node[7],node[10];
rz(0.001953125*pi) node[12];
cx node[1],node[0];
cx node[5],node[8];
cx node[10],node[7];
cx node[13],node[12];
rz(3.75*pi) node[0];
rz(0.5*pi) node[1];
cx node[8],node[5];
cx node[7],node[6];
rz(3.998046875*pi) node[12];
cx node[14],node[13];
sx node[1];
cx node[5],node[3];
rz(0.03125*pi) node[6];
cx node[9],node[8];
cx node[15],node[12];
cx node[13],node[14];
rz(3.5*pi) node[1];
cx node[3],node[5];
cx node[7],node[6];
cx node[8],node[9];
cx node[12],node[15];
cx node[14],node[13];
sx node[1];
cx node[5],node[3];
cx node[7],node[4];
rz(3.96875*pi) node[6];
cx node[9],node[8];
cx node[15],node[12];
rz(1.0*pi) node[1];
cx node[2],node[3];
rz(0.0625*pi) node[4];
cx node[8],node[11];
cx node[12],node[10];
cx node[18],node[15];
cx node[0],node[1];
rz(1.52587890625e-05*pi) node[3];
cx node[7],node[4];
rz(0.0078125*pi) node[10];
rz(0.0001220703125*pi) node[11];
rz(0.0009765625*pi) node[15];
cx node[1],node[0];
cx node[2],node[3];
rz(3.9375*pi) node[4];
cx node[8],node[11];
cx node[12],node[10];
cx node[18],node[15];
cx node[0],node[1];
rz(3.9999847412109375*pi) node[3];
cx node[7],node[4];
rz(3.9921875*pi) node[10];
rz(3.9998779296875*pi) node[11];
rz(3.9990234375*pi) node[15];
cx node[2],node[3];
cx node[4],node[7];
cx node[8],node[11];
cx node[12],node[10];
cx node[3],node[2];
cx node[7],node[4];
cx node[11],node[8];
cx node[10],node[12];
cx node[4],node[1];
cx node[2],node[3];
cx node[6],node[7];
cx node[8],node[11];
cx node[12],node[10];
rz(0.125*pi) node[1];
cx node[3],node[5];
cx node[7],node[6];
cx node[9],node[8];
cx node[12],node[13];
cx node[4],node[1];
cx node[5],node[3];
cx node[6],node[7];
rz(6.103515625e-05*pi) node[8];
cx node[13],node[12];
rz(3.875*pi) node[1];
cx node[3],node[5];
cx node[10],node[7];
cx node[9],node[8];
cx node[12],node[13];
cx node[4],node[1];
rz(0.015625*pi) node[7];
rz(3.99993896484375*pi) node[8];
cx node[12],node[15];
cx node[14],node[13];
cx node[1],node[4];
cx node[5],node[8];
cx node[10],node[7];
rz(0.00390625*pi) node[13];
rz(0.00048828125*pi) node[15];
cx node[4],node[1];
rz(3.984375*pi) node[7];
rz(3.0517578125e-05*pi) node[8];
cx node[12],node[15];
cx node[14],node[13];
cx node[1],node[0];
cx node[5],node[8];
cx node[10],node[7];
rz(3.99609375*pi) node[13];
rz(3.99951171875*pi) node[15];
rz(0.25*pi) node[0];
cx node[7],node[10];
rz(3.999969482421875*pi) node[8];
cx node[13],node[12];
cx node[1],node[0];
cx node[10],node[7];
cx node[9],node[8];
cx node[12],node[13];
rz(3.75*pi) node[0];
rz(0.5*pi) node[1];
cx node[7],node[6];
cx node[8],node[9];
cx node[13],node[12];
sx node[1];
rz(0.03125*pi) node[6];
cx node[9],node[8];
cx node[12],node[15];
rz(3.5*pi) node[1];
cx node[7],node[6];
cx node[15],node[12];
sx node[1];
rz(3.96875*pi) node[6];
cx node[12],node[15];
rz(1.0*pi) node[1];
cx node[6],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[0],node[1];
cx node[7],node[6];
cx node[12],node[13];
rz(0.001953125*pi) node[15];
cx node[1],node[0];
cx node[6],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[0],node[1];
cx node[14],node[13];
rz(3.998046875*pi) node[15];
cx node[12],node[15];
cx node[13],node[14];
cx node[14],node[13];
rz(0.0009765625*pi) node[15];
cx node[11],node[14];
cx node[12],node[15];
cx node[13],node[12];
rz(0.000244140625*pi) node[14];
rz(3.9990234375*pi) node[15];
cx node[11],node[14];
cx node[12],node[13];
cx node[18],node[15];
cx node[13],node[12];
rz(3.999755859375*pi) node[14];
cx node[15],node[18];
cx node[12],node[10];
cx node[14],node[11];
cx node[18],node[15];
rz(0.0078125*pi) node[10];
cx node[11],node[14];
cx node[12],node[10];
cx node[14],node[11];
cx node[8],node[11];
rz(3.9921875*pi) node[10];
cx node[12],node[10];
rz(0.0001220703125*pi) node[11];
cx node[8],node[11];
cx node[10],node[12];
cx node[12],node[10];
rz(3.9998779296875*pi) node[11];
cx node[10],node[7];
cx node[11],node[8];
cx node[15],node[12];
rz(0.015625*pi) node[7];
cx node[8],node[11];
rz(0.00390625*pi) node[12];
cx node[10],node[7];
cx node[11],node[8];
cx node[15],node[12];
cx node[5],node[8];
rz(3.984375*pi) node[7];
rz(3.99609375*pi) node[12];
cx node[18],node[15];
cx node[7],node[10];
rz(6.103515625e-05*pi) node[8];
cx node[13],node[12];
cx node[15],node[18];
cx node[5],node[8];
cx node[10],node[7];
rz(0.001953125*pi) node[12];
cx node[18],node[15];
cx node[7],node[10];
rz(3.99993896484375*pi) node[8];
cx node[13],node[12];
cx node[5],node[8];
cx node[6],node[7];
rz(3.998046875*pi) node[12];
cx node[8],node[5];
cx node[7],node[6];
cx node[15],node[12];
cx node[5],node[8];
cx node[6],node[7];
cx node[12],node[15];
cx node[7],node[4];
cx node[15],node[12];
rz(0.0625*pi) node[4];
cx node[12],node[13];
cx node[7],node[4];
cx node[13],node[12];
rz(3.9375*pi) node[4];
cx node[12],node[13];
cx node[7],node[4];
cx node[10],node[12];
cx node[14],node[13];
cx node[4],node[7];
cx node[12],node[10];
rz(0.00048828125*pi) node[13];
cx node[7],node[4];
cx node[10],node[12];
cx node[14],node[13];
cx node[4],node[1];
cx node[6],node[7];
cx node[12],node[15];
rz(3.99951171875*pi) node[13];
rz(0.125*pi) node[1];
rz(0.03125*pi) node[7];
cx node[15],node[12];
cx node[14],node[13];
cx node[4],node[1];
cx node[6],node[7];
cx node[12],node[15];
cx node[13],node[14];
rz(3.875*pi) node[1];
rz(3.96875*pi) node[7];
cx node[14],node[13];
cx node[18],node[15];
cx node[4],node[1];
cx node[7],node[10];
cx node[11],node[14];
cx node[13],node[12];
rz(0.0078125*pi) node[15];
cx node[1],node[4];
cx node[10],node[7];
rz(0.0009765625*pi) node[12];
rz(0.000244140625*pi) node[14];
cx node[18],node[15];
cx node[4],node[1];
cx node[7],node[10];
cx node[11],node[14];
cx node[13],node[12];
rz(3.9921875*pi) node[15];
cx node[1],node[0];
rz(3.9990234375*pi) node[12];
rz(3.999755859375*pi) node[14];
rz(0.25*pi) node[0];
cx node[14],node[11];
cx node[15],node[12];
cx node[1],node[0];
cx node[11],node[14];
cx node[12],node[15];
rz(3.75*pi) node[0];
rz(0.5*pi) node[1];
cx node[14],node[11];
cx node[15],node[12];
sx node[1];
cx node[8],node[11];
cx node[12],node[10];
cx node[18],node[15];
rz(3.5*pi) node[1];
cx node[10],node[12];
rz(0.0001220703125*pi) node[11];
cx node[15],node[18];
sx node[1];
cx node[8],node[11];
cx node[12],node[10];
cx node[18],node[15];
rz(1.0*pi) node[1];
cx node[7],node[10];
rz(3.9998779296875*pi) node[11];
cx node[15],node[12];
cx node[0],node[1];
cx node[8],node[11];
rz(0.00390625*pi) node[10];
rz(0.015625*pi) node[12];
cx node[1],node[0];
cx node[7],node[10];
cx node[11],node[8];
cx node[15],node[12];
cx node[0],node[1];
cx node[8],node[11];
rz(3.99609375*pi) node[10];
rz(3.984375*pi) node[12];
cx node[12],node[10];
cx node[10],node[12];
cx node[12],node[10];
cx node[7],node[10];
cx node[13],node[12];
rz(0.0078125*pi) node[10];
rz(0.001953125*pi) node[12];
cx node[7],node[10];
cx node[13],node[12];
cx node[6],node[7];
rz(3.9921875*pi) node[10];
rz(3.998046875*pi) node[12];
cx node[7],node[6];
cx node[15],node[12];
cx node[6],node[7];
cx node[12],node[15];
cx node[7],node[4];
cx node[15],node[12];
rz(0.0625*pi) node[4];
cx node[12],node[10];
cx node[18],node[15];
cx node[7],node[4];
cx node[10],node[12];
cx node[15],node[18];
rz(3.9375*pi) node[4];
cx node[12],node[10];
cx node[18],node[15];
cx node[7],node[4];
cx node[13],node[12];
cx node[4],node[7];
rz(0.00390625*pi) node[12];
cx node[7],node[4];
cx node[13],node[12];
cx node[4],node[1];
cx node[10],node[7];
rz(3.99609375*pi) node[12];
rz(0.125*pi) node[1];
rz(0.03125*pi) node[7];
cx node[15],node[12];
cx node[4],node[1];
cx node[10],node[7];
cx node[12],node[15];
rz(3.875*pi) node[1];
rz(3.96875*pi) node[7];
cx node[15],node[12];
cx node[4],node[1];
cx node[6],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[1],node[4];
rz(0.015625*pi) node[7];
cx node[12],node[13];
cx node[15],node[18];
cx node[4],node[1];
cx node[6],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[1],node[0];
rz(3.984375*pi) node[7];
cx node[14],node[13];
rz(0.25*pi) node[0];
cx node[7],node[10];
rz(0.00048828125*pi) node[13];
cx node[1],node[0];
cx node[10],node[7];
cx node[14],node[13];
rz(3.75*pi) node[0];
rz(0.5*pi) node[1];
cx node[7],node[10];
rz(3.99951171875*pi) node[13];
sx node[1];
cx node[7],node[4];
cx node[12],node[10];
cx node[14],node[13];
rz(3.5*pi) node[1];
rz(0.0625*pi) node[4];
rz(0.0078125*pi) node[10];
cx node[13],node[14];
sx node[1];
cx node[7],node[4];
cx node[12],node[10];
cx node[14],node[13];
rz(1.0*pi) node[1];
rz(3.9375*pi) node[4];
rz(3.9921875*pi) node[10];
cx node[11],node[14];
cx node[13],node[12];
cx node[0],node[1];
cx node[7],node[4];
cx node[12],node[13];
rz(0.000244140625*pi) node[14];
cx node[1],node[0];
cx node[4],node[7];
cx node[11],node[14];
cx node[13],node[12];
cx node[0],node[1];
cx node[7],node[4];
cx node[12],node[15];
rz(3.999755859375*pi) node[14];
cx node[4],node[1];
cx node[6],node[7];
cx node[11],node[14];
rz(0.0009765625*pi) node[15];
rz(0.125*pi) node[1];
rz(0.03125*pi) node[7];
cx node[14],node[11];
cx node[12],node[15];
cx node[4],node[1];
cx node[6],node[7];
cx node[11],node[14];
rz(3.9990234375*pi) node[15];
rz(3.875*pi) node[1];
rz(3.96875*pi) node[7];
cx node[15],node[12];
cx node[14],node[13];
cx node[4],node[1];
cx node[12],node[15];
cx node[13],node[14];
cx node[1],node[4];
cx node[15],node[12];
cx node[14],node[13];
cx node[4],node[1];
cx node[13],node[12];
cx node[15],node[18];
cx node[1],node[0];
rz(0.00048828125*pi) node[12];
rz(0.001953125*pi) node[18];
rz(0.25*pi) node[0];
cx node[13],node[12];
cx node[15],node[18];
cx node[1],node[0];
rz(3.99951171875*pi) node[12];
rz(3.998046875*pi) node[18];
rz(3.75*pi) node[0];
rz(0.5*pi) node[1];
cx node[13],node[12];
sx node[1];
cx node[12],node[13];
rz(3.5*pi) node[1];
cx node[13],node[12];
sx node[1];
cx node[15],node[12];
cx node[14],node[13];
rz(1.0*pi) node[1];
cx node[12],node[15];
cx node[13],node[14];
cx node[0],node[1];
cx node[15],node[12];
cx node[14],node[13];
cx node[1],node[0];
cx node[12],node[10];
cx node[15],node[18];
cx node[0],node[1];
rz(0.00390625*pi) node[10];
rz(0.0009765625*pi) node[18];
cx node[12],node[10];
cx node[15],node[18];
rz(3.99609375*pi) node[10];
cx node[13],node[12];
rz(3.9990234375*pi) node[18];
cx node[12],node[13];
cx node[13],node[12];
cx node[12],node[10];
cx node[10],node[12];
cx node[12],node[10];
cx node[10],node[7];
cx node[15],node[12];
rz(0.015625*pi) node[7];
rz(0.001953125*pi) node[12];
cx node[10],node[7];
cx node[15],node[12];
rz(3.984375*pi) node[7];
rz(3.998046875*pi) node[12];
cx node[7],node[10];
cx node[13],node[12];
cx node[10],node[7];
cx node[12],node[13];
cx node[7],node[10];
cx node[13],node[12];
cx node[6],node[7];
cx node[12],node[10];
cx node[7],node[6];
rz(0.0078125*pi) node[10];
cx node[6],node[7];
cx node[12],node[10];
cx node[7],node[4];
rz(3.9921875*pi) node[10];
rz(0.0625*pi) node[4];
cx node[12],node[10];
cx node[7],node[4];
cx node[10],node[12];
rz(3.9375*pi) node[4];
cx node[12],node[10];
cx node[7],node[4];
cx node[15],node[12];
cx node[4],node[7];
rz(0.00390625*pi) node[12];
cx node[7],node[4];
cx node[15],node[12];
cx node[4],node[1];
cx node[6],node[7];
rz(3.99609375*pi) node[12];
rz(0.125*pi) node[1];
rz(0.03125*pi) node[7];
cx node[15],node[12];
cx node[4],node[1];
cx node[6],node[7];
cx node[12],node[15];
rz(3.875*pi) node[1];
rz(3.96875*pi) node[7];
cx node[15],node[12];
cx node[4],node[1];
cx node[10],node[7];
cx node[1],node[4];
rz(0.015625*pi) node[7];
cx node[4],node[1];
cx node[10],node[7];
cx node[1],node[0];
rz(3.984375*pi) node[7];
rz(0.25*pi) node[0];
cx node[7],node[10];
cx node[1],node[0];
cx node[10],node[7];
rz(3.75*pi) node[0];
rz(0.5*pi) node[1];
cx node[7],node[10];
sx node[1];
cx node[6],node[7];
cx node[12],node[10];
rz(3.5*pi) node[1];
cx node[7],node[6];
rz(0.0078125*pi) node[10];
sx node[1];
cx node[6],node[7];
cx node[12],node[10];
rz(1.0*pi) node[1];
cx node[7],node[4];
rz(3.9921875*pi) node[10];
cx node[0],node[1];
rz(0.0625*pi) node[4];
cx node[12],node[10];
cx node[1],node[0];
cx node[7],node[4];
cx node[10],node[12];
cx node[0],node[1];
rz(3.9375*pi) node[4];
cx node[12],node[10];
cx node[7],node[4];
cx node[4],node[7];
cx node[7],node[4];
cx node[4],node[1];
cx node[6],node[7];
rz(0.125*pi) node[1];
rz(0.03125*pi) node[7];
cx node[4],node[1];
cx node[6],node[7];
rz(3.875*pi) node[1];
rz(3.96875*pi) node[7];
cx node[4],node[1];
cx node[10],node[7];
cx node[1],node[4];
rz(0.015625*pi) node[7];
cx node[4],node[1];
cx node[10],node[7];
cx node[1],node[0];
rz(3.984375*pi) node[7];
rz(0.25*pi) node[0];
cx node[6],node[7];
cx node[1],node[0];
cx node[7],node[6];
rz(3.75*pi) node[0];
rz(0.5*pi) node[1];
cx node[6],node[7];
sx node[1];
cx node[7],node[4];
rz(3.5*pi) node[1];
rz(0.0625*pi) node[4];
sx node[1];
cx node[7],node[4];
rz(1.0*pi) node[1];
rz(3.9375*pi) node[4];
cx node[0],node[1];
cx node[7],node[4];
cx node[1],node[0];
cx node[4],node[7];
cx node[0],node[1];
cx node[7],node[4];
cx node[4],node[1];
cx node[10],node[7];
rz(0.125*pi) node[1];
rz(0.03125*pi) node[7];
cx node[4],node[1];
cx node[10],node[7];
rz(3.875*pi) node[1];
rz(3.96875*pi) node[7];
cx node[4],node[1];
cx node[10],node[7];
cx node[1],node[4];
cx node[7],node[10];
cx node[4],node[1];
cx node[10],node[7];
cx node[1],node[0];
cx node[7],node[4];
rz(0.25*pi) node[0];
rz(0.0625*pi) node[4];
cx node[1],node[0];
cx node[7],node[4];
rz(3.75*pi) node[0];
rz(0.5*pi) node[1];
rz(3.9375*pi) node[4];
sx node[1];
cx node[7],node[4];
rz(3.5*pi) node[1];
cx node[4],node[7];
sx node[1];
cx node[7],node[4];
rz(1.0*pi) node[1];
cx node[4],node[1];
cx node[1],node[4];
cx node[4],node[1];
cx node[1],node[0];
rz(0.125*pi) node[0];
cx node[1],node[0];
rz(3.875*pi) node[0];
cx node[1],node[4];
rz(0.25*pi) node[4];
cx node[1],node[4];
rz(0.5*pi) node[1];
rz(3.75*pi) node[4];
sx node[1];
rz(3.5*pi) node[1];
sx node[1];
rz(1.0*pi) node[1];
barrier node[2],node[9],node[5],node[8],node[11],node[14],node[18],node[13],node[15],node[12],node[6],node[10],node[7],node[0],node[4],node[1],node[3];
measure node[2] -> c[0];
measure node[9] -> c[1];
measure node[5] -> c[2];
measure node[8] -> c[3];
measure node[11] -> c[4];
measure node[14] -> c[5];
measure node[18] -> c[6];
measure node[13] -> c[7];
measure node[15] -> c[8];
measure node[12] -> c[9];
measure node[6] -> c[10];
measure node[10] -> c[11];
measure node[7] -> c[12];
measure node[0] -> c[13];
measure node[4] -> c[14];
measure node[1] -> c[15];
