OPENQASM 2.0;
include "qelib1.inc";

qreg node[26];
creg c[18];
creg meas[18];
sx node[24];
rz(0.5*pi) node[24];
sx node[24];
rz(0.4999961853027348*pi) node[24];
cx node[24],node[25];
rz(3.75*pi) node[25];
cx node[24],node[25];
cx node[24],node[23];
rz(0.25*pi) node[25];
rz(3.875*pi) node[23];
sx node[25];
cx node[24],node[23];
rz(0.5*pi) node[25];
rz(0.125*pi) node[23];
sx node[25];
cx node[24],node[23];
rz(0.4999923706054692*pi) node[25];
cx node[23],node[24];
cx node[24],node[23];
cx node[23],node[21];
cx node[25],node[24];
rz(3.9375*pi) node[21];
rz(3.75*pi) node[24];
cx node[23],node[21];
cx node[25],node[24];
rz(0.0625*pi) node[21];
rz(0.25*pi) node[24];
cx node[23],node[21];
sx node[24];
cx node[21],node[23];
rz(0.5*pi) node[24];
cx node[23],node[21];
sx node[24];
cx node[21],node[18];
rz(0.49998474121093794*pi) node[24];
rz(3.96875*pi) node[18];
cx node[25],node[24];
cx node[21],node[18];
cx node[24],node[25];
rz(0.03125*pi) node[18];
cx node[25],node[24];
cx node[21],node[18];
cx node[24],node[23];
cx node[18],node[21];
rz(3.875*pi) node[23];
cx node[21],node[18];
cx node[24],node[23];
cx node[18],node[15];
rz(0.125*pi) node[23];
rz(3.984375*pi) node[15];
cx node[24],node[23];
cx node[18],node[15];
cx node[23],node[24];
rz(0.015625*pi) node[15];
cx node[18],node[17];
cx node[24],node[23];
rz(3.9921875*pi) node[17];
cx node[23],node[21];
cx node[25],node[24];
cx node[18],node[17];
rz(3.9375*pi) node[21];
rz(3.75*pi) node[24];
cx node[18],node[15];
rz(0.0078125*pi) node[17];
cx node[23],node[21];
cx node[25],node[24];
cx node[15],node[18];
rz(0.0625*pi) node[21];
rz(0.25*pi) node[24];
cx node[18],node[15];
cx node[23],node[21];
sx node[24];
cx node[15],node[12];
cx node[21],node[23];
rz(0.5*pi) node[24];
rz(3.99609375*pi) node[12];
cx node[23],node[21];
sx node[24];
cx node[15],node[12];
cx node[21],node[18];
rz(0.49996948242187544*pi) node[24];
rz(0.00390625*pi) node[12];
rz(3.96875*pi) node[18];
cx node[25],node[24];
cx node[15],node[12];
cx node[21],node[18];
cx node[24],node[25];
cx node[12],node[15];
rz(0.03125*pi) node[18];
cx node[25],node[24];
cx node[15],node[12];
cx node[21],node[18];
cx node[24],node[23];
cx node[12],node[10];
cx node[18],node[21];
rz(3.875*pi) node[23];
rz(3.998046875*pi) node[10];
cx node[21],node[18];
cx node[24],node[23];
cx node[12],node[10];
cx node[18],node[17];
rz(0.125*pi) node[23];
rz(0.001953125*pi) node[10];
cx node[12],node[13];
rz(3.984375*pi) node[17];
cx node[24],node[23];
rz(3.9990234375*pi) node[13];
cx node[18],node[17];
cx node[23],node[24];
cx node[12],node[13];
cx node[18],node[15];
rz(0.015625*pi) node[17];
cx node[24],node[23];
cx node[12],node[10];
rz(0.0009765625*pi) node[13];
rz(3.9921875*pi) node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[10],node[12];
cx node[18],node[15];
rz(3.9375*pi) node[21];
rz(3.75*pi) node[24];
cx node[12],node[10];
rz(0.0078125*pi) node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[10],node[7];
cx node[18],node[15];
rz(0.0625*pi) node[21];
rz(0.25*pi) node[24];
rz(3.99951171875*pi) node[7];
cx node[15],node[18];
cx node[23],node[21];
sx node[24];
cx node[10],node[7];
cx node[18],node[15];
cx node[21],node[23];
rz(0.5*pi) node[24];
rz(0.00048828125*pi) node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
cx node[10],node[7];
rz(3.99609375*pi) node[12];
cx node[18],node[17];
rz(0.49993896484375044*pi) node[24];
cx node[7],node[10];
cx node[15],node[12];
cx node[17],node[18];
cx node[25],node[24];
cx node[10],node[7];
rz(0.00390625*pi) node[12];
cx node[21],node[18];
cx node[24],node[25];
cx node[7],node[4];
cx node[15],node[12];
rz(3.96875*pi) node[18];
cx node[25],node[24];
rz(3.999755859375*pi) node[4];
cx node[12],node[15];
cx node[21],node[18];
cx node[24],node[23];
cx node[7],node[4];
cx node[15],node[12];
rz(0.03125*pi) node[18];
rz(3.875*pi) node[23];
rz(0.000244140625*pi) node[4];
cx node[7],node[6];
cx node[12],node[13];
cx node[21],node[18];
cx node[24],node[23];
rz(3.9998779296875*pi) node[6];
rz(3.998046875*pi) node[13];
cx node[18],node[21];
rz(0.125*pi) node[23];
cx node[7],node[6];
cx node[12],node[13];
cx node[21],node[18];
cx node[23],node[24];
cx node[7],node[4];
rz(0.0001220703125*pi) node[6];
cx node[12],node[10];
rz(0.001953125*pi) node[13];
cx node[18],node[17];
cx node[24],node[23];
cx node[4],node[7];
rz(3.9990234375*pi) node[10];
rz(3.984375*pi) node[17];
cx node[23],node[24];
cx node[7],node[4];
cx node[12],node[10];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
cx node[4],node[1];
rz(0.0009765625*pi) node[10];
cx node[18],node[15];
rz(0.015625*pi) node[17];
rz(3.9375*pi) node[21];
rz(3.75*pi) node[24];
rz(3.99993896484375*pi) node[1];
cx node[12],node[10];
rz(3.9921875*pi) node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[4],node[1];
cx node[10],node[12];
cx node[18],node[15];
rz(0.0625*pi) node[21];
rz(0.25*pi) node[24];
rz(6.103515625e-05*pi) node[1];
cx node[12],node[10];
rz(0.0078125*pi) node[15];
cx node[23],node[21];
sx node[24];
cx node[4],node[1];
cx node[10],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[21],node[23];
rz(0.5*pi) node[24];
cx node[1],node[4];
rz(3.99951171875*pi) node[7];
cx node[12],node[13];
cx node[15],node[18];
cx node[23],node[21];
sx node[24];
cx node[4],node[1];
cx node[10],node[7];
cx node[13],node[12];
cx node[18],node[15];
rz(0.49987792968750044*pi) node[24];
cx node[1],node[0];
rz(0.00048828125*pi) node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[25],node[24];
rz(3.999969482421875*pi) node[0];
cx node[10],node[7];
rz(3.99609375*pi) node[12];
cx node[18],node[17];
cx node[24],node[25];
cx node[1],node[0];
cx node[7],node[10];
cx node[15],node[12];
cx node[17],node[18];
cx node[25],node[24];
rz(3.0517578125e-05*pi) node[0];
cx node[1],node[2];
cx node[10],node[7];
rz(0.00390625*pi) node[12];
cx node[21],node[18];
cx node[24],node[23];
rz(3.9999847412109375*pi) node[2];
cx node[7],node[6];
cx node[12],node[15];
rz(3.96875*pi) node[18];
rz(3.875*pi) node[23];
cx node[1],node[2];
rz(3.999755859375*pi) node[6];
cx node[15],node[12];
cx node[21],node[18];
cx node[24],node[23];
rz(1.52587890625e-05*pi) node[2];
cx node[7],node[6];
cx node[12],node[15];
rz(0.03125*pi) node[18];
rz(0.125*pi) node[23];
cx node[1],node[2];
cx node[7],node[4];
rz(0.000244140625*pi) node[6];
cx node[12],node[13];
cx node[21],node[18];
cx node[24],node[23];
cx node[2],node[1];
rz(3.9998779296875*pi) node[4];
rz(3.998046875*pi) node[13];
cx node[18],node[21];
cx node[23],node[24];
cx node[1],node[2];
cx node[7],node[4];
cx node[12],node[13];
cx node[21],node[18];
cx node[24],node[23];
cx node[0],node[1];
cx node[2],node[3];
rz(0.0001220703125*pi) node[4];
cx node[12],node[10];
rz(0.001953125*pi) node[13];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
cx node[1],node[0];
rz(3.9999923706054688*pi) node[3];
cx node[7],node[4];
rz(3.9990234375*pi) node[10];
rz(3.984375*pi) node[17];
rz(3.9375*pi) node[21];
rz(3.75*pi) node[24];
cx node[0],node[1];
cx node[2],node[3];
cx node[4],node[7];
cx node[12],node[10];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
rz(7.62939453125e-06*pi) node[3];
cx node[7],node[4];
rz(0.0009765625*pi) node[10];
cx node[18],node[15];
rz(0.015625*pi) node[17];
rz(0.0625*pi) node[21];
rz(0.25*pi) node[24];
cx node[4],node[1];
cx node[3],node[2];
cx node[6],node[7];
cx node[12],node[10];
rz(3.9921875*pi) node[15];
cx node[23],node[21];
sx node[24];
rz(3.99993896484375*pi) node[1];
cx node[2],node[3];
cx node[7],node[6];
cx node[10],node[12];
cx node[18],node[15];
cx node[21],node[23];
rz(0.5*pi) node[24];
cx node[4],node[1];
cx node[3],node[2];
cx node[6],node[7];
cx node[12],node[10];
rz(0.0078125*pi) node[15];
cx node[23],node[21];
sx node[24];
rz(6.103515625e-05*pi) node[1];
cx node[3],node[5];
cx node[10],node[7];
cx node[13],node[12];
cx node[18],node[15];
rz(0.49975585937500044*pi) node[24];
cx node[4],node[1];
rz(3.9999961853027344*pi) node[5];
rz(3.99951171875*pi) node[7];
cx node[12],node[13];
cx node[15],node[18];
cx node[25],node[24];
cx node[1],node[4];
cx node[3],node[5];
cx node[10],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[24],node[25];
cx node[4],node[1];
rz(3.814697265625e-06*pi) node[5];
rz(0.00048828125*pi) node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[25],node[24];
cx node[1],node[0];
cx node[5],node[3];
cx node[10],node[7];
rz(3.99609375*pi) node[12];
cx node[18],node[17];
cx node[24],node[23];
rz(3.999969482421875*pi) node[0];
cx node[3],node[5];
cx node[7],node[10];
cx node[15],node[12];
cx node[17],node[18];
rz(3.875*pi) node[23];
cx node[1],node[0];
cx node[5],node[3];
cx node[10],node[7];
rz(0.00390625*pi) node[12];
cx node[21],node[18];
cx node[24],node[23];
rz(3.0517578125e-05*pi) node[0];
cx node[1],node[2];
cx node[7],node[6];
cx node[12],node[15];
rz(3.96875*pi) node[18];
rz(0.125*pi) node[23];
rz(3.9999847412109375*pi) node[2];
rz(3.999755859375*pi) node[6];
cx node[15],node[12];
cx node[21],node[18];
cx node[23],node[24];
cx node[1],node[2];
cx node[7],node[6];
cx node[12],node[15];
rz(0.03125*pi) node[18];
cx node[24],node[23];
rz(1.52587890625e-05*pi) node[2];
cx node[7],node[4];
rz(0.000244140625*pi) node[6];
cx node[12],node[13];
cx node[21],node[18];
cx node[23],node[24];
cx node[1],node[2];
rz(3.9998779296875*pi) node[4];
rz(3.998046875*pi) node[13];
cx node[18],node[21];
cx node[25],node[24];
cx node[2],node[1];
cx node[7],node[4];
cx node[12],node[13];
cx node[21],node[18];
rz(3.75*pi) node[24];
cx node[1],node[2];
rz(0.0001220703125*pi) node[4];
cx node[12],node[10];
rz(0.001953125*pi) node[13];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
cx node[0],node[1];
cx node[2],node[3];
cx node[7],node[4];
rz(3.9990234375*pi) node[10];
rz(3.984375*pi) node[17];
rz(3.9375*pi) node[21];
rz(0.25*pi) node[24];
cx node[1],node[0];
rz(3.9999923706054688*pi) node[3];
cx node[4],node[7];
cx node[12],node[10];
cx node[18],node[17];
cx node[23],node[21];
sx node[24];
cx node[0],node[1];
cx node[2],node[3];
cx node[7],node[4];
rz(0.0009765625*pi) node[10];
cx node[18],node[15];
rz(0.015625*pi) node[17];
rz(0.0625*pi) node[21];
rz(0.5*pi) node[24];
cx node[4],node[1];
rz(7.62939453125e-06*pi) node[3];
cx node[6],node[7];
cx node[12],node[10];
rz(3.9921875*pi) node[15];
cx node[23],node[21];
sx node[24];
rz(3.99993896484375*pi) node[1];
cx node[3],node[2];
cx node[7],node[6];
cx node[10],node[12];
cx node[18],node[15];
cx node[21],node[23];
rz(0.49951171875000044*pi) node[24];
cx node[4],node[1];
cx node[2],node[3];
cx node[6],node[7];
cx node[12],node[10];
rz(0.0078125*pi) node[15];
cx node[23],node[21];
cx node[25],node[24];
rz(6.103515625e-05*pi) node[1];
cx node[3],node[2];
cx node[10],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[24],node[25];
cx node[4],node[1];
rz(3.99951171875*pi) node[7];
cx node[12],node[13];
cx node[15],node[18];
cx node[25],node[24];
cx node[1],node[4];
cx node[10],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[24],node[23];
cx node[4],node[1];
rz(0.00048828125*pi) node[7];
cx node[15],node[12];
cx node[17],node[18];
rz(3.875*pi) node[23];
cx node[1],node[0];
cx node[10],node[7];
rz(3.99609375*pi) node[12];
cx node[18],node[17];
cx node[24],node[23];
rz(3.999969482421875*pi) node[0];
cx node[7],node[10];
cx node[15],node[12];
cx node[17],node[18];
rz(0.125*pi) node[23];
cx node[1],node[0];
cx node[10],node[7];
rz(0.00390625*pi) node[12];
cx node[21],node[18];
cx node[23],node[24];
rz(3.0517578125e-05*pi) node[0];
cx node[1],node[2];
cx node[7],node[6];
cx node[12],node[15];
rz(3.96875*pi) node[18];
cx node[24],node[23];
rz(3.9999847412109375*pi) node[2];
rz(3.999755859375*pi) node[6];
cx node[15],node[12];
cx node[21],node[18];
cx node[23],node[24];
cx node[1],node[2];
cx node[7],node[6];
cx node[12],node[15];
rz(0.03125*pi) node[18];
cx node[25],node[24];
cx node[0],node[1];
rz(1.52587890625e-05*pi) node[2];
cx node[7],node[4];
rz(0.000244140625*pi) node[6];
cx node[12],node[13];
cx node[21],node[18];
rz(3.75*pi) node[24];
cx node[1],node[0];
rz(3.9998779296875*pi) node[4];
rz(3.998046875*pi) node[13];
cx node[18],node[21];
cx node[25],node[24];
cx node[0],node[1];
cx node[7],node[4];
cx node[12],node[13];
cx node[21],node[18];
rz(0.25*pi) node[24];
rz(0.0001220703125*pi) node[4];
cx node[12],node[10];
rz(0.001953125*pi) node[13];
cx node[18],node[17];
cx node[23],node[21];
sx node[24];
cx node[7],node[4];
rz(3.9990234375*pi) node[10];
rz(3.984375*pi) node[17];
rz(3.9375*pi) node[21];
rz(0.5*pi) node[24];
cx node[4],node[7];
cx node[12],node[10];
cx node[18],node[17];
cx node[23],node[21];
sx node[24];
cx node[7],node[4];
rz(0.0009765625*pi) node[10];
cx node[18],node[15];
rz(0.015625*pi) node[17];
rz(0.0625*pi) node[21];
rz(0.49902343750000044*pi) node[24];
cx node[4],node[1];
cx node[6],node[7];
cx node[12],node[10];
rz(3.9921875*pi) node[15];
cx node[23],node[21];
cx node[25],node[24];
rz(3.99993896484375*pi) node[1];
cx node[7],node[6];
cx node[10],node[12];
cx node[18],node[15];
cx node[21],node[23];
cx node[24],node[25];
cx node[4],node[1];
cx node[6],node[7];
cx node[12],node[10];
rz(0.0078125*pi) node[15];
cx node[23],node[21];
cx node[25],node[24];
rz(6.103515625e-05*pi) node[1];
cx node[10],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[24],node[23];
cx node[1],node[4];
rz(3.99951171875*pi) node[7];
cx node[12],node[13];
cx node[15],node[18];
rz(3.875*pi) node[23];
cx node[4],node[1];
cx node[10],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[24],node[23];
cx node[1],node[4];
rz(0.00048828125*pi) node[7];
cx node[15],node[12];
cx node[17],node[18];
rz(0.125*pi) node[23];
cx node[1],node[2];
cx node[10],node[7];
rz(3.99609375*pi) node[12];
cx node[18],node[17];
cx node[23],node[24];
rz(3.999969482421875*pi) node[2];
cx node[7],node[10];
cx node[15],node[12];
cx node[17],node[18];
cx node[24],node[23];
cx node[1],node[2];
cx node[10],node[7];
rz(0.00390625*pi) node[12];
cx node[21],node[18];
cx node[23],node[24];
rz(3.0517578125e-05*pi) node[2];
cx node[7],node[6];
cx node[12],node[15];
rz(3.96875*pi) node[18];
cx node[25],node[24];
cx node[2],node[1];
rz(3.999755859375*pi) node[6];
cx node[15],node[12];
cx node[21],node[18];
rz(3.75*pi) node[24];
cx node[1],node[2];
cx node[7],node[6];
cx node[12],node[15];
rz(0.03125*pi) node[18];
cx node[25],node[24];
cx node[2],node[1];
cx node[7],node[4];
rz(0.000244140625*pi) node[6];
cx node[12],node[13];
cx node[21],node[18];
rz(0.25*pi) node[24];
rz(3.9998779296875*pi) node[4];
rz(3.998046875*pi) node[13];
cx node[18],node[21];
sx node[24];
cx node[7],node[4];
cx node[12],node[13];
cx node[21],node[18];
rz(0.5*pi) node[24];
rz(0.0001220703125*pi) node[4];
cx node[12],node[10];
rz(0.001953125*pi) node[13];
cx node[18],node[17];
cx node[23],node[21];
sx node[24];
cx node[7],node[4];
rz(3.9990234375*pi) node[10];
rz(3.984375*pi) node[17];
rz(3.9375*pi) node[21];
rz(0.49804687500000044*pi) node[24];
cx node[4],node[7];
cx node[12],node[10];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
cx node[7],node[4];
rz(0.0009765625*pi) node[10];
cx node[18],node[15];
rz(0.015625*pi) node[17];
rz(0.0625*pi) node[21];
cx node[24],node[25];
cx node[4],node[1];
cx node[6],node[7];
cx node[12],node[10];
rz(3.9921875*pi) node[15];
cx node[23],node[21];
cx node[25],node[24];
rz(3.99993896484375*pi) node[1];
cx node[7],node[6];
cx node[10],node[12];
cx node[18],node[15];
cx node[21],node[23];
cx node[4],node[1];
cx node[6],node[7];
cx node[12],node[10];
rz(0.0078125*pi) node[15];
cx node[23],node[21];
rz(6.103515625e-05*pi) node[1];
cx node[10],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[24],node[23];
cx node[1],node[4];
rz(3.99951171875*pi) node[7];
cx node[12],node[13];
cx node[15],node[18];
rz(3.875*pi) node[23];
cx node[4],node[1];
cx node[10],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[24],node[23];
cx node[1],node[4];
rz(0.00048828125*pi) node[7];
cx node[15],node[12];
cx node[17],node[18];
rz(0.125*pi) node[23];
cx node[10],node[7];
rz(3.99609375*pi) node[12];
cx node[18],node[17];
cx node[23],node[24];
cx node[7],node[10];
cx node[15],node[12];
cx node[17],node[18];
cx node[24],node[23];
cx node[10],node[7];
rz(0.00390625*pi) node[12];
cx node[21],node[18];
cx node[23],node[24];
cx node[7],node[6];
cx node[12],node[15];
rz(3.96875*pi) node[18];
cx node[25],node[24];
rz(3.999755859375*pi) node[6];
cx node[15],node[12];
cx node[21],node[18];
rz(3.75*pi) node[24];
cx node[7],node[6];
cx node[12],node[15];
rz(0.03125*pi) node[18];
cx node[25],node[24];
cx node[7],node[4];
rz(0.000244140625*pi) node[6];
cx node[12],node[13];
cx node[21],node[18];
rz(0.25*pi) node[24];
rz(3.9998779296875*pi) node[4];
rz(3.998046875*pi) node[13];
cx node[18],node[21];
sx node[24];
cx node[7],node[4];
cx node[12],node[13];
cx node[21],node[18];
rz(0.5*pi) node[24];
rz(0.0001220703125*pi) node[4];
cx node[6],node[7];
cx node[12],node[10];
rz(0.001953125*pi) node[13];
cx node[18],node[17];
cx node[23],node[21];
sx node[24];
cx node[7],node[6];
rz(3.9990234375*pi) node[10];
rz(3.984375*pi) node[17];
rz(3.9375*pi) node[21];
rz(0.49609375000000044*pi) node[24];
cx node[6],node[7];
cx node[12],node[10];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
rz(0.0009765625*pi) node[10];
cx node[18],node[15];
rz(0.015625*pi) node[17];
rz(0.0625*pi) node[21];
cx node[24],node[25];
cx node[12],node[10];
rz(3.9921875*pi) node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[10],node[12];
cx node[18],node[15];
cx node[21],node[23];
cx node[12],node[10];
rz(0.0078125*pi) node[15];
cx node[23],node[21];
cx node[10],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[24],node[23];
rz(3.99951171875*pi) node[7];
cx node[12],node[13];
cx node[15],node[18];
rz(3.875*pi) node[23];
cx node[10],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[24],node[23];
rz(0.00048828125*pi) node[7];
cx node[15],node[12];
cx node[17],node[18];
rz(0.125*pi) node[23];
cx node[10],node[7];
rz(3.99609375*pi) node[12];
cx node[18],node[17];
cx node[23],node[24];
cx node[7],node[10];
cx node[15],node[12];
cx node[17],node[18];
cx node[24],node[23];
cx node[10],node[7];
rz(0.00390625*pi) node[12];
cx node[21],node[18];
cx node[23],node[24];
cx node[7],node[4];
cx node[12],node[15];
rz(3.96875*pi) node[18];
cx node[25],node[24];
rz(3.999755859375*pi) node[4];
cx node[15],node[12];
cx node[21],node[18];
rz(3.75*pi) node[24];
cx node[7],node[4];
cx node[12],node[15];
rz(0.03125*pi) node[18];
cx node[25],node[24];
rz(0.000244140625*pi) node[4];
cx node[12],node[13];
cx node[21],node[18];
rz(0.25*pi) node[24];
cx node[4],node[7];
rz(3.998046875*pi) node[13];
cx node[18],node[21];
sx node[24];
cx node[7],node[4];
cx node[12],node[13];
cx node[21],node[18];
rz(0.5*pi) node[24];
cx node[4],node[7];
cx node[12],node[10];
rz(0.001953125*pi) node[13];
cx node[18],node[17];
cx node[23],node[21];
sx node[24];
rz(3.9990234375*pi) node[10];
rz(3.984375*pi) node[17];
rz(3.9375*pi) node[21];
rz(0.49218750000000044*pi) node[24];
cx node[12],node[10];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
rz(0.0009765625*pi) node[10];
cx node[18],node[15];
rz(0.015625*pi) node[17];
rz(0.0625*pi) node[21];
cx node[24],node[25];
cx node[12],node[10];
rz(3.9921875*pi) node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[10],node[12];
cx node[18],node[15];
cx node[21],node[23];
cx node[12],node[10];
rz(0.0078125*pi) node[15];
cx node[23],node[21];
cx node[10],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[24],node[23];
rz(3.99951171875*pi) node[7];
cx node[12],node[13];
cx node[15],node[18];
rz(3.875*pi) node[23];
cx node[10],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[24],node[23];
rz(0.00048828125*pi) node[7];
cx node[15],node[12];
cx node[17],node[18];
rz(0.125*pi) node[23];
cx node[7],node[10];
rz(3.99609375*pi) node[12];
cx node[18],node[17];
cx node[23],node[24];
cx node[10],node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[24],node[23];
cx node[7],node[10];
rz(0.00390625*pi) node[12];
cx node[21],node[18];
cx node[23],node[24];
cx node[12],node[15];
rz(3.96875*pi) node[18];
cx node[25],node[24];
cx node[15],node[12];
cx node[21],node[18];
rz(3.75*pi) node[24];
cx node[12],node[15];
rz(0.03125*pi) node[18];
cx node[25],node[24];
cx node[12],node[13];
cx node[21],node[18];
rz(0.25*pi) node[24];
rz(3.998046875*pi) node[13];
cx node[18],node[21];
sx node[24];
cx node[12],node[13];
cx node[21],node[18];
rz(0.5*pi) node[24];
cx node[12],node[10];
rz(0.001953125*pi) node[13];
cx node[18],node[17];
cx node[23],node[21];
sx node[24];
rz(3.9990234375*pi) node[10];
rz(3.984375*pi) node[17];
rz(3.9375*pi) node[21];
rz(0.48437500000000044*pi) node[24];
cx node[12],node[10];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
rz(0.0009765625*pi) node[10];
cx node[13],node[12];
cx node[18],node[15];
rz(0.015625*pi) node[17];
rz(0.0625*pi) node[21];
cx node[24],node[25];
cx node[12],node[13];
rz(3.9921875*pi) node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[13],node[12];
cx node[18],node[15];
cx node[21],node[23];
rz(0.0078125*pi) node[15];
cx node[23],node[21];
cx node[18],node[15];
cx node[24],node[23];
cx node[15],node[18];
rz(3.875*pi) node[23];
cx node[18],node[15];
cx node[24],node[23];
cx node[15],node[12];
cx node[17],node[18];
rz(0.125*pi) node[23];
rz(3.99609375*pi) node[12];
cx node[18],node[17];
cx node[23],node[24];
cx node[15],node[12];
cx node[17],node[18];
cx node[24],node[23];
rz(0.00390625*pi) node[12];
cx node[21],node[18];
cx node[23],node[24];
cx node[15],node[12];
rz(3.96875*pi) node[18];
cx node[25],node[24];
cx node[12],node[15];
cx node[21],node[18];
rz(3.75*pi) node[24];
cx node[15],node[12];
rz(0.03125*pi) node[18];
cx node[25],node[24];
cx node[12],node[10];
cx node[21],node[18];
rz(0.25*pi) node[24];
rz(3.998046875*pi) node[10];
cx node[18],node[21];
sx node[24];
cx node[12],node[10];
cx node[21],node[18];
rz(0.5*pi) node[24];
rz(0.001953125*pi) node[10];
cx node[18],node[17];
cx node[23],node[21];
sx node[24];
cx node[10],node[12];
rz(3.984375*pi) node[17];
rz(3.9375*pi) node[21];
rz(0.46875000000000044*pi) node[24];
cx node[12],node[10];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
cx node[10],node[12];
cx node[18],node[15];
rz(0.015625*pi) node[17];
rz(0.0625*pi) node[21];
cx node[24],node[25];
rz(3.9921875*pi) node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[18],node[15];
cx node[21],node[23];
rz(0.0078125*pi) node[15];
cx node[23],node[21];
cx node[18],node[15];
cx node[24],node[23];
cx node[15],node[18];
rz(3.875*pi) node[23];
cx node[18],node[15];
cx node[24],node[23];
cx node[15],node[12];
cx node[17],node[18];
rz(0.125*pi) node[23];
rz(3.99609375*pi) node[12];
cx node[18],node[17];
cx node[23],node[24];
cx node[15],node[12];
cx node[17],node[18];
cx node[24],node[23];
rz(0.00390625*pi) node[12];
cx node[21],node[18];
cx node[23],node[24];
cx node[12],node[15];
rz(3.96875*pi) node[18];
cx node[25],node[24];
cx node[15],node[12];
cx node[21],node[18];
rz(3.75*pi) node[24];
cx node[12],node[15];
rz(0.03125*pi) node[18];
cx node[25],node[24];
cx node[21],node[18];
rz(0.25*pi) node[24];
cx node[18],node[21];
sx node[24];
cx node[21],node[18];
rz(0.5*pi) node[24];
cx node[18],node[17];
cx node[23],node[21];
sx node[24];
rz(3.984375*pi) node[17];
rz(3.9375*pi) node[21];
rz(0.43750000000000044*pi) node[24];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
cx node[18],node[15];
rz(0.015625*pi) node[17];
rz(0.0625*pi) node[21];
cx node[24],node[25];
rz(3.9921875*pi) node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[18],node[15];
cx node[21],node[23];
rz(0.0078125*pi) node[15];
cx node[17],node[18];
cx node[23],node[21];
cx node[18],node[17];
cx node[24],node[23];
cx node[17],node[18];
rz(3.875*pi) node[23];
cx node[21],node[18];
cx node[24],node[23];
rz(3.96875*pi) node[18];
rz(0.125*pi) node[23];
cx node[21],node[18];
cx node[23],node[24];
rz(0.03125*pi) node[18];
cx node[24],node[23];
cx node[21],node[18];
cx node[23],node[24];
cx node[18],node[21];
cx node[25],node[24];
cx node[21],node[18];
rz(3.75*pi) node[24];
cx node[18],node[15];
cx node[23],node[21];
cx node[25],node[24];
rz(3.984375*pi) node[15];
rz(3.9375*pi) node[21];
rz(0.25*pi) node[24];
cx node[18],node[15];
cx node[23],node[21];
sx node[24];
rz(0.015625*pi) node[15];
rz(0.0625*pi) node[21];
rz(0.5*pi) node[24];
cx node[15],node[18];
cx node[23],node[21];
sx node[24];
cx node[18],node[15];
cx node[21],node[23];
rz(0.37500000000000044*pi) node[24];
cx node[15],node[18];
cx node[23],node[21];
cx node[25],node[24];
cx node[21],node[18];
cx node[24],node[25];
rz(3.96875*pi) node[18];
cx node[25],node[24];
cx node[21],node[18];
cx node[24],node[23];
rz(0.03125*pi) node[18];
rz(3.875*pi) node[23];
cx node[18],node[21];
cx node[24],node[23];
cx node[21],node[18];
rz(0.125*pi) node[23];
cx node[18],node[21];
cx node[24],node[23];
cx node[23],node[24];
cx node[24],node[23];
cx node[23],node[21];
cx node[25],node[24];
rz(3.9375*pi) node[21];
rz(3.75*pi) node[24];
cx node[23],node[21];
cx node[25],node[24];
rz(0.0625*pi) node[21];
rz(0.25*pi) node[24];
cx node[21],node[23];
sx node[24];
cx node[23],node[21];
rz(0.5*pi) node[24];
cx node[21],node[23];
sx node[24];
rz(0.25*pi) node[24];
cx node[23],node[24];
cx node[24],node[23];
cx node[23],node[24];
cx node[25],node[24];
rz(3.875*pi) node[24];
cx node[25],node[24];
rz(0.125*pi) node[24];
cx node[23],node[24];
rz(3.75*pi) node[24];
cx node[23],node[24];
rz(3.25*pi) node[24];
sx node[24];
rz(1.5*pi) node[24];
sx node[24];
rz(1.0*pi) node[24];
barrier node[5],node[3],node[0],node[2],node[1],node[6],node[4],node[7],node[13],node[10],node[12],node[17],node[15],node[18],node[21],node[25],node[23],node[24];
measure node[5] -> meas[0];
measure node[3] -> meas[1];
measure node[0] -> meas[2];
measure node[2] -> meas[3];
measure node[1] -> meas[4];
measure node[6] -> meas[5];
measure node[4] -> meas[6];
measure node[7] -> meas[7];
measure node[13] -> meas[8];
measure node[10] -> meas[9];
measure node[12] -> meas[10];
measure node[17] -> meas[11];
measure node[15] -> meas[12];
measure node[18] -> meas[13];
measure node[21] -> meas[14];
measure node[25] -> meas[15];
measure node[23] -> meas[16];
measure node[24] -> meas[17];
