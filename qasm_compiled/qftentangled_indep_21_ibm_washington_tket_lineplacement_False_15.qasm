OPENQASM 2.0;
include "qelib1.inc";

qreg node[27];
creg meas[21];
sx node[4];
rz(0.5*pi) node[4];
sx node[4];
cx node[4],node[3];
cx node[3],node[2];
sx node[4];
cx node[2],node[1];
rz(0.5*pi) node[4];
cx node[1],node[0];
sx node[4];
cx node[0],node[14];
rz(0.49999952316284224*pi) node[4];
cx node[4],node[3];
cx node[14],node[18];
rz(3.75*pi) node[3];
cx node[18],node[19];
cx node[4],node[3];
cx node[19],node[20];
rz(0.25*pi) node[3];
cx node[20],node[21];
sx node[3];
cx node[21],node[22];
rz(0.5*pi) node[3];
cx node[22],node[15];
sx node[3];
rz(0.49999904632568404*pi) node[3];
cx node[4],node[3];
cx node[3],node[4];
cx node[4],node[3];
cx node[3],node[2];
rz(3.875*pi) node[2];
cx node[3],node[2];
rz(0.125*pi) node[2];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[1];
cx node[4],node[3];
rz(3.9375*pi) node[1];
rz(3.75*pi) node[3];
cx node[2],node[1];
cx node[4],node[3];
rz(0.0625*pi) node[1];
rz(0.25*pi) node[3];
cx node[2],node[1];
sx node[3];
cx node[1],node[2];
rz(0.5*pi) node[3];
cx node[2],node[1];
sx node[3];
cx node[1],node[0];
rz(0.49999809265136763*pi) node[3];
rz(3.96875*pi) node[0];
cx node[4],node[3];
cx node[1],node[0];
cx node[3],node[4];
rz(0.03125*pi) node[0];
cx node[4],node[3];
cx node[1],node[0];
cx node[3],node[2];
cx node[0],node[1];
rz(3.875*pi) node[2];
cx node[1],node[0];
cx node[3],node[2];
cx node[0],node[14];
rz(0.125*pi) node[2];
cx node[3],node[2];
rz(3.984375*pi) node[14];
cx node[0],node[14];
cx node[2],node[3];
cx node[3],node[2];
rz(0.015625*pi) node[14];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[3];
cx node[14],node[0];
rz(3.9375*pi) node[1];
rz(3.75*pi) node[3];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[3];
rz(0.0625*pi) node[1];
rz(0.25*pi) node[3];
cx node[14],node[18];
cx node[2],node[1];
sx node[3];
rz(3.9921875*pi) node[18];
cx node[1],node[2];
rz(0.5*pi) node[3];
cx node[14],node[18];
cx node[2],node[1];
sx node[3];
rz(0.0078125*pi) node[18];
cx node[1],node[0];
rz(0.4999961853027348*pi) node[3];
cx node[14],node[18];
rz(3.96875*pi) node[0];
cx node[4],node[3];
cx node[18],node[14];
cx node[1],node[0];
cx node[3],node[4];
cx node[14],node[18];
rz(0.03125*pi) node[0];
cx node[4],node[3];
cx node[18],node[19];
cx node[1],node[0];
cx node[3],node[2];
rz(3.99609375*pi) node[19];
cx node[0],node[1];
rz(3.875*pi) node[2];
cx node[18],node[19];
cx node[1],node[0];
cx node[3],node[2];
rz(0.00390625*pi) node[19];
cx node[0],node[14];
rz(0.125*pi) node[2];
cx node[18],node[19];
cx node[3],node[2];
rz(3.984375*pi) node[14];
cx node[19],node[18];
cx node[0],node[14];
cx node[2],node[3];
cx node[18],node[19];
cx node[3],node[2];
rz(0.015625*pi) node[14];
cx node[19],node[20];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[3];
rz(3.998046875*pi) node[20];
cx node[14],node[0];
rz(3.9375*pi) node[1];
rz(3.75*pi) node[3];
cx node[19],node[20];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[3];
rz(0.001953125*pi) node[20];
rz(0.0625*pi) node[1];
rz(0.25*pi) node[3];
cx node[14],node[18];
cx node[19],node[20];
cx node[2],node[1];
sx node[3];
rz(3.9921875*pi) node[18];
cx node[20],node[19];
cx node[1],node[2];
rz(0.5*pi) node[3];
cx node[14],node[18];
cx node[19],node[20];
cx node[2],node[1];
sx node[3];
rz(0.0078125*pi) node[18];
cx node[20],node[21];
cx node[1],node[0];
rz(0.4999923706054692*pi) node[3];
cx node[14],node[18];
rz(3.9990234375*pi) node[21];
rz(3.96875*pi) node[0];
cx node[4],node[3];
cx node[18],node[14];
cx node[20],node[21];
cx node[1],node[0];
cx node[3],node[4];
cx node[14],node[18];
rz(0.0009765625*pi) node[21];
rz(0.03125*pi) node[0];
cx node[4],node[3];
cx node[18],node[19];
cx node[20],node[21];
cx node[1],node[0];
cx node[3],node[2];
rz(3.99609375*pi) node[19];
cx node[21],node[20];
cx node[0],node[1];
rz(3.875*pi) node[2];
cx node[18],node[19];
cx node[20],node[21];
cx node[1],node[0];
cx node[3],node[2];
rz(0.00390625*pi) node[19];
cx node[21],node[22];
cx node[0],node[14];
rz(0.125*pi) node[2];
cx node[18],node[19];
rz(3.99951171875*pi) node[22];
cx node[3],node[2];
rz(3.984375*pi) node[14];
cx node[19],node[18];
cx node[21],node[22];
cx node[0],node[14];
cx node[2],node[3];
cx node[18],node[19];
rz(0.00048828125*pi) node[22];
cx node[3],node[2];
rz(0.015625*pi) node[14];
cx node[19],node[20];
cx node[22],node[21];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[3];
rz(3.998046875*pi) node[20];
cx node[21],node[22];
cx node[14],node[0];
rz(3.9375*pi) node[1];
rz(3.75*pi) node[3];
cx node[19],node[20];
cx node[22],node[21];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[3];
rz(0.001953125*pi) node[20];
rz(0.0625*pi) node[1];
rz(0.25*pi) node[3];
cx node[14],node[18];
cx node[19],node[20];
cx node[2],node[1];
sx node[3];
rz(3.9921875*pi) node[18];
cx node[20],node[19];
cx node[1],node[2];
rz(0.5*pi) node[3];
cx node[14],node[18];
cx node[19],node[20];
cx node[2],node[1];
sx node[3];
rz(0.0078125*pi) node[18];
cx node[20],node[21];
cx node[1],node[0];
rz(0.49998474121093794*pi) node[3];
cx node[14],node[18];
rz(3.9990234375*pi) node[21];
rz(3.96875*pi) node[0];
cx node[4],node[3];
cx node[18],node[14];
cx node[20],node[21];
cx node[1],node[0];
cx node[3],node[4];
cx node[14],node[18];
rz(0.0009765625*pi) node[21];
rz(0.03125*pi) node[0];
cx node[4],node[3];
cx node[18],node[19];
cx node[21],node[20];
cx node[1],node[0];
cx node[3],node[2];
rz(3.99609375*pi) node[19];
cx node[20],node[21];
cx node[0],node[1];
rz(3.875*pi) node[2];
cx node[18],node[19];
cx node[21],node[20];
cx node[1],node[0];
cx node[3],node[2];
rz(0.00390625*pi) node[19];
cx node[0],node[14];
rz(0.125*pi) node[2];
cx node[19],node[18];
cx node[3],node[2];
rz(3.984375*pi) node[14];
cx node[18],node[19];
cx node[0],node[14];
cx node[2],node[3];
cx node[19],node[18];
cx node[3],node[2];
rz(0.015625*pi) node[14];
cx node[19],node[20];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[3];
rz(3.998046875*pi) node[20];
cx node[14],node[0];
rz(3.9375*pi) node[1];
rz(3.75*pi) node[3];
cx node[19],node[20];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[3];
rz(0.001953125*pi) node[20];
rz(0.0625*pi) node[1];
rz(0.25*pi) node[3];
cx node[14],node[18];
cx node[20],node[19];
cx node[2],node[1];
sx node[3];
rz(3.9921875*pi) node[18];
cx node[19],node[20];
cx node[1],node[2];
rz(0.5*pi) node[3];
cx node[14],node[18];
cx node[20],node[19];
cx node[2],node[1];
sx node[3];
rz(0.0078125*pi) node[18];
cx node[1],node[0];
rz(0.49996948242187544*pi) node[3];
cx node[18],node[14];
rz(3.96875*pi) node[0];
cx node[4],node[3];
cx node[14],node[18];
cx node[1],node[0];
cx node[3],node[4];
cx node[18],node[14];
rz(0.03125*pi) node[0];
cx node[4],node[3];
cx node[18],node[19];
cx node[1],node[0];
cx node[3],node[2];
rz(3.99609375*pi) node[19];
cx node[0],node[1];
rz(3.875*pi) node[2];
cx node[18],node[19];
cx node[1],node[0];
cx node[3],node[2];
rz(0.00390625*pi) node[19];
cx node[0],node[14];
rz(0.125*pi) node[2];
cx node[19],node[18];
cx node[3],node[2];
rz(3.984375*pi) node[14];
cx node[18],node[19];
cx node[0],node[14];
cx node[2],node[3];
cx node[19],node[18];
cx node[3],node[2];
rz(0.015625*pi) node[14];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[3];
cx node[14],node[0];
rz(3.9375*pi) node[1];
rz(3.75*pi) node[3];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[3];
rz(0.0625*pi) node[1];
rz(0.25*pi) node[3];
cx node[14],node[18];
cx node[2],node[1];
sx node[3];
rz(3.9921875*pi) node[18];
cx node[1],node[2];
rz(0.5*pi) node[3];
cx node[14],node[18];
cx node[2],node[1];
sx node[3];
rz(0.0078125*pi) node[18];
cx node[1],node[0];
rz(0.49993896484375044*pi) node[3];
cx node[18],node[14];
rz(3.96875*pi) node[0];
cx node[4],node[3];
cx node[14],node[18];
cx node[1],node[0];
cx node[3],node[4];
cx node[18],node[14];
rz(0.03125*pi) node[0];
cx node[4],node[3];
cx node[1],node[0];
cx node[3],node[2];
cx node[0],node[1];
rz(3.875*pi) node[2];
cx node[1],node[0];
cx node[3],node[2];
cx node[0],node[14];
rz(0.125*pi) node[2];
cx node[3],node[2];
rz(3.984375*pi) node[14];
cx node[0],node[14];
cx node[2],node[3];
cx node[3],node[2];
rz(0.015625*pi) node[14];
cx node[14],node[0];
cx node[2],node[1];
cx node[4],node[3];
cx node[0],node[14];
rz(3.9375*pi) node[1];
rz(3.75*pi) node[3];
cx node[14],node[0];
cx node[2],node[1];
cx node[4],node[3];
rz(0.0625*pi) node[1];
rz(0.25*pi) node[3];
cx node[1],node[2];
sx node[3];
cx node[2],node[1];
rz(0.5*pi) node[3];
cx node[1],node[2];
sx node[3];
cx node[1],node[0];
rz(0.49987792968750044*pi) node[3];
rz(3.96875*pi) node[0];
cx node[4],node[3];
cx node[1],node[0];
cx node[3],node[4];
rz(0.03125*pi) node[0];
cx node[4],node[3];
cx node[0],node[1];
cx node[3],node[2];
cx node[1],node[0];
rz(3.875*pi) node[2];
cx node[0],node[1];
cx node[3],node[2];
rz(0.125*pi) node[2];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[1];
cx node[4],node[3];
rz(3.9375*pi) node[1];
rz(3.75*pi) node[3];
cx node[2],node[1];
cx node[4],node[3];
rz(0.0625*pi) node[1];
rz(0.25*pi) node[3];
cx node[1],node[2];
sx node[3];
cx node[2],node[1];
rz(0.5*pi) node[3];
cx node[1],node[2];
sx node[3];
rz(0.49975585937500044*pi) node[3];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[3];
cx node[4],node[3];
rz(3.875*pi) node[3];
cx node[4],node[3];
rz(0.125*pi) node[3];
cx node[5],node[4];
cx node[2],node[3];
cx node[4],node[5];
rz(3.75*pi) node[3];
cx node[5],node[4];
cx node[2],node[3];
cx node[15],node[4];
rz(0.25*pi) node[3];
cx node[5],node[4];
cx node[22],node[15];
sx node[3];
cx node[4],node[5];
rz(3.999755859375*pi) node[15];
rz(0.5*pi) node[3];
cx node[5],node[4];
cx node[22],node[15];
sx node[3];
cx node[5],node[6];
rz(0.000244140625*pi) node[15];
rz(0.49951171875000044*pi) node[3];
cx node[6],node[7];
cx node[15],node[22];
cx node[4],node[3];
cx node[7],node[8];
cx node[22],node[15];
cx node[3],node[4];
cx node[8],node[16];
cx node[15],node[22];
cx node[4],node[3];
cx node[16],node[26];
cx node[21],node[22];
rz(3.99951171875*pi) node[22];
cx node[26],node[25];
cx node[21],node[22];
cx node[25],node[24];
rz(0.00048828125*pi) node[22];
cx node[24],node[23];
rz(1.9073486328125e-06*pi) node[25];
cx node[22],node[21];
rz(3.337860107421875e-06*pi) node[23];
rz(2.86102294921875e-06*pi) node[24];
cx node[21],node[22];
cx node[22],node[21];
cx node[20],node[21];
rz(3.9990234375*pi) node[21];
cx node[20],node[21];
rz(0.0009765625*pi) node[21];
cx node[21],node[20];
cx node[20],node[21];
cx node[21],node[20];
cx node[19],node[20];
rz(3.998046875*pi) node[20];
cx node[19],node[20];
rz(0.001953125*pi) node[20];
cx node[20],node[19];
cx node[19],node[20];
cx node[20],node[19];
cx node[18],node[19];
rz(3.99609375*pi) node[19];
cx node[18],node[19];
rz(0.00390625*pi) node[19];
cx node[19],node[18];
cx node[18],node[19];
cx node[19],node[18];
cx node[14],node[18];
rz(3.9921875*pi) node[18];
cx node[14],node[18];
rz(0.0078125*pi) node[18];
cx node[18],node[14];
cx node[14],node[18];
cx node[18],node[14];
cx node[0],node[14];
rz(3.984375*pi) node[14];
cx node[0],node[14];
rz(0.015625*pi) node[14];
cx node[14],node[0];
cx node[0],node[14];
cx node[14],node[0];
cx node[1],node[0];
rz(3.96875*pi) node[0];
cx node[1],node[0];
rz(0.03125*pi) node[0];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[3],node[2];
rz(3.9375*pi) node[2];
cx node[3],node[2];
rz(0.0625*pi) node[2];
cx node[4],node[3];
cx node[1],node[2];
cx node[3],node[4];
rz(3.875*pi) node[2];
cx node[4],node[3];
cx node[1],node[2];
cx node[5],node[4];
rz(0.125*pi) node[2];
cx node[4],node[5];
cx node[3],node[2];
cx node[5],node[4];
rz(3.75*pi) node[2];
cx node[15],node[4];
cx node[3],node[2];
rz(3.9998779296875*pi) node[4];
rz(0.25*pi) node[2];
cx node[15],node[4];
sx node[2];
rz(0.0001220703125*pi) node[4];
rz(0.5*pi) node[2];
cx node[4],node[15];
sx node[2];
cx node[15],node[4];
rz(0.49902343750000044*pi) node[2];
cx node[4],node[15];
cx node[3],node[2];
cx node[4],node[5];
cx node[22],node[15];
cx node[2],node[3];
cx node[5],node[4];
rz(3.999755859375*pi) node[15];
cx node[3],node[2];
cx node[4],node[5];
cx node[22],node[15];
cx node[5],node[6];
rz(0.000244140625*pi) node[15];
rz(3.99993896484375*pi) node[6];
cx node[15],node[22];
cx node[5],node[6];
cx node[22],node[15];
rz(6.103515625e-05*pi) node[6];
cx node[15],node[22];
cx node[15],node[4];
cx node[5],node[6];
cx node[21],node[22];
cx node[4],node[15];
cx node[6],node[5];
rz(3.99951171875*pi) node[22];
cx node[15],node[4];
cx node[5],node[6];
cx node[21],node[22];
cx node[4],node[5];
cx node[6],node[7];
rz(0.00048828125*pi) node[22];
rz(3.9998779296875*pi) node[5];
rz(3.999969482421875*pi) node[7];
cx node[22],node[21];
cx node[4],node[5];
cx node[6],node[7];
cx node[21],node[22];
rz(0.0001220703125*pi) node[5];
rz(3.0517578125e-05*pi) node[7];
cx node[22],node[21];
cx node[4],node[5];
cx node[6],node[7];
cx node[22],node[15];
cx node[20],node[21];
cx node[5],node[4];
cx node[7],node[6];
cx node[15],node[22];
rz(3.9990234375*pi) node[21];
cx node[4],node[5];
cx node[6],node[7];
cx node[22],node[15];
cx node[20],node[21];
cx node[15],node[4];
cx node[5],node[6];
cx node[7],node[8];
rz(0.0009765625*pi) node[21];
rz(3.999755859375*pi) node[4];
rz(3.99993896484375*pi) node[6];
rz(3.9999847412109375*pi) node[8];
cx node[21],node[20];
cx node[15],node[4];
cx node[5],node[6];
cx node[7],node[8];
cx node[20],node[21];
rz(0.000244140625*pi) node[4];
rz(6.103515625e-05*pi) node[6];
rz(1.52587890625e-05*pi) node[8];
cx node[21],node[20];
cx node[4],node[15];
cx node[5],node[6];
cx node[8],node[7];
cx node[19],node[20];
cx node[21],node[22];
cx node[15],node[4];
cx node[6],node[5];
cx node[7],node[8];
rz(3.998046875*pi) node[20];
cx node[22],node[21];
cx node[4],node[15];
cx node[5],node[6];
cx node[8],node[7];
cx node[19],node[20];
cx node[21],node[22];
cx node[4],node[5];
cx node[6],node[7];
cx node[8],node[16];
cx node[22],node[15];
rz(0.001953125*pi) node[20];
rz(3.9998779296875*pi) node[5];
rz(3.999969482421875*pi) node[7];
rz(3.99951171875*pi) node[15];
rz(3.9999923706054688*pi) node[16];
cx node[20],node[19];
cx node[4],node[5];
cx node[6],node[7];
cx node[8],node[16];
cx node[22],node[15];
cx node[19],node[20];
rz(0.0001220703125*pi) node[5];
rz(3.0517578125e-05*pi) node[7];
rz(0.00048828125*pi) node[15];
rz(7.62939453125e-06*pi) node[16];
cx node[20],node[19];
cx node[4],node[5];
cx node[6],node[7];
cx node[16],node[8];
cx node[22],node[15];
cx node[18],node[19];
cx node[20],node[21];
cx node[5],node[4];
cx node[7],node[6];
cx node[8],node[16];
cx node[15],node[22];
rz(3.99609375*pi) node[19];
cx node[21],node[20];
cx node[4],node[5];
cx node[6],node[7];
cx node[16],node[8];
cx node[22],node[15];
cx node[18],node[19];
cx node[20],node[21];
cx node[15],node[4];
cx node[5],node[6];
cx node[7],node[8];
cx node[16],node[26];
rz(0.00390625*pi) node[19];
cx node[21],node[22];
rz(3.999755859375*pi) node[4];
rz(3.99993896484375*pi) node[6];
rz(3.9999847412109375*pi) node[8];
cx node[19],node[18];
rz(3.9990234375*pi) node[22];
rz(3.9999961853027344*pi) node[26];
cx node[15],node[4];
cx node[5],node[6];
cx node[7],node[8];
cx node[16],node[26];
cx node[18],node[19];
cx node[21],node[22];
rz(0.000244140625*pi) node[4];
rz(6.103515625e-05*pi) node[6];
rz(1.52587890625e-05*pi) node[8];
cx node[19],node[18];
rz(0.0009765625*pi) node[22];
rz(3.814697265625e-06*pi) node[26];
cx node[15],node[4];
cx node[5],node[6];
cx node[7],node[8];
cx node[14],node[18];
cx node[26],node[16];
cx node[20],node[19];
cx node[21],node[22];
cx node[4],node[15];
cx node[6],node[5];
cx node[8],node[7];
cx node[16],node[26];
rz(3.9921875*pi) node[18];
cx node[19],node[20];
cx node[22],node[21];
cx node[15],node[4];
cx node[5],node[6];
cx node[7],node[8];
cx node[14],node[18];
cx node[26],node[16];
cx node[20],node[19];
cx node[21],node[22];
cx node[4],node[5];
cx node[6],node[7];
cx node[8],node[16];
cx node[22],node[15];
rz(0.0078125*pi) node[18];
cx node[20],node[21];
cx node[25],node[26];
rz(3.9998779296875*pi) node[5];
rz(3.999969482421875*pi) node[7];
cx node[18],node[14];
rz(3.99951171875*pi) node[15];
rz(3.9999923706054688*pi) node[16];
rz(3.998046875*pi) node[21];
cx node[26],node[25];
cx node[4],node[5];
cx node[6],node[7];
cx node[8],node[16];
cx node[14],node[18];
cx node[22],node[15];
cx node[20],node[21];
cx node[25],node[26];
rz(0.0001220703125*pi) node[5];
rz(3.0517578125e-05*pi) node[7];
cx node[18],node[14];
rz(0.00048828125*pi) node[15];
rz(7.62939453125e-06*pi) node[16];
rz(0.001953125*pi) node[21];
cx node[24],node[25];
cx node[0],node[14];
cx node[4],node[5];
cx node[6],node[7];
cx node[16],node[8];
cx node[22],node[15];
cx node[19],node[18];
cx node[21],node[20];
cx node[25],node[24];
cx node[5],node[4];
cx node[7],node[6];
cx node[8],node[16];
rz(3.984375*pi) node[14];
cx node[15],node[22];
cx node[18],node[19];
cx node[20],node[21];
cx node[24],node[25];
cx node[0],node[14];
cx node[4],node[5];
cx node[6],node[7];
cx node[16],node[8];
cx node[22],node[15];
cx node[19],node[18];
cx node[21],node[20];
cx node[23],node[24];
cx node[15],node[4];
cx node[5],node[6];
cx node[7],node[8];
rz(0.015625*pi) node[14];
cx node[16],node[26];
cx node[19],node[20];
cx node[21],node[22];
cx node[24],node[23];
rz(3.999755859375*pi) node[4];
rz(3.99993896484375*pi) node[6];
rz(3.9999847412109375*pi) node[8];
cx node[18],node[14];
rz(3.99609375*pi) node[20];
rz(3.9990234375*pi) node[22];
cx node[23],node[24];
rz(3.9999961853027344*pi) node[26];
cx node[15],node[4];
cx node[5],node[6];
cx node[7],node[8];
rz(3.96875*pi) node[14];
cx node[16],node[26];
cx node[19],node[20];
cx node[21],node[22];
rz(0.000244140625*pi) node[4];
rz(6.103515625e-05*pi) node[6];
rz(1.52587890625e-05*pi) node[8];
cx node[18],node[14];
rz(0.00390625*pi) node[20];
rz(0.0009765625*pi) node[22];
rz(3.814697265625e-06*pi) node[26];
cx node[4],node[15];
cx node[5],node[6];
cx node[7],node[8];
rz(0.03125*pi) node[14];
cx node[20],node[19];
cx node[21],node[22];
cx node[0],node[14];
cx node[15],node[4];
cx node[6],node[5];
cx node[8],node[7];
cx node[19],node[20];
cx node[22],node[21];
cx node[14],node[0];
cx node[4],node[15];
cx node[5],node[6];
cx node[7],node[8];
cx node[20],node[19];
cx node[21],node[22];
cx node[0],node[14];
cx node[4],node[5];
cx node[6],node[7];
cx node[8],node[16];
cx node[22],node[15];
cx node[20],node[21];
cx node[1],node[0];
rz(3.9998779296875*pi) node[5];
rz(3.999969482421875*pi) node[7];
cx node[16],node[8];
cx node[14],node[18];
rz(3.99951171875*pi) node[15];
rz(3.998046875*pi) node[21];
rz(3.9375*pi) node[0];
cx node[4],node[5];
cx node[6],node[7];
cx node[8],node[16];
cx node[18],node[14];
cx node[22],node[15];
cx node[20],node[21];
cx node[1],node[0];
rz(0.0001220703125*pi) node[5];
rz(3.0517578125e-05*pi) node[7];
cx node[14],node[18];
rz(0.00048828125*pi) node[15];
cx node[16],node[26];
rz(0.001953125*pi) node[21];
rz(0.0625*pi) node[0];
cx node[4],node[5];
cx node[6],node[7];
cx node[22],node[15];
cx node[18],node[19];
cx node[20],node[21];
rz(3.9999923706054688*pi) node[26];
cx node[0],node[1];
cx node[5],node[4];
cx node[7],node[6];
cx node[15],node[22];
cx node[16],node[26];
rz(3.9921875*pi) node[19];
cx node[21],node[20];
cx node[1],node[0];
cx node[4],node[5];
cx node[6],node[7];
cx node[22],node[15];
cx node[18],node[19];
cx node[20],node[21];
rz(7.62939453125e-06*pi) node[26];
cx node[0],node[1];
cx node[15],node[4];
cx node[5],node[6];
cx node[7],node[8];
cx node[26],node[16];
rz(0.0078125*pi) node[19];
cx node[21],node[22];
cx node[2],node[1];
rz(3.999755859375*pi) node[4];
rz(3.99993896484375*pi) node[6];
cx node[8],node[7];
cx node[16],node[26];
cx node[19],node[18];
rz(3.9990234375*pi) node[22];
rz(3.875*pi) node[1];
cx node[15],node[4];
cx node[5],node[6];
cx node[7],node[8];
cx node[26],node[16];
cx node[18],node[19];
cx node[21],node[22];
cx node[2],node[1];
rz(0.000244140625*pi) node[4];
rz(6.103515625e-05*pi) node[6];
cx node[8],node[16];
cx node[19],node[18];
rz(0.0009765625*pi) node[22];
cx node[26],node[25];
rz(0.125*pi) node[1];
cx node[15],node[4];
cx node[5],node[6];
cx node[14],node[18];
rz(3.9999847412109375*pi) node[16];
cx node[19],node[20];
cx node[21],node[22];
rz(3.9999961853027344*pi) node[25];
cx node[2],node[1];
cx node[4],node[15];
cx node[6],node[5];
cx node[8],node[16];
rz(3.984375*pi) node[18];
rz(3.99609375*pi) node[20];
cx node[22],node[21];
cx node[26],node[25];
cx node[1],node[2];
cx node[15],node[4];
cx node[5],node[6];
cx node[14],node[18];
rz(1.52587890625e-05*pi) node[16];
cx node[19],node[20];
cx node[21],node[22];
rz(3.814697265625e-06*pi) node[25];
cx node[2],node[1];
cx node[4],node[5];
cx node[6],node[7];
cx node[8],node[16];
cx node[22],node[15];
rz(0.015625*pi) node[18];
rz(0.00390625*pi) node[20];
cx node[3],node[2];
rz(3.9998779296875*pi) node[5];
cx node[7],node[6];
cx node[16],node[8];
cx node[18],node[14];
rz(3.99951171875*pi) node[15];
cx node[20],node[19];
rz(3.75*pi) node[2];
cx node[4],node[5];
cx node[6],node[7];
cx node[8],node[16];
cx node[14],node[18];
cx node[22],node[15];
cx node[19],node[20];
cx node[3],node[2];
rz(0.0001220703125*pi) node[5];
cx node[7],node[8];
cx node[18],node[14];
rz(0.00048828125*pi) node[15];
cx node[16],node[26];
cx node[20],node[19];
cx node[0],node[14];
rz(0.25*pi) node[2];
cx node[4],node[5];
rz(3.999969482421875*pi) node[8];
cx node[22],node[15];
cx node[26],node[16];
cx node[18],node[19];
cx node[20],node[21];
sx node[2];
cx node[5],node[4];
cx node[7],node[8];
rz(3.96875*pi) node[14];
cx node[15],node[22];
cx node[16],node[26];
rz(3.9921875*pi) node[19];
rz(3.998046875*pi) node[21];
cx node[0],node[14];
rz(0.5*pi) node[2];
cx node[4],node[5];
rz(3.0517578125e-05*pi) node[8];
cx node[22],node[15];
cx node[18],node[19];
cx node[20],node[21];
cx node[26],node[25];
sx node[2];
cx node[15],node[4];
cx node[8],node[7];
rz(0.03125*pi) node[14];
rz(0.0078125*pi) node[19];
rz(0.001953125*pi) node[21];
rz(3.9999923706054688*pi) node[25];
cx node[0],node[14];
rz(0.49804687500000044*pi) node[2];
rz(3.999755859375*pi) node[4];
cx node[7],node[8];
cx node[19],node[18];
cx node[21],node[20];
cx node[26],node[25];
cx node[14],node[0];
cx node[3],node[2];
cx node[15],node[4];
cx node[8],node[7];
cx node[18],node[19];
cx node[20],node[21];
rz(7.62939453125e-06*pi) node[25];
cx node[0],node[14];
cx node[2],node[3];
rz(0.000244140625*pi) node[4];
cx node[7],node[6];
cx node[19],node[18];
cx node[21],node[20];
cx node[25],node[26];
cx node[1],node[0];
cx node[3],node[2];
cx node[15],node[4];
cx node[6],node[7];
cx node[14],node[18];
cx node[19],node[20];
cx node[21],node[22];
cx node[26],node[25];
rz(3.9375*pi) node[0];
cx node[4],node[15];
cx node[7],node[6];
rz(3.984375*pi) node[18];
rz(3.99609375*pi) node[20];
rz(3.9990234375*pi) node[22];
cx node[25],node[26];
cx node[1],node[0];
cx node[15],node[4];
cx node[5],node[6];
cx node[14],node[18];
cx node[26],node[16];
cx node[19],node[20];
cx node[21],node[22];
cx node[25],node[24];
rz(0.0625*pi) node[0];
rz(3.99993896484375*pi) node[6];
cx node[16],node[26];
rz(0.015625*pi) node[18];
rz(0.00390625*pi) node[20];
rz(0.0009765625*pi) node[22];
rz(3.9999961853027344*pi) node[24];
cx node[0],node[1];
cx node[5],node[6];
cx node[14],node[18];
cx node[26],node[16];
cx node[20],node[19];
cx node[21],node[22];
cx node[25],node[24];
cx node[1],node[0];
rz(6.103515625e-05*pi) node[6];
cx node[8],node[16];
cx node[18],node[14];
cx node[19],node[20];
cx node[22],node[21];
rz(3.814697265625e-06*pi) node[24];
cx node[0],node[1];
cx node[5],node[6];
cx node[14],node[18];
rz(3.9999847412109375*pi) node[16];
cx node[20],node[19];
cx node[21],node[22];
cx node[24],node[25];
cx node[0],node[14];
cx node[2],node[1];
cx node[6],node[5];
cx node[8],node[16];
cx node[22],node[15];
cx node[18],node[19];
cx node[20],node[21];
cx node[25],node[24];
rz(3.875*pi) node[1];
cx node[5],node[6];
rz(3.96875*pi) node[14];
rz(3.99951171875*pi) node[15];
rz(1.52587890625e-05*pi) node[16];
rz(3.9921875*pi) node[19];
rz(3.998046875*pi) node[21];
cx node[24],node[25];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[5];
cx node[6],node[7];
cx node[8],node[16];
cx node[22],node[15];
cx node[18],node[19];
cx node[20],node[21];
rz(0.125*pi) node[1];
rz(3.9998779296875*pi) node[5];
cx node[7],node[6];
cx node[16],node[8];
rz(0.03125*pi) node[14];
rz(0.00048828125*pi) node[15];
rz(0.0078125*pi) node[19];
rz(0.001953125*pi) node[21];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[5];
cx node[6],node[7];
cx node[8],node[16];
cx node[22],node[15];
cx node[18],node[19];
cx node[20],node[21];
cx node[14],node[0];
cx node[1],node[2];
rz(0.0001220703125*pi) node[5];
cx node[7],node[8];
cx node[15],node[22];
cx node[16],node[26];
cx node[19],node[18];
cx node[21],node[20];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[5];
rz(3.999969482421875*pi) node[8];
cx node[22],node[15];
cx node[26],node[16];
cx node[18],node[19];
cx node[20],node[21];
cx node[1],node[0];
cx node[3],node[2];
cx node[5],node[4];
cx node[7],node[8];
cx node[14],node[18];
cx node[16],node[26];
cx node[19],node[20];
cx node[21],node[22];
rz(3.9375*pi) node[0];
rz(3.75*pi) node[2];
cx node[4],node[5];
rz(3.0517578125e-05*pi) node[8];
rz(3.984375*pi) node[18];
rz(3.99609375*pi) node[20];
rz(3.9990234375*pi) node[22];
cx node[26],node[25];
cx node[1],node[0];
cx node[3],node[2];
cx node[15],node[4];
cx node[7],node[8];
cx node[14],node[18];
cx node[19],node[20];
cx node[21],node[22];
rz(3.9999923706054688*pi) node[25];
rz(0.0625*pi) node[0];
rz(0.25*pi) node[2];
rz(3.999755859375*pi) node[4];
cx node[8],node[7];
rz(0.015625*pi) node[18];
rz(0.00390625*pi) node[20];
rz(0.0009765625*pi) node[22];
cx node[26],node[25];
cx node[0],node[1];
sx node[2];
cx node[15],node[4];
cx node[7],node[8];
cx node[14],node[18];
cx node[19],node[20];
cx node[21],node[22];
rz(7.62939453125e-06*pi) node[25];
cx node[1],node[0];
rz(0.5*pi) node[2];
rz(0.000244140625*pi) node[4];
cx node[7],node[6];
cx node[8],node[16];
cx node[18],node[14];
cx node[20],node[19];
cx node[22],node[21];
cx node[25],node[26];
cx node[0],node[1];
sx node[2];
cx node[4],node[15];
cx node[6],node[7];
cx node[16],node[8];
cx node[14],node[18];
cx node[19],node[20];
cx node[21],node[22];
cx node[26],node[25];
cx node[0],node[14];
rz(0.49609375000000044*pi) node[2];
cx node[15],node[4];
cx node[7],node[6];
cx node[8],node[16];
cx node[18],node[19];
cx node[20],node[21];
cx node[25],node[26];
cx node[3],node[2];
cx node[4],node[15];
cx node[5],node[6];
rz(3.96875*pi) node[14];
cx node[16],node[26];
rz(3.9921875*pi) node[19];
rz(3.998046875*pi) node[21];
cx node[0],node[14];
cx node[2],node[3];
rz(3.99993896484375*pi) node[6];
cx node[22],node[15];
cx node[18],node[19];
cx node[20],node[21];
rz(3.9999847412109375*pi) node[26];
cx node[3],node[2];
cx node[5],node[6];
rz(0.03125*pi) node[14];
rz(3.99951171875*pi) node[15];
cx node[16],node[26];
rz(0.0078125*pi) node[19];
rz(0.001953125*pi) node[21];
cx node[0],node[14];
cx node[2],node[1];
rz(6.103515625e-05*pi) node[6];
cx node[22],node[15];
cx node[18],node[19];
cx node[20],node[21];
rz(1.52587890625e-05*pi) node[26];
cx node[14],node[0];
rz(3.875*pi) node[1];
cx node[5],node[6];
rz(0.00048828125*pi) node[15];
cx node[26],node[16];
cx node[19],node[18];
cx node[21],node[20];
cx node[0],node[14];
cx node[2],node[1];
cx node[6],node[5];
cx node[22],node[15];
cx node[16],node[26];
cx node[18],node[19];
cx node[20],node[21];
rz(0.125*pi) node[1];
cx node[5],node[6];
cx node[14],node[18];
cx node[15],node[22];
cx node[26],node[16];
cx node[19],node[20];
cx node[2],node[1];
cx node[4],node[5];
cx node[6],node[7];
cx node[22],node[15];
rz(3.984375*pi) node[18];
rz(3.99609375*pi) node[20];
cx node[1],node[2];
rz(3.9998779296875*pi) node[5];
cx node[7],node[6];
cx node[14],node[18];
cx node[19],node[20];
cx node[21],node[22];
cx node[2],node[1];
cx node[4],node[5];
cx node[6],node[7];
rz(0.015625*pi) node[18];
rz(0.00390625*pi) node[20];
rz(3.9990234375*pi) node[22];
cx node[1],node[0];
cx node[3],node[2];
rz(0.0001220703125*pi) node[5];
cx node[7],node[8];
cx node[14],node[18];
cx node[19],node[20];
cx node[21],node[22];
rz(3.9375*pi) node[0];
rz(3.75*pi) node[2];
cx node[4],node[5];
cx node[8],node[7];
cx node[18],node[14];
cx node[20],node[19];
rz(0.0009765625*pi) node[22];
cx node[1],node[0];
cx node[3],node[2];
cx node[5],node[4];
cx node[7],node[8];
cx node[14],node[18];
cx node[19],node[20];
cx node[21],node[22];
rz(0.0625*pi) node[0];
rz(0.25*pi) node[2];
cx node[4],node[5];
cx node[8],node[16];
cx node[18],node[19];
cx node[22],node[21];
cx node[0],node[1];
sx node[2];
cx node[15],node[4];
cx node[5],node[6];
rz(3.999969482421875*pi) node[16];
rz(3.9921875*pi) node[19];
cx node[21],node[22];
cx node[1],node[0];
rz(0.5*pi) node[2];
rz(3.999755859375*pi) node[4];
cx node[6],node[5];
cx node[8],node[16];
cx node[18],node[19];
cx node[20],node[21];
cx node[0],node[1];
sx node[2];
cx node[15],node[4];
cx node[5],node[6];
rz(3.0517578125e-05*pi) node[16];
rz(0.0078125*pi) node[19];
rz(3.998046875*pi) node[21];
cx node[0],node[14];
rz(0.49218750000000044*pi) node[2];
rz(0.000244140625*pi) node[4];
cx node[6],node[7];
cx node[16],node[8];
cx node[18],node[19];
cx node[20],node[21];
cx node[3],node[2];
cx node[15],node[4];
cx node[7],node[6];
cx node[8],node[16];
rz(3.96875*pi) node[14];
cx node[19],node[18];
rz(0.001953125*pi) node[21];
cx node[0],node[14];
cx node[2],node[3];
cx node[4],node[15];
cx node[6],node[7];
cx node[16],node[8];
cx node[18],node[19];
cx node[20],node[21];
cx node[3],node[2];
cx node[15],node[4];
cx node[7],node[8];
rz(0.03125*pi) node[14];
cx node[21],node[20];
cx node[0],node[14];
cx node[2],node[1];
rz(3.99993896484375*pi) node[8];
cx node[22],node[15];
cx node[20],node[21];
cx node[14],node[0];
rz(3.875*pi) node[1];
cx node[7],node[8];
rz(3.99951171875*pi) node[15];
cx node[19],node[20];
cx node[0],node[14];
cx node[2],node[1];
rz(6.103515625e-05*pi) node[8];
cx node[22],node[15];
rz(3.99609375*pi) node[20];
rz(0.125*pi) node[1];
cx node[8],node[7];
cx node[14],node[18];
rz(0.00048828125*pi) node[15];
cx node[19],node[20];
cx node[2],node[1];
cx node[7],node[8];
cx node[22],node[15];
rz(3.984375*pi) node[18];
rz(0.00390625*pi) node[20];
cx node[1],node[2];
cx node[8],node[7];
cx node[14],node[18];
cx node[15],node[22];
cx node[19],node[20];
cx node[2],node[1];
cx node[7],node[6];
cx node[22],node[15];
rz(0.015625*pi) node[18];
cx node[20],node[19];
cx node[1],node[0];
cx node[3],node[2];
cx node[6],node[7];
cx node[14],node[18];
cx node[19],node[20];
cx node[21],node[22];
rz(3.9375*pi) node[0];
rz(3.75*pi) node[2];
cx node[7],node[6];
cx node[18],node[14];
rz(3.9990234375*pi) node[22];
cx node[1],node[0];
cx node[3],node[2];
cx node[6],node[5];
cx node[14],node[18];
cx node[21],node[22];
rz(0.0625*pi) node[0];
rz(0.25*pi) node[2];
cx node[5],node[6];
cx node[18],node[19];
rz(0.0009765625*pi) node[22];
cx node[0],node[1];
sx node[2];
cx node[6],node[5];
rz(3.9921875*pi) node[19];
cx node[21],node[22];
cx node[1],node[0];
rz(0.5*pi) node[2];
cx node[4],node[5];
cx node[18],node[19];
cx node[22],node[21];
cx node[0],node[1];
sx node[2];
rz(3.9998779296875*pi) node[5];
rz(0.0078125*pi) node[19];
cx node[21],node[22];
cx node[0],node[14];
rz(0.48437500000000044*pi) node[2];
cx node[4],node[5];
cx node[18],node[19];
cx node[20],node[21];
cx node[3],node[2];
rz(0.0001220703125*pi) node[5];
rz(3.96875*pi) node[14];
cx node[19],node[18];
rz(3.998046875*pi) node[21];
cx node[0],node[14];
cx node[2],node[3];
cx node[5],node[4];
cx node[18],node[19];
cx node[20],node[21];
cx node[3],node[2];
cx node[4],node[5];
rz(0.03125*pi) node[14];
rz(0.001953125*pi) node[21];
cx node[0],node[14];
cx node[2],node[1];
cx node[5],node[4];
cx node[21],node[20];
cx node[14],node[0];
rz(3.875*pi) node[1];
cx node[15],node[4];
cx node[20],node[21];
cx node[0],node[14];
cx node[2],node[1];
rz(3.999755859375*pi) node[4];
cx node[21],node[20];
rz(0.125*pi) node[1];
cx node[15],node[4];
cx node[14],node[18];
cx node[19],node[20];
cx node[2],node[1];
rz(0.000244140625*pi) node[4];
rz(3.984375*pi) node[18];
rz(3.99609375*pi) node[20];
cx node[1],node[2];
cx node[4],node[15];
cx node[14],node[18];
cx node[19],node[20];
cx node[2],node[1];
cx node[15],node[4];
rz(0.015625*pi) node[18];
rz(0.00390625*pi) node[20];
cx node[1],node[0];
cx node[3],node[2];
cx node[4],node[15];
cx node[14],node[18];
cx node[20],node[19];
rz(3.9375*pi) node[0];
rz(3.75*pi) node[2];
cx node[18],node[14];
cx node[22],node[15];
cx node[19],node[20];
cx node[1],node[0];
cx node[3],node[2];
cx node[14],node[18];
rz(3.99951171875*pi) node[15];
cx node[20],node[19];
rz(0.0625*pi) node[0];
rz(0.25*pi) node[2];
cx node[22],node[15];
cx node[18],node[19];
cx node[0],node[1];
sx node[2];
rz(0.00048828125*pi) node[15];
rz(3.9921875*pi) node[19];
cx node[1],node[0];
rz(0.5*pi) node[2];
cx node[15],node[22];
cx node[18],node[19];
cx node[0],node[1];
sx node[2];
cx node[22],node[15];
rz(0.0078125*pi) node[19];
cx node[0],node[14];
rz(0.46875000000000044*pi) node[2];
cx node[15],node[22];
cx node[18],node[19];
cx node[3],node[2];
rz(3.96875*pi) node[14];
cx node[19],node[18];
cx node[21],node[22];
cx node[0],node[14];
cx node[2],node[3];
cx node[18],node[19];
rz(3.9990234375*pi) node[22];
cx node[3],node[2];
rz(0.03125*pi) node[14];
cx node[21],node[22];
cx node[0],node[14];
cx node[2],node[1];
rz(0.0009765625*pi) node[22];
cx node[14],node[0];
rz(3.875*pi) node[1];
cx node[22],node[21];
cx node[0],node[14];
cx node[2],node[1];
cx node[21],node[22];
rz(0.125*pi) node[1];
cx node[14],node[18];
cx node[22],node[21];
cx node[2],node[1];
rz(3.984375*pi) node[18];
cx node[20],node[21];
cx node[1],node[2];
cx node[14],node[18];
rz(3.998046875*pi) node[21];
cx node[2],node[1];
rz(0.015625*pi) node[18];
cx node[20],node[21];
cx node[1],node[0];
cx node[3],node[2];
cx node[14],node[18];
rz(0.001953125*pi) node[21];
rz(3.9375*pi) node[0];
rz(3.75*pi) node[2];
cx node[18],node[14];
cx node[21],node[20];
cx node[1],node[0];
cx node[3],node[2];
cx node[14],node[18];
cx node[20],node[21];
rz(0.0625*pi) node[0];
rz(0.25*pi) node[2];
cx node[21],node[20];
cx node[0],node[1];
sx node[2];
cx node[19],node[20];
cx node[1],node[0];
rz(0.5*pi) node[2];
rz(3.99609375*pi) node[20];
cx node[0],node[1];
sx node[2];
cx node[19],node[20];
cx node[0],node[14];
rz(0.43750000000000044*pi) node[2];
rz(0.00390625*pi) node[20];
cx node[3],node[2];
rz(3.96875*pi) node[14];
cx node[20],node[19];
cx node[0],node[14];
cx node[2],node[3];
cx node[19],node[20];
cx node[3],node[2];
rz(0.03125*pi) node[14];
cx node[20],node[19];
cx node[0],node[14];
cx node[2],node[1];
cx node[18],node[19];
cx node[14],node[0];
rz(3.875*pi) node[1];
rz(3.9921875*pi) node[19];
cx node[0],node[14];
cx node[2],node[1];
cx node[18],node[19];
rz(0.125*pi) node[1];
rz(0.0078125*pi) node[19];
cx node[2],node[1];
cx node[19],node[18];
cx node[1],node[2];
cx node[18],node[19];
cx node[2],node[1];
cx node[19],node[18];
cx node[1],node[0];
cx node[3],node[2];
cx node[14],node[18];
rz(3.9375*pi) node[0];
rz(3.75*pi) node[2];
rz(3.984375*pi) node[18];
cx node[1],node[0];
cx node[3],node[2];
cx node[14],node[18];
rz(0.0625*pi) node[0];
rz(0.25*pi) node[2];
rz(0.015625*pi) node[18];
cx node[0],node[1];
sx node[2];
cx node[18],node[14];
cx node[1],node[0];
rz(0.5*pi) node[2];
cx node[14],node[18];
cx node[0],node[1];
sx node[2];
cx node[18],node[14];
cx node[0],node[14];
rz(0.37500000000000044*pi) node[2];
cx node[3],node[2];
rz(3.96875*pi) node[14];
cx node[0],node[14];
cx node[2],node[3];
cx node[3],node[2];
rz(0.03125*pi) node[14];
cx node[14],node[0];
cx node[2],node[1];
cx node[0],node[14];
rz(3.875*pi) node[1];
cx node[14],node[0];
cx node[2],node[1];
rz(0.125*pi) node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[1],node[0];
cx node[3],node[2];
rz(3.9375*pi) node[0];
rz(3.75*pi) node[2];
cx node[1],node[0];
cx node[3],node[2];
rz(0.0625*pi) node[0];
rz(0.25*pi) node[2];
cx node[0],node[1];
sx node[2];
cx node[1],node[0];
rz(0.5*pi) node[2];
cx node[0],node[1];
sx node[2];
rz(0.25*pi) node[2];
cx node[1],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[3],node[2];
rz(3.875*pi) node[2];
cx node[3],node[2];
rz(0.125*pi) node[2];
cx node[1],node[2];
rz(3.75*pi) node[2];
cx node[1],node[2];
rz(3.25*pi) node[2];
sx node[2];
rz(1.5*pi) node[2];
sx node[2];
rz(1.0*pi) node[2];
barrier node[23],node[6],node[7],node[24],node[25],node[26],node[16],node[8],node[5],node[4],node[15],node[22],node[21],node[20],node[19],node[18],node[14],node[0],node[3],node[1],node[2];
measure node[23] -> meas[0];
measure node[6] -> meas[1];
measure node[7] -> meas[2];
measure node[24] -> meas[3];
measure node[25] -> meas[4];
measure node[26] -> meas[5];
measure node[16] -> meas[6];
measure node[8] -> meas[7];
measure node[5] -> meas[8];
measure node[4] -> meas[9];
measure node[15] -> meas[10];
measure node[22] -> meas[11];
measure node[21] -> meas[12];
measure node[20] -> meas[13];
measure node[19] -> meas[14];
measure node[18] -> meas[15];
measure node[14] -> meas[16];
measure node[0] -> meas[17];
measure node[3] -> meas[18];
measure node[1] -> meas[19];
measure node[2] -> meas[20];
