OPENQASM 2.0;
include "qelib1.inc";

qreg node[27];
creg c[18];
sx node[0];
rz(1.5*pi) node[1];
x node[4];
x node[6];
rz(1.5*pi) node[7];
sx node[8];
rz(3.5*pi) node[10];
sx node[11];
rz(1.5*pi) node[12];
sx node[13];
sx node[14];
rz(3.5*pi) node[15];
sx node[16];
sx node[19];
sx node[20];
sx node[22];
sx node[24];
rz(1.5714015833386856*pi) node[25];
sx node[26];
rz(0.5*pi) node[0];
sx node[1];
sx node[4];
sx node[7];
rz(3.5*pi) node[8];
x node[10];
rz(3.5*pi) node[11];
sx node[12];
rz(3.5*pi) node[13];
rz(3.5*pi) node[14];
x node[15];
rz(3.5*pi) node[16];
rz(3.5*pi) node[19];
rz(3.5*pi) node[20];
rz(3.5*pi) node[22];
rz(3.5*pi) node[24];
rz(3.5*pi) node[26];
sx node[0];
rz(3.5*pi) node[1];
rz(3.5*pi) node[7];
sx node[8];
rz(0.5*pi) node[10];
sx node[11];
rz(0.5*pi) node[12];
sx node[13];
sx node[14];
rz(0.5*pi) node[15];
sx node[16];
sx node[19];
sx node[20];
sx node[22];
sx node[24];
sx node[26];
sx node[1];
sx node[7];
rz(1.0*pi) node[8];
rz(1.0*pi) node[11];
sx node[12];
rz(1.0*pi) node[13];
rz(1.0*pi) node[14];
rz(1.0*pi) node[16];
rz(1.0*pi) node[19];
rz(1.0*pi) node[20];
rz(1.0*pi) node[22];
rz(1.0*pi) node[24];
rz(1.0*pi) node[26];
rz(0.5*pi) node[1];
rz(0.5*pi) node[7];
rz(0.5*pi) node[12];
cx node[25],node[24];
rz(0.9285984048445683*pi) node[24];
x node[24];
rz(0.5*pi) node[24];
cx node[25],node[24];
cx node[25],node[22];
rz(3.571405409852697*pi) node[24];
rz(0.35719680764870454*pi) node[22];
x node[22];
rz(0.5*pi) node[22];
cx node[25],node[22];
rz(2.1428108217458264*pi) node[22];
cx node[25],node[26];
rz(0.21439362166360676*pi) node[26];
x node[26];
rz(0.5*pi) node[26];
cx node[25],node[26];
cx node[25],node[22];
rz(2.2856216371254554*pi) node[26];
cx node[22],node[25];
cx node[25],node[22];
cx node[22],node[19];
cx node[26],node[25];
rz(0.9287872262690526*pi) node[19];
cx node[25],node[26];
x node[19];
cx node[26],node[25];
rz(0.5*pi) node[19];
cx node[22],node[19];
rz(1.571243291309072*pi) node[19];
cx node[22],node[19];
cx node[19],node[22];
cx node[22],node[19];
cx node[19],node[16];
rz(0.35757446323006836*pi) node[16];
x node[16];
rz(0.5*pi) node[16];
cx node[19],node[16];
rz(2.1424865719261814*pi) node[16];
cx node[19],node[20];
rz(0.21514892327703805*pi) node[20];
x node[20];
rz(0.5*pi) node[20];
cx node[19],node[20];
cx node[19],node[16];
rz(2.2849731470354615*pi) node[20];
cx node[16],node[19];
cx node[19],node[16];
cx node[16],node[14];
cx node[20],node[19];
rz(0.9302978454114093*pi) node[14];
cx node[19],node[20];
x node[14];
cx node[20],node[19];
rz(0.5*pi) node[14];
cx node[16],node[14];
rz(1.5699462952135903*pi) node[14];
cx node[16],node[14];
cx node[14],node[16];
cx node[16],node[14];
cx node[14],node[11];
rz(0.36059570310633127*pi) node[11];
x node[11];
rz(0.5*pi) node[11];
cx node[14],node[11];
rz(2.139892578143668*pi) node[11];
cx node[14],node[13];
cx node[8],node[11];
rz(0.22119139984646485*pi) node[13];
cx node[11],node[8];
x node[13];
cx node[8],node[11];
rz(0.5*pi) node[13];
cx node[14],node[13];
cx node[14],node[11];
rz(2.278808600153535*pi) node[13];
rz(0.9423828144657573*pi) node[11];
x node[11];
rz(0.5*pi) node[11];
cx node[14],node[11];
rz(3.0576171855342427*pi) node[11];
rz(0.5*pi) node[14];
sx node[14];
rz(3.5*pi) node[14];
sx node[14];
rz(1.0*pi) node[14];
cx node[13],node[14];
cx node[14],node[13];
cx node[13],node[14];
cx node[11],node[14];
cx node[13],node[12];
rz(0.5019531249999996*pi) node[11];
rz(1.5*pi) node[13];
cx node[11],node[14];
sx node[13];
rz(0.11523437470046671*pi) node[13];
rz(0.0009765624999995559*pi) node[14];
cx node[11],node[14];
sx node[13];
cx node[14],node[11];
rz(1.5*pi) node[13];
cx node[11],node[14];
cx node[13],node[12];
sx node[12];
sx node[13];
rz(3.5*pi) node[12];
sx node[12];
rz(3.6191406247004667*pi) node[12];
cx node[13],node[12];
cx node[12],node[13];
cx node[13],node[12];
cx node[12],node[10];
rz(1.5*pi) node[12];
sx node[12];
rz(0.23046875576713166*pi) node[12];
sx node[12];
rz(1.5*pi) node[12];
cx node[12],node[10];
sx node[10];
cx node[12],node[15];
rz(3.5*pi) node[10];
rz(1.5*pi) node[12];
sx node[10];
sx node[12];
rz(1.2382812557671303*pi) node[10];
rz(0.46093749561876884*pi) node[12];
sx node[12];
rz(0.5*pi) node[12];
cx node[12],node[15];
sx node[12];
sx node[15];
cx node[12],node[10];
rz(3.5*pi) node[15];
cx node[10],node[12];
sx node[15];
cx node[12],node[10];
rz(1.4765624956187677*pi) node[15];
cx node[10],node[7];
cx node[15],node[12];
rz(0.5*pi) node[10];
cx node[12],node[15];
sx node[10];
cx node[15],node[12];
rz(1.9218750003379501*pi) node[10];
sx node[10];
rz(1.5*pi) node[10];
cx node[10],node[7];
sx node[7];
rz(3.5*pi) node[7];
sx node[7];
rz(0.4531250003379498*pi) node[7];
cx node[10],node[7];
cx node[7],node[10];
cx node[10],node[7];
cx node[7],node[4];
rz(1.5*pi) node[7];
sx node[7];
rz(0.15625000000000022*pi) node[7];
sx node[7];
rz(1.5*pi) node[7];
cx node[7],node[4];
sx node[4];
sx node[7];
rz(2.5*pi) node[4];
cx node[7],node[6];
sx node[4];
rz(1.5*pi) node[7];
rz(1.4062499999999996*pi) node[4];
sx node[7];
rz(0.3124999999999998*pi) node[7];
sx node[7];
rz(1.5*pi) node[7];
cx node[7],node[6];
sx node[6];
sx node[7];
cx node[7],node[4];
rz(2.5*pi) node[6];
cx node[4],node[7];
sx node[6];
cx node[7],node[4];
rz(1.8124999999999993*pi) node[6];
cx node[4],node[1];
cx node[6],node[7];
rz(1.5*pi) node[4];
cx node[7],node[6];
sx node[4];
cx node[6],node[7];
rz(3.374999999999999*pi) node[4];
sx node[4];
rz(1.0*pi) node[4];
cx node[4],node[1];
sx node[1];
rz(0.5*pi) node[4];
rz(3.5*pi) node[1];
sx node[4];
sx node[1];
rz(3.5*pi) node[4];
rz(0.12499999999999867*pi) node[1];
sx node[4];
cx node[0],node[1];
rz(1.0*pi) node[4];
cx node[1],node[0];
cx node[0],node[1];
cx node[4],node[1];
rz(0.25*pi) node[1];
cx node[4],node[1];
rz(0.75*pi) node[1];
sx node[1];
rz(3.5*pi) node[1];
sx node[1];
rz(1.0*pi) node[1];
cx node[0],node[1];
rz(0.25*pi) node[1];
cx node[0],node[1];
rz(0.5*pi) node[0];
rz(3.75*pi) node[1];
sx node[0];
cx node[1],node[4];
rz(3.5*pi) node[0];
cx node[4],node[1];
sx node[0];
cx node[1],node[4];
rz(1.0*pi) node[0];
cx node[7],node[4];
cx node[0],node[1];
rz(0.125*pi) node[4];
cx node[1],node[0];
cx node[7],node[4];
cx node[0],node[1];
rz(3.875*pi) node[4];
cx node[7],node[4];
cx node[4],node[7];
cx node[7],node[4];
cx node[4],node[1];
cx node[6],node[7];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[7];
cx node[4],node[1];
cx node[6],node[7];
rz(3.75*pi) node[1];
rz(0.5*pi) node[4];
rz(3.9375*pi) node[7];
sx node[4];
cx node[10],node[7];
rz(3.5*pi) node[4];
rz(0.03125*pi) node[7];
sx node[4];
cx node[10],node[7];
rz(1.0*pi) node[4];
rz(3.96875*pi) node[7];
cx node[1],node[4];
cx node[7],node[10];
cx node[4],node[1];
cx node[10],node[7];
cx node[1],node[4];
cx node[7],node[10];
cx node[6],node[7];
cx node[12],node[10];
cx node[7],node[6];
rz(0.015625*pi) node[10];
cx node[6],node[7];
cx node[12],node[10];
cx node[7],node[4];
rz(3.984375*pi) node[10];
rz(0.125*pi) node[4];
cx node[12],node[10];
cx node[7],node[4];
cx node[10],node[12];
rz(3.875*pi) node[4];
cx node[12],node[10];
cx node[7],node[4];
cx node[15],node[12];
cx node[4],node[7];
rz(0.0078125*pi) node[12];
cx node[7],node[4];
cx node[15],node[12];
cx node[4],node[1];
cx node[6],node[7];
rz(3.9921875*pi) node[12];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[7];
cx node[13],node[12];
cx node[4],node[1];
cx node[6],node[7];
rz(0.00390625*pi) node[12];
rz(3.75*pi) node[1];
rz(0.5*pi) node[4];
rz(3.9375*pi) node[7];
cx node[13],node[12];
sx node[4];
cx node[10],node[7];
rz(3.99609375*pi) node[12];
rz(3.5*pi) node[4];
rz(0.03125*pi) node[7];
cx node[12],node[13];
sx node[4];
cx node[10],node[7];
cx node[13],node[12];
rz(1.0*pi) node[4];
rz(3.96875*pi) node[7];
cx node[12],node[13];
cx node[1],node[4];
cx node[7],node[10];
cx node[15],node[12];
cx node[14],node[13];
cx node[4],node[1];
cx node[10],node[7];
cx node[12],node[15];
rz(0.001953125*pi) node[13];
cx node[1],node[4];
cx node[7],node[10];
cx node[15],node[12];
cx node[14],node[13];
cx node[6],node[7];
cx node[12],node[10];
rz(3.998046875*pi) node[13];
cx node[7],node[6];
rz(0.015625*pi) node[10];
cx node[13],node[14];
cx node[6],node[7];
cx node[12],node[10];
cx node[14],node[13];
cx node[7],node[4];
rz(3.984375*pi) node[10];
cx node[13],node[14];
rz(0.125*pi) node[4];
cx node[12],node[10];
cx node[11],node[14];
cx node[7],node[4];
cx node[10],node[12];
rz(0.0009765625*pi) node[14];
rz(3.875*pi) node[4];
cx node[12],node[10];
cx node[11],node[14];
cx node[7],node[4];
cx node[15],node[12];
rz(3.9990234375*pi) node[14];
cx node[4],node[7];
cx node[14],node[11];
rz(0.0078125*pi) node[12];
cx node[7],node[4];
cx node[11],node[14];
cx node[15],node[12];
cx node[4],node[1];
cx node[6],node[7];
cx node[14],node[11];
rz(3.9921875*pi) node[12];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[7];
cx node[8],node[11];
cx node[13],node[12];
cx node[4],node[1];
cx node[6],node[7];
rz(0.00048828125*pi) node[11];
rz(0.00390625*pi) node[12];
rz(3.75*pi) node[1];
rz(0.5*pi) node[4];
rz(3.9375*pi) node[7];
cx node[8],node[11];
cx node[13],node[12];
sx node[4];
cx node[10],node[7];
rz(3.99951171875*pi) node[11];
rz(3.99609375*pi) node[12];
rz(3.5*pi) node[4];
rz(0.03125*pi) node[7];
cx node[12],node[13];
sx node[4];
cx node[10],node[7];
cx node[13],node[12];
rz(1.0*pi) node[4];
rz(3.96875*pi) node[7];
cx node[12],node[13];
cx node[1],node[4];
cx node[7],node[10];
cx node[15],node[12];
cx node[14],node[13];
cx node[4],node[1];
cx node[10],node[7];
cx node[12],node[15];
rz(0.001953125*pi) node[13];
cx node[1],node[4];
cx node[7],node[10];
cx node[15],node[12];
cx node[14],node[13];
cx node[6],node[7];
cx node[12],node[10];
rz(3.998046875*pi) node[13];
cx node[7],node[6];
rz(0.015625*pi) node[10];
cx node[13],node[14];
cx node[6],node[7];
cx node[12],node[10];
cx node[14],node[13];
cx node[7],node[4];
rz(3.984375*pi) node[10];
cx node[13],node[14];
rz(0.125*pi) node[4];
cx node[12],node[10];
cx node[14],node[11];
cx node[7],node[4];
cx node[10],node[12];
cx node[11],node[14];
rz(3.875*pi) node[4];
cx node[12],node[10];
cx node[14],node[11];
cx node[7],node[4];
cx node[8],node[11];
cx node[15],node[12];
cx node[16],node[14];
cx node[4],node[7];
rz(0.0009765625*pi) node[11];
rz(0.0078125*pi) node[12];
rz(0.000244140625*pi) node[14];
cx node[7],node[4];
cx node[8],node[11];
cx node[15],node[12];
cx node[16],node[14];
cx node[4],node[1];
cx node[6],node[7];
rz(3.9990234375*pi) node[11];
rz(3.9921875*pi) node[12];
rz(3.999755859375*pi) node[14];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[7];
cx node[13],node[12];
cx node[16],node[14];
cx node[4],node[1];
cx node[6],node[7];
rz(0.00390625*pi) node[12];
cx node[14],node[16];
rz(3.75*pi) node[1];
rz(0.5*pi) node[4];
rz(3.9375*pi) node[7];
cx node[13],node[12];
cx node[16],node[14];
sx node[4];
cx node[10],node[7];
cx node[14],node[11];
rz(3.99609375*pi) node[12];
cx node[19],node[16];
rz(3.5*pi) node[4];
rz(0.03125*pi) node[7];
rz(0.00048828125*pi) node[11];
cx node[12],node[13];
rz(0.0001220703125*pi) node[16];
sx node[4];
cx node[10],node[7];
cx node[14],node[11];
cx node[13],node[12];
cx node[19],node[16];
rz(1.0*pi) node[4];
rz(3.96875*pi) node[7];
rz(3.99951171875*pi) node[11];
cx node[12],node[13];
rz(3.9998779296875*pi) node[16];
cx node[1],node[4];
cx node[7],node[10];
cx node[11],node[14];
cx node[15],node[12];
cx node[19],node[16];
cx node[4],node[1];
cx node[10],node[7];
cx node[14],node[11];
cx node[12],node[15];
cx node[16],node[19];
cx node[1],node[4];
cx node[7],node[10];
cx node[11],node[14];
cx node[15],node[12];
cx node[19],node[16];
cx node[6],node[7];
cx node[8],node[11];
cx node[12],node[10];
cx node[16],node[14];
cx node[20],node[19];
cx node[7],node[6];
cx node[11],node[8];
rz(0.015625*pi) node[10];
rz(0.000244140625*pi) node[14];
rz(6.103515625e-05*pi) node[19];
cx node[6],node[7];
cx node[8],node[11];
cx node[12],node[10];
cx node[16],node[14];
cx node[20],node[19];
cx node[7],node[4];
rz(3.984375*pi) node[10];
rz(3.999755859375*pi) node[14];
rz(3.99993896484375*pi) node[19];
rz(0.125*pi) node[4];
cx node[12],node[10];
cx node[14],node[16];
cx node[22],node[19];
cx node[7],node[4];
cx node[10],node[12];
cx node[16],node[14];
rz(3.0517578125e-05*pi) node[19];
rz(3.875*pi) node[4];
cx node[12],node[10];
cx node[14],node[16];
cx node[22],node[19];
cx node[7],node[4];
cx node[15],node[12];
cx node[13],node[14];
rz(3.999969482421875*pi) node[19];
cx node[4],node[7];
rz(0.0078125*pi) node[12];
cx node[14],node[13];
cx node[19],node[22];
cx node[7],node[4];
cx node[15],node[12];
cx node[13],node[14];
cx node[22],node[19];
cx node[4],node[1];
cx node[6],node[7];
cx node[11],node[14];
rz(3.9921875*pi) node[12];
cx node[19],node[22];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[7];
rz(0.001953125*pi) node[14];
cx node[20],node[19];
cx node[25],node[22];
cx node[4],node[1];
cx node[6],node[7];
cx node[11],node[14];
cx node[19],node[20];
rz(1.52587890625e-05*pi) node[22];
rz(3.75*pi) node[1];
rz(0.5*pi) node[4];
rz(3.9375*pi) node[7];
rz(3.998046875*pi) node[14];
cx node[20],node[19];
cx node[25],node[22];
sx node[4];
cx node[10],node[7];
cx node[14],node[11];
cx node[19],node[16];
rz(3.9999847412109375*pi) node[22];
rz(3.5*pi) node[4];
rz(0.03125*pi) node[7];
cx node[11],node[14];
rz(0.0001220703125*pi) node[16];
cx node[25],node[22];
sx node[4];
cx node[10],node[7];
cx node[14],node[11];
cx node[19],node[16];
cx node[22],node[25];
rz(1.0*pi) node[4];
rz(3.96875*pi) node[7];
cx node[8],node[11];
cx node[13],node[14];
rz(3.9998779296875*pi) node[16];
cx node[25],node[22];
cx node[1],node[4];
cx node[7],node[10];
rz(0.0009765625*pi) node[11];
cx node[14],node[13];
cx node[19],node[16];
cx node[26],node[25];
cx node[4],node[1];
cx node[10],node[7];
cx node[8],node[11];
cx node[13],node[14];
cx node[16],node[19];
rz(7.62939453125e-06*pi) node[25];
cx node[1],node[4];
cx node[7],node[10];
rz(3.9990234375*pi) node[11];
cx node[13],node[12];
cx node[19],node[16];
cx node[26],node[25];
cx node[6],node[7];
cx node[14],node[11];
rz(0.00390625*pi) node[12];
cx node[20],node[19];
rz(3.9999923706054688*pi) node[25];
cx node[7],node[6];
rz(0.00048828125*pi) node[11];
cx node[13],node[12];
rz(6.103515625e-05*pi) node[19];
cx node[24],node[25];
cx node[6],node[7];
cx node[14],node[11];
rz(3.99609375*pi) node[12];
cx node[20],node[19];
rz(3.814697265625e-06*pi) node[25];
cx node[7],node[4];
rz(3.99951171875*pi) node[11];
cx node[12],node[13];
rz(3.99993896484375*pi) node[19];
cx node[24],node[25];
rz(0.125*pi) node[4];
cx node[13],node[12];
cx node[22],node[19];
rz(3.9999961853027344*pi) node[25];
cx node[7],node[4];
cx node[12],node[13];
rz(3.0517578125e-05*pi) node[19];
rz(3.875*pi) node[4];
cx node[15],node[12];
cx node[13],node[14];
cx node[22],node[19];
cx node[7],node[4];
cx node[12],node[15];
cx node[14],node[13];
rz(3.999969482421875*pi) node[19];
cx node[4],node[7];
cx node[15],node[12];
cx node[13],node[14];
cx node[19],node[22];
cx node[7],node[4];
cx node[12],node[10];
cx node[14],node[11];
cx node[22],node[19];
cx node[4],node[1];
cx node[6],node[7];
rz(0.015625*pi) node[10];
cx node[11],node[14];
cx node[19],node[22];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[7];
cx node[12],node[10];
cx node[14],node[11];
cx node[20],node[19];
cx node[22],node[25];
cx node[4],node[1];
cx node[6],node[7];
cx node[8],node[11];
rz(3.984375*pi) node[10];
cx node[16],node[14];
cx node[19],node[20];
cx node[25],node[22];
rz(3.75*pi) node[1];
rz(0.5*pi) node[4];
rz(3.9375*pi) node[7];
cx node[12],node[10];
rz(0.001953125*pi) node[11];
rz(0.000244140625*pi) node[14];
cx node[20],node[19];
cx node[22],node[25];
sx node[4];
cx node[8],node[11];
cx node[10],node[12];
cx node[16],node[14];
cx node[26],node[25];
rz(3.5*pi) node[4];
cx node[12],node[10];
rz(3.998046875*pi) node[11];
rz(3.999755859375*pi) node[14];
rz(1.52587890625e-05*pi) node[25];
sx node[4];
cx node[10],node[7];
cx node[8],node[11];
cx node[15],node[12];
cx node[14],node[16];
cx node[26],node[25];
rz(1.0*pi) node[4];
rz(0.03125*pi) node[7];
cx node[11],node[8];
rz(0.0078125*pi) node[12];
cx node[16],node[14];
rz(3.9999847412109375*pi) node[25];
cx node[1],node[4];
cx node[10],node[7];
cx node[8],node[11];
cx node[15],node[12];
cx node[14],node[16];
cx node[24],node[25];
cx node[4],node[1];
rz(3.96875*pi) node[7];
cx node[11],node[14];
rz(3.9921875*pi) node[12];
cx node[19],node[16];
rz(7.62939453125e-06*pi) node[25];
cx node[1],node[4];
cx node[7],node[10];
cx node[14],node[11];
rz(0.0001220703125*pi) node[16];
cx node[24],node[25];
cx node[10],node[7];
cx node[11],node[14];
cx node[19],node[16];
rz(3.9999923706054688*pi) node[25];
cx node[7],node[10];
cx node[8],node[11];
cx node[13],node[14];
rz(3.9998779296875*pi) node[16];
cx node[26],node[25];
cx node[6],node[7];
cx node[11],node[8];
cx node[14],node[13];
cx node[19],node[16];
cx node[25],node[26];
cx node[7],node[6];
cx node[8],node[11];
cx node[13],node[14];
cx node[16],node[19];
cx node[26],node[25];
cx node[6],node[7];
cx node[14],node[11];
cx node[13],node[12];
cx node[19],node[16];
cx node[25],node[22];
cx node[7],node[4];
rz(0.0009765625*pi) node[11];
rz(0.00390625*pi) node[12];
cx node[20],node[19];
cx node[22],node[25];
rz(0.125*pi) node[4];
cx node[14],node[11];
cx node[13],node[12];
rz(6.103515625e-05*pi) node[19];
cx node[25],node[22];
cx node[7],node[4];
rz(3.9990234375*pi) node[11];
rz(3.99609375*pi) node[12];
cx node[20],node[19];
cx node[24],node[25];
rz(3.875*pi) node[4];
cx node[8],node[11];
cx node[12],node[13];
rz(3.99993896484375*pi) node[19];
cx node[25],node[24];
cx node[7],node[4];
rz(0.00048828125*pi) node[11];
cx node[13],node[12];
cx node[22],node[19];
cx node[24],node[25];
cx node[4],node[7];
cx node[8],node[11];
cx node[12],node[13];
rz(3.0517578125e-05*pi) node[19];
cx node[7],node[4];
rz(3.99951171875*pi) node[11];
cx node[15],node[12];
cx node[14],node[13];
cx node[22],node[19];
cx node[4],node[1];
cx node[6],node[7];
cx node[12],node[15];
rz(0.001953125*pi) node[13];
rz(3.999969482421875*pi) node[19];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[7];
cx node[15],node[12];
cx node[14],node[13];
cx node[19],node[22];
cx node[4],node[1];
cx node[6],node[7];
cx node[12],node[10];
rz(3.998046875*pi) node[13];
cx node[22],node[19];
rz(3.75*pi) node[1];
rz(0.5*pi) node[4];
rz(3.9375*pi) node[7];
rz(0.015625*pi) node[10];
cx node[13],node[14];
cx node[19],node[22];
sx node[4];
cx node[12],node[10];
cx node[14],node[13];
cx node[20],node[19];
cx node[25],node[22];
rz(3.5*pi) node[4];
rz(3.984375*pi) node[10];
cx node[13],node[14];
cx node[19],node[20];
rz(1.52587890625e-05*pi) node[22];
sx node[4];
cx node[12],node[10];
cx node[14],node[11];
cx node[20],node[19];
cx node[25],node[22];
rz(1.0*pi) node[4];
cx node[10],node[12];
cx node[11],node[14];
rz(3.9999847412109375*pi) node[22];
cx node[1],node[4];
cx node[12],node[10];
cx node[14],node[11];
cx node[25],node[22];
cx node[4],node[1];
cx node[10],node[7];
cx node[8],node[11];
cx node[15],node[12];
cx node[16],node[14];
cx node[22],node[25];
cx node[1],node[4];
rz(0.03125*pi) node[7];
rz(0.0009765625*pi) node[11];
rz(0.0078125*pi) node[12];
rz(0.000244140625*pi) node[14];
cx node[25],node[22];
cx node[10],node[7];
cx node[8],node[11];
cx node[15],node[12];
cx node[16],node[14];
rz(3.96875*pi) node[7];
rz(3.9990234375*pi) node[11];
rz(3.9921875*pi) node[12];
rz(3.999755859375*pi) node[14];
cx node[7],node[10];
cx node[8],node[11];
cx node[13],node[12];
cx node[16],node[14];
cx node[10],node[7];
cx node[11],node[8];
rz(0.00390625*pi) node[12];
cx node[14],node[16];
cx node[7],node[10];
cx node[8],node[11];
cx node[13],node[12];
cx node[16],node[14];
cx node[6],node[7];
cx node[14],node[11];
rz(3.99609375*pi) node[12];
cx node[19],node[16];
cx node[7],node[6];
cx node[11],node[14];
cx node[12],node[13];
rz(0.0001220703125*pi) node[16];
cx node[6],node[7];
cx node[14],node[11];
cx node[13],node[12];
cx node[19],node[16];
cx node[7],node[4];
cx node[11],node[8];
cx node[12],node[13];
rz(3.9998779296875*pi) node[16];
rz(0.125*pi) node[4];
rz(0.00048828125*pi) node[8];
cx node[15],node[12];
cx node[14],node[13];
cx node[19],node[16];
cx node[7],node[4];
cx node[11],node[8];
cx node[12],node[15];
rz(0.001953125*pi) node[13];
cx node[16],node[19];
rz(3.875*pi) node[4];
rz(3.99951171875*pi) node[8];
cx node[15],node[12];
cx node[14],node[13];
cx node[19],node[16];
cx node[7],node[4];
cx node[12],node[10];
rz(3.998046875*pi) node[13];
cx node[16],node[14];
cx node[20],node[19];
cx node[4],node[7];
rz(0.015625*pi) node[10];
cx node[14],node[16];
rz(6.103515625e-05*pi) node[19];
cx node[7],node[4];
cx node[12],node[10];
cx node[16],node[14];
cx node[20],node[19];
cx node[4],node[1];
cx node[6],node[7];
rz(3.984375*pi) node[10];
cx node[14],node[11];
rz(3.99993896484375*pi) node[19];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[7];
cx node[12],node[10];
cx node[11],node[14];
cx node[20],node[19];
cx node[4],node[1];
cx node[6],node[7];
cx node[10],node[12];
cx node[14],node[11];
cx node[19],node[20];
rz(3.75*pi) node[1];
rz(0.5*pi) node[4];
rz(3.9375*pi) node[7];
cx node[11],node[8];
cx node[12],node[10];
cx node[14],node[13];
cx node[20],node[19];
sx node[4];
cx node[10],node[7];
rz(0.000244140625*pi) node[8];
cx node[15],node[12];
rz(0.0009765625*pi) node[13];
cx node[19],node[16];
rz(3.5*pi) node[4];
rz(0.03125*pi) node[7];
cx node[11],node[8];
rz(0.0078125*pi) node[12];
cx node[14],node[13];
cx node[16],node[19];
sx node[4];
cx node[10],node[7];
rz(3.999755859375*pi) node[8];
cx node[15],node[12];
rz(3.9990234375*pi) node[13];
cx node[19],node[16];
rz(1.0*pi) node[4];
rz(3.96875*pi) node[7];
rz(3.9921875*pi) node[12];
cx node[16],node[14];
cx node[22],node[19];
cx node[1],node[4];
cx node[7],node[10];
cx node[14],node[16];
cx node[19],node[22];
cx node[4],node[1];
cx node[10],node[7];
cx node[16],node[14];
cx node[22],node[19];
cx node[1],node[4];
cx node[7],node[10];
cx node[14],node[11];
cx node[19],node[20];
cx node[6],node[7];
cx node[11],node[14];
rz(3.0517578125e-05*pi) node[20];
cx node[7],node[6];
cx node[14],node[11];
cx node[19],node[20];
cx node[6],node[7];
cx node[11],node[8];
cx node[14],node[13];
cx node[19],node[16];
rz(3.999969482421875*pi) node[20];
cx node[7],node[4];
rz(0.0001220703125*pi) node[8];
rz(0.00048828125*pi) node[13];
cx node[16],node[19];
rz(0.125*pi) node[4];
cx node[11],node[8];
cx node[14],node[13];
cx node[19],node[16];
cx node[7],node[4];
rz(3.9998779296875*pi) node[8];
rz(3.99951171875*pi) node[13];
cx node[16],node[14];
cx node[22],node[19];
rz(3.875*pi) node[4];
cx node[14],node[16];
cx node[19],node[22];
cx node[7],node[4];
cx node[16],node[14];
cx node[22],node[19];
cx node[4],node[7];
cx node[14],node[11];
cx node[19],node[16];
cx node[7],node[4];
cx node[11],node[14];
cx node[16],node[19];
cx node[4],node[1];
cx node[6],node[7];
cx node[14],node[11];
cx node[19],node[16];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[7];
cx node[11],node[8];
cx node[14],node[13];
cx node[22],node[19];
cx node[4],node[1];
cx node[6],node[7];
rz(6.103515625e-05*pi) node[8];
rz(0.000244140625*pi) node[13];
cx node[19],node[22];
rz(3.75*pi) node[1];
rz(0.5*pi) node[4];
rz(3.9375*pi) node[7];
cx node[11],node[8];
cx node[14],node[13];
cx node[22],node[19];
sx node[4];
rz(3.99993896484375*pi) node[8];
rz(3.999755859375*pi) node[13];
cx node[16],node[14];
rz(3.5*pi) node[4];
cx node[14],node[16];
sx node[4];
cx node[16],node[14];
rz(1.0*pi) node[4];
cx node[13],node[14];
cx node[1],node[4];
cx node[14],node[13];
cx node[4],node[1];
cx node[13],node[14];
cx node[1],node[4];
cx node[11],node[14];
cx node[13],node[12];
rz(0.00390625*pi) node[12];
rz(0.0001220703125*pi) node[14];
cx node[11],node[14];
cx node[13],node[12];
rz(3.99609375*pi) node[12];
rz(3.9998779296875*pi) node[14];
cx node[12],node[13];
cx node[13],node[12];
cx node[12],node[13];
cx node[15],node[12];
cx node[13],node[14];
cx node[12],node[15];
cx node[14],node[13];
cx node[15],node[12];
cx node[13],node[14];
cx node[12],node[10];
cx node[14],node[16];
rz(0.015625*pi) node[10];
cx node[16],node[14];
cx node[12],node[10];
cx node[14],node[16];
rz(3.984375*pi) node[10];
cx node[19],node[16];
cx node[12],node[10];
rz(0.001953125*pi) node[16];
cx node[10],node[12];
cx node[19],node[16];
cx node[12],node[10];
rz(3.998046875*pi) node[16];
cx node[10],node[7];
cx node[15],node[12];
cx node[19],node[16];
rz(0.03125*pi) node[7];
rz(0.0078125*pi) node[12];
cx node[16],node[19];
cx node[10],node[7];
cx node[15],node[12];
cx node[19],node[16];
rz(3.96875*pi) node[7];
rz(3.9921875*pi) node[12];
cx node[16],node[14];
cx node[22],node[19];
cx node[7],node[10];
cx node[12],node[13];
cx node[14],node[16];
rz(0.0009765625*pi) node[19];
cx node[10],node[7];
cx node[13],node[12];
cx node[16],node[14];
cx node[22],node[19];
cx node[7],node[10];
cx node[12],node[13];
rz(3.9990234375*pi) node[19];
cx node[6],node[7];
cx node[15],node[12];
cx node[14],node[13];
cx node[16],node[19];
cx node[7],node[6];
cx node[12],node[15];
rz(0.00390625*pi) node[13];
rz(0.00048828125*pi) node[19];
cx node[6],node[7];
cx node[15],node[12];
cx node[14],node[13];
cx node[16],node[19];
cx node[7],node[4];
cx node[12],node[10];
rz(3.99609375*pi) node[13];
rz(3.99951171875*pi) node[19];
rz(0.125*pi) node[4];
rz(0.015625*pi) node[10];
cx node[13],node[14];
cx node[19],node[16];
cx node[7],node[4];
cx node[12],node[10];
cx node[14],node[13];
cx node[16],node[19];
rz(3.875*pi) node[4];
rz(3.984375*pi) node[10];
cx node[13],node[14];
cx node[19],node[16];
cx node[7],node[4];
cx node[12],node[10];
cx node[16],node[14];
cx node[22],node[19];
cx node[4],node[7];
cx node[10],node[12];
cx node[14],node[16];
cx node[19],node[22];
cx node[7],node[4];
cx node[12],node[10];
cx node[16],node[14];
cx node[22],node[19];
cx node[4],node[1];
cx node[6],node[7];
cx node[11],node[14];
cx node[13],node[12];
cx node[19],node[16];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[7];
rz(0.0078125*pi) node[12];
rz(0.000244140625*pi) node[14];
rz(0.001953125*pi) node[16];
cx node[4],node[1];
cx node[6],node[7];
cx node[11],node[14];
cx node[13],node[12];
cx node[19],node[16];
rz(3.75*pi) node[1];
rz(0.5*pi) node[4];
rz(3.9375*pi) node[7];
rz(3.9921875*pi) node[12];
rz(3.999755859375*pi) node[14];
rz(3.998046875*pi) node[16];
sx node[4];
cx node[10],node[7];
cx node[11],node[14];
cx node[13],node[12];
cx node[19],node[16];
rz(3.5*pi) node[4];
rz(0.03125*pi) node[7];
cx node[14],node[11];
cx node[12],node[13];
cx node[16],node[19];
sx node[4];
cx node[10],node[7];
cx node[11],node[14];
cx node[13],node[12];
cx node[19],node[16];
rz(1.0*pi) node[4];
rz(3.96875*pi) node[7];
cx node[16],node[14];
cx node[22],node[19];
cx node[1],node[4];
cx node[7],node[10];
cx node[14],node[16];
rz(0.0009765625*pi) node[19];
cx node[4],node[1];
cx node[10],node[7];
cx node[16],node[14];
cx node[22],node[19];
cx node[1],node[4];
cx node[7],node[10];
cx node[14],node[13];
rz(3.9990234375*pi) node[19];
cx node[6],node[7];
cx node[12],node[10];
rz(0.00390625*pi) node[13];
cx node[16],node[19];
cx node[7],node[6];
rz(0.015625*pi) node[10];
cx node[14],node[13];
rz(0.00048828125*pi) node[19];
cx node[6],node[7];
cx node[12],node[10];
rz(3.99609375*pi) node[13];
cx node[16],node[19];
cx node[7],node[4];
rz(3.984375*pi) node[10];
cx node[13],node[14];
rz(3.99951171875*pi) node[19];
rz(0.125*pi) node[4];
cx node[10],node[12];
cx node[14],node[13];
cx node[22],node[19];
cx node[7],node[4];
cx node[12],node[10];
cx node[13],node[14];
cx node[19],node[22];
rz(3.875*pi) node[4];
cx node[10],node[12];
cx node[22],node[19];
cx node[7],node[4];
cx node[13],node[12];
cx node[19],node[16];
cx node[4],node[7];
rz(0.0078125*pi) node[12];
cx node[16],node[19];
cx node[7],node[4];
cx node[13],node[12];
cx node[19],node[16];
cx node[4],node[1];
cx node[6],node[7];
rz(3.9921875*pi) node[12];
cx node[16],node[14];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[7];
cx node[12],node[13];
rz(0.001953125*pi) node[14];
cx node[4],node[1];
cx node[6],node[7];
cx node[13],node[12];
cx node[16],node[14];
rz(3.75*pi) node[1];
rz(0.5*pi) node[4];
rz(3.9375*pi) node[7];
cx node[12],node[13];
rz(3.998046875*pi) node[14];
sx node[4];
cx node[10],node[7];
cx node[16],node[14];
rz(3.5*pi) node[4];
rz(0.03125*pi) node[7];
cx node[14],node[16];
sx node[4];
cx node[10],node[7];
cx node[16],node[14];
rz(1.0*pi) node[4];
rz(3.96875*pi) node[7];
cx node[14],node[13];
cx node[19],node[16];
cx node[1],node[4];
cx node[7],node[10];
rz(0.00390625*pi) node[13];
rz(0.0009765625*pi) node[16];
cx node[4],node[1];
cx node[10],node[7];
cx node[14],node[13];
cx node[19],node[16];
cx node[1],node[4];
cx node[7],node[10];
rz(3.99609375*pi) node[13];
rz(3.9990234375*pi) node[16];
cx node[6],node[7];
cx node[12],node[10];
cx node[13],node[14];
cx node[19],node[16];
cx node[7],node[6];
rz(0.015625*pi) node[10];
cx node[14],node[13];
cx node[16],node[19];
cx node[6],node[7];
cx node[12],node[10];
cx node[13],node[14];
cx node[19],node[16];
cx node[7],node[4];
rz(3.984375*pi) node[10];
cx node[16],node[14];
rz(0.125*pi) node[4];
cx node[10],node[12];
rz(0.001953125*pi) node[14];
cx node[7],node[4];
cx node[12],node[10];
cx node[16],node[14];
rz(3.875*pi) node[4];
cx node[10],node[12];
rz(3.998046875*pi) node[14];
cx node[7],node[4];
cx node[13],node[12];
cx node[16],node[14];
cx node[4],node[7];
rz(0.0078125*pi) node[12];
cx node[14],node[16];
cx node[7],node[4];
cx node[13],node[12];
cx node[16],node[14];
cx node[4],node[1];
cx node[6],node[7];
rz(3.9921875*pi) node[12];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[7];
cx node[12],node[13];
cx node[4],node[1];
cx node[6],node[7];
cx node[13],node[12];
rz(3.75*pi) node[1];
rz(0.5*pi) node[4];
rz(3.9375*pi) node[7];
cx node[12],node[13];
sx node[4];
cx node[10],node[7];
cx node[14],node[13];
rz(3.5*pi) node[4];
rz(0.03125*pi) node[7];
rz(0.00390625*pi) node[13];
sx node[4];
cx node[10],node[7];
cx node[14],node[13];
rz(1.0*pi) node[4];
rz(3.96875*pi) node[7];
rz(3.99609375*pi) node[13];
cx node[1],node[4];
cx node[7],node[10];
cx node[14],node[13];
cx node[4],node[1];
cx node[10],node[7];
cx node[13],node[14];
cx node[1],node[4];
cx node[7],node[10];
cx node[14],node[13];
cx node[6],node[7];
cx node[12],node[10];
cx node[7],node[6];
rz(0.015625*pi) node[10];
cx node[6],node[7];
cx node[12],node[10];
cx node[7],node[4];
rz(3.984375*pi) node[10];
rz(0.125*pi) node[4];
cx node[10],node[12];
cx node[7],node[4];
cx node[12],node[10];
rz(3.875*pi) node[4];
cx node[10],node[12];
cx node[7],node[4];
cx node[13],node[12];
cx node[4],node[7];
rz(0.0078125*pi) node[12];
cx node[7],node[4];
cx node[13],node[12];
cx node[4],node[1];
cx node[6],node[7];
rz(3.9921875*pi) node[12];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[7];
cx node[13],node[12];
cx node[4],node[1];
cx node[6],node[7];
cx node[12],node[13];
rz(3.75*pi) node[1];
rz(0.5*pi) node[4];
rz(3.9375*pi) node[7];
cx node[13],node[12];
sx node[4];
cx node[10],node[7];
rz(3.5*pi) node[4];
rz(0.03125*pi) node[7];
sx node[4];
cx node[10],node[7];
rz(1.0*pi) node[4];
rz(3.96875*pi) node[7];
cx node[1],node[4];
cx node[7],node[10];
cx node[4],node[1];
cx node[10],node[7];
cx node[1],node[4];
cx node[7],node[10];
cx node[6],node[7];
cx node[12],node[10];
cx node[7],node[6];
rz(0.015625*pi) node[10];
cx node[6],node[7];
cx node[12],node[10];
cx node[7],node[4];
rz(3.984375*pi) node[10];
rz(0.125*pi) node[4];
cx node[12],node[10];
cx node[7],node[4];
cx node[10],node[12];
rz(3.875*pi) node[4];
cx node[12],node[10];
cx node[7],node[4];
cx node[4],node[7];
cx node[7],node[4];
cx node[4],node[1];
cx node[6],node[7];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[7];
cx node[4],node[1];
cx node[6],node[7];
rz(3.75*pi) node[1];
rz(0.5*pi) node[4];
rz(3.9375*pi) node[7];
sx node[4];
cx node[10],node[7];
rz(3.5*pi) node[4];
rz(0.03125*pi) node[7];
sx node[4];
cx node[10],node[7];
rz(1.0*pi) node[4];
rz(3.96875*pi) node[7];
cx node[1],node[4];
cx node[6],node[7];
cx node[4],node[1];
cx node[7],node[6];
cx node[1],node[4];
cx node[6],node[7];
cx node[7],node[4];
rz(0.125*pi) node[4];
cx node[7],node[4];
rz(3.875*pi) node[4];
cx node[7],node[4];
cx node[4],node[7];
cx node[7],node[4];
cx node[4],node[1];
cx node[10],node[7];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[7];
cx node[4],node[1];
cx node[10],node[7];
rz(3.75*pi) node[1];
rz(0.5*pi) node[4];
rz(3.9375*pi) node[7];
sx node[4];
cx node[10],node[7];
rz(3.5*pi) node[4];
cx node[7],node[10];
sx node[4];
cx node[10],node[7];
rz(1.0*pi) node[4];
cx node[7],node[4];
cx node[4],node[7];
cx node[7],node[4];
cx node[4],node[1];
rz(0.125*pi) node[1];
cx node[4],node[1];
rz(3.875*pi) node[1];
cx node[4],node[7];
rz(0.25*pi) node[7];
cx node[4],node[7];
rz(0.5*pi) node[4];
rz(3.75*pi) node[7];
sx node[4];
rz(3.5*pi) node[4];
sx node[4];
rz(1.0*pi) node[4];
barrier node[24],node[26],node[25],node[20],node[8],node[15],node[11],node[22],node[19],node[16],node[14],node[13],node[12],node[6],node[10],node[1],node[7],node[4],node[0];
measure node[24] -> c[0];
measure node[26] -> c[1];
measure node[25] -> c[2];
measure node[20] -> c[3];
measure node[8] -> c[4];
measure node[15] -> c[5];
measure node[11] -> c[6];
measure node[22] -> c[7];
measure node[19] -> c[8];
measure node[16] -> c[9];
measure node[14] -> c[10];
measure node[13] -> c[11];
measure node[12] -> c[12];
measure node[6] -> c[13];
measure node[10] -> c[14];
measure node[1] -> c[15];
measure node[7] -> c[16];
measure node[4] -> c[17];
