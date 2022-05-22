OPENQASM 2.0;
include "qelib1.inc";

qreg node[34];
creg meas[16];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
rz(0.5*pi) node[4];
sx node[5];
sx node[6];
sx node[14];
sx node[15];
sx node[18];
sx node[19];
sx node[20];
sx node[21];
sx node[22];
sx node[23];
sx node[33];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
sx node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[33];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
rz(3.5*pi) node[4];
sx node[5];
sx node[6];
sx node[14];
sx node[15];
sx node[18];
sx node[19];
sx node[20];
sx node[21];
sx node[22];
sx node[23];
sx node[33];
rz(0.0009765624999995559*pi) node[0];
rz(0.0004882812499995559*pi) node[1];
rz(0.0002441406249995559*pi) node[2];
rz(3.051757812455591e-05*pi) node[3];
sx node[4];
rz(6.103515624955591e-05*pi) node[5];
rz(0.062499999999999556*pi) node[6];
rz(0.001953124999999556*pi) node[14];
rz(0.00012207031249955591*pi) node[15];
rz(0.003906249999999556*pi) node[18];
rz(0.007812499999999556*pi) node[19];
rz(0.015624999999999556*pi) node[20];
rz(0.12499999999999956*pi) node[21];
rz(0.031249999999999556*pi) node[22];
rz(0.5*pi) node[23];
rz(0.25*pi) node[33];
rz(0.7951672359369731*pi) node[4];
cx node[3],node[4];
rz(3.7048327640630268*pi) node[4];
cx node[3],node[4];
rz(0.29516723593697314*pi) node[4];
cx node[5],node[4];
rz(3.409665540858449*pi) node[4];
cx node[5],node[4];
rz(0.5903344591415509*pi) node[4];
cx node[6],node[5];
cx node[15],node[4];
cx node[5],node[6];
rz(2.3193310498859097*pi) node[4];
cx node[6],node[5];
sx node[4];
rz(3.5*pi) node[4];
sx node[4];
rz(1.5*pi) node[4];
cx node[15],node[4];
rz(3.5*pi) node[4];
cx node[22],node[15];
sx node[4];
cx node[15],node[22];
rz(0.5*pi) node[4];
cx node[22],node[15];
sx node[4];
cx node[21],node[22];
rz(0.6806689523994236*pi) node[4];
cx node[22],node[21];
cx node[4],node[3];
cx node[21],node[22];
cx node[3],node[4];
cx node[4],node[3];
cx node[2],node[3];
cx node[15],node[4];
rz(1.6386621316028078*pi) node[3];
cx node[4],node[15];
cx node[2],node[3];
cx node[15],node[4];
rz(0.3613378706825252*pi) node[3];
cx node[22],node[15];
cx node[3],node[2];
cx node[15],node[22];
cx node[2],node[3];
cx node[22],node[15];
cx node[3],node[2];
cx node[23],node[22];
cx node[1],node[2];
cx node[4],node[3];
cx node[22],node[23];
rz(3.2773243905295706*pi) node[2];
cx node[3],node[4];
cx node[23],node[22];
cx node[1],node[2];
cx node[4],node[3];
rz(0.7226757731960389*pi) node[2];
cx node[5],node[4];
cx node[2],node[1];
cx node[4],node[5];
cx node[1],node[2];
cx node[5],node[4];
cx node[2],node[1];
cx node[0],node[1];
cx node[3],node[2];
rz(2.0546484627492543*pi) node[1];
cx node[2],node[3];
sx node[1];
cx node[3],node[2];
rz(3.5*pi) node[1];
cx node[4],node[3];
sx node[1];
cx node[3],node[4];
rz(1.5*pi) node[1];
cx node[4],node[3];
cx node[0],node[1];
cx node[15],node[4];
cx node[14],node[0];
rz(3.5*pi) node[1];
cx node[4],node[15];
cx node[0],node[14];
sx node[1];
cx node[15],node[4];
cx node[14],node[0];
rz(0.5*pi) node[1];
cx node[22],node[15];
sx node[1];
cx node[18],node[14];
cx node[15],node[22];
rz(0.9453515168464228*pi) node[1];
cx node[14],node[18];
cx node[22],node[15];
cx node[0],node[1];
cx node[18],node[14];
rz(1.1092969254985086*pi) node[1];
cx node[19],node[18];
cx node[0],node[1];
cx node[18],node[19];
cx node[14],node[0];
rz(0.8907030632385013*pi) node[1];
cx node[19],node[18];
cx node[0],node[14];
cx node[20],node[19];
cx node[14],node[0];
cx node[19],node[20];
cx node[0],node[1];
cx node[18],node[14];
cx node[20],node[19];
rz(1.7185932143772433*pi) node[1];
cx node[14],node[18];
cx node[33],node[20];
sx node[1];
cx node[18],node[14];
cx node[20],node[33];
rz(3.5*pi) node[1];
cx node[19],node[18];
cx node[33],node[20];
sx node[1];
cx node[18],node[19];
rz(1.5*pi) node[1];
cx node[19],node[18];
cx node[0],node[1];
cx node[20],node[19];
cx node[14],node[0];
rz(3.5*pi) node[1];
cx node[19],node[20];
cx node[0],node[14];
sx node[1];
cx node[20],node[19];
cx node[14],node[0];
rz(0.5*pi) node[1];
sx node[1];
cx node[18],node[14];
rz(1.2814061192130388*pi) node[1];
cx node[14],node[18];
cx node[0],node[1];
cx node[18],node[14];
rz(3.9371864287544867*pi) node[1];
cx node[19],node[18];
sx node[1];
cx node[18],node[19];
rz(3.5*pi) node[1];
cx node[19],node[18];
sx node[1];
rz(1.5*pi) node[1];
cx node[0],node[1];
cx node[14],node[0];
rz(3.5*pi) node[1];
cx node[0],node[14];
sx node[1];
cx node[14],node[0];
rz(0.5*pi) node[1];
sx node[1];
cx node[18],node[14];
rz(1.0628122256936823*pi) node[1];
cx node[14],node[18];
cx node[0],node[1];
cx node[18],node[14];
rz(0.37437604060784224*pi) node[1];
sx node[1];
rz(3.5*pi) node[1];
sx node[1];
rz(1.5*pi) node[1];
cx node[0],node[1];
cx node[14],node[0];
rz(3.5*pi) node[1];
cx node[0],node[14];
sx node[1];
cx node[14],node[0];
rz(0.5*pi) node[1];
sx node[1];
rz(0.6256244832183533*pi) node[1];
cx node[2],node[1];
rz(1.7487520812156845*pi) node[1];
cx node[2],node[1];
rz(0.25124894823587907*pi) node[1];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[1];
cx node[4],node[3];
rz(3.497504162431369*pi) node[1];
cx node[3],node[4];
cx node[2],node[1];
cx node[4],node[3];
rz(0.5024978964717581*pi) node[1];
cx node[3],node[2];
cx node[15],node[4];
cx node[2],node[3];
cx node[4],node[15];
cx node[3],node[2];
cx node[15],node[4];
cx node[2],node[1];
cx node[4],node[3];
rz(2.495008324862738*pi) node[1];
cx node[3],node[4];
sx node[1];
cx node[4],node[3];
rz(3.5*pi) node[1];
sx node[1];
rz(1.5*pi) node[1];
cx node[2],node[1];
rz(3.5*pi) node[1];
cx node[3],node[2];
sx node[1];
cx node[2],node[3];
rz(0.5*pi) node[1];
cx node[3],node[2];
sx node[1];
rz(0.5049957952288497*pi) node[1];
cx node[0],node[1];
rz(1.9900166497254759*pi) node[1];
cx node[0],node[1];
rz(0.009991584670957399*pi) node[1];
cx node[2],node[1];
rz(3.979969637473914*pi) node[1];
cx node[2],node[1];
rz(3.520030362526086*pi) node[1];
rz(0.5*pi) node[2];
sx node[1];
sx node[2];
rz(3.5*pi) node[1];
rz(3.5*pi) node[2];
sx node[1];
sx node[2];
rz(1.5*pi) node[1];
rz(1.0*pi) node[2];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[14],node[0];
cx node[1],node[2];
cx node[0],node[14];
rz(0.25*pi) node[2];
cx node[14],node[0];
cx node[1],node[2];
rz(0.5*pi) node[1];
rz(3.75*pi) node[2];
sx node[1];
cx node[3],node[2];
rz(3.5*pi) node[1];
rz(0.125*pi) node[2];
sx node[1];
cx node[3],node[2];
rz(1.0*pi) node[1];
rz(3.875*pi) node[2];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[1];
cx node[4],node[3];
rz(0.25*pi) node[1];
rz(0.0625*pi) node[3];
cx node[2],node[1];
cx node[4],node[3];
rz(3.75*pi) node[1];
rz(0.5*pi) node[2];
rz(3.9375*pi) node[3];
sx node[2];
cx node[4],node[3];
rz(3.5*pi) node[2];
cx node[3],node[4];
sx node[2];
cx node[4],node[3];
rz(1.0*pi) node[2];
cx node[15],node[4];
rz(0.03125*pi) node[4];
cx node[15],node[4];
rz(3.96875*pi) node[4];
cx node[4],node[3];
cx node[3],node[4];
cx node[4],node[3];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[0],node[1];
cx node[2],node[3];
rz(0.015625*pi) node[1];
cx node[3],node[2];
cx node[0],node[1];
cx node[2],node[3];
rz(3.984375*pi) node[1];
cx node[4],node[3];
cx node[1],node[0];
rz(0.125*pi) node[3];
cx node[0],node[1];
cx node[4],node[3];
cx node[1],node[0];
rz(3.875*pi) node[3];
cx node[0],node[14];
cx node[4],node[3];
cx node[14],node[0];
cx node[3],node[4];
cx node[0],node[14];
cx node[4],node[3];
cx node[3],node[2];
cx node[18],node[14];
rz(0.25*pi) node[2];
rz(0.0078125*pi) node[14];
cx node[3],node[2];
cx node[18],node[14];
rz(3.75*pi) node[2];
rz(0.5*pi) node[3];
rz(3.9921875*pi) node[14];
sx node[3];
cx node[14],node[18];
rz(3.5*pi) node[3];
cx node[18],node[14];
sx node[3];
cx node[14],node[18];
rz(1.0*pi) node[3];
cx node[19],node[18];
rz(0.00390625*pi) node[18];
cx node[19],node[18];
rz(3.99609375*pi) node[18];
cx node[18],node[19];
cx node[19],node[18];
cx node[18],node[19];
cx node[20],node[19];
rz(0.001953125*pi) node[19];
cx node[20],node[19];
rz(3.998046875*pi) node[19];
cx node[19],node[20];
cx node[20],node[19];
cx node[19],node[20];
cx node[33],node[20];
rz(0.0009765625*pi) node[20];
cx node[33],node[20];
rz(3.9990234375*pi) node[20];
cx node[20],node[21];
cx node[21],node[20];
cx node[20],node[21];
cx node[22],node[21];
rz(0.00048828125*pi) node[21];
cx node[22],node[21];
rz(3.99951171875*pi) node[21];
cx node[21],node[22];
cx node[22],node[21];
cx node[21],node[22];
cx node[22],node[15];
cx node[20],node[21];
cx node[15],node[22];
cx node[21],node[20];
cx node[22],node[15];
cx node[20],node[21];
cx node[4],node[15];
cx node[33],node[20];
cx node[15],node[4];
cx node[20],node[33];
cx node[4],node[15];
cx node[33],node[20];
cx node[5],node[4];
cx node[22],node[15];
rz(0.000244140625*pi) node[4];
rz(0.0625*pi) node[15];
cx node[5],node[4];
cx node[22],node[15];
rz(3.999755859375*pi) node[4];
cx node[6],node[5];
rz(3.9375*pi) node[15];
cx node[15],node[4];
cx node[5],node[6];
cx node[4],node[15];
cx node[6],node[5];
cx node[15],node[4];
cx node[4],node[3];
cx node[22],node[15];
cx node[3],node[4];
cx node[15],node[22];
cx node[4],node[3];
cx node[22],node[15];
cx node[3],node[2];
cx node[5],node[4];
cx node[21],node[22];
cx node[2],node[3];
cx node[4],node[5];
rz(0.0001220703125*pi) node[22];
cx node[3],node[2];
cx node[5],node[4];
cx node[21],node[22];
cx node[1],node[2];
cx node[15],node[4];
rz(3.9998779296875*pi) node[22];
rz(0.03125*pi) node[2];
cx node[4],node[15];
cx node[1],node[2];
cx node[15],node[4];
rz(3.96875*pi) node[2];
cx node[4],node[3];
cx node[15],node[22];
cx node[2],node[1];
rz(0.125*pi) node[3];
rz(6.103515625e-05*pi) node[22];
cx node[1],node[2];
cx node[4],node[3];
cx node[15],node[22];
cx node[2],node[1];
rz(3.875*pi) node[3];
cx node[4],node[5];
rz(3.99993896484375*pi) node[22];
cx node[1],node[0];
cx node[2],node[3];
rz(0.25*pi) node[5];
cx node[23],node[22];
cx node[0],node[1];
rz(0.0625*pi) node[3];
cx node[4],node[5];
rz(3.0517578125e-05*pi) node[22];
cx node[1],node[0];
cx node[2],node[3];
rz(0.5*pi) node[4];
rz(3.75*pi) node[5];
cx node[23],node[22];
cx node[14],node[0];
rz(3.9375*pi) node[3];
sx node[4];
rz(3.999969482421875*pi) node[22];
rz(0.015625*pi) node[0];
cx node[3],node[2];
rz(3.5*pi) node[4];
cx node[14],node[0];
cx node[2],node[3];
sx node[4];
rz(3.984375*pi) node[0];
cx node[3],node[2];
rz(1.0*pi) node[4];
cx node[0],node[14];
cx node[14],node[0];
cx node[0],node[14];
cx node[0],node[1];
cx node[18],node[14];
cx node[1],node[0];
rz(0.0078125*pi) node[14];
cx node[0],node[1];
cx node[18],node[14];
cx node[1],node[2];
rz(3.9921875*pi) node[14];
rz(0.03125*pi) node[2];
cx node[14],node[18];
cx node[1],node[2];
cx node[18],node[14];
rz(3.96875*pi) node[2];
cx node[14],node[18];
cx node[2],node[1];
cx node[19],node[18];
cx node[1],node[2];
rz(0.00390625*pi) node[18];
cx node[2],node[1];
cx node[19],node[18];
cx node[1],node[0];
rz(3.99609375*pi) node[18];
cx node[0],node[1];
cx node[18],node[19];
cx node[1],node[0];
cx node[19],node[18];
cx node[14],node[0];
cx node[18],node[19];
rz(0.015625*pi) node[0];
cx node[20],node[19];
cx node[14],node[0];
rz(0.001953125*pi) node[19];
rz(3.984375*pi) node[0];
cx node[20],node[19];
cx node[0],node[14];
rz(3.998046875*pi) node[19];
cx node[14],node[0];
cx node[20],node[19];
cx node[0],node[14];
cx node[19],node[20];
cx node[0],node[1];
cx node[18],node[14];
cx node[20],node[19];
cx node[1],node[0];
rz(0.0078125*pi) node[14];
cx node[33],node[20];
cx node[0],node[1];
cx node[18],node[14];
rz(0.0009765625*pi) node[20];
rz(3.9921875*pi) node[14];
cx node[33],node[20];
cx node[14],node[18];
rz(3.9990234375*pi) node[20];
cx node[18],node[14];
cx node[20],node[21];
cx node[14],node[18];
cx node[21],node[20];
cx node[14],node[0];
cx node[19],node[18];
cx node[20],node[21];
cx node[0],node[14];
rz(0.00390625*pi) node[18];
cx node[21],node[22];
cx node[14],node[0];
cx node[19],node[18];
cx node[22],node[21];
rz(3.99609375*pi) node[18];
cx node[21],node[22];
cx node[22],node[15];
cx node[18],node[19];
cx node[20],node[21];
cx node[15],node[22];
cx node[19],node[18];
cx node[21],node[20];
cx node[22],node[15];
cx node[18],node[19];
cx node[20],node[21];
cx node[15],node[4];
cx node[19],node[20];
cx node[21],node[22];
cx node[4],node[15];
cx node[20],node[19];
cx node[22],node[21];
cx node[15],node[4];
cx node[19],node[20];
cx node[21],node[22];
cx node[5],node[4];
cx node[33],node[20];
cx node[4],node[5];
rz(0.001953125*pi) node[20];
cx node[5],node[4];
cx node[33],node[20];
cx node[3],node[4];
cx node[6],node[5];
rz(3.998046875*pi) node[20];
rz(0.125*pi) node[4];
rz(0.00048828125*pi) node[5];
cx node[20],node[21];
cx node[3],node[4];
cx node[6],node[5];
cx node[21],node[20];
rz(3.875*pi) node[4];
rz(3.99951171875*pi) node[5];
cx node[20],node[21];
cx node[4],node[3];
cx node[3],node[4];
cx node[4],node[3];
cx node[2],node[3];
cx node[4],node[15];
rz(0.0625*pi) node[3];
rz(0.25*pi) node[15];
cx node[2],node[3];
cx node[4],node[15];
rz(3.9375*pi) node[3];
rz(0.5*pi) node[4];
rz(3.75*pi) node[15];
cx node[3],node[2];
sx node[4];
cx node[2],node[3];
rz(3.5*pi) node[4];
cx node[3],node[2];
sx node[4];
cx node[1],node[2];
rz(1.0*pi) node[4];
rz(0.03125*pi) node[2];
cx node[5],node[4];
cx node[1],node[2];
cx node[4],node[5];
rz(3.96875*pi) node[2];
cx node[5],node[4];
cx node[1],node[2];
cx node[15],node[4];
cx node[2],node[1];
cx node[4],node[15];
cx node[1],node[2];
cx node[15],node[4];
cx node[0],node[1];
cx node[3],node[4];
cx node[22],node[15];
rz(0.015625*pi) node[1];
rz(0.125*pi) node[4];
rz(0.000244140625*pi) node[15];
cx node[0],node[1];
cx node[3],node[4];
cx node[22],node[15];
rz(3.984375*pi) node[1];
rz(3.875*pi) node[4];
rz(3.999755859375*pi) node[15];
cx node[21],node[22];
cx node[1],node[0];
cx node[22],node[21];
cx node[0],node[1];
cx node[21],node[22];
cx node[1],node[0];
cx node[22],node[15];
cx node[0],node[14];
cx node[15],node[22];
cx node[14],node[0];
cx node[22],node[15];
cx node[0],node[14];
cx node[15],node[4];
cx node[21],node[22];
cx node[4],node[15];
cx node[18],node[14];
cx node[22],node[21];
cx node[15],node[4];
rz(0.0078125*pi) node[14];
cx node[21],node[22];
cx node[5],node[4];
cx node[18],node[14];
cx node[20],node[21];
cx node[4],node[5];
rz(3.9921875*pi) node[14];
rz(0.0001220703125*pi) node[21];
cx node[5],node[4];
cx node[18],node[14];
cx node[20],node[21];
cx node[3],node[4];
cx node[6],node[5];
cx node[14],node[18];
rz(3.9998779296875*pi) node[21];
rz(0.25*pi) node[4];
rz(0.0009765625*pi) node[5];
cx node[18],node[14];
cx node[14],node[0];
cx node[3],node[4];
cx node[6],node[5];
cx node[0],node[14];
rz(0.5*pi) node[3];
rz(3.75*pi) node[4];
rz(3.9990234375*pi) node[5];
cx node[14],node[0];
sx node[3];
cx node[5],node[4];
rz(3.5*pi) node[3];
cx node[4],node[5];
sx node[3];
cx node[5],node[4];
rz(1.0*pi) node[3];
cx node[4],node[15];
cx node[15],node[4];
cx node[4],node[15];
cx node[4],node[3];
cx node[22],node[15];
cx node[3],node[4];
rz(0.00048828125*pi) node[15];
cx node[4],node[3];
cx node[22],node[15];
cx node[2],node[3];
rz(3.99951171875*pi) node[15];
rz(0.0625*pi) node[3];
cx node[15],node[22];
cx node[2],node[3];
cx node[22],node[15];
rz(3.9375*pi) node[3];
cx node[15],node[22];
cx node[3],node[2];
cx node[22],node[21];
cx node[2],node[3];
cx node[21],node[22];
cx node[3],node[2];
cx node[22],node[21];
cx node[1],node[2];
cx node[20],node[21];
cx node[23],node[22];
rz(0.03125*pi) node[2];
rz(0.000244140625*pi) node[21];
rz(6.103515625e-05*pi) node[22];
cx node[1],node[2];
cx node[20],node[21];
cx node[23],node[22];
rz(3.96875*pi) node[2];
cx node[33],node[20];
rz(3.999755859375*pi) node[21];
rz(3.99993896484375*pi) node[22];
cx node[2],node[1];
cx node[20],node[33];
cx node[1],node[2];
cx node[33],node[20];
cx node[2],node[1];
cx node[20],node[19];
cx node[0],node[1];
cx node[19],node[20];
rz(0.015625*pi) node[1];
cx node[20],node[19];
cx node[0],node[1];
cx node[19],node[18];
rz(3.984375*pi) node[1];
rz(0.00390625*pi) node[18];
cx node[0],node[1];
cx node[19],node[18];
cx node[1],node[0];
rz(3.99609375*pi) node[18];
cx node[0],node[1];
cx node[18],node[19];
cx node[19],node[18];
cx node[18],node[19];
cx node[18],node[14];
cx node[19],node[20];
cx node[14],node[18];
cx node[20],node[19];
cx node[18],node[14];
cx node[19],node[20];
cx node[14],node[0];
cx node[20],node[21];
rz(0.0078125*pi) node[0];
cx node[21],node[20];
cx node[14],node[0];
cx node[20],node[21];
rz(3.9921875*pi) node[0];
cx node[21],node[22];
cx node[22],node[21];
cx node[21],node[22];
cx node[22],node[15];
cx node[20],node[21];
cx node[15],node[22];
cx node[21],node[20];
cx node[22],node[15];
cx node[20],node[21];
cx node[15],node[4];
cx node[33],node[20];
cx node[4],node[15];
cx node[20],node[33];
cx node[15],node[4];
cx node[33],node[20];
cx node[5],node[4];
cx node[4],node[5];
cx node[5],node[4];
cx node[3],node[4];
cx node[6],node[5];
rz(0.125*pi) node[4];
rz(0.001953125*pi) node[5];
cx node[3],node[4];
cx node[6],node[5];
rz(3.875*pi) node[4];
rz(3.998046875*pi) node[5];
cx node[4],node[3];
cx node[3],node[4];
cx node[4],node[3];
cx node[2],node[3];
cx node[4],node[15];
rz(0.0625*pi) node[3];
rz(0.25*pi) node[15];
cx node[2],node[3];
cx node[4],node[15];
rz(3.9375*pi) node[3];
rz(0.5*pi) node[4];
rz(3.75*pi) node[15];
cx node[2],node[3];
sx node[4];
cx node[3],node[2];
rz(3.5*pi) node[4];
cx node[2],node[3];
sx node[4];
cx node[1],node[2];
rz(1.0*pi) node[4];
rz(0.03125*pi) node[2];
cx node[5],node[4];
cx node[1],node[2];
cx node[4],node[5];
rz(3.96875*pi) node[2];
cx node[5],node[4];
cx node[1],node[2];
cx node[15],node[4];
cx node[2],node[1];
cx node[4],node[15];
cx node[1],node[2];
cx node[15],node[4];
cx node[0],node[1];
cx node[3],node[4];
cx node[22],node[15];
cx node[1],node[0];
rz(0.125*pi) node[4];
rz(0.0009765625*pi) node[15];
cx node[0],node[1];
cx node[3],node[4];
cx node[22],node[15];
cx node[14],node[0];
rz(3.875*pi) node[4];
rz(3.9990234375*pi) node[15];
rz(0.015625*pi) node[0];
cx node[4],node[3];
cx node[15],node[22];
cx node[14],node[0];
cx node[3],node[4];
cx node[22],node[15];
rz(3.984375*pi) node[0];
cx node[4],node[3];
cx node[15],node[22];
cx node[2],node[3];
cx node[4],node[5];
cx node[21],node[22];
rz(0.0625*pi) node[3];
rz(0.25*pi) node[5];
cx node[22],node[21];
cx node[2],node[3];
cx node[4],node[5];
cx node[21],node[22];
rz(3.9375*pi) node[3];
rz(0.5*pi) node[4];
rz(3.75*pi) node[5];
cx node[20],node[21];
cx node[23],node[22];
cx node[2],node[3];
sx node[4];
rz(0.00048828125*pi) node[21];
rz(0.0001220703125*pi) node[22];
cx node[3],node[2];
rz(3.5*pi) node[4];
cx node[20],node[21];
cx node[23],node[22];
cx node[2],node[3];
sx node[4];
cx node[20],node[19];
rz(3.99951171875*pi) node[21];
rz(3.9998779296875*pi) node[22];
cx node[1],node[2];
rz(1.0*pi) node[4];
cx node[19],node[20];
cx node[23],node[22];
cx node[2],node[1];
cx node[5],node[4];
cx node[20],node[19];
cx node[22],node[23];
cx node[1],node[2];
cx node[4],node[5];
cx node[23],node[22];
cx node[0],node[1];
cx node[2],node[3];
cx node[5],node[4];
cx node[22],node[21];
cx node[1],node[0];
cx node[3],node[2];
rz(0.000244140625*pi) node[21];
cx node[0],node[1];
cx node[2],node[3];
cx node[22],node[21];
cx node[14],node[0];
cx node[4],node[3];
rz(3.999755859375*pi) node[21];
rz(0.03125*pi) node[0];
cx node[3],node[4];
cx node[22],node[21];
cx node[14],node[0];
cx node[4],node[3];
cx node[21],node[22];
rz(3.96875*pi) node[0];
cx node[2],node[3];
cx node[5],node[4];
cx node[22],node[21];
rz(0.125*pi) node[3];
cx node[4],node[5];
cx node[21],node[20];
cx node[2],node[3];
cx node[5],node[4];
cx node[20],node[21];
cx node[1],node[2];
rz(3.875*pi) node[3];
cx node[6],node[5];
cx node[21],node[20];
cx node[2],node[1];
rz(0.00390625*pi) node[5];
cx node[1],node[2];
cx node[6],node[5];
cx node[2],node[3];
rz(3.99609375*pi) node[5];
cx node[3],node[2];
cx node[2],node[3];
cx node[1],node[2];
cx node[4],node[3];
cx node[2],node[1];
cx node[3],node[4];
cx node[1],node[2];
cx node[4],node[3];
cx node[0],node[1];
cx node[2],node[3];
cx node[4],node[5];
cx node[1],node[0];
rz(0.25*pi) node[3];
cx node[5],node[4];
cx node[0],node[1];
cx node[2],node[3];
cx node[4],node[5];
cx node[14],node[0];
rz(0.5*pi) node[2];
rz(3.75*pi) node[3];
cx node[15],node[4];
cx node[6],node[5];
rz(0.0625*pi) node[0];
sx node[2];
rz(0.001953125*pi) node[4];
rz(0.0078125*pi) node[5];
cx node[14],node[0];
rz(3.5*pi) node[2];
cx node[15],node[4];
cx node[6],node[5];
rz(3.9375*pi) node[0];
sx node[2];
rz(3.998046875*pi) node[4];
rz(3.9921875*pi) node[5];
rz(1.0*pi) node[2];
cx node[1],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[3];
cx node[2],node[1];
cx node[3],node[4];
cx node[1],node[2];
cx node[4],node[3];
cx node[2],node[1];
cx node[3],node[4];
cx node[0],node[1];
cx node[4],node[5];
cx node[1],node[0];
cx node[5],node[4];
cx node[0],node[1];
cx node[4],node[5];
cx node[14],node[0];
cx node[1],node[2];
cx node[15],node[4];
cx node[6],node[5];
rz(0.125*pi) node[0];
cx node[2],node[1];
rz(0.00390625*pi) node[4];
rz(0.015625*pi) node[5];
cx node[14],node[0];
cx node[1],node[2];
cx node[15],node[4];
cx node[6],node[5];
rz(3.875*pi) node[0];
cx node[2],node[3];
rz(3.99609375*pi) node[4];
rz(3.984375*pi) node[5];
cx node[0],node[1];
cx node[3],node[2];
cx node[1],node[0];
cx node[2],node[3];
cx node[0],node[1];
cx node[3],node[4];
cx node[14],node[0];
cx node[1],node[2];
cx node[4],node[3];
rz(0.25*pi) node[0];
cx node[2],node[1];
cx node[3],node[4];
cx node[14],node[0];
cx node[1],node[2];
cx node[4],node[5];
rz(3.75*pi) node[0];
cx node[2],node[3];
cx node[5],node[4];
rz(0.5*pi) node[14];
cx node[0],node[1];
cx node[3],node[2];
cx node[4],node[5];
sx node[14];
cx node[1],node[0];
cx node[2],node[3];
cx node[15],node[4];
cx node[6],node[5];
rz(3.5*pi) node[14];
cx node[0],node[1];
rz(0.0078125*pi) node[4];
rz(0.03125*pi) node[5];
sx node[14];
cx node[1],node[2];
cx node[15],node[4];
cx node[6],node[5];
rz(1.0*pi) node[14];
cx node[14],node[0];
cx node[2],node[1];
rz(3.9921875*pi) node[4];
rz(3.96875*pi) node[5];
cx node[0],node[14];
cx node[1],node[2];
cx node[3],node[4];
cx node[14],node[0];
cx node[4],node[3];
cx node[0],node[1];
cx node[3],node[4];
cx node[14],node[18];
cx node[1],node[0];
cx node[2],node[3];
cx node[4],node[5];
cx node[18],node[14];
cx node[0],node[1];
cx node[3],node[2];
cx node[5],node[4];
cx node[14],node[18];
cx node[0],node[14];
cx node[2],node[3];
cx node[4],node[5];
cx node[19],node[18];
cx node[14],node[0];
cx node[1],node[2];
cx node[15],node[4];
cx node[6],node[5];
rz(0.0009765625*pi) node[18];
cx node[0],node[14];
cx node[2],node[1];
rz(0.015625*pi) node[4];
rz(0.0625*pi) node[5];
cx node[19],node[18];
cx node[1],node[2];
cx node[15],node[4];
cx node[6],node[5];
rz(3.9990234375*pi) node[18];
cx node[1],node[0];
rz(3.984375*pi) node[4];
rz(3.9375*pi) node[5];
cx node[19],node[18];
cx node[0],node[1];
cx node[3],node[4];
cx node[18],node[19];
cx node[1],node[0];
cx node[4],node[3];
cx node[19],node[18];
cx node[3],node[4];
cx node[18],node[14];
cx node[20],node[19];
cx node[2],node[3];
cx node[4],node[5];
rz(0.001953125*pi) node[14];
rz(0.00048828125*pi) node[19];
cx node[3],node[2];
cx node[5],node[4];
cx node[18],node[14];
cx node[20],node[19];
cx node[2],node[3];
cx node[4],node[5];
rz(3.998046875*pi) node[14];
rz(3.99951171875*pi) node[19];
cx node[2],node[1];
cx node[15],node[4];
cx node[6],node[5];
cx node[18],node[14];
cx node[20],node[19];
cx node[1],node[2];
rz(0.03125*pi) node[4];
rz(0.125*pi) node[5];
cx node[14],node[18];
cx node[19],node[20];
cx node[2],node[1];
cx node[15],node[4];
cx node[6],node[5];
cx node[18],node[14];
cx node[20],node[19];
cx node[14],node[0];
rz(3.96875*pi) node[4];
rz(3.875*pi) node[5];
cx node[19],node[18];
rz(0.00390625*pi) node[0];
cx node[3],node[4];
rz(0.0009765625*pi) node[18];
cx node[14],node[0];
cx node[4],node[3];
cx node[19],node[18];
rz(3.99609375*pi) node[0];
cx node[3],node[4];
rz(3.9990234375*pi) node[18];
cx node[0],node[14];
cx node[3],node[2];
cx node[4],node[5];
cx node[19],node[18];
cx node[14],node[0];
cx node[2],node[3];
cx node[5],node[4];
cx node[18],node[19];
cx node[0],node[14];
cx node[3],node[2];
cx node[4],node[5];
cx node[19],node[18];
cx node[0],node[1];
cx node[15],node[4];
cx node[6],node[5];
cx node[18],node[14];
rz(0.0078125*pi) node[1];
rz(0.0625*pi) node[4];
rz(0.25*pi) node[5];
rz(0.001953125*pi) node[14];
cx node[0],node[1];
cx node[15],node[4];
cx node[6],node[5];
cx node[18],node[14];
rz(3.9921875*pi) node[1];
rz(3.9375*pi) node[4];
rz(3.75*pi) node[5];
rz(0.5*pi) node[6];
rz(3.998046875*pi) node[14];
cx node[0],node[1];
cx node[4],node[3];
sx node[6];
cx node[18],node[14];
cx node[1],node[0];
cx node[3],node[4];
rz(3.5*pi) node[6];
cx node[14],node[18];
cx node[0],node[1];
cx node[4],node[3];
sx node[6];
cx node[18],node[14];
cx node[14],node[0];
cx node[1],node[2];
cx node[15],node[4];
rz(1.0*pi) node[6];
rz(0.00390625*pi) node[0];
rz(0.015625*pi) node[2];
cx node[4],node[15];
cx node[14],node[0];
cx node[1],node[2];
cx node[15],node[4];
rz(3.99609375*pi) node[0];
rz(3.984375*pi) node[2];
cx node[4],node[5];
cx node[14],node[0];
cx node[1],node[2];
rz(0.125*pi) node[5];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[5];
cx node[14],node[0];
cx node[1],node[2];
rz(3.875*pi) node[5];
cx node[0],node[1];
cx node[2],node[3];
cx node[5],node[4];
rz(0.0078125*pi) node[1];
rz(0.03125*pi) node[3];
cx node[4],node[5];
cx node[0],node[1];
cx node[2],node[3];
cx node[5],node[4];
rz(3.9921875*pi) node[1];
rz(3.96875*pi) node[3];
cx node[5],node[6];
cx node[0],node[1];
cx node[3],node[2];
rz(0.25*pi) node[6];
cx node[1],node[0];
cx node[2],node[3];
cx node[5],node[6];
cx node[0],node[1];
cx node[3],node[2];
rz(0.5*pi) node[5];
rz(3.75*pi) node[6];
cx node[1],node[2];
cx node[3],node[4];
sx node[5];
rz(0.015625*pi) node[2];
rz(0.0625*pi) node[4];
rz(3.5*pi) node[5];
cx node[1],node[2];
cx node[3],node[4];
sx node[5];
rz(3.984375*pi) node[2];
rz(3.9375*pi) node[4];
rz(1.0*pi) node[5];
cx node[1],node[2];
cx node[3],node[4];
cx node[6],node[5];
cx node[2],node[1];
cx node[4],node[3];
cx node[5],node[6];
cx node[1],node[2];
cx node[3],node[4];
cx node[6],node[5];
cx node[2],node[3];
cx node[4],node[5];
rz(0.03125*pi) node[3];
rz(0.125*pi) node[5];
cx node[2],node[3];
cx node[4],node[5];
rz(3.96875*pi) node[3];
rz(3.875*pi) node[5];
cx node[2],node[3];
cx node[5],node[4];
cx node[3],node[2];
cx node[4],node[5];
cx node[2],node[3];
cx node[5],node[4];
cx node[3],node[4];
cx node[5],node[6];
rz(0.0625*pi) node[4];
rz(0.25*pi) node[6];
cx node[3],node[4];
cx node[5],node[6];
rz(3.9375*pi) node[4];
rz(0.5*pi) node[5];
rz(3.75*pi) node[6];
cx node[3],node[4];
sx node[5];
cx node[4],node[3];
rz(3.5*pi) node[5];
cx node[3],node[4];
sx node[5];
rz(1.0*pi) node[5];
cx node[4],node[5];
cx node[5],node[4];
cx node[4],node[5];
cx node[5],node[6];
rz(0.125*pi) node[6];
cx node[5],node[6];
cx node[5],node[4];
rz(3.875*pi) node[6];
rz(0.25*pi) node[4];
cx node[5],node[4];
rz(3.75*pi) node[4];
rz(0.5*pi) node[5];
sx node[5];
rz(3.5*pi) node[5];
sx node[5];
rz(1.0*pi) node[5];
barrier node[5],node[4],node[6],node[3],node[2],node[1],node[0],node[14],node[18],node[19],node[20],node[22],node[23],node[33],node[21],node[15];
measure node[5] -> meas[0];
measure node[4] -> meas[1];
measure node[6] -> meas[2];
measure node[3] -> meas[3];
measure node[2] -> meas[4];
measure node[1] -> meas[5];
measure node[0] -> meas[6];
measure node[14] -> meas[7];
measure node[18] -> meas[8];
measure node[19] -> meas[9];
measure node[20] -> meas[10];
measure node[22] -> meas[11];
measure node[23] -> meas[12];
measure node[33] -> meas[13];
measure node[21] -> meas[14];
measure node[15] -> meas[15];
