OPENQASM 2.0;
include "qelib1.inc";

qreg node[34];
creg c[16];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
rz(1.4288864115556332*pi) node[4];
sx node[5];
rz(3.5*pi) node[6];
sx node[7];
sx node[14];
sx node[15];
sx node[18];
sx node[19];
rz(3.5*pi) node[20];
rz(1.0*pi) node[21];
rz(1.0*pi) node[22];
sx node[23];
rz(1.0*pi) node[33];
rz(3.5*pi) node[0];
rz(3.5*pi) node[1];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
rz(3.5*pi) node[5];
x node[6];
rz(0.5*pi) node[7];
rz(3.5*pi) node[14];
rz(3.5*pi) node[15];
rz(3.5*pi) node[18];
x node[20];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[5];
rz(0.5*pi) node[6];
sx node[7];
sx node[14];
sx node[15];
sx node[18];
rz(0.5*pi) node[20];
rz(1.0*pi) node[0];
rz(1.0*pi) node[1];
rz(1.0*pi) node[2];
rz(1.0*pi) node[3];
rz(1.0*pi) node[5];
rz(1.0*pi) node[14];
rz(1.0*pi) node[15];
rz(1.0*pi) node[18];
cx node[4],node[3];
rz(0.0711135883704126*pi) node[3];
x node[3];
rz(0.5*pi) node[3];
cx node[4],node[3];
rz(0.42890167041864946*pi) node[3];
cx node[4],node[5];
rz(0.6422271724150597*pi) node[5];
x node[5];
rz(0.5*pi) node[5];
cx node[4],node[5];
cx node[4],node[15];
rz(1.857803345163065*pi) node[5];
cx node[6],node[5];
rz(0.7844543511963169*pi) node[15];
cx node[5],node[6];
x node[15];
cx node[6],node[5];
rz(0.5*pi) node[15];
cx node[4],node[15];
cx node[7],node[6];
cx node[4],node[3];
cx node[6],node[7];
rz(1.7156066839599329*pi) node[15];
cx node[3],node[4];
cx node[7],node[6];
cx node[22],node[15];
cx node[4],node[3];
cx node[15],node[22];
cx node[3],node[2];
cx node[22],node[15];
rz(0.06890868761980595*pi) node[2];
cx node[15],node[4];
cx node[21],node[22];
x node[2];
cx node[4],node[15];
cx node[22],node[21];
rz(0.5*pi) node[2];
cx node[15],node[4];
cx node[21],node[22];
cx node[3],node[2];
cx node[22],node[15];
rz(2.4312133826926936*pi) node[2];
cx node[15],node[22];
cx node[3],node[2];
cx node[22],node[15];
cx node[2],node[3];
cx node[23],node[22];
cx node[3],node[2];
cx node[22],node[23];
cx node[2],node[1];
cx node[4],node[3];
cx node[23],node[22];
rz(0.6378173820546926*pi) node[1];
cx node[3],node[4];
x node[1];
cx node[4],node[3];
rz(0.5*pi) node[1];
cx node[5],node[4];
cx node[2],node[1];
cx node[4],node[5];
rz(1.8624267585703067*pi) node[1];
cx node[5],node[4];
cx node[2],node[1];
cx node[6],node[5];
cx node[1],node[2];
cx node[5],node[6];
cx node[2],node[1];
cx node[6],node[5];
cx node[1],node[0];
cx node[3],node[2];
rz(0.7756347641093853*pi) node[0];
cx node[2],node[3];
x node[0];
cx node[3],node[2];
rz(0.5*pi) node[0];
cx node[4],node[3];
cx node[1],node[0];
cx node[3],node[4];
rz(1.7248535171406143*pi) node[0];
cx node[4],node[3];
cx node[14],node[0];
cx node[15],node[4];
cx node[0],node[14];
cx node[4],node[15];
cx node[14],node[0];
cx node[15],node[4];
cx node[1],node[0];
cx node[18],node[14];
cx node[22],node[15];
rz(0.051269529361436916*pi) node[0];
cx node[14],node[18];
cx node[15],node[22];
x node[0];
cx node[18],node[14];
cx node[22],node[15];
rz(0.5*pi) node[0];
cx node[19],node[18];
cx node[1],node[0];
cx node[18],node[19];
rz(2.4497070331385626*pi) node[0];
cx node[19],node[18];
cx node[14],node[0];
cx node[20],node[19];
cx node[0],node[14];
cx node[19],node[20];
cx node[14],node[0];
cx node[20],node[19];
cx node[1],node[0];
cx node[18],node[14];
cx node[33],node[20];
rz(0.6025390623548557*pi) node[0];
cx node[14],node[18];
cx node[20],node[33];
x node[0];
cx node[18],node[14];
cx node[33],node[20];
rz(0.5*pi) node[0];
cx node[19],node[18];
cx node[1],node[0];
cx node[18],node[19];
rz(1.899414062645144*pi) node[0];
rz(0.5*pi) node[1];
cx node[19],node[18];
cx node[14],node[0];
sx node[1];
cx node[20],node[19];
cx node[0],node[14];
rz(3.5*pi) node[1];
cx node[19],node[20];
cx node[14],node[0];
sx node[1];
cx node[20],node[19];
rz(1.0*pi) node[1];
cx node[18],node[14];
cx node[1],node[0];
cx node[14],node[18];
rz(1.5*pi) node[1];
cx node[18],node[14];
sx node[1];
cx node[19],node[18];
rz(0.2050781310759089*pi) node[1];
cx node[18],node[19];
sx node[1];
cx node[19],node[18];
rz(2.5*pi) node[1];
cx node[1],node[0];
sx node[0];
sx node[1];
rz(2.5*pi) node[0];
sx node[0];
rz(2.2988281189240913*pi) node[0];
cx node[14],node[0];
cx node[0],node[14];
cx node[14],node[0];
cx node[1],node[0];
cx node[18],node[14];
rz(0.5*pi) node[1];
cx node[14],node[18];
sx node[1];
cx node[18],node[14];
rz(2.5898437537636765*pi) node[1];
sx node[1];
rz(1.5*pi) node[1];
cx node[1],node[0];
sx node[0];
cx node[1],node[2];
rz(3.5*pi) node[0];
rz(1.5*pi) node[1];
sx node[0];
sx node[1];
rz(3.5976562537636756*pi) node[0];
rz(3.1796875063846866*pi) node[1];
cx node[14],node[0];
sx node[1];
cx node[0],node[14];
rz(1.5*pi) node[1];
cx node[14],node[0];
cx node[1],node[2];
sx node[2];
rz(3.5*pi) node[2];
sx node[2];
rz(1.1953125063846857*pi) node[2];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[1],node[2];
cx node[4],node[3];
rz(1.5*pi) node[1];
cx node[3],node[4];
sx node[1];
cx node[4],node[3];
rz(3.359374996853879*pi) node[1];
cx node[15],node[4];
sx node[1];
cx node[4],node[15];
rz(2.5*pi) node[1];
cx node[15],node[4];
cx node[1],node[2];
sx node[2];
rz(3.5*pi) node[2];
sx node[2];
rz(0.39062499685387764*pi) node[2];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[1],node[2];
cx node[4],node[3];
rz(0.5*pi) node[1];
cx node[3],node[4];
sx node[1];
cx node[4],node[3];
rz(1.7187499999999996*pi) node[1];
cx node[5],node[4];
sx node[1];
cx node[4],node[5];
rz(1.5*pi) node[1];
cx node[5],node[4];
cx node[1],node[2];
cx node[1],node[0];
sx node[2];
rz(1.5*pi) node[1];
rz(3.5*pi) node[2];
sx node[1];
sx node[2];
rz(3.4375*pi) node[1];
rz(3.781249999999999*pi) node[2];
sx node[1];
cx node[3],node[2];
rz(1.5*pi) node[1];
cx node[2],node[3];
cx node[1],node[0];
cx node[3],node[2];
sx node[0];
sx node[1];
cx node[4],node[3];
rz(3.5*pi) node[0];
cx node[1],node[2];
cx node[3],node[4];
sx node[0];
rz(1.5*pi) node[1];
cx node[4],node[3];
rz(3.5624999999999996*pi) node[0];
sx node[1];
rz(0.12500000000000022*pi) node[1];
sx node[1];
rz(1.0*pi) node[1];
cx node[1],node[2];
rz(0.5*pi) node[1];
sx node[2];
sx node[1];
rz(2.5*pi) node[2];
rz(3.5*pi) node[1];
sx node[2];
sx node[1];
rz(2.624999999999999*pi) node[2];
rz(1.0*pi) node[1];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[1],node[2];
rz(0.25*pi) node[2];
cx node[1],node[2];
cx node[0],node[1];
rz(0.75*pi) node[2];
cx node[1],node[0];
sx node[2];
cx node[0],node[1];
rz(3.5*pi) node[2];
cx node[14],node[0];
sx node[2];
cx node[0],node[14];
rz(1.0*pi) node[2];
cx node[14],node[0];
cx node[3],node[2];
rz(0.25*pi) node[2];
cx node[18],node[14];
cx node[3],node[2];
cx node[14],node[18];
rz(3.75*pi) node[2];
rz(0.5*pi) node[3];
cx node[18],node[14];
cx node[1],node[2];
sx node[3];
cx node[19],node[18];
rz(0.125*pi) node[2];
rz(3.5*pi) node[3];
cx node[18],node[19];
cx node[1],node[2];
sx node[3];
cx node[19],node[18];
rz(3.875*pi) node[2];
rz(1.0*pi) node[3];
cx node[20],node[19];
cx node[3],node[2];
cx node[19],node[20];
cx node[2],node[3];
cx node[20],node[19];
cx node[3],node[2];
cx node[33],node[20];
cx node[1],node[2];
cx node[4],node[3];
cx node[20],node[33];
rz(0.25*pi) node[2];
rz(0.0625*pi) node[3];
cx node[33],node[20];
cx node[1],node[2];
cx node[4],node[3];
rz(0.5*pi) node[1];
rz(3.75*pi) node[2];
rz(3.9375*pi) node[3];
sx node[1];
cx node[4],node[3];
rz(3.5*pi) node[1];
cx node[3],node[4];
sx node[1];
cx node[4],node[3];
rz(1.0*pi) node[1];
cx node[3],node[2];
cx node[5],node[4];
rz(0.125*pi) node[2];
rz(0.03125*pi) node[4];
cx node[3],node[2];
cx node[5],node[4];
rz(3.875*pi) node[2];
rz(3.96875*pi) node[4];
cx node[15],node[4];
rz(0.015625*pi) node[4];
cx node[15],node[4];
rz(3.984375*pi) node[4];
cx node[4],node[3];
cx node[3],node[4];
cx node[4],node[3];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[1];
cx node[4],node[3];
cx node[1],node[2];
cx node[3],node[4];
cx node[2],node[1];
cx node[4],node[3];
cx node[0],node[1];
cx node[3],node[2];
cx node[5],node[4];
rz(0.0078125*pi) node[1];
rz(0.25*pi) node[2];
rz(0.0625*pi) node[4];
cx node[0],node[1];
cx node[3],node[2];
cx node[5],node[4];
rz(3.9921875*pi) node[1];
rz(3.75*pi) node[2];
rz(0.5*pi) node[3];
rz(3.9375*pi) node[4];
cx node[1],node[0];
sx node[3];
cx node[15],node[4];
cx node[0],node[1];
rz(3.5*pi) node[3];
rz(0.03125*pi) node[4];
cx node[1],node[0];
sx node[3];
cx node[15],node[4];
cx node[14],node[0];
rz(1.0*pi) node[3];
rz(3.96875*pi) node[4];
rz(0.00390625*pi) node[0];
cx node[4],node[3];
cx node[14],node[0];
cx node[3],node[4];
rz(3.99609375*pi) node[0];
cx node[4],node[3];
cx node[0],node[14];
cx node[3],node[2];
cx node[14],node[0];
cx node[2],node[3];
cx node[0],node[14];
cx node[3],node[2];
cx node[1],node[2];
cx node[18],node[14];
rz(0.015625*pi) node[2];
rz(0.001953125*pi) node[14];
cx node[1],node[2];
cx node[18],node[14];
rz(3.984375*pi) node[2];
rz(3.998046875*pi) node[14];
cx node[2],node[1];
cx node[18],node[14];
cx node[1],node[2];
cx node[14],node[18];
cx node[2],node[1];
cx node[18],node[14];
cx node[0],node[1];
cx node[19],node[18];
rz(0.0078125*pi) node[1];
rz(0.0009765625*pi) node[18];
cx node[0],node[1];
cx node[19],node[18];
rz(3.9921875*pi) node[1];
rz(3.9990234375*pi) node[18];
cx node[1],node[0];
cx node[19],node[18];
cx node[0],node[1];
cx node[18],node[19];
cx node[1],node[0];
cx node[19],node[18];
cx node[14],node[0];
cx node[20],node[19];
rz(0.00390625*pi) node[0];
rz(0.00048828125*pi) node[19];
cx node[14],node[0];
cx node[20],node[19];
rz(3.99609375*pi) node[0];
rz(3.99951171875*pi) node[19];
cx node[0],node[14];
cx node[20],node[19];
cx node[14],node[0];
cx node[19],node[20];
cx node[0],node[14];
cx node[20],node[19];
cx node[18],node[14];
cx node[20],node[21];
rz(0.001953125*pi) node[14];
cx node[21],node[20];
cx node[18],node[14];
cx node[20],node[21];
rz(3.998046875*pi) node[14];
cx node[22],node[21];
cx node[18],node[14];
rz(0.000244140625*pi) node[21];
cx node[14],node[18];
cx node[22],node[21];
cx node[18],node[14];
rz(3.999755859375*pi) node[21];
cx node[19],node[18];
cx node[21],node[22];
rz(0.0009765625*pi) node[18];
cx node[22],node[21];
cx node[19],node[18];
cx node[21],node[22];
cx node[22],node[15];
rz(3.9990234375*pi) node[18];
cx node[20],node[21];
cx node[15],node[22];
cx node[19],node[18];
cx node[21],node[20];
cx node[22],node[15];
cx node[18],node[19];
cx node[20],node[21];
cx node[15],node[4];
cx node[19],node[18];
cx node[4],node[15];
cx node[20],node[19];
cx node[15],node[4];
rz(0.00048828125*pi) node[19];
cx node[5],node[4];
cx node[20],node[19];
cx node[4],node[5];
rz(3.99951171875*pi) node[19];
cx node[5],node[4];
cx node[19],node[20];
cx node[4],node[3];
cx node[6],node[5];
cx node[20],node[19];
rz(0.125*pi) node[3];
rz(0.0001220703125*pi) node[5];
cx node[19],node[20];
cx node[4],node[3];
cx node[6],node[5];
rz(3.875*pi) node[3];
cx node[4],node[15];
rz(3.9998779296875*pi) node[5];
cx node[7],node[6];
cx node[6],node[7];
rz(0.25*pi) node[15];
cx node[4],node[15];
cx node[7],node[6];
rz(0.5*pi) node[4];
rz(3.75*pi) node[15];
sx node[4];
cx node[22],node[15];
rz(3.5*pi) node[4];
cx node[15],node[22];
sx node[4];
cx node[22],node[15];
rz(1.0*pi) node[4];
cx node[5],node[4];
cx node[4],node[5];
cx node[5],node[4];
cx node[15],node[4];
cx node[4],node[15];
cx node[15],node[4];
cx node[4],node[3];
cx node[22],node[15];
rz(0.0625*pi) node[3];
cx node[15],node[22];
cx node[4],node[3];
cx node[22],node[15];
rz(3.9375*pi) node[3];
cx node[4],node[15];
cx node[21],node[22];
cx node[2],node[3];
rz(0.125*pi) node[15];
rz(6.103515625e-05*pi) node[22];
rz(0.03125*pi) node[3];
cx node[4],node[15];
cx node[21],node[22];
cx node[2],node[3];
cx node[4],node[5];
rz(3.875*pi) node[15];
cx node[20],node[21];
rz(3.99993896484375*pi) node[22];
rz(3.96875*pi) node[3];
rz(0.25*pi) node[5];
cx node[21],node[20];
cx node[3],node[2];
cx node[4],node[5];
cx node[20],node[21];
cx node[2],node[3];
rz(0.5*pi) node[4];
rz(3.75*pi) node[5];
cx node[3],node[2];
sx node[4];
cx node[6],node[5];
cx node[1],node[2];
rz(3.5*pi) node[4];
cx node[5],node[6];
rz(0.015625*pi) node[2];
sx node[4];
cx node[6],node[5];
cx node[1],node[2];
rz(1.0*pi) node[4];
rz(3.984375*pi) node[2];
cx node[5],node[4];
cx node[2],node[1];
cx node[4],node[5];
cx node[1],node[2];
cx node[5],node[4];
cx node[2],node[1];
cx node[15],node[4];
cx node[6],node[5];
cx node[0],node[1];
cx node[4],node[15];
cx node[5],node[6];
rz(0.0078125*pi) node[1];
cx node[15],node[4];
cx node[6],node[5];
cx node[0],node[1];
cx node[3],node[4];
cx node[15],node[22];
rz(3.9921875*pi) node[1];
rz(0.0625*pi) node[4];
rz(3.0517578125e-05*pi) node[22];
cx node[0],node[1];
cx node[3],node[4];
cx node[15],node[22];
cx node[1],node[0];
rz(3.9375*pi) node[4];
rz(3.999969482421875*pi) node[22];
cx node[0],node[1];
cx node[4],node[3];
cx node[23],node[22];
cx node[14],node[0];
cx node[3],node[4];
rz(1.52587890625e-05*pi) node[22];
rz(0.00390625*pi) node[0];
cx node[4],node[3];
cx node[23],node[22];
cx node[14],node[0];
cx node[2],node[3];
rz(3.9999847412109375*pi) node[22];
rz(3.99609375*pi) node[0];
rz(0.03125*pi) node[3];
cx node[21],node[22];
cx node[0],node[14];
cx node[2],node[3];
cx node[22],node[21];
cx node[14],node[0];
rz(3.96875*pi) node[3];
cx node[21],node[22];
cx node[0],node[14];
cx node[3],node[2];
cx node[22],node[15];
cx node[20],node[21];
cx node[2],node[3];
cx node[18],node[14];
cx node[15],node[22];
cx node[21],node[20];
cx node[3],node[2];
rz(0.001953125*pi) node[14];
cx node[22],node[15];
cx node[20],node[21];
cx node[1],node[2];
cx node[4],node[15];
cx node[18],node[14];
cx node[21],node[22];
rz(0.015625*pi) node[2];
cx node[15],node[4];
rz(3.998046875*pi) node[14];
cx node[22],node[21];
cx node[1],node[2];
cx node[4],node[15];
cx node[18],node[14];
cx node[21],node[22];
rz(3.984375*pi) node[2];
cx node[4],node[5];
cx node[14],node[18];
cx node[1],node[2];
cx node[5],node[4];
cx node[18],node[14];
cx node[2],node[1];
cx node[4],node[5];
cx node[19],node[18];
cx node[1],node[2];
cx node[15],node[4];
cx node[5],node[6];
rz(0.0009765625*pi) node[18];
cx node[0],node[1];
rz(0.125*pi) node[4];
cx node[6],node[5];
cx node[19],node[18];
rz(0.0078125*pi) node[1];
cx node[15],node[4];
cx node[5],node[6];
rz(3.9990234375*pi) node[18];
cx node[0],node[1];
rz(3.875*pi) node[4];
cx node[7],node[6];
cx node[18],node[19];
rz(3.9921875*pi) node[1];
cx node[3],node[4];
rz(0.000244140625*pi) node[6];
cx node[19],node[18];
cx node[0],node[1];
rz(0.0625*pi) node[4];
cx node[7],node[6];
cx node[18],node[19];
cx node[1],node[0];
cx node[3],node[4];
rz(3.999755859375*pi) node[6];
cx node[19],node[20];
cx node[0],node[1];
rz(3.9375*pi) node[4];
cx node[5],node[6];
cx node[20],node[19];
cx node[14],node[0];
cx node[6],node[5];
cx node[19],node[20];
rz(0.00390625*pi) node[0];
cx node[5],node[6];
cx node[20],node[21];
cx node[14],node[0];
cx node[4],node[5];
cx node[21],node[20];
rz(3.99609375*pi) node[0];
cx node[5],node[4];
cx node[20],node[21];
cx node[0],node[14];
cx node[4],node[5];
cx node[22],node[21];
cx node[14],node[0];
cx node[21],node[22];
cx node[0],node[14];
cx node[22],node[21];
cx node[18],node[14];
cx node[15],node[22];
rz(0.001953125*pi) node[14];
cx node[22],node[15];
cx node[18],node[14];
cx node[15],node[22];
cx node[4],node[15];
rz(3.998046875*pi) node[14];
cx node[0],node[14];
cx node[15],node[4];
cx node[14],node[0];
cx node[4],node[15];
cx node[0],node[14];
cx node[5],node[4];
cx node[22],node[15];
cx node[4],node[5];
cx node[15],node[22];
cx node[5],node[4];
cx node[22],node[15];
cx node[4],node[3];
cx node[5],node[6];
cx node[21],node[22];
cx node[3],node[4];
cx node[6],node[5];
rz(0.0001220703125*pi) node[22];
cx node[4],node[3];
cx node[5],node[6];
cx node[21],node[22];
cx node[2],node[3];
cx node[7],node[6];
rz(3.9998779296875*pi) node[22];
rz(0.03125*pi) node[3];
rz(0.00048828125*pi) node[6];
cx node[21],node[22];
cx node[2],node[3];
cx node[7],node[6];
cx node[22],node[21];
rz(3.96875*pi) node[3];
rz(3.99951171875*pi) node[6];
cx node[21],node[22];
cx node[3],node[2];
cx node[5],node[6];
cx node[20],node[21];
cx node[2],node[3];
cx node[6],node[5];
rz(6.103515625e-05*pi) node[21];
cx node[3],node[2];
cx node[5],node[6];
cx node[20],node[21];
cx node[1],node[2];
cx node[5],node[4];
rz(3.99993896484375*pi) node[21];
rz(0.015625*pi) node[2];
cx node[4],node[5];
cx node[1],node[2];
cx node[5],node[4];
cx node[0],node[1];
rz(3.984375*pi) node[2];
cx node[4],node[15];
cx node[6],node[5];
cx node[1],node[0];
cx node[15],node[4];
cx node[5],node[6];
cx node[0],node[1];
cx node[4],node[15];
cx node[6],node[5];
cx node[1],node[2];
cx node[22],node[15];
cx node[2],node[1];
rz(0.000244140625*pi) node[15];
cx node[1],node[2];
cx node[22],node[15];
cx node[0],node[1];
cx node[2],node[3];
rz(3.999755859375*pi) node[15];
cx node[1],node[0];
cx node[3],node[2];
cx node[15],node[22];
cx node[0],node[1];
cx node[2],node[3];
cx node[22],node[15];
cx node[14],node[0];
cx node[3],node[4];
cx node[15],node[22];
rz(0.0078125*pi) node[0];
cx node[4],node[3];
cx node[22],node[21];
cx node[14],node[0];
cx node[3],node[4];
cx node[21],node[22];
rz(3.9921875*pi) node[0];
cx node[5],node[4];
cx node[22],node[21];
cx node[0],node[14];
cx node[4],node[5];
cx node[20],node[21];
cx node[23],node[22];
cx node[14],node[0];
cx node[5],node[4];
rz(0.0001220703125*pi) node[21];
rz(3.0517578125e-05*pi) node[22];
cx node[0],node[14];
cx node[3],node[4];
cx node[6],node[5];
cx node[20],node[21];
cx node[23],node[22];
rz(0.25*pi) node[4];
cx node[5],node[6];
cx node[18],node[14];
rz(3.9998779296875*pi) node[21];
rz(3.999969482421875*pi) node[22];
cx node[3],node[4];
cx node[6],node[5];
rz(0.00390625*pi) node[14];
rz(0.5*pi) node[3];
rz(3.75*pi) node[4];
cx node[7],node[6];
cx node[18],node[14];
sx node[3];
cx node[5],node[4];
rz(0.0009765625*pi) node[6];
rz(3.99609375*pi) node[14];
rz(3.5*pi) node[3];
rz(0.125*pi) node[4];
cx node[7],node[6];
sx node[3];
cx node[5],node[4];
rz(3.9990234375*pi) node[6];
rz(1.0*pi) node[3];
rz(3.875*pi) node[4];
cx node[4],node[3];
cx node[3],node[4];
cx node[4],node[3];
cx node[2],node[3];
cx node[5],node[4];
rz(0.0625*pi) node[3];
rz(0.25*pi) node[4];
cx node[2],node[3];
cx node[5],node[4];
rz(3.9375*pi) node[3];
rz(3.75*pi) node[4];
rz(0.5*pi) node[5];
cx node[3],node[2];
sx node[5];
cx node[2],node[3];
rz(3.5*pi) node[5];
cx node[3],node[2];
sx node[5];
cx node[1],node[2];
cx node[3],node[4];
rz(1.0*pi) node[5];
rz(0.03125*pi) node[2];
rz(0.125*pi) node[4];
cx node[1],node[2];
cx node[3],node[4];
rz(3.96875*pi) node[2];
rz(3.875*pi) node[4];
cx node[1],node[2];
cx node[15],node[4];
cx node[2],node[1];
cx node[4],node[15];
cx node[1],node[2];
cx node[15],node[4];
cx node[0],node[1];
cx node[5],node[4];
rz(0.015625*pi) node[1];
cx node[4],node[5];
cx node[0],node[1];
cx node[5],node[4];
cx node[14],node[0];
rz(3.984375*pi) node[1];
cx node[3],node[4];
cx node[5],node[6];
cx node[0],node[14];
rz(0.25*pi) node[4];
rz(0.00048828125*pi) node[6];
cx node[14],node[0];
cx node[3],node[4];
cx node[5],node[6];
cx node[0],node[1];
rz(0.5*pi) node[3];
rz(3.75*pi) node[4];
rz(3.99951171875*pi) node[6];
cx node[1],node[0];
sx node[3];
cx node[6],node[5];
cx node[0],node[1];
rz(3.5*pi) node[3];
cx node[5],node[6];
cx node[0],node[14];
cx node[1],node[2];
sx node[3];
cx node[6],node[5];
cx node[14],node[0];
cx node[2],node[1];
rz(1.0*pi) node[3];
cx node[5],node[4];
cx node[7],node[6];
cx node[0],node[14];
cx node[1],node[2];
cx node[4],node[5];
cx node[6],node[7];
cx node[2],node[3];
cx node[5],node[4];
cx node[7],node[6];
cx node[18],node[14];
cx node[3],node[2];
cx node[15],node[4];
cx node[6],node[5];
rz(0.0078125*pi) node[14];
cx node[2],node[3];
cx node[4],node[15];
cx node[5],node[6];
cx node[18],node[14];
cx node[1],node[2];
cx node[15],node[4];
cx node[6],node[5];
rz(3.9921875*pi) node[14];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[3];
cx node[15],node[22];
cx node[14],node[0];
cx node[1],node[2];
cx node[3],node[4];
cx node[22],node[15];
cx node[0],node[14];
cx node[4],node[3];
cx node[15],node[22];
cx node[0],node[1];
cx node[2],node[3];
cx node[5],node[4];
cx node[22],node[21];
cx node[1],node[0];
rz(0.0625*pi) node[3];
rz(0.001953125*pi) node[4];
cx node[21],node[22];
cx node[0],node[1];
cx node[2],node[3];
cx node[5],node[4];
cx node[22],node[21];
rz(3.9375*pi) node[3];
rz(3.998046875*pi) node[4];
cx node[20],node[21];
cx node[23],node[22];
cx node[2],node[3];
cx node[4],node[5];
rz(0.000244140625*pi) node[21];
rz(6.103515625e-05*pi) node[22];
cx node[3],node[2];
cx node[5],node[4];
cx node[20],node[21];
cx node[23],node[22];
cx node[2],node[3];
cx node[4],node[5];
rz(3.999755859375*pi) node[21];
rz(3.99993896484375*pi) node[22];
cx node[1],node[2];
cx node[3],node[4];
cx node[6],node[5];
cx node[20],node[21];
cx node[2],node[1];
cx node[4],node[3];
cx node[5],node[6];
cx node[21],node[20];
cx node[1],node[2];
cx node[3],node[4];
cx node[6],node[5];
cx node[20],node[21];
cx node[0],node[1];
cx node[3],node[2];
cx node[4],node[5];
cx node[7],node[6];
cx node[21],node[22];
cx node[1],node[0];
rz(0.00390625*pi) node[2];
rz(0.125*pi) node[5];
rz(0.0009765625*pi) node[6];
cx node[22],node[21];
cx node[0],node[1];
cx node[3],node[2];
cx node[4],node[5];
cx node[7],node[6];
cx node[21],node[22];
cx node[14],node[0];
rz(3.99609375*pi) node[2];
rz(3.875*pi) node[5];
rz(3.9990234375*pi) node[6];
cx node[22],node[15];
cx node[20],node[21];
rz(0.03125*pi) node[0];
cx node[2],node[3];
cx node[6],node[5];
cx node[15],node[22];
cx node[21],node[20];
cx node[14],node[0];
cx node[3],node[2];
cx node[5],node[6];
cx node[22],node[15];
cx node[20],node[21];
rz(3.96875*pi) node[0];
cx node[2],node[3];
cx node[6],node[5];
cx node[23],node[22];
cx node[0],node[14];
cx node[4],node[3];
cx node[22],node[23];
cx node[14],node[0];
cx node[3],node[4];
cx node[23],node[22];
cx node[0],node[14];
cx node[4],node[3];
cx node[22],node[21];
cx node[0],node[1];
cx node[4],node[5];
cx node[18],node[14];
rz(0.0001220703125*pi) node[21];
cx node[1],node[0];
cx node[5],node[4];
rz(0.015625*pi) node[14];
cx node[22],node[21];
cx node[0],node[1];
cx node[4],node[5];
cx node[18],node[14];
rz(3.9998779296875*pi) node[21];
cx node[1],node[2];
cx node[15],node[4];
cx node[5],node[6];
rz(3.984375*pi) node[14];
cx node[2],node[1];
rz(0.00048828125*pi) node[4];
cx node[6],node[5];
cx node[1],node[2];
cx node[15],node[4];
cx node[5],node[6];
cx node[0],node[1];
cx node[3],node[2];
rz(3.99951171875*pi) node[4];
cx node[7],node[6];
cx node[1],node[0];
cx node[2],node[3];
cx node[15],node[4];
rz(0.001953125*pi) node[6];
cx node[0],node[1];
cx node[3],node[2];
cx node[4],node[15];
cx node[7],node[6];
cx node[0],node[14];
cx node[2],node[1];
cx node[15],node[4];
rz(3.998046875*pi) node[6];
rz(0.25*pi) node[1];
cx node[5],node[4];
cx node[7],node[6];
rz(0.0078125*pi) node[14];
cx node[22],node[15];
cx node[0],node[14];
cx node[2],node[1];
cx node[4],node[5];
cx node[6],node[7];
rz(0.000244140625*pi) node[15];
rz(3.75*pi) node[1];
rz(0.5*pi) node[2];
cx node[5],node[4];
cx node[7],node[6];
rz(3.9921875*pi) node[14];
cx node[22],node[15];
cx node[14],node[0];
sx node[2];
cx node[6],node[5];
rz(3.999755859375*pi) node[15];
cx node[0],node[14];
rz(3.5*pi) node[2];
cx node[5],node[6];
cx node[22],node[15];
cx node[14],node[0];
sx node[2];
cx node[6],node[5];
cx node[15],node[22];
cx node[0],node[1];
rz(1.0*pi) node[2];
cx node[6],node[7];
cx node[18],node[14];
cx node[22],node[15];
cx node[1],node[0];
rz(0.0009765625*pi) node[7];
cx node[14],node[18];
cx node[0],node[1];
cx node[6],node[7];
cx node[18],node[14];
cx node[1],node[2];
rz(3.9990234375*pi) node[7];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[3];
cx node[4],node[3];
cx node[3],node[4];
cx node[4],node[3];
cx node[2],node[3];
cx node[5],node[4];
rz(0.0625*pi) node[3];
rz(0.00390625*pi) node[4];
cx node[2],node[3];
cx node[5],node[4];
rz(3.9375*pi) node[3];
rz(3.99609375*pi) node[4];
cx node[2],node[3];
cx node[4],node[5];
cx node[3],node[2];
cx node[5],node[4];
cx node[2],node[3];
cx node[4],node[5];
cx node[2],node[1];
cx node[6],node[5];
cx node[1],node[2];
rz(0.001953125*pi) node[5];
cx node[2],node[1];
cx node[6],node[5];
cx node[0],node[1];
cx node[3],node[2];
rz(3.998046875*pi) node[5];
cx node[7],node[6];
cx node[1],node[0];
cx node[2],node[3];
cx node[6],node[7];
cx node[0],node[1];
cx node[3],node[2];
cx node[7],node[6];
cx node[14],node[0];
cx node[2],node[1];
cx node[6],node[5];
rz(0.03125*pi) node[0];
rz(0.125*pi) node[1];
cx node[5],node[6];
cx node[14],node[0];
cx node[2],node[1];
cx node[6],node[5];
rz(3.96875*pi) node[0];
rz(3.875*pi) node[1];
cx node[2],node[3];
cx node[0],node[14];
rz(0.25*pi) node[3];
cx node[14],node[0];
cx node[2],node[3];
cx node[0],node[14];
rz(0.5*pi) node[2];
rz(3.75*pi) node[3];
cx node[0],node[1];
sx node[2];
cx node[18],node[14];
rz(0.0625*pi) node[1];
rz(3.5*pi) node[2];
rz(0.015625*pi) node[14];
cx node[0],node[1];
sx node[2];
cx node[18],node[14];
rz(3.9375*pi) node[1];
rz(1.0*pi) node[2];
rz(3.984375*pi) node[14];
cx node[0],node[14];
cx node[14],node[0];
cx node[0],node[14];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[0],node[14];
cx node[1],node[2];
cx node[14],node[0];
cx node[2],node[1];
cx node[0],node[14];
cx node[1],node[2];
cx node[3],node[2];
cx node[18],node[14];
cx node[2],node[3];
rz(0.03125*pi) node[14];
cx node[3],node[2];
cx node[18],node[14];
cx node[4],node[3];
rz(3.96875*pi) node[14];
cx node[0],node[14];
rz(0.0078125*pi) node[3];
cx node[14],node[0];
cx node[4],node[3];
cx node[0],node[14];
rz(3.9921875*pi) node[3];
cx node[0],node[1];
cx node[3],node[4];
cx node[1],node[0];
cx node[4],node[3];
cx node[0],node[1];
cx node[3],node[4];
cx node[1],node[2];
cx node[4],node[5];
cx node[2],node[1];
cx node[5],node[4];
cx node[1],node[2];
cx node[4],node[5];
cx node[1],node[0];
cx node[3],node[2];
cx node[15],node[4];
cx node[5],node[6];
cx node[0],node[1];
rz(0.015625*pi) node[2];
rz(0.00048828125*pi) node[4];
cx node[6],node[5];
cx node[1],node[0];
cx node[3],node[2];
cx node[15],node[4];
cx node[5],node[6];
cx node[14],node[0];
rz(3.984375*pi) node[2];
rz(3.99951171875*pi) node[4];
cx node[7],node[6];
rz(0.125*pi) node[0];
cx node[2],node[3];
rz(0.00390625*pi) node[6];
cx node[14],node[0];
cx node[3],node[2];
cx node[7],node[6];
rz(3.875*pi) node[0];
cx node[2],node[3];
rz(3.99609375*pi) node[6];
cx node[0],node[14];
cx node[3],node[4];
cx node[14],node[0];
cx node[4],node[3];
cx node[0],node[14];
cx node[3],node[4];
cx node[0],node[1];
cx node[4],node[5];
cx node[18],node[14];
rz(0.25*pi) node[1];
cx node[5],node[4];
rz(0.0625*pi) node[14];
cx node[0],node[1];
cx node[4],node[5];
cx node[18],node[14];
rz(0.5*pi) node[0];
rz(3.75*pi) node[1];
cx node[15],node[4];
cx node[5],node[6];
rz(3.9375*pi) node[14];
sx node[0];
rz(0.0009765625*pi) node[4];
cx node[6],node[5];
rz(3.5*pi) node[0];
cx node[15],node[4];
cx node[5],node[6];
sx node[0];
rz(3.9990234375*pi) node[4];
cx node[7],node[6];
rz(1.0*pi) node[0];
cx node[15],node[4];
rz(0.0078125*pi) node[6];
cx node[14],node[0];
cx node[4],node[15];
cx node[7],node[6];
cx node[0],node[14];
cx node[15],node[4];
rz(3.9921875*pi) node[6];
cx node[14],node[0];
cx node[4],node[5];
cx node[7],node[6];
cx node[0],node[1];
rz(0.001953125*pi) node[5];
cx node[6],node[7];
cx node[18],node[14];
cx node[1],node[0];
cx node[4],node[5];
cx node[7],node[6];
cx node[14],node[18];
cx node[0],node[1];
rz(3.998046875*pi) node[5];
cx node[18],node[14];
cx node[14],node[0];
cx node[2],node[1];
cx node[6],node[5];
rz(0.125*pi) node[0];
rz(0.03125*pi) node[1];
cx node[5],node[6];
cx node[14],node[0];
cx node[2],node[1];
cx node[6],node[5];
rz(3.875*pi) node[0];
rz(3.96875*pi) node[1];
cx node[5],node[4];
cx node[7],node[6];
cx node[14],node[18];
cx node[1],node[2];
cx node[4],node[5];
cx node[6],node[7];
rz(0.25*pi) node[18];
cx node[2],node[1];
cx node[5],node[4];
cx node[7],node[6];
cx node[14],node[18];
cx node[1],node[2];
cx node[5],node[6];
rz(0.5*pi) node[14];
rz(3.75*pi) node[18];
cx node[1],node[0];
cx node[2],node[3];
rz(0.00390625*pi) node[6];
sx node[14];
rz(0.0625*pi) node[0];
cx node[3],node[2];
cx node[5],node[6];
rz(3.5*pi) node[14];
cx node[1],node[0];
cx node[2],node[3];
rz(3.99609375*pi) node[6];
sx node[14];
rz(3.9375*pi) node[0];
cx node[4],node[3];
rz(1.0*pi) node[14];
cx node[0],node[1];
rz(0.015625*pi) node[3];
cx node[18],node[14];
cx node[1],node[0];
cx node[4],node[3];
cx node[14],node[18];
cx node[0],node[1];
rz(3.984375*pi) node[3];
cx node[18],node[14];
cx node[0],node[14];
cx node[3],node[4];
cx node[4],node[3];
rz(0.125*pi) node[14];
cx node[0],node[14];
cx node[3],node[4];
cx node[3],node[2];
cx node[5],node[4];
rz(3.875*pi) node[14];
cx node[0],node[14];
cx node[2],node[3];
rz(0.0078125*pi) node[4];
cx node[14],node[0];
cx node[3],node[2];
cx node[5],node[4];
cx node[0],node[14];
cx node[2],node[1];
rz(3.9921875*pi) node[4];
rz(0.03125*pi) node[1];
cx node[5],node[4];
cx node[14],node[18];
cx node[2],node[1];
cx node[4],node[5];
rz(0.25*pi) node[18];
rz(3.96875*pi) node[1];
cx node[5],node[4];
cx node[14],node[18];
cx node[1],node[2];
cx node[4],node[3];
rz(0.5*pi) node[14];
rz(3.75*pi) node[18];
cx node[2],node[1];
cx node[3],node[4];
sx node[14];
cx node[1],node[2];
cx node[4],node[3];
rz(3.5*pi) node[14];
cx node[1],node[0];
cx node[3],node[2];
sx node[14];
rz(0.0625*pi) node[0];
rz(0.015625*pi) node[2];
rz(1.0*pi) node[14];
cx node[1],node[0];
cx node[3],node[2];
cx node[18],node[14];
rz(3.9375*pi) node[0];
rz(3.984375*pi) node[2];
cx node[14],node[18];
cx node[1],node[0];
cx node[3],node[2];
cx node[18],node[14];
cx node[0],node[1];
cx node[2],node[3];
cx node[1],node[0];
cx node[3],node[2];
cx node[0],node[14];
cx node[2],node[1];
rz(0.03125*pi) node[1];
rz(0.125*pi) node[14];
cx node[0],node[14];
cx node[2],node[1];
rz(3.96875*pi) node[1];
rz(3.875*pi) node[14];
cx node[0],node[14];
cx node[2],node[1];
cx node[14],node[0];
cx node[1],node[2];
cx node[0],node[14];
cx node[2],node[1];
cx node[1],node[0];
cx node[14],node[18];
rz(0.0625*pi) node[0];
rz(0.25*pi) node[18];
cx node[1],node[0];
cx node[14],node[18];
rz(3.9375*pi) node[0];
rz(0.5*pi) node[14];
rz(3.75*pi) node[18];
cx node[1],node[0];
sx node[14];
cx node[0],node[1];
rz(3.5*pi) node[14];
cx node[1],node[0];
sx node[14];
rz(1.0*pi) node[14];
cx node[0],node[14];
cx node[14],node[0];
cx node[0],node[14];
cx node[14],node[18];
rz(0.125*pi) node[18];
cx node[14],node[18];
cx node[14],node[0];
rz(3.875*pi) node[18];
rz(0.25*pi) node[0];
cx node[14],node[0];
rz(3.75*pi) node[0];
rz(0.5*pi) node[14];
sx node[14];
rz(3.5*pi) node[14];
sx node[14];
rz(1.0*pi) node[14];
barrier node[19],node[23],node[20],node[21],node[22],node[4],node[15],node[7],node[6],node[5],node[3],node[2],node[1],node[18],node[0],node[14],node[33];
measure node[19] -> c[0];
measure node[23] -> c[1];
measure node[20] -> c[2];
measure node[21] -> c[3];
measure node[22] -> c[4];
measure node[4] -> c[5];
measure node[15] -> c[6];
measure node[7] -> c[7];
measure node[6] -> c[8];
measure node[5] -> c[9];
measure node[3] -> c[10];
measure node[2] -> c[11];
measure node[1] -> c[12];
measure node[18] -> c[13];
measure node[0] -> c[14];
measure node[14] -> c[15];
