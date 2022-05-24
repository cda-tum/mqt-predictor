OPENQASM 2.0;
include "qelib1.inc";

qreg node[15];
creg meas[11];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[8];
sx node[9];
sx node[11];
rz(0.5*pi) node[13];
sx node[14];
rz(3.2266869758318837*pi) node[0];
rz(3.0389084553335515*pi) node[1];
rz(3.037225109536415*pi) node[2];
rz(3.058395284349395*pi) node[3];
rz(3.007210539683948*pi) node[4];
rz(3.0264780237512716*pi) node[5];
rz(3.20522647034908*pi) node[8];
rz(3.1700072261469474*pi) node[9];
rz(3.2095845208417577*pi) node[11];
sx node[13];
rz(3.0002631166502414*pi) node[14];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[8];
sx node[9];
sx node[11];
rz(3.5*pi) node[13];
sx node[14];
rz(1.0*pi) node[0];
rz(1.0*pi) node[1];
rz(1.0*pi) node[2];
rz(1.0*pi) node[3];
rz(1.0*pi) node[4];
rz(1.0*pi) node[5];
rz(1.0*pi) node[8];
rz(1.0*pi) node[9];
rz(1.0*pi) node[11];
sx node[13];
rz(1.0*pi) node[14];
cx node[1],node[0];
rz(0.762011044047092*pi) node[13];
cx node[1],node[2];
cx node[1],node[4];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[0],node[1];
cx node[2],node[3];
cx node[0],node[1];
cx node[2],node[3];
cx node[1],node[0];
cx node[3],node[2];
cx node[0],node[1];
cx node[2],node[3];
cx node[1],node[4];
cx node[3],node[5];
cx node[1],node[2];
cx node[3],node[5];
cx node[1],node[2];
cx node[5],node[3];
cx node[2],node[1];
cx node[3],node[5];
cx node[1],node[2];
cx node[5],node[8];
cx node[0],node[1];
cx node[2],node[3];
cx node[8],node[5];
cx node[1],node[0];
cx node[2],node[3];
cx node[5],node[8];
cx node[0],node[1];
cx node[3],node[2];
cx node[8],node[5];
cx node[1],node[4];
cx node[2],node[3];
cx node[8],node[11];
cx node[1],node[0];
cx node[3],node[5];
cx node[8],node[9];
cx node[1],node[2];
cx node[3],node[5];
cx node[11],node[8];
cx node[1],node[2];
cx node[5],node[3];
cx node[8],node[11];
cx node[2],node[1];
cx node[3],node[5];
cx node[11],node[8];
cx node[1],node[2];
cx node[5],node[8];
cx node[11],node[14];
cx node[4],node[1];
cx node[2],node[3];
cx node[5],node[8];
cx node[14],node[11];
cx node[1],node[4];
cx node[2],node[3];
cx node[8],node[5];
cx node[11],node[14];
cx node[4],node[1];
cx node[3],node[2];
cx node[5],node[8];
cx node[14],node[11];
cx node[1],node[0];
cx node[2],node[3];
cx node[8],node[9];
cx node[14],node[13];
cx node[1],node[4];
cx node[3],node[5];
cx node[8],node[11];
sx node[14];
cx node[1],node[2];
cx node[3],node[5];
cx node[8],node[11];
rz(3.273983107441757*pi) node[14];
cx node[1],node[2];
cx node[5],node[3];
cx node[11],node[8];
sx node[14];
cx node[2],node[1];
cx node[3],node[5];
cx node[8],node[11];
rz(1.0*pi) node[14];
cx node[1],node[2];
cx node[5],node[8];
cx node[13],node[14];
cx node[0],node[1];
cx node[2],node[3];
cx node[8],node[5];
cx node[14],node[13];
cx node[1],node[0];
cx node[2],node[3];
cx node[5],node[8];
cx node[13],node[14];
cx node[0],node[1];
cx node[3],node[2];
cx node[8],node[9];
cx node[11],node[14];
cx node[1],node[4];
cx node[2],node[3];
cx node[8],node[5];
sx node[11];
cx node[1],node[0];
cx node[3],node[5];
rz(3.1318389075633477*pi) node[11];
cx node[1],node[2];
cx node[5],node[3];
sx node[11];
cx node[1],node[2];
cx node[3],node[5];
rz(1.0*pi) node[11];
cx node[2],node[1];
cx node[14],node[11];
cx node[1],node[2];
cx node[11],node[14];
cx node[4],node[1];
cx node[14],node[11];
cx node[1],node[4];
cx node[8],node[11];
cx node[13],node[14];
cx node[4],node[1];
sx node[8];
cx node[13],node[14];
cx node[1],node[0];
rz(3.2071168341329144*pi) node[8];
cx node[14],node[13];
cx node[1],node[4];
sx node[8];
cx node[13],node[14];
rz(1.0*pi) node[8];
cx node[8],node[11];
cx node[11],node[8];
cx node[8],node[11];
cx node[9],node[8];
cx node[14],node[11];
cx node[8],node[9];
cx node[14],node[11];
cx node[9],node[8];
cx node[11],node[14];
cx node[5],node[8];
cx node[14],node[11];
cx node[5],node[3];
cx node[13],node[14];
cx node[2],node[3];
cx node[8],node[5];
cx node[13],node[14];
cx node[3],node[2];
cx node[5],node[8];
cx node[14],node[13];
cx node[2],node[3];
cx node[8],node[5];
cx node[13],node[14];
cx node[3],node[5];
cx node[8],node[9];
cx node[3],node[2];
sx node[8];
cx node[3],node[5];
rz(3.0727579259845834*pi) node[8];
cx node[5],node[3];
sx node[8];
cx node[3],node[5];
rz(1.0*pi) node[8];
cx node[3],node[2];
cx node[11],node[8];
cx node[2],node[3];
cx node[8],node[11];
cx node[3],node[2];
cx node[11],node[8];
cx node[1],node[2];
cx node[8],node[11];
cx node[1],node[2];
cx node[5],node[8];
cx node[14],node[11];
cx node[2],node[1];
cx node[8],node[5];
cx node[14],node[11];
cx node[1],node[2];
cx node[5],node[8];
cx node[11],node[14];
cx node[0],node[1];
cx node[2],node[3];
cx node[8],node[9];
cx node[14],node[11];
cx node[1],node[0];
cx node[2],node[3];
sx node[8];
cx node[13],node[14];
cx node[0],node[1];
cx node[3],node[2];
rz(3.2747386778866203*pi) node[8];
cx node[13],node[14];
cx node[1],node[4];
cx node[2],node[3];
sx node[8];
cx node[14],node[13];
cx node[1],node[0];
rz(1.0*pi) node[8];
cx node[13],node[14];
cx node[1],node[2];
cx node[5],node[8];
cx node[1],node[2];
cx node[3],node[5];
cx node[11],node[8];
cx node[2],node[1];
cx node[5],node[3];
cx node[8],node[11];
cx node[1],node[2];
cx node[3],node[5];
cx node[11],node[8];
cx node[4],node[1];
cx node[8],node[11];
cx node[1],node[4];
cx node[9],node[8];
cx node[14],node[11];
cx node[4],node[1];
cx node[8],node[9];
cx node[14],node[11];
cx node[1],node[0];
cx node[9],node[8];
cx node[11],node[14];
cx node[1],node[4];
cx node[5],node[8];
cx node[14],node[11];
sx node[5];
cx node[13],node[14];
rz(3.230850023964508*pi) node[5];
cx node[13],node[14];
sx node[5];
cx node[14],node[13];
rz(1.0*pi) node[5];
cx node[13],node[14];
cx node[3],node[5];
cx node[2],node[3];
cx node[8],node[5];
cx node[3],node[2];
cx node[5],node[8];
cx node[2],node[3];
cx node[8],node[5];
cx node[3],node[5];
cx node[9],node[8];
sx node[3];
cx node[11],node[8];
rz(3.0052612218106285*pi) node[3];
cx node[8],node[11];
sx node[3];
cx node[11],node[8];
rz(1.0*pi) node[3];
cx node[8],node[11];
cx node[2],node[3];
cx node[14],node[11];
cx node[3],node[5];
cx node[14],node[11];
cx node[5],node[3];
cx node[11],node[14];
cx node[3],node[5];
cx node[14],node[11];
cx node[3],node[2];
cx node[5],node[8];
cx node[13],node[14];
cx node[2],node[3];
cx node[8],node[5];
cx node[13],node[14];
cx node[3],node[2];
cx node[5],node[8];
cx node[14],node[13];
cx node[1],node[2];
cx node[9],node[8];
cx node[13],node[14];
sx node[1];
cx node[5],node[8];
rz(3.219640754076898*pi) node[1];
cx node[11],node[8];
sx node[1];
cx node[8],node[11];
rz(1.0*pi) node[1];
cx node[11],node[8];
cx node[1],node[2];
cx node[8],node[11];
cx node[2],node[1];
cx node[9],node[8];
cx node[14],node[11];
cx node[1],node[2];
cx node[8],node[9];
cx node[14],node[11];
cx node[0],node[1];
cx node[3],node[2];
cx node[9],node[8];
cx node[11],node[14];
cx node[1],node[0];
cx node[2],node[3];
cx node[14],node[11];
cx node[0],node[1];
cx node[3],node[2];
cx node[13],node[14];
cx node[1],node[4];
cx node[2],node[3];
cx node[13],node[14];
cx node[1],node[0];
cx node[3],node[5];
cx node[14],node[13];
sx node[1];
cx node[5],node[3];
cx node[13],node[14];
rz(3.127279711875553*pi) node[1];
cx node[3],node[5];
sx node[1];
cx node[8],node[5];
rz(1.0*pi) node[1];
cx node[3],node[5];
cx node[2],node[1];
cx node[8],node[5];
cx node[1],node[2];
cx node[5],node[8];
cx node[2],node[1];
cx node[8],node[5];
cx node[1],node[2];
cx node[9],node[8];
cx node[4],node[1];
cx node[2],node[3];
cx node[11],node[8];
cx node[1],node[4];
cx node[3],node[2];
cx node[8],node[11];
cx node[4],node[1];
cx node[2],node[3];
cx node[11],node[8];
cx node[1],node[0];
cx node[5],node[3];
cx node[8],node[11];
rz(0.07055486090008123*pi) node[0];
sx node[1];
cx node[2],node[3];
cx node[14],node[11];
rz(3.1102867234865927*pi) node[1];
cx node[3],node[5];
cx node[14],node[11];
sx node[1];
cx node[5],node[3];
cx node[11],node[14];
rz(1.0*pi) node[1];
cx node[3],node[5];
cx node[14],node[11];
cx node[4],node[1];
cx node[3],node[2];
cx node[5],node[8];
cx node[13],node[14];
cx node[2],node[3];
cx node[8],node[5];
cx node[13],node[14];
cx node[3],node[2];
cx node[5],node[8];
cx node[14],node[13];
cx node[2],node[1];
cx node[9],node[8];
cx node[13],node[14];
cx node[1],node[2];
cx node[5],node[8];
cx node[2],node[1];
cx node[11],node[8];
cx node[1],node[2];
cx node[8],node[11];
cx node[0],node[1];
cx node[3],node[2];
cx node[11],node[8];
cx node[1],node[0];
cx node[2],node[3];
cx node[8],node[11];
cx node[0],node[1];
cx node[3],node[2];
cx node[9],node[8];
cx node[14],node[11];
cx node[4],node[1];
cx node[2],node[3];
cx node[8],node[9];
cx node[14],node[11];
cx node[0],node[1];
cx node[3],node[5];
sx node[4];
cx node[9],node[8];
cx node[11],node[14];
sx node[0];
cx node[2],node[1];
cx node[5],node[3];
rz(3.090968371319463*pi) node[4];
cx node[14],node[11];
rz(3.2084782420851594*pi) node[0];
sx node[2];
cx node[3],node[5];
sx node[4];
cx node[13],node[14];
sx node[0];
rz(3.167385089255107*pi) node[2];
rz(1.0*pi) node[4];
cx node[8],node[5];
cx node[13],node[14];
rz(1.0*pi) node[0];
sx node[2];
cx node[3],node[5];
cx node[14],node[13];
rz(1.0*pi) node[2];
cx node[8],node[5];
cx node[13],node[14];
cx node[1],node[2];
cx node[5],node[8];
cx node[2],node[1];
cx node[8],node[5];
cx node[1],node[2];
cx node[9],node[8];
cx node[4],node[1];
cx node[2],node[3];
cx node[11],node[8];
cx node[1],node[4];
cx node[3],node[2];
cx node[8],node[11];
cx node[4],node[1];
cx node[2],node[3];
cx node[11],node[8];
cx node[1],node[0];
cx node[5],node[3];
cx node[8],node[11];
cx node[1],node[4];
cx node[2],node[3];
sx node[5];
cx node[14],node[11];
sx node[2];
rz(3.1919364622215385*pi) node[5];
cx node[14],node[11];
rz(3.18068207245864*pi) node[2];
sx node[5];
cx node[11],node[14];
sx node[2];
rz(1.0*pi) node[5];
cx node[14],node[11];
rz(1.0*pi) node[2];
cx node[3],node[5];
cx node[13],node[14];
cx node[5],node[3];
cx node[13],node[14];
cx node[3],node[5];
cx node[14],node[13];
cx node[3],node[2];
cx node[5],node[8];
cx node[13],node[14];
cx node[2],node[3];
cx node[8],node[5];
cx node[3],node[2];
cx node[5],node[8];
cx node[1],node[2];
cx node[9],node[8];
cx node[1],node[2];
cx node[5],node[8];
sx node[9];
cx node[2],node[1];
sx node[5];
cx node[11],node[8];
rz(3.2241042207932313*pi) node[9];
cx node[1],node[2];
rz(3.082734817446464*pi) node[5];
sx node[9];
sx node[11];
cx node[0],node[1];
cx node[2],node[3];
sx node[5];
rz(1.0*pi) node[9];
rz(3.1516101942084718*pi) node[11];
cx node[1],node[0];
cx node[2],node[3];
rz(1.0*pi) node[5];
sx node[11];
cx node[0],node[1];
cx node[3],node[2];
rz(1.0*pi) node[11];
cx node[1],node[4];
cx node[2],node[3];
cx node[8],node[11];
cx node[1],node[0];
cx node[3],node[5];
cx node[11],node[8];
cx node[1],node[2];
cx node[5],node[3];
cx node[8],node[11];
cx node[1],node[2];
cx node[3],node[5];
cx node[9],node[8];
cx node[14],node[11];
cx node[2],node[1];
cx node[8],node[9];
sx node[14];
cx node[1],node[2];
cx node[9],node[8];
rz(3.179682582234809*pi) node[14];
cx node[4],node[1];
cx node[5],node[8];
sx node[14];
cx node[1],node[4];
cx node[5],node[3];
rz(1.0*pi) node[14];
cx node[4],node[1];
cx node[2],node[3];
cx node[8],node[5];
cx node[14],node[11];
cx node[1],node[0];
cx node[3],node[2];
cx node[5],node[8];
cx node[11],node[14];
cx node[1],node[4];
cx node[2],node[3];
cx node[8],node[5];
cx node[14],node[11];
cx node[3],node[5];
cx node[8],node[9];
cx node[13],node[14];
cx node[3],node[2];
cx node[8],node[11];
sx node[13];
rz(0.15704539418662677*pi) node[14];
cx node[3],node[5];
cx node[8],node[11];
rz(3.182436465173331*pi) node[13];
cx node[5],node[3];
cx node[11],node[8];
sx node[13];
cx node[3],node[5];
cx node[8],node[11];
rz(1.0*pi) node[13];
cx node[3],node[2];
cx node[5],node[8];
cx node[13],node[14];
cx node[2],node[3];
cx node[8],node[5];
cx node[14],node[13];
cx node[3],node[2];
cx node[5],node[8];
cx node[13],node[14];
cx node[1],node[2];
cx node[8],node[9];
cx node[11],node[14];
cx node[1],node[2];
cx node[8],node[5];
cx node[14],node[11];
cx node[2],node[1];
cx node[11],node[14];
cx node[1],node[2];
cx node[14],node[11];
cx node[0],node[1];
cx node[2],node[3];
cx node[8],node[11];
cx node[14],node[13];
cx node[1],node[0];
cx node[2],node[3];
cx node[8],node[11];
sx node[14];
cx node[0],node[1];
cx node[3],node[2];
cx node[11],node[8];
rz(3.006303155102227*pi) node[14];
cx node[1],node[4];
cx node[2],node[3];
cx node[8],node[11];
sx node[14];
cx node[1],node[0];
cx node[3],node[5];
cx node[9],node[8];
rz(1.0*pi) node[14];
cx node[1],node[2];
cx node[5],node[3];
cx node[8],node[9];
cx node[13],node[14];
cx node[1],node[2];
cx node[3],node[5];
cx node[9],node[8];
cx node[14],node[13];
cx node[2],node[1];
cx node[5],node[8];
cx node[13],node[14];
cx node[1],node[2];
cx node[5],node[3];
cx node[11],node[14];
cx node[4],node[1];
cx node[2],node[3];
cx node[8],node[5];
sx node[11];
cx node[1],node[4];
cx node[3],node[2];
cx node[5],node[8];
rz(3.3171169194754997*pi) node[11];
cx node[4],node[1];
cx node[2],node[3];
cx node[8],node[5];
sx node[11];
cx node[1],node[0];
cx node[3],node[5];
cx node[8],node[9];
rz(1.0*pi) node[11];
cx node[1],node[4];
cx node[3],node[2];
cx node[14],node[11];
cx node[3],node[5];
cx node[11],node[14];
cx node[5],node[3];
cx node[14],node[11];
cx node[3],node[5];
cx node[8],node[11];
cx node[3],node[2];
sx node[8];
cx node[2],node[3];
rz(3.1218279720273276*pi) node[8];
cx node[3],node[2];
sx node[8];
cx node[1],node[2];
rz(1.0*pi) node[8];
cx node[1],node[2];
cx node[5],node[8];
cx node[2],node[1];
cx node[8],node[5];
cx node[1],node[2];
cx node[5],node[8];
cx node[0],node[1];
cx node[2],node[3];
cx node[8],node[9];
cx node[1],node[0];
cx node[2],node[3];
cx node[8],node[11];
cx node[0],node[1];
cx node[3],node[2];
sx node[8];
cx node[1],node[4];
cx node[2],node[3];
rz(3.073555577725561*pi) node[8];
cx node[1],node[0];
cx node[3],node[5];
sx node[8];
cx node[1],node[2];
cx node[5],node[3];
rz(1.0*pi) node[8];
cx node[1],node[2];
cx node[3],node[5];
cx node[2],node[1];
cx node[5],node[8];
cx node[1],node[2];
cx node[8],node[5];
cx node[4],node[1];
cx node[2],node[3];
cx node[5],node[8];
cx node[1],node[4];
cx node[3],node[2];
cx node[8],node[9];
cx node[4],node[1];
cx node[2],node[3];
cx node[8],node[11];
cx node[1],node[0];
cx node[3],node[5];
sx node[8];
cx node[1],node[4];
cx node[5],node[3];
rz(3.30037509443268*pi) node[8];
cx node[1],node[2];
cx node[3],node[5];
sx node[8];
cx node[2],node[1];
rz(1.0*pi) node[8];
cx node[1],node[2];
cx node[5],node[8];
cx node[0],node[1];
cx node[2],node[3];
cx node[8],node[5];
cx node[1],node[0];
cx node[3],node[2];
cx node[5],node[8];
cx node[0],node[1];
cx node[2],node[3];
cx node[8],node[9];
cx node[1],node[4];
cx node[8],node[11];
cx node[1],node[2];
sx node[8];
cx node[2],node[1];
rz(3.063856439501428*pi) node[8];
cx node[1],node[2];
sx node[8];
cx node[4],node[1];
rz(1.0*pi) node[8];
cx node[1],node[4];
cx node[9],node[8];
cx node[4],node[1];
cx node[8],node[9];
cx node[9],node[8];
cx node[8],node[5];
cx node[5],node[8];
cx node[8],node[5];
cx node[3],node[5];
cx node[11],node[8];
cx node[5],node[3];
cx node[8],node[11];
cx node[3],node[5];
cx node[11],node[8];
cx node[5],node[3];
cx node[2],node[3];
cx node[5],node[8];
cx node[2],node[3];
sx node[5];
cx node[3],node[2];
rz(3.0676886311267575*pi) node[5];
cx node[2],node[3];
sx node[5];
cx node[1],node[2];
rz(1.0*pi) node[5];
cx node[8],node[5];
cx node[5],node[8];
cx node[8],node[5];
cx node[3],node[5];
sx node[3];
rz(3.111123324027308*pi) node[3];
sx node[3];
rz(1.0*pi) node[3];
cx node[5],node[3];
cx node[3],node[5];
cx node[5],node[3];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[1],node[2];
sx node[1];
cx node[3],node[2];
rz(3.1999631767563086*pi) node[1];
rz(3.5707609837463643*pi) node[2];
sx node[3];
sx node[1];
sx node[2];
rz(3.2783864787278083*pi) node[3];
rz(1.0*pi) node[1];
rz(3.5*pi) node[2];
sx node[3];
sx node[2];
rz(1.0*pi) node[3];
rz(1.5*pi) node[2];
barrier node[13],node[14],node[0],node[4],node[11],node[9],node[8],node[5],node[1],node[3],node[2];
measure node[13] -> meas[0];
measure node[14] -> meas[1];
measure node[0] -> meas[2];
measure node[4] -> meas[3];
measure node[11] -> meas[4];
measure node[9] -> meas[5];
measure node[8] -> meas[6];
measure node[5] -> meas[7];
measure node[1] -> meas[8];
measure node[3] -> meas[9];
measure node[2] -> meas[10];
