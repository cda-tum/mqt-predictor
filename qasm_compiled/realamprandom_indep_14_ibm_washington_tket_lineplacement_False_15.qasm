OPENQASM 2.0;
include "qelib1.inc";

qreg node[34];
creg meas[14];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[14];
sx node[15];
sx node[19];
sx node[20];
sx node[21];
sx node[22];
sx node[23];
rz(0.5*pi) node[33];
rz(3.019666454602864*pi) node[0];
rz(3.0769341705952984*pi) node[1];
rz(3.1835296533569992*pi) node[2];
rz(3.177898168541351*pi) node[3];
rz(3.0988431671806955*pi) node[4];
rz(3.165792781563*pi) node[5];
rz(3.2041080820895567*pi) node[14];
rz(3.0334520949909973*pi) node[15];
rz(3.1511840955638073*pi) node[19];
rz(3.0571992806701687*pi) node[20];
rz(3.307526009932718*pi) node[21];
rz(3.30431206309885*pi) node[22];
rz(3.3109572708399813*pi) node[23];
sx node[33];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[14];
sx node[15];
sx node[19];
sx node[20];
sx node[21];
sx node[22];
sx node[23];
rz(3.5*pi) node[33];
rz(1.0*pi) node[0];
rz(1.0*pi) node[1];
rz(1.0*pi) node[2];
rz(1.0*pi) node[3];
rz(1.0*pi) node[4];
rz(1.0*pi) node[5];
rz(1.0*pi) node[14];
rz(1.0*pi) node[15];
rz(1.0*pi) node[19];
rz(1.0*pi) node[20];
rz(1.0*pi) node[21];
rz(1.0*pi) node[22];
rz(1.0*pi) node[23];
sx node[33];
cx node[0],node[1];
rz(0.6029585407020154*pi) node[33];
cx node[0],node[14];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[0],node[14];
cx node[1],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[0],node[1];
cx node[2],node[3];
cx node[0],node[1];
cx node[3],node[2];
cx node[1],node[0];
cx node[2],node[3];
cx node[0],node[1];
cx node[3],node[2];
cx node[14],node[0];
cx node[1],node[2];
cx node[3],node[4];
cx node[14],node[0];
cx node[1],node[2];
cx node[4],node[3];
cx node[0],node[14];
cx node[2],node[1];
cx node[3],node[4];
cx node[14],node[0];
cx node[1],node[2];
cx node[4],node[3];
cx node[0],node[1];
cx node[2],node[3];
cx node[4],node[5];
cx node[0],node[1];
cx node[2],node[3];
cx node[4],node[15];
cx node[1],node[0];
cx node[3],node[2];
cx node[4],node[15];
cx node[0],node[1];
cx node[2],node[3];
cx node[15],node[4];
cx node[14],node[0];
cx node[1],node[2];
cx node[4],node[15];
cx node[14],node[0];
cx node[2],node[1];
cx node[5],node[4];
cx node[15],node[22];
cx node[0],node[14];
cx node[1],node[2];
cx node[4],node[5];
cx node[22],node[15];
cx node[14],node[0];
cx node[2],node[1];
cx node[5],node[4];
cx node[15],node[22];
cx node[0],node[1];
cx node[3],node[4];
cx node[22],node[15];
cx node[0],node[1];
cx node[4],node[3];
cx node[22],node[21];
cx node[1],node[0];
cx node[3],node[4];
cx node[22],node[23];
cx node[0],node[1];
cx node[4],node[3];
cx node[21],node[22];
cx node[14],node[0];
cx node[2],node[3];
cx node[4],node[5];
cx node[22],node[21];
cx node[14],node[0];
cx node[2],node[3];
cx node[4],node[15];
cx node[21],node[22];
cx node[0],node[14];
cx node[3],node[2];
cx node[4],node[15];
cx node[21],node[20];
cx node[14],node[0];
cx node[2],node[3];
cx node[15],node[4];
cx node[21],node[20];
cx node[1],node[2];
cx node[4],node[15];
cx node[20],node[21];
cx node[2],node[1];
cx node[5],node[4];
cx node[15],node[22];
cx node[21],node[20];
cx node[1],node[2];
cx node[4],node[5];
cx node[15],node[22];
cx node[20],node[19];
cx node[2],node[1];
cx node[5],node[4];
cx node[22],node[15];
cx node[20],node[33];
cx node[0],node[1];
cx node[3],node[4];
cx node[15],node[22];
sx node[20];
cx node[0],node[1];
cx node[4],node[3];
rz(3.206896711211086*pi) node[20];
cx node[22],node[23];
cx node[1],node[0];
cx node[3],node[4];
sx node[20];
cx node[22],node[21];
cx node[0],node[1];
cx node[4],node[3];
rz(1.0*pi) node[20];
cx node[22],node[21];
cx node[14],node[0];
cx node[2],node[3];
cx node[4],node[5];
cx node[19],node[20];
cx node[21],node[22];
cx node[14],node[0];
cx node[2],node[3];
cx node[4],node[15];
cx node[20],node[19];
cx node[22],node[21];
cx node[0],node[14];
cx node[3],node[2];
cx node[4],node[15];
cx node[19],node[20];
cx node[23],node[22];
cx node[14],node[0];
cx node[2],node[3];
cx node[15],node[4];
cx node[21],node[20];
cx node[22],node[23];
cx node[1],node[2];
cx node[4],node[15];
cx node[20],node[21];
cx node[23],node[22];
cx node[2],node[1];
cx node[5],node[4];
cx node[15],node[22];
cx node[21],node[20];
cx node[1],node[2];
cx node[4],node[5];
cx node[22],node[15];
cx node[20],node[21];
cx node[2],node[1];
cx node[5],node[4];
cx node[15],node[22];
cx node[20],node[33];
cx node[0],node[1];
cx node[3],node[4];
cx node[22],node[15];
sx node[20];
cx node[0],node[1];
cx node[4],node[3];
rz(3.2797730192409933*pi) node[20];
cx node[22],node[23];
cx node[1],node[0];
cx node[3],node[4];
sx node[20];
cx node[22],node[21];
cx node[0],node[1];
cx node[4],node[3];
rz(1.0*pi) node[20];
cx node[22],node[21];
cx node[14],node[0];
cx node[2],node[3];
cx node[4],node[5];
cx node[19],node[20];
cx node[21],node[22];
cx node[14],node[0];
cx node[2],node[3];
cx node[4],node[15];
cx node[33],node[20];
cx node[22],node[21];
cx node[0],node[14];
cx node[3],node[2];
cx node[4],node[15];
cx node[20],node[33];
cx node[23],node[22];
cx node[14],node[0];
cx node[2],node[3];
cx node[15],node[4];
cx node[33],node[20];
cx node[22],node[23];
cx node[1],node[2];
cx node[4],node[15];
cx node[21],node[20];
cx node[23],node[22];
cx node[2],node[1];
cx node[5],node[4];
cx node[15],node[22];
sx node[21];
cx node[1],node[2];
cx node[4],node[5];
cx node[22],node[15];
rz(3.070026528747442*pi) node[21];
cx node[2],node[1];
cx node[5],node[4];
cx node[15],node[22];
sx node[21];
cx node[0],node[1];
cx node[3],node[4];
cx node[22],node[15];
rz(1.0*pi) node[21];
cx node[0],node[1];
cx node[4],node[3];
cx node[21],node[20];
cx node[22],node[23];
cx node[1],node[0];
cx node[3],node[4];
cx node[20],node[21];
cx node[0],node[1];
cx node[4],node[3];
cx node[21],node[20];
cx node[14],node[0];
cx node[2],node[3];
cx node[4],node[5];
cx node[19],node[20];
cx node[22],node[21];
cx node[14],node[0];
cx node[2],node[3];
cx node[4],node[15];
cx node[33],node[20];
sx node[22];
cx node[0],node[14];
cx node[3],node[2];
cx node[4],node[15];
cx node[19],node[20];
rz(3.161540576967364*pi) node[22];
cx node[14],node[0];
cx node[2],node[3];
cx node[15],node[4];
cx node[20],node[19];
sx node[22];
cx node[1],node[2];
cx node[4],node[15];
cx node[19],node[20];
rz(1.0*pi) node[22];
cx node[2],node[1];
cx node[5],node[4];
cx node[22],node[21];
cx node[1],node[2];
cx node[4],node[5];
cx node[21],node[22];
cx node[2],node[1];
cx node[5],node[4];
cx node[22],node[21];
cx node[0],node[1];
cx node[3],node[4];
cx node[20],node[21];
cx node[23],node[22];
cx node[0],node[1];
cx node[4],node[3];
cx node[20],node[21];
cx node[22],node[23];
cx node[1],node[0];
cx node[3],node[4];
cx node[21],node[20];
cx node[23],node[22];
cx node[0],node[1];
cx node[4],node[3];
cx node[15],node[22];
cx node[20],node[21];
cx node[14],node[0];
cx node[2],node[3];
cx node[4],node[5];
cx node[22],node[15];
cx node[33],node[20];
cx node[14],node[0];
cx node[2],node[3];
cx node[15],node[22];
cx node[19],node[20];
cx node[0],node[14];
cx node[3],node[2];
cx node[22],node[15];
cx node[33],node[20];
cx node[14],node[0];
cx node[2],node[3];
cx node[4],node[15];
cx node[20],node[33];
cx node[22],node[23];
cx node[1],node[2];
cx node[4],node[15];
cx node[33],node[20];
sx node[22];
cx node[2],node[1];
cx node[15],node[4];
rz(3.265964554444177*pi) node[22];
cx node[1],node[2];
cx node[4],node[15];
sx node[22];
cx node[2],node[1];
cx node[5],node[4];
rz(1.0*pi) node[22];
cx node[0],node[1];
cx node[4],node[5];
cx node[21],node[22];
cx node[0],node[1];
cx node[5],node[4];
cx node[22],node[21];
cx node[1],node[0];
cx node[3],node[4];
cx node[21],node[22];
cx node[0],node[1];
cx node[4],node[3];
cx node[22],node[21];
cx node[14],node[0];
cx node[3],node[4];
cx node[20],node[21];
cx node[23],node[22];
cx node[14],node[0];
cx node[4],node[3];
cx node[21],node[20];
cx node[22],node[23];
cx node[0],node[14];
cx node[2],node[3];
cx node[4],node[5];
cx node[20],node[21];
cx node[23],node[22];
cx node[14],node[0];
cx node[2],node[3];
cx node[15],node[22];
cx node[21],node[20];
cx node[3],node[2];
sx node[15];
cx node[19],node[20];
cx node[2],node[3];
rz(3.2772202367688696*pi) node[15];
cx node[33],node[20];
cx node[1],node[2];
sx node[15];
cx node[19],node[20];
cx node[2],node[1];
rz(1.0*pi) node[15];
cx node[20],node[19];
cx node[1],node[2];
cx node[22],node[15];
cx node[19],node[20];
cx node[2],node[1];
cx node[15],node[22];
cx node[0],node[1];
cx node[22],node[15];
cx node[0],node[1];
cx node[4],node[15];
cx node[23],node[22];
cx node[1],node[0];
sx node[4];
cx node[21],node[22];
cx node[0],node[1];
rz(3.2120671969264483*pi) node[4];
cx node[22],node[21];
cx node[14],node[0];
sx node[4];
cx node[21],node[22];
cx node[14],node[0];
rz(1.0*pi) node[4];
cx node[22],node[21];
cx node[0],node[14];
cx node[4],node[15];
cx node[20],node[21];
cx node[23],node[22];
cx node[14],node[0];
cx node[15],node[4];
cx node[20],node[21];
cx node[22],node[23];
cx node[4],node[15];
cx node[21],node[20];
cx node[23],node[22];
cx node[5],node[4];
cx node[22],node[15];
cx node[20],node[21];
cx node[4],node[5];
cx node[22],node[15];
cx node[33],node[20];
cx node[5],node[4];
cx node[15],node[22];
cx node[19],node[20];
cx node[3],node[4];
cx node[22],node[15];
cx node[33],node[20];
cx node[4],node[3];
cx node[20],node[33];
cx node[23],node[22];
cx node[3],node[4];
cx node[33],node[20];
cx node[21],node[22];
cx node[4],node[3];
cx node[22],node[21];
cx node[2],node[3];
cx node[4],node[5];
cx node[21],node[22];
cx node[2],node[3];
sx node[4];
cx node[22],node[21];
cx node[3],node[2];
rz(3.0123191793914295*pi) node[4];
cx node[20],node[21];
cx node[23],node[22];
cx node[2],node[3];
sx node[4];
cx node[21],node[20];
cx node[22],node[23];
cx node[1],node[2];
rz(1.0*pi) node[4];
cx node[20],node[21];
cx node[23],node[22];
cx node[2],node[1];
cx node[15],node[4];
cx node[21],node[20];
cx node[1],node[2];
cx node[4],node[15];
cx node[19],node[20];
cx node[2],node[1];
cx node[15],node[4];
cx node[33],node[20];
cx node[0],node[1];
cx node[4],node[15];
cx node[19],node[20];
cx node[0],node[1];
cx node[5],node[4];
cx node[22],node[15];
cx node[20],node[19];
cx node[1],node[0];
cx node[4],node[5];
cx node[22],node[15];
cx node[19],node[20];
cx node[0],node[1];
cx node[5],node[4];
cx node[15],node[22];
cx node[14],node[0];
cx node[3],node[4];
cx node[22],node[15];
cx node[14],node[0];
sx node[3];
cx node[23],node[22];
cx node[0],node[14];
rz(3.038463658496235*pi) node[3];
cx node[21],node[22];
cx node[14],node[0];
sx node[3];
cx node[22],node[21];
rz(1.0*pi) node[3];
cx node[21],node[22];
cx node[4],node[3];
cx node[22],node[21];
cx node[3],node[4];
cx node[20],node[21];
cx node[23],node[22];
cx node[4],node[3];
cx node[20],node[21];
cx node[22],node[23];
cx node[2],node[3];
cx node[5],node[4];
cx node[21],node[20];
cx node[23],node[22];
sx node[2];
cx node[15],node[4];
cx node[20],node[21];
rz(3.075865932776882*pi) node[2];
cx node[4],node[15];
cx node[33],node[20];
sx node[2];
cx node[15],node[4];
cx node[19],node[20];
rz(1.0*pi) node[2];
cx node[4],node[15];
cx node[33],node[20];
cx node[2],node[3];
cx node[5],node[4];
cx node[22],node[15];
cx node[20],node[33];
cx node[3],node[2];
cx node[4],node[5];
cx node[22],node[15];
cx node[33],node[20];
cx node[2],node[3];
cx node[5],node[4];
cx node[15],node[22];
cx node[1],node[2];
cx node[4],node[3];
cx node[22],node[15];
sx node[1];
cx node[4],node[3];
cx node[23],node[22];
rz(3.236949062322827*pi) node[1];
cx node[3],node[4];
cx node[21],node[22];
sx node[1];
cx node[4],node[3];
cx node[22],node[21];
rz(1.0*pi) node[1];
cx node[5],node[4];
cx node[21],node[22];
cx node[2],node[1];
cx node[15],node[4];
cx node[22],node[21];
cx node[1],node[2];
cx node[4],node[15];
cx node[20],node[21];
cx node[23],node[22];
cx node[2],node[1];
cx node[15],node[4];
cx node[21],node[20];
cx node[22],node[23];
cx node[0],node[1];
cx node[3],node[2];
cx node[4],node[15];
cx node[20],node[21];
cx node[23],node[22];
sx node[0];
cx node[2],node[3];
cx node[5],node[4];
cx node[22],node[15];
cx node[21],node[20];
rz(3.07030799484519*pi) node[0];
cx node[3],node[2];
cx node[4],node[5];
cx node[22],node[15];
cx node[19],node[20];
sx node[0];
cx node[2],node[3];
cx node[5],node[4];
cx node[15],node[22];
cx node[33],node[20];
rz(1.0*pi) node[0];
cx node[4],node[3];
cx node[22],node[15];
cx node[19],node[20];
cx node[0],node[1];
cx node[4],node[3];
cx node[20],node[19];
cx node[23],node[22];
cx node[1],node[0];
cx node[3],node[4];
cx node[19],node[20];
cx node[21],node[22];
cx node[0],node[1];
cx node[4],node[3];
cx node[22],node[21];
cx node[14],node[0];
cx node[2],node[1];
cx node[5],node[4];
cx node[21],node[22];
rz(0.17746544700728717*pi) node[0];
cx node[2],node[1];
cx node[15],node[4];
sx node[14];
cx node[22],node[21];
cx node[1],node[2];
cx node[4],node[15];
rz(3.0776963871411893*pi) node[14];
cx node[20],node[21];
cx node[23],node[22];
cx node[2],node[1];
cx node[15],node[4];
sx node[14];
cx node[20],node[21];
cx node[22],node[23];
cx node[3],node[2];
cx node[4],node[15];
rz(1.0*pi) node[14];
cx node[21],node[20];
cx node[23],node[22];
cx node[14],node[0];
cx node[2],node[3];
cx node[5],node[4];
cx node[22],node[15];
cx node[20],node[21];
cx node[0],node[14];
cx node[3],node[2];
cx node[4],node[5];
cx node[22],node[15];
cx node[33],node[20];
cx node[14],node[0];
cx node[2],node[3];
cx node[5],node[4];
cx node[15],node[22];
cx node[19],node[20];
cx node[1],node[0];
cx node[4],node[3];
cx node[22],node[15];
cx node[33],node[20];
cx node[0],node[1];
cx node[4],node[3];
cx node[20],node[33];
cx node[23],node[22];
cx node[1],node[0];
cx node[3],node[4];
cx node[33],node[20];
cx node[21],node[22];
cx node[0],node[1];
cx node[4],node[3];
cx node[22],node[21];
cx node[0],node[14];
cx node[2],node[1];
cx node[5],node[4];
cx node[21],node[22];
sx node[0];
cx node[2],node[1];
cx node[15],node[4];
cx node[22],node[21];
rz(3.2266299336557087*pi) node[0];
cx node[1],node[2];
cx node[4],node[15];
cx node[20],node[21];
cx node[23],node[22];
sx node[0];
cx node[2],node[1];
cx node[15],node[4];
cx node[21],node[20];
cx node[22],node[23];
rz(1.0*pi) node[0];
cx node[3],node[2];
cx node[4],node[15];
cx node[20],node[21];
cx node[23],node[22];
cx node[14],node[0];
cx node[2],node[3];
cx node[5],node[4];
cx node[22],node[15];
cx node[21],node[20];
cx node[0],node[14];
cx node[3],node[2];
cx node[4],node[5];
cx node[22],node[15];
cx node[19],node[20];
cx node[14],node[0];
cx node[2],node[3];
cx node[5],node[4];
cx node[15],node[22];
cx node[33],node[20];
cx node[1],node[0];
cx node[4],node[3];
cx node[22],node[15];
cx node[19],node[20];
sx node[1];
cx node[4],node[3];
cx node[20],node[19];
cx node[23],node[22];
rz(3.255930469546258*pi) node[1];
cx node[3],node[4];
cx node[19],node[20];
cx node[21],node[22];
sx node[1];
cx node[4],node[3];
cx node[22],node[21];
rz(1.0*pi) node[1];
cx node[5],node[4];
cx node[21],node[22];
cx node[0],node[1];
cx node[15],node[4];
cx node[22],node[21];
cx node[1],node[0];
cx node[4],node[15];
cx node[20],node[21];
cx node[23],node[22];
cx node[0],node[1];
cx node[15],node[4];
cx node[20],node[21];
cx node[22],node[23];
cx node[14],node[0];
cx node[2],node[1];
cx node[4],node[15];
cx node[21],node[20];
cx node[23],node[22];
cx node[14],node[0];
sx node[2];
cx node[5],node[4];
cx node[22],node[15];
cx node[20],node[21];
cx node[0],node[14];
rz(3.0523056656769993*pi) node[2];
cx node[4],node[5];
cx node[22],node[15];
cx node[33],node[20];
cx node[14],node[0];
sx node[2];
cx node[5],node[4];
cx node[15],node[22];
cx node[19],node[20];
rz(1.0*pi) node[2];
cx node[22],node[15];
cx node[33],node[20];
cx node[2],node[1];
cx node[20],node[33];
cx node[23],node[22];
cx node[1],node[2];
cx node[33],node[20];
cx node[21],node[22];
cx node[2],node[1];
cx node[22],node[21];
cx node[0],node[1];
cx node[3],node[2];
cx node[21],node[22];
cx node[0],node[1];
sx node[3];
cx node[22],node[21];
cx node[1],node[0];
rz(3.275309972808956*pi) node[3];
cx node[20],node[21];
cx node[23],node[22];
cx node[0],node[1];
sx node[3];
cx node[21],node[20];
cx node[22],node[23];
cx node[14],node[0];
rz(1.0*pi) node[3];
cx node[20],node[21];
cx node[23],node[22];
cx node[14],node[0];
cx node[2],node[3];
cx node[21],node[20];
cx node[0],node[14];
cx node[3],node[2];
cx node[19],node[20];
cx node[14],node[0];
cx node[2],node[3];
cx node[33],node[20];
cx node[1],node[2];
cx node[4],node[3];
cx node[19],node[20];
cx node[2],node[1];
sx node[4];
cx node[20],node[19];
cx node[1],node[2];
rz(3.2112696910072906*pi) node[4];
cx node[19],node[20];
cx node[2],node[1];
sx node[4];
cx node[0],node[1];
rz(1.0*pi) node[4];
cx node[0],node[1];
cx node[4],node[3];
cx node[1],node[0];
cx node[3],node[4];
cx node[0],node[1];
cx node[4],node[3];
cx node[14],node[0];
cx node[2],node[3];
cx node[5],node[4];
cx node[14],node[0];
cx node[2],node[3];
cx node[15],node[4];
sx node[5];
cx node[0],node[14];
cx node[3],node[2];
rz(3.1293313542780123*pi) node[5];
sx node[15];
cx node[14],node[0];
cx node[2],node[3];
sx node[5];
rz(3.1151881969413133*pi) node[15];
cx node[1],node[2];
rz(1.0*pi) node[5];
sx node[15];
cx node[2],node[1];
rz(1.0*pi) node[15];
cx node[1],node[2];
cx node[4],node[15];
cx node[2],node[1];
cx node[15],node[4];
cx node[0],node[1];
cx node[4],node[15];
cx node[0],node[1];
cx node[5],node[4];
cx node[22],node[15];
cx node[1],node[0];
cx node[4],node[5];
sx node[22];
cx node[0],node[1];
cx node[5],node[4];
rz(3.2536616576970454*pi) node[22];
cx node[14],node[0];
cx node[3],node[4];
sx node[22];
cx node[14],node[0];
cx node[4],node[3];
rz(1.0*pi) node[22];
cx node[0],node[14];
cx node[3],node[4];
cx node[22],node[15];
cx node[14],node[0];
cx node[4],node[3];
cx node[15],node[22];
cx node[2],node[3];
cx node[4],node[5];
cx node[22],node[15];
cx node[2],node[3];
cx node[4],node[15];
cx node[23],node[22];
cx node[3],node[2];
cx node[4],node[15];
cx node[21],node[22];
sx node[23];
cx node[2],node[3];
cx node[15],node[4];
sx node[21];
rz(3.1620485980026896*pi) node[23];
cx node[1],node[2];
cx node[4],node[15];
rz(3.236804888139387*pi) node[21];
sx node[23];
cx node[2],node[1];
cx node[5],node[4];
sx node[21];
rz(1.0*pi) node[23];
cx node[1],node[2];
cx node[4],node[5];
rz(1.0*pi) node[21];
cx node[2],node[1];
cx node[5],node[4];
cx node[22],node[21];
cx node[0],node[1];
cx node[3],node[4];
cx node[21],node[22];
cx node[0],node[1];
cx node[4],node[3];
cx node[22],node[21];
cx node[1],node[0];
cx node[3],node[4];
cx node[20],node[21];
cx node[23],node[22];
cx node[0],node[1];
cx node[4],node[3];
sx node[20];
cx node[22],node[23];
cx node[14],node[0];
cx node[2],node[3];
cx node[4],node[5];
rz(3.128328276539044*pi) node[20];
cx node[23],node[22];
cx node[14],node[0];
cx node[2],node[3];
cx node[15],node[22];
sx node[20];
cx node[0],node[14];
cx node[3],node[2];
cx node[22],node[15];
rz(1.0*pi) node[20];
cx node[14],node[0];
cx node[2],node[3];
cx node[15],node[22];
cx node[20],node[21];
cx node[1],node[2];
cx node[22],node[15];
cx node[21],node[20];
cx node[2],node[1];
cx node[4],node[15];
cx node[20],node[21];
cx node[22],node[23];
cx node[1],node[2];
cx node[4],node[15];
cx node[33],node[20];
cx node[22],node[21];
cx node[2],node[1];
cx node[15],node[4];
cx node[19],node[20];
cx node[22],node[21];
sx node[33];
cx node[0],node[1];
cx node[4],node[15];
sx node[19];
rz(0.22316798328844034*pi) node[20];
cx node[21],node[22];
rz(3.274986378402191*pi) node[33];
cx node[0],node[1];
cx node[5],node[4];
rz(3.009874807988039*pi) node[19];
cx node[22],node[21];
sx node[33];
cx node[1],node[0];
cx node[4],node[5];
sx node[19];
cx node[23],node[22];
rz(1.0*pi) node[33];
cx node[0],node[1];
cx node[5],node[4];
rz(1.0*pi) node[19];
cx node[33],node[20];
cx node[22],node[23];
cx node[14],node[0];
cx node[3],node[4];
cx node[20],node[33];
cx node[23],node[22];
cx node[14],node[0];
cx node[4],node[3];
cx node[15],node[22];
cx node[33],node[20];
cx node[0],node[14];
cx node[3],node[4];
cx node[22],node[15];
cx node[21],node[20];
cx node[14],node[0];
cx node[4],node[3];
cx node[15],node[22];
cx node[21],node[20];
cx node[2],node[3];
cx node[4],node[5];
cx node[22],node[15];
cx node[20],node[21];
cx node[2],node[3];
cx node[4],node[15];
cx node[21],node[20];
cx node[22],node[23];
cx node[3],node[2];
cx node[4],node[15];
cx node[20],node[19];
cx node[22],node[21];
cx node[2],node[3];
cx node[15],node[4];
cx node[20],node[33];
cx node[22],node[21];
cx node[1],node[2];
cx node[4],node[15];
sx node[20];
cx node[21],node[22];
cx node[2],node[1];
cx node[5],node[4];
rz(3.2889236213886703*pi) node[20];
cx node[22],node[21];
cx node[1],node[2];
cx node[4],node[5];
sx node[20];
cx node[23],node[22];
cx node[2],node[1];
cx node[5],node[4];
rz(1.0*pi) node[20];
cx node[22],node[23];
cx node[0],node[1];
cx node[3],node[4];
cx node[19],node[20];
cx node[23],node[22];
cx node[0],node[1];
cx node[4],node[3];
cx node[15],node[22];
cx node[20],node[19];
cx node[1],node[0];
cx node[3],node[4];
cx node[22],node[15];
cx node[19],node[20];
cx node[0],node[1];
cx node[4],node[3];
cx node[15],node[22];
cx node[21],node[20];
cx node[14],node[0];
cx node[2],node[3];
cx node[4],node[5];
cx node[22],node[15];
cx node[20],node[21];
cx node[14],node[0];
cx node[2],node[3];
cx node[4],node[15];
cx node[21],node[20];
cx node[22],node[23];
cx node[0],node[14];
cx node[3],node[2];
cx node[4],node[15];
cx node[20],node[21];
cx node[14],node[0];
cx node[2],node[3];
cx node[15],node[4];
cx node[20],node[33];
cx node[22],node[21];
cx node[1],node[2];
cx node[4],node[15];
sx node[20];
cx node[22],node[21];
cx node[2],node[1];
cx node[5],node[4];
rz(3.085072860092394*pi) node[20];
cx node[21],node[22];
cx node[1],node[2];
cx node[4],node[5];
sx node[20];
cx node[22],node[21];
cx node[2],node[1];
cx node[5],node[4];
rz(1.0*pi) node[20];
cx node[23],node[22];
cx node[0],node[1];
cx node[3],node[4];
cx node[33],node[20];
cx node[22],node[23];
cx node[0],node[1];
cx node[4],node[3];
cx node[20],node[33];
cx node[23],node[22];
cx node[1],node[0];
cx node[3],node[4];
cx node[15],node[22];
cx node[33],node[20];
cx node[0],node[1];
cx node[4],node[3];
cx node[22],node[15];
cx node[21],node[20];
cx node[14],node[0];
cx node[2],node[3];
cx node[4],node[5];
cx node[15],node[22];
sx node[21];
cx node[14],node[0];
cx node[2],node[3];
cx node[22],node[15];
rz(3.013036192895428*pi) node[21];
cx node[0],node[14];
cx node[3],node[2];
cx node[4],node[15];
sx node[21];
cx node[22],node[23];
cx node[14],node[0];
cx node[2],node[3];
cx node[4],node[15];
rz(1.0*pi) node[21];
cx node[1],node[2];
cx node[15],node[4];
cx node[20],node[21];
cx node[2],node[1];
cx node[4],node[15];
cx node[21],node[20];
cx node[1],node[2];
cx node[5],node[4];
cx node[20],node[21];
cx node[2],node[1];
cx node[4],node[5];
cx node[22],node[21];
cx node[0],node[1];
cx node[5],node[4];
sx node[22];
cx node[0],node[1];
cx node[3],node[4];
rz(3.230398128204956*pi) node[22];
cx node[1],node[0];
cx node[4],node[3];
sx node[22];
cx node[0],node[1];
cx node[3],node[4];
rz(1.0*pi) node[22];
cx node[14],node[0];
cx node[4],node[3];
cx node[23],node[22];
cx node[14],node[0];
cx node[2],node[3];
cx node[4],node[5];
cx node[22],node[23];
cx node[0],node[14];
cx node[2],node[3];
cx node[23],node[22];
cx node[14],node[0];
cx node[3],node[2];
cx node[15],node[22];
cx node[2],node[3];
cx node[22],node[15];
cx node[1],node[2];
cx node[15],node[22];
cx node[2],node[1];
cx node[22],node[15];
cx node[1],node[2];
cx node[4],node[15];
cx node[22],node[21];
cx node[2],node[1];
cx node[4],node[15];
sx node[22];
cx node[0],node[1];
cx node[15],node[4];
rz(3.2180792225232593*pi) node[22];
cx node[0],node[1];
cx node[4],node[15];
sx node[22];
cx node[1],node[0];
cx node[5],node[4];
rz(1.0*pi) node[22];
cx node[0],node[1];
cx node[4],node[5];
cx node[21],node[22];
cx node[14],node[0];
cx node[5],node[4];
cx node[22],node[21];
cx node[14],node[0];
cx node[3],node[4];
cx node[21],node[22];
cx node[0],node[14];
cx node[4],node[3];
cx node[15],node[22];
cx node[14],node[0];
cx node[3],node[4];
sx node[15];
cx node[4],node[3];
rz(3.0696085856032447*pi) node[15];
cx node[2],node[3];
cx node[4],node[5];
sx node[15];
cx node[2],node[3];
rz(1.0*pi) node[15];
cx node[3],node[2];
cx node[22],node[15];
cx node[2],node[3];
cx node[15],node[22];
cx node[1],node[2];
cx node[22],node[15];
cx node[2],node[1];
cx node[4],node[15];
cx node[1],node[2];
sx node[4];
cx node[2],node[1];
rz(3.164865108060461*pi) node[4];
cx node[0],node[1];
sx node[4];
cx node[0],node[1];
rz(1.0*pi) node[4];
cx node[1],node[0];
cx node[5],node[4];
cx node[0],node[1];
cx node[4],node[5];
cx node[14],node[0];
cx node[5],node[4];
cx node[14],node[0];
cx node[3],node[4];
cx node[0],node[14];
cx node[4],node[3];
cx node[14],node[0];
cx node[3],node[4];
cx node[4],node[3];
cx node[2],node[3];
cx node[4],node[15];
cx node[2],node[3];
sx node[4];
cx node[3],node[2];
rz(3.062140482236428*pi) node[4];
cx node[2],node[3];
sx node[4];
cx node[1],node[2];
rz(1.0*pi) node[4];
cx node[2],node[1];
cx node[15],node[4];
cx node[1],node[2];
cx node[4],node[15];
cx node[2],node[1];
cx node[15],node[4];
cx node[0],node[1];
cx node[3],node[4];
cx node[14],node[0];
sx node[3];
cx node[0],node[1];
rz(3.2340715922650287*pi) node[3];
cx node[14],node[0];
sx node[3];
cx node[0],node[1];
rz(1.0*pi) node[3];
cx node[4],node[3];
cx node[3],node[4];
cx node[4],node[3];
cx node[2],node[3];
sx node[2];
rz(3.2475828629108627*pi) node[2];
sx node[2];
rz(1.0*pi) node[2];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[0],node[1];
sx node[0];
rz(3.0808086324398616*pi) node[0];
sx node[0];
rz(1.0*pi) node[0];
cx node[14],node[0];
cx node[0],node[14];
cx node[14],node[0];
cx node[0],node[1];
sx node[0];
cx node[2],node[1];
rz(3.214006868676754*pi) node[0];
rz(3.7994719708053513*pi) node[1];
sx node[2];
sx node[0];
sx node[1];
rz(3.236310405018057*pi) node[2];
rz(1.0*pi) node[0];
rz(3.5*pi) node[1];
sx node[2];
sx node[1];
rz(1.0*pi) node[2];
rz(1.5*pi) node[1];
barrier node[19],node[33],node[20],node[23],node[21],node[22],node[5],node[15],node[4],node[3],node[14],node[0],node[2],node[1];
measure node[19] -> meas[0];
measure node[33] -> meas[1];
measure node[20] -> meas[2];
measure node[23] -> meas[3];
measure node[21] -> meas[4];
measure node[22] -> meas[5];
measure node[5] -> meas[6];
measure node[15] -> meas[7];
measure node[4] -> meas[8];
measure node[3] -> meas[9];
measure node[14] -> meas[10];
measure node[0] -> meas[11];
measure node[2] -> meas[12];
measure node[1] -> meas[13];
