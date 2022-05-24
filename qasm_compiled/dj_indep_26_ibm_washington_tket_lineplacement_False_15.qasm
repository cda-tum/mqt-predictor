OPENQASM 2.0;
include "qelib1.inc";

qreg node[42];
creg c[25];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
sx node[7];
sx node[8];
sx node[14];
sx node[15];
sx node[16];
sx node[18];
sx node[19];
sx node[20];
sx node[21];
sx node[22];
sx node[23];
sx node[24];
sx node[25];
sx node[33];
sx node[34];
sx node[38];
sx node[39];
sx node[40];
sx node[41];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(3.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[33];
rz(0.5*pi) node[34];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
sx node[7];
sx node[8];
sx node[14];
sx node[15];
sx node[16];
sx node[18];
sx node[19];
sx node[20];
sx node[21];
sx node[22];
sx node[23];
sx node[24];
sx node[25];
sx node[33];
sx node[34];
sx node[38];
sx node[39];
sx node[40];
sx node[41];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[33];
rz(0.5*pi) node[34];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
cx node[3],node[4];
rz(0.5*pi) node[3];
cx node[5],node[4];
sx node[3];
cx node[15],node[4];
rz(0.5*pi) node[5];
rz(3.5*pi) node[3];
sx node[5];
rz(0.5*pi) node[15];
sx node[3];
rz(3.5*pi) node[5];
sx node[15];
rz(1.0*pi) node[3];
sx node[5];
rz(3.5*pi) node[15];
cx node[4],node[3];
rz(1.0*pi) node[5];
sx node[15];
cx node[3],node[4];
cx node[6],node[5];
rz(1.0*pi) node[15];
cx node[4],node[3];
cx node[5],node[6];
cx node[22],node[15];
cx node[2],node[3];
cx node[6],node[5];
cx node[15],node[22];
rz(0.5*pi) node[2];
cx node[7],node[6];
cx node[22],node[15];
sx node[2];
cx node[15],node[4];
cx node[6],node[7];
cx node[21],node[22];
rz(3.5*pi) node[2];
cx node[4],node[15];
cx node[7],node[6];
cx node[22],node[21];
sx node[2];
cx node[15],node[4];
cx node[8],node[7];
cx node[21],node[22];
rz(1.0*pi) node[2];
cx node[7],node[8];
cx node[22],node[15];
cx node[3],node[2];
cx node[8],node[7];
cx node[15],node[22];
cx node[2],node[3];
cx node[16],node[8];
cx node[22],node[15];
cx node[3],node[2];
cx node[8],node[16];
cx node[23],node[22];
cx node[1],node[2];
cx node[4],node[3];
cx node[16],node[8];
cx node[22],node[23];
rz(0.5*pi) node[1];
cx node[3],node[4];
cx node[23],node[22];
sx node[1];
cx node[4],node[3];
cx node[24],node[23];
rz(3.5*pi) node[1];
cx node[5],node[4];
cx node[23],node[24];
sx node[1];
cx node[4],node[5];
cx node[24],node[23];
rz(1.0*pi) node[1];
cx node[5],node[4];
cx node[25],node[24];
cx node[2],node[1];
cx node[6],node[5];
cx node[24],node[25];
cx node[1],node[2];
cx node[5],node[6];
cx node[25],node[24];
cx node[2],node[1];
cx node[6],node[5];
cx node[0],node[1];
cx node[3],node[2];
cx node[7],node[6];
rz(0.5*pi) node[0];
cx node[2],node[3];
cx node[6],node[7];
sx node[0];
cx node[3],node[2];
cx node[7],node[6];
rz(3.5*pi) node[0];
cx node[4],node[3];
cx node[8],node[7];
sx node[0];
cx node[3],node[4];
cx node[7],node[8];
rz(1.0*pi) node[0];
cx node[4],node[3];
cx node[8],node[7];
cx node[14],node[0];
cx node[15],node[4];
cx node[0],node[14];
cx node[4],node[15];
cx node[14],node[0];
cx node[15],node[4];
cx node[0],node[1];
cx node[18],node[14];
cx node[22],node[15];
rz(0.5*pi) node[0];
cx node[14],node[18];
cx node[15],node[22];
sx node[0];
cx node[18],node[14];
cx node[22],node[15];
rz(3.5*pi) node[0];
cx node[19],node[18];
cx node[23],node[22];
sx node[0];
cx node[18],node[19];
cx node[22],node[23];
rz(1.0*pi) node[0];
cx node[19],node[18];
cx node[23],node[22];
cx node[14],node[0];
cx node[20],node[19];
cx node[24],node[23];
cx node[0],node[14];
cx node[19],node[20];
cx node[23],node[24];
cx node[14],node[0];
cx node[20],node[19];
cx node[24],node[23];
cx node[0],node[1];
cx node[18],node[14];
cx node[33],node[20];
cx node[34],node[24];
rz(0.5*pi) node[0];
cx node[14],node[18];
cx node[20],node[33];
cx node[24],node[34];
sx node[0];
cx node[18],node[14];
cx node[33],node[20];
cx node[34],node[24];
rz(3.5*pi) node[0];
cx node[19],node[18];
cx node[39],node[33];
sx node[0];
cx node[18],node[19];
cx node[33],node[39];
rz(1.0*pi) node[0];
cx node[19],node[18];
cx node[39],node[33];
cx node[14],node[0];
cx node[20],node[19];
cx node[40],node[39];
cx node[0],node[14];
cx node[19],node[20];
cx node[39],node[40];
cx node[14],node[0];
cx node[20],node[19];
cx node[40],node[39];
cx node[0],node[1];
cx node[18],node[14];
cx node[33],node[20];
cx node[41],node[40];
rz(0.5*pi) node[0];
cx node[14],node[18];
cx node[20],node[33];
cx node[40],node[41];
sx node[0];
cx node[18],node[14];
cx node[33],node[20];
cx node[41],node[40];
rz(3.5*pi) node[0];
cx node[19],node[18];
cx node[39],node[33];
sx node[0];
cx node[18],node[19];
cx node[33],node[39];
rz(1.0*pi) node[0];
cx node[19],node[18];
cx node[39],node[33];
cx node[14],node[0];
cx node[20],node[19];
cx node[38],node[39];
cx node[0],node[14];
cx node[19],node[20];
cx node[39],node[38];
cx node[14],node[0];
cx node[20],node[19];
cx node[38],node[39];
cx node[0],node[1];
cx node[18],node[14];
cx node[33],node[20];
rz(0.5*pi) node[0];
cx node[2],node[1];
cx node[14],node[18];
cx node[20],node[33];
sx node[0];
rz(0.5*pi) node[2];
cx node[18],node[14];
cx node[33],node[20];
rz(3.5*pi) node[0];
sx node[2];
cx node[19],node[18];
cx node[39],node[33];
sx node[0];
rz(3.5*pi) node[2];
cx node[18],node[19];
cx node[33],node[39];
rz(1.0*pi) node[0];
sx node[2];
cx node[19],node[18];
cx node[39],node[33];
cx node[14],node[0];
rz(1.0*pi) node[2];
cx node[20],node[19];
cx node[40],node[39];
cx node[0],node[14];
cx node[3],node[2];
cx node[19],node[20];
cx node[39],node[40];
cx node[14],node[0];
cx node[2],node[3];
cx node[20],node[19];
cx node[40],node[39];
cx node[3],node[2];
cx node[18],node[14];
cx node[33],node[20];
cx node[2],node[1];
cx node[4],node[3];
cx node[14],node[18];
cx node[20],node[33];
rz(0.5*pi) node[2];
cx node[3],node[4];
cx node[18],node[14];
cx node[33],node[20];
sx node[2];
cx node[4],node[3];
cx node[19],node[18];
cx node[39],node[33];
rz(3.5*pi) node[2];
cx node[15],node[4];
cx node[18],node[19];
cx node[33],node[39];
sx node[2];
cx node[4],node[15];
cx node[19],node[18];
cx node[39],node[33];
rz(1.0*pi) node[2];
cx node[15],node[4];
cx node[20],node[19];
cx node[3],node[2];
cx node[22],node[15];
cx node[19],node[20];
cx node[2],node[3];
cx node[15],node[22];
cx node[20],node[19];
cx node[3],node[2];
cx node[22],node[15];
cx node[33],node[20];
cx node[2],node[1];
cx node[4],node[3];
cx node[20],node[33];
cx node[23],node[22];
cx node[0],node[1];
rz(0.5*pi) node[2];
cx node[3],node[4];
cx node[33],node[20];
cx node[22],node[23];
rz(0.5*pi) node[0];
sx node[2];
cx node[4],node[3];
cx node[23],node[22];
sx node[0];
rz(3.5*pi) node[2];
cx node[5],node[4];
cx node[24],node[23];
rz(3.5*pi) node[0];
sx node[2];
cx node[4],node[5];
cx node[23],node[24];
sx node[0];
rz(1.0*pi) node[2];
cx node[5],node[4];
cx node[24],node[23];
rz(1.0*pi) node[0];
cx node[3],node[2];
cx node[6],node[5];
cx node[14],node[0];
cx node[2],node[3];
cx node[5],node[6];
cx node[0],node[14];
cx node[3],node[2];
cx node[6],node[5];
cx node[14],node[0];
cx node[2],node[1];
cx node[4],node[3];
cx node[7],node[6];
rz(0.5*pi) node[2];
cx node[3],node[4];
cx node[6],node[7];
cx node[18],node[14];
sx node[2];
cx node[4],node[3];
cx node[7],node[6];
cx node[14],node[18];
rz(3.5*pi) node[2];
cx node[15],node[4];
cx node[18],node[14];
sx node[2];
cx node[4],node[15];
cx node[19],node[18];
rz(1.0*pi) node[2];
cx node[15],node[4];
cx node[18],node[19];
cx node[3],node[2];
cx node[22],node[15];
cx node[19],node[18];
cx node[2],node[3];
cx node[15],node[22];
cx node[20],node[19];
cx node[3],node[2];
cx node[22],node[15];
cx node[19],node[20];
cx node[2],node[1];
cx node[4],node[3];
cx node[20],node[19];
cx node[23],node[22];
cx node[0],node[1];
rz(0.5*pi) node[2];
cx node[3],node[4];
cx node[22],node[23];
rz(0.5*pi) node[0];
sx node[2];
cx node[4],node[3];
cx node[23],node[22];
sx node[0];
rz(3.5*pi) node[2];
cx node[5],node[4];
rz(3.5*pi) node[0];
sx node[2];
cx node[4],node[5];
sx node[0];
rz(1.0*pi) node[2];
cx node[5],node[4];
rz(1.0*pi) node[0];
cx node[3],node[2];
cx node[6],node[5];
cx node[14],node[0];
cx node[2],node[3];
cx node[5],node[6];
cx node[0],node[14];
cx node[3],node[2];
cx node[6],node[5];
cx node[14],node[0];
cx node[2],node[1];
cx node[4],node[3];
rz(0.5*pi) node[2];
cx node[3],node[4];
cx node[18],node[14];
sx node[2];
cx node[4],node[3];
cx node[14],node[18];
rz(3.5*pi) node[2];
cx node[15],node[4];
cx node[18],node[14];
sx node[2];
cx node[4],node[15];
cx node[19],node[18];
rz(1.0*pi) node[2];
cx node[15],node[4];
cx node[18],node[19];
cx node[3],node[2];
cx node[22],node[15];
cx node[19],node[18];
cx node[2],node[3];
cx node[15],node[22];
cx node[3],node[2];
cx node[22],node[15];
cx node[2],node[1];
cx node[4],node[3];
cx node[0],node[1];
rz(0.5*pi) node[2];
cx node[3],node[4];
rz(0.5*pi) node[0];
sx node[2];
cx node[4],node[3];
sx node[0];
rz(3.5*pi) node[2];
cx node[15],node[4];
rz(3.5*pi) node[0];
sx node[2];
cx node[4],node[15];
sx node[0];
rz(1.0*pi) node[2];
cx node[15],node[4];
rz(1.0*pi) node[0];
cx node[3],node[2];
cx node[14],node[0];
cx node[2],node[3];
cx node[0],node[14];
cx node[3],node[2];
cx node[14],node[0];
cx node[4],node[3];
cx node[0],node[1];
cx node[3],node[4];
cx node[18],node[14];
rz(0.5*pi) node[0];
cx node[2],node[1];
cx node[4],node[3];
cx node[14],node[18];
sx node[0];
rz(0.5*pi) node[2];
cx node[5],node[4];
cx node[18],node[14];
rz(3.5*pi) node[0];
sx node[2];
cx node[4],node[5];
sx node[0];
rz(3.5*pi) node[2];
cx node[5],node[4];
rz(1.0*pi) node[0];
sx node[2];
cx node[14],node[0];
rz(1.0*pi) node[2];
cx node[0],node[14];
cx node[3],node[2];
cx node[14],node[0];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[1];
cx node[4],node[3];
rz(0.5*pi) node[2];
cx node[3],node[4];
sx node[2];
cx node[4],node[3];
rz(3.5*pi) node[2];
sx node[2];
rz(1.0*pi) node[2];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[1];
cx node[0],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[0];
sx node[2];
sx node[0];
rz(3.5*pi) node[2];
rz(3.5*pi) node[0];
sx node[2];
sx node[0];
rz(1.0*pi) node[2];
rz(1.0*pi) node[0];
barrier node[25],node[16],node[21],node[8],node[34],node[41],node[38],node[40],node[39],node[33],node[24],node[7],node[23],node[20],node[6],node[22],node[19],node[15],node[5],node[18],node[14],node[4],node[3],node[2],node[0],node[1];
measure node[25] -> c[0];
measure node[16] -> c[1];
measure node[21] -> c[2];
measure node[8] -> c[3];
measure node[34] -> c[4];
measure node[41] -> c[5];
measure node[38] -> c[6];
measure node[40] -> c[7];
measure node[39] -> c[8];
measure node[33] -> c[9];
measure node[24] -> c[10];
measure node[7] -> c[11];
measure node[23] -> c[12];
measure node[20] -> c[13];
measure node[6] -> c[14];
measure node[22] -> c[15];
measure node[19] -> c[16];
measure node[15] -> c[17];
measure node[5] -> c[18];
measure node[18] -> c[19];
measure node[14] -> c[20];
measure node[4] -> c[21];
measure node[3] -> c[22];
measure node[2] -> c[23];
measure node[0] -> c[24];
