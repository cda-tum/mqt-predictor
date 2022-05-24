OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[6];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[14],q[16];
rz(-1.75673946673406) q[16];
cx q[14],q[16];
cx q[13],q[14];
rz(-1.75673946673406) q[14];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/2) q[14];
sx q[14];
rz(12.0455905594774) q[14];
sx q[14];
rz(5*pi/2) q[14];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[19],q[16];
rz(-1.75673946673406) q[16];
cx q[19],q[16];
rz(pi/2) q[16];
sx q[16];
rz(12.0455905594774) q[16];
sx q[16];
rz(5*pi/2) q[16];
cx q[14],q[16];
rz(5.08712887134137) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[18],q[21];
rz(-1.75673946673406) q[21];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
rz(-1.75673946673406) q[15];
cx q[12],q[15];
rz(pi/2) q[12];
sx q[12];
rz(12.0455905594774) q[12];
sx q[12];
rz(5*pi/2) q[12];
cx q[12],q[13];
rz(5.08712887134137) q[13];
cx q[12],q[13];
rz(pi/2) q[13];
sx q[13];
rz(14.8674278535498) q[13];
sx q[13];
rz(5*pi/2) q[13];
rz(pi/2) q[15];
sx q[15];
rz(12.0455905594774) q[15];
sx q[15];
rz(5*pi/2) q[15];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[25],q[24];
rz(-1.75673946673406) q[24];
cx q[25],q[24];
rz(pi/2) q[24];
sx q[24];
rz(12.0455905594774) q[24];
sx q[24];
rz(5*pi/2) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[18],q[21];
rz(5.08712887134137) q[21];
cx q[18],q[21];
cx q[15],q[18];
rz(5.08712887134137) q[18];
cx q[15],q[18];
rz(pi/2) q[15];
sx q[15];
rz(14.8674278535498) q[15];
sx q[15];
rz(5*pi/2) q[15];
rz(pi/2) q[18];
sx q[18];
rz(14.8674278535498) q[18];
sx q[18];
rz(5*pi/2) q[18];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(pi/2) q[25];
sx q[25];
rz(12.0455905594774) q[25];
sx q[25];
rz(5*pi/2) q[25];
cx q[25],q[22];
rz(5.08712887134137) q[22];
cx q[25],q[22];
rz(pi/2) q[22];
sx q[22];
rz(14.8674278535498) q[22];
sx q[22];
rz(5*pi/2) q[22];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
rz(5.08712887134137) q[23];
cx q[24],q[23];
rz(pi/2) q[23];
sx q[23];
rz(14.8674278535498) q[23];
sx q[23];
rz(5*pi/2) q[23];
rz(pi/2) q[24];
sx q[24];
rz(14.8674278535498) q[24];
sx q[24];
rz(5*pi/2) q[24];
barrier q[18],q[25],q[23],q[1],q[7],q[4],q[10],q[15],q[24],q[22],q[16],q[19],q[5],q[2],q[8],q[11],q[17],q[13],q[20],q[26],q[0],q[21],q[3],q[6],q[14],q[9],q[12];
measure q[15] -> meas[0];
measure q[24] -> meas[1];
measure q[18] -> meas[2];
measure q[13] -> meas[3];
measure q[23] -> meas[4];
measure q[22] -> meas[5];
