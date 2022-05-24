OPENQASM 2.0;
include "qelib1.inc";

qreg node[126];
creg meas[6];
rz(0.5*pi) node[111];
rz(0.5*pi) node[121];
rz(0.5*pi) node[122];
sx node[123];
rz(0.5*pi) node[124];
rz(0.5*pi) node[125];
sx node[111];
sx node[121];
sx node[122];
rz(2.867005179833682*pi) node[123];
sx node[124];
sx node[125];
rz(1.5*pi) node[111];
rz(1.5*pi) node[121];
rz(3.5*pi) node[122];
sx node[123];
rz(3.5*pi) node[124];
rz(1.5*pi) node[125];
sx node[111];
sx node[121];
sx node[122];
sx node[124];
sx node[125];
rz(3.52165973781717*pi) node[111];
rz(3.601594639534314*pi) node[121];
rz(1.4902786802229122*pi) node[122];
rz(1.063993118194454*pi) node[124];
rz(0.09043341544141803*pi) node[125];
cx node[123],node[124];
sx node[124];
rz(2.5*pi) node[124];
sx node[124];
rz(1.5*pi) node[124];
cx node[125],node[124];
cx node[124],node[125];
cx node[125],node[124];
cx node[123],node[124];
cx node[123],node[122];
cx node[125],node[124];
cx node[123],node[122];
sx node[124];
cx node[122],node[123];
rz(2.5*pi) node[124];
cx node[123],node[122];
sx node[124];
cx node[122],node[111];
rz(1.5*pi) node[124];
cx node[122],node[121];
cx node[125],node[124];
sx node[122];
cx node[124],node[125];
rz(0.280504392181123*pi) node[122];
cx node[125],node[124];
sx node[122];
cx node[124],node[123];
rz(1.0*pi) node[122];
cx node[124],node[123];
cx node[111],node[122];
cx node[123],node[124];
cx node[122],node[111];
cx node[124],node[123];
cx node[111],node[122];
cx node[125],node[124];
cx node[123],node[122];
sx node[124];
cx node[123],node[122];
rz(2.5*pi) node[124];
cx node[122],node[123];
sx node[124];
cx node[123],node[122];
rz(1.5*pi) node[124];
cx node[122],node[121];
cx node[125],node[124];
sx node[122];
cx node[124],node[125];
rz(3.7931718516431423*pi) node[122];
cx node[125],node[124];
sx node[122];
cx node[124],node[123];
rz(1.0*pi) node[122];
cx node[125],node[124];
cx node[121],node[122];
cx node[124],node[123];
cx node[122],node[121];
cx node[125],node[124];
cx node[121],node[122];
cx node[124],node[123];
sx node[123];
rz(2.5*pi) node[123];
sx node[123];
rz(1.5*pi) node[123];
cx node[122],node[123];
cx node[123],node[122];
cx node[122],node[123];
cx node[124],node[123];
sx node[124];
rz(3.082078860176358*pi) node[124];
sx node[124];
rz(1.0*pi) node[124];
cx node[125],node[124];
cx node[124],node[125];
cx node[125],node[124];
cx node[124],node[123];
cx node[122],node[123];
sx node[124];
sx node[122];
rz(1.6450677265696312*pi) node[123];
rz(0.7091474779596603*pi) node[124];
rz(0.3851686822152931*pi) node[122];
sx node[123];
sx node[124];
sx node[122];
rz(0.5*pi) node[123];
rz(1.0*pi) node[124];
rz(1.0*pi) node[122];
sx node[123];
rz(1.5*pi) node[123];
barrier node[111],node[121],node[125],node[124],node[122],node[123];
measure node[111] -> meas[0];
measure node[121] -> meas[1];
measure node[125] -> meas[2];
measure node[124] -> meas[3];
measure node[122] -> meas[4];
measure node[123] -> meas[5];
