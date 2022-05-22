OPENQASM 2.0;
include "qelib1.inc";

qreg node[127];
creg meas[6];
sx node[111];
sx node[122];
sx node[123];
sx node[124];
rz(0.5*pi) node[125];
sx node[126];
rz(0.5*pi) node[111];
rz(0.5*pi) node[122];
rz(0.5*pi) node[123];
rz(0.5*pi) node[124];
sx node[125];
rz(0.5*pi) node[126];
sx node[111];
sx node[122];
sx node[123];
sx node[124];
rz(3.5*pi) node[125];
sx node[126];
rz(0.5*pi) node[111];
rz(0.25*pi) node[122];
rz(0.12499999999999956*pi) node[123];
rz(0.031249999999999556*pi) node[124];
sx node[125];
rz(0.062499999999999556*pi) node[126];
rz(0.7951672359369731*pi) node[125];
cx node[124],node[125];
rz(3.7048327640630268*pi) node[125];
cx node[124],node[125];
rz(0.29516723593697314*pi) node[125];
cx node[126],node[125];
rz(3.409665540858449*pi) node[125];
cx node[126],node[125];
rz(0.5903344591415509*pi) node[125];
cx node[125],node[124];
cx node[124],node[125];
cx node[125],node[124];
cx node[123],node[124];
cx node[126],node[125];
rz(2.3193310498859097*pi) node[124];
cx node[125],node[126];
sx node[124];
cx node[126],node[125];
rz(3.5*pi) node[124];
sx node[124];
rz(1.5*pi) node[124];
cx node[123],node[124];
rz(3.5*pi) node[124];
sx node[124];
rz(0.5*pi) node[124];
sx node[124];
rz(0.6806689523994236*pi) node[124];
cx node[124],node[123];
cx node[123],node[124];
cx node[124],node[123];
cx node[122],node[123];
rz(1.6386621316028078*pi) node[123];
cx node[122],node[123];
cx node[111],node[122];
rz(0.3613378706825252*pi) node[123];
cx node[122],node[111];
cx node[111],node[122];
cx node[122],node[123];
rz(3.2773243905295706*pi) node[123];
cx node[122],node[123];
rz(0.5*pi) node[122];
rz(0.22267560947042941*pi) node[123];
sx node[122];
sx node[123];
rz(3.5*pi) node[122];
rz(3.5*pi) node[123];
sx node[122];
sx node[123];
rz(1.0*pi) node[122];
rz(1.5*pi) node[123];
cx node[111],node[122];
rz(0.25*pi) node[122];
cx node[111],node[122];
rz(0.5*pi) node[111];
rz(3.75*pi) node[122];
sx node[111];
cx node[122],node[123];
rz(3.5*pi) node[111];
cx node[123],node[122];
sx node[111];
cx node[122],node[123];
rz(1.0*pi) node[111];
cx node[124],node[123];
cx node[111],node[122];
rz(0.125*pi) node[123];
cx node[122],node[111];
cx node[124],node[123];
cx node[111],node[122];
rz(3.875*pi) node[123];
cx node[124],node[123];
cx node[123],node[124];
cx node[124],node[123];
cx node[123],node[122];
cx node[125],node[124];
rz(0.25*pi) node[122];
rz(0.0625*pi) node[124];
cx node[123],node[122];
cx node[125],node[124];
rz(3.75*pi) node[122];
rz(0.5*pi) node[123];
rz(3.9375*pi) node[124];
sx node[123];
cx node[125],node[124];
rz(3.5*pi) node[123];
cx node[124],node[125];
sx node[123];
cx node[125],node[124];
rz(1.0*pi) node[123];
cx node[126],node[125];
cx node[122],node[123];
rz(0.03125*pi) node[125];
cx node[123],node[122];
cx node[126],node[125];
cx node[122],node[123];
rz(3.96875*pi) node[125];
cx node[124],node[123];
cx node[126],node[125];
rz(0.125*pi) node[123];
cx node[125],node[126];
cx node[124],node[123];
cx node[126],node[125];
rz(3.875*pi) node[123];
cx node[124],node[123];
cx node[123],node[124];
cx node[124],node[123];
cx node[123],node[122];
cx node[125],node[124];
rz(0.25*pi) node[122];
rz(0.0625*pi) node[124];
cx node[123],node[122];
cx node[125],node[124];
rz(3.75*pi) node[122];
rz(0.5*pi) node[123];
rz(3.9375*pi) node[124];
sx node[123];
cx node[125],node[124];
rz(3.5*pi) node[123];
cx node[124],node[125];
sx node[123];
cx node[125],node[124];
rz(1.0*pi) node[123];
cx node[124],node[123];
cx node[123],node[124];
cx node[124],node[123];
cx node[123],node[122];
rz(0.125*pi) node[122];
cx node[123],node[122];
rz(3.875*pi) node[122];
cx node[123],node[124];
rz(0.25*pi) node[124];
cx node[123],node[124];
rz(0.5*pi) node[123];
rz(3.75*pi) node[124];
sx node[123];
rz(3.5*pi) node[123];
sx node[123];
rz(1.0*pi) node[123];
barrier node[123],node[124],node[122],node[125],node[126],node[111];
measure node[123] -> meas[0];
measure node[124] -> meas[1];
measure node[122] -> meas[2];
measure node[125] -> meas[3];
measure node[126] -> meas[4];
measure node[111] -> meas[5];
