OPENQASM 2.0;
include "qelib1.inc";

qreg node[8];
creg meas[7];
sx node[0];
rz(0.5*pi) node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[7];
rz(0.5*pi) node[0];
sx node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[7];
sx node[0];
rz(3.5*pi) node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[7];
rz(0.015624999999999556*pi) node[0];
sx node[1];
rz(0.031249999999999556*pi) node[2];
rz(0.12499999999999956*pi) node[3];
rz(0.062499999999999556*pi) node[4];
rz(0.5*pi) node[5];
rz(0.25*pi) node[7];
rz(0.7951672359369731*pi) node[1];
cx node[0],node[1];
rz(3.7048327640630268*pi) node[1];
cx node[0],node[1];
rz(0.29516723593697314*pi) node[1];
cx node[2],node[1];
rz(3.409665540858449*pi) node[1];
cx node[2],node[1];
rz(0.5903344591415509*pi) node[1];
cx node[3],node[2];
cx node[4],node[1];
cx node[2],node[3];
rz(2.3193310498859097*pi) node[1];
cx node[3],node[2];
sx node[1];
cx node[5],node[3];
rz(3.5*pi) node[1];
cx node[3],node[5];
sx node[1];
cx node[5],node[3];
rz(1.5*pi) node[1];
cx node[4],node[1];
rz(3.5*pi) node[1];
cx node[7],node[4];
sx node[1];
cx node[4],node[7];
rz(0.5*pi) node[1];
cx node[7],node[4];
sx node[1];
rz(0.6806689523994236*pi) node[1];
cx node[2],node[1];
rz(1.6386621316028078*pi) node[1];
cx node[2],node[1];
rz(0.3613378706825252*pi) node[1];
cx node[3],node[2];
cx node[4],node[1];
cx node[2],node[3];
rz(3.2773243905295706*pi) node[1];
cx node[3],node[2];
cx node[4],node[1];
rz(0.7226757731960389*pi) node[1];
cx node[2],node[1];
rz(2.5546484627492543*pi) node[1];
cx node[2],node[1];
rz(0.9453515372507457*pi) node[1];
rz(0.5*pi) node[2];
sx node[1];
sx node[2];
rz(3.5*pi) node[1];
rz(3.5*pi) node[2];
sx node[1];
sx node[2];
rz(1.5*pi) node[1];
rz(1.0*pi) node[2];
cx node[4],node[1];
cx node[1],node[4];
cx node[4],node[1];
cx node[1],node[2];
cx node[7],node[4];
rz(0.25*pi) node[2];
cx node[4],node[7];
cx node[1],node[2];
cx node[7],node[4];
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
cx node[1],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[4],node[1];
cx node[3],node[2];
rz(0.0625*pi) node[1];
rz(0.25*pi) node[2];
cx node[4],node[1];
cx node[3],node[2];
rz(3.9375*pi) node[1];
rz(3.75*pi) node[2];
rz(0.5*pi) node[3];
cx node[1],node[2];
sx node[3];
cx node[2],node[1];
rz(3.5*pi) node[3];
cx node[1],node[2];
sx node[3];
cx node[4],node[1];
rz(1.0*pi) node[3];
rz(0.125*pi) node[1];
cx node[2],node[3];
cx node[4],node[1];
cx node[3],node[2];
rz(3.875*pi) node[1];
cx node[2],node[3];
cx node[1],node[2];
cx node[5],node[3];
cx node[2],node[1];
rz(0.03125*pi) node[3];
cx node[1],node[2];
cx node[5],node[3];
cx node[4],node[1];
rz(3.96875*pi) node[3];
rz(0.25*pi) node[1];
cx node[3],node[2];
cx node[4],node[1];
cx node[2],node[3];
rz(3.75*pi) node[1];
cx node[3],node[2];
rz(0.5*pi) node[4];
cx node[2],node[1];
cx node[5],node[3];
sx node[4];
cx node[1],node[2];
rz(0.0625*pi) node[3];
rz(3.5*pi) node[4];
cx node[2],node[1];
cx node[5],node[3];
sx node[4];
cx node[0],node[1];
rz(3.9375*pi) node[3];
rz(1.0*pi) node[4];
rz(0.015625*pi) node[1];
cx node[3],node[2];
cx node[0],node[1];
cx node[2],node[3];
rz(3.984375*pi) node[1];
cx node[3],node[2];
cx node[4],node[1];
cx node[5],node[3];
cx node[1],node[4];
rz(0.125*pi) node[3];
cx node[4],node[1];
cx node[5],node[3];
cx node[2],node[1];
rz(3.875*pi) node[3];
cx node[1],node[2];
cx node[2],node[1];
cx node[0],node[1];
cx node[3],node[2];
rz(0.03125*pi) node[1];
cx node[2],node[3];
cx node[0],node[1];
cx node[3],node[2];
rz(3.96875*pi) node[1];
cx node[5],node[3];
cx node[0],node[1];
rz(0.25*pi) node[3];
cx node[1],node[0];
cx node[5],node[3];
cx node[0],node[1];
rz(3.75*pi) node[3];
rz(0.5*pi) node[5];
cx node[1],node[2];
sx node[5];
rz(0.0625*pi) node[2];
rz(3.5*pi) node[5];
cx node[1],node[2];
sx node[5];
rz(3.9375*pi) node[2];
rz(1.0*pi) node[5];
cx node[1],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[3];
rz(0.125*pi) node[3];
cx node[2],node[3];
rz(3.875*pi) node[3];
cx node[5],node[3];
cx node[3],node[5];
cx node[5],node[3];
cx node[2],node[3];
rz(0.25*pi) node[3];
cx node[2],node[3];
rz(0.5*pi) node[2];
rz(3.75*pi) node[3];
sx node[2];
rz(3.5*pi) node[2];
sx node[2];
rz(1.0*pi) node[2];
barrier node[2],node[3],node[5],node[1],node[0],node[4],node[7];
measure node[2] -> meas[0];
measure node[3] -> meas[1];
measure node[5] -> meas[2];
measure node[1] -> meas[3];
measure node[0] -> meas[4];
measure node[4] -> meas[5];
measure node[7] -> meas[6];
