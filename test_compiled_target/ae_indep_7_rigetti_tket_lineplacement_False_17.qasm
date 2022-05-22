OPENQASM 2.0;
include "qelib1.inc";

qreg node[15];
creg meas[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[1];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
rz(3.5*pi) node[7];
rz(3.5*pi) node[13];
rz(3.5*pi) node[14];
rx(3.5*pi) node[0];
rx(2.2951672359369732*pi) node[1];
rx(3.5*pi) node[2];
rx(3.5*pi) node[3];
rx(3.5*pi) node[7];
rx(1.5*pi) node[13];
rx(3.5*pi) node[14];
rz(3.5156249999999996*pi) node[0];
rz(0.5*pi) node[1];
rz(3.5312499999999996*pi) node[2];
rz(3.75*pi) node[3];
rz(3.6249999999999996*pi) node[7];
rz(3.5624999999999996*pi) node[14];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[0],node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
rz(3.5*pi) node[1];
rx(3.7048327640630268*pi) node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[0],node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[7],node[0];
rz(3.5*pi) node[1];
rz(0.5*pi) node[0];
rx(0.29516723593697364*pi) node[1];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[7];
cz node[0],node[7];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[7];
rx(0.5*pi) node[0];
cz node[2],node[1];
rx(0.5*pi) node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[7];
cz node[7],node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rz(3.5*pi) node[1];
rz(0.5*pi) node[0];
rx(3.409665540858449*pi) node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[2],node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(3.5*pi) node[1];
cz node[3],node[2];
rx(0.5903344591415508*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[1];
cz node[2],node[3];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
cz node[14],node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[1];
cz node[3],node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(3.5*pi) node[1];
rx(0.5*pi) node[2];
rx(2.8193310498859097*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[14],node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.5*pi) node[14];
rz(3.5*pi) node[1];
rx(1.1806689523994232*pi) node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[0],node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
rz(3.5*pi) node[1];
rx(1.638662131602808*pi) node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[0],node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
rz(3.5*pi) node[1];
rx(0.36133787068252504*pi) node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[2],node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
rz(3.5*pi) node[1];
rx(3.2773243905295706*pi) node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[2],node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(3.5*pi) node[1];
cz node[13],node[2];
rx(0.7226757731960389*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[13];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rx(0.5*pi) node[1];
cz node[2],node[13];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rx(0.5*pi) node[2];
rx(0.5*pi) node[13];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
cz node[13],node[2];
rz(0.5*pi) node[2];
rx(0.5*pi) node[2];
rz(0.5*pi) node[2];
cz node[2],node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
rz(3.5*pi) node[1];
rx(2.5546484627492543*pi) node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[2],node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(3.5*pi) node[1];
rx(0.5*pi) node[2];
rx(3.4453515372507453*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
cz node[13],node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[0],node[1];
rz(0.25*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[1],node[0];
cz node[13],node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[13];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[13],node[14];
cz node[0],node[1];
rz(3.75*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[14],node[13];
cz node[7],node[0];
cz node[1],node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[0];
rx(0.5*pi) node[2];
rx(0.5*pi) node[7];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
cz node[13],node[14];
cz node[0],node[7];
rz(0.1250000000000001*pi) node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
rx(0.5*pi) node[14];
rx(0.5*pi) node[0];
rx(0.5*pi) node[2];
rx(0.5*pi) node[7];
rz(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[7];
rz(0.5*pi) node[14];
cz node[7],node[0];
cz node[1],node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[0];
cz node[1],node[14];
rx(0.5*pi) node[2];
rz(0.5*pi) node[0];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(3.8750000000000004*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[2];
rz(0.25*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
cz node[13],node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
cz node[1],node[14];
rx(0.5*pi) node[2];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(0.5*pi) node[1];
rz(0.06250000000000044*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[2];
rz(3.75*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
cz node[13],node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[2];
cz node[13],node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(3.9375*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[2];
rz(0.1250000000000001*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
cz node[3],node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[2];
cz node[13],node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rz(0.031250000000000555*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[2];
rz(3.8750000000000004*pi) node[14];
rz(0.5*pi) node[2];
cz node[3],node[2];
rz(0.5*pi) node[2];
rx(0.5*pi) node[2];
rz(0.5*pi) node[2];
rz(3.96875*pi) node[2];
rz(0.5*pi) node[2];
rx(0.5*pi) node[2];
rz(0.5*pi) node[2];
cz node[1],node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[2],node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[1],node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[0],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
cz node[13],node[2];
rz(0.01562500000000011*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.25*pi) node[2];
cz node[0],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
cz node[13],node[2];
rz(1.984375*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[13];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[13];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[1];
rz(3.75*pi) node[2];
rx(0.5*pi) node[13];
cz node[14],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
cz node[3],node[2];
rz(0.5*pi) node[14];
cz node[1],node[14];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[14];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[14];
rz(0.5*pi) node[1];
cz node[2],node[3];
rz(0.5*pi) node[14];
cz node[14],node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[1];
cz node[3],node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
cz node[2],node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
rz(0.06250000000000044*pi) node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[2],node[1];
rz(0.5*pi) node[1];
cz node[2],node[3];
rx(0.5*pi) node[1];
rz(0.5*pi) node[3];
rz(0.5*pi) node[1];
rx(0.5*pi) node[3];
rz(3.9375*pi) node[1];
rz(0.5*pi) node[3];
rz(0.5*pi) node[1];
rz(0.1250000000000001*pi) node[3];
rx(0.5*pi) node[1];
rz(0.5*pi) node[3];
rz(0.5*pi) node[1];
rx(0.5*pi) node[3];
cz node[0],node[1];
rz(0.5*pi) node[3];
rz(0.5*pi) node[1];
cz node[2],node[3];
rx(0.5*pi) node[1];
cz node[2],node[13];
rz(0.5*pi) node[3];
rz(0.5*pi) node[1];
rx(0.5*pi) node[3];
rz(0.5*pi) node[13];
rz(0.031250000000000555*pi) node[1];
rz(0.5*pi) node[3];
rx(0.5*pi) node[13];
rz(0.5*pi) node[1];
rz(3.8750000000000004*pi) node[3];
rz(0.5*pi) node[13];
rx(0.5*pi) node[1];
rz(0.5*pi) node[3];
rz(0.25*pi) node[13];
rz(0.5*pi) node[1];
rx(0.5*pi) node[3];
rz(0.5*pi) node[13];
cz node[0],node[1];
rz(0.5*pi) node[3];
rx(0.5*pi) node[13];
rz(0.5*pi) node[1];
rz(0.5*pi) node[13];
rx(0.5*pi) node[1];
cz node[2],node[13];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[13];
rz(1.96875*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[13];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(3.75*pi) node[13];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[13];
cz node[0],node[1];
rx(0.5*pi) node[13];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[13];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[1],node[0];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
cz node[0],node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[1],node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[2],node[1];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[1],node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[2],node[3];
rz(0.5*pi) node[3];
rx(0.5*pi) node[3];
rz(0.5*pi) node[3];
rz(0.06250000000000044*pi) node[3];
rz(0.5*pi) node[3];
rx(0.5*pi) node[3];
rz(0.5*pi) node[3];
cz node[2],node[3];
cz node[2],node[13];
rz(0.5*pi) node[3];
rx(0.5*pi) node[3];
rz(0.5*pi) node[13];
rz(0.5*pi) node[3];
rx(0.5*pi) node[13];
rz(1.9375*pi) node[3];
rz(0.5*pi) node[13];
rz(0.1250000000000001*pi) node[13];
rz(0.5*pi) node[13];
rx(0.5*pi) node[13];
rz(0.5*pi) node[13];
cz node[2],node[13];
cz node[2],node[1];
rz(0.5*pi) node[13];
rz(0.5*pi) node[1];
rx(0.5*pi) node[13];
rx(0.5*pi) node[1];
rz(0.5*pi) node[13];
rz(0.5*pi) node[1];
rz(1.875*pi) node[13];
rz(0.25*pi) node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[1];
rz(0.5*pi) node[1];
cz node[2],node[1];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[1];
rz(1.75*pi) node[1];
barrier node[2],node[1],node[13],node[3],node[0],node[14],node[7];
measure node[2] -> meas[0];
measure node[1] -> meas[1];
measure node[13] -> meas[2];
measure node[3] -> meas[3];
measure node[0] -> meas[4];
measure node[14] -> meas[5];
measure node[7] -> meas[6];
