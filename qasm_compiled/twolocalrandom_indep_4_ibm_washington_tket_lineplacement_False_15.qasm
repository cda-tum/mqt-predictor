OPENQASM 2.0;
include "qelib1.inc";

qreg node[15];
creg meas[4];
sx node[0];
sx node[1];
rz(0.5*pi) node[2];
sx node[14];
rz(3.1678689628420598*pi) node[0];
rz(3.2383072846694105*pi) node[1];
sx node[2];
rz(3.055575023334653*pi) node[14];
sx node[0];
sx node[1];
rz(3.5*pi) node[2];
sx node[14];
rz(1.0*pi) node[0];
rz(1.0*pi) node[1];
sx node[2];
rz(1.0*pi) node[14];
cx node[0],node[1];
rz(0.5020554710908982*pi) node[2];
cx node[0],node[14];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[0],node[14];
cx node[1],node[2];
sx node[1];
rz(3.1209934563167208*pi) node[1];
sx node[1];
rz(1.0*pi) node[1];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[0],node[1];
sx node[0];
rz(3.1450997678743935*pi) node[0];
sx node[0];
rz(1.0*pi) node[0];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[14],node[0];
cx node[2],node[1];
rz(0.10009496024660103*pi) node[0];
cx node[2],node[1];
sx node[14];
cx node[1],node[2];
rz(3.154265406271735*pi) node[14];
cx node[2],node[1];
sx node[14];
rz(1.0*pi) node[14];
cx node[14],node[0];
cx node[0],node[14];
cx node[14],node[0];
cx node[1],node[0];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[0],node[14];
cx node[2],node[1];
sx node[0];
cx node[2],node[1];
rz(3.0264359377657484*pi) node[0];
cx node[1],node[2];
sx node[0];
cx node[2],node[1];
rz(1.0*pi) node[0];
cx node[14],node[0];
cx node[0],node[14];
cx node[14],node[0];
cx node[1],node[0];
sx node[1];
rz(3.215878420477776*pi) node[1];
sx node[1];
rz(1.0*pi) node[1];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[14],node[0];
cx node[2],node[1];
cx node[14],node[0];
rz(0.29116922253632427*pi) node[1];
sx node[2];
cx node[0],node[14];
rz(3.0744864714256916*pi) node[2];
cx node[14],node[0];
sx node[2];
rz(1.0*pi) node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[0],node[1];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[14],node[0];
cx node[1],node[2];
sx node[1];
rz(3.190311901282958*pi) node[1];
sx node[1];
rz(1.0*pi) node[1];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[1],node[0];
cx node[14],node[0];
cx node[1],node[0];
sx node[14];
rz(3.786904826701036*pi) node[0];
sx node[1];
rz(3.1445073839734174*pi) node[14];
sx node[0];
rz(3.0704558404045623*pi) node[1];
sx node[14];
rz(3.5*pi) node[0];
sx node[1];
rz(1.0*pi) node[14];
sx node[0];
rz(1.0*pi) node[1];
rz(1.5*pi) node[0];
barrier node[2],node[14],node[1],node[0];
measure node[2] -> meas[0];
measure node[14] -> meas[1];
measure node[1] -> meas[2];
measure node[0] -> meas[3];
