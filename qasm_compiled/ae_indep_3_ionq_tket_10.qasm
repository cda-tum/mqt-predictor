OPENQASM 2.0;
include "qelib1.inc";

qreg eval[2];
qreg q[1];
creg meas[3];
rz(0.5*pi) eval[0];
rz(3.0*pi) eval[1];
rz(3.5*pi) q[0];
rx(1.0*pi) eval[0];
rx(1.3210321340289208*pi) eval[1];
rx(3.2951672359369732*pi) q[0];
rz(0.5*pi) eval[0];
rz(1.0*pi) eval[1];
ry(0.5*pi) eval[0];
ry(0.5*pi) eval[1];
rxx(0.5*pi) eval[0],q[0];
ry(3.5*pi) eval[0];
rx(3.5*pi) q[0];
rz(3.5*pi) eval[0];
rx(3.7048327640630268*pi) eval[0];
rz(0.5*pi) eval[0];
ry(0.5*pi) eval[0];
rxx(0.5*pi) eval[0],q[0];
ry(3.5*pi) eval[0];
rx(3.5*pi) q[0];
rz(3.5*pi) eval[0];
rz(3.0*pi) q[0];
rz(3.5*pi) eval[0];
rx(1.9582855093103997*pi) q[0];
rx(2.546613475870144*pi) eval[0];
rxx(0.5*pi) eval[1],q[0];
ry(0.5*pi) eval[0];
ry(3.5*pi) eval[1];
rx(3.5*pi) q[0];
rz(3.5*pi) eval[1];
rx(3.5903344591415505*pi) eval[1];
rz(0.5*pi) eval[1];
ry(0.5*pi) eval[1];
rxx(0.5*pi) eval[1],q[0];
ry(3.5*pi) eval[1];
rx(3.5*pi) q[0];
rz(3.5*pi) eval[1];
rz(1.0*pi) q[0];
rz(3.405897269871177*pi) eval[1];
rx(0.8437872043889239*pi) q[0];
rx(3.814605267549765*pi) eval[1];
rz(0.5*pi) q[0];
rz(0.5792790741825866*pi) eval[1];
rxx(0.5*pi) eval[0],eval[1];
ry(3.5*pi) eval[0];
rx(3.5*pi) eval[1];
rz(3.5*pi) eval[0];
rx(3.75*pi) eval[0];
rz(0.5*pi) eval[0];
ry(0.5*pi) eval[0];
rxx(0.5*pi) eval[0],eval[1];
ry(3.5*pi) eval[0];
rx(3.5*pi) eval[1];
rz(3.5*pi) eval[0];
rz(3.5*pi) eval[1];
rz(3.5*pi) eval[0];
rx(0.5*pi) eval[1];
rx(0.20338652412985744*pi) eval[0];
rz(3.8012322521261215*pi) eval[1];
barrier eval[0],eval[1],q[0];
measure eval[0] -> meas[0];
measure eval[1] -> meas[1];
measure q[0] -> meas[2];
