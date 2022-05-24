OPENQASM 2.0;
include "qelib1.inc";

qreg eval[4];
qreg q[1];
creg meas[5];
rx(1.25866431275903*pi) eval[0];
rz(3.0*pi) eval[1];
ry(0.5*pi) eval[2];
rz(3.5*pi) eval[3];
rz(3.5*pi) q[0];
rz(1.0*pi) eval[0];
rx(3.583821584994456*pi) eval[1];
rx(1.0*pi) eval[3];
rx(2.6720246423777954*pi) q[0];
ry(0.5*pi) eval[0];
rz(1.0*pi) eval[1];
rz(0.5*pi) eval[3];
rz(1.0*pi) q[0];
rxx(0.5*pi) eval[0],q[0];
ry(0.5*pi) eval[1];
ry(0.5*pi) eval[3];
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
rx(0.497868585509393*pi) q[0];
rz(3.5*pi) eval[0];
rxx(0.5*pi) eval[1],q[0];
rx(2.584982244459698*pi) eval[0];
ry(3.5*pi) eval[1];
rx(3.5*pi) q[0];
rz(1.0*pi) eval[0];
rz(3.5*pi) eval[1];
ry(0.5*pi) eval[0];
rx(3.5903344591415505*pi) eval[1];
rz(0.5*pi) eval[1];
ry(0.5*pi) eval[1];
rxx(0.5*pi) eval[1],q[0];
ry(3.5*pi) eval[1];
rx(3.5*pi) q[0];
rz(3.5*pi) eval[1];
rz(1.0*pi) q[0];
rz(3.5*pi) eval[1];
rx(3.006512874147095*pi) q[0];
rx(2.504601171865258*pi) eval[1];
rxx(0.5*pi) eval[2],q[0];
rz(1.0*pi) eval[1];
ry(3.5*pi) eval[2];
rx(3.5*pi) q[0];
ry(0.5*pi) eval[1];
rz(3.5*pi) eval[2];
rx(3.8193310498859097*pi) eval[2];
rz(0.5*pi) eval[2];
ry(0.5*pi) eval[2];
rxx(0.5*pi) eval[2],q[0];
ry(3.5*pi) eval[2];
rx(3.5*pi) q[0];
rz(3.5*pi) eval[2];
rz(3.0*pi) q[0];
rz(3.5*pi) eval[2];
rx(3.819331047600577*pi) q[0];
rx(3.75*pi) eval[2];
rxx(0.5*pi) eval[3],q[0];
rz(3.0*pi) eval[2];
ry(3.5*pi) eval[3];
rx(3.5*pi) q[0];
ry(0.5*pi) eval[2];
rz(3.5*pi) eval[3];
rz(1.0*pi) eval[3];
rx(1.3613378683971924*pi) eval[3];
rz(0.5*pi) eval[3];
ry(0.5*pi) eval[3];
rxx(0.5*pi) eval[3],q[0];
ry(3.5*pi) eval[3];
rx(3.5*pi) q[0];
rz(3.5*pi) eval[3];
rz(1.0*pi) q[0];
rx(0.5*pi) eval[3];
rx(2.361337868397192*pi) q[0];
rz(0.5*pi) eval[3];
rz(0.5*pi) q[0];
rxx(0.5*pi) eval[2],eval[3];
ry(3.5*pi) eval[2];
rx(3.5*pi) eval[3];
rz(3.5*pi) eval[2];
rx(2.75*pi) eval[2];
rz(0.5*pi) eval[2];
ry(0.5*pi) eval[2];
rxx(0.5*pi) eval[2],eval[3];
ry(3.5*pi) eval[2];
rx(3.5*pi) eval[3];
rz(3.5*pi) eval[2];
rx(2.9876060570882026*pi) eval[3];
rxx(0.5*pi) eval[1],eval[3];
rx(0.5*pi) eval[2];
ry(3.5*pi) eval[1];
rz(0.5*pi) eval[2];
rx(3.5*pi) eval[3];
rz(3.5*pi) eval[1];
rx(3.8750000000000004*pi) eval[1];
rz(0.5*pi) eval[1];
ry(0.5*pi) eval[1];
rxx(0.5*pi) eval[1],eval[3];
ry(3.5*pi) eval[1];
rx(3.5*pi) eval[3];
rz(3.5*pi) eval[1];
rz(3.0*pi) eval[3];
rz(3.5*pi) eval[1];
rx(2.810096875985978*pi) eval[3];
rxx(0.5*pi) eval[0],eval[3];
rx(3.2957795868708013*pi) eval[1];
ry(3.5*pi) eval[0];
ry(0.5*pi) eval[1];
rx(3.5*pi) eval[3];
rz(3.5*pi) eval[0];
rxx(0.5*pi) eval[1],eval[2];
rx(3.9375*pi) eval[0];
ry(3.5*pi) eval[1];
rx(3.5*pi) eval[2];
rz(0.5*pi) eval[0];
rz(3.5*pi) eval[1];
ry(0.5*pi) eval[0];
rx(3.75*pi) eval[1];
rxx(0.5*pi) eval[0],eval[3];
rz(0.5*pi) eval[1];
ry(3.5*pi) eval[0];
ry(0.5*pi) eval[1];
rx(3.5*pi) eval[3];
rz(3.5*pi) eval[0];
rxx(0.5*pi) eval[1],eval[2];
rz(3.5*pi) eval[3];
rz(3.5*pi) eval[0];
ry(3.5*pi) eval[1];
rx(3.5*pi) eval[2];
rx(0.5*pi) eval[3];
rx(0.5624092823058239*pi) eval[0];
rz(3.5*pi) eval[1];
rx(1.4911615574084454*pi) eval[2];
rz(0.8849908188977755*pi) eval[3];
rz(1.0*pi) eval[0];
rz(0.5512322521261224*pi) eval[1];
ry(0.5*pi) eval[0];
rx(0.5*pi) eval[1];
rxx(0.5*pi) eval[0],eval[2];
rz(0.5*pi) eval[1];
ry(3.5*pi) eval[0];
rx(3.5*pi) eval[2];
rz(3.5*pi) eval[0];
rx(3.8750000000000004*pi) eval[0];
rz(0.5*pi) eval[0];
ry(0.5*pi) eval[0];
rxx(0.5*pi) eval[0],eval[2];
ry(3.5*pi) eval[0];
rx(3.5*pi) eval[2];
rz(3.5*pi) eval[0];
rz(3.5*pi) eval[2];
rz(3.5*pi) eval[0];
rx(3.5*pi) eval[2];
rx(1.8769778747350134*pi) eval[0];
rz(2.6338384425915544*pi) eval[2];
rz(1.0*pi) eval[0];
ry(0.5*pi) eval[0];
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
barrier eval[0],eval[1],eval[2],eval[3],q[0];
measure eval[0] -> meas[0];
measure eval[1] -> meas[1];
measure eval[2] -> meas[2];
measure eval[3] -> meas[3];
measure q[0] -> meas[4];
