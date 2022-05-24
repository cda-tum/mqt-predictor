OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg meas[8];
ry(2.2328727) q[0];
ry(0.32030234) q[1];
rxx(pi/2) q[0],q[1];
rx(-3*pi) q[0];
rx(-pi/2) q[1];
ry(pi/2) q[1];
ry(0.34576806) q[2];
rxx(pi/2) q[0],q[2];
rx(-pi) q[2];
rxx(pi/2) q[1],q[2];
rx(-5*pi/2) q[1];
ry(pi/2) q[2];
ry(0.68401222) q[3];
rxx(pi/2) q[0],q[3];
rx(-3*pi/2) q[3];
rxx(pi/2) q[1],q[3];
rxx(pi/2) q[2],q[3];
ry(pi/2) q[3];
ry(0.54563649) q[4];
rxx(pi/2) q[0],q[4];
rxx(pi/2) q[1],q[4];
rxx(pi/2) q[2],q[4];
rxx(pi/2) q[3],q[4];
rx(-3*pi/2) q[3];
ry(pi/2) q[4];
ry(0.24846481) q[5];
rxx(pi/2) q[0],q[5];
rx(-5*pi/2) q[5];
rxx(pi/2) q[1],q[5];
rxx(pi/2) q[2],q[5];
rxx(pi/2) q[3],q[5];
rxx(pi/2) q[4],q[5];
rx(-pi) q[4];
ry(pi/2) q[5];
ry(0.15047851) q[6];
rxx(pi/2) q[0],q[6];
rx(-3*pi) q[6];
rxx(pi/2) q[1],q[6];
rxx(pi/2) q[2],q[6];
rxx(pi/2) q[3],q[6];
rxx(pi/2) q[4],q[6];
rxx(pi/2) q[5],q[6];
rx(-pi/2) q[5];
ry(pi/2) q[6];
ry(0.96337861) q[7];
rxx(pi/2) q[0],q[7];
rz(0.3366067) q[0];
rx(-7*pi/2) q[0];
rx(-7*pi/2) q[7];
rxx(pi/2) q[1],q[7];
rz(-0.88273007) q[1];
rx(-pi) q[1];
rxx(pi/2) q[0],q[1];
ry(pi/2) q[1];
rxx(pi/2) q[2],q[7];
rz(-1.0318637) q[2];
rx(-3*pi/2) q[2];
rxx(pi/2) q[0],q[2];
rxx(pi/2) q[1],q[2];
rx(-5*pi/2) q[1];
ry(pi/2) q[2];
rxx(pi/2) q[3],q[7];
rz(-0.75670242) q[3];
rxx(pi/2) q[0],q[3];
rxx(pi/2) q[1],q[3];
rxx(pi/2) q[2],q[3];
ry(pi/2) q[3];
rxx(pi/2) q[4],q[7];
rz(-1.199661) q[4];
rx(-5*pi/2) q[4];
rxx(pi/2) q[0],q[4];
rxx(pi/2) q[1],q[4];
rxx(pi/2) q[2],q[4];
rxx(pi/2) q[3],q[4];
rx(-3*pi/2) q[3];
ry(pi/2) q[4];
rxx(pi/2) q[5],q[7];
rz(-0.60448519) q[5];
rx(-3*pi) q[5];
rxx(pi/2) q[0],q[5];
rxx(pi/2) q[1],q[5];
rxx(pi/2) q[2],q[5];
rxx(pi/2) q[3],q[5];
rxx(pi/2) q[4],q[5];
rx(-pi) q[4];
ry(pi/2) q[5];
rxx(pi/2) q[6],q[7];
rz(-1.3241579) q[6];
rx(-7*pi/2) q[6];
rxx(pi/2) q[0],q[6];
rxx(pi/2) q[1],q[6];
rxx(pi/2) q[2],q[6];
rxx(pi/2) q[3],q[6];
rxx(pi/2) q[4],q[6];
rxx(pi/2) q[5],q[6];
rx(-pi/2) q[5];
ry(pi/2) q[6];
ry(0.39865192) q[7];
rxx(pi/2) q[0],q[7];
rz(0.39500606) q[0];
rx(-7*pi/2) q[0];
rx(-7*pi/2) q[7];
rxx(pi/2) q[1],q[7];
rz(-0.93156575) q[1];
rx(-pi) q[1];
rxx(pi/2) q[0],q[1];
ry(pi/2) q[1];
rxx(pi/2) q[2],q[7];
rz(-1.0089156) q[2];
rx(-3*pi/2) q[2];
rxx(pi/2) q[0],q[2];
rxx(pi/2) q[1],q[2];
rx(-5*pi/2) q[1];
ry(pi/2) q[2];
rxx(pi/2) q[3],q[7];
rz(-1.4281629) q[3];
rxx(pi/2) q[0],q[3];
rxx(pi/2) q[1],q[3];
rxx(pi/2) q[2],q[3];
ry(pi/2) q[3];
rxx(pi/2) q[4],q[7];
rz(-1.1395228) q[4];
rx(-5*pi/2) q[4];
rxx(pi/2) q[0],q[4];
rxx(pi/2) q[1],q[4];
rxx(pi/2) q[2],q[4];
rxx(pi/2) q[3],q[4];
rx(-3*pi/2) q[3];
ry(pi/2) q[4];
rxx(pi/2) q[5],q[7];
rz(-1.1301496) q[5];
rx(-3*pi) q[5];
rxx(pi/2) q[0],q[5];
rxx(pi/2) q[1],q[5];
rxx(pi/2) q[2],q[5];
rxx(pi/2) q[3],q[5];
rxx(pi/2) q[4],q[5];
rx(-pi) q[4];
ry(pi/2) q[5];
rxx(pi/2) q[6],q[7];
rz(-1.4099057) q[6];
rx(-7*pi/2) q[6];
rxx(pi/2) q[0],q[6];
rxx(pi/2) q[1],q[6];
rxx(pi/2) q[2],q[6];
rxx(pi/2) q[3],q[6];
rxx(pi/2) q[4],q[6];
rxx(pi/2) q[5],q[6];
rx(-pi/2) q[5];
ry(pi/2) q[6];
ry(0.43326276) q[7];
rxx(pi/2) q[0],q[7];
rz(-1.4233664) q[0];
rx(-pi/2) q[0];
rx(-7*pi/2) q[7];
rxx(pi/2) q[1],q[7];
rz(-0.67972535) q[1];
rx(-pi/2) q[1];
rxx(pi/2) q[2],q[7];
rz(-0.97040211) q[2];
rx(-pi/2) q[2];
rxx(pi/2) q[3],q[7];
rz(-1.5015121) q[3];
rx(-pi/2) q[3];
rxx(pi/2) q[4],q[7];
rz(-0.68146657) q[4];
rx(-pi/2) q[4];
rxx(pi/2) q[5],q[7];
rz(-1.4834665) q[5];
rx(-pi/2) q[5];
rxx(pi/2) q[6],q[7];
rz(-0.9599054) q[6];
rx(-pi/2) q[6];
ry(0.65515809) q[7];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
