OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
qreg q[8];
creg c[1];
rz(2.8015193) q[5];
sx q[5];
rz(-1.1422834) q[5];
sx q[5];
rz(-2.276168) q[5];
rz(pi/2) q[6];
ecr q[6],q[5];
rz(-2.9316226) q[5];
sx q[5];
rz(-2.4729073) q[5];
sx q[5];
rz(0.20997004) q[5];
ecr q[6],q[5];
rz(3.0086483) q[5];
sx q[5];
rz(-1.4914315) q[5];
sx q[5];
rz(-2.6064593) q[5];
x q[6];
rz(3*pi/4) q[6];
barrier q[5],q[6];
measure q[5] -> c[0];
