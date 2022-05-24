OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg meas[2];
rz(1.7352333) q[0];
ry(1.4801179) q[0];
rz(-2.1866523) q[0];
rz(0.67106392) q[1];
ry(1.2360668) q[1];
rz(1.9436477) q[1];
rxx(pi/2) q[0],q[1];
rz(1.8047972) q[0];
ry(1.5431681) q[0];
rz(-1.3367955) q[0];
rz(0.58758334) q[1];
ry(2.0769741) q[1];
rz(-0.62921812) q[1];
rxx(pi/2) q[0],q[1];
rz(-3.1379116) q[0];
ry(3.1342307) q[0];
rz(-0.0036810122) q[0];
rz(0.61547971) q[1];
ry(2*pi/3) q[1];
rz(2.5261129) q[1];
rxx(pi/2) q[0],q[1];
rz(-0.68829514) q[0];
ry(2.3999717) q[0];
rz(1.0313002) q[0];
rz(1.7444376) q[1];
ry(0.60900109) q[1];
rz(-2.5728059) q[1];
barrier q[0],q[1];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
