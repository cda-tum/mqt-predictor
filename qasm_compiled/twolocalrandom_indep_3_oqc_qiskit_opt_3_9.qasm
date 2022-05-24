OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
qreg q[8];
creg meas[3];
rz(1.6601181) q[3];
sx q[3];
rz(-0.54282273) q[3];
sx q[3];
rz(1.6750126) q[3];
sx q[4];
rz(0.77768748) q[4];
sx q[4];
ecr q[4],q[3];
rz(1.9136783) q[3];
sx q[3];
rz(-2.5743502) q[3];
sx q[3];
rz(-0.29244199) q[3];
rz(2.067082) q[4];
sx q[4];
sx q[5];
rz(1.1839805) q[5];
sx q[5];
rz(-pi/2) q[5];
ecr q[4],q[5];
rz(pi/2) q[4];
sx q[4];
rz(-pi/2) q[4];
rz(-2.6664158) q[5];
sx q[5];
rz(-1.8386952) q[5];
sx q[5];
rz(0.47517685) q[5];
ecr q[4],q[5];
rz(-pi) q[4];
sx q[4];
rz(pi/2) q[4];
ecr q[4],q[3];
rz(2.159778) q[3];
sx q[3];
rz(-0.96945442) q[3];
sx q[3];
rz(2.6203582) q[3];
sx q[4];
rz(0.60406017) q[4];
rz(0.39205103) q[5];
sx q[5];
rz(-1.0760628) q[5];
sx q[5];
rz(-2.7191578) q[5];
ecr q[4],q[5];
rz(-pi/2) q[4];
sx q[4];
rz(-pi) q[4];
rz(0.29941689) q[5];
sx q[5];
rz(-2.0277055) q[5];
sx q[5];
rz(0.96026218) q[5];
ecr q[4],q[5];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(0.70882804) q[5];
sx q[5];
rz(-2.3968969) q[5];
sx q[5];
rz(-2.4327646) q[5];
ecr q[4],q[5];
rz(-1.2492475) q[4];
sx q[4];
rz(-0.74469578) q[4];
sx q[4];
rz(2.2796244) q[4];
ecr q[4],q[3];
rz(-1.7881732) q[3];
sx q[3];
rz(-1.4537369) q[3];
sx q[3];
rz(-0.87804181) q[3];
rz(-pi/2) q[4];
rz(pi/2) q[5];
sx q[5];
rz(-0.37860475) q[5];
sx q[5];
rz(pi/2) q[5];
ecr q[4],q[5];
rz(-pi) q[4];
sx q[4];
rz(2.0293017) q[4];
ecr q[4],q[3];
rz(0.29941689) q[3];
sx q[3];
rz(-2.0277055) q[3];
sx q[3];
rz(0.96026218) q[3];
rz(-pi/2) q[4];
sx q[4];
rz(-pi) q[4];
ecr q[4],q[3];
rz(0.70882804) q[3];
sx q[3];
rz(-2.3968969) q[3];
sx q[3];
rz(-2.4327646) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
ecr q[4],q[3];
rz(-2.829129) q[3];
sx q[3];
rz(-1.0973256) q[3];
sx q[3];
rz(2.1871834) q[3];
rz(-1.2492475) q[4];
sx q[4];
rz(-0.74469578) q[4];
sx q[4];
rz(2.2796244) q[4];
rz(-pi/2) q[5];
sx q[5];
rz(-0.29364643) q[5];
sx q[5];
rz(-pi/2) q[5];
ecr q[4],q[5];
sx q[4];
rz(0.75926755) q[4];
sx q[4];
rz(-pi/2) q[4];
ecr q[4],q[3];
rz(-pi/2) q[3];
sx q[3];
rz(-1.685245) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi) q[4];
sx q[4];
rz(-pi/2) q[4];
rz(2.860916) q[5];
sx q[5];
rz(-2.9271763) q[5];
sx q[5];
rz(-2.7760617) q[5];
ecr q[4],q[5];
rz(-pi/2) q[4];
sx q[4];
rz(-pi) q[4];
rz(0.29941689) q[5];
sx q[5];
rz(-2.0277055) q[5];
sx q[5];
rz(0.96026218) q[5];
ecr q[4],q[5];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(0.70882804) q[5];
sx q[5];
rz(-2.3968969) q[5];
sx q[5];
rz(-2.4327646) q[5];
ecr q[4],q[5];
rz(1.8923451) q[4];
sx q[4];
rz(-2.3968969) q[4];
sx q[4];
rz(-2.2796244) q[4];
ecr q[4],q[3];
rz(2.4030577) q[3];
sx q[3];
rz(-1.0764487) q[3];
sx q[3];
rz(2.0512196) q[3];
rz(pi/2) q[4];
rz(-pi/2) q[5];
sx q[5];
rz(-1*pi/6) q[5];
sx q[5];
rz(-pi/2) q[5];
ecr q[4],q[5];
sx q[4];
rz(1.9487778) q[4];
sx q[4];
rz(-pi) q[4];
rz(0.0065882134) q[5];
sx q[5];
rz(-1.9440512) q[5];
sx q[5];
rz(-0.018065615) q[5];
barrier q[0],q[6],q[4],q[2],q[1],q[3],q[7],q[5];
measure q[3] -> meas[0];
measure q[5] -> meas[1];
measure q[4] -> meas[2];
