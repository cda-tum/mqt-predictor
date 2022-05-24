OPENQASM 2.0;
include "qelib1.inc";

qreg node[5];
creg meas[5];
sx node[0];
sx node[1];
sx node[2];
rz(0.5*pi) node[3];
sx node[4];
rz(3.2879466542784512*pi) node[0];
rz(3.2235365105327363*pi) node[1];
rz(3.104357078336898*pi) node[2];
sx node[3];
rz(3.212796000817804*pi) node[4];
sx node[0];
sx node[1];
sx node[2];
rz(3.5*pi) node[3];
sx node[4];
rz(1.0*pi) node[0];
rz(1.0*pi) node[1];
rz(1.0*pi) node[2];
sx node[3];
rz(1.0*pi) node[4];
cx node[1],node[0];
rz(0.7197952040703297*pi) node[3];
cx node[1],node[2];
cx node[1],node[4];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[0],node[1];
cx node[2],node[3];
cx node[0],node[1];
sx node[2];
cx node[1],node[0];
rz(3.028071285346509*pi) node[2];
cx node[0],node[1];
sx node[2];
cx node[1],node[4];
rz(1.0*pi) node[2];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[1],node[2];
sx node[1];
rz(3.3020274104572813*pi) node[1];
sx node[1];
rz(1.0*pi) node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[0],node[1];
cx node[3],node[2];
cx node[1],node[0];
cx node[3],node[2];
cx node[0],node[1];
cx node[2],node[3];
cx node[1],node[4];
cx node[3],node[2];
cx node[1],node[0];
sx node[1];
rz(3.2492806295422354*pi) node[1];
sx node[1];
rz(1.0*pi) node[1];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[4],node[1];
cx node[3],node[2];
cx node[1],node[4];
cx node[3],node[2];
cx node[4],node[1];
cx node[2],node[3];
cx node[1],node[0];
cx node[3],node[2];
rz(0.1642114752892383*pi) node[0];
sx node[1];
rz(3.2360573636040506*pi) node[1];
sx node[1];
rz(1.0*pi) node[1];
cx node[4],node[1];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[0],node[1];
cx node[3],node[2];
cx node[1],node[0];
cx node[3],node[2];
cx node[0],node[1];
cx node[2],node[3];
cx node[4],node[1];
cx node[3],node[2];
cx node[0],node[1];
sx node[4];
sx node[0];
cx node[2],node[1];
rz(3.3065191824673295*pi) node[4];
rz(3.1315783488598363*pi) node[0];
sx node[2];
sx node[4];
sx node[0];
rz(3.0470504029773986*pi) node[2];
rz(1.0*pi) node[4];
rz(1.0*pi) node[0];
sx node[2];
rz(1.0*pi) node[2];
cx node[1],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[4],node[1];
cx node[3],node[2];
cx node[1],node[4];
rz(0.14774744849576704*pi) node[2];
sx node[3];
cx node[4],node[1];
rz(3.2114912686825803*pi) node[3];
cx node[1],node[0];
sx node[3];
cx node[1],node[4];
rz(1.0*pi) node[3];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[1],node[2];
cx node[1],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[0],node[1];
cx node[2],node[3];
cx node[1],node[0];
sx node[2];
cx node[0],node[1];
rz(3.1939613262665096*pi) node[2];
cx node[1],node[4];
sx node[2];
cx node[1],node[0];
rz(1.0*pi) node[2];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[1],node[2];
sx node[1];
rz(3.0216440695213214*pi) node[1];
sx node[1];
rz(1.0*pi) node[1];
cx node[4],node[1];
cx node[1],node[4];
cx node[4],node[1];
cx node[1],node[0];
cx node[1],node[2];
sx node[1];
rz(3.231112663781883*pi) node[1];
sx node[1];
rz(1.0*pi) node[1];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[0],node[1];
sx node[0];
rz(3.6621148486634874*pi) node[1];
rz(3.1074054739606733*pi) node[0];
sx node[1];
sx node[0];
rz(3.5*pi) node[1];
rz(1.0*pi) node[0];
sx node[1];
rz(1.5*pi) node[1];
barrier node[3],node[4],node[2],node[0],node[1];
measure node[3] -> meas[0];
measure node[4] -> meas[1];
measure node[2] -> meas[2];
measure node[0] -> meas[3];
measure node[1] -> meas[4];
