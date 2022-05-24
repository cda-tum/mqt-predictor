OPENQASM 2.0;
include "qelib1.inc";

qreg node[15];
creg meas[7];
sx node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[14];
rz(0.8816405685007196*pi) node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[14];
sx node[0];
rz(0.5*pi) node[1];
rz(2.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[14];
rz(1.0*pi) node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[14];
rz(1.2356210282382625*pi) node[1];
rz(3.5962910187015735*pi) node[2];
rz(0.7618399328203966*pi) node[3];
rz(0.8306067664522822*pi) node[4];
rz(1.052049654414824*pi) node[5];
rz(1.0264727806036038*pi) node[14];
cx node[0],node[1];
cx node[0],node[14];
sx node[1];
rz(2.5*pi) node[1];
sx node[1];
rz(1.5*pi) node[1];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[0],node[14];
cx node[1],node[2];
cx node[2],node[1];
sx node[14];
cx node[1],node[2];
rz(2.5*pi) node[14];
cx node[2],node[1];
sx node[14];
cx node[0],node[1];
cx node[2],node[3];
rz(1.5*pi) node[14];
cx node[0],node[1];
cx node[3],node[2];
cx node[1],node[0];
cx node[2],node[3];
cx node[0],node[1];
cx node[3],node[2];
cx node[14],node[0];
cx node[1],node[2];
cx node[3],node[4];
sx node[0];
cx node[1],node[2];
cx node[4],node[3];
rz(2.5*pi) node[0];
cx node[2],node[1];
cx node[3],node[4];
sx node[0];
cx node[1],node[2];
cx node[4],node[3];
rz(1.5*pi) node[0];
cx node[2],node[3];
cx node[4],node[5];
cx node[14],node[0];
cx node[2],node[3];
sx node[4];
cx node[0],node[14];
cx node[3],node[2];
rz(0.6849479253232105*pi) node[4];
cx node[14],node[0];
cx node[2],node[3];
sx node[4];
cx node[0],node[1];
rz(1.0*pi) node[4];
cx node[0],node[1];
cx node[5],node[4];
cx node[1],node[0];
cx node[4],node[5];
cx node[0],node[1];
cx node[5],node[4];
cx node[14],node[0];
cx node[1],node[2];
cx node[3],node[4];
sx node[0];
cx node[2],node[1];
rz(0.5*pi) node[3];
rz(2.5*pi) node[0];
cx node[1],node[2];
sx node[3];
sx node[0];
cx node[2],node[1];
rz(2.5*pi) node[3];
rz(1.5*pi) node[0];
sx node[3];
cx node[14],node[0];
rz(0.1677156578160317*pi) node[3];
cx node[0],node[14];
cx node[4],node[3];
cx node[14],node[0];
cx node[3],node[4];
cx node[0],node[1];
cx node[4],node[3];
cx node[0],node[1];
cx node[2],node[3];
cx node[5],node[4];
cx node[1],node[0];
rz(0.5*pi) node[2];
sx node[4];
cx node[0],node[1];
sx node[2];
rz(2.5*pi) node[4];
cx node[14],node[0];
rz(0.5*pi) node[2];
sx node[4];
sx node[0];
sx node[2];
rz(1.5*pi) node[4];
rz(2.5*pi) node[0];
rz(1.3505356112064133*pi) node[2];
cx node[5],node[4];
sx node[0];
cx node[2],node[3];
cx node[4],node[5];
rz(1.5*pi) node[0];
cx node[3],node[2];
cx node[5],node[4];
cx node[14],node[0];
cx node[2],node[3];
cx node[0],node[14];
cx node[1],node[2];
cx node[4],node[3];
cx node[14],node[0];
rz(0.5*pi) node[1];
cx node[4],node[3];
sx node[1];
cx node[3],node[4];
rz(0.5*pi) node[1];
cx node[4],node[3];
sx node[1];
cx node[5],node[4];
rz(3.8479152251025073*pi) node[1];
sx node[4];
cx node[2],node[1];
rz(2.5*pi) node[4];
cx node[1],node[2];
sx node[4];
cx node[2],node[1];
rz(1.5*pi) node[4];
cx node[0],node[1];
cx node[3],node[2];
cx node[5],node[4];
rz(0.5*pi) node[0];
cx node[2],node[3];
cx node[4],node[5];
sx node[0];
cx node[3],node[2];
cx node[5],node[4];
rz(2.5*pi) node[0];
cx node[2],node[3];
sx node[0];
cx node[4],node[3];
rz(3.9612768615471974*pi) node[0];
cx node[4],node[3];
cx node[0],node[1];
cx node[3],node[4];
cx node[1],node[0];
cx node[4],node[3];
cx node[0],node[1];
cx node[5],node[4];
cx node[14],node[0];
cx node[2],node[1];
sx node[4];
rz(1.5536277889790338*pi) node[0];
cx node[2],node[1];
rz(2.5*pi) node[4];
rz(0.5*pi) node[14];
cx node[1],node[2];
sx node[4];
sx node[14];
cx node[2],node[1];
rz(1.5*pi) node[4];
rz(2.5*pi) node[14];
cx node[3],node[2];
cx node[5],node[4];
sx node[14];
cx node[2],node[3];
cx node[4],node[5];
rz(1.1296252914548852*pi) node[14];
cx node[14],node[0];
cx node[3],node[2];
cx node[5],node[4];
cx node[0],node[14];
cx node[2],node[3];
cx node[14],node[0];
cx node[4],node[3];
cx node[1],node[0];
cx node[4],node[3];
cx node[0],node[1];
cx node[3],node[4];
cx node[1],node[0];
cx node[4],node[3];
cx node[0],node[1];
cx node[5],node[4];
cx node[0],node[14];
cx node[2],node[1];
sx node[4];
sx node[0];
cx node[2],node[1];
rz(2.5*pi) node[4];
rz(1.0447842046791136*pi) node[0];
cx node[1],node[2];
sx node[4];
sx node[0];
cx node[2],node[1];
rz(1.5*pi) node[4];
rz(1.0*pi) node[0];
cx node[3],node[2];
cx node[5],node[4];
cx node[14],node[0];
cx node[2],node[3];
cx node[4],node[5];
cx node[0],node[14];
cx node[3],node[2];
cx node[5],node[4];
cx node[14],node[0];
cx node[2],node[3];
cx node[1],node[0];
cx node[4],node[3];
rz(0.5*pi) node[1];
cx node[4],node[3];
sx node[1];
cx node[3],node[4];
rz(2.5*pi) node[1];
cx node[4],node[3];
sx node[1];
cx node[5],node[4];
rz(3.54090434992347*pi) node[1];
sx node[4];
cx node[0],node[1];
rz(2.5*pi) node[4];
cx node[1],node[0];
sx node[4];
cx node[0],node[1];
rz(1.5*pi) node[4];
cx node[14],node[0];
cx node[2],node[1];
cx node[5],node[4];
sx node[0];
rz(0.5*pi) node[2];
cx node[4],node[5];
rz(2.5*pi) node[0];
sx node[2];
cx node[5],node[4];
sx node[0];
rz(0.5*pi) node[2];
rz(1.5*pi) node[0];
sx node[2];
cx node[14],node[0];
rz(0.395128535875076*pi) node[2];
cx node[0],node[14];
cx node[2],node[1];
cx node[14],node[0];
cx node[1],node[2];
cx node[2],node[1];
cx node[0],node[1];
cx node[3],node[2];
cx node[0],node[1];
rz(0.5*pi) node[3];
cx node[1],node[0];
sx node[3];
cx node[0],node[1];
rz(2.5*pi) node[3];
cx node[14],node[0];
sx node[3];
sx node[0];
rz(0.8325203869373581*pi) node[3];
rz(2.5*pi) node[0];
cx node[2],node[3];
sx node[0];
cx node[3],node[2];
rz(1.5*pi) node[0];
cx node[2],node[3];
cx node[14],node[0];
cx node[1],node[2];
cx node[4],node[3];
cx node[0],node[14];
cx node[2],node[1];
rz(0.5*pi) node[4];
cx node[14],node[0];
cx node[1],node[2];
sx node[4];
cx node[2],node[1];
rz(0.5*pi) node[4];
cx node[0],node[1];
sx node[4];
cx node[0],node[1];
rz(3.8157346618259744*pi) node[4];
cx node[1],node[0];
cx node[4],node[3];
cx node[0],node[1];
cx node[3],node[4];
cx node[14],node[0];
cx node[4],node[3];
sx node[0];
cx node[2],node[3];
cx node[5],node[4];
rz(2.5*pi) node[0];
cx node[2],node[3];
rz(1.739943025836574*pi) node[4];
rz(0.5*pi) node[5];
sx node[0];
cx node[3],node[2];
sx node[5];
rz(1.5*pi) node[0];
cx node[2],node[3];
rz(2.5*pi) node[5];
cx node[14],node[0];
cx node[1],node[2];
sx node[5];
cx node[0],node[14];
cx node[2],node[1];
rz(0.8366160899597177*pi) node[5];
cx node[14],node[0];
cx node[1],node[2];
cx node[5],node[4];
cx node[2],node[1];
cx node[4],node[5];
cx node[0],node[1];
cx node[5],node[4];
cx node[0],node[1];
cx node[3],node[4];
cx node[1],node[0];
cx node[4],node[3];
cx node[0],node[1];
cx node[3],node[4];
cx node[14],node[0];
cx node[4],node[3];
sx node[0];
cx node[2],node[3];
cx node[4],node[5];
rz(2.5*pi) node[0];
cx node[2],node[3];
sx node[4];
sx node[0];
cx node[3],node[2];
rz(1.8740668821708013*pi) node[4];
rz(1.5*pi) node[0];
cx node[2],node[3];
sx node[4];
cx node[14],node[0];
cx node[1],node[2];
rz(1.0*pi) node[4];
cx node[0],node[14];
cx node[2],node[1];
cx node[5],node[4];
cx node[14],node[0];
cx node[1],node[2];
cx node[4],node[5];
cx node[2],node[1];
cx node[5],node[4];
cx node[0],node[1];
cx node[3],node[4];
cx node[14],node[0];
sx node[3];
cx node[0],node[1];
rz(0.3341550270794911*pi) node[3];
cx node[14],node[0];
sx node[3];
cx node[0],node[1];
rz(1.0*pi) node[3];
sx node[1];
cx node[4],node[3];
rz(2.5*pi) node[1];
cx node[3],node[4];
sx node[1];
cx node[4],node[3];
rz(1.5*pi) node[1];
cx node[2],node[3];
sx node[2];
rz(2.7846945883508996*pi) node[2];
sx node[2];
rz(1.0*pi) node[2];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[0],node[1];
sx node[0];
rz(2.7161793402759677*pi) node[0];
sx node[0];
rz(1.0*pi) node[0];
cx node[14],node[0];
cx node[0],node[14];
cx node[14],node[0];
cx node[0],node[1];
sx node[0];
cx node[2],node[1];
rz(1.8326620943896055*pi) node[0];
rz(0.40914778937609064*pi) node[1];
sx node[2];
sx node[0];
sx node[1];
rz(1.9090791877419622*pi) node[2];
rz(1.0*pi) node[0];
rz(2.5*pi) node[1];
sx node[2];
sx node[1];
rz(1.0*pi) node[2];
rz(1.5*pi) node[1];
barrier node[5],node[4],node[3],node[14],node[0],node[2],node[1];
measure node[5] -> meas[0];
measure node[4] -> meas[1];
measure node[3] -> meas[2];
measure node[14] -> meas[3];
measure node[0] -> meas[4];
measure node[2] -> meas[5];
measure node[1] -> meas[6];
