OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }

qreg node[8];
creg meas[8];
sx node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(1.410295355797988*pi) node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
sx node[7];
sx node[0];
rz(2.5*pi) node[1];
rz(2.5*pi) node[2];
rz(0.5*pi) node[3];
rz(2.5*pi) node[4];
rz(0.5*pi) node[5];
rz(2.5*pi) node[6];
rz(0.5*pi) node[7];
rz(1.0*pi) node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
sx node[7];
x node[0];
rz(0.9317787657714864*pi) node[1];
rz(1.476740440658*pi) node[2];
rz(1.4576397464931983*pi) node[3];
rz(3.5519122529832554*pi) node[4];
rz(3.757501569963002*pi) node[5];
rz(3.9582480195283534*pi) node[6];
rz(3.921252260451238*pi) node[7];
rz(3.5*pi) node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
sx node[7];
ecr node[0],node[1];
x node[0];
sx node[1];
rz(3.5*pi) node[0];
rz(2.5*pi) node[1];
ecr node[0],node[7];
sx node[1];
x node[0];
rz(1.5*pi) node[1];
sx node[7];
rz(3.5*pi) node[0];
sx node[1];
ecr node[0],node[1];
sx node[0];
x node[1];
rz(3.5*pi) node[1];
ecr node[1],node[0];
x node[0];
sx node[1];
rz(3.5*pi) node[0];
ecr node[0],node[1];
x node[0];
x node[1];
rz(3.5*pi) node[0];
rz(3.5*pi) node[1];
ecr node[0],node[7];
ecr node[1],node[2];
x node[0];
sx node[1];
x node[2];
sx node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[2];
rz(2.5*pi) node[7];
ecr node[2],node[1];
sx node[7];
x node[1];
sx node[2];
rz(1.5*pi) node[7];
rz(3.5*pi) node[1];
x node[7];
ecr node[1],node[2];
rz(3.5*pi) node[7];
sx node[1];
x node[2];
rz(3.5*pi) node[2];
ecr node[2],node[1];
sx node[1];
x node[2];
ecr node[0],node[1];
rz(3.5*pi) node[2];
x node[0];
sx node[1];
ecr node[2],node[3];
rz(3.5*pi) node[0];
sx node[2];
x node[3];
ecr node[0],node[1];
rz(3.5*pi) node[3];
sx node[0];
x node[1];
ecr node[3],node[2];
rz(3.5*pi) node[1];
x node[2];
sx node[3];
ecr node[1],node[0];
rz(3.5*pi) node[2];
x node[0];
sx node[1];
ecr node[2],node[3];
rz(3.5*pi) node[0];
sx node[2];
x node[3];
ecr node[0],node[1];
rz(3.5*pi) node[3];
sx node[0];
x node[1];
ecr node[3],node[2];
ecr node[7],node[0];
rz(3.5*pi) node[1];
sx node[2];
x node[3];
sx node[0];
ecr node[1],node[2];
rz(3.5*pi) node[3];
x node[7];
rz(2.5*pi) node[0];
x node[1];
sx node[2];
ecr node[3],node[4];
rz(3.5*pi) node[7];
sx node[0];
rz(3.5*pi) node[1];
sx node[3];
x node[4];
rz(1.5*pi) node[0];
ecr node[1],node[2];
rz(3.5*pi) node[4];
sx node[0];
sx node[1];
x node[2];
ecr node[4],node[3];
ecr node[7],node[0];
rz(3.5*pi) node[2];
x node[3];
sx node[4];
x node[0];
ecr node[2],node[1];
rz(3.5*pi) node[3];
sx node[7];
rz(3.5*pi) node[0];
x node[1];
sx node[2];
ecr node[3],node[4];
ecr node[0],node[7];
rz(3.5*pi) node[1];
sx node[3];
x node[4];
sx node[0];
ecr node[1],node[2];
rz(3.5*pi) node[4];
x node[7];
sx node[1];
x node[2];
ecr node[4],node[3];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[2];
sx node[3];
x node[4];
x node[0];
ecr node[2],node[3];
rz(3.5*pi) node[4];
x node[7];
rz(3.5*pi) node[0];
x node[2];
sx node[3];
ecr node[4],node[5];
rz(3.5*pi) node[7];
ecr node[0],node[1];
rz(3.5*pi) node[2];
x node[4];
sx node[5];
x node[0];
sx node[1];
ecr node[2],node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[0];
sx node[2];
x node[3];
ecr node[4],node[5];
ecr node[0],node[1];
rz(3.5*pi) node[3];
x node[4];
x node[5];
sx node[0];
x node[1];
ecr node[3],node[2];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
rz(3.5*pi) node[1];
x node[2];
sx node[3];
ecr node[5],node[6];
ecr node[1],node[0];
rz(3.5*pi) node[2];
sx node[5];
sx node[6];
x node[0];
sx node[1];
ecr node[2],node[3];
ecr node[4],node[5];
rz(3.5*pi) node[0];
sx node[2];
x node[3];
sx node[4];
x node[5];
ecr node[0],node[1];
rz(3.5*pi) node[3];
rz(3.770963686995615*pi) node[4];
rz(3.5*pi) node[5];
sx node[0];
x node[1];
sx node[4];
ecr node[5],node[6];
ecr node[7],node[0];
rz(3.5*pi) node[1];
rz(1.0*pi) node[4];
sx node[5];
sx node[6];
sx node[0];
ecr node[1],node[2];
sx node[4];
x node[7];
rz(2.5*pi) node[0];
x node[1];
sx node[2];
ecr node[3],node[4];
rz(3.5*pi) node[7];
sx node[0];
rz(3.5*pi) node[1];
sx node[3];
x node[4];
rz(1.5*pi) node[0];
ecr node[1],node[2];
rz(3.5*pi) node[4];
sx node[0];
sx node[1];
x node[2];
ecr node[4],node[3];
ecr node[7],node[0];
rz(3.5*pi) node[2];
x node[3];
sx node[4];
x node[0];
ecr node[2],node[1];
rz(3.5*pi) node[3];
x node[7];
rz(3.5*pi) node[0];
x node[1];
sx node[2];
ecr node[3],node[4];
rz(3.5*pi) node[7];
rz(3.5*pi) node[1];
x node[3];
x node[4];
ecr node[1],node[2];
rz(3.5*pi) node[3];
rz(3.5*pi) node[4];
sx node[1];
x node[2];
ecr node[4],node[5];
ecr node[0],node[1];
rz(3.5*pi) node[2];
x node[4];
sx node[5];
sx node[0];
sx node[1];
rz(3.5*pi) node[4];
ecr node[7],node[0];
ecr node[4],node[5];
x node[0];
x node[4];
x node[5];
x node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
rz(3.5*pi) node[7];
ecr node[0],node[1];
ecr node[5],node[6];
x node[0];
sx node[1];
sx node[5];
sx node[6];
rz(3.5*pi) node[0];
ecr node[4],node[5];
ecr node[0],node[1];
rz(0.5*pi) node[4];
x node[5];
x node[0];
sx node[1];
sx node[4];
rz(3.5*pi) node[5];
rz(3.5*pi) node[0];
rz(2.5*pi) node[1];
rz(2.5*pi) node[4];
ecr node[5],node[6];
sx node[1];
sx node[4];
sx node[5];
x node[6];
rz(1.5*pi) node[1];
rz(3.5359614610887684*pi) node[4];
rz(3.5*pi) node[6];
x node[1];
sx node[4];
rz(3.5*pi) node[1];
ecr node[3],node[4];
sx node[3];
sx node[4];
ecr node[2],node[3];
rz(2.5*pi) node[4];
sx node[2];
x node[3];
sx node[4];
rz(3.5*pi) node[3];
rz(1.5*pi) node[4];
ecr node[3],node[2];
sx node[4];
x node[2];
sx node[3];
rz(3.5*pi) node[2];
ecr node[2],node[3];
x node[2];
x node[3];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
ecr node[3],node[4];
sx node[3];
x node[4];
rz(3.5*pi) node[4];
ecr node[4],node[3];
x node[3];
sx node[4];
rz(3.5*pi) node[3];
ecr node[3],node[4];
sx node[3];
x node[4];
rz(3.5*pi) node[4];
ecr node[4],node[5];
x node[4];
sx node[5];
rz(3.5*pi) node[4];
ecr node[6],node[5];
x node[5];
sx node[6];
rz(3.5*pi) node[5];
ecr node[5],node[6];
sx node[5];
x node[6];
rz(3.5*pi) node[6];
ecr node[6],node[5];
sx node[5];
sx node[6];
ecr node[4],node[5];
ecr node[7],node[6];
rz(0.5*pi) node[4];
sx node[5];
x node[6];
sx node[7];
sx node[4];
rz(3.5*pi) node[6];
rz(0.5*pi) node[4];
ecr node[6],node[7];
sx node[4];
sx node[6];
x node[7];
rz(3.828757886567342*pi) node[4];
rz(3.5*pi) node[7];
x node[4];
ecr node[7],node[6];
rz(3.5*pi) node[4];
x node[6];
sx node[7];
ecr node[4],node[3];
rz(3.5*pi) node[6];
x node[3];
sx node[4];
ecr node[6],node[7];
rz(3.5*pi) node[3];
x node[6];
sx node[7];
ecr node[0],node[7];
ecr node[3],node[4];
rz(3.5*pi) node[6];
x node[0];
sx node[3];
x node[4];
ecr node[6],node[5];
sx node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[4];
x node[5];
rz(0.5*pi) node[6];
ecr node[0],node[7];
ecr node[4],node[3];
rz(3.5*pi) node[5];
sx node[6];
sx node[0];
sx node[3];
x node[4];
rz(0.5*pi) node[6];
x node[7];
ecr node[2],node[3];
rz(3.5*pi) node[4];
sx node[6];
rz(3.5*pi) node[7];
ecr node[7],node[0];
x node[2];
sx node[3];
rz(0.5846586100243475*pi) node[6];
x node[0];
rz(3.5*pi) node[2];
ecr node[4],node[3];
sx node[6];
sx node[7];
rz(3.5*pi) node[0];
sx node[3];
sx node[4];
ecr node[5],node[6];
ecr node[0],node[7];
rz(2.5*pi) node[3];
sx node[5];
x node[6];
sx node[0];
sx node[3];
rz(3.5*pi) node[6];
x node[7];
ecr node[1],node[0];
rz(1.5*pi) node[3];
ecr node[6],node[5];
rz(3.5*pi) node[7];
sx node[0];
x node[1];
sx node[3];
x node[5];
sx node[6];
rz(2.5*pi) node[0];
rz(3.5*pi) node[1];
ecr node[2],node[3];
rz(3.5*pi) node[5];
sx node[0];
x node[2];
x node[3];
ecr node[5],node[6];
rz(1.5*pi) node[0];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
x node[5];
sx node[6];
sx node[0];
rz(3.5*pi) node[5];
ecr node[7],node[6];
ecr node[1],node[0];
ecr node[5],node[4];
sx node[6];
rz(0.5*pi) node[7];
x node[0];
sx node[1];
x node[4];
sx node[5];
sx node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[4];
rz(2.5*pi) node[7];
ecr node[0],node[1];
ecr node[4],node[5];
sx node[7];
sx node[0];
x node[1];
sx node[4];
x node[5];
rz(3.6758056258073575*pi) node[7];
rz(3.5*pi) node[1];
rz(3.5*pi) node[5];
sx node[7];
ecr node[1],node[0];
ecr node[5],node[4];
x node[0];
x node[1];
sx node[4];
x node[5];
rz(3.5*pi) node[0];
rz(3.5*pi) node[1];
ecr node[3],node[4];
rz(3.5*pi) node[5];
ecr node[0],node[7];
sx node[3];
sx node[4];
sx node[0];
ecr node[2],node[3];
x node[7];
x node[2];
x node[3];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
x node[0];
ecr node[3],node[4];
sx node[7];
rz(3.5*pi) node[0];
x node[3];
sx node[4];
ecr node[0],node[7];
rz(3.5*pi) node[3];
ecr node[5],node[4];
sx node[0];
sx node[4];
x node[5];
x node[7];
ecr node[1],node[0];
ecr node[3],node[4];
rz(3.5*pi) node[5];
rz(3.5*pi) node[7];
x node[0];
sx node[1];
sx node[3];
sx node[4];
ecr node[7],node[6];
rz(3.5*pi) node[0];
rz(2.5*pi) node[4];
sx node[6];
rz(0.5*pi) node[7];
ecr node[0],node[1];
sx node[4];
sx node[7];
sx node[0];
x node[1];
rz(1.5*pi) node[4];
rz(0.5*pi) node[7];
rz(3.5*pi) node[1];
sx node[4];
sx node[7];
ecr node[1],node[0];
ecr node[5],node[4];
rz(0.48255261399112537*pi) node[7];
x node[0];
sx node[1];
x node[4];
x node[5];
sx node[7];
rz(3.5*pi) node[0];
ecr node[2],node[1];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
ecr node[0],node[7];
sx node[1];
x node[2];
sx node[0];
rz(3.5*pi) node[2];
x node[7];
ecr node[2],node[1];
rz(3.5*pi) node[7];
ecr node[7],node[0];
x node[1];
sx node[2];
x node[0];
rz(3.5*pi) node[1];
sx node[7];
rz(3.5*pi) node[0];
ecr node[1],node[2];
ecr node[0],node[7];
sx node[1];
x node[2];
sx node[0];
rz(3.5*pi) node[2];
x node[7];
ecr node[2],node[1];
rz(3.5*pi) node[7];
x node[1];
x node[2];
ecr node[7],node[6];
rz(3.5*pi) node[1];
rz(3.5*pi) node[2];
rz(2.3878919171956485*pi) node[6];
rz(0.5*pi) node[7];
ecr node[1],node[0];
ecr node[2],node[3];
x node[6];
sx node[7];
sx node[0];
x node[1];
sx node[2];
x node[3];
rz(3.5*pi) node[6];
rz(2.5*pi) node[7];
rz(3.5*pi) node[1];
rz(3.5*pi) node[3];
sx node[7];
ecr node[3],node[2];
rz(0.5091340598345908*pi) node[7];
x node[2];
sx node[3];
x node[7];
rz(3.5*pi) node[2];
rz(3.5*pi) node[7];
ecr node[7],node[0];
ecr node[2],node[3];
x node[0];
x node[2];
sx node[3];
sx node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[2];
ecr node[4],node[3];
ecr node[0],node[7];
sx node[3];
sx node[4];
sx node[0];
ecr node[5],node[4];
x node[7];
x node[4];
x node[5];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
sx node[0];
ecr node[4],node[3];
sx node[7];
ecr node[1],node[0];
sx node[3];
x node[4];
ecr node[6],node[7];
sx node[0];
x node[1];
ecr node[2],node[3];
rz(3.5*pi) node[4];
sx node[6];
x node[7];
rz(3.5*pi) node[1];
x node[2];
sx node[3];
rz(3.5*pi) node[7];
rz(3.5*pi) node[2];
ecr node[4],node[3];
ecr node[7],node[6];
sx node[3];
sx node[4];
x node[6];
sx node[7];
rz(2.5*pi) node[3];
rz(3.5*pi) node[6];
sx node[3];
ecr node[6],node[7];
rz(1.5*pi) node[3];
sx node[6];
x node[7];
sx node[3];
ecr node[5],node[6];
rz(3.5*pi) node[7];
ecr node[7],node[0];
ecr node[2],node[3];
sx node[5];
x node[6];
x node[0];
x node[2];
x node[3];
rz(3.5*pi) node[6];
sx node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
ecr node[6],node[5];
ecr node[0],node[7];
x node[5];
sx node[6];
sx node[0];
rz(3.5*pi) node[5];
x node[7];
ecr node[5],node[6];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[5];
x node[6];
sx node[0];
rz(3.5*pi) node[6];
sx node[7];
ecr node[1],node[0];
ecr node[6],node[5];
x node[0];
sx node[1];
x node[5];
x node[6];
rz(3.5*pi) node[0];
rz(1.5897576647492668*pi) node[1];
rz(3.5*pi) node[5];
rz(3.5*pi) node[6];
sx node[1];
ecr node[5],node[4];
ecr node[6],node[7];
rz(1.0*pi) node[1];
x node[4];
sx node[5];
x node[6];
sx node[7];
ecr node[0],node[7];
x node[1];
rz(3.5*pi) node[4];
rz(3.5*pi) node[6];
sx node[0];
rz(3.5*pi) node[1];
ecr node[4],node[5];
x node[7];
sx node[4];
x node[5];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[5];
x node[0];
ecr node[5],node[4];
sx node[7];
rz(3.5*pi) node[0];
sx node[4];
x node[5];
ecr node[0],node[7];
ecr node[3],node[4];
rz(3.5*pi) node[5];
sx node[0];
sx node[3];
sx node[4];
sx node[7];
ecr node[1],node[0];
ecr node[2],node[3];
ecr node[6],node[7];
x node[0];
sx node[1];
x node[2];
x node[3];
rz(0.5*pi) node[6];
sx node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
sx node[6];
ecr node[0],node[1];
ecr node[3],node[4];
rz(0.5*pi) node[6];
sx node[0];
x node[1];
x node[3];
sx node[4];
sx node[6];
rz(3.5*pi) node[1];
rz(3.5*pi) node[3];
ecr node[5],node[4];
rz(0.5463492656057083*pi) node[6];
ecr node[1],node[0];
sx node[4];
x node[5];
sx node[6];
x node[0];
sx node[1];
ecr node[3],node[4];
rz(3.5*pi) node[5];
rz(3.5*pi) node[0];
ecr node[2],node[1];
sx node[3];
sx node[4];
ecr node[0],node[7];
sx node[1];
x node[2];
rz(2.5*pi) node[4];
sx node[0];
rz(3.5*pi) node[2];
sx node[4];
x node[7];
ecr node[2],node[1];
rz(1.5*pi) node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
x node[1];
sx node[2];
sx node[4];
x node[0];
rz(3.5*pi) node[1];
ecr node[5],node[4];
sx node[7];
rz(3.5*pi) node[0];
ecr node[1],node[2];
x node[4];
x node[5];
ecr node[0],node[7];
sx node[1];
x node[2];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
sx node[0];
rz(3.5*pi) node[2];
x node[7];
ecr node[2],node[1];
rz(3.5*pi) node[7];
x node[1];
x node[2];
ecr node[7],node[6];
rz(3.5*pi) node[1];
rz(3.5*pi) node[2];
sx node[6];
x node[7];
ecr node[1],node[0];
ecr node[2],node[3];
rz(2.5*pi) node[6];
rz(3.5*pi) node[7];
sx node[0];
rz(0.5*pi) node[1];
sx node[2];
x node[3];
sx node[6];
ecr node[7],node[0];
sx node[1];
rz(3.5*pi) node[3];
rz(1.5*pi) node[6];
x node[0];
rz(0.5*pi) node[1];
ecr node[3],node[2];
x node[6];
sx node[7];
rz(3.5*pi) node[0];
sx node[1];
x node[2];
sx node[3];
rz(3.5*pi) node[6];
ecr node[0],node[7];
rz(0.7426110816666864*pi) node[1];
rz(3.5*pi) node[2];
sx node[0];
sx node[1];
ecr node[2],node[3];
x node[7];
x node[2];
sx node[3];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[2];
ecr node[4],node[3];
x node[0];
sx node[3];
sx node[4];
sx node[7];
rz(3.5*pi) node[0];
ecr node[5],node[4];
ecr node[6],node[7];
ecr node[0],node[1];
x node[4];
x node[5];
sx node[6];
x node[7];
sx node[0];
sx node[1];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
rz(3.5*pi) node[7];
ecr node[4],node[3];
ecr node[7],node[6];
sx node[3];
x node[4];
x node[6];
sx node[7];
ecr node[2],node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[6];
x node[2];
sx node[3];
ecr node[6],node[7];
rz(3.5*pi) node[2];
ecr node[4],node[3];
sx node[6];
x node[7];
sx node[3];
sx node[4];
ecr node[5],node[6];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(2.5*pi) node[3];
rz(0.5*pi) node[5];
x node[6];
x node[0];
sx node[3];
sx node[5];
rz(3.5*pi) node[6];
sx node[7];
rz(3.5*pi) node[0];
rz(1.5*pi) node[3];
rz(2.5*pi) node[5];
ecr node[0],node[7];
sx node[3];
sx node[5];
sx node[0];
ecr node[2],node[3];
rz(1.4008296159366294*pi) node[5];
x node[7];
x node[2];
x node[3];
sx node[5];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
ecr node[6],node[5];
x node[0];
x node[5];
sx node[6];
x node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[5];
rz(3.5*pi) node[7];
ecr node[0],node[1];
ecr node[5],node[6];
x node[0];
sx node[1];
sx node[5];
x node[6];
rz(3.5*pi) node[0];
rz(2.5*pi) node[1];
rz(3.5*pi) node[6];
sx node[1];
ecr node[6],node[5];
rz(1.5*pi) node[1];
x node[5];
sx node[6];
x node[1];
rz(3.5*pi) node[5];
ecr node[7],node[6];
rz(3.5*pi) node[1];
ecr node[5],node[4];
sx node[6];
sx node[7];
ecr node[0],node[7];
x node[4];
sx node[5];
sx node[0];
rz(3.5*pi) node[4];
x node[7];
ecr node[4],node[5];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[4];
x node[5];
x node[0];
rz(3.5*pi) node[5];
sx node[7];
rz(3.5*pi) node[0];
ecr node[5],node[4];
ecr node[0],node[7];
sx node[4];
x node[5];
sx node[0];
ecr node[3],node[4];
rz(3.5*pi) node[5];
x node[7];
ecr node[1],node[0];
sx node[3];
sx node[4];
rz(3.5*pi) node[7];
x node[0];
sx node[1];
ecr node[2],node[3];
ecr node[7],node[6];
rz(3.5*pi) node[0];
rz(0.5*pi) node[2];
x node[3];
sx node[6];
sx node[7];
ecr node[0],node[1];
sx node[2];
rz(3.5*pi) node[3];
sx node[0];
x node[1];
rz(0.5*pi) node[2];
ecr node[3],node[4];
rz(3.5*pi) node[1];
sx node[2];
x node[3];
sx node[4];
ecr node[1],node[0];
rz(0.19569071051579356*pi) node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
x node[0];
x node[1];
sx node[2];
sx node[4];
rz(0.5*pi) node[5];
rz(3.5*pi) node[0];
rz(3.5*pi) node[1];
ecr node[3],node[4];
sx node[5];
ecr node[0],node[7];
ecr node[1],node[2];
rz(0.5*pi) node[3];
rz(0.970220075963008*pi) node[4];
rz(0.5*pi) node[5];
sx node[0];
sx node[1];
x node[2];
sx node[3];
sx node[4];
sx node[5];
x node[7];
rz(3.5*pi) node[2];
rz(0.5*pi) node[3];
rz(1.433463065849815*pi) node[5];
rz(3.5*pi) node[7];
ecr node[7],node[0];
ecr node[2],node[1];
sx node[3];
sx node[5];
x node[0];
x node[1];
sx node[2];
rz(0.40205688892941904*pi) node[3];
sx node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[1];
sx node[3];
ecr node[0],node[7];
ecr node[1],node[2];
x node[0];
sx node[1];
x node[2];
x node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[2];
rz(3.5*pi) node[7];
ecr node[2],node[1];
ecr node[7],node[6];
sx node[1];
x node[2];
sx node[6];
x node[7];
ecr node[0],node[1];
rz(3.5*pi) node[2];
rz(2.5*pi) node[6];
rz(3.5*pi) node[7];
sx node[0];
sx node[1];
ecr node[2],node[3];
sx node[6];
ecr node[7],node[0];
sx node[2];
x node[3];
rz(1.5*pi) node[6];
x node[0];
rz(3.5*pi) node[3];
x node[6];
sx node[7];
rz(3.5*pi) node[0];
ecr node[3],node[2];
rz(3.5*pi) node[6];
ecr node[0],node[7];
x node[2];
sx node[3];
sx node[0];
rz(3.5*pi) node[2];
x node[7];
ecr node[2],node[3];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[2];
x node[3];
x node[0];
rz(3.5*pi) node[3];
sx node[7];
rz(3.5*pi) node[0];
ecr node[3],node[4];
ecr node[6],node[7];
ecr node[0],node[1];
x node[3];
x node[4];
sx node[6];
x node[7];
sx node[0];
sx node[1];
rz(3.5*pi) node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[7];
ecr node[4],node[5];
ecr node[7],node[6];
sx node[4];
sx node[5];
x node[6];
sx node[7];
ecr node[3],node[4];
rz(3.5*pi) node[6];
x node[3];
x node[4];
ecr node[6],node[7];
rz(3.5*pi) node[3];
rz(3.5*pi) node[4];
x node[6];
x node[7];
ecr node[3],node[2];
ecr node[4],node[5];
rz(3.5*pi) node[6];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[2];
x node[3];
sx node[4];
sx node[5];
x node[0];
rz(3.5*pi) node[3];
ecr node[6],node[5];
sx node[7];
rz(3.5*pi) node[0];
ecr node[3],node[4];
sx node[5];
x node[6];
ecr node[0],node[7];
sx node[3];
sx node[4];
rz(3.5*pi) node[6];
sx node[0];
rz(3.787975957165828*pi) node[3];
ecr node[6],node[5];
x node[7];
sx node[3];
x node[5];
sx node[6];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(1.0*pi) node[3];
rz(3.5*pi) node[5];
x node[0];
sx node[3];
ecr node[5],node[6];
x node[7];
rz(3.5*pi) node[0];
sx node[5];
x node[6];
rz(3.5*pi) node[7];
ecr node[0],node[1];
rz(3.5*pi) node[6];
x node[0];
sx node[1];
ecr node[6],node[5];
rz(3.5*pi) node[0];
rz(2.5*pi) node[1];
x node[5];
sx node[6];
sx node[1];
rz(3.5*pi) node[5];
ecr node[7],node[6];
rz(1.5*pi) node[1];
ecr node[5],node[4];
sx node[6];
sx node[7];
ecr node[0],node[7];
x node[1];
x node[4];
sx node[5];
sx node[0];
rz(3.5*pi) node[1];
rz(3.5*pi) node[4];
x node[7];
ecr node[4],node[5];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[4];
x node[5];
x node[0];
rz(3.5*pi) node[5];
sx node[7];
rz(3.5*pi) node[0];
ecr node[5],node[4];
ecr node[0],node[7];
x node[4];
sx node[5];
sx node[0];
rz(3.5*pi) node[4];
x node[7];
ecr node[1],node[0];
ecr node[4],node[3];
rz(3.5*pi) node[7];
x node[0];
sx node[1];
x node[3];
x node[4];
ecr node[7],node[6];
rz(3.5*pi) node[0];
rz(3.5*pi) node[3];
rz(3.5*pi) node[4];
sx node[6];
sx node[7];
ecr node[0],node[1];
ecr node[3],node[2];
sx node[0];
x node[1];
sx node[2];
sx node[3];
rz(3.5*pi) node[1];
ecr node[4],node[3];
ecr node[1],node[0];
x node[3];
x node[4];
x node[0];
x node[1];
rz(3.5*pi) node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[0];
rz(3.5*pi) node[1];
ecr node[3],node[2];
ecr node[4],node[5];
ecr node[0],node[7];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[0];
ecr node[1],node[2];
rz(2.2927282137846827*pi) node[4];
x node[7];
sx node[1];
x node[2];
sx node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[2];
rz(1.0*pi) node[4];
x node[0];
ecr node[2],node[1];
sx node[4];
sx node[7];
rz(3.5*pi) node[0];
x node[1];
sx node[2];
ecr node[0],node[7];
rz(3.5*pi) node[1];
x node[0];
ecr node[1],node[2];
x node[7];
rz(3.5*pi) node[0];
sx node[1];
x node[2];
rz(3.5*pi) node[7];
rz(3.5*pi) node[2];
ecr node[7],node[6];
ecr node[2],node[1];
sx node[6];
x node[7];
sx node[1];
x node[2];
rz(2.5*pi) node[6];
rz(3.5*pi) node[7];
ecr node[0],node[1];
rz(3.5*pi) node[2];
sx node[6];
sx node[0];
sx node[1];
ecr node[2],node[3];
rz(1.5*pi) node[6];
ecr node[7],node[0];
sx node[2];
x node[3];
x node[6];
x node[0];
rz(3.5*pi) node[3];
rz(3.5*pi) node[6];
sx node[7];
rz(3.5*pi) node[0];
ecr node[3],node[2];
ecr node[0],node[7];
x node[2];
sx node[3];
sx node[0];
rz(3.5*pi) node[2];
x node[7];
ecr node[2],node[3];
rz(3.5*pi) node[7];
ecr node[7],node[0];
x node[3];
x node[0];
rz(3.5*pi) node[3];
sx node[7];
rz(3.5*pi) node[0];
ecr node[3],node[4];
ecr node[6],node[7];
ecr node[0],node[1];
x node[3];
x node[4];
sx node[6];
x node[7];
sx node[0];
sx node[1];
rz(3.5*pi) node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[7];
ecr node[4],node[5];
ecr node[7],node[6];
sx node[4];
sx node[5];
x node[6];
sx node[7];
ecr node[3],node[4];
rz(3.5*pi) node[6];
sx node[3];
x node[4];
ecr node[6],node[7];
rz(3.4859577944282822*pi) node[3];
rz(3.5*pi) node[4];
x node[6];
x node[7];
sx node[3];
ecr node[4],node[5];
rz(3.5*pi) node[6];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(1.0*pi) node[3];
sx node[5];
x node[0];
ecr node[6],node[5];
sx node[7];
rz(3.5*pi) node[0];
x node[5];
sx node[6];
ecr node[0],node[7];
rz(3.5*pi) node[5];
rz(1.0228506521222056*pi) node[6];
sx node[0];
sx node[6];
x node[7];
rz(1.0*pi) node[6];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[6];
x node[0];
ecr node[5],node[6];
x node[7];
rz(3.5*pi) node[0];
sx node[5];
x node[6];
rz(3.5*pi) node[7];
ecr node[0],node[1];
rz(3.5*pi) node[6];
x node[0];
sx node[1];
ecr node[6],node[5];
rz(3.5*pi) node[0];
rz(2.5*pi) node[1];
x node[5];
sx node[6];
sx node[1];
rz(3.5*pi) node[5];
rz(1.5*pi) node[1];
ecr node[5],node[6];
x node[1];
sx node[6];
rz(3.5*pi) node[1];
ecr node[7],node[6];
x node[6];
sx node[7];
rz(3.5*pi) node[6];
rz(3.9012489093057825*pi) node[7];
sx node[7];
rz(1.0*pi) node[7];
sx node[7];
ecr node[6],node[7];
sx node[6];
x node[7];
rz(3.5*pi) node[7];
ecr node[7],node[6];
x node[6];
sx node[7];
rz(3.5*pi) node[6];
ecr node[6],node[7];
sx node[7];
ecr node[0],node[7];
sx node[0];
x node[7];
rz(1.8107018488425113*pi) node[0];
rz(3.5*pi) node[7];
sx node[0];
rz(1.0*pi) node[0];
sx node[0];
ecr node[7],node[0];
x node[0];
sx node[7];
rz(3.5*pi) node[0];
ecr node[0],node[7];
sx node[0];
x node[7];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[0];
ecr node[1],node[0];
rz(0.7565238178680824*pi) node[0];
sx node[1];
sx node[0];
rz(1.6482873280007282*pi) node[1];
rz(2.5*pi) node[0];
sx node[1];
sx node[0];
rz(1.0*pi) node[1];
rz(1.5*pi) node[0];
barrier node[2],node[4],node[3],node[5],node[6],node[7],node[1],node[0];
measure node[2] -> meas[0];
measure node[4] -> meas[1];
measure node[3] -> meas[2];
measure node[5] -> meas[3];
measure node[6] -> meas[4];
measure node[7] -> meas[5];
measure node[1] -> meas[6];
measure node[0] -> meas[7];
