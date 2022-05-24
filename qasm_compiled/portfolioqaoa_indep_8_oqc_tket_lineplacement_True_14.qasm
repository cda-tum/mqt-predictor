OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }

qreg node[8];
creg meas[8];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
sx node[7];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
sx node[7];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
x node[4];
sx node[5];
sx node[6];
sx node[7];
rz(3.5*pi) node[4];
ecr node[4],node[5];
x node[4];
rz(1.6195529137375821*pi) node[5];
rz(3.5*pi) node[4];
sx node[5];
ecr node[4],node[5];
x node[4];
x node[5];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
ecr node[4],node[3];
rz(3.61950293908545*pi) node[3];
x node[4];
sx node[3];
rz(3.5*pi) node[4];
ecr node[4],node[3];
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
ecr node[4],node[3];
x node[3];
sx node[4];
rz(3.5*pi) node[3];
ecr node[5],node[4];
ecr node[3],node[2];
rz(1.6176144065307234*pi) node[4];
x node[5];
rz(3.61941222076789*pi) node[2];
x node[3];
sx node[4];
rz(3.5*pi) node[5];
sx node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
ecr node[3],node[2];
sx node[4];
x node[5];
sx node[2];
x node[3];
rz(3.5*pi) node[5];
rz(3.5*pi) node[3];
ecr node[5],node[4];
ecr node[3],node[2];
x node[4];
sx node[5];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[2];
ecr node[4],node[5];
ecr node[2],node[3];
sx node[4];
x node[5];
sx node[2];
x node[3];
rz(3.5*pi) node[5];
rz(3.5*pi) node[3];
ecr node[5],node[4];
ecr node[3],node[2];
x node[4];
x node[5];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
rz(3.5*pi) node[2];
ecr node[4],node[3];
ecr node[2],node[1];
rz(3.619432592600605*pi) node[3];
x node[4];
rz(3.6195102602128326*pi) node[1];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
sx node[1];
rz(3.5*pi) node[2];
ecr node[4],node[3];
ecr node[2],node[1];
sx node[3];
x node[4];
sx node[1];
x node[2];
rz(3.5*pi) node[4];
rz(3.5*pi) node[2];
ecr node[4],node[3];
ecr node[2],node[1];
x node[3];
sx node[4];
x node[1];
sx node[2];
rz(3.5*pi) node[3];
rz(3.5*pi) node[1];
ecr node[3],node[4];
ecr node[1],node[2];
sx node[3];
x node[4];
sx node[1];
x node[2];
rz(3.5*pi) node[4];
rz(3.5*pi) node[2];
ecr node[4],node[3];
ecr node[2],node[1];
x node[3];
sx node[4];
x node[1];
sx node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
rz(3.5*pi) node[1];
ecr node[3],node[2];
rz(1.619539863032248*pi) node[4];
x node[5];
ecr node[1],node[0];
rz(3.6194278179523103*pi) node[2];
x node[3];
sx node[4];
rz(3.5*pi) node[5];
rz(3.6194714264067187*pi) node[0];
x node[1];
sx node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
sx node[0];
rz(3.5*pi) node[1];
ecr node[3],node[2];
sx node[4];
x node[5];
ecr node[1],node[0];
sx node[2];
x node[3];
rz(3.5*pi) node[5];
x node[0];
sx node[1];
rz(3.5*pi) node[3];
ecr node[5],node[4];
rz(3.5*pi) node[0];
ecr node[3],node[2];
x node[4];
sx node[5];
ecr node[0],node[1];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
sx node[0];
x node[1];
rz(3.5*pi) node[2];
ecr node[4],node[5];
rz(3.5*pi) node[1];
ecr node[2],node[3];
sx node[4];
x node[5];
ecr node[1],node[0];
sx node[2];
x node[3];
rz(3.5*pi) node[5];
x node[0];
sx node[1];
rz(3.5*pi) node[3];
ecr node[5],node[4];
rz(3.5*pi) node[0];
ecr node[3],node[2];
x node[4];
x node[5];
ecr node[0],node[1];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
x node[0];
sx node[1];
rz(3.5*pi) node[2];
ecr node[4],node[3];
rz(3.5*pi) node[0];
ecr node[2],node[1];
rz(3.619411902458003*pi) node[3];
x node[4];
ecr node[0],node[7];
rz(3.6194236799237913*pi) node[1];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
x node[0];
sx node[1];
rz(3.5*pi) node[2];
ecr node[4],node[3];
rz(3.61946792499797*pi) node[7];
rz(3.5*pi) node[0];
ecr node[2],node[1];
sx node[3];
x node[4];
sx node[7];
ecr node[0],node[7];
sx node[1];
x node[2];
rz(3.5*pi) node[4];
x node[0];
rz(3.5*pi) node[2];
ecr node[4],node[3];
sx node[7];
rz(3.5*pi) node[0];
ecr node[2],node[1];
x node[3];
sx node[4];
ecr node[0],node[7];
x node[1];
sx node[2];
rz(3.5*pi) node[3];
sx node[0];
rz(3.5*pi) node[1];
ecr node[3],node[4];
x node[7];
ecr node[1],node[2];
sx node[3];
x node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[1];
x node[2];
rz(3.5*pi) node[4];
x node[0];
rz(3.5*pi) node[2];
ecr node[4],node[3];
sx node[7];
rz(3.5*pi) node[0];
ecr node[2],node[1];
x node[3];
sx node[4];
ecr node[0],node[7];
x node[1];
sx node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
sx node[0];
rz(3.5*pi) node[1];
ecr node[3],node[2];
rz(1.6194968911976133*pi) node[4];
x node[5];
x node[7];
ecr node[1],node[0];
rz(3.619701564454431*pi) node[2];
x node[3];
sx node[4];
rz(3.5*pi) node[5];
rz(3.5*pi) node[7];
rz(3.6194201785150426*pi) node[0];
x node[1];
sx node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
ecr node[7],node[6];
sx node[0];
rz(3.5*pi) node[1];
ecr node[3],node[2];
sx node[4];
x node[5];
rz(3.6195147165512385*pi) node[6];
rz(0.4975324532332228*pi) node[7];
ecr node[1],node[0];
sx node[2];
x node[3];
rz(3.5*pi) node[5];
sx node[6];
x node[7];
sx node[0];
x node[1];
rz(3.5*pi) node[3];
ecr node[5],node[4];
rz(3.5*pi) node[7];
rz(3.5*pi) node[1];
ecr node[3],node[2];
x node[4];
sx node[5];
ecr node[7],node[6];
ecr node[1],node[0];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
x node[6];
sx node[7];
x node[0];
sx node[1];
rz(3.5*pi) node[2];
ecr node[4],node[5];
rz(3.5*pi) node[6];
rz(1.8632727724585898*pi) node[7];
rz(3.5*pi) node[0];
ecr node[2],node[3];
sx node[4];
x node[5];
sx node[7];
ecr node[0],node[1];
sx node[2];
x node[3];
rz(3.5*pi) node[5];
rz(2.5*pi) node[7];
sx node[0];
x node[1];
rz(3.5*pi) node[3];
ecr node[5],node[4];
sx node[7];
rz(3.5*pi) node[1];
ecr node[3],node[2];
x node[4];
x node[5];
ecr node[6],node[7];
ecr node[1],node[0];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
sx node[6];
x node[7];
x node[0];
sx node[1];
rz(3.5*pi) node[2];
ecr node[4],node[3];
rz(3.5*pi) node[7];
rz(3.5*pi) node[0];
ecr node[2],node[1];
rz(3.619463468659564*pi) node[3];
x node[4];
ecr node[7],node[6];
rz(3.62033627436748*pi) node[1];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
x node[6];
sx node[7];
sx node[1];
rz(3.5*pi) node[2];
ecr node[4],node[3];
rz(3.5*pi) node[6];
ecr node[2],node[1];
sx node[3];
x node[4];
ecr node[6],node[7];
sx node[1];
x node[2];
rz(3.5*pi) node[4];
x node[6];
sx node[7];
ecr node[0],node[7];
rz(3.5*pi) node[2];
ecr node[4],node[3];
rz(3.5*pi) node[6];
rz(0.49913648041168024*pi) node[0];
ecr node[2],node[1];
x node[3];
sx node[4];
rz(3.6197840067149514*pi) node[7];
x node[0];
x node[1];
sx node[2];
rz(3.5*pi) node[3];
sx node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[1];
ecr node[3],node[4];
ecr node[0],node[7];
ecr node[1],node[2];
sx node[3];
x node[4];
sx node[0];
sx node[1];
x node[2];
rz(3.5*pi) node[4];
sx node[7];
rz(3.8632727724585902*pi) node[0];
rz(3.5*pi) node[2];
ecr node[4],node[3];
sx node[0];
ecr node[2],node[1];
x node[3];
sx node[4];
rz(0.5*pi) node[0];
x node[1];
sx node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
x node[0];
rz(3.5*pi) node[1];
ecr node[3],node[2];
rz(1.619426544712768*pi) node[4];
x node[5];
rz(3.5*pi) node[0];
rz(3.6194548742926376*pi) node[2];
x node[3];
sx node[4];
rz(3.5*pi) node[5];
ecr node[0],node[7];
sx node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
sx node[0];
ecr node[3],node[2];
sx node[4];
x node[5];
x node[7];
sx node[2];
x node[3];
rz(3.5*pi) node[5];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[3];
ecr node[5],node[4];
x node[0];
ecr node[3],node[2];
x node[4];
sx node[5];
sx node[7];
rz(3.5*pi) node[0];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
ecr node[0],node[7];
rz(3.5*pi) node[2];
ecr node[4],node[5];
sx node[0];
ecr node[2],node[3];
sx node[4];
x node[5];
sx node[7];
ecr node[1],node[0];
sx node[2];
x node[3];
rz(3.5*pi) node[5];
ecr node[6],node[7];
rz(3.6200899025155735*pi) node[0];
rz(0.6729373706627095*pi) node[1];
rz(3.5*pi) node[3];
ecr node[5],node[4];
x node[6];
rz(0.8920430363690386*pi) node[7];
sx node[0];
x node[1];
ecr node[3],node[2];
x node[4];
x node[5];
rz(3.5*pi) node[6];
sx node[7];
rz(3.5*pi) node[1];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
ecr node[6],node[7];
ecr node[1],node[0];
rz(3.5*pi) node[2];
ecr node[4],node[3];
x node[6];
sx node[7];
sx node[0];
sx node[1];
rz(3.619449781334458*pi) node[3];
x node[4];
rz(3.5*pi) node[6];
rz(3.8632727724585902*pi) node[1];
sx node[3];
rz(3.5*pi) node[4];
ecr node[6],node[7];
sx node[1];
ecr node[4],node[3];
sx node[6];
x node[7];
rz(0.5*pi) node[1];
sx node[3];
x node[4];
rz(3.5*pi) node[7];
x node[1];
rz(3.5*pi) node[4];
ecr node[7],node[6];
rz(3.5*pi) node[1];
ecr node[4],node[3];
x node[6];
sx node[7];
ecr node[1],node[0];
x node[3];
sx node[4];
rz(3.5*pi) node[6];
x node[0];
sx node[1];
rz(3.5*pi) node[3];
ecr node[6],node[7];
rz(3.5*pi) node[0];
ecr node[3],node[4];
x node[6];
x node[7];
ecr node[0],node[1];
sx node[3];
x node[4];
rz(3.5*pi) node[6];
rz(3.5*pi) node[7];
sx node[0];
x node[1];
rz(3.5*pi) node[4];
rz(3.5*pi) node[1];
ecr node[4],node[3];
ecr node[1],node[0];
x node[3];
sx node[4];
sx node[0];
sx node[1];
rz(3.5*pi) node[3];
ecr node[5],node[4];
ecr node[7],node[0];
ecr node[2],node[1];
rz(1.61939535034392*pi) node[4];
x node[5];
rz(2.892132163137168*pi) node[0];
rz(3.6194625137299052*pi) node[1];
rz(0.5032659782271196*pi) node[2];
sx node[4];
rz(3.5*pi) node[5];
x node[7];
sx node[0];
sx node[1];
x node[2];
ecr node[5],node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[2];
sx node[4];
x node[5];
x node[0];
ecr node[2],node[1];
rz(3.5*pi) node[5];
sx node[7];
rz(3.5*pi) node[0];
x node[1];
sx node[2];
ecr node[5],node[4];
ecr node[0],node[7];
rz(3.5*pi) node[1];
rz(3.8632727724585902*pi) node[2];
x node[4];
sx node[5];
sx node[0];
sx node[2];
rz(3.5*pi) node[4];
x node[7];
rz(0.5*pi) node[2];
ecr node[4],node[5];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[2];
sx node[4];
x node[5];
x node[0];
ecr node[1],node[2];
rz(3.5*pi) node[5];
sx node[7];
rz(3.5*pi) node[0];
sx node[1];
x node[2];
ecr node[5],node[4];
ecr node[0],node[7];
rz(3.5*pi) node[2];
x node[4];
x node[5];
x node[0];
ecr node[2],node[1];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
sx node[7];
rz(3.5*pi) node[0];
x node[1];
sx node[2];
ecr node[6],node[7];
rz(3.5*pi) node[1];
x node[6];
rz(0.8954881042672014*pi) node[7];
ecr node[1],node[2];
rz(3.5*pi) node[6];
sx node[7];
sx node[1];
sx node[2];
ecr node[6],node[7];
ecr node[0],node[1];
ecr node[3],node[2];
x node[6];
sx node[7];
x node[0];
rz(2.892293227939577*pi) node[1];
rz(3.61948543204171*pi) node[2];
rz(0.5002838283964297*pi) node[3];
rz(3.5*pi) node[6];
rz(3.5*pi) node[0];
sx node[1];
sx node[2];
x node[3];
ecr node[6],node[7];
ecr node[0],node[1];
rz(3.5*pi) node[3];
sx node[6];
x node[7];
x node[0];
sx node[1];
ecr node[3],node[2];
rz(3.5*pi) node[7];
rz(3.5*pi) node[0];
x node[2];
sx node[3];
ecr node[7],node[6];
ecr node[0],node[1];
rz(3.5*pi) node[2];
rz(3.8632727724585902*pi) node[3];
x node[6];
sx node[7];
sx node[0];
x node[1];
sx node[3];
rz(3.5*pi) node[6];
rz(3.5*pi) node[1];
rz(0.5*pi) node[3];
ecr node[6],node[7];
ecr node[1],node[0];
sx node[3];
x node[6];
x node[7];
x node[0];
sx node[1];
ecr node[2],node[3];
rz(3.5*pi) node[6];
rz(3.5*pi) node[7];
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
x node[2];
sx node[3];
rz(2.892257258922438*pi) node[0];
rz(3.5*pi) node[2];
x node[7];
sx node[0];
ecr node[2],node[3];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[2];
sx node[3];
x node[0];
ecr node[1],node[2];
ecr node[4],node[3];
sx node[7];
rz(3.5*pi) node[0];
x node[1];
rz(2.8921191124318355*pi) node[2];
rz(3.6195532320474673*pi) node[3];
rz(0.4971265762973498*pi) node[4];
ecr node[0],node[7];
rz(3.5*pi) node[1];
sx node[2];
sx node[3];
x node[4];
sx node[0];
ecr node[1],node[2];
rz(3.5*pi) node[4];
x node[7];
x node[1];
sx node[2];
ecr node[4],node[3];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[1];
sx node[3];
sx node[4];
x node[0];
ecr node[1],node[2];
rz(3.8632727724585902*pi) node[4];
sx node[7];
rz(3.5*pi) node[0];
sx node[1];
x node[2];
sx node[4];
ecr node[0],node[7];
rz(3.5*pi) node[2];
rz(0.5*pi) node[4];
x node[0];
ecr node[2],node[1];
x node[4];
sx node[7];
rz(3.5*pi) node[0];
x node[1];
sx node[2];
rz(3.5*pi) node[4];
ecr node[6],node[7];
rz(3.5*pi) node[1];
ecr node[4],node[3];
x node[6];
rz(0.8920669096104987*pi) node[7];
ecr node[1],node[2];
x node[3];
sx node[4];
rz(3.5*pi) node[6];
sx node[7];
sx node[1];
x node[2];
rz(3.5*pi) node[3];
ecr node[6],node[7];
ecr node[0],node[1];
rz(3.5*pi) node[2];
ecr node[3],node[4];
x node[6];
sx node[7];
x node[0];
rz(2.8922655349794795*pi) node[1];
sx node[3];
x node[4];
rz(3.5*pi) node[6];
rz(3.5*pi) node[0];
sx node[1];
rz(3.5*pi) node[4];
ecr node[6],node[7];
ecr node[0],node[1];
ecr node[4],node[3];
sx node[6];
x node[7];
x node[0];
sx node[1];
sx node[3];
sx node[4];
rz(3.5*pi) node[7];
rz(3.5*pi) node[0];
ecr node[2],node[3];
ecr node[5],node[4];
ecr node[7],node[6];
ecr node[0],node[1];
x node[2];
rz(2.892188503987022*pi) node[3];
rz(3.6194360940093517*pi) node[4];
rz(0.5048346730082107*pi) node[5];
x node[6];
sx node[7];
sx node[0];
x node[1];
rz(3.5*pi) node[2];
sx node[3];
sx node[4];
x node[5];
rz(3.5*pi) node[6];
rz(3.5*pi) node[1];
ecr node[2],node[3];
rz(3.5*pi) node[5];
ecr node[6],node[7];
ecr node[1],node[0];
x node[2];
sx node[3];
ecr node[5],node[4];
x node[6];
x node[7];
x node[0];
sx node[1];
rz(3.5*pi) node[2];
rz(0.5097028407455282*pi) node[4];
sx node[5];
rz(3.5*pi) node[6];
rz(3.5*pi) node[7];
rz(3.5*pi) node[0];
ecr node[2],node[3];
sx node[4];
rz(3.8632727724585902*pi) node[5];
ecr node[0],node[1];
sx node[2];
x node[3];
rz(3.8632727724585902*pi) node[4];
sx node[5];
sx node[0];
x node[1];
rz(3.5*pi) node[3];
sx node[4];
rz(0.5*pi) node[5];
ecr node[7],node[0];
rz(3.5*pi) node[1];
ecr node[3],node[2];
rz(0.5*pi) node[4];
x node[5];
rz(2.892293864559351*pi) node[0];
x node[2];
sx node[3];
sx node[4];
rz(3.5*pi) node[5];
x node[7];
sx node[0];
rz(3.5*pi) node[2];
ecr node[5],node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
ecr node[2],node[3];
x node[4];
sx node[5];
x node[0];
sx node[2];
x node[3];
rz(3.5*pi) node[4];
sx node[7];
rz(3.5*pi) node[0];
ecr node[1],node[2];
rz(3.5*pi) node[3];
ecr node[4],node[5];
ecr node[0],node[7];
x node[1];
rz(2.892273174416749*pi) node[2];
sx node[4];
x node[5];
sx node[0];
rz(3.5*pi) node[1];
sx node[2];
rz(3.5*pi) node[5];
x node[7];
ecr node[1],node[2];
ecr node[5],node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[1];
x node[2];
sx node[4];
sx node[5];
x node[0];
rz(3.5*pi) node[2];
ecr node[3],node[4];
sx node[7];
rz(3.5*pi) node[0];
ecr node[2],node[1];
x node[3];
rz(2.8921942335649735*pi) node[4];
ecr node[0],node[7];
x node[1];
sx node[2];
rz(3.5*pi) node[3];
sx node[4];
x node[0];
rz(3.5*pi) node[1];
ecr node[3],node[4];
sx node[7];
rz(3.5*pi) node[0];
ecr node[1],node[2];
sx node[3];
x node[4];
ecr node[6],node[7];
sx node[1];
x node[2];
rz(3.5*pi) node[4];
x node[6];
rz(0.8921429856732956*pi) node[7];
rz(3.5*pi) node[2];
ecr node[4],node[3];
rz(3.5*pi) node[6];
sx node[7];
ecr node[2],node[1];
x node[3];
sx node[4];
ecr node[6],node[7];
sx node[1];
x node[2];
rz(3.5*pi) node[3];
x node[6];
sx node[7];
ecr node[0],node[1];
rz(3.5*pi) node[2];
ecr node[3],node[4];
rz(3.5*pi) node[6];
x node[0];
rz(2.8917794757832755*pi) node[1];
sx node[3];
x node[4];
ecr node[6],node[7];
rz(3.5*pi) node[0];
sx node[1];
rz(3.5*pi) node[4];
sx node[6];
x node[7];
ecr node[0],node[1];
ecr node[4],node[3];
rz(3.5*pi) node[7];
sx node[0];
x node[1];
sx node[3];
x node[4];
ecr node[7],node[6];
rz(3.5*pi) node[1];
ecr node[2],node[3];
rz(3.5*pi) node[4];
x node[6];
sx node[7];
ecr node[1],node[0];
x node[2];
rz(2.892278903994699*pi) node[3];
ecr node[4],node[5];
rz(3.5*pi) node[6];
x node[0];
sx node[1];
rz(3.5*pi) node[2];
sx node[3];
rz(0.5043848056460671*pi) node[4];
rz(2.892111154684681*pi) node[5];
ecr node[6],node[7];
rz(3.5*pi) node[0];
ecr node[2],node[3];
x node[4];
sx node[5];
x node[6];
x node[7];
ecr node[0],node[1];
sx node[2];
x node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[6];
rz(3.5*pi) node[7];
sx node[0];
x node[1];
rz(3.5*pi) node[3];
ecr node[4],node[5];
rz(3.5*pi) node[1];
ecr node[3],node[2];
sx node[4];
x node[5];
ecr node[1],node[0];
x node[2];
sx node[3];
rz(1.219367392908981*pi) node[4];
rz(3.5*pi) node[5];
sx node[0];
x node[1];
rz(3.5*pi) node[2];
sx node[4];
ecr node[7],node[0];
rz(3.5*pi) node[1];
ecr node[2],node[3];
rz(2.5*pi) node[4];
rz(2.892202509622015*pi) node[0];
sx node[2];
x node[3];
sx node[4];
x node[7];
sx node[0];
rz(3.5*pi) node[3];
ecr node[5],node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
ecr node[3],node[2];
x node[4];
sx node[5];
x node[0];
sx node[2];
x node[3];
rz(3.5*pi) node[4];
sx node[7];
rz(3.5*pi) node[0];
ecr node[1],node[2];
rz(3.5*pi) node[3];
ecr node[4],node[5];
ecr node[0],node[7];
x node[1];
rz(2.890651703856525*pi) node[2];
sx node[4];
x node[5];
sx node[0];
rz(3.5*pi) node[1];
sx node[2];
rz(3.5*pi) node[5];
x node[7];
ecr node[1],node[2];
ecr node[5],node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
x node[1];
sx node[2];
sx node[4];
x node[5];
x node[0];
rz(3.5*pi) node[1];
ecr node[3],node[4];
rz(3.5*pi) node[5];
sx node[7];
rz(3.5*pi) node[0];
ecr node[1],node[2];
rz(0.5015344997702345*pi) node[3];
rz(2.8916330532356316*pi) node[4];
ecr node[0],node[7];
sx node[1];
x node[2];
x node[3];
sx node[4];
x node[0];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
sx node[7];
rz(3.5*pi) node[0];
ecr node[2],node[1];
ecr node[3],node[4];
ecr node[6],node[7];
x node[1];
sx node[2];
sx node[3];
x node[4];
x node[6];
rz(0.8922680814585675*pi) node[7];
rz(3.5*pi) node[1];
rz(3.219367392908981*pi) node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[6];
sx node[7];
ecr node[1],node[2];
sx node[3];
ecr node[6],node[7];
sx node[1];
x node[2];
rz(0.5*pi) node[3];
x node[6];
sx node[7];
ecr node[0],node[1];
rz(3.5*pi) node[2];
sx node[3];
rz(3.5*pi) node[6];
x node[0];
rz(2.892217470186665*pi) node[1];
ecr node[4],node[3];
ecr node[6],node[7];
rz(3.5*pi) node[0];
sx node[1];
x node[3];
sx node[4];
sx node[6];
x node[7];
ecr node[0],node[1];
rz(3.5*pi) node[3];
rz(3.5*pi) node[7];
x node[0];
sx node[1];
ecr node[3],node[4];
ecr node[7],node[6];
rz(3.5*pi) node[0];
sx node[3];
x node[4];
x node[6];
sx node[7];
ecr node[0],node[1];
rz(3.5*pi) node[4];
rz(3.5*pi) node[6];
sx node[0];
x node[1];
ecr node[4],node[3];
ecr node[6],node[7];
rz(3.5*pi) node[1];
sx node[3];
sx node[4];
x node[6];
x node[7];
ecr node[1],node[0];
ecr node[2],node[3];
ecr node[5],node[4];
rz(3.5*pi) node[6];
rz(3.5*pi) node[7];
x node[0];
sx node[1];
rz(0.19269054162966714*pi) node[2];
rz(2.8910890616401446*pi) node[3];
rz(0.39276222527730464*pi) node[4];
x node[5];
rz(3.5*pi) node[0];
x node[2];
sx node[3];
sx node[4];
rz(3.5*pi) node[5];
ecr node[0],node[1];
rz(3.5*pi) node[2];
ecr node[5],node[4];
sx node[0];
x node[1];
ecr node[2],node[3];
sx node[4];
x node[5];
ecr node[7],node[0];
rz(3.5*pi) node[1];
sx node[2];
sx node[3];
rz(3.5*pi) node[5];
rz(2.8922270194832507*pi) node[0];
rz(3.219367392908981*pi) node[2];
ecr node[5],node[4];
x node[7];
sx node[0];
sx node[2];
x node[4];
sx node[5];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(0.5*pi) node[2];
rz(3.5*pi) node[4];
x node[0];
x node[2];
ecr node[4],node[5];
sx node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[2];
sx node[4];
x node[5];
ecr node[0],node[7];
ecr node[2],node[3];
rz(3.5*pi) node[5];
sx node[0];
sx node[2];
x node[3];
ecr node[5],node[4];
x node[7];
rz(3.5*pi) node[3];
x node[4];
x node[5];
rz(3.5*pi) node[7];
ecr node[7],node[0];
ecr node[3],node[2];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
x node[0];
x node[2];
sx node[3];
sx node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[2];
ecr node[0],node[7];
ecr node[2],node[3];
x node[0];
sx node[2];
sx node[3];
sx node[7];
rz(3.5*pi) node[0];
ecr node[1],node[2];
ecr node[4],node[3];
ecr node[6],node[7];
rz(0.49419637464008503*pi) node[1];
rz(2.892203782861559*pi) node[2];
rz(2.3927284844293686*pi) node[3];
x node[4];
x node[6];
rz(0.8923234673787661*pi) node[7];
x node[1];
sx node[2];
sx node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[6];
sx node[7];
rz(3.5*pi) node[1];
ecr node[4],node[3];
ecr node[6],node[7];
ecr node[1],node[2];
sx node[3];
x node[4];
x node[6];
sx node[7];
sx node[1];
x node[2];
rz(3.5*pi) node[4];
rz(3.5*pi) node[6];
rz(3.219367392908981*pi) node[1];
rz(3.5*pi) node[2];
ecr node[4],node[3];
ecr node[6],node[7];
sx node[1];
x node[3];
sx node[4];
sx node[6];
x node[7];
rz(0.5*pi) node[1];
rz(3.5*pi) node[3];
rz(3.5*pi) node[7];
sx node[1];
ecr node[3],node[4];
ecr node[7],node[6];
ecr node[2],node[1];
sx node[3];
x node[4];
x node[6];
sx node[7];
x node[1];
sx node[2];
rz(3.5*pi) node[4];
rz(3.5*pi) node[6];
rz(3.5*pi) node[1];
ecr node[4],node[3];
ecr node[6],node[7];
ecr node[1],node[2];
x node[3];
sx node[4];
x node[6];
x node[7];
sx node[1];
x node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
rz(3.5*pi) node[6];
rz(3.5*pi) node[7];
rz(3.5*pi) node[2];
rz(0.3914651124911055*pi) node[4];
x node[5];
ecr node[2],node[1];
sx node[4];
rz(3.5*pi) node[5];
sx node[1];
sx node[2];
ecr node[5],node[4];
ecr node[0],node[1];
ecr node[3],node[2];
sx node[4];
x node[5];
rz(0.49949562945626136*pi) node[0];
rz(2.892163039196129*pi) node[1];
rz(2.392668005550994*pi) node[2];
x node[3];
rz(3.5*pi) node[5];
x node[0];
sx node[1];
sx node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
rz(3.5*pi) node[0];
ecr node[3],node[2];
x node[4];
sx node[5];
ecr node[0],node[1];
sx node[2];
x node[3];
rz(3.5*pi) node[4];
sx node[0];
x node[1];
rz(3.5*pi) node[3];
ecr node[4],node[5];
rz(3.219367392908981*pi) node[0];
rz(3.5*pi) node[1];
ecr node[3],node[2];
sx node[4];
x node[5];
sx node[0];
x node[2];
sx node[3];
rz(3.5*pi) node[5];
rz(0.5*pi) node[0];
rz(3.5*pi) node[2];
ecr node[5],node[4];
sx node[0];
ecr node[2],node[3];
x node[4];
x node[5];
ecr node[1],node[0];
sx node[2];
x node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
x node[0];
sx node[1];
rz(3.5*pi) node[3];
rz(3.5*pi) node[0];
ecr node[3],node[2];
ecr node[0],node[1];
x node[2];
sx node[3];
sx node[0];
x node[1];
rz(3.5*pi) node[2];
ecr node[4],node[3];
rz(3.5*pi) node[1];
rz(2.392681374566213*pi) node[3];
x node[4];
ecr node[1],node[0];
sx node[3];
rz(3.5*pi) node[4];
sx node[0];
sx node[1];
ecr node[4],node[3];
ecr node[7],node[0];
ecr node[2],node[1];
sx node[3];
x node[4];
rz(2.89204271805915*pi) node[0];
rz(2.392733577387548*pi) node[1];
x node[2];
rz(3.5*pi) node[4];
rz(0.5051060958481597*pi) node[7];
sx node[0];
sx node[1];
rz(3.5*pi) node[2];
ecr node[4],node[3];
x node[7];
ecr node[2],node[1];
x node[3];
sx node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[1];
x node[2];
rz(3.5*pi) node[3];
x node[0];
rz(3.5*pi) node[2];
ecr node[3],node[4];
sx node[7];
rz(3.5*pi) node[0];
ecr node[2],node[1];
sx node[3];
x node[4];
rz(3.219367392908981*pi) node[7];
x node[1];
sx node[2];
rz(3.5*pi) node[4];
sx node[7];
rz(3.5*pi) node[1];
ecr node[4],node[3];
rz(0.5*pi) node[7];
ecr node[1],node[2];
x node[3];
sx node[4];
sx node[7];
ecr node[0],node[7];
sx node[1];
x node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
sx node[0];
rz(3.5*pi) node[2];
rz(0.3927533126004912*pi) node[4];
x node[5];
x node[7];
ecr node[2],node[1];
sx node[4];
rz(3.5*pi) node[5];
rz(3.5*pi) node[7];
ecr node[7],node[0];
x node[1];
sx node[2];
ecr node[5],node[4];
x node[0];
rz(3.5*pi) node[1];
ecr node[3],node[2];
sx node[4];
x node[5];
sx node[7];
rz(3.5*pi) node[0];
rz(2.3926781914673514*pi) node[2];
x node[3];
rz(3.5*pi) node[5];
ecr node[0],node[7];
sx node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
sx node[0];
ecr node[3],node[2];
x node[4];
sx node[5];
sx node[7];
ecr node[1],node[0];
sx node[2];
x node[3];
rz(3.5*pi) node[4];
ecr node[6],node[7];
rz(2.39270747597688*pi) node[0];
x node[1];
rz(3.5*pi) node[3];
ecr node[4],node[5];
rz(0.4914087758118304*pi) node[6];
rz(2.892250892724716*pi) node[7];
sx node[0];
rz(3.5*pi) node[1];
ecr node[3],node[2];
sx node[4];
x node[5];
x node[6];
sx node[7];
ecr node[1],node[0];
x node[2];
sx node[3];
rz(3.5*pi) node[5];
rz(3.5*pi) node[6];
sx node[0];
x node[1];
rz(3.5*pi) node[2];
ecr node[5],node[4];
ecr node[6],node[7];
rz(3.5*pi) node[1];
ecr node[2],node[3];
x node[4];
x node[5];
sx node[6];
rz(0.4827580362040249*pi) node[7];
ecr node[1],node[0];
sx node[2];
x node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
rz(3.219367392908981*pi) node[6];
sx node[7];
x node[0];
sx node[1];
rz(3.5*pi) node[3];
sx node[6];
rz(3.219367392908981*pi) node[7];
rz(3.5*pi) node[0];
ecr node[3],node[2];
rz(0.5*pi) node[6];
sx node[7];
ecr node[0],node[1];
x node[2];
sx node[3];
x node[6];
rz(0.5*pi) node[7];
sx node[0];
x node[1];
rz(3.5*pi) node[2];
ecr node[4],node[3];
rz(3.5*pi) node[6];
sx node[7];
rz(3.5*pi) node[1];
rz(2.392667687241108*pi) node[3];
x node[4];
ecr node[6],node[7];
ecr node[1],node[0];
sx node[3];
rz(3.5*pi) node[4];
sx node[6];
x node[7];
x node[0];
sx node[1];
ecr node[4],node[3];
rz(3.5*pi) node[7];
rz(3.5*pi) node[0];
ecr node[2],node[1];
sx node[3];
x node[4];
ecr node[7],node[6];
rz(2.3926756449882616*pi) node[1];
x node[2];
rz(3.5*pi) node[4];
x node[6];
sx node[7];
sx node[1];
rz(3.5*pi) node[2];
ecr node[4],node[3];
rz(3.5*pi) node[6];
ecr node[2],node[1];
x node[3];
sx node[4];
ecr node[6],node[7];
x node[1];
sx node[2];
rz(3.5*pi) node[3];
sx node[6];
sx node[7];
ecr node[0],node[7];
rz(3.5*pi) node[1];
ecr node[3],node[4];
x node[0];
ecr node[1],node[2];
sx node[3];
x node[4];
rz(2.392705247807677*pi) node[7];
rz(3.5*pi) node[0];
sx node[1];
x node[2];
rz(3.5*pi) node[4];
sx node[7];
ecr node[0],node[7];
rz(3.5*pi) node[2];
ecr node[4],node[3];
x node[0];
ecr node[2],node[1];
x node[3];
sx node[4];
sx node[7];
rz(3.5*pi) node[0];
x node[1];
sx node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
ecr node[0],node[7];
rz(3.5*pi) node[1];
rz(0.39272434640084786*pi) node[4];
x node[5];
sx node[0];
ecr node[1],node[2];
sx node[4];
rz(3.5*pi) node[5];
x node[7];
x node[1];
sx node[2];
ecr node[5],node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[1];
ecr node[3],node[2];
sx node[4];
x node[5];
x node[0];
rz(2.392861219651908*pi) node[2];
x node[3];
rz(3.5*pi) node[5];
sx node[7];
rz(3.5*pi) node[0];
sx node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
ecr node[0],node[7];
ecr node[3],node[2];
x node[4];
sx node[5];
sx node[0];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
x node[7];
ecr node[1],node[0];
rz(3.5*pi) node[2];
ecr node[4],node[5];
rz(3.5*pi) node[7];
rz(2.3926734168190587*pi) node[0];
x node[1];
ecr node[2],node[3];
sx node[4];
x node[5];
ecr node[7],node[6];
sx node[0];
rz(3.5*pi) node[1];
sx node[2];
x node[3];
rz(3.5*pi) node[5];
rz(2.392736442176523*pi) node[6];
rz(3.4983489988114225*pi) node[7];
ecr node[1],node[0];
rz(3.5*pi) node[3];
ecr node[5],node[4];
sx node[6];
x node[7];
x node[0];
sx node[1];
ecr node[3],node[2];
x node[4];
x node[5];
rz(3.5*pi) node[7];
rz(3.5*pi) node[0];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
ecr node[7],node[6];
ecr node[0],node[1];
rz(3.5*pi) node[2];
x node[6];
sx node[7];
sx node[0];
x node[1];
ecr node[2],node[3];
rz(3.5*pi) node[6];
rz(3.2319939025726936*pi) node[7];
rz(3.5*pi) node[1];
x node[2];
sx node[3];
sx node[7];
ecr node[1],node[0];
rz(3.5*pi) node[2];
ecr node[4],node[3];
rz(1.5*pi) node[7];
x node[0];
sx node[1];
rz(2.392702064708815*pi) node[3];
x node[4];
sx node[7];
rz(3.5*pi) node[0];
sx node[3];
rz(3.5*pi) node[4];
ecr node[6],node[7];
ecr node[0],node[1];
ecr node[4],node[3];
sx node[6];
x node[7];
x node[0];
sx node[1];
sx node[3];
x node[4];
rz(3.5*pi) node[7];
rz(3.5*pi) node[0];
ecr node[2],node[1];
rz(3.5*pi) node[4];
ecr node[7],node[6];
rz(2.3932861633499627*pi) node[1];
x node[2];
ecr node[4],node[3];
x node[6];
sx node[7];
sx node[1];
rz(3.5*pi) node[2];
x node[3];
sx node[4];
rz(3.5*pi) node[6];
ecr node[2],node[1];
rz(3.5*pi) node[3];
ecr node[6],node[7];
sx node[1];
x node[2];
ecr node[3],node[4];
sx node[6];
sx node[7];
ecr node[0],node[7];
rz(3.5*pi) node[2];
sx node[3];
x node[4];
rz(3.4994222124236796*pi) node[0];
ecr node[2],node[1];
rz(3.5*pi) node[4];
rz(2.392916605572103*pi) node[7];
x node[0];
x node[1];
sx node[2];
ecr node[4],node[3];
sx node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[1];
x node[3];
sx node[4];
ecr node[0],node[7];
ecr node[1],node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
sx node[0];
sx node[1];
x node[2];
rz(0.3926775548475794*pi) node[4];
x node[5];
x node[7];
rz(3.2319939025726936*pi) node[0];
rz(3.5*pi) node[2];
sx node[4];
rz(3.5*pi) node[5];
rz(3.5*pi) node[7];
sx node[0];
ecr node[2],node[1];
ecr node[5],node[4];
rz(1.5*pi) node[0];
x node[1];
sx node[2];
sx node[4];
x node[5];
sx node[0];
rz(3.5*pi) node[1];
ecr node[3],node[2];
rz(3.5*pi) node[5];
ecr node[7],node[0];
rz(2.3926963351308643*pi) node[2];
x node[3];
x node[0];
sx node[2];
rz(3.5*pi) node[3];
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
sx node[2];
x node[3];
sx node[0];
rz(3.5*pi) node[3];
sx node[7];
ecr node[1],node[0];
ecr node[3],node[2];
rz(2.393121278828919*pi) node[0];
rz(3.6157099175093634*pi) node[1];
x node[2];
sx node[3];
sx node[0];
x node[1];
rz(3.5*pi) node[2];
rz(3.5*pi) node[1];
ecr node[2],node[3];
ecr node[1],node[0];
x node[2];
x node[3];
sx node[0];
sx node[1];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
rz(3.2319939025726936*pi) node[1];
ecr node[3],node[4];
sx node[1];
sx node[3];
x node[4];
rz(1.5*pi) node[1];
rz(3.5*pi) node[4];
sx node[1];
ecr node[4],node[3];
ecr node[2],node[1];
x node[3];
sx node[4];
x node[1];
sx node[2];
rz(3.5*pi) node[3];
rz(3.5*pi) node[1];
ecr node[3],node[4];
ecr node[1],node[2];
x node[3];
sx node[4];
sx node[1];
x node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
rz(3.5*pi) node[2];
rz(2.3926928337221165*pi) node[4];
x node[5];
ecr node[2],node[1];
sx node[4];
rz(3.5*pi) node[5];
x node[1];
sx node[2];
ecr node[5],node[4];
rz(3.5*pi) node[1];
sx node[4];
x node[5];
ecr node[1],node[0];
ecr node[3],node[4];
rz(3.5*pi) node[5];
rz(2.392701428089043*pi) node[0];
rz(3.502185205897732*pi) node[1];
x node[3];
rz(0.3926565463950915*pi) node[4];
ecr node[5],node[6];
sx node[0];
x node[1];
rz(3.5*pi) node[3];
sx node[4];
sx node[5];
x node[6];
rz(3.5*pi) node[1];
ecr node[3],node[4];
rz(3.5*pi) node[6];
ecr node[1],node[0];
x node[3];
x node[4];
ecr node[6],node[5];
sx node[0];
sx node[1];
rz(3.5*pi) node[3];
rz(3.5*pi) node[4];
x node[5];
sx node[6];
rz(3.2319939025726936*pi) node[1];
rz(3.5*pi) node[5];
sx node[1];
ecr node[5],node[6];
rz(1.5*pi) node[1];
x node[6];
sx node[1];
rz(3.5*pi) node[6];
ecr node[6],node[7];
sx node[6];
x node[7];
rz(3.5*pi) node[7];
ecr node[7],node[6];
x node[6];
sx node[7];
rz(3.5*pi) node[6];
ecr node[6],node[7];
x node[7];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(2.392717025273466*pi) node[0];
rz(3.5001899122071776*pi) node[7];
sx node[0];
x node[7];
rz(3.5*pi) node[7];
ecr node[7],node[0];
x node[0];
sx node[7];
rz(3.5*pi) node[0];
rz(3.2319939025726936*pi) node[7];
ecr node[0],node[1];
sx node[7];
sx node[0];
x node[1];
rz(1.5*pi) node[7];
rz(3.5*pi) node[1];
ecr node[1],node[0];
x node[0];
sx node[1];
rz(3.5*pi) node[0];
ecr node[0],node[1];
x node[1];
rz(3.5*pi) node[1];
ecr node[1],node[2];
sx node[1];
x node[2];
rz(3.5*pi) node[2];
ecr node[2],node[1];
x node[1];
sx node[2];
rz(3.5*pi) node[1];
ecr node[1],node[2];
sx node[2];
ecr node[3],node[2];
rz(2.3927622252773046*pi) node[2];
rz(3.4980774168165305*pi) node[3];
sx node[2];
x node[3];
rz(3.5*pi) node[3];
ecr node[3],node[2];
sx node[2];
sx node[3];
rz(3.2319939025726936*pi) node[3];
sx node[3];
rz(1.5*pi) node[3];
sx node[3];
ecr node[4],node[3];
x node[3];
sx node[4];
rz(3.5*pi) node[3];
ecr node[3],node[4];
sx node[3];
x node[4];
rz(3.5*pi) node[4];
ecr node[4],node[3];
x node[3];
rz(3.5*pi) node[3];
ecr node[3],node[2];
rz(2.392683921045303*pi) node[2];
rz(3.5032348009164345*pi) node[3];
sx node[2];
x node[3];
rz(3.5*pi) node[3];
ecr node[3],node[2];
rz(3.5064920341507646*pi) node[2];
sx node[3];
sx node[2];
rz(3.2319939025726936*pi) node[3];
rz(3.2319939025726936*pi) node[2];
sx node[3];
sx node[2];
rz(1.5*pi) node[3];
rz(1.5*pi) node[2];
barrier node[2],node[3],node[4],node[7],node[0],node[1],node[6],node[5];
measure node[2] -> meas[0];
measure node[3] -> meas[1];
measure node[4] -> meas[2];
measure node[7] -> meas[3];
measure node[0] -> meas[4];
measure node[1] -> meas[5];
measure node[6] -> meas[6];
measure node[5] -> meas[7];
