OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }

qreg node[8];
creg meas[5];
sx node[0];
sx node[1];
sx node[2];
rz(3.6954586693103977*pi) node[3];
sx node[7];
rz(3.158993715951444*pi) node[0];
rz(1.2435558630192505*pi) node[1];
rz(1.2171646407501067*pi) node[2];
sx node[3];
rz(1.1630567474795495*pi) node[7];
sx node[0];
sx node[1];
sx node[2];
rz(2.534977967604596*pi) node[3];
sx node[7];
rz(1.2930607375045815*pi) node[0];
rz(1.2787552959799946*pi) node[1];
rz(1.0785840836869474*pi) node[2];
sx node[3];
rz(1.1388998377944952*pi) node[7];
x node[0];
sx node[1];
sx node[2];
rz(1.5491229695280442*pi) node[3];
sx node[7];
rz(3.5*pi) node[0];
sx node[3];
ecr node[0],node[1];
x node[0];
sx node[1];
rz(3.5*pi) node[0];
ecr node[0],node[7];
x node[0];
sx node[7];
rz(3.5*pi) node[0];
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
x node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[2];
rz(3.5*pi) node[7];
ecr node[2],node[1];
x node[1];
sx node[2];
rz(3.5*pi) node[1];
ecr node[1],node[2];
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
rz(3.1031408478848284*pi) node[2];
rz(3.5*pi) node[3];
sx node[0];
x node[1];
sx node[2];
rz(3.5*pi) node[1];
rz(1.0908934389293634*pi) node[2];
ecr node[1],node[0];
sx node[2];
x node[0];
sx node[1];
ecr node[3],node[2];
rz(3.5*pi) node[0];
x node[2];
sx node[3];
ecr node[0],node[1];
rz(3.5*pi) node[2];
sx node[0];
x node[1];
ecr node[2],node[3];
ecr node[7],node[0];
rz(3.5*pi) node[1];
sx node[2];
x node[3];
sx node[0];
rz(3.5*pi) node[3];
x node[7];
ecr node[3],node[2];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[2];
x node[3];
x node[0];
ecr node[1],node[2];
rz(3.5*pi) node[3];
sx node[7];
rz(3.5*pi) node[0];
sx node[1];
sx node[2];
ecr node[0],node[7];
rz(1.0007902911273239*pi) node[1];
sx node[0];
sx node[1];
x node[7];
rz(1.1380490018349247*pi) node[1];
rz(3.5*pi) node[7];
ecr node[7],node[0];
x node[1];
x node[0];
rz(3.5*pi) node[1];
x node[7];
rz(3.5*pi) node[0];
ecr node[1],node[2];
rz(3.5*pi) node[7];
sx node[1];
x node[2];
rz(3.5*pi) node[2];
ecr node[2],node[1];
x node[1];
sx node[2];
rz(3.5*pi) node[1];
ecr node[1],node[2];
sx node[1];
sx node[2];
ecr node[0],node[1];
ecr node[3],node[2];
sx node[0];
sx node[1];
sx node[2];
x node[3];
rz(1.2347465318768513*pi) node[0];
rz(3.5*pi) node[3];
sx node[0];
ecr node[3],node[2];
rz(1.285994203918621*pi) node[0];
x node[2];
sx node[3];
x node[0];
rz(3.5*pi) node[2];
rz(3.5*pi) node[0];
ecr node[2],node[3];
ecr node[0],node[1];
sx node[2];
x node[3];
sx node[0];
x node[1];
rz(3.5*pi) node[3];
rz(3.5*pi) node[1];
ecr node[3],node[2];
ecr node[1],node[0];
x node[2];
x node[3];
x node[0];
sx node[1];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
rz(3.5*pi) node[0];
ecr node[0],node[1];
sx node[0];
sx node[1];
ecr node[7],node[0];
ecr node[2],node[1];
rz(2.2940173319238375*pi) node[0];
x node[1];
sx node[2];
sx node[7];
sx node[0];
rz(3.5*pi) node[1];
rz(1.0849903025126126*pi) node[7];
rz(0.7230326729323915*pi) node[0];
ecr node[1],node[2];
sx node[7];
sx node[0];
sx node[1];
x node[2];
rz(1.2030528939743612*pi) node[7];
rz(1.0*pi) node[0];
rz(3.5*pi) node[2];
x node[7];
sx node[0];
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
x node[1];
sx node[2];
sx node[0];
rz(3.5*pi) node[1];
ecr node[3],node[2];
x node[7];
sx node[2];
x node[3];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[3];
sx node[0];
ecr node[3],node[2];
sx node[7];
ecr node[1],node[0];
x node[2];
sx node[3];
x node[0];
sx node[1];
rz(3.5*pi) node[2];
rz(3.5*pi) node[0];
ecr node[2],node[3];
ecr node[0],node[1];
sx node[2];
x node[3];
sx node[0];
x node[1];
rz(3.5*pi) node[3];
rz(3.5*pi) node[1];
ecr node[3],node[2];
ecr node[1],node[0];
x node[2];
x node[3];
x node[0];
sx node[1];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
rz(3.5*pi) node[0];
ecr node[0],node[1];
x node[0];
sx node[1];
rz(3.5*pi) node[0];
ecr node[2],node[1];
ecr node[0],node[7];
x node[1];
sx node[2];
sx node[0];
rz(3.5*pi) node[1];
x node[7];
rz(3.077213056162078*pi) node[0];
ecr node[1],node[2];
rz(3.5*pi) node[7];
sx node[0];
sx node[1];
x node[2];
rz(1.1368691639600907*pi) node[0];
rz(3.5*pi) node[2];
sx node[0];
ecr node[2],node[1];
ecr node[7],node[0];
x node[1];
sx node[2];
x node[0];
rz(3.5*pi) node[1];
sx node[7];
rz(3.5*pi) node[0];
ecr node[1],node[2];
ecr node[0],node[7];
x node[1];
sx node[2];
sx node[0];
rz(3.5*pi) node[1];
ecr node[3],node[2];
x node[7];
sx node[2];
x node[3];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[3];
sx node[0];
ecr node[3],node[2];
x node[7];
ecr node[1],node[0];
x node[2];
sx node[3];
rz(3.5*pi) node[7];
x node[0];
sx node[1];
rz(3.5*pi) node[2];
rz(3.5*pi) node[0];
rz(1.1865312707968014*pi) node[1];
ecr node[2],node[3];
sx node[1];
sx node[2];
x node[3];
rz(1.256194974571357*pi) node[1];
rz(3.5*pi) node[3];
sx node[1];
ecr node[3],node[2];
ecr node[0],node[1];
x node[2];
x node[3];
sx node[0];
x node[1];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
rz(3.5*pi) node[1];
ecr node[1],node[0];
x node[0];
sx node[1];
rz(3.5*pi) node[0];
ecr node[0],node[1];
sx node[0];
sx node[1];
ecr node[7],node[0];
ecr node[2],node[1];
sx node[0];
x node[1];
sx node[2];
x node[7];
rz(3.5*pi) node[1];
rz(1.0721256238427603*pi) node[2];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[2];
x node[0];
rz(1.2878228305527697*pi) node[2];
sx node[7];
rz(3.5*pi) node[0];
sx node[2];
ecr node[0],node[7];
ecr node[1],node[2];
sx node[0];
sx node[1];
x node[2];
x node[7];
rz(3.5*pi) node[2];
rz(3.5*pi) node[7];
ecr node[7],node[0];
ecr node[2],node[1];
x node[0];
x node[1];
sx node[2];
x node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[1];
rz(3.5*pi) node[7];
ecr node[1],node[2];
sx node[1];
sx node[2];
ecr node[0],node[1];
ecr node[3],node[2];
x node[0];
sx node[1];
rz(2.0017774447281127*pi) node[2];
sx node[3];
rz(3.5*pi) node[0];
sx node[2];
rz(1.2834722092255957*pi) node[3];
ecr node[0],node[1];
rz(0.8021312440714767*pi) node[2];
sx node[3];
sx node[0];
x node[1];
sx node[2];
rz(1.27807358442915*pi) node[3];
rz(3.5*pi) node[1];
rz(1.0*pi) node[2];
x node[3];
ecr node[1],node[0];
sx node[2];
rz(3.5*pi) node[3];
x node[0];
sx node[1];
ecr node[3],node[2];
rz(3.5*pi) node[0];
x node[2];
sx node[3];
ecr node[0],node[1];
rz(3.5*pi) node[2];
sx node[0];
x node[1];
ecr node[2],node[3];
ecr node[7],node[0];
rz(3.5*pi) node[1];
sx node[2];
x node[3];
sx node[0];
rz(3.5*pi) node[3];
x node[7];
ecr node[3],node[2];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[2];
sx node[3];
x node[0];
ecr node[1],node[2];
sx node[7];
rz(3.5*pi) node[0];
x node[1];
sx node[2];
ecr node[0],node[7];
rz(3.5*pi) node[1];
sx node[0];
ecr node[1],node[2];
x node[7];
sx node[1];
x node[2];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[2];
x node[0];
ecr node[2],node[1];
x node[7];
rz(3.5*pi) node[0];
x node[1];
sx node[2];
rz(3.5*pi) node[7];
rz(3.5*pi) node[1];
ecr node[1],node[2];
sx node[1];
x node[2];
ecr node[0],node[1];
rz(3.5*pi) node[2];
sx node[0];
sx node[1];
ecr node[2],node[3];
ecr node[7],node[0];
sx node[2];
x node[3];
x node[0];
rz(3.2783365465923247*pi) node[2];
rz(3.5*pi) node[3];
x node[7];
rz(3.5*pi) node[0];
sx node[2];
rz(3.5*pi) node[7];
ecr node[0],node[1];
rz(1.0674069853575774*pi) node[2];
sx node[0];
sx node[1];
sx node[2];
ecr node[7],node[0];
ecr node[3],node[2];
x node[0];
x node[2];
sx node[3];
x node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[2];
rz(3.5*pi) node[7];
ecr node[0],node[1];
ecr node[2],node[3];
x node[0];
sx node[1];
sx node[2];
x node[3];
rz(3.5*pi) node[0];
rz(3.5*pi) node[3];
ecr node[3],node[2];
x node[2];
rz(3.5*pi) node[2];
ecr node[2],node[1];
x node[1];
sx node[2];
rz(3.5*pi) node[1];
ecr node[1],node[2];
sx node[1];
x node[2];
rz(3.5*pi) node[2];
ecr node[2],node[1];
sx node[1];
x node[2];
ecr node[0],node[1];
rz(3.5*pi) node[2];
sx node[0];
sx node[1];
rz(3.284999014425665*pi) node[0];
sx node[0];
rz(1.2742903409248028*pi) node[0];
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
x node[0];
rz(3.5*pi) node[0];
ecr node[0],node[1];
sx node[0];
sx node[1];
rz(3.2547502901388246*pi) node[0];
ecr node[2],node[1];
sx node[0];
rz(3.5720675609364227*pi) node[1];
sx node[2];
rz(1.1941560498949537*pi) node[0];
sx node[1];
rz(3.2136281510695284*pi) node[2];
rz(3.5*pi) node[1];
sx node[2];
sx node[1];
rz(1.2336833800400968*pi) node[2];
rz(1.7110080023399994*pi) node[1];
barrier node[3],node[7],node[0],node[2],node[1];
measure node[3] -> meas[0];
measure node[7] -> meas[1];
measure node[0] -> meas[2];
measure node[2] -> meas[3];
measure node[1] -> meas[4];
