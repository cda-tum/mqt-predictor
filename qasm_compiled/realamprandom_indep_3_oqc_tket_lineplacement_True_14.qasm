OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }

qreg node[7];
creg meas[3];
rz(0.5*pi) node[4];
sx node[5];
sx node[6];
sx node[4];
rz(3.2914810222071806*pi) node[5];
rz(1.2126094028056098*pi) node[6];
rz(3.5*pi) node[4];
sx node[5];
sx node[6];
sx node[4];
rz(1.0*pi) node[5];
rz(1.0*pi) node[6];
rz(0.7151980943503662*pi) node[4];
sx node[5];
x node[6];
sx node[4];
rz(3.5*pi) node[6];
ecr node[6],node[5];
sx node[5];
x node[6];
rz(3.5*pi) node[6];
ecr node[6],node[5];
x node[5];
x node[6];
rz(3.5*pi) node[5];
rz(3.5*pi) node[6];
ecr node[5],node[4];
sx node[4];
sx node[5];
ecr node[6],node[5];
x node[5];
sx node[6];
rz(3.5*pi) node[5];
rz(1.1264772388844388*pi) node[6];
ecr node[5],node[4];
sx node[6];
sx node[4];
x node[5];
rz(1.0*pi) node[6];
rz(3.5*pi) node[5];
x node[6];
ecr node[5],node[4];
rz(3.5*pi) node[6];
rz(0.1286703368898976*pi) node[4];
sx node[5];
sx node[4];
rz(3.0992367923798*pi) node[5];
sx node[5];
rz(1.0*pi) node[5];
sx node[5];
ecr node[6],node[5];
sx node[5];
x node[6];
rz(3.5*pi) node[6];
ecr node[6],node[5];
x node[5];
x node[6];
rz(3.5*pi) node[5];
rz(3.5*pi) node[6];
ecr node[5],node[4];
sx node[4];
sx node[5];
ecr node[6],node[5];
x node[5];
sx node[6];
rz(3.5*pi) node[5];
rz(1.18801602208455*pi) node[6];
ecr node[5],node[4];
sx node[6];
sx node[4];
x node[5];
rz(1.0*pi) node[6];
rz(3.5*pi) node[5];
x node[6];
ecr node[5],node[4];
rz(3.5*pi) node[6];
rz(0.14355193570618752*pi) node[4];
sx node[5];
sx node[4];
rz(3.3036345279996873*pi) node[5];
sx node[5];
rz(1.0*pi) node[5];
sx node[5];
ecr node[6],node[5];
sx node[5];
x node[6];
rz(3.5*pi) node[6];
ecr node[6],node[5];
x node[5];
x node[6];
rz(3.5*pi) node[5];
rz(3.5*pi) node[6];
ecr node[5],node[4];
sx node[4];
sx node[5];
ecr node[6],node[5];
x node[5];
sx node[6];
rz(3.5*pi) node[5];
rz(3.149050127322802*pi) node[6];
ecr node[5],node[4];
sx node[6];
sx node[4];
x node[5];
rz(1.0*pi) node[6];
rz(3.5*pi) node[5];
ecr node[5],node[4];
rz(3.5328215023058984*pi) node[4];
sx node[5];
sx node[4];
rz(3.234076017758947*pi) node[5];
rz(3.5*pi) node[4];
sx node[5];
sx node[4];
rz(1.0*pi) node[5];
rz(1.5*pi) node[4];
barrier node[6],node[5],node[4];
measure node[6] -> meas[0];
measure node[5] -> meas[1];
measure node[4] -> meas[2];
