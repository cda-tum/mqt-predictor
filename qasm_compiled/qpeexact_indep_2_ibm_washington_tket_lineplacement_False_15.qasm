OPENQASM 2.0;
include "qelib1.inc";

qreg node[2];
creg c[1];
x node[0];
cx node[0],node[1];
barrier node[1],node[0];
measure node[1] -> c[0];
