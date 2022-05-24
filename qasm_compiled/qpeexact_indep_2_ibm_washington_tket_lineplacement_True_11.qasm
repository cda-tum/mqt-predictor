OPENQASM 2.0;
include "qelib1.inc";

qreg node[126];
creg c[1];
x node[124];
cx node[124],node[125];
barrier node[125],node[124];
measure node[125] -> c[0];
