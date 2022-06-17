OPENQASM 2.0;
include "qelib1.inc";

qreg node[26];
creg c[1];
x node[25];
cx node[25],node[24];
barrier node[24],node[25];
measure node[24] -> c[0];
