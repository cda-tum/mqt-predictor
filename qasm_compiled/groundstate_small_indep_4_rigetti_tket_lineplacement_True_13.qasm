OPENQASM 2.0;
include "qelib1.inc";

qreg node[80];
creg meas[4];
rz(0.5*pi) node[65];
rz(0.7868820996780246*pi) node[66];
rz(3.3654418931375574*pi) node[77];
rz(0.7957693785453521*pi) node[78];
rz(0.9598248996871763*pi) node[79];
rx(0.5*pi) node[65];
rz(0.5*pi) node[66];
rx(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[65];
rx(0.5*pi) node[66];
rz(3.7470783715094114*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[66];
rz(0.5*pi) node[78];
cz node[77],node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
rz(3.5*pi) node[78];
rx(3.5*pi) node[78];
rz(3.37553913040158*pi) node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[79],node[78];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[78],node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[79],node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[77],node[78];
cz node[77],node[66];
rz(0.5*pi) node[78];
rz(0.5*pi) node[66];
rx(0.5*pi) node[77];
rx(0.5*pi) node[78];
rx(0.5*pi) node[66];
rz(0.4963691897541641*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[66];
rz(0.5*pi) node[78];
cz node[66],node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
cz node[79],node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[78];
cz node[65],node[66];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rx(0.5*pi) node[66];
rz(3.5*pi) node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[66];
rx(3.5*pi) node[78];
cz node[66],node[65];
rz(1.9542707190122357*pi) node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[65];
cz node[79],node[78];
rx(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[78],node[65];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[65];
cz node[79],node[78];
rx(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[65];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[78],node[65];
rz(0.5*pi) node[65];
rx(0.5*pi) node[65];
rz(0.5*pi) node[65];
rz(0.5*pi) node[65];
rx(0.5*pi) node[65];
rz(0.5*pi) node[65];
cz node[78],node[65];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(3.5*pi) node[65];
cz node[77],node[78];
rx(1.5*pi) node[65];
rz(0.5*pi) node[78];
rz(1.3217404624530564*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[78],node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[77],node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[78];
rx(0.5*pi) node[79];
rx(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[78];
cz node[78],node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[77],node[78];
rz(3.5*pi) node[79];
rz(0.5*pi) node[78];
rx(3.5*pi) node[79];
rx(0.5*pi) node[78];
rz(0.6200856598527547*pi) node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[79],node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
rz(3.5*pi) node[78];
rx(3.5*pi) node[78];
rz(1.212450156208937*pi) node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[65],node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
cz node[78],node[65];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rx(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
cz node[65],node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[77],node[78];
rx(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[77];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[79],node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[79];
rx(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[78];
rz(0.5*pi) node[78];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
cz node[65],node[78];
rx(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[78];
rx(0.2913704463065361*pi) node[78];
barrier node[77],node[79],node[65],node[78];
measure node[77] -> meas[0];
measure node[79] -> meas[1];
measure node[65] -> meas[2];
measure node[78] -> meas[3];
