OPENQASM 2.0;
include "qelib1.inc";

qreg q[10];
creg meas[10];
rz(3.5*pi) q[0];
rz(0.7613063722314442*pi) q[1];
rz(3.959617569508955*pi) q[2];
rz(0.005138404953688669*pi) q[3];
rz(3.992199998900558*pi) q[4];
rz(3.9597561224874744*pi) q[5];
rz(3.974832867126171*pi) q[6];
rz(0.019005213493945017*pi) q[7];
rz(0.020990248173235138*pi) q[8];
rz(0.9552097961166925*pi) q[9];
rx(1.5327909561022126*pi) q[0];
rx(3.4840578661415096*pi) q[1];
rx(3.5048564615097364*pi) q[2];
rx(3.4596522913128718*pi) q[3];
rx(3.460079096274238*pi) q[4];
rx(3.5059004463074293*pi) q[5];
rx(3.531983566982842*pi) q[6];
rx(3.535979731411344*pi) q[7];
rx(3.465137809803572*pi) q[8];
rx(3.579947776383545*pi) q[9];
rz(0.26087813260603376*pi) q[0];
rz(3.517093280163416*pi) q[1];
rz(1.4621065950984455*pi) q[2];
rz(0.04043118645547428*pi) q[3];
rz(3.938411021895015*pi) q[4];
rz(1.453906769536096*pi) q[5];
rz(1.2127967412580607*pi) q[6];
rz(0.8448748107805129*pi) q[7];
rz(0.1729958743345008*pi) q[8];
rz(3.1648928390762276*pi) q[9];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[0],q[1];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
rz(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(0.19845001914517357*pi) q[0];
rx(0.5*pi) q[1];
ry(0.5*pi) q[0];
rz(2.5217562652120677*pi) q[1];
rxx(0.5*pi) q[0],q[2];
ry(0.5*pi) q[1];
ry(3.5*pi) q[0];
rx(3.5*pi) q[2];
rz(3.5*pi) q[0];
rx(0.8015499808548282*pi) q[2];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[2];
rxx(0.5*pi) q[0],q[3];
ry(3.5*pi) q[1];
rx(3.5*pi) q[2];
ry(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rx(3.5*pi) q[3];
rz(3.5*pi) q[0];
rz(1.1984500191451737*pi) q[1];
rx(0.5*pi) q[2];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
rz(2.5217562652120677*pi) q[2];
rxx(0.5*pi) q[0],q[4];
rxx(0.5*pi) q[1],q[3];
ry(0.5*pi) q[2];
ry(3.5*pi) q[0];
ry(3.5*pi) q[1];
rx(3.5*pi) q[3];
rx(3.5*pi) q[4];
rz(3.5*pi) q[0];
rz(3.5*pi) q[1];
rx(0.8015499808548282*pi) q[3];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
rxx(0.5*pi) q[2],q[3];
rxx(0.5*pi) q[0],q[5];
rxx(0.5*pi) q[1],q[4];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
ry(3.5*pi) q[0];
ry(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rx(3.5*pi) q[4];
rx(3.5*pi) q[5];
rz(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(0.19845001914517357*pi) q[2];
rx(0.5*pi) q[3];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rz(2.5217562652120677*pi) q[3];
rxx(0.5*pi) q[0],q[6];
rxx(0.5*pi) q[1],q[5];
rxx(0.5*pi) q[2],q[4];
ry(0.5*pi) q[3];
ry(3.5*pi) q[0];
ry(3.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[4];
rx(3.5*pi) q[5];
rx(3.5*pi) q[6];
rz(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rx(0.8015499808548282*pi) q[4];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rxx(0.5*pi) q[3],q[4];
rxx(0.5*pi) q[0],q[7];
rxx(0.5*pi) q[1],q[6];
rxx(0.5*pi) q[2],q[5];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
ry(3.5*pi) q[0];
ry(3.5*pi) q[1];
ry(3.5*pi) q[2];
rz(3.5*pi) q[3];
rz(3.5*pi) q[4];
rx(3.5*pi) q[5];
rx(3.5*pi) q[6];
rx(3.5*pi) q[7];
rz(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(1.1984500191451737*pi) q[3];
rx(0.5*pi) q[4];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
ry(0.5*pi) q[3];
rz(2.5217562652120677*pi) q[4];
rxx(0.5*pi) q[0],q[8];
rxx(0.5*pi) q[1],q[7];
rxx(0.5*pi) q[2],q[6];
rxx(0.5*pi) q[3],q[5];
ry(0.5*pi) q[4];
ry(3.5*pi) q[0];
ry(3.5*pi) q[1];
ry(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[5];
rx(3.5*pi) q[6];
rx(3.5*pi) q[7];
rx(3.5*pi) q[8];
rz(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rx(0.8015499808548282*pi) q[5];
rz(0.63207967500859*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
ry(0.5*pi) q[3];
rxx(0.5*pi) q[4],q[5];
rx(1.0*pi) q[0];
rxx(0.5*pi) q[1],q[8];
rxx(0.5*pi) q[2],q[7];
rxx(0.5*pi) q[3],q[6];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
rz(0.5*pi) q[0];
ry(3.5*pi) q[1];
ry(3.5*pi) q[2];
ry(3.5*pi) q[3];
rz(3.5*pi) q[4];
rz(3.5*pi) q[5];
rx(3.5*pi) q[6];
rx(3.5*pi) q[7];
rx(3.5*pi) q[8];
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rz(0.19845001914517357*pi) q[4];
rx(0.5*pi) q[5];
rxx(0.5*pi) q[0],q[9];
rz(0.21866291201504529*pi) q[1];
ry(0.5*pi) q[2];
ry(0.5*pi) q[3];
ry(0.5*pi) q[4];
rz(2.5217562652120677*pi) q[5];
ry(3.5*pi) q[0];
rx(1.0*pi) q[1];
rxx(0.5*pi) q[2],q[8];
rxx(0.5*pi) q[3],q[7];
rxx(0.5*pi) q[4],q[6];
ry(0.5*pi) q[5];
rx(3.5*pi) q[9];
rz(3.5*pi) q[0];
rz(0.5*pi) q[1];
ry(3.5*pi) q[2];
ry(3.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[6];
rx(3.5*pi) q[7];
rx(3.5*pi) q[8];
rz(3.0*pi) q[9];
rz(3.591407826759797*pi) q[0];
ry(0.5*pi) q[1];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rz(3.5*pi) q[4];
rx(0.8015499808548282*pi) q[6];
rx(0.5865832370064549*pi) q[9];
rx(0.09944962366593547*pi) q[0];
rz(0.749790423634271*pi) q[2];
ry(0.5*pi) q[3];
ry(0.5*pi) q[4];
rxx(0.5*pi) q[5],q[6];
rz(3.0*pi) q[9];
rz(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[9];
ry(0.5*pi) q[2];
rxx(0.5*pi) q[3],q[8];
rxx(0.5*pi) q[4],q[7];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
ry(3.5*pi) q[1];
ry(3.5*pi) q[3];
ry(3.5*pi) q[4];
rz(3.5*pi) q[5];
rz(3.5*pi) q[6];
rx(3.5*pi) q[7];
rx(3.5*pi) q[8];
rx(3.5*pi) q[9];
rz(3.5*pi) q[1];
rz(3.5*pi) q[3];
rz(3.5*pi) q[4];
rz(1.1984500191451737*pi) q[5];
rx(0.5*pi) q[6];
rx(3.9688724883807747*pi) q[9];
rz(3.177991063766253*pi) q[1];
rz(1.5762138201555778*pi) q[3];
ry(0.5*pi) q[4];
ry(0.5*pi) q[5];
rz(2.5217562652120677*pi) q[6];
rz(1.0*pi) q[9];
rx(0.9512929899061902*pi) q[1];
rxx(0.5*pi) q[2],q[9];
rx(1.0*pi) q[3];
rxx(0.5*pi) q[4],q[8];
rxx(0.5*pi) q[5],q[7];
ry(0.5*pi) q[6];
rz(0.5*pi) q[1];
ry(3.5*pi) q[2];
rz(0.5*pi) q[3];
ry(3.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[7];
rx(3.5*pi) q[8];
rx(3.5*pi) q[9];
rz(3.5*pi) q[2];
ry(0.5*pi) q[3];
rz(3.5*pi) q[4];
rz(3.5*pi) q[5];
rx(0.8015499808548282*pi) q[7];
rz(3.0*pi) q[9];
rz(0.7908814246145217*pi) q[2];
rz(1.077976443531481*pi) q[4];
ry(0.5*pi) q[5];
rxx(0.5*pi) q[6],q[7];
rx(2.6735766034786934*pi) q[9];
rx(3.5907106015636394*pi) q[2];
rxx(0.5*pi) q[3],q[9];
rx(1.0*pi) q[4];
rxx(0.5*pi) q[5],q[8];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
rz(0.5*pi) q[2];
ry(3.5*pi) q[3];
rz(0.5*pi) q[4];
ry(3.5*pi) q[5];
rz(3.5*pi) q[6];
rz(3.5*pi) q[7];
rx(3.5*pi) q[8];
rx(3.5*pi) q[9];
rz(3.5*pi) q[3];
ry(0.5*pi) q[4];
rz(3.5*pi) q[5];
rz(0.19845001914517357*pi) q[6];
rx(0.5*pi) q[7];
rz(3.0*pi) q[9];
rz(0.5355419719067851*pi) q[3];
rz(0.9321706667915368*pi) q[5];
ry(0.5*pi) q[6];
rz(2.5217562652120677*pi) q[7];
rx(2.501762623375904*pi) q[9];
rx(3.609644782288888*pi) q[3];
rx(1.0*pi) q[5];
rxx(0.5*pi) q[6],q[8];
ry(0.5*pi) q[7];
rz(1.0*pi) q[9];
rz(0.5*pi) q[3];
rxx(0.5*pi) q[4],q[9];
rz(0.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[8];
ry(3.5*pi) q[4];
ry(0.5*pi) q[5];
rz(3.5*pi) q[6];
rx(0.8015499808548282*pi) q[8];
rx(3.5*pi) q[9];
rz(3.5*pi) q[4];
rz(1.2319253831124946*pi) q[6];
rxx(0.5*pi) q[7],q[8];
rx(2.1458057767399437*pi) q[9];
rz(0.037304595282678044*pi) q[4];
rxx(0.5*pi) q[5],q[9];
ry(0.5*pi) q[6];
ry(3.5*pi) q[7];
rx(3.5*pi) q[8];
rx(3.9945635027949455*pi) q[4];
ry(3.5*pi) q[5];
rz(3.5*pi) q[7];
rz(3.5*pi) q[8];
rx(3.5*pi) q[9];
rz(0.5*pi) q[4];
rz(3.5*pi) q[5];
rz(1.5727465854620566*pi) q[7];
rx(3.5*pi) q[8];
rz(3.0*pi) q[9];
rz(3.891498818542744*pi) q[5];
ry(0.5*pi) q[7];
rz(1.9719326888598454*pi) q[8];
rx(1.7997547163209573*pi) q[9];
rx(0.6007921929759696*pi) q[5];
rxx(0.5*pi) q[6],q[9];
ry(0.5*pi) q[8];
rz(0.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[9];
rz(3.5*pi) q[6];
rz(1.0*pi) q[9];
rz(0.30874646513629855*pi) q[6];
rx(0.8576288167956118*pi) q[9];
rx(3.8337188017966746*pi) q[6];
rz(1.0*pi) q[9];
rz(0.5*pi) q[6];
rxx(0.5*pi) q[7],q[9];
ry(3.5*pi) q[7];
rx(3.5*pi) q[9];
rz(3.5*pi) q[7];
rz(3.0*pi) q[9];
rz(1.1663752819319098*pi) q[7];
rx(2.0664355395339684*pi) q[9];
rx(3.1708658068093833*pi) q[7];
rxx(0.5*pi) q[8],q[9];
rz(0.5*pi) q[7];
ry(3.5*pi) q[8];
rx(3.5*pi) q[9];
rz(3.5*pi) q[8];
rz(0.2682093143922395*pi) q[9];
rz(0.7671891785341212*pi) q[8];
rx(0.7031106396712814*pi) q[9];
rx(0.5060778499548029*pi) q[8];
rz(1.1874680283003602*pi) q[9];
rz(0.5*pi) q[8];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
