OPENQASM 2.0;
include "qelib1.inc";

qreg q[11];
creg meas[11];
rz(3.5*pi) q[0];
rz(0.8236025709458832*pi) q[1];
rz(3.9855709987400725*pi) q[2];
rz(0.034292858559582085*pi) q[3];
rz(3.9718312533223274*pi) q[4];
rz(0.017813959748521513*pi) q[5];
rz(3.9737126086470576*pi) q[6];
rz(3.9659172022000115*pi) q[7];
rz(0.0393698662259061*pi) q[8];
rz(0.03513710097418821*pi) q[9];
rz(1.0409424484699334*pi) q[10];
rx(2.985524395233791*pi) q[0];
rx(3.329216688346824*pi) q[1];
rx(3.5380393920102944*pi) q[2];
rx(3.4780897647869744*pi) q[3];
rx(3.5293764557496203*pi) q[4];
rx(3.5365822462632948*pi) q[5];
rx(3.531070393406503*pi) q[6];
rx(3.477763682529911*pi) q[7];
rx(3.4897653730017133*pi) q[8];
rx(3.520525506809432*pi) q[9];
rx(3.3845267311244105*pi) q[10];
rz(0.26087813260603376*pi) q[0];
rz(3.719754303546117*pi) q[1];
rz(1.1157274259279588*pi) q[2];
rz(0.3197006822637707*pi) q[3];
rz(1.2439622702422382*pi) q[4];
rz(0.8553479573773558*pi) q[5];
rz(1.2241246685264349*pi) q[6];
rz(3.683330925845671*pi) q[7];
rz(0.41945285104038743*pi) q[8];
rz(0.6676263302073825*pi) q[9];
rz(0.11125219519842955*pi) q[10];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[0],q[1];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
rz(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(1.1984500191451737*pi) q[0];
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
rz(0.19845001914517357*pi) q[1];
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
rz(1.1984500191451737*pi) q[2];
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
rz(0.19845001914517357*pi) q[3];
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
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
ry(0.5*pi) q[3];
rxx(0.5*pi) q[4],q[5];
rxx(0.5*pi) q[0],q[9];
rxx(0.5*pi) q[1],q[8];
rxx(0.5*pi) q[2],q[7];
rxx(0.5*pi) q[3],q[6];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
ry(3.5*pi) q[0];
ry(3.5*pi) q[1];
ry(3.5*pi) q[2];
ry(3.5*pi) q[3];
rz(3.5*pi) q[4];
rz(3.5*pi) q[5];
rx(3.5*pi) q[6];
rx(3.5*pi) q[7];
rx(3.5*pi) q[8];
rx(3.5*pi) q[9];
rz(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rz(1.1984500191451737*pi) q[4];
rx(0.5*pi) q[5];
rz(0.6628851589317832*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
ry(0.5*pi) q[3];
ry(0.5*pi) q[4];
rz(2.5217562652120677*pi) q[5];
rx(1.0*pi) q[0];
rxx(0.5*pi) q[1],q[9];
rxx(0.5*pi) q[2],q[8];
rxx(0.5*pi) q[3],q[7];
rxx(0.5*pi) q[4],q[6];
ry(0.5*pi) q[5];
rz(0.5*pi) q[0];
ry(3.5*pi) q[1];
ry(3.5*pi) q[2];
ry(3.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[6];
rx(3.5*pi) q[7];
rx(3.5*pi) q[8];
rx(3.5*pi) q[9];
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rz(3.5*pi) q[4];
rx(0.8015499808548282*pi) q[6];
rxx(0.5*pi) q[0],q[10];
rz(1.66841418627744*pi) q[1];
ry(0.5*pi) q[2];
ry(0.5*pi) q[3];
ry(0.5*pi) q[4];
rxx(0.5*pi) q[5],q[6];
ry(3.5*pi) q[0];
rx(1.0*pi) q[1];
rxx(0.5*pi) q[2],q[9];
rxx(0.5*pi) q[3],q[8];
rxx(0.5*pi) q[4],q[7];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
rx(3.5*pi) q[10];
rz(3.5*pi) q[0];
rz(0.5*pi) q[1];
ry(3.5*pi) q[2];
ry(3.5*pi) q[3];
ry(3.5*pi) q[4];
rz(3.5*pi) q[5];
rz(3.5*pi) q[6];
rx(3.5*pi) q[7];
rx(3.5*pi) q[8];
rx(3.5*pi) q[9];
rz(3.0*pi) q[10];
rz(3.6222133106829904*pi) q[0];
ry(0.5*pi) q[1];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rz(3.5*pi) q[4];
rz(0.19845001914517357*pi) q[5];
rx(0.5*pi) q[6];
rx(0.0055290273456571025*pi) q[10];
rx(0.5802018540530872*pi) q[0];
rz(0.08718578815388767*pi) q[2];
ry(0.5*pi) q[3];
ry(0.5*pi) q[4];
ry(0.5*pi) q[5];
rz(2.5217562652120677*pi) q[6];
rz(1.0*pi) q[10];
rz(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[10];
rx(1.0*pi) q[2];
rxx(0.5*pi) q[3],q[9];
rxx(0.5*pi) q[4],q[8];
rxx(0.5*pi) q[5],q[7];
ry(0.5*pi) q[6];
ry(3.5*pi) q[1];
rz(0.5*pi) q[2];
ry(3.5*pi) q[3];
ry(3.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[7];
rx(3.5*pi) q[8];
rx(3.5*pi) q[9];
rx(3.5*pi) q[10];
rz(3.5*pi) q[1];
ry(0.5*pi) q[2];
rz(3.5*pi) q[3];
rz(3.5*pi) q[4];
rz(3.5*pi) q[5];
rx(0.8015499808548282*pi) q[7];
rx(3.5812283981235526*pi) q[10];
rz(0.6277423380286478*pi) q[1];
rxx(0.5*pi) q[2],q[10];
rz(1.6649748757730274*pi) q[3];
ry(0.5*pi) q[4];
ry(0.5*pi) q[5];
rxx(0.5*pi) q[6],q[7];
rx(0.9553706891625662*pi) q[1];
ry(3.5*pi) q[2];
rx(1.0*pi) q[3];
rxx(0.5*pi) q[4],q[9];
rxx(0.5*pi) q[5],q[8];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
rx(3.5*pi) q[10];
rz(0.5*pi) q[1];
rz(3.5*pi) q[2];
rz(0.5*pi) q[3];
ry(3.5*pi) q[4];
ry(3.5*pi) q[5];
rz(3.5*pi) q[6];
rz(3.5*pi) q[7];
rx(3.5*pi) q[8];
rx(3.5*pi) q[9];
rx(1.4222109123808602*pi) q[10];
rz(3.0465139399050956*pi) q[2];
ry(0.5*pi) q[3];
rz(3.5*pi) q[4];
rz(3.5*pi) q[5];
rz(1.1984500191451737*pi) q[6];
rx(0.5*pi) q[7];
rx(0.9505130668911815*pi) q[2];
rxx(0.5*pi) q[3],q[10];
rz(0.2521822655084789*pi) q[4];
ry(0.5*pi) q[5];
ry(0.5*pi) q[6];
rz(2.5217562652120677*pi) q[7];
rz(0.5*pi) q[2];
ry(3.5*pi) q[3];
rx(1.0*pi) q[4];
rxx(0.5*pi) q[5],q[9];
rxx(0.5*pi) q[6],q[8];
ry(0.5*pi) q[7];
rx(3.5*pi) q[10];
rz(3.5*pi) q[3];
rz(0.5*pi) q[4];
ry(3.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[8];
rx(3.5*pi) q[9];
rz(3.0*pi) q[10];
rz(0.6243030275242349*pi) q[3];
ry(0.5*pi) q[4];
rz(3.5*pi) q[5];
rz(3.5*pi) q[6];
rx(0.8015499808548282*pi) q[8];
rx(0.5872073897354516*pi) q[10];
rx(0.8249239946166794*pi) q[3];
rz(3.7751494640089893*pi) q[5];
ry(0.5*pi) q[6];
rxx(0.5*pi) q[7],q[8];
rz(3.0*pi) q[10];
rz(0.5*pi) q[3];
rxx(0.5*pi) q[4],q[10];
rx(1.0*pi) q[5];
rxx(0.5*pi) q[6],q[9];
ry(3.5*pi) q[7];
rx(3.5*pi) q[8];
ry(3.5*pi) q[4];
rz(0.5*pi) q[5];
ry(3.5*pi) q[6];
rz(3.5*pi) q[7];
rz(3.5*pi) q[8];
rx(3.5*pi) q[9];
rx(3.5*pi) q[10];
rz(3.5*pi) q[4];
ry(0.5*pi) q[5];
rz(3.5*pi) q[6];
rz(0.19845001914517357*pi) q[7];
rx(0.5*pi) q[8];
rz(1.0*pi) q[10];
rz(3.2115104172596864*pi) q[4];
rz(0.9775261536108082*pi) q[6];
ry(0.5*pi) q[7];
rz(2.5217562652120677*pi) q[8];
rx(0.5229671985005107*pi) q[10];
rx(3.898642845526178*pi) q[4];
rx(1.0*pi) q[6];
rxx(0.5*pi) q[7],q[9];
ry(0.5*pi) q[8];
rz(3.0*pi) q[10];
rz(0.5*pi) q[4];
rxx(0.5*pi) q[5],q[10];
rz(0.5*pi) q[6];
ry(3.5*pi) q[7];
rx(3.5*pi) q[9];
ry(3.5*pi) q[5];
ry(0.5*pi) q[6];
rz(3.5*pi) q[7];
rx(0.8015499808548282*pi) q[9];
rx(3.5*pi) q[10];
rz(3.5*pi) q[5];
rz(2.079729865017994*pi) q[7];
rxx(0.5*pi) q[8],q[9];
rx(3.797623310398181*pi) q[10];
rz(2.7344776157601967*pi) q[5];
rxx(0.5*pi) q[6],q[10];
ry(0.5*pi) q[7];
ry(3.5*pi) q[8];
rx(3.5*pi) q[9];
rx(3.08836465290633*pi) q[5];
ry(3.5*pi) q[6];
rz(3.5*pi) q[8];
rz(3.5*pi) q[9];
rx(3.5*pi) q[10];
rz(0.5*pi) q[5];
rz(3.5*pi) q[6];
rz(2.6618946686670566*pi) q[8];
rx(0.5*pi) q[9];
rz(3.0*pi) q[10];
rz(3.9368543053620155*pi) q[6];
ry(0.5*pi) q[8];
rz(2.000024788814316*pi) q[9];
rx(1.6022037114071868*pi) q[10];
rx(0.5330549466961257*pi) q[6];
rxx(0.5*pi) q[7],q[10];
ry(0.5*pi) q[9];
rz(0.5*pi) q[6];
ry(3.5*pi) q[7];
rx(3.5*pi) q[10];
rz(3.5*pi) q[7];
rx(2.383714784503888*pi) q[10];
rz(3.4609419832307986*pi) q[7];
rxx(0.5*pi) q[8],q[10];
rx(0.6881349700678471*pi) q[7];
ry(3.5*pi) q[8];
rx(3.5*pi) q[10];
rz(0.5*pi) q[7];
rz(3.5*pi) q[8];
rx(1.8163738549351915*pi) q[10];
rz(0.07722719872691008*pi) q[8];
rxx(0.5*pi) q[9],q[10];
rx(0.08017838579096173*pi) q[8];
ry(3.5*pi) q[9];
rx(3.5*pi) q[10];
rz(0.5*pi) q[8];
rz(3.5*pi) q[9];
rz(3.54820389250242*pi) q[10];
rz(0.26085334379171865*pi) q[9];
rx(0.4554413547060331*pi) q[10];
rx(0.919717219990689*pi) q[9];
rz(1.2357607808278752*pi) q[10];
rz(0.5*pi) q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
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
measure q[10] -> meas[10];
