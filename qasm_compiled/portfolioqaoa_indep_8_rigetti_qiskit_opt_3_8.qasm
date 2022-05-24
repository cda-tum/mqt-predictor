OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg meas[8];
rz(-pi/2) q[10];
rz(pi/2) q[19];
rx(pi/2) q[19];
rz(3.3989164) q[19];
rz(pi/2) q[20];
rx(pi/2) q[20];
rx(pi/2) q[21];
rz(-pi) q[21];
rz(pi/2) q[56];
rx(pi/2) q[56];
rz(2.992173) q[56];
rz(-pi) q[57];
rx(-2.3098172) q[57];
cz q[56],q[57];
rx(pi) q[56];
rx(1.1952098) q[57];
cz q[56],q[57];
rz(-0.14941968) q[56];
rx(-pi) q[56];
rx(-2.3098172) q[57];
rx(-2.9863321) q[63];
rz(-pi) q[63];
cz q[56],q[63];
rz(pi/2) q[56];
rx(pi) q[56];
rx(1.1953668) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi/2) q[56];
cz q[19],q[56];
rx(pi) q[19];
rx(1.1956518) q[56];
rz(-pi/2) q[56];
cz q[19],q[56];
rz(-2.8842689) q[19];
rx(pi/2) q[19];
rz(-pi/2) q[19];
rx(-pi/2) q[56];
cz q[56],q[57];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[57];
rx(2.9863321) q[63];
cz q[56],q[63];
rx(-1.2012998) q[63];
cz q[56],q[63];
cz q[56],q[19];
rx(-1.1955878) q[19];
cz q[56],q[19];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rx(pi/2) q[63];
cz q[20],q[63];
rx(-pi/2) q[20];
rz(-pi/2) q[20];
rx(pi/2) q[63];
cz q[20],q[63];
cz q[20],q[19];
rx(-1.1952508) q[19];
cz q[20],q[19];
rz(-pi/2) q[19];
rx(-pi/2) q[19];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
rz(-pi) q[70];
rx(-2.4932607) q[70];
cz q[57],q[70];
rz(pi/2) q[57];
rx(pi) q[57];
rx(1.1953438) q[70];
cz q[57],q[70];
rx(-pi) q[57];
rz(-pi/2) q[57];
cz q[57],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
rz(-0.99424327) q[57];
cz q[57],q[56];
rx(2.6120455) q[56];
rz(-pi/2) q[56];
cz q[56],q[63];
rx(pi/2) q[56];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-0.37533056) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
cz q[56],q[63];
rz(2.1003434) q[63];
rx(-pi/2) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(-pi/2) q[63];
rx(pi/2) q[63];
cz q[20],q[63];
rx(-pi/2) q[20];
rz(pi/2) q[20];
rx(pi/2) q[63];
cz q[20],q[63];
rx(-pi/2) q[20];
rz(0.88720053) q[20];
rx(-pi/2) q[20];
cz q[20],q[21];
rx(pi/2) q[20];
rx(pi/2) q[21];
cz q[20],q[21];
rx(-0.37531956) q[20];
rz(3*pi/2) q[20];
rx(pi/2) q[21];
cz q[20],q[21];
rz(2.1173217) q[21];
rx(-pi/2) q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rz(-pi/2) q[21];
rx(pi/2) q[21];
cz q[10],q[21];
rx(0.37546656) q[10];
rz(-1.1622416) q[10];
rx(pi/2) q[21];
cz q[10],q[21];
rx(1.7073201) q[10];
rz(-0.30459779) q[10];
rz(pi/2) q[21];
rx(pi/2) q[21];
rz(pi) q[63];
rx(-2.4932607) q[70];
cz q[57],q[70];
rx(-1.1956028) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(-pi/2) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-0.37518056) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(-0.57655306) q[56];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(-pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(2.677945) q[56];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rx(-0.29407223) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rx(pi/2) q[63];
cz q[20],q[63];
rx(-0.37516956) q[20];
rz(pi/2) q[20];
rx(pi/2) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(0.69883136) q[20];
rx(pi/2) q[20];
cz q[20],q[21];
rx(pi/2) q[20];
rz(-pi/2) q[21];
rx(pi/2) q[21];
cz q[20],q[21];
rx(-0.37631256) q[20];
rz(pi) q[20];
rx(pi/2) q[21];
cz q[20],q[21];
rx(-pi/2) q[20];
rz(-1.9782683) q[21];
rx(pi/2) q[21];
rz(-2.0003376) q[21];
cz q[10],q[21];
rx(2.8024358) q[21];
cz q[10],q[21];
rx(-pi/2) q[21];
rz(-pi/2) q[63];
rx(pi/2) q[63];
rx(pi/2) q[70];
rz(pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(-pi/2) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(0.37514356) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(pi) q[56];
cz q[19],q[56];
rx(-1.1953858) q[56];
cz q[19],q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
cz q[56],q[63];
rx(pi/2) q[56];
rz(-3.1437876) q[57];
rz(-pi/2) q[63];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-pi/2) q[56];
rz(3*pi/2) q[56];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-pi/2) q[63];
rx(-pi/2) q[70];
rz(-pi/2) q[70];
cz q[57],q[70];
rx(-1.1947428) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(-pi/2) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-0.37804756) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-1.1049537) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi) q[19];
rz(-pi/2) q[56];
rx(-pi/2) q[56];
cz q[56],q[19];
rx(pi/2) q[19];
rz(1.6867597) q[19];
rx(pi/2) q[56];
rz(pi/2) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
cz q[19],q[20];
rx(pi/2) q[19];
rx(pi/2) q[20];
cz q[19],q[20];
rx(0.37727356) q[19];
rz(pi/2) q[19];
rx(pi/2) q[20];
cz q[19],q[20];
rx(-pi) q[19];
rx(-0.46630692) q[20];
rz(1.957133) q[20];
rx(2.9541656) q[20];
cz q[20],q[21];
rx(pi/2) q[20];
rx(pi/2) q[21];
cz q[20],q[21];
rx(-pi/2) q[20];
rz(pi/2) q[20];
rx(pi/2) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rz(pi/2) q[21];
cz q[10],q[21];
rx(2.8027158) q[21];
cz q[10],q[21];
cz q[20],q[21];
rx(2.8132588) q[21];
cz q[20],q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rz(pi) q[10];
rz(-pi/2) q[21];
rx(-pi/2) q[21];
cz q[21],q[10];
rx(pi/2) q[10];
rz(pi) q[10];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[10],q[21];
rx(pi/2) q[21];
rz(pi/2) q[21];
rx(-1.3734008) q[56];
rz(pi/2) q[56];
rx(-pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-0.37530556) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rz(0.19739556) q[57];
rx(pi) q[57];
cz q[63],q[56];
rx(-1.1956068) q[56];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi/2) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rx(pi/2) q[56];
cz q[19],q[56];
rx(-pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[56];
cz q[19],q[56];
rx(-pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
rz(pi/2) q[70];
cz q[57],q[70];
rx(-1.1955178) q[70];
cz q[57],q[70];
rz(-2.5850961) q[57];
rx(-pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(-pi/2) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-0.37530256) q[56];
rz(4.3393089) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(1.7888576) q[56];
rz(3.6465152) q[56];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(-pi/2) q[57];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(-1.3435744) q[56];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rx(-pi/2) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rx(pi/2) q[63];
cz q[20],q[63];
rx(-pi/2) q[20];
rx(pi/2) q[63];
cz q[20],q[63];
cz q[21],q[20];
rx(2.8032218) q[20];
cz q[21],q[20];
rx(-pi/2) q[21];
rx(-pi/2) q[63];
rz(-pi/2) q[63];
cz q[63],q[20];
rx(2.8031088) q[20];
cz q[63],q[20];
rz(-pi/2) q[20];
cz q[20],q[21];
rx(pi/2) q[20];
rx(pi/2) q[21];
cz q[20],q[21];
rx(-pi/2) q[20];
rz(3.9743994) q[20];
rx(pi/2) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rx(-pi/2) q[21];
rz(-pi/2) q[21];
cz q[10],q[21];
rx(2.8025108) q[21];
cz q[10],q[21];
rx(-pi/2) q[21];
rx(pi/2) q[70];
rz(pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(-pi/2) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-0.37526256) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
cz q[19],q[56];
rx(-1.1957048) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(-pi/2) q[56];
rz(-0.22722191) q[57];
rx(-pi/2) q[70];
rz(-pi/2) q[70];
cz q[57],q[70];
rx(-1.1954218) q[70];
cz q[57],q[70];
rx(1.1412552) q[57];
rz(1.5711677) q[57];
rx(0.00081067088) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(-pi/2) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(-pi/2) q[56];
rx(pi/2) q[56];
cz q[19],q[56];
rx(-pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(pi) q[19];
cz q[19],q[20];
rx(pi/2) q[19];
rz(-pi/2) q[20];
rx(pi/2) q[20];
cz q[19],q[20];
rx(1.2318785) q[19];
rz(pi/2) q[19];
rx(pi/2) q[20];
cz q[19],q[20];
rx(-pi/2) q[19];
rz(-3.9743994) q[19];
rx(-pi/2) q[20];
rz(pi/2) q[20];
rz(3*pi/2) q[56];
cz q[63],q[20];
rx(2.8031348) q[20];
cz q[63],q[20];
rz(pi/2) q[20];
rx(pi/2) q[20];
cz q[20],q[21];
rx(pi/2) q[20];
rx(pi/2) q[21];
cz q[20],q[21];
rx(-pi/2) q[20];
rz(pi/2) q[20];
rx(pi/2) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rz(-pi/2) q[20];
rz(pi/2) q[21];
cz q[10],q[21];
rx(2.8032238) q[21];
cz q[10],q[21];
cz q[20],q[21];
rx(2.8027498) q[21];
cz q[20],q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rz(pi) q[10];
rx(-pi/2) q[20];
rz(-pi/2) q[21];
rx(-pi/2) q[21];
cz q[21],q[10];
rx(pi/2) q[10];
rz(0.96922702) q[10];
rx(pi/2) q[21];
rz(pi) q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rx(pi/2) q[21];
rz(pi/2) q[21];
rx(-pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(1.172541) q[57];
cz q[56],q[57];
rx(pi) q[56];
rx(1.1952088) q[57];
rz(pi/2) q[57];
cz q[56],q[57];
rz(-1.5617692) q[56];
rx(pi/2) q[56];
rz(1.1412551) q[56];
cz q[19],q[56];
rx(2.8028928) q[56];
cz q[19],q[56];
rx(pi/2) q[57];
rz(4.3513748) q[57];
cz q[63],q[56];
rx(2.8031588) q[56];
cz q[63],q[56];
rx(-pi/2) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rx(pi/2) q[56];
cz q[19],q[56];
rx(-pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
cz q[19],q[20];
rx(pi/2) q[19];
rz(-pi/2) q[20];
rx(pi/2) q[20];
cz q[19],q[20];
rx(-pi/2) q[19];
rz(pi/2) q[19];
rx(pi/2) q[20];
cz q[19],q[20];
rx(pi/2) q[19];
rz(pi/2) q[19];
rz(pi/2) q[20];
cz q[21],q[20];
rx(2.8016078) q[20];
cz q[21],q[20];
cz q[19],q[20];
rx(2.8029368) q[20];
cz q[19],q[20];
rz(-pi/2) q[20];
rx(-pi/2) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rx(pi/2) q[21];
cz q[20],q[21];
rx(-pi/2) q[20];
rz(3*pi/2) q[20];
rx(pi/2) q[21];
cz q[20],q[21];
rz(-pi/2) q[21];
rx(-pi/2) q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rz(-pi/2) q[21];
rx(pi/2) q[21];
cz q[10],q[21];
rx(1.2323465) q[10];
rz(pi/2) q[10];
rx(pi/2) q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rx(-pi/2) q[21];
rz(pi/2) q[21];
rx(2.5400233) q[21];
rx(-pi/2) q[56];
rz(pi) q[56];
rx(-pi/2) q[63];
rz(-pi/2) q[70];
rx(pi/2) q[70];
cz q[57],q[70];
rx(pi) q[57];
rx(1.1955768) q[70];
cz q[57],q[70];
rz(-1.5640376) q[57];
rx(-1.1412551) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(-pi/2) q[57];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[57];
rz(-pi/2) q[57];
rx(1.5859849) q[70];
rz(2.7120514) q[70];
cz q[57],q[70];
rx(2.8029108) q[70];
cz q[57],q[70];
cz q[57],q[56];
rx(2.8026498) q[56];
cz q[57],q[56];
rz(-pi/2) q[56];
cz q[56],q[63];
rx(pi/2) q[56];
rx(-2.4523831) q[57];
rz(1.560165) q[57];
rx(3.1328327) q[57];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-pi/2) q[56];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-pi/2) q[63];
rz(-pi/2) q[63];
rx(-pi/2) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
rx(-pi/2) q[57];
rx(pi/2) q[70];
cz q[57],q[70];
cz q[56],q[57];
rx(2.8031768) q[57];
cz q[56],q[57];
cz q[56],q[63];
rx(-pi/2) q[57];
rx(2.8011478) q[63];
cz q[56],q[63];
rx(-2.452424) q[56];
rz(1.5670758) q[56];
rx(3.1385272) q[56];
cz q[56],q[57];
rx(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rz(pi/2) q[57];
rx(-pi/2) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rx(pi/2) q[63];
cz q[20],q[63];
rx(-pi/2) q[20];
rz(pi/2) q[20];
rx(pi/2) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(1.2803412) q[20];
rx(-pi/2) q[63];
rz(-pi/2) q[63];
cz q[63],q[56];
rx(2.7980648) q[56];
cz q[63],q[56];
cz q[19],q[56];
rx(2.8029838) q[56];
cz q[19],q[56];
rz(-pi/2) q[56];
rx(pi/2) q[63];
rz(-pi/2) q[63];
cz q[20],q[63];
rx(pi) q[20];
rx(0.3421538) q[63];
rz(pi/2) q[63];
cz q[20],q[63];
rz(-1.8612515) q[20];
rx(pi/2) q[20];
rz(-pi/2) q[20];
cz q[19],q[20];
rx(2.8029408) q[20];
cz q[19],q[20];
rx(0.68924457) q[19];
rz(1.5567251) q[19];
rx(1.5592016) q[19];
rz(-pi/2) q[20];
cz q[20],q[21];
rx(pi/2) q[20];
rz(-pi/2) q[21];
rx(pi/2) q[21];
cz q[20],q[21];
rx(-pi/2) q[20];
rx(pi/2) q[21];
cz q[20],q[21];
rx(-pi/2) q[21];
rz(-pi/2) q[21];
rx(0.43842737) q[63];
rz(1.0205543) q[63];
rx(0.83982636) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rz(-pi/2) q[63];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-pi/2) q[56];
rz(3*pi/2) q[56];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-pi/2) q[63];
rz(-pi/2) q[63];
cz q[20],q[63];
rx(2.8030138) q[63];
cz q[20],q[63];
cz q[20],q[21];
rx(2.8028128) q[21];
cz q[20],q[21];
rx(-2.452429) q[20];
rz(1.5720192) q[20];
rx(-3.1405851) q[20];
rx(-pi/2) q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rx(pi/2) q[21];
cz q[10],q[21];
rx(-pi/2) q[10];
rx(pi/2) q[21];
cz q[10],q[21];
rx(-pi/2) q[21];
rz(-4.6963477) q[21];
rx(-pi/2) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rx(pi/2) q[63];
cz q[20],q[63];
rx(-pi/2) q[20];
rx(pi/2) q[63];
cz q[20],q[63];
cz q[21],q[20];
rx(2.8033168) q[20];
cz q[21],q[20];
rz(pi/2) q[20];
rx(pi/2) q[20];
cz q[21],q[10];
rx(2.8024348) q[10];
cz q[21],q[10];
rx(-2.5933411) q[10];
rz(pi/2) q[10];
rx(pi/2) q[10];
rx(-2.2599593) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rz(-pi/2) q[21];
rx(pi/2) q[21];
cz q[20],q[21];
rx(-pi/2) q[20];
rx(pi/2) q[21];
cz q[20],q[21];
rz(-pi/2) q[21];
rx(pi/2) q[21];
cz q[10],q[21];
rz(pi/2) q[10];
rx(pi) q[10];
rx(0.3385038) q[21];
rz(pi/2) q[21];
cz q[10],q[21];
rx(0.78537538) q[10];
rz(2.0234109) q[10];
rx(1.9830376) q[10];
rx(pi/2) q[21];
rz(3.1146025) q[21];
rx(-2.2599593) q[21];
rz(-pi/2) q[63];
rx(-pi/2) q[63];
rx(-pi/2) q[70];
cz q[70],q[57];
rx(1.2338989) q[57];
cz q[70],q[57];
rx(-pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi/2) q[56];
rz(pi/2) q[57];
cz q[70],q[57];
rx(1.2337929) q[57];
cz q[70],q[57];
cz q[56],q[57];
rx(1.2298239) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rx(pi/2) q[56];
cz q[19],q[56];
rx(-pi/2) q[19];
rz(1.7330552) q[19];
rx(pi/2) q[56];
cz q[19],q[56];
rx(-pi/2) q[56];
rz(-pi/2) q[56];
cz q[70],q[57];
rz(-pi/2) q[57];
rx(-pi/2) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[57],q[56];
rx(1.2336029) q[56];
cz q[57],q[56];
cz q[19],q[56];
rx(1.2336449) q[56];
cz q[19],q[56];
rz(-pi/2) q[56];
rx(-pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(-0.14388994) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
cz q[56],q[63];
rx(pi/2) q[56];
rx(-pi/2) q[57];
rz(-pi/2) q[57];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-0.33698741) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
cz q[56],q[63];
cz q[19],q[56];
rx(1.2336349) q[56];
cz q[19],q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
rz(2.9977027) q[63];
rx(pi) q[63];
cz q[63],q[20];
rx(1.2337269) q[20];
cz q[63],q[20];
cz q[19],q[20];
rx(1.2336269) q[20];
cz q[19],q[20];
rx(-pi/2) q[19];
rz(-pi/2) q[20];
rx(-pi/2) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rx(pi/2) q[63];
cz q[20],q[63];
rx(-pi/2) q[20];
rz(-1.5638387) q[20];
rx(pi/2) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
cz q[20],q[21];
rx(pi/2) q[20];
rz(-pi/2) q[21];
rx(pi/2) q[21];
cz q[20],q[21];
rx(-0.33707641) q[20];
rx(pi/2) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rz(-pi) q[20];
cz q[19],q[20];
rx(pi/2) q[19];
rx(pi/2) q[20];
cz q[19],q[20];
rx(-0.33717641) q[19];
rz(pi/2) q[19];
rx(pi/2) q[20];
cz q[19],q[20];
rz(-2.5138256) q[20];
rx(pi/2) q[20];
rz(-2.0057222) q[21];
rx(pi/2) q[21];
cz q[10],q[21];
rx(pi/2) q[10];
rz(-pi/2) q[21];
rx(pi/2) q[21];
cz q[10],q[21];
rx(-0.33697841) q[10];
rz(-0.88745531) q[10];
rx(pi/2) q[21];
cz q[10],q[21];
rx(1.2940181) q[10];
rz(-1.2469943) q[10];
rz(-pi/2) q[21];
rx(-pi/2) q[21];
cz q[20],q[21];
rx(pi/2) q[20];
rz(-pi/2) q[21];
rx(pi/2) q[21];
cz q[20],q[21];
rx(-0.33641241) q[20];
rz(pi/2) q[20];
rx(pi/2) q[21];
cz q[20],q[21];
rx(-pi/2) q[20];
rz(-2.0344893) q[21];
rx(2.4127623) q[21];
rx(-pi/2) q[63];
cz q[70],q[57];
rx(1.2338709) q[57];
cz q[70],q[57];
rx(-pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi) q[56];
rz(pi/2) q[57];
cz q[70],q[57];
rx(1.2336019) q[57];
cz q[70],q[57];
cz q[56],q[57];
rx(1.2337799) q[57];
cz q[56],q[57];
cz q[56],q[63];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[56];
rx(pi/2) q[56];
rz(pi) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
cz q[56],q[57];
rx(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
cz q[19],q[56];
rx(pi/2) q[19];
rz(-pi/2) q[56];
rx(pi/2) q[56];
cz q[19],q[56];
rx(-pi/2) q[19];
rz(-pi/2) q[19];
rx(pi/2) q[56];
cz q[19],q[56];
rx(-pi/2) q[57];
rz(-pi/2) q[57];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[70],q[57];
rx(1.2342099) q[57];
cz q[70],q[57];
rx(-pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rz(pi/2) q[57];
cz q[63],q[56];
rx(1.2337099) q[56];
cz q[63],q[56];
cz q[19],q[56];
rx(1.2336329) q[56];
cz q[19],q[56];
rz(-pi/2) q[56];
cz q[70],q[57];
rx(1.2355449) q[57];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(-pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[57];
cz q[56],q[57];
rx(pi/2) q[56];
rz(pi/2) q[56];
rz(pi) q[57];
cz q[63],q[56];
rx(1.2336919) q[56];
cz q[63],q[56];
cz q[19],q[56];
rx(1.2336809) q[56];
cz q[19],q[56];
rx(pi) q[19];
cz q[57],q[56];
rx(1.2335669) q[56];
cz q[57],q[56];
rz(pi/2) q[56];
rx(pi/2) q[56];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi/2) q[63];
cz q[20],q[63];
rx(-pi/2) q[63];
cz q[56],q[63];
rx(pi/2) q[56];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-pi/2) q[56];
rz(pi/2) q[56];
rx(pi/2) q[63];
cz q[56],q[63];
rx(-pi/2) q[56];
rz(-0.087641995) q[56];
rz(pi) q[63];
cz q[20],q[63];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[63],q[20];
rx(pi/2) q[20];
rz(pi) q[20];
rx(pi/2) q[63];
rz(pi) q[63];
cz q[20],q[63];
rx(pi/2) q[63];
rz(1.5776614) q[63];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi/2) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(-pi/2) q[57];
cz q[56],q[57];
rx(pi) q[56];
rx(1.2350269) q[57];
cz q[56],q[57];
rz(3.0539507) q[56];
rx(pi/2) q[56];
rz(pi/2) q[56];
rx(1.9343098) q[57];
rz(0.84196599) q[57];
cz q[63],q[56];
rx(1.2337079) q[56];
cz q[63],q[56];
rx(1.9080005) q[56];
rz(pi) q[56];
cz q[19],q[56];
rz(pi/2) q[19];
rx(pi) q[19];
rx(1.2337569) q[56];
cz q[19],q[56];
rz(-1.5701997) q[19];
rx(pi/2) q[19];
rz(2.4127623) q[19];
cz q[20],q[19];
rx(pi/2) q[19];
rz(pi) q[19];
rx(pi/2) q[20];
rz(pi) q[20];
cz q[19],q[20];
rx(pi/2) q[19];
rx(pi/2) q[20];
rz(pi/2) q[20];
cz q[20],q[19];
rx(-pi/2) q[19];
rx(-1.9080005) q[56];
rx(-0.72883034) q[63];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi) q[70];
cz q[57],q[70];
rx(pi/2) q[57];
rz(pi) q[57];
rx(pi/2) q[70];
rz(pi/2) q[70];
cz q[70],q[57];
rx(pi/2) q[57];
rz(1.5647564) q[57];
cz q[57],q[56];
rx(1.2338989) q[56];
cz q[57],q[56];
rx(0.95414677) q[56];
cz q[19],q[56];
rz(pi/2) q[19];
rx(pi) q[19];
rx(1.2336529) q[56];
rz(4.2241989) q[56];
cz q[19],q[56];
rz(3.1314302) q[19];
rx(2.4127623) q[19];
rx(1.0058543) q[56];
rz(0.78136479) q[56];
rx(-0.72883034) q[57];
barrier q[30],q[39],q[48],q[21],q[54],q[70],q[8],q[72],q[17],q[14],q[26],q[23],q[32],q[41],q[50],q[47],q[10],q[1],q[65],q[56],q[74],q[63],q[16],q[25],q[34],q[43],q[40],q[52],q[49],q[58],q[3],q[67],q[12],q[9],q[76],q[73],q[18],q[27],q[36],q[45],q[42],q[51],q[60],q[5],q[2],q[69],q[66],q[11],q[78],q[75],q[57],q[29],q[38],q[35],q[44],q[53],q[62],q[7],q[59],q[4],q[71],q[68],q[13],q[77],q[22],q[31],q[28],q[37],q[46],q[55],q[0],q[64],q[61],q[6],q[20],q[15],q[79],q[24],q[19],q[33];
measure q[56] -> meas[0];
measure q[19] -> meas[1];
measure q[57] -> meas[2];
measure q[20] -> meas[3];
measure q[63] -> meas[4];
measure q[70] -> meas[5];
measure q[21] -> meas[6];
measure q[10] -> meas[7];
