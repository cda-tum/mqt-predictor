OPENQASM 2.0;
include "qelib1.inc";

qreg q[9];
creg meas[9];
rz(3.5*pi) q[0];
rz(2.9037430936383473*pi) q[1];
rz(3.5*pi) q[2];
rz(3.0*pi) q[3];
rz(3.5*pi) q[4];
rz(3.5*pi) q[5];
rz(3.0*pi) q[6];
rz(0.5*pi) q[7];
rz(2.8459329402424816*pi) q[8];
rx(1.174085055060313*pi) q[0];
rx(3.3950916995435447*pi) q[1];
rx(3.3271338067589937*pi) q[2];
rx(2.5*pi) q[3];
rx(2.809875419452454*pi) q[4];
rx(1.344047520966737*pi) q[5];
rx(2.5*pi) q[6];
rx(2.560043880831723*pi) q[7];
rx(3.2945687082835837*pi) q[8];
rz(3.6411778800231955*pi) q[0];
rz(2.7558470944709077*pi) q[1];
rz(0.5*pi) q[2];
rz(0.19135880305312847*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.08140095218577575*pi) q[6];
rz(0.5*pi) q[7];
rz(1.7713660229382358*pi) q[8];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[0],q[1];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
rz(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.3588221199768045*pi) q[0];
rx(0.5*pi) q[1];
rx(2.961670624690602*pi) q[0];
rz(2.6411778800231955*pi) q[1];
rz(0.5*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[2];
ry(3.5*pi) q[1];
rx(3.5*pi) q[2];
rz(3.5*pi) q[1];
rz(0.5*pi) q[2];
rz(3.5*pi) q[1];
rx(0.5*pi) q[2];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(3.2700122703292838*pi) q[1];
ry(0.5*pi) q[2];
rxx(0.5*pi) q[0],q[1];
rxx(0.5*pi) q[2],q[3];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
rz(3.5*pi) q[0];
rz(0.5*pi) q[1];
rz(3.5*pi) q[2];
rz(0.5*pi) q[3];
rz(3.5*pi) q[0];
rx(0.5*pi) q[1];
rz(3.5*pi) q[2];
rx(0.5*pi) q[3];
rx(2.444452699363756*pi) q[0];
rz(0.5*pi) q[1];
rx(2.2353239652498287*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[0];
ry(0.5*pi) q[1];
rz(0.5*pi) q[2];
ry(0.5*pi) q[3];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[2];
rxx(0.5*pi) q[3],q[4];
ry(3.5*pi) q[1];
rx(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rz(3.5*pi) q[4];
rz(3.5*pi) q[1];
rx(3.5*pi) q[2];
rx(2.2040080102438946*pi) q[3];
rx(3.5*pi) q[4];
rx(2.5*pi) q[1];
rz(1.0*pi) q[2];
rz(0.5*pi) q[3];
rz(1.0*pi) q[4];
rz(2.1148084226216604*pi) q[1];
ry(0.5*pi) q[2];
ry(0.5*pi) q[4];
rxx(0.5*pi) q[0],q[1];
rxx(0.5*pi) q[2],q[3];
rxx(0.5*pi) q[4],q[5];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
rz(3.5*pi) q[0];
rz(0.5*pi) q[1];
rz(3.5*pi) q[2];
rz(0.5*pi) q[3];
rz(3.5*pi) q[4];
rz(0.5*pi) q[5];
rz(3.5*pi) q[0];
rx(0.5*pi) q[1];
rx(0.3128370734869077*pi) q[2];
rx(0.5*pi) q[3];
rz(3.5*pi) q[4];
rx(0.5*pi) q[5];
rx(3.483213082411835*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(2.5*pi) q[4];
rz(0.5*pi) q[5];
rz(2.2285664625360333*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[3];
rz(0.0287764987174981*pi) q[4];
ry(0.5*pi) q[5];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[2];
rxx(0.5*pi) q[3],q[4];
rxx(0.5*pi) q[5],q[6];
ry(3.5*pi) q[1];
rx(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rz(0.5*pi) q[4];
rz(3.5*pi) q[5];
rz(0.5*pi) q[6];
rz(3.761237587360254*pi) q[1];
rx(3.5*pi) q[2];
rz(2.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[5];
rx(0.5*pi) q[6];
rx(3.584323633387496*pi) q[1];
rz(1.0*pi) q[2];
rx(2.648119133455985*pi) q[3];
rz(0.5*pi) q[4];
rx(2.3611200740413856*pi) q[5];
rz(0.5*pi) q[6];
rz(2.5761668897708843*pi) q[1];
ry(0.5*pi) q[2];
rz(0.5*pi) q[3];
ry(0.5*pi) q[4];
rz(0.5*pi) q[5];
ry(0.5*pi) q[6];
rxx(0.5*pi) q[0],q[1];
rxx(0.5*pi) q[2],q[3];
rxx(0.5*pi) q[4],q[5];
rxx(0.5*pi) q[6],q[7];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
rz(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(0.5*pi) q[3];
rz(3.5*pi) q[4];
rz(3.5*pi) q[5];
rz(3.5*pi) q[6];
rz(3.5*pi) q[7];
rz(3.7714335374639663*pi) q[0];
rx(3.5*pi) q[1];
rz(3.5*pi) q[2];
rx(0.5*pi) q[3];
rz(3.5*pi) q[4];
rx(3.5*pi) q[5];
rx(1.7832068946498345*pi) q[6];
rx(0.5*pi) q[7];
rx(1.8651899870948943*pi) q[0];
rz(2.2714335374639667*pi) q[1];
rx(2.5*pi) q[2];
rz(0.5*pi) q[3];
rx(3.5*pi) q[4];
rz(1.0*pi) q[5];
rz(0.5*pi) q[6];
rz(3.25*pi) q[7];
rz(0.5*pi) q[0];
ry(0.5*pi) q[1];
rz(1.7096106966477826*pi) q[2];
ry(0.5*pi) q[3];
rz(0.21035052488187028*pi) q[4];
ry(0.5*pi) q[5];
ry(0.5*pi) q[7];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[2];
rxx(0.5*pi) q[3],q[4];
rxx(0.5*pi) q[5],q[6];
rxx(0.5*pi) q[7],q[8];
ry(3.5*pi) q[1];
rx(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
ry(3.5*pi) q[7];
rx(3.5*pi) q[8];
rz(3.5*pi) q[1];
rz(0.5*pi) q[2];
rz(3.5*pi) q[3];
rz(0.5*pi) q[4];
rz(3.5*pi) q[5];
rz(3.5*pi) q[6];
rz(3.5*pi) q[7];
rz(1.0023838161852792*pi) q[8];
rx(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(3.5*pi) q[3];
rx(0.5*pi) q[4];
rx(2.984336092182184*pi) q[5];
rx(3.5*pi) q[6];
rz(1.75*pi) q[7];
rx(3.752573198672973*pi) q[8];
rz(3.504577003779934*pi) q[1];
rz(0.5*pi) q[2];
rx(2.16950466857606*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(1.0*pi) q[6];
rx(1.4897895567073338*pi) q[7];
rz(0.8151476476910734*pi) q[8];
rxx(0.5*pi) q[0],q[1];
ry(0.5*pi) q[2];
rz(0.5*pi) q[3];
ry(0.5*pi) q[4];
ry(0.5*pi) q[6];
rz(0.5*pi) q[7];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[3];
rxx(0.5*pi) q[4],q[5];
rxx(0.5*pi) q[6],q[7];
rz(3.5*pi) q[0];
rz(3.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
rz(3.5*pi) q[0];
rx(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(0.5*pi) q[3];
rz(3.5*pi) q[4];
rz(0.5*pi) q[5];
rz(3.5*pi) q[6];
rz(3.5*pi) q[7];
rx(3.9035544200095154*pi) q[0];
rz(1.0*pi) q[1];
rz(3.5*pi) q[2];
rx(0.5*pi) q[3];
rz(3.5*pi) q[4];
rx(0.5*pi) q[5];
rz(3.5*pi) q[6];
rx(0.5*pi) q[7];
rz(0.5*pi) q[0];
ry(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(2.5*pi) q[4];
rz(0.5*pi) q[5];
rx(2.5*pi) q[6];
rz(0.4969276061680275*pi) q[7];
rz(3.756525793706074*pi) q[2];
ry(0.5*pi) q[3];
rz(0.5536370976160017*pi) q[4];
ry(0.5*pi) q[5];
rz(2.178349786782868*pi) q[6];
ry(0.5*pi) q[7];
rxx(0.5*pi) q[1],q[2];
rxx(0.5*pi) q[3],q[4];
rxx(0.5*pi) q[5],q[6];
rxx(0.5*pi) q[7],q[8];
ry(3.5*pi) q[1];
rx(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
ry(3.5*pi) q[7];
rx(3.5*pi) q[8];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rz(0.5*pi) q[4];
rz(3.5*pi) q[5];
rz(3.5*pi) q[6];
rz(3.5*pi) q[7];
rz(3.1268502144690835*pi) q[8];
rz(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(2.768159345132159*pi) q[3];
rx(0.5*pi) q[4];
rz(2.5*pi) q[5];
rx(3.5*pi) q[6];
rz(0.003072393831972442*pi) q[7];
rx(3.4335150816196895*pi) q[8];
rx(3.9477714057908244*pi) q[1];
rz(2.75*pi) q[2];
rx(3.3964147910695286*pi) q[3];
rz(0.5*pi) q[4];
rx(0.3679004464317338*pi) q[5];
rz(1.0*pi) q[6];
rx(1.5*pi) q[7];
rz(2.001216459136776*pi) q[8];
rz(0.5*pi) q[1];
ry(0.5*pi) q[2];
rz(1.6095536799040808*pi) q[3];
ry(0.5*pi) q[4];
rz(0.5*pi) q[5];
ry(0.5*pi) q[6];
rz(3.370145757505214*pi) q[7];
rxx(0.5*pi) q[2],q[3];
rxx(0.5*pi) q[4],q[5];
rxx(0.5*pi) q[6],q[7];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rz(3.5*pi) q[4];
rz(3.5*pi) q[5];
rz(3.5*pi) q[6];
rz(3.5*pi) q[7];
rz(0.75*pi) q[2];
rx(3.5*pi) q[3];
rz(3.0*pi) q[4];
rx(3.5*pi) q[5];
rz(3.5*pi) q[6];
rx(1.5*pi) q[7];
rx(3.911187903829234*pi) q[2];
rz(0.75*pi) q[3];
rx(1.9245943863166945*pi) q[4];
rz(1.0*pi) q[5];
rx(3.217189596747479*pi) q[6];
rz(0.43634718703950415*pi) q[7];
rz(0.5*pi) q[2];
ry(0.5*pi) q[3];
rz(0.5*pi) q[4];
ry(0.5*pi) q[5];
rz(0.5*pi) q[6];
ry(0.5*pi) q[7];
rxx(0.5*pi) q[3],q[4];
rxx(0.5*pi) q[5],q[6];
rxx(0.5*pi) q[7],q[8];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
ry(3.5*pi) q[7];
rx(3.5*pi) q[8];
rz(3.5*pi) q[3];
rz(0.5*pi) q[4];
rz(3.5*pi) q[5];
rz(3.5*pi) q[6];
rz(3.5*pi) q[7];
rz(0.5690516448524559*pi) q[8];
rx(3.7142794912776966*pi) q[3];
rx(0.5*pi) q[4];
rz(3.5*pi) q[5];
rx(3.5*pi) q[6];
rz(3.0636528129604956*pi) q[7];
rx(3.2832011152888043*pi) q[8];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(1.5*pi) q[5];
rz(1.0*pi) q[6];
rx(2.573728125006305*pi) q[7];
rz(3.4928505403093233*pi) q[8];
ry(0.5*pi) q[4];
rz(3.0204240352247016*pi) q[5];
ry(0.5*pi) q[6];
rz(0.5*pi) q[7];
rxx(0.5*pi) q[4],q[5];
rxx(0.5*pi) q[6],q[7];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
rz(3.5*pi) q[4];
rz(3.5*pi) q[5];
rz(3.5*pi) q[6];
rz(3.5*pi) q[7];
rz(3.5*pi) q[4];
rx(1.5*pi) q[5];
rz(3.627839955915175*pi) q[6];
rx(0.5*pi) q[7];
rx(0.06644993855091598*pi) q[4];
rz(1.1475836176504333*pi) q[5];
rx(3.42416049322651*pi) q[6];
rz(3.093307247577039*pi) q[7];
rz(0.5*pi) q[4];
ry(0.5*pi) q[5];
rz(3.3385502334153876*pi) q[6];
ry(0.5*pi) q[7];
rxx(0.5*pi) q[5],q[6];
rxx(0.5*pi) q[7],q[8];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
ry(3.5*pi) q[7];
rx(3.5*pi) q[8];
rz(3.5*pi) q[5];
rz(3.5*pi) q[6];
rz(3.5*pi) q[7];
rz(3.202916297690746*pi) q[8];
rz(1.3524163823495687*pi) q[5];
rx(3.5*pi) q[6];
rz(1.4066927524229604*pi) q[7];
rx(3.5947247482268656*pi) q[8];
rx(3.042423256105876*pi) q[5];
rz(0.852416382349567*pi) q[6];
rx(0.5*pi) q[7];
rz(2.471763011027005*pi) q[8];
rz(0.5*pi) q[5];
ry(0.5*pi) q[6];
rz(3.7703188202613207*pi) q[7];
rxx(0.5*pi) q[6],q[7];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
rz(3.5*pi) q[6];
rz(3.5*pi) q[7];
rz(0.5*pi) q[6];
rx(0.5*pi) q[7];
rx(0.16379430287508512*pi) q[6];
rz(3.8061546609547765*pi) q[7];
rz(0.5*pi) q[6];
ry(0.5*pi) q[7];
rxx(0.5*pi) q[7],q[8];
ry(3.5*pi) q[7];
rx(3.5*pi) q[8];
rz(3.5*pi) q[7];
rz(0.020603175461803125*pi) q[8];
rz(3.6938453390452235*pi) q[7];
rx(0.693532377174901*pi) q[8];
rx(3.851937928306287*pi) q[7];
rz(1.0117801709656322*pi) q[8];
rz(0.5*pi) q[7];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
