OPENQASM 2.0;
include "qelib1.inc";

qreg q[8];
creg meas[8];
rz(3.5*pi) q[0];
rz(1.7645499850955613*pi) q[1];
rz(1.039446388575316*pi) q[2];
rz(0.9970144902768983*pi) q[3];
rz(2.9945743703932615*pi) q[4];
rz(3.0066389026756957*pi) q[5];
rz(1.029505470211228*pi) q[6];
rz(2.8434075503698164*pi) q[7];
rx(2.410295355797988*pi) q[0];
rx(3.546371464602042*pi) q[1];
rx(3.509934153663826*pi) q[2];
rx(3.5405627202142886*pi) q[3];
rx(3.540310288026369*pi) q[4];
rx(3.540129265746387*pi) q[5];
rx(3.528033546601045*pi) q[6];
rx(3.480190378221006*pi) q[7];
rz(3.2608781326060337*pi) q[0];
rz(3.4497832880134185*pi) q[1];
rz(0.5781313583081309*pi) q[2];
rz(1.023450042625512*pi) q[3];
rz(1.0427042684808243*pi) q[4];
rz(0.9476686941502064*pi) q[5];
rz(0.7411973865479682*pi) q[6];
rz(0.5367770979157453*pi) q[7];
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
rz(2.198410329891062*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rxx(0.5*pi) q[3],q[4];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[6];
rxx(0.5*pi) q[2],q[5];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
rxx(0.5*pi) q[0],q[7];
ry(3.5*pi) q[1];
ry(3.5*pi) q[2];
rz(3.5*pi) q[3];
rz(3.5*pi) q[4];
rx(3.5*pi) q[5];
rx(3.5*pi) q[6];
ry(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(1.1984500191451737*pi) q[3];
rx(0.5*pi) q[4];
rx(3.5*pi) q[7];
rz(3.5*pi) q[0];
rz(1.779793715642759*pi) q[1];
ry(0.5*pi) q[2];
ry(0.5*pi) q[3];
rz(2.5217562652120677*pi) q[4];
rx(0.5813833857516973*pi) q[7];
rz(0.34226151835773044*pi) q[0];
ry(0.5*pi) q[1];
rxx(0.5*pi) q[2],q[6];
rxx(0.5*pi) q[3],q[5];
ry(0.5*pi) q[4];
rx(3.2290363130043844*pi) q[0];
rxx(0.5*pi) q[1],q[7];
ry(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[5];
rx(3.5*pi) q[6];
rz(0.26087813260603376*pi) q[0];
ry(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rx(0.8015499808548282*pi) q[5];
rx(3.5*pi) q[7];
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
rz(2.0406718482487927*pi) q[2];
ry(0.5*pi) q[3];
rxx(0.5*pi) q[4],q[5];
rx(2.2608781326060337*pi) q[7];
rz(0.29426757943111836*pi) q[1];
ry(0.5*pi) q[2];
rxx(0.5*pi) q[3],q[6];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
rx(3.262767976377144*pi) q[1];
rxx(0.5*pi) q[2],q[7];
ry(3.5*pi) q[3];
rz(3.5*pi) q[4];
rz(3.5*pi) q[5];
rx(3.5*pi) q[6];
rz(2.0490237812266505*pi) q[1];
ry(3.5*pi) q[2];
rz(3.5*pi) q[3];
rz(0.19845001914517357*pi) q[4];
rx(0.5*pi) q[5];
rx(3.5*pi) q[7];
rxx(0.5*pi) q[0],q[1];
rz(3.5*pi) q[2];
rz(2.3838990977298335*pi) q[3];
ry(0.5*pi) q[4];
rz(2.5217562652120677*pi) q[5];
rz(1.0*pi) q[7];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
rz(1.034976782079104*pi) q[2];
ry(0.5*pi) q[3];
rxx(0.5*pi) q[4],q[6];
ry(0.5*pi) q[5];
rx(1.6567727505189596*pi) q[7];
rz(3.5*pi) q[0];
rz(3.5*pi) q[1];
rx(3.520798240872455*pi) q[2];
ry(3.5*pi) q[4];
rx(3.5*pi) q[6];
rz(1.0*pi) q[7];
rz(0.19845001914517357*pi) q[0];
rx(0.5*pi) q[1];
rz(0.6700978719155191*pi) q[2];
rxx(0.5*pi) q[3],q[7];
rz(3.5*pi) q[4];
rx(0.8015499808548282*pi) q[6];
ry(0.5*pi) q[0];
rz(2.5217562652120677*pi) q[1];
ry(3.5*pi) q[3];
rz(0.040671848248792664*pi) q[4];
rxx(0.5*pi) q[5],q[6];
rx(3.5*pi) q[7];
rxx(0.5*pi) q[0],q[2];
ry(0.5*pi) q[1];
rz(3.5*pi) q[3];
rx(3.0*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
rz(3.0*pi) q[7];
ry(3.5*pi) q[0];
rx(3.5*pi) q[2];
rz(2.6460281504465413*pi) q[3];
rz(0.5*pi) q[4];
rz(3.5*pi) q[5];
rz(3.5*pi) q[6];
rx(2.84322724948104*pi) q[7];
rz(3.5*pi) q[0];
rx(0.8015499808548282*pi) q[2];
rx(3.460765600646849*pi) q[3];
ry(0.5*pi) q[4];
rz(0.1430602344462616*pi) q[5];
rx(3.5*pi) q[6];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[2];
rz(1.9146783059716996*pi) q[3];
rxx(0.5*pi) q[4],q[7];
ry(0.5*pi) q[5];
rz(3.3468262264990747*pi) q[6];
rxx(0.5*pi) q[0],q[3];
ry(3.5*pi) q[1];
rx(3.5*pi) q[2];
ry(3.5*pi) q[4];
ry(0.5*pi) q[6];
rx(3.5*pi) q[7];
ry(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rx(3.5*pi) q[3];
rz(3.5*pi) q[4];
rz(3.0*pi) q[7];
rz(3.5*pi) q[0];
rz(1.1984500191451737*pi) q[1];
rx(0.5*pi) q[2];
rz(0.47857683124139205*pi) q[4];
rx(3.403938367052296*pi) q[7];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
rz(2.5217562652120677*pi) q[2];
rx(3.4654014960380213*pi) q[4];
rxx(0.5*pi) q[5],q[7];
rxx(0.5*pi) q[1],q[3];
ry(0.5*pi) q[2];
rz(1.8230284984309195*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[7];
rxx(0.5*pi) q[0],q[4];
ry(3.5*pi) q[1];
rx(3.5*pi) q[3];
rz(3.5*pi) q[5];
rx(3.9883572738425954*pi) q[7];
ry(3.5*pi) q[0];
rz(3.5*pi) q[1];
rx(0.8015499808548282*pi) q[3];
rx(3.5*pi) q[4];
rz(0.09830201460504262*pi) q[5];
rz(3.0*pi) q[7];
rz(3.5*pi) q[0];
ry(0.5*pi) q[1];
rxx(0.5*pi) q[2],q[3];
rx(3.4593895679310123*pi) q[5];
rxx(0.5*pi) q[6],q[7];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[4];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
rz(0.017590496320066706*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
rxx(0.5*pi) q[0],q[5];
ry(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rx(3.5*pi) q[4];
rz(3.5*pi) q[6];
rz(2.7270836763955018*pi) q[7];
ry(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(0.19845001914517357*pi) q[2];
rx(0.5*pi) q[3];
rx(3.5*pi) q[5];
rz(0.8934689665879034*pi) q[6];
rx(3.2464604644118906*pi) q[7];
rz(3.5*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rz(2.5217562652120677*pi) q[3];
rx(3.5406550124935205*pi) q[6];
rz(1.1470117245673044*pi) q[7];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[5];
rxx(0.5*pi) q[2],q[4];
ry(0.5*pi) q[3];
rz(0.9907909086033495*pi) q[6];
rxx(0.5*pi) q[0],q[6];
ry(3.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[4];
rx(3.5*pi) q[5];
ry(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rx(0.8015499808548282*pi) q[4];
rx(3.5*pi) q[6];
rz(3.5*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rxx(0.5*pi) q[3],q[4];
rz(1.7311277480561622*pi) q[0];
rxx(0.5*pi) q[1],q[6];
rxx(0.5*pi) q[2],q[5];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
rx(1.0*pi) q[0];
ry(3.5*pi) q[1];
ry(3.5*pi) q[2];
rz(3.5*pi) q[3];
rz(3.5*pi) q[4];
rx(3.5*pi) q[5];
rx(3.5*pi) q[6];
rz(0.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(1.1984500191451737*pi) q[3];
rx(0.5*pi) q[4];
ry(0.5*pi) q[0];
rz(2.0406718482487927*pi) q[1];
ry(0.5*pi) q[2];
ry(0.5*pi) q[3];
rz(2.5217562652120677*pi) q[4];
rxx(0.5*pi) q[0],q[7];
ry(0.5*pi) q[1];
rxx(0.5*pi) q[2],q[6];
rxx(0.5*pi) q[3],q[5];
ry(0.5*pi) q[4];
ry(3.5*pi) q[0];
ry(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[5];
rx(3.5*pi) q[6];
rx(3.5*pi) q[7];
rz(3.5*pi) q[0];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rx(0.8015499808548282*pi) q[5];
rz(3.0*pi) q[7];
rz(3.6904558998073695*pi) q[0];
rz(0.2797937156427587*pi) q[2];
ry(0.5*pi) q[3];
rxx(0.5*pi) q[4],q[5];
rx(2.80954410019263*pi) q[7];
rx(2.4102423352507314*pi) q[0];
rxx(0.5*pi) q[1],q[7];
rx(1.0*pi) q[2];
rxx(0.5*pi) q[3],q[6];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
rz(0.26087813260603376*pi) q[0];
ry(3.5*pi) q[1];
rz(0.5*pi) q[2];
ry(3.5*pi) q[3];
rz(3.5*pi) q[4];
rz(3.5*pi) q[5];
rx(3.5*pi) q[6];
rx(3.5*pi) q[7];
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
ry(0.5*pi) q[2];
rz(3.5*pi) q[3];
rz(0.19845001914517357*pi) q[4];
rx(3.5*pi) q[5];
rx(3.7391218673939663*pi) q[7];
rz(3.9571290527760414*pi) q[1];
rz(0.944610215301088*pi) q[3];
ry(0.5*pi) q[4];
rz(0.9782437347879326*pi) q[5];
rz(3.0*pi) q[7];
rx(3.735990400754684*pi) q[1];
rxx(0.5*pi) q[2],q[7];
ry(0.5*pi) q[3];
rxx(0.5*pi) q[4],q[6];
ry(0.5*pi) q[5];
rz(3.063032920217247*pi) q[1];
ry(3.5*pi) q[2];
ry(3.5*pi) q[4];
rx(3.5*pi) q[6];
rx(3.5*pi) q[7];
rxx(0.5*pi) q[0],q[1];
rz(3.5*pi) q[2];
rz(3.5*pi) q[4];
rx(1.8015499808548283*pi) q[6];
rz(3.0*pi) q[7];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
rz(0.7672860423432091*pi) q[2];
rz(3.5406718482487927*pi) q[4];
rxx(0.5*pi) q[5],q[6];
rx(1.1648164996583295*pi) q[7];
rz(3.5*pi) q[0];
rz(3.5*pi) q[1];
rx(3.5293808378353617*pi) q[2];
rxx(0.5*pi) q[3],q[7];
rx(3.0*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
rz(0.19845001914517357*pi) q[0];
rx(0.5*pi) q[1];
rz(0.7560873336240447*pi) q[2];
ry(3.5*pi) q[3];
rz(0.5*pi) q[4];
rz(3.5*pi) q[5];
rz(3.5*pi) q[6];
rx(3.5*pi) q[7];
ry(0.5*pi) q[0];
rz(2.5217562652120677*pi) q[1];
rz(3.5*pi) q[3];
ry(0.5*pi) q[4];
ry(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(3.0*pi) q[7];
rxx(0.5*pi) q[0],q[2];
ry(0.5*pi) q[1];
rz(2.0835323784212223*pi) q[3];
rz(2.879159061120282*pi) q[6];
rx(0.9039383670522958*pi) q[7];
ry(3.5*pi) q[0];
rx(3.5*pi) q[2];
rx(3.538703895194052*pi) q[3];
rxx(0.5*pi) q[4],q[7];
ry(0.5*pi) q[6];
rz(3.5*pi) q[0];
rx(0.8015499808548282*pi) q[2];
rz(1.0999331498203166*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[7];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[2];
rz(3.5*pi) q[4];
rz(3.0*pi) q[7];
rxx(0.5*pi) q[0],q[3];
ry(3.5*pi) q[1];
rx(3.5*pi) q[2];
rz(1.9667150133828404*pi) q[4];
rx(1.2608781326060339*pi) q[7];
ry(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rx(3.5*pi) q[3];
rx(3.523416010974949*pi) q[4];
rz(3.0*pi) q[7];
rz(3.5*pi) q[0];
rz(1.1984500191451737*pi) q[1];
rx(0.5*pi) q[2];
rz(1.3055352388607924*pi) q[4];
rxx(0.5*pi) q[5],q[7];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
rz(2.5217562652120677*pi) q[2];
ry(3.5*pi) q[5];
rx(3.5*pi) q[7];
rxx(0.5*pi) q[0],q[4];
rxx(0.5*pi) q[1],q[3];
ry(0.5*pi) q[2];
rz(3.5*pi) q[5];
rx(0.1425972040917855*pi) q[7];
ry(3.5*pi) q[0];
ry(3.5*pi) q[1];
rx(3.5*pi) q[3];
rx(3.5*pi) q[4];
rz(0.269362293610069*pi) q[5];
rz(1.0*pi) q[7];
rz(3.5*pi) q[0];
rz(3.5*pi) q[1];
rx(0.8015499808548282*pi) q[3];
rx(3.460218177124726*pi) q[5];
rxx(0.5*pi) q[6],q[7];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
rxx(0.5*pi) q[2],q[3];
rz(0.06706782455516624*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
rxx(0.5*pi) q[0],q[5];
rxx(0.5*pi) q[1],q[4];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
rz(3.5*pi) q[6];
rz(1.0527698432957366*pi) q[7];
ry(3.5*pi) q[0];
ry(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rx(3.5*pi) q[4];
rx(3.5*pi) q[5];
rz(0.8940983805282854*pi) q[6];
rx(3.042901107937682*pi) q[7];
rz(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(0.19845001914517357*pi) q[2];
rx(0.5*pi) q[3];
rx(3.4612480983690297*pi) q[6];
rz(0.08123350362405413*pi) q[7];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rz(2.5217562652120677*pi) q[3];
rz(0.09869768310524796*pi) q[6];
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
rz(1.9643897452869874*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rxx(0.5*pi) q[3],q[4];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[6];
rxx(0.5*pi) q[2],q[5];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
rxx(0.5*pi) q[0],q[7];
ry(3.5*pi) q[1];
ry(3.5*pi) q[2];
rz(3.5*pi) q[3];
rz(3.5*pi) q[4];
rx(3.5*pi) q[5];
rx(3.5*pi) q[6];
ry(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(1.1984500191451737*pi) q[3];
rx(0.5*pi) q[4];
rx(3.5*pi) q[7];
rz(3.5*pi) q[0];
rz(0.9219330154951004*pi) q[1];
ry(0.5*pi) q[2];
ry(0.5*pi) q[3];
rz(2.5217562652120677*pi) q[4];
rz(3.0*pi) q[7];
rz(3.5762821029618053*pi) q[0];
rx(1.0*pi) q[1];
rxx(0.5*pi) q[2],q[6];
rxx(0.5*pi) q[3],q[5];
ry(0.5*pi) q[4];
rx(2.5424567297918865*pi) q[7];
rx(0.7879759571658281*pi) q[0];
rz(0.5*pi) q[1];
ry(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[5];
rx(3.5*pi) q[6];
rz(0.5*pi) q[0];
ry(0.5*pi) q[1];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rx(0.8015499808548282*pi) q[5];
rxx(0.5*pi) q[1],q[7];
rz(2.197864960072292*pi) q[2];
ry(0.5*pi) q[3];
rxx(0.5*pi) q[4],q[5];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
rxx(0.5*pi) q[3],q[6];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
rx(3.5*pi) q[7];
rz(3.5*pi) q[1];
ry(3.5*pi) q[3];
rz(3.5*pi) q[4];
rz(3.5*pi) q[5];
rx(3.5*pi) q[6];
rz(3.0*pi) q[7];
rz(3.8812611672463078*pi) q[1];
rz(3.5*pi) q[3];
rz(0.19845001914517357*pi) q[4];
rx(0.5*pi) q[5];
rx(1.775931944577192*pi) q[7];
rx(0.2927282137846835*pi) q[1];
rxx(0.5*pi) q[2],q[7];
rz(3.054234443676747*pi) q[3];
ry(0.5*pi) q[4];
rz(2.5217562652120677*pi) q[5];
rz(0.5*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
rxx(0.5*pi) q[4],q[6];
ry(0.5*pi) q[5];
rx(3.5*pi) q[7];
rz(3.5*pi) q[2];
ry(3.5*pi) q[4];
rx(3.5*pi) q[6];
rz(1.0*pi) q[7];
rz(3.3428068881765007*pi) q[2];
rz(3.5*pi) q[4];
rx(0.8015499808548282*pi) q[6];
rx(1.1436305163955356*pi) q[7];
rx(0.4859577944282824*pi) q[2];
rz(3.645093306179828*pi) q[4];
rxx(0.5*pi) q[5],q[6];
rz(1.0*pi) q[7];
rz(0.5*pi) q[2];
rxx(0.5*pi) q[3],q[7];
rx(1.0*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
ry(3.5*pi) q[3];
rz(0.5*pi) q[4];
rz(3.5*pi) q[5];
rz(3.5*pi) q[6];
rx(3.5*pi) q[7];
rz(3.5*pi) q[3];
ry(0.5*pi) q[4];
rz(3.717882280169364*pi) q[5];
rx(0.5*pi) q[6];
rz(3.0*pi) q[7];
rz(0.4864374045720472*pi) q[3];
rx(1.0*pi) q[5];
rz(3.1847830324313238*pi) q[6];
rx(1.9091411374969243*pi) q[7];
rx(0.02285065212220831*pi) q[3];
rxx(0.5*pi) q[4],q[7];
rz(0.5*pi) q[5];
ry(0.5*pi) q[6];
rz(0.5*pi) q[3];
ry(3.5*pi) q[4];
ry(0.5*pi) q[5];
rx(3.5*pi) q[7];
rz(3.5*pi) q[4];
rx(0.12566104515563475*pi) q[7];
rz(0.6044214579310347*pi) q[4];
rxx(0.5*pi) q[5],q[7];
rx(3.9012489093057834*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[7];
rz(0.5*pi) q[4];
rz(3.5*pi) q[5];
rx(1.5548555129501034*pi) q[7];
rz(3.4787604127753977*pi) q[5];
rz(1.0*pi) q[7];
rx(3.810701848842511*pi) q[5];
rxx(0.5*pi) q[6],q[7];
rz(0.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
rz(3.5*pi) q[6];
rz(0.4288639474860978*pi) q[7];
rz(3.07609510017471*pi) q[6];
rx(0.26510264821789087*pi) q[7];
rx(0.648287328000729*pi) q[6];
rz(3.6037001219647884*pi) q[7];
rz(0.5*pi) q[6];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
