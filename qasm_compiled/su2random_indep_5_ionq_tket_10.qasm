OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];
creg meas[5];
rz(3.5*pi) q[0];
rz(0.7089395285333957*pi) q[1];
rz(0.9574937966120258*pi) q[2];
rz(2.864174522860192*pi) q[3];
rz(0.1828142411137954*pi) q[4];
rx(0.15899371595144357*pi) q[0];
rx(3.8445005839638213*pi) q[1];
rx(3.7761891799352427*pi) q[2];
rx(3.728506935581603*pi) q[3];
rx(3.3880149090030667*pi) q[4];
rz(1.1454771198541478*pi) q[0];
rz(0.6058740356833259*pi) q[1];
rz(0.2410730815620694*pi) q[2];
rz(0.3014873360142432*pi) q[3];
rz(1.9479622747227365*pi) q[4];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[0],q[1];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
rz(3.5*pi) q[0];
rx(2.147583617650433*pi) q[1];
rz(1.3975836176504333*pi) q[0];
rz(3.352416382349567*pi) q[1];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
rxx(0.5*pi) q[0],q[2];
ry(3.5*pi) q[0];
rx(3.5*pi) q[2];
rz(3.5*pi) q[0];
rz(3.0*pi) q[2];
ry(0.5*pi) q[0];
rx(1.897583617650434*pi) q[2];
rxx(0.5*pi) q[0],q[3];
rz(1.0*pi) q[2];
ry(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[2];
rx(3.5*pi) q[3];
rz(3.5*pi) q[0];
ry(3.5*pi) q[1];
rx(3.5*pi) q[2];
rz(0.32797913037736937*pi) q[0];
rz(3.5*pi) q[1];
rx(3.1475836176504335*pi) q[2];
rx(1.0*pi) q[0];
rz(0.39758361765043326*pi) q[1];
rz(3.647583617650433*pi) q[2];
rz(0.5*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[3];
rxx(0.5*pi) q[0],q[4];
ry(3.5*pi) q[1];
rx(3.5*pi) q[3];
ry(3.5*pi) q[0];
rz(3.5*pi) q[1];
rx(3.102416382349566*pi) q[3];
rx(3.5*pi) q[4];
rz(3.5*pi) q[0];
rz(2.25*pi) q[1];
rxx(0.5*pi) q[2],q[3];
rz(3.0*pi) q[4];
rz(3.0779791303773694*pi) q[0];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
rx(2.9220208696226306*pi) q[4];
rx(3.103140847884828*pi) q[0];
rxx(0.5*pi) q[1],q[4];
rz(3.5*pi) q[2];
rx(1.147583617650433*pi) q[3];
rz(0.9433098212789299*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
rz(3.6586484399891046*pi) q[3];
rx(3.5*pi) q[4];
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
ry(0.5*pi) q[3];
rx(2.852416382349566*pi) q[4];
rz(3.1366113868438523*pi) q[1];
rz(1.0*pi) q[4];
rx(3.8527468634981297*pi) q[1];
rxx(0.5*pi) q[2],q[4];
rz(0.0016068846596223318*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[4];
rxx(0.5*pi) q[0],q[1];
rz(3.5*pi) q[2];
rx(2.511064822338671*pi) q[4];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
rz(0.3997654085685294*pi) q[2];
rxx(0.5*pi) q[3],q[4];
rz(3.5*pi) q[0];
rx(2.147583617650433*pi) q[1];
rx(3.1469047755590775*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
rz(1.3975836176504333*pi) q[0];
rz(3.352416382349567*pi) q[1];
rz(3.3895947665495862*pi) q[2];
rz(3.5*pi) q[3];
rz(0.6388857068917924*pi) q[4];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
rz(3.2239266880492075*pi) q[3];
rx(3.6236966424180133*pi) q[4];
rxx(0.5*pi) q[0],q[2];
rx(3.208437619886278*pi) q[3];
rz(3.9702381400645286*pi) q[4];
ry(3.5*pi) q[0];
rx(3.5*pi) q[2];
rz(3.1131487817447985*pi) q[3];
rz(3.5*pi) q[0];
rz(3.0*pi) q[2];
ry(0.5*pi) q[0];
rx(1.897583617650434*pi) q[2];
rxx(0.5*pi) q[0],q[3];
rz(1.0*pi) q[2];
ry(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[2];
rx(3.5*pi) q[3];
rz(3.5*pi) q[0];
ry(3.5*pi) q[1];
rx(3.5*pi) q[2];
rz(1.5525684567112537*pi) q[0];
rz(3.5*pi) q[1];
rx(2.147583617650433*pi) q[2];
ry(0.5*pi) q[0];
rz(0.39758361765043326*pi) q[1];
rz(3.352416382349567*pi) q[2];
rxx(0.5*pi) q[0],q[4];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
ry(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[3];
rx(3.5*pi) q[4];
rz(3.5*pi) q[0];
ry(3.5*pi) q[1];
rx(3.5*pi) q[3];
rz(3.0*pi) q[4];
rz(1.1974315432887452*pi) q[0];
rz(3.5*pi) q[1];
rz(3.0*pi) q[3];
rx(1.7245893263338843*pi) q[4];
rx(3.9227869438379215*pi) q[0];
rz(0.8279791303773694*pi) q[1];
rx(1.897583617650434*pi) q[3];
rz(3.9892855463096595*pi) q[0];
rx(1.0*pi) q[1];
rz(1.0*pi) q[3];
ry(0.5*pi) q[0];
rz(0.5*pi) q[1];
rxx(0.5*pi) q[2],q[3];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
rxx(0.5*pi) q[1],q[4];
rz(3.5*pi) q[2];
rx(2.147583617650433*pi) q[3];
ry(3.5*pi) q[1];
rz(3.325391302139786*pi) q[2];
rz(3.4600152286253083*pi) q[3];
rx(3.5*pi) q[4];
rz(3.5*pi) q[1];
rx(1.0*pi) q[2];
ry(0.5*pi) q[3];
rx(1.4001714458880166*pi) q[4];
rz(1.2785621109234062*pi) q[1];
rz(0.5*pi) q[2];
rx(3.125353556339137*pi) q[1];
ry(0.5*pi) q[2];
rz(3.4866510547573046*pi) q[1];
rxx(0.5*pi) q[2],q[4];
rxx(0.5*pi) q[0],q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[4];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
rz(3.5*pi) q[2];
rx(1.2177924558640445*pi) q[4];
rz(3.5*pi) q[0];
rx(2.147583617650433*pi) q[1];
rz(3.9489419002849915*pi) q[2];
rz(1.0*pi) q[4];
rz(1.3975836176504333*pi) q[0];
rz(3.352416382349567*pi) q[1];
rx(3.197349436563909*pi) q[2];
rxx(0.5*pi) q[3],q[4];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
rz(3.076841038120122*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
rxx(0.5*pi) q[0],q[2];
rz(3.5*pi) q[3];
rz(3.2192393252783913*pi) q[4];
ry(3.5*pi) q[0];
rx(3.5*pi) q[2];
rz(1.402805258782902*pi) q[3];
rx(3.2460387872207774*pi) q[4];
rz(3.5*pi) q[0];
rz(3.0*pi) q[2];
rx(3.8334410192663557*pi) q[3];
rz(0.031257543603472104*pi) q[4];
ry(0.5*pi) q[0];
rx(1.897583617650434*pi) q[2];
rz(0.4851403492013261*pi) q[3];
rxx(0.5*pi) q[0],q[3];
rz(1.0*pi) q[2];
ry(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[2];
rx(3.5*pi) q[3];
rz(3.5*pi) q[0];
ry(3.5*pi) q[1];
rx(3.5*pi) q[2];
rz(0.47114206162369543*pi) q[0];
rz(3.5*pi) q[1];
rx(2.147583617650433*pi) q[2];
rx(1.0*pi) q[0];
rz(0.39758361765043326*pi) q[1];
rz(3.352416382349567*pi) q[2];
rz(0.5*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[3];
rxx(0.5*pi) q[0],q[4];
ry(3.5*pi) q[1];
rx(3.5*pi) q[3];
ry(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.0*pi) q[3];
rx(3.5*pi) q[4];
rz(3.5*pi) q[0];
rz(1.483262291643426*pi) q[1];
rx(1.897583617650434*pi) q[3];
rz(3.0*pi) q[4];
rz(0.22114206162369543*pi) q[0];
ry(0.5*pi) q[1];
rz(1.0*pi) q[3];
rx(2.0121202300197303*pi) q[4];
rx(0.721663453407675*pi) q[0];
rxx(0.5*pi) q[1],q[4];
rxx(0.5*pi) q[2],q[3];
rz(1.5674069853575774*pi) q[0];
ry(3.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
rx(3.5*pi) q[4];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rx(2.147583617650433*pi) q[3];
rx(3.992300456384377*pi) q[4];
rz(0.26673770835657407*pi) q[1];
rz(0.3731463656782359*pi) q[2];
rz(3.9585473873809116*pi) q[3];
rx(0.28499901442566505*pi) q[1];
ry(0.5*pi) q[2];
ry(0.5*pi) q[3];
rz(0.7742903409248028*pi) q[1];
rxx(0.5*pi) q[2],q[4];
ry(3.5*pi) q[2];
rx(3.5*pi) q[4];
rz(3.5*pi) q[2];
rx(1.7329846393531096*pi) q[4];
rz(0.7744372519721974*pi) q[2];
rxx(0.5*pi) q[3],q[4];
rx(0.25475029013882466*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
rz(0.6941560498949538*pi) q[2];
rz(3.5*pi) q[3];
rz(0.07266193680459465*pi) q[4];
rz(3.5414526126190884*pi) q[3];
rx(0.5403889204110931*pi) q[4];
rx(0.213628151069528*pi) q[3];
rz(1.220363452389257*pi) q[4];
rz(0.7336833800400968*pi) q[3];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
