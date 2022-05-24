OPENQASM 2.0;
include "qelib1.inc";

qreg q[9];
creg meas[9];
rz(3.5*pi) q[0];
rz(0.8453230433193262*pi) q[1];
rz(2.7854557979415215*pi) q[2];
rz(2.9118144796132874*pi) q[3];
rz(0.7896467864719243*pi) q[4];
rz(0.9583407233509817*pi) q[5];
rz(1.8242212533049007*pi) q[6];
rz(3.8320473236303405*pi) q[7];
rz(0.12290840158667593*pi) q[8];
rx(2.258088469429002*pi) q[0];
rx(3.8312119917267218*pi) q[1];
rx(3.6398632408483578*pi) q[2];
rx(3.737116976242874*pi) q[3];
rx(3.646698160029351*pi) q[4];
rx(3.7472423177277068*pi) q[5];
rx(3.3119806796825575*pi) q[6];
rx(3.3051453813437606*pi) q[7];
rx(3.7235749777690828*pi) q[8];
rz(2.852416382349567*pi) q[0];
rz(0.17487302214752898*pi) q[1];
rz(2.3442396237381935*pi) q[2];
rz(0.12642142076893415*pi) q[3];
rz(0.33460813905655085*pi) q[4];
rz(0.0590857953699741*pi) q[5];
rz(2.7339380088957532*pi) q[6];
rz(0.7477673212611269*pi) q[7];
rz(0.8212153580391528*pi) q[8];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[0],q[1];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
rz(3.5*pi) q[0];
rx(2.147583617650433*pi) q[1];
rz(0.8975836176504333*pi) q[0];
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
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
rx(2.147583617650433*pi) q[2];
rxx(0.5*pi) q[0],q[4];
rz(0.39758361765043326*pi) q[1];
rz(3.352416382349567*pi) q[2];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rx(3.5*pi) q[4];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[3];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
rx(3.5*pi) q[3];
rxx(0.5*pi) q[0],q[5];
rz(3.5*pi) q[1];
rz(3.0*pi) q[3];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
rx(1.897583617650434*pi) q[3];
rx(3.5*pi) q[5];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[4];
rz(1.0*pi) q[3];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[3];
rx(3.5*pi) q[4];
rxx(0.5*pi) q[0],q[6];
rz(3.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
rz(3.5*pi) q[2];
rx(2.147583617650433*pi) q[3];
rx(3.5*pi) q[6];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[5];
rz(1.3975836176504333*pi) q[2];
rz(3.352416382349567*pi) q[3];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
ry(0.5*pi) q[3];
rx(3.5*pi) q[5];
rxx(0.5*pi) q[0],q[7];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[4];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[4];
rx(3.5*pi) q[7];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[6];
rz(3.5*pi) q[2];
rz(3.0*pi) q[4];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
rx(1.897583617650434*pi) q[4];
rx(3.5*pi) q[6];
rxx(0.5*pi) q[0],q[8];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[5];
rz(1.0*pi) q[4];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[4];
rx(3.5*pi) q[5];
rx(3.5*pi) q[8];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[7];
rz(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
rz(3.0*pi) q[8];
rz(3.25*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
rz(3.5*pi) q[3];
rx(2.147583617650433*pi) q[4];
rx(3.5*pi) q[7];
rx(1.25*pi) q[8];
rx(1.8555311748478476*pi) q[0];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[6];
rz(0.39758361765043326*pi) q[3];
rz(3.352416382349567*pi) q[4];
rz(3.852416382349567*pi) q[0];
rz(1.25*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
ry(0.5*pi) q[4];
rx(3.5*pi) q[6];
ry(0.5*pi) q[0];
rx(3.0*pi) q[1];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[5];
rz(0.5*pi) q[1];
ry(0.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[5];
ry(0.5*pi) q[1];
rxx(0.5*pi) q[2],q[7];
rz(3.5*pi) q[3];
rz(3.0*pi) q[5];
rxx(0.5*pi) q[1],q[8];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
rx(1.897583617650434*pi) q[5];
rx(3.5*pi) q[7];
ry(3.5*pi) q[1];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[6];
rz(1.0*pi) q[5];
rx(3.5*pi) q[8];
rz(3.5*pi) q[1];
rz(1.1475836176504333*pi) q[2];
ry(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[5];
rx(3.5*pi) q[6];
rx(1.1024163823495676*pi) q[8];
rz(1.815985920654914*pi) q[1];
ry(0.5*pi) q[2];
rz(3.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
rz(1.0*pi) q[8];
rx(3.3051529328205618*pi) q[1];
rxx(0.5*pi) q[2],q[8];
ry(0.5*pi) q[3];
rz(3.5*pi) q[4];
rx(2.147583617650433*pi) q[5];
rz(3.385825693622861*pi) q[1];
ry(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[7];
rz(3.897583617650433*pi) q[4];
rz(3.352416382349567*pi) q[5];
rx(3.5*pi) q[8];
rxx(0.5*pi) q[0],q[1];
rz(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.0*pi) q[4];
ry(0.5*pi) q[5];
rx(3.5*pi) q[7];
rx(2.602416382349567*pi) q[8];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
rz(3.893816217290937*pi) q[2];
rz(3.5*pi) q[3];
rz(0.5*pi) q[4];
rz(3.5*pi) q[0];
rx(2.147583617650433*pi) q[1];
rx(3.649398672176178*pi) q[2];
rz(2.25*pi) q[3];
ry(0.5*pi) q[4];
rz(1.3975836176504333*pi) q[0];
rz(3.352416382349567*pi) q[1];
rz(2.3306996882476576*pi) q[2];
ry(0.5*pi) q[3];
rxx(0.5*pi) q[4],q[6];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
rxx(0.5*pi) q[3],q[8];
ry(3.5*pi) q[4];
rx(3.5*pi) q[6];
rxx(0.5*pi) q[0],q[2];
ry(3.5*pi) q[3];
rz(3.5*pi) q[4];
rz(3.0*pi) q[6];
rx(3.5*pi) q[8];
ry(3.5*pi) q[0];
rx(3.5*pi) q[2];
rz(3.5*pi) q[3];
ry(0.5*pi) q[4];
rx(1.897583617650434*pi) q[6];
rx(3.25*pi) q[8];
rz(3.5*pi) q[0];
rz(3.0*pi) q[2];
rz(0.7834777825933772*pi) q[3];
rxx(0.5*pi) q[4],q[7];
rz(1.0*pi) q[6];
rz(3.0*pi) q[8];
ry(0.5*pi) q[0];
rx(1.897583617650434*pi) q[2];
rx(3.636432372271939*pi) q[3];
ry(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[6];
rx(3.5*pi) q[7];
rz(1.0*pi) q[2];
rz(0.34894336626269695*pi) q[3];
rz(3.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
rxx(0.5*pi) q[0],q[3];
rxx(0.5*pi) q[1],q[2];
ry(0.5*pi) q[4];
rz(3.5*pi) q[5];
rx(2.147583617650433*pi) q[6];
ry(3.5*pi) q[0];
ry(3.5*pi) q[1];
rx(3.5*pi) q[2];
rx(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[8];
rz(0.39758361765043326*pi) q[5];
rz(3.352416382349567*pi) q[6];
rz(3.5*pi) q[0];
rz(3.5*pi) q[1];
rx(2.147583617650433*pi) q[2];
ry(3.5*pi) q[4];
ry(0.5*pi) q[5];
ry(0.5*pi) q[6];
rx(3.5*pi) q[8];
ry(0.5*pi) q[0];
rz(0.39758361765043326*pi) q[1];
rz(3.352416382349567*pi) q[2];
rz(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[7];
rx(0.25*pi) q[8];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rz(0.9142833918817395*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[7];
rxx(0.5*pi) q[1],q[3];
rx(3.302169842568782*pi) q[4];
rz(3.5*pi) q[5];
rz(3.0*pi) q[7];
ry(3.5*pi) q[1];
rx(3.5*pi) q[3];
rz(1.2458895882343366*pi) q[4];
rz(3.75*pi) q[5];
rx(1.897583617650434*pi) q[7];
rxx(0.5*pi) q[0],q[4];
rz(3.5*pi) q[1];
rz(3.0*pi) q[3];
rx(3.0*pi) q[5];
rz(1.0*pi) q[7];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
rx(1.897583617650434*pi) q[3];
rx(3.5*pi) q[4];
rz(0.5*pi) q[5];
rxx(0.5*pi) q[6],q[7];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[4];
rz(1.0*pi) q[3];
ry(0.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[3];
rx(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[8];
rz(3.5*pi) q[6];
rx(2.147583617650433*pi) q[7];
rz(3.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
ry(3.5*pi) q[5];
rz(0.22556274802780263*pi) q[6];
rz(3.480131475694459*pi) q[7];
rx(3.5*pi) q[8];
ry(0.5*pi) q[1];
rz(3.5*pi) q[2];
rx(2.147583617650433*pi) q[3];
rz(3.5*pi) q[5];
ry(0.5*pi) q[6];
ry(0.5*pi) q[7];
rz(3.0*pi) q[8];
rz(1.3975836176504333*pi) q[2];
rz(3.352416382349567*pi) q[3];
rz(0.1937705893235111*pi) q[5];
rx(3.0779791303773694*pi) q[8];
ry(0.5*pi) q[2];
ry(0.5*pi) q[3];
rx(3.3307520136573547*pi) q[5];
rxx(0.5*pi) q[6],q[8];
rxx(0.5*pi) q[2],q[4];
rz(3.2998312843363418*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[8];
rxx(0.5*pi) q[0],q[5];
ry(3.5*pi) q[2];
rx(3.5*pi) q[4];
rz(3.5*pi) q[6];
rz(3.0*pi) q[8];
ry(3.5*pi) q[0];
rz(3.5*pi) q[2];
rz(3.0*pi) q[4];
rx(3.5*pi) q[5];
rz(2.353510775843343*pi) q[6];
rx(1.5978476546829106*pi) q[8];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[5];
ry(0.5*pi) q[2];
rx(1.897583617650434*pi) q[4];
rx(3.7423905437613234*pi) q[6];
rz(1.0*pi) q[8];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
rz(1.0*pi) q[4];
rx(3.5*pi) q[5];
rz(0.09766685639633588*pi) q[6];
rxx(0.5*pi) q[7],q[8];
rxx(0.5*pi) q[0],q[6];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[5];
rxx(0.5*pi) q[3],q[4];
ry(3.5*pi) q[7];
rx(3.5*pi) q[8];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
rx(3.5*pi) q[5];
rx(3.5*pi) q[6];
rz(3.5*pi) q[7];
rz(3.8495337550535154*pi) q[8];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[6];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rx(2.147583617650433*pi) q[4];
rz(1.5056281267689338*pi) q[7];
rx(3.906790433565696*pi) q[8];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
rz(0.39758361765043326*pi) q[3];
rz(3.352416382349567*pi) q[4];
rx(3.5*pi) q[6];
rx(3.749681034182846*pi) q[7];
rz(2.1589964737638114*pi) q[8];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[6];
ry(0.5*pi) q[3];
ry(0.5*pi) q[4];
rz(0.020145692624330136*pi) q[7];
rxx(0.5*pi) q[0],q[7];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[5];
rx(3.5*pi) q[6];
ry(3.5*pi) q[0];
rz(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[5];
rx(3.5*pi) q[7];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[7];
ry(0.5*pi) q[2];
rz(3.5*pi) q[3];
rz(3.0*pi) q[5];
rz(1.1475836176504333*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[3];
rx(1.897583617650434*pi) q[5];
rx(3.5*pi) q[7];
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[7];
rxx(0.5*pi) q[3],q[6];
rz(1.0*pi) q[5];
rxx(0.5*pi) q[0],q[8];
rz(0.75*pi) q[1];
ry(3.5*pi) q[2];
ry(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[5];
rx(3.5*pi) q[6];
rx(3.5*pi) q[7];
ry(3.5*pi) q[0];
rx(3.0*pi) q[1];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
rx(3.5*pi) q[8];
rz(3.5*pi) q[0];
rz(0.5*pi) q[1];
rz(1.25*pi) q[2];
ry(0.5*pi) q[3];
rz(3.5*pi) q[4];
rx(2.147583617650433*pi) q[5];
rz(1.0*pi) q[8];
rz(3.602416382349567*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rxx(0.5*pi) q[3],q[7];
rz(1.3975836176504333*pi) q[4];
rz(3.352416382349567*pi) q[5];
rx(1.8975836176504333*pi) q[8];
rx(1.7820457541900414*pi) q[0];
rxx(0.5*pi) q[1],q[8];
ry(3.5*pi) q[3];
ry(0.5*pi) q[4];
ry(0.5*pi) q[5];
rx(3.5*pi) q[7];
rz(3.852416382349567*pi) q[0];
ry(3.5*pi) q[1];
rz(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[6];
rx(3.5*pi) q[8];
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[8];
rz(0.25*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[6];
rz(3.2155321990021193*pi) q[1];
ry(3.5*pi) q[2];
rx(3.0*pi) q[3];
rz(3.5*pi) q[4];
rz(3.0*pi) q[6];
rx(3.5*pi) q[8];
rx(3.1945328671269655*pi) q[1];
rz(3.5*pi) q[2];
rz(0.5*pi) q[3];
ry(0.5*pi) q[4];
rx(1.897583617650434*pi) q[6];
rz(3.24703981497127*pi) q[1];
rz(2.923638510458944*pi) q[2];
ry(0.5*pi) q[3];
rxx(0.5*pi) q[4],q[7];
rz(1.0*pi) q[6];
rxx(0.5*pi) q[0],q[1];
rx(3.7404716408533014*pi) q[2];
rxx(0.5*pi) q[3],q[8];
ry(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[6];
rx(3.5*pi) q[7];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
rz(0.10908109957429668*pi) q[2];
ry(3.5*pi) q[3];
rz(3.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
rx(3.5*pi) q[8];
rz(3.5*pi) q[0];
rx(2.147583617650433*pi) q[1];
rz(3.5*pi) q[3];
rz(2.4371670418109987*pi) q[4];
rz(3.5*pi) q[5];
rx(2.147583617650433*pi) q[6];
rx(0.812832958189001*pi) q[8];
rz(1.3975836176504333*pi) q[0];
rz(3.352416382349567*pi) q[1];
rz(3.603420970166521*pi) q[3];
ry(0.5*pi) q[4];
rz(0.39758361765043326*pi) q[5];
rz(3.352416382349567*pi) q[6];
rz(1.0*pi) q[8];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
rx(3.731902720238348*pi) q[3];
rxx(0.5*pi) q[4],q[8];
ry(0.5*pi) q[5];
ry(0.5*pi) q[6];
rxx(0.5*pi) q[0],q[2];
rz(1.8509085043532714*pi) q[3];
ry(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[7];
rx(3.5*pi) q[8];
ry(3.5*pi) q[0];
rx(3.5*pi) q[2];
rz(3.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[7];
rx(0.3128329581890011*pi) q[8];
rz(3.5*pi) q[0];
rz(3.0*pi) q[2];
rz(3.7214057879714826*pi) q[4];
rz(3.5*pi) q[5];
rz(3.0*pi) q[7];
rz(1.0*pi) q[8];
ry(0.5*pi) q[0];
rx(1.897583617650434*pi) q[2];
rx(3.2639052832806175*pi) q[4];
rz(0.75*pi) q[5];
rx(1.897583617650434*pi) q[7];
rxx(0.5*pi) q[0],q[3];
rz(1.0*pi) q[2];
rz(2.8687885048142165*pi) q[4];
rx(3.0*pi) q[5];
rz(1.0*pi) q[7];
ry(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[2];
rx(3.5*pi) q[3];
rz(0.5*pi) q[5];
rxx(0.5*pi) q[6],q[7];
rz(3.5*pi) q[0];
ry(3.5*pi) q[1];
rx(3.5*pi) q[2];
ry(0.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
rx(2.147583617650433*pi) q[2];
rxx(0.5*pi) q[5],q[8];
rz(3.5*pi) q[6];
rx(1.147583617650433*pi) q[7];
rxx(0.5*pi) q[0],q[4];
rz(2.397583617650433*pi) q[1];
rz(3.352416382349567*pi) q[2];
ry(3.5*pi) q[5];
rz(0.897583617650433*pi) q[6];
rz(0.11119595738220689*pi) q[7];
rx(3.5*pi) q[8];
ry(3.5*pi) q[0];
rx(3.0*pi) q[1];
ry(0.5*pi) q[2];
rx(3.5*pi) q[4];
rz(3.5*pi) q[5];
rx(1.0*pi) q[6];
ry(0.5*pi) q[7];
rz(3.0*pi) q[8];
rz(3.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.12512750119098892*pi) q[5];
rz(0.5*pi) q[6];
rx(1.25*pi) q[8];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
rx(3.7224946967031927*pi) q[5];
ry(0.5*pi) q[6];
rz(1.0*pi) q[8];
rxx(0.5*pi) q[1],q[3];
rz(3.8177735368219246*pi) q[5];
rxx(0.5*pi) q[6],q[8];
rxx(0.5*pi) q[0],q[5];
ry(3.5*pi) q[1];
rx(3.5*pi) q[3];
ry(3.5*pi) q[6];
rx(3.5*pi) q[8];
ry(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.0*pi) q[3];
rx(3.5*pi) q[5];
rz(3.5*pi) q[6];
rx(0.36119595738220606*pi) q[8];
rz(3.5*pi) q[0];
ry(0.5*pi) q[1];
rx(1.897583617650434*pi) q[3];
rz(0.7884552987026849*pi) q[6];
rxx(0.5*pi) q[7],q[8];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[4];
rz(1.0*pi) q[3];
rx(3.747654215062138*pi) q[6];
ry(3.5*pi) q[7];
rx(3.5*pi) q[8];
ry(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[3];
rx(3.5*pi) q[4];
rz(3.9454820376946116*pi) q[6];
rz(3.5*pi) q[7];
rz(3.215969045797779*pi) q[8];
rxx(0.5*pi) q[0],q[6];
rz(3.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
rz(0.10482459083085471*pi) q[7];
rx(0.5879613163188039*pi) q[8];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
rz(3.5*pi) q[2];
rx(2.147583617650433*pi) q[3];
rx(3.5*pi) q[6];
rx(3.6373159123685848*pi) q[7];
rz(3.06891465755266*pi) q[8];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[5];
rz(1.3975836176504333*pi) q[2];
rz(3.352416382349567*pi) q[3];
rz(1.6522600405267194*pi) q[7];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
ry(0.5*pi) q[3];
rx(3.5*pi) q[5];
rxx(0.5*pi) q[0],q[7];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[4];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[4];
rx(3.5*pi) q[7];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[6];
rz(3.5*pi) q[2];
rz(3.0*pi) q[4];
rz(3.75*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
rx(1.897583617650434*pi) q[4];
rx(3.5*pi) q[6];
rx(3.0*pi) q[0];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[5];
rz(1.0*pi) q[4];
rz(0.5*pi) q[0];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[4];
rx(3.5*pi) q[5];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[7];
rz(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
rxx(0.5*pi) q[0],q[8];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
rz(3.5*pi) q[3];
rx(2.147583617650433*pi) q[4];
rx(3.5*pi) q[7];
ry(3.5*pi) q[0];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[6];
rz(0.39758361765043326*pi) q[3];
rz(3.352416382349567*pi) q[4];
rx(3.5*pi) q[8];
rz(3.5*pi) q[0];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
ry(0.5*pi) q[4];
rx(3.5*pi) q[6];
rz(3.0*pi) q[8];
rz(2.5*pi) q[0];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[5];
rx(1.25*pi) q[8];
rx(3.1666317979535794*pi) q[0];
rxx(0.5*pi) q[1],q[8];
ry(0.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[5];
rz(0.5*pi) q[0];
ry(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[7];
rz(3.5*pi) q[3];
rz(3.0*pi) q[5];
rx(3.5*pi) q[8];
rz(3.5*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
rx(1.897583617650434*pi) q[5];
rx(3.5*pi) q[7];
rz(1.75*pi) q[1];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[6];
rz(1.0*pi) q[5];
rx(3.2023292138427584*pi) q[1];
ry(0.5*pi) q[2];
ry(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[5];
rx(3.5*pi) q[6];
rz(0.5*pi) q[1];
rxx(0.5*pi) q[2],q[8];
rz(3.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
rz(3.5*pi) q[4];
rx(2.147583617650433*pi) q[5];
rx(3.5*pi) q[8];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[7];
rz(1.3975836176504333*pi) q[4];
rz(3.352416382349567*pi) q[5];
rz(3.75*pi) q[2];
ry(3.5*pi) q[3];
ry(0.5*pi) q[4];
ry(0.5*pi) q[5];
rx(3.5*pi) q[7];
rx(0.15608362841881349*pi) q[2];
rz(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[6];
rz(0.5*pi) q[2];
ry(0.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[6];
rxx(0.5*pi) q[3],q[8];
rz(3.5*pi) q[4];
rz(3.0*pi) q[6];
ry(3.5*pi) q[3];
ry(0.5*pi) q[4];
rx(1.897583617650434*pi) q[6];
rx(3.5*pi) q[8];
rz(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[7];
rz(1.0*pi) q[6];
rx(3.75*pi) q[8];
rz(3.75*pi) q[3];
ry(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[6];
rx(3.5*pi) q[7];
rx(0.18188804256884666*pi) q[3];
rz(3.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
rz(0.5*pi) q[3];
rz(2.25*pi) q[4];
rz(3.5*pi) q[5];
rx(2.147583617650433*pi) q[6];
ry(0.5*pi) q[4];
rz(0.39758361765043326*pi) q[5];
rz(3.352416382349567*pi) q[6];
rxx(0.5*pi) q[4],q[8];
ry(0.5*pi) q[5];
ry(0.5*pi) q[6];
ry(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[7];
rx(3.5*pi) q[8];
rz(3.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[7];
rz(1.0*pi) q[8];
rz(3.5*pi) q[4];
rz(3.5*pi) q[5];
rz(3.0*pi) q[7];
rx(2.2219380415110117*pi) q[8];
rx(0.29812384923534335*pi) q[4];
rz(1.5280619584889887*pi) q[5];
rx(1.897583617650434*pi) q[7];
rz(1.0*pi) q[8];
rz(0.5*pi) q[4];
ry(0.5*pi) q[5];
rz(1.0*pi) q[7];
rxx(0.5*pi) q[5],q[8];
rxx(0.5*pi) q[6],q[7];
ry(3.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
rx(3.5*pi) q[8];
rz(3.5*pi) q[5];
rz(3.5*pi) q[6];
rx(2.147583617650433*pi) q[7];
rz(3.0*pi) q[8];
rz(0.2219380415110117*pi) q[5];
rz(0.14758361765043326*pi) q[6];
rz(3.956353990884495*pi) q[7];
rx(2.7780619584889887*pi) q[8];
rx(0.19542610595504162*pi) q[5];
rx(3.0*pi) q[6];
ry(0.5*pi) q[7];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
ry(0.5*pi) q[6];
rxx(0.5*pi) q[6],q[8];
ry(3.5*pi) q[6];
rx(3.5*pi) q[8];
rz(3.5*pi) q[6];
rz(3.0*pi) q[8];
rz(3.5*pi) q[6];
rx(0.054602033329815856*pi) q[8];
rx(3.031455160039848*pi) q[6];
rxx(0.5*pi) q[7],q[8];
rz(0.5*pi) q[6];
ry(3.5*pi) q[7];
rx(3.5*pi) q[8];
rz(3.5*pi) q[7];
rz(0.4548331123223238*pi) q[8];
rz(3.5436460091155055*pi) q[7];
rx(0.717920366200437*pi) q[8];
rx(0.11235522301428987*pi) q[7];
rz(1.4292797472001864*pi) q[8];
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
