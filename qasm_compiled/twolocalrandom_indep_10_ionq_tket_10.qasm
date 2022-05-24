OPENQASM 2.0;
include "qelib1.inc";

qreg q[10];
creg meas[10];
rz(3.5*pi) q[0];
rz(0.7884359784194837*pi) q[1];
rz(2.782051883564712*pi) q[2];
rz(2.874013141003588*pi) q[3];
rz(0.9360456933807562*pi) q[4];
rz(0.9275373436547709*pi) q[5];
rz(1.8260441898258746*pi) q[6];
rz(3.8683605855997123*pi) q[7];
rz(3.927967711129565*pi) q[8];
rz(0.9027178054559015*pi) q[9];
rx(2.0413441974766053*pi) q[0];
rx(3.8076576429972477*pi) q[1];
rx(3.633868954120193*pi) q[2];
rx(3.7220685424150055*pi) q[3];
rx(3.743396362415494*pi) q[4];
rx(3.7414543383154015*pi) q[5];
rx(3.3103255929102695*pi) q[6];
rx(3.2808467064526825*pi) q[7];
rx(3.258440816687138*pi) q[8];
rx(3.610022790166864*pi) q[9];
rz(2.852416382349567*pi) q[0];
rz(0.24220630487763462*pi) q[1];
rz(2.352404686050709*pi) q[2];
rz(0.18356308791926867*pi) q[3];
rz(0.09107533679523594*pi) q[4];
rz(0.1034039569103482*pi) q[5];
rz(2.7371985586806513*pi) q[6];
rz(0.807590040243818*pi) q[7];
rz(2.897221586105419*pi) q[8];
rz(3.2386701023200724*pi) q[9];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[0],q[1];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
rz(3.5*pi) q[0];
rx(2.147583617650433*pi) q[1];
rz(0.39758361765043326*pi) q[0];
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
rz(1.3975836176504333*pi) q[1];
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
rz(0.39758361765043326*pi) q[2];
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
rz(1.8955561935630412*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
rz(3.5*pi) q[3];
rx(2.147583617650433*pi) q[4];
rx(3.5*pi) q[7];
rx(1.0*pi) q[0];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[6];
rz(1.3975836176504333*pi) q[3];
rz(3.352416382349567*pi) q[4];
rz(0.5*pi) q[0];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
ry(0.5*pi) q[4];
rx(3.5*pi) q[6];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[8];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[5];
rxx(0.5*pi) q[0],q[9];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[5];
rx(3.5*pi) q[8];
ry(3.5*pi) q[0];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[7];
rz(3.5*pi) q[3];
rz(3.0*pi) q[5];
rx(3.5*pi) q[9];
rz(3.5*pi) q[0];
rz(1.75*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
rx(1.897583617650434*pi) q[5];
rx(3.5*pi) q[7];
rz(3.0*pi) q[9];
rz(3.6455561935630407*pi) q[0];
ry(0.5*pi) q[1];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[6];
rz(1.0*pi) q[5];
rx(2.854443806436959*pi) q[9];
rx(2.984128229137556*pi) q[0];
rxx(0.5*pi) q[1],q[9];
ry(0.5*pi) q[2];
ry(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[5];
rx(3.5*pi) q[6];
rz(3.852416382349567*pi) q[0];
ry(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[8];
rz(3.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
rx(3.5*pi) q[9];
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
rz(3.5*pi) q[4];
rx(2.147583617650433*pi) q[5];
rx(3.5*pi) q[8];
rz(3.211600407745193*pi) q[1];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[7];
rz(3.397583617650433*pi) q[4];
rz(3.352416382349567*pi) q[5];
rx(3.7418537019330436*pi) q[1];
rz(2.25*pi) q[2];
ry(3.5*pi) q[3];
rx(3.0*pi) q[4];
ry(0.5*pi) q[5];
rx(3.5*pi) q[7];
rz(2.3424332387550866*pi) q[1];
ry(0.5*pi) q[2];
rz(3.5*pi) q[3];
rz(0.5*pi) q[4];
rxx(0.5*pi) q[0],q[1];
rxx(0.5*pi) q[2],q[9];
ry(0.5*pi) q[3];
ry(0.5*pi) q[4];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
ry(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[8];
rxx(0.5*pi) q[4],q[6];
rx(3.5*pi) q[9];
rz(3.5*pi) q[0];
rx(2.147583617650433*pi) q[1];
rz(3.5*pi) q[2];
ry(3.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[6];
rx(3.5*pi) q[8];
rz(0.39758361765043326*pi) q[0];
rz(3.352416382349567*pi) q[1];
rz(2.980054095214766*pi) q[2];
rz(3.5*pi) q[3];
rz(3.5*pi) q[4];
rz(3.0*pi) q[6];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
rx(3.7493734346600736*pi) q[2];
rz(1.75*pi) q[3];
ry(0.5*pi) q[4];
rx(1.897583617650434*pi) q[6];
rz(0.02822629061974702*pi) q[2];
ry(0.5*pi) q[3];
rxx(0.5*pi) q[4],q[7];
rz(1.0*pi) q[6];
rxx(0.5*pi) q[0],q[2];
rxx(0.5*pi) q[3],q[9];
ry(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[6];
rx(3.5*pi) q[7];
ry(3.5*pi) q[0];
rx(3.5*pi) q[2];
ry(3.5*pi) q[3];
rz(3.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
rx(3.5*pi) q[9];
rz(3.5*pi) q[0];
rz(3.0*pi) q[2];
rz(3.5*pi) q[3];
ry(0.5*pi) q[4];
rz(3.5*pi) q[5];
rx(2.147583617650433*pi) q[6];
rz(1.0*pi) q[9];
ry(0.5*pi) q[0];
rx(1.897583617650434*pi) q[2];
rz(1.2876876314682928*pi) q[3];
rxx(0.5*pi) q[4],q[8];
rz(1.3975836176504333*pi) q[5];
rz(3.352416382349567*pi) q[6];
rx(1.75*pi) q[9];
rz(1.0*pi) q[2];
rx(3.6435725879123084*pi) q[3];
ry(3.5*pi) q[4];
ry(0.5*pi) q[5];
ry(0.5*pi) q[6];
rx(3.5*pi) q[8];
rxx(0.5*pi) q[1],q[2];
rz(0.33905740549289476*pi) q[3];
rz(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[7];
rxx(0.5*pi) q[0],q[3];
ry(3.5*pi) q[1];
rx(3.5*pi) q[2];
ry(0.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[7];
ry(3.5*pi) q[0];
rz(3.5*pi) q[1];
rx(2.147583617650433*pi) q[2];
rx(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[9];
rz(3.5*pi) q[5];
rz(3.0*pi) q[7];
rz(3.5*pi) q[0];
rz(1.3975836176504333*pi) q[1];
rz(3.352416382349567*pi) q[2];
ry(3.5*pi) q[4];
ry(0.5*pi) q[5];
rx(1.897583617650434*pi) q[7];
rx(3.5*pi) q[9];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rz(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[8];
rz(1.0*pi) q[7];
rx(3.75*pi) q[9];
rxx(0.5*pi) q[1],q[3];
rz(3.4295358409597188*pi) q[4];
ry(3.5*pi) q[5];
rxx(0.5*pi) q[6],q[7];
rx(3.5*pi) q[8];
ry(3.5*pi) q[1];
rx(3.5*pi) q[3];
rx(3.315521341614992*pi) q[4];
rz(3.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
rz(3.5*pi) q[1];
rz(3.0*pi) q[3];
rz(3.2728643620291185*pi) q[4];
rz(2.75*pi) q[5];
rz(3.5*pi) q[6];
rx(2.147583617650433*pi) q[7];
rxx(0.5*pi) q[0],q[4];
ry(0.5*pi) q[1];
rx(1.897583617650434*pi) q[3];
ry(0.5*pi) q[5];
rz(0.39758361765043326*pi) q[6];
rz(3.352416382349567*pi) q[7];
ry(3.5*pi) q[0];
rz(1.0*pi) q[3];
rx(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[9];
ry(0.5*pi) q[6];
ry(0.5*pi) q[7];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[4];
rxx(0.5*pi) q[2],q[3];
ry(3.5*pi) q[5];
rxx(0.5*pi) q[6],q[8];
rx(3.5*pi) q[9];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
rx(3.5*pi) q[4];
rz(3.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[8];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rx(2.147583617650433*pi) q[3];
rz(3.4977443651315485*pi) q[5];
rz(3.5*pi) q[6];
rz(3.0*pi) q[8];
ry(0.5*pi) q[1];
rz(0.39758361765043326*pi) q[2];
rz(3.352416382349567*pi) q[3];
rx(3.749992007695622*pi) q[5];
rz(1.25*pi) q[6];
rx(1.897583617650434*pi) q[8];
ry(0.5*pi) q[2];
ry(0.5*pi) q[3];
rz(2.003189976121322*pi) q[5];
ry(0.5*pi) q[6];
rz(1.0*pi) q[8];
rxx(0.5*pi) q[0],q[5];
rxx(0.5*pi) q[2],q[4];
rxx(0.5*pi) q[6],q[9];
rxx(0.5*pi) q[7],q[8];
ry(3.5*pi) q[0];
ry(3.5*pi) q[2];
rx(3.5*pi) q[4];
rx(3.5*pi) q[5];
ry(3.5*pi) q[6];
ry(3.5*pi) q[7];
rx(3.5*pi) q[8];
rx(3.5*pi) q[9];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[5];
rz(3.5*pi) q[2];
rz(3.0*pi) q[4];
rz(3.5*pi) q[6];
rz(3.5*pi) q[7];
rx(2.147583617650433*pi) q[8];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
rx(1.897583617650434*pi) q[4];
rx(3.5*pi) q[5];
rz(2.8437593755995683*pi) q[6];
rz(3.147583617650433*pi) q[7];
rz(0.10728369722787445*pi) q[8];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[5];
rz(1.0*pi) q[4];
rx(3.7038933221506936*pi) q[6];
rx(3.0*pi) q[7];
ry(0.5*pi) q[8];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[4];
rx(3.5*pi) q[5];
rz(0.23225913807662013*pi) q[6];
rz(0.5*pi) q[7];
rxx(0.5*pi) q[0],q[6];
rz(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
ry(0.5*pi) q[7];
ry(3.5*pi) q[0];
ry(0.5*pi) q[2];
rz(3.5*pi) q[3];
rx(2.147583617650433*pi) q[4];
rx(3.5*pi) q[6];
rxx(0.5*pi) q[7],q[9];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[6];
rz(1.3975836176504333*pi) q[3];
rz(3.352416382349567*pi) q[4];
ry(3.5*pi) q[7];
rx(3.5*pi) q[9];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[3];
ry(0.5*pi) q[4];
rx(3.5*pi) q[6];
rz(3.5*pi) q[7];
rx(2.3927163027721257*pi) q[9];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[6];
rxx(0.5*pi) q[3],q[5];
rz(2.2170137555138503*pi) q[7];
rz(1.0*pi) q[9];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[5];
rx(3.5*pi) q[6];
rx(3.3644424602240073*pi) q[7];
rxx(0.5*pi) q[8],q[9];
rz(3.5*pi) q[2];
rz(3.5*pi) q[3];
rz(3.0*pi) q[5];
rz(3.3501296192714074*pi) q[7];
ry(3.5*pi) q[8];
rx(3.5*pi) q[9];
rxx(0.5*pi) q[0],q[7];
ry(0.5*pi) q[2];
ry(0.5*pi) q[3];
rx(1.897583617650434*pi) q[5];
rz(3.5*pi) q[8];
rz(2.0362580401427985*pi) q[9];
ry(3.5*pi) q[0];
rxx(0.5*pi) q[3],q[6];
rz(1.0*pi) q[5];
rx(3.5*pi) q[7];
rz(3.67829143491345*pi) q[8];
rx(3.4958519594098902*pi) q[9];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[7];
ry(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[5];
rx(3.5*pi) q[6];
rx(3.359934203562797*pi) q[8];
rz(0.012614992704323758*pi) q[9];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
rz(3.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
rx(3.5*pi) q[7];
rz(2.656040715078916*pi) q[8];
rxx(0.5*pi) q[0],q[8];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[7];
ry(0.5*pi) q[3];
rz(3.5*pi) q[4];
rx(2.147583617650433*pi) q[5];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
rz(0.39758361765043326*pi) q[4];
rz(3.352416382349567*pi) q[5];
rx(3.5*pi) q[7];
rx(3.5*pi) q[8];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[8];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[7];
ry(0.5*pi) q[4];
ry(0.5*pi) q[5];
rz(1.852416382349567*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
ry(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[6];
rx(3.5*pi) q[7];
rx(3.5*pi) q[8];
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[8];
rz(3.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[6];
rxx(0.5*pi) q[0],q[9];
rz(0.25*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
rz(3.5*pi) q[4];
rz(3.0*pi) q[6];
rx(3.5*pi) q[8];
ry(3.5*pi) q[0];
rx(3.0*pi) q[1];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[8];
ry(0.5*pi) q[4];
rx(1.897583617650434*pi) q[6];
rx(3.5*pi) q[9];
rz(3.5*pi) q[0];
rz(0.5*pi) q[1];
rz(1.25*pi) q[2];
ry(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[7];
rz(1.0*pi) q[6];
rx(3.5*pi) q[8];
rz(3.0*pi) q[9];
rz(2.897583617650433*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rz(3.5*pi) q[3];
ry(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[6];
rx(3.5*pi) q[7];
rx(3.1024163823495665*pi) q[9];
rx(1.7931824081530239*pi) q[0];
rxx(0.5*pi) q[1],q[9];
rz(3.75*pi) q[3];
rz(3.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
rz(3.852416382349567*pi) q[0];
ry(3.5*pi) q[1];
rx(3.0*pi) q[3];
ry(0.5*pi) q[4];
rz(3.5*pi) q[5];
rx(2.147583617650433*pi) q[6];
rx(3.5*pi) q[9];
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[9];
rz(0.5*pi) q[3];
rxx(0.5*pi) q[4],q[8];
rz(1.3975836176504333*pi) q[5];
rz(3.352416382349567*pi) q[6];
rz(0.6691597819606336*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
ry(3.5*pi) q[4];
ry(0.5*pi) q[5];
ry(0.5*pi) q[6];
rx(3.5*pi) q[8];
rx(3.5*pi) q[9];
rx(3.1736059898250293*pi) q[1];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[9];
rz(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[7];
rz(3.191735104137797*pi) q[1];
rz(2.804537356941823*pi) q[2];
ry(3.5*pi) q[3];
rz(0.8524163823495667*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[7];
rx(3.5*pi) q[9];
rxx(0.5*pi) q[0],q[1];
rx(3.6672166508669823*pi) q[2];
rz(3.5*pi) q[3];
ry(0.5*pi) q[4];
rz(3.5*pi) q[5];
rz(3.0*pi) q[7];
rz(3.0*pi) q[9];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
rz(0.30318680510651885*pi) q[2];
rz(2.077849279483048*pi) q[3];
ry(0.5*pi) q[5];
rx(1.897583617650434*pi) q[7];
rx(3.102416382349567*pi) q[9];
rz(3.5*pi) q[0];
rx(2.147583617650433*pi) q[1];
rx(3.259919145885031*pi) q[3];
rxx(0.5*pi) q[4],q[9];
rxx(0.5*pi) q[5],q[8];
rz(1.0*pi) q[7];
rz(0.39758361765043326*pi) q[0];
rz(3.352416382349567*pi) q[1];
rz(3.1112524818551814*pi) q[3];
ry(3.5*pi) q[4];
ry(3.5*pi) q[5];
rxx(0.5*pi) q[6],q[7];
rx(3.5*pi) q[8];
rx(3.5*pi) q[9];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
rz(3.5*pi) q[4];
rz(3.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
rx(0.39758361765043326*pi) q[9];
rxx(0.5*pi) q[0],q[2];
rz(1.3469238414793223*pi) q[4];
rz(2.75*pi) q[5];
rz(3.5*pi) q[6];
rx(2.147583617650433*pi) q[7];
ry(3.5*pi) q[0];
rx(3.5*pi) q[2];
rx(3.2541008299079794*pi) q[4];
ry(0.5*pi) q[5];
rz(2.897583617650433*pi) q[6];
rz(3.352416382349567*pi) q[7];
rz(3.5*pi) q[0];
rz(3.0*pi) q[2];
rz(2.928047124469524*pi) q[4];
rxx(0.5*pi) q[5],q[9];
rx(3.0*pi) q[6];
ry(0.5*pi) q[7];
ry(0.5*pi) q[0];
rx(1.897583617650434*pi) q[2];
ry(3.5*pi) q[5];
rz(0.5*pi) q[6];
rx(3.5*pi) q[9];
rxx(0.5*pi) q[0],q[3];
rz(1.0*pi) q[2];
rz(3.5*pi) q[5];
ry(0.5*pi) q[6];
rz(1.0*pi) q[9];
ry(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[2];
rx(3.5*pi) q[3];
rz(1.3157067009228873*pi) q[5];
rxx(0.5*pi) q[6],q[8];
rx(0.25*pi) q[9];
rz(3.5*pi) q[0];
ry(3.5*pi) q[1];
rx(3.5*pi) q[2];
rx(3.3202726145300305*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[8];
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
rx(2.147583617650433*pi) q[2];
rz(0.7183488443075309*pi) q[5];
rz(3.5*pi) q[6];
rz(3.0*pi) q[8];
rxx(0.5*pi) q[0],q[4];
rz(1.3975836176504333*pi) q[1];
rz(3.352416382349567*pi) q[2];
ry(0.5*pi) q[6];
rx(1.897583617650434*pi) q[8];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rx(3.5*pi) q[4];
rxx(0.5*pi) q[6],q[9];
rz(1.0*pi) q[8];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[3];
ry(3.5*pi) q[6];
rxx(0.5*pi) q[7],q[8];
rx(3.5*pi) q[9];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
rx(3.5*pi) q[3];
rz(3.5*pi) q[6];
ry(3.5*pi) q[7];
rx(3.5*pi) q[8];
rz(1.0*pi) q[9];
rxx(0.5*pi) q[0],q[5];
rz(3.5*pi) q[1];
rz(3.0*pi) q[3];
rz(0.8282488671374897*pi) q[6];
rz(3.5*pi) q[7];
rx(3.147583617650433*pi) q[8];
rx(3.75*pi) q[9];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
rx(1.897583617650434*pi) q[3];
rx(3.5*pi) q[5];
rx(3.2600256061393402*pi) q[6];
rz(1.647583617650433*pi) q[7];
rz(3.037582001545956*pi) q[8];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[4];
rz(1.0*pi) q[3];
rz(3.1118361533994765*pi) q[6];
ry(0.5*pi) q[7];
ry(0.5*pi) q[8];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[3];
rx(3.5*pi) q[4];
rxx(0.5*pi) q[7],q[9];
rxx(0.5*pi) q[0],q[6];
rz(3.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
ry(3.5*pi) q[7];
rx(3.5*pi) q[9];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
rz(3.5*pi) q[2];
rx(2.147583617650433*pi) q[3];
rx(3.5*pi) q[6];
rz(3.5*pi) q[7];
rz(1.0*pi) q[9];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[5];
rz(3.397583617650433*pi) q[2];
rz(3.352416382349567*pi) q[3];
rz(0.8018812595058149*pi) q[7];
rx(1.5375820015459558*pi) q[9];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
rx(3.0*pi) q[2];
ry(0.5*pi) q[3];
rx(3.5*pi) q[5];
rx(3.663918309029484*pi) q[7];
rxx(0.5*pi) q[8],q[9];
rz(3.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.308530829740754*pi) q[7];
ry(3.5*pi) q[8];
rx(3.5*pi) q[9];
rxx(0.5*pi) q[0],q[7];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rz(3.5*pi) q[8];
rx(2.962417998454045*pi) q[9];
ry(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[6];
rxx(0.5*pi) q[2],q[4];
rx(3.5*pi) q[7];
rz(0.10886603978542087*pi) q[8];
rz(3.9935576954822483*pi) q[9];
rz(3.5*pi) q[0];
ry(3.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[4];
rx(3.5*pi) q[6];
rx(3.2894872869300182*pi) q[8];
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rz(3.0*pi) q[4];
rz(3.21610124016747*pi) q[8];
rxx(0.5*pi) q[0],q[8];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rx(1.897583617650434*pi) q[4];
ry(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[7];
rxx(0.5*pi) q[2],q[5];
rz(1.0*pi) q[4];
rx(3.5*pi) q[8];
rz(3.5*pi) q[0];
ry(3.5*pi) q[1];
ry(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[4];
rx(3.5*pi) q[5];
rx(3.5*pi) q[7];
rz(1.25*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rz(3.5*pi) q[3];
rx(2.147583617650433*pi) q[4];
rxx(0.5*pi) q[0],q[9];
rxx(0.5*pi) q[1],q[8];
rxx(0.5*pi) q[2],q[6];
rz(0.39758361765043326*pi) q[3];
rz(3.352416382349567*pi) q[4];
ry(3.5*pi) q[0];
ry(3.5*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
ry(0.5*pi) q[4];
rx(3.5*pi) q[6];
rx(3.5*pi) q[8];
rx(3.5*pi) q[9];
rz(3.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[5];
rz(3.5*pi) q[0];
rz(0.25*pi) q[1];
ry(0.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[5];
rx(0.317047413862183*pi) q[0];
rx(3.0*pi) q[1];
rxx(0.5*pi) q[2],q[7];
rz(3.5*pi) q[3];
rz(3.0*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
rx(1.897583617650434*pi) q[5];
rx(3.5*pi) q[7];
ry(0.5*pi) q[1];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[6];
rz(1.0*pi) q[5];
rxx(0.5*pi) q[1],q[9];
ry(0.5*pi) q[2];
ry(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[5];
rx(3.5*pi) q[6];
ry(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[8];
rz(3.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
rx(3.5*pi) q[9];
rz(3.5*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
rz(3.5*pi) q[4];
rx(2.147583617650433*pi) q[5];
rx(3.5*pi) q[8];
rx(0.75*pi) q[9];
rz(3.0*pi) q[1];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[7];
rz(3.397583617650433*pi) q[4];
rz(3.352416382349567*pi) q[5];
rx(3.16484528624404*pi) q[1];
ry(0.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.0*pi) q[4];
ry(0.5*pi) q[5];
rx(3.5*pi) q[7];
rz(0.5*pi) q[1];
rxx(0.5*pi) q[2],q[9];
rz(3.5*pi) q[3];
rz(0.5*pi) q[4];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
ry(0.5*pi) q[4];
rx(3.5*pi) q[9];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[8];
rxx(0.5*pi) q[4],q[6];
rz(1.75*pi) q[2];
ry(3.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[6];
rx(3.5*pi) q[8];
rx(3.214964383019033*pi) q[2];
rz(3.5*pi) q[3];
rz(3.5*pi) q[4];
rz(3.0*pi) q[6];
rz(0.5*pi) q[2];
ry(0.5*pi) q[3];
ry(0.5*pi) q[4];
rx(1.897583617650434*pi) q[6];
rxx(0.5*pi) q[3],q[9];
rxx(0.5*pi) q[4],q[7];
rz(1.0*pi) q[6];
ry(3.5*pi) q[3];
ry(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[6];
rx(3.5*pi) q[7];
rx(3.5*pi) q[9];
rz(3.5*pi) q[3];
rz(3.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
rz(3.75*pi) q[3];
ry(0.5*pi) q[4];
rz(3.5*pi) q[5];
rx(2.147583617650433*pi) q[6];
rx(0.2845123513321332*pi) q[3];
rxx(0.5*pi) q[4],q[8];
rz(1.3975836176504333*pi) q[5];
rz(3.352416382349567*pi) q[6];
rz(0.5*pi) q[3];
ry(3.5*pi) q[4];
ry(0.5*pi) q[5];
ry(0.5*pi) q[6];
rx(3.5*pi) q[8];
rz(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[7];
ry(0.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[7];
rxx(0.5*pi) q[4],q[9];
rz(3.5*pi) q[5];
rz(3.0*pi) q[7];
ry(3.5*pi) q[4];
ry(0.5*pi) q[5];
rx(1.897583617650434*pi) q[7];
rx(3.5*pi) q[9];
rz(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[8];
rz(1.0*pi) q[7];
rz(3.0*pi) q[9];
rz(1.75*pi) q[4];
ry(3.5*pi) q[5];
rxx(0.5*pi) q[6],q[7];
rx(3.5*pi) q[8];
rx(1.647583617650433*pi) q[9];
rx(3.3073817348093275*pi) q[4];
rz(3.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
rz(0.5*pi) q[4];
rz(2.647583617650433*pi) q[5];
rz(3.5*pi) q[6];
rx(2.147583617650433*pi) q[7];
ry(0.5*pi) q[5];
rz(0.39758361765043326*pi) q[6];
rz(3.352416382349567*pi) q[7];
rxx(0.5*pi) q[5],q[9];
ry(0.5*pi) q[6];
ry(0.5*pi) q[7];
ry(3.5*pi) q[5];
rxx(0.5*pi) q[6],q[8];
rx(3.5*pi) q[9];
rz(3.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[8];
rx(2.102416382349566*pi) q[9];
rz(3.102416382349567*pi) q[5];
rz(3.5*pi) q[6];
rz(3.0*pi) q[8];
rx(0.08071408117202489*pi) q[5];
rz(2.25*pi) q[6];
rx(1.897583617650434*pi) q[8];
rz(0.5*pi) q[5];
ry(0.5*pi) q[6];
rz(1.0*pi) q[8];
rxx(0.5*pi) q[6],q[9];
rxx(0.5*pi) q[7],q[8];
ry(3.5*pi) q[6];
ry(3.5*pi) q[7];
rx(3.5*pi) q[8];
rx(3.5*pi) q[9];
rz(3.5*pi) q[6];
rz(3.5*pi) q[7];
rx(1.147583617650433*pi) q[8];
rx(3.25*pi) q[9];
rz(3.5*pi) q[6];
rz(0.39758361765043326*pi) q[7];
rz(2.9077660420196505*pi) q[8];
rx(0.0018555169951221641*pi) q[6];
ry(0.5*pi) q[7];
ry(0.5*pi) q[8];
rz(0.5*pi) q[6];
rxx(0.5*pi) q[7],q[9];
ry(3.5*pi) q[7];
rx(3.5*pi) q[9];
rz(3.5*pi) q[7];
rz(3.0*pi) q[9];
rz(0.75*pi) q[7];
rx(2.1577660420196505*pi) q[9];
rx(0.19062625884872195*pi) q[7];
rxx(0.5*pi) q[8],q[9];
rz(0.5*pi) q[7];
ry(3.5*pi) q[8];
rx(3.5*pi) q[9];
rz(3.5*pi) q[8];
rz(3.9006250755973357*pi) q[9];
rz(2.5922339579803495*pi) q[8];
rx(0.588004467076874*pi) q[9];
rx(3.3102776105214984*pi) q[8];
rz(3.972029241848566*pi) q[9];
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
