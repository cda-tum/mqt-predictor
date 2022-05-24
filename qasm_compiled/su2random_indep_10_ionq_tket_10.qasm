OPENQASM 2.0;
include "qelib1.inc";

qreg q[10];
creg meas[10];
rz(3.5*pi) q[0];
rz(0.6859456757637594*pi) q[1];
rz(0.8260554584015154*pi) q[2];
rz(2.8783855005324317*pi) q[3];
rz(1.1788425267880143*pi) q[4];
rz(1.124569466896041*pi) q[5];
rz(0.9231647904026987*pi) q[6];
rz(2.878305528279951*pi) q[7];
rz(0.8404675516542734*pi) q[8];
rz(1.3379532154138718*pi) q[9];
rx(0.0363159080696323*pi) q[0];
rx(3.7110912714689643*pi) q[1];
rx(3.7231250006060033*pi) q[2];
rx(3.7383386478835776*pi) q[3];
rx(3.7747715220041567*pi) q[4];
rx(3.8062784898085504*pi) q[5];
rx(3.7635471074228657*pi) q[6];
rx(3.7245056274062356*pi) q[7];
rx(3.72081919915755*pi) q[8];
rx(3.5265356573114968*pi) q[9];
rz(0.8567226758419846*pi) q[0];
rz(0.4612559917234631*pi) q[1];
rz(0.40987029068494657*pi) q[2];
rz(0.29236235859327975*pi) q[3];
rz(0.0509054894689025*pi) q[4];
rz(0.13777575769599842*pi) q[5];
rz(0.26618256737830837*pi) q[6];
rz(0.19622499346788103*pi) q[7];
rz(0.35914811334686575*pi) q[8];
rz(0.1229697849460114*pi) q[9];
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
rz(2.0779791303773694*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
rz(3.5*pi) q[3];
rx(2.147583617650433*pi) q[4];
rx(3.5*pi) q[7];
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[6];
rz(1.3975836176504333*pi) q[3];
rz(3.352416382349567*pi) q[4];
rxx(0.5*pi) q[0],q[9];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
ry(0.5*pi) q[4];
rx(3.5*pi) q[6];
ry(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[8];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[5];
rx(3.5*pi) q[9];
rz(3.5*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[5];
rx(3.5*pi) q[8];
rx(1.376853634321764*pi) q[9];
rz(0.6720208696226289*pi) q[0];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[7];
rz(3.5*pi) q[3];
rz(3.0*pi) q[5];
rx(3.9177061355473355*pi) q[0];
rz(1.9548327646991335*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
rx(1.897583617650434*pi) q[5];
rx(3.5*pi) q[7];
rz(3.970643199430895*pi) q[0];
ry(0.5*pi) q[1];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[6];
rz(1.0*pi) q[5];
ry(0.5*pi) q[0];
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
rx(1.397583617650433*pi) q[9];
rz(3.1147802322119826*pi) q[1];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[7];
rz(0.39758361765043326*pi) q[4];
rz(3.352416382349567*pi) q[5];
rz(3.0*pi) q[9];
rx(3.8620991899390114*pi) q[1];
rz(1.352416382349567*pi) q[2];
ry(3.5*pi) q[3];
ry(0.5*pi) q[4];
ry(0.5*pi) q[5];
rx(3.5*pi) q[7];
rz(2.4210769131063317*pi) q[1];
rx(1.0*pi) q[2];
rz(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[6];
rxx(0.5*pi) q[0],q[1];
rz(0.5*pi) q[2];
ry(0.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[6];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
ry(0.5*pi) q[2];
rxx(0.5*pi) q[3],q[8];
rz(3.5*pi) q[4];
rz(3.0*pi) q[6];
rz(3.5*pi) q[0];
rx(2.147583617650433*pi) q[1];
rxx(0.5*pi) q[2],q[9];
ry(3.5*pi) q[3];
ry(0.5*pi) q[4];
rx(1.897583617650434*pi) q[6];
rx(3.5*pi) q[8];
rz(2.897583617650433*pi) q[0];
rz(3.352416382349567*pi) q[1];
ry(3.5*pi) q[2];
rz(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[7];
rz(1.0*pi) q[6];
rx(3.5*pi) q[9];
rx(3.0*pi) q[0];
ry(0.5*pi) q[1];
rz(3.5*pi) q[2];
rz(1.297864712134245*pi) q[3];
ry(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[6];
rx(3.5*pi) q[7];
rz(3.0*pi) q[9];
rz(0.5*pi) q[0];
rz(2.642414021724961*pi) q[2];
ry(0.5*pi) q[3];
rz(3.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
rx(1.9454483297846803*pi) q[9];
ry(0.5*pi) q[0];
rx(3.784334201005242*pi) q[2];
rxx(0.5*pi) q[3],q[9];
ry(0.5*pi) q[4];
rz(3.5*pi) q[5];
rx(2.147583617650433*pi) q[6];
rz(3.7409565512690803*pi) q[2];
ry(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[8];
rz(1.3975836176504333*pi) q[5];
rz(3.352416382349567*pi) q[6];
rx(3.5*pi) q[9];
rxx(0.5*pi) q[0],q[2];
rz(3.5*pi) q[3];
ry(3.5*pi) q[4];
ry(0.5*pi) q[5];
ry(0.5*pi) q[6];
rx(3.5*pi) q[8];
rx(3.6569680525648876*pi) q[9];
ry(3.5*pi) q[0];
rx(3.5*pi) q[2];
rz(0.771526018521927*pi) q[3];
rz(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[7];
rz(1.0*pi) q[9];
rz(3.5*pi) q[0];
rz(3.0*pi) q[2];
rx(3.2990952176018005*pi) q[3];
rz(1.9548327646991335*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[7];
ry(0.5*pi) q[0];
rx(1.897583617650434*pi) q[2];
rz(0.6175566095690873*pi) q[3];
rx(1.0*pi) q[4];
rz(3.5*pi) q[5];
rz(3.0*pi) q[7];
rxx(0.5*pi) q[0],q[3];
rz(1.0*pi) q[2];
rz(0.5*pi) q[4];
ry(0.5*pi) q[5];
rx(1.897583617650434*pi) q[7];
ry(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[2];
rx(3.5*pi) q[3];
ry(0.5*pi) q[4];
rxx(0.5*pi) q[5],q[8];
rz(1.0*pi) q[7];
rz(3.5*pi) q[0];
ry(3.5*pi) q[1];
rx(3.5*pi) q[2];
rxx(0.5*pi) q[4],q[9];
ry(3.5*pi) q[5];
rxx(0.5*pi) q[6],q[7];
rx(3.5*pi) q[8];
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
rx(2.147583617650433*pi) q[2];
ry(3.5*pi) q[4];
rz(3.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
rx(3.5*pi) q[9];
rz(1.3975836176504333*pi) q[1];
rz(3.352416382349567*pi) q[2];
rz(3.5*pi) q[4];
rz(1.5872514067288517*pi) q[5];
rz(3.5*pi) q[6];
rx(2.147583617650433*pi) q[7];
rx(0.3675813579702817*pi) q[9];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rz(3.3493340954206916*pi) q[4];
ry(0.5*pi) q[5];
rz(0.39758361765043326*pi) q[6];
rz(3.352416382349567*pi) q[7];
rz(1.0*pi) q[9];
rxx(0.5*pi) q[1],q[3];
rx(3.751102604546179*pi) q[4];
rxx(0.5*pi) q[5],q[9];
ry(0.5*pi) q[6];
ry(0.5*pi) q[7];
ry(3.5*pi) q[1];
rx(3.5*pi) q[3];
rz(1.6133158876798923*pi) q[4];
ry(3.5*pi) q[5];
rxx(0.5*pi) q[6],q[8];
rx(3.5*pi) q[9];
rxx(0.5*pi) q[0],q[4];
rz(3.5*pi) q[1];
rz(3.0*pi) q[3];
rz(3.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[8];
rz(3.0*pi) q[9];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
rx(1.897583617650434*pi) q[3];
rx(3.5*pi) q[4];
rz(0.6324154080707229*pi) q[5];
rz(3.5*pi) q[6];
rz(3.0*pi) q[8];
rx(2.7348350243792847*pi) q[9];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[4];
rz(1.0*pi) q[3];
rx(3.141449821917548*pi) q[5];
rz(0.8524163823495667*pi) q[6];
rx(1.897583617650434*pi) q[8];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[3];
rx(3.5*pi) q[4];
rz(0.6234341954956503*pi) q[5];
rx(1.0*pi) q[6];
rz(1.0*pi) q[8];
rxx(0.5*pi) q[0],q[5];
rz(3.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
rz(0.5*pi) q[6];
rxx(0.5*pi) q[7],q[8];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
rz(3.5*pi) q[2];
rx(2.147583617650433*pi) q[3];
rx(3.5*pi) q[5];
ry(0.5*pi) q[6];
ry(3.5*pi) q[7];
rx(3.5*pi) q[8];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[5];
rz(0.39758361765043326*pi) q[2];
rz(3.352416382349567*pi) q[3];
rxx(0.5*pi) q[6],q[9];
rz(3.5*pi) q[7];
rx(1.147583617650433*pi) q[8];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
ry(0.5*pi) q[3];
rx(3.5*pi) q[5];
ry(3.5*pi) q[6];
rz(0.15752756547402247*pi) q[7];
rz(3.584220317482921*pi) q[8];
rx(3.5*pi) q[9];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[4];
rz(3.5*pi) q[6];
ry(0.5*pi) q[7];
ry(0.5*pi) q[8];
rx(1.0924724345259775*pi) q[9];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[4];
rz(2.2309744703503362*pi) q[6];
rz(1.0*pi) q[9];
rz(3.5*pi) q[2];
rz(3.0*pi) q[4];
rx(3.7280840528031467*pi) q[6];
rxx(0.5*pi) q[7],q[9];
ry(0.5*pi) q[2];
rx(1.897583617650434*pi) q[4];
rz(3.7287399587568784*pi) q[6];
ry(3.5*pi) q[7];
rx(3.5*pi) q[9];
rxx(0.5*pi) q[0],q[6];
rxx(0.5*pi) q[2],q[5];
rz(1.0*pi) q[4];
rz(3.5*pi) q[7];
rz(1.0*pi) q[9];
ry(3.5*pi) q[0];
ry(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[4];
rx(3.5*pi) q[5];
rx(3.5*pi) q[6];
rz(3.347260519232947*pi) q[7];
rx(0.5941642653065117*pi) q[9];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[6];
rz(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
rx(3.284758025238447*pi) q[7];
rxx(0.5*pi) q[8],q[9];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
rz(3.5*pi) q[3];
rx(2.147583617650433*pi) q[4];
rx(3.5*pi) q[6];
rz(0.7416901283710577*pi) q[7];
ry(3.5*pi) q[8];
rx(3.5*pi) q[9];
rxx(0.5*pi) q[0],q[7];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[6];
rz(1.3975836176504333*pi) q[3];
rz(3.352416382349567*pi) q[4];
rz(3.5*pi) q[8];
rz(2.754127870498051*pi) q[9];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
ry(0.5*pi) q[4];
rx(3.5*pi) q[6];
rx(3.5*pi) q[7];
rz(0.37223700469191834*pi) q[8];
rx(3.474528951355921*pi) q[9];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[7];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[5];
rx(3.773662198217282*pi) q[8];
rz(0.7845561905623915*pi) q[9];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[5];
rx(3.5*pi) q[7];
rz(3.8791009484059895*pi) q[8];
rxx(0.5*pi) q[0],q[8];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[7];
rz(3.5*pi) q[3];
rz(3.0*pi) q[5];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
rx(1.897583617650434*pi) q[5];
rx(3.5*pi) q[7];
rx(3.5*pi) q[8];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[8];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[6];
rz(1.0*pi) q[5];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
ry(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[5];
rx(3.5*pi) q[6];
rx(3.5*pi) q[8];
rxx(0.5*pi) q[0],q[9];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[8];
rz(3.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
ry(3.5*pi) q[0];
rz(3.25*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
rz(3.5*pi) q[4];
rx(2.147583617650433*pi) q[5];
rx(3.5*pi) q[8];
rx(3.5*pi) q[9];
rz(3.5*pi) q[0];
rx(3.0*pi) q[1];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[7];
rz(0.39758361765043326*pi) q[4];
rz(3.352416382349567*pi) q[5];
rz(3.0*pi) q[9];
rz(3.25*pi) q[0];
rz(0.5*pi) q[1];
rz(3.75*pi) q[2];
ry(3.5*pi) q[3];
ry(0.5*pi) q[4];
ry(0.5*pi) q[5];
rx(3.5*pi) q[7];
rx(0.75*pi) q[9];
rx(3.124612867792604*pi) q[0];
ry(0.5*pi) q[1];
rx(3.0*pi) q[2];
rz(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[6];
rz(0.8660868077469916*pi) q[0];
rxx(0.5*pi) q[1],q[9];
rz(0.5*pi) q[2];
ry(0.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[6];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
rxx(0.5*pi) q[3],q[8];
rz(3.5*pi) q[4];
rz(3.0*pi) q[6];
rx(3.5*pi) q[9];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[9];
ry(3.5*pi) q[3];
ry(0.5*pi) q[4];
rx(1.897583617650434*pi) q[6];
rx(3.5*pi) q[8];
rz(0.5671068880012611*pi) q[1];
ry(3.5*pi) q[2];
rz(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[7];
rz(1.0*pi) q[6];
rx(3.5*pi) q[9];
rx(3.141188748883179*pi) q[1];
rz(3.5*pi) q[2];
rz(2.25*pi) q[3];
ry(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[6];
rx(3.5*pi) q[7];
rz(3.198393766204247*pi) q[1];
rz(3.0691657374330923*pi) q[2];
ry(0.5*pi) q[3];
rz(3.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
rxx(0.5*pi) q[0],q[1];
rx(3.2372260766814263*pi) q[2];
rxx(0.5*pi) q[3],q[9];
ry(0.5*pi) q[4];
rz(3.5*pi) q[5];
rx(2.147583617650433*pi) q[6];
ry(3.5*pi) q[0];
rx(3.5*pi) q[1];
rz(1.2478865163006405*pi) q[2];
ry(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[8];
rz(1.3975836176504333*pi) q[5];
rz(3.352416382349567*pi) q[6];
rx(3.5*pi) q[9];
rz(3.5*pi) q[0];
rx(2.147583617650433*pi) q[1];
rz(3.5*pi) q[3];
ry(3.5*pi) q[4];
ry(0.5*pi) q[5];
ry(0.5*pi) q[6];
rx(3.5*pi) q[8];
rz(1.0*pi) q[9];
rz(0.39758361765043326*pi) q[0];
rz(3.352416382349567*pi) q[1];
rz(1.2278987134034396*pi) q[3];
rz(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[7];
rx(1.6871670418109983*pi) q[9];
ry(0.5*pi) q[0];
ry(0.5*pi) q[1];
rx(3.7625207256550426*pi) q[3];
rz(3.0628329581890013*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[7];
rz(1.0*pi) q[9];
rxx(0.5*pi) q[0],q[2];
rz(0.019839764139362503*pi) q[3];
ry(0.5*pi) q[4];
rz(3.5*pi) q[5];
rz(3.0*pi) q[7];
ry(3.5*pi) q[0];
rx(3.5*pi) q[2];
rxx(0.5*pi) q[4],q[9];
ry(0.5*pi) q[5];
rx(1.897583617650434*pi) q[7];
rz(3.5*pi) q[0];
rz(3.0*pi) q[2];
ry(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[8];
rz(1.0*pi) q[7];
rx(3.5*pi) q[9];
ry(0.5*pi) q[0];
rx(1.897583617650434*pi) q[2];
rz(3.5*pi) q[4];
ry(3.5*pi) q[5];
rxx(0.5*pi) q[6],q[7];
rx(3.5*pi) q[8];
rz(1.0*pi) q[9];
rxx(0.5*pi) q[0],q[3];
rz(1.0*pi) q[2];
rz(2.078588595901877*pi) q[4];
rz(3.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
rx(1.3078597709100424*pi) q[9];
ry(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[2];
rx(3.5*pi) q[3];
rx(3.7627817999029123*pi) q[4];
rz(0.7549731872789505*pi) q[5];
rz(3.5*pi) q[6];
rx(2.147583617650433*pi) q[7];
rz(3.5*pi) q[0];
ry(3.5*pi) q[1];
rx(3.5*pi) q[2];
rz(0.3310587772052196*pi) q[4];
rx(1.0*pi) q[5];
rz(0.39758361765043326*pi) q[6];
rz(3.352416382349567*pi) q[7];
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
rx(2.147583617650433*pi) q[2];
rz(0.5*pi) q[5];
ry(0.5*pi) q[6];
ry(0.5*pi) q[7];
rxx(0.5*pi) q[0],q[4];
rz(1.3975836176504333*pi) q[1];
rz(3.352416382349567*pi) q[2];
ry(0.5*pi) q[5];
rxx(0.5*pi) q[6],q[8];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
rx(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[9];
ry(3.5*pi) q[6];
rx(3.5*pi) q[8];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[3];
ry(3.5*pi) q[5];
rz(3.5*pi) q[6];
rz(3.0*pi) q[8];
rx(3.5*pi) q[9];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
rx(3.5*pi) q[3];
rz(3.5*pi) q[5];
rz(1.0451672353008665*pi) q[6];
rx(1.897583617650434*pi) q[8];
rx(0.7098059519780837*pi) q[9];
rz(3.5*pi) q[1];
rz(3.0*pi) q[3];
rz(1.1892563396773865*pi) q[5];
ry(0.5*pi) q[6];
rz(1.0*pi) q[8];
rz(1.0*pi) q[9];
ry(0.5*pi) q[1];
rx(1.897583617650434*pi) q[3];
rx(3.3185941138156454*pi) q[5];
rxx(0.5*pi) q[6],q[9];
rxx(0.5*pi) q[7],q[8];
rxx(0.5*pi) q[1],q[4];
rz(1.0*pi) q[3];
rz(3.3170315897842806*pi) q[5];
ry(3.5*pi) q[6];
ry(3.5*pi) q[7];
rx(3.5*pi) q[8];
rx(3.5*pi) q[9];
rxx(0.5*pi) q[0],q[5];
ry(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[3];
rx(3.5*pi) q[4];
rz(3.5*pi) q[6];
rz(3.5*pi) q[7];
rx(2.147583617650433*pi) q[8];
rx(3.8448390071739955*pi) q[9];
ry(3.5*pi) q[0];
rz(3.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[3];
rx(3.5*pi) q[5];
rz(2.1839191017884247*pi) q[6];
rz(1.2875898601252949*pi) q[7];
rz(3.642260704571144*pi) q[8];
rz(1.0*pi) q[9];
rz(3.5*pi) q[0];
ry(0.5*pi) q[1];
rz(3.5*pi) q[2];
rx(2.147583617650433*pi) q[3];
rx(3.790992511931346*pi) q[6];
rx(1.0*pi) q[7];
ry(0.5*pi) q[8];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[5];
rz(0.39758361765043326*pi) q[2];
rz(3.352416382349567*pi) q[3];
rz(0.2476078110216604*pi) q[6];
rz(0.5*pi) q[7];
rxx(0.5*pi) q[0],q[6];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
ry(0.5*pi) q[3];
rx(3.5*pi) q[5];
ry(0.5*pi) q[7];
ry(3.5*pi) q[0];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[4];
rx(3.5*pi) q[6];
rxx(0.5*pi) q[7],q[9];
rz(3.5*pi) q[0];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
rx(3.5*pi) q[4];
ry(3.5*pi) q[7];
rx(3.5*pi) q[9];
ry(0.5*pi) q[0];
rxx(0.5*pi) q[1],q[6];
rz(3.5*pi) q[2];
rz(3.0*pi) q[4];
rz(3.5*pi) q[7];
rx(1.9977455379037172*pi) q[9];
ry(3.5*pi) q[1];
ry(0.5*pi) q[2];
rx(1.897583617650434*pi) q[4];
rx(3.5*pi) q[6];
rz(0.1505972819866409*pi) q[7];
rz(1.0*pi) q[9];
rz(3.5*pi) q[1];
rxx(0.5*pi) q[2],q[5];
rz(1.0*pi) q[4];
rx(3.227457866601376*pi) q[7];
rxx(0.5*pi) q[8],q[9];
ry(0.5*pi) q[1];
ry(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[4];
rx(3.5*pi) q[5];
rz(3.18026482167239*pi) q[7];
ry(3.5*pi) q[8];
rx(3.5*pi) q[9];
rxx(0.5*pi) q[0],q[7];
rz(3.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[4];
rz(3.5*pi) q[8];
rz(0.20929686385938295*pi) q[9];
ry(3.5*pi) q[0];
ry(0.5*pi) q[2];
rz(3.5*pi) q[3];
rx(2.147583617650433*pi) q[4];
rx(3.5*pi) q[7];
rz(1.2689752102127634*pi) q[8];
rx(3.4813831856653*pi) q[9];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[7];
rxx(0.5*pi) q[2],q[6];
rz(1.3975836176504333*pi) q[3];
rz(3.352416382349567*pi) q[4];
rx(3.737026517114741*pi) q[8];
rz(0.19592278462406543*pi) q[9];
ry(0.5*pi) q[0];
ry(3.5*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
ry(0.5*pi) q[4];
rx(3.5*pi) q[6];
rx(3.5*pi) q[7];
rz(0.13743717446322157*pi) q[8];
rxx(0.5*pi) q[0],q[8];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[5];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
ry(0.5*pi) q[2];
ry(3.5*pi) q[3];
rx(3.5*pi) q[5];
rx(3.5*pi) q[8];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[8];
rxx(0.5*pi) q[2],q[7];
rz(3.5*pi) q[3];
rz(3.0*pi) q[5];
rz(2.25*pi) q[0];
ry(3.5*pi) q[1];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
rx(1.897583617650434*pi) q[5];
rx(3.5*pi) q[7];
rx(3.5*pi) q[8];
ry(0.5*pi) q[0];
rz(3.5*pi) q[1];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[6];
rz(1.0*pi) q[5];
rxx(0.5*pi) q[0],q[9];
rz(2.397583617650433*pi) q[1];
ry(0.5*pi) q[2];
ry(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[5];
rx(3.5*pi) q[6];
ry(3.5*pi) q[0];
ry(0.5*pi) q[1];
rxx(0.5*pi) q[2],q[8];
rz(3.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[5];
rx(3.5*pi) q[9];
rz(3.5*pi) q[0];
ry(3.5*pi) q[2];
ry(0.5*pi) q[3];
rz(3.5*pi) q[4];
rx(2.147583617650433*pi) q[5];
rx(3.5*pi) q[8];
rx(0.6475836176504337*pi) q[9];
rz(3.5*pi) q[0];
rxx(0.5*pi) q[1],q[9];
rz(3.5*pi) q[2];
rxx(0.5*pi) q[3],q[7];
rz(0.39758361765043326*pi) q[4];
rz(3.352416382349567*pi) q[5];
rx(0.08223212506713914*pi) q[0];
ry(3.5*pi) q[1];
rz(2.5336361026633027*pi) q[2];
ry(3.5*pi) q[3];
ry(0.5*pi) q[4];
ry(0.5*pi) q[5];
rx(3.5*pi) q[7];
rx(3.5*pi) q[9];
rz(0.5269311780135856*pi) q[0];
rz(3.5*pi) q[1];
ry(0.5*pi) q[2];
rz(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[6];
rz(3.0*pi) q[9];
rz(3.352416382349567*pi) q[1];
ry(0.5*pi) q[3];
ry(3.5*pi) q[4];
rx(3.5*pi) q[6];
rx(1.3639475149871307*pi) q[9];
rx(0.2556417392567748*pi) q[1];
rxx(0.5*pi) q[3],q[8];
rz(3.5*pi) q[4];
rz(3.0*pi) q[6];
rz(1.0*pi) q[9];
rz(0.5005520474139187*pi) q[1];
rxx(0.5*pi) q[2],q[9];
ry(3.5*pi) q[3];
ry(0.5*pi) q[4];
rx(1.897583617650434*pi) q[6];
rx(3.5*pi) q[8];
ry(3.5*pi) q[2];
rz(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[7];
rz(1.0*pi) q[6];
rx(3.5*pi) q[9];
rz(3.5*pi) q[2];
rz(1.1024163823495667*pi) q[3];
ry(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[6];
rx(3.5*pi) q[7];
rx(1.068780279686264*pi) q[9];
rz(3.2163638973366973*pi) q[2];
ry(0.5*pi) q[3];
rz(3.5*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[6];
rx(0.1412618976852069*pi) q[2];
rxx(0.5*pi) q[3],q[9];
ry(0.5*pi) q[4];
rz(3.5*pi) q[5];
rx(2.147583617650433*pi) q[6];
rz(0.7332248832969387*pi) q[2];
ry(3.5*pi) q[3];
rxx(0.5*pi) q[4],q[8];
rz(1.897583617650433*pi) q[5];
rz(3.352416382349567*pi) q[6];
rx(3.5*pi) q[9];
rz(3.5*pi) q[3];
ry(3.5*pi) q[4];
ry(0.5*pi) q[5];
ry(0.5*pi) q[6];
rx(3.5*pi) q[8];
rz(3.0*pi) q[9];
rz(0.6475836176504335*pi) q[3];
rz(3.5*pi) q[4];
rxx(0.5*pi) q[5],q[7];
rx(2.352416382349567*pi) q[9];
rx(0.18385148671009655*pi) q[3];
rz(0.75*pi) q[4];
ry(3.5*pi) q[5];
rx(3.5*pi) q[7];
rz(0.7314742903313882*pi) q[3];
rx(3.0*pi) q[4];
rz(3.5*pi) q[5];
rz(3.0*pi) q[7];
rz(0.5*pi) q[4];
ry(0.5*pi) q[5];
rx(1.897583617650434*pi) q[7];
ry(0.5*pi) q[4];
rxx(0.5*pi) q[5],q[8];
rz(1.0*pi) q[7];
rxx(0.5*pi) q[4],q[9];
ry(3.5*pi) q[5];
rxx(0.5*pi) q[6],q[7];
rx(3.5*pi) q[8];
ry(3.5*pi) q[4];
rz(3.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[7];
rx(3.5*pi) q[9];
rz(3.5*pi) q[4];
ry(0.5*pi) q[5];
rz(3.5*pi) q[6];
rx(2.147583617650433*pi) q[7];
rx(1.25*pi) q[9];
rz(3.5*pi) q[4];
rz(0.39758361765043326*pi) q[6];
rz(3.352416382349567*pi) q[7];
rz(1.0*pi) q[9];
rx(3.032916855048612*pi) q[4];
rxx(0.5*pi) q[5],q[9];
ry(0.5*pi) q[6];
ry(0.5*pi) q[7];
rz(2.6464573325489043*pi) q[4];
ry(3.5*pi) q[5];
rxx(0.5*pi) q[6],q[8];
rx(3.5*pi) q[9];
rz(3.5*pi) q[5];
ry(3.5*pi) q[6];
rx(3.5*pi) q[8];
rx(1.0046888083207686*pi) q[9];
rz(3.25*pi) q[5];
rz(3.5*pi) q[6];
rz(3.0*pi) q[8];
rx(0.2799846756055128*pi) q[5];
rz(3.0046888083207683*pi) q[6];
rx(1.897583617650434*pi) q[8];
rz(0.7340123278426769*pi) q[5];
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
rx(3.2453111916792317*pi) q[9];
rz(0.7453111916792317*pi) q[6];
rz(1.1475836176504333*pi) q[7];
rz(3.878637400696867*pi) q[8];
rx(0.30011203041319157*pi) q[6];
ry(0.5*pi) q[7];
ry(0.5*pi) q[8];
rz(0.6417797655883362*pi) q[6];
rxx(0.5*pi) q[7],q[9];
ry(3.5*pi) q[7];
rx(3.5*pi) q[9];
rz(3.5*pi) q[7];
rx(3.1213625993031333*pi) q[9];
rx(0.2932643762542674*pi) q[7];
rz(1.0*pi) q[9];
rz(0.5131225583790733*pi) q[7];
rxx(0.5*pi) q[8],q[9];
ry(3.5*pi) q[8];
rx(3.5*pi) q[9];
rz(3.5*pi) q[8];
rz(3.217998790486118*pi) q[9];
rz(2.621362599303133*pi) q[8];
rx(0.4041719318564173*pi) q[9];
rx(0.7613560946091095*pi) q[8];
rz(1.0819238013559906*pi) q[9];
rz(3.525575452281565*pi) q[8];
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
