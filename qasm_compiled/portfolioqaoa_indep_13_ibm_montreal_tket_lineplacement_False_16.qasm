OPENQASM 2.0;
include "qelib1.inc";

qreg node[17];
creg meas[13];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[8];
sx node[9];
sx node[11];
sx node[12];
sx node[13];
sx node[14];
sx node[16];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[16];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[8];
sx node[9];
sx node[11];
sx node[12];
sx node[13];
sx node[14];
sx node[16];
rz(3.06796010085707*pi) node[1];
cx node[1],node[2];
rz(0.44842392670966547*pi) node[2];
cx node[1],node[2];
cx node[1],node[0];
rz(3.0430983162107537*pi) node[2];
rz(0.44851432671733704*pi) node[0];
cx node[1],node[0];
cx node[1],node[4];
rz(0.449502360604054*pi) node[4];
cx node[1],node[4];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[1],node[0];
cx node[2],node[3];
rz(0.4485515689740218*pi) node[0];
rz(0.4482787774015655*pi) node[3];
cx node[1],node[0];
cx node[2],node[3];
rz(3.0614794707293114*pi) node[0];
cx node[1],node[4];
cx node[3],node[2];
cx node[2],node[3];
rz(0.4486632957440726*pi) node[4];
cx node[1],node[4];
cx node[3],node[2];
cx node[1],node[2];
cx node[3],node[5];
rz(0.4484850422078104*pi) node[2];
rz(0.44896314365685797*pi) node[5];
cx node[1],node[2];
cx node[3],node[5];
cx node[1],node[2];
cx node[5],node[3];
cx node[2],node[1];
cx node[3],node[5];
cx node[1],node[2];
cx node[5],node[3];
cx node[0],node[1];
cx node[2],node[3];
cx node[5],node[8];
cx node[1],node[0];
rz(0.44848154079906166*pi) node[3];
rz(0.4481326731638049*pi) node[8];
cx node[0],node[1];
cx node[2],node[3];
cx node[5],node[8];
cx node[1],node[4];
cx node[2],node[3];
cx node[8],node[5];
cx node[3],node[2];
rz(0.4485210112249476*pi) node[4];
cx node[5],node[8];
cx node[1],node[4];
cx node[2],node[3];
cx node[8],node[5];
cx node[1],node[0];
cx node[3],node[5];
rz(3.0635644641457924*pi) node[4];
cx node[8],node[11];
rz(0.44777489285173644*pi) node[0];
rz(0.44852419432381296*pi) node[5];
rz(0.448532152070964*pi) node[11];
cx node[1],node[0];
cx node[3],node[5];
cx node[8],node[11];
cx node[1],node[2];
cx node[3],node[5];
cx node[8],node[9];
rz(0.4484767661507689*pi) node[2];
cx node[5],node[3];
rz(0.4434347375536163*pi) node[9];
cx node[1],node[2];
cx node[3],node[5];
cx node[8],node[9];
cx node[1],node[2];
cx node[11],node[8];
cx node[2],node[1];
cx node[8],node[11];
cx node[1],node[2];
cx node[11],node[8];
cx node[4],node[1];
cx node[2],node[3];
cx node[5],node[8];
cx node[11],node[14];
cx node[1],node[4];
rz(0.4485499774245909*pi) node[3];
rz(0.4484939548846256*pi) node[8];
rz(0.4477535660893608*pi) node[14];
cx node[4],node[1];
cx node[2],node[3];
cx node[5],node[8];
cx node[11],node[14];
cx node[1],node[0];
cx node[2],node[3];
cx node[5],node[8];
cx node[14],node[11];
rz(0.44715227871435914*pi) node[0];
cx node[3],node[2];
cx node[8],node[5];
cx node[11],node[14];
cx node[1],node[0];
cx node[2],node[3];
cx node[5],node[8];
cx node[14],node[11];
rz(0.895013042759407*pi) node[0];
cx node[1],node[4];
cx node[3],node[5];
cx node[8],node[9];
cx node[14],node[13];
rz(0.44851114361847877*pi) node[4];
rz(0.448468490093731*pi) node[5];
rz(0.44949599440633037*pi) node[9];
rz(0.44840387318683383*pi) node[13];
cx node[1],node[4];
cx node[3],node[5];
cx node[8],node[9];
cx node[14],node[13];
cx node[1],node[2];
cx node[3],node[5];
cx node[8],node[11];
cx node[14],node[16];
rz(0.4484570309378242*pi) node[2];
cx node[5],node[3];
rz(0.44842647318874995*pi) node[11];
rz(0.44812980837482996*pi) node[16];
cx node[1],node[2];
cx node[3],node[5];
cx node[8],node[11];
cx node[14],node[16];
cx node[1],node[2];
cx node[8],node[11];
cx node[13],node[14];
cx node[2],node[1];
cx node[11],node[8];
cx node[14],node[13];
cx node[1],node[2];
cx node[8],node[11];
cx node[13],node[14];
cx node[0],node[1];
cx node[2],node[3];
cx node[5],node[8];
cx node[11],node[14];
cx node[13],node[12];
cx node[1],node[0];
rz(0.44832620557460245*pi) node[3];
cx node[8],node[5];
rz(0.4477717097528746*pi) node[12];
rz(0.4484789943199736*pi) node[14];
cx node[0],node[1];
cx node[2],node[3];
cx node[5],node[8];
cx node[11],node[14];
cx node[13],node[12];
cx node[1],node[4];
cx node[2],node[3];
cx node[8],node[9];
cx node[11],node[14];
sx node[13];
cx node[3],node[2];
rz(0.44861172954251316*pi) node[4];
rz(0.44902935211318606*pi) node[9];
cx node[14],node[11];
rz(3.536761409240354*pi) node[13];
cx node[1],node[4];
cx node[2],node[3];
cx node[8],node[9];
cx node[11],node[14];
sx node[13];
cx node[1],node[0];
rz(3.0459745325113214*pi) node[4];
cx node[8],node[5];
rz(1.9909252949257068*pi) node[13];
cx node[14],node[16];
rz(0.44853660840937337*pi) node[0];
rz(0.44856493798924646*pi) node[5];
cx node[12],node[13];
rz(0.4485452027763017*pi) node[16];
cx node[1],node[0];
cx node[8],node[5];
cx node[13],node[12];
cx node[14],node[16];
cx node[1],node[2];
cx node[3],node[5];
cx node[8],node[11];
cx node[12],node[13];
rz(0.4492830450924714*pi) node[2];
cx node[5],node[3];
rz(0.4484914084055376*pi) node[11];
cx node[14],node[13];
cx node[1],node[2];
cx node[3],node[5];
cx node[8],node[11];
rz(0.44838318304423197*pi) node[13];
cx node[1],node[2];
cx node[8],node[11];
cx node[14],node[13];
cx node[2],node[1];
cx node[11],node[8];
sx node[14];
cx node[1],node[2];
cx node[8],node[11];
rz(3.536761409240354*pi) node[14];
cx node[4],node[1];
cx node[9],node[8];
sx node[14];
cx node[1],node[4];
cx node[8],node[9];
rz(1.5*pi) node[14];
cx node[4],node[1];
cx node[9],node[8];
cx node[14],node[13];
cx node[1],node[0];
cx node[5],node[8];
cx node[13],node[14];
rz(0.44856016334095017*pi) node[0];
rz(0.4500829578364538*pi) node[8];
cx node[14],node[13];
cx node[1],node[0];
cx node[5],node[8];
cx node[11],node[14];
cx node[12],node[13];
rz(3.0606963010853447*pi) node[0];
cx node[1],node[4];
cx node[5],node[3];
cx node[14],node[11];
rz(0.9990217712822833*pi) node[13];
rz(0.4483516703655006*pi) node[3];
rz(0.4485502957344778*pi) node[4];
cx node[11],node[14];
cx node[12],node[13];
cx node[1],node[4];
cx node[5],node[3];
rz(0.5061799142492007*pi) node[13];
cx node[14],node[16];
cx node[2],node[3];
cx node[8],node[5];
rz(0.448570030947419*pi) node[16];
cx node[3],node[2];
cx node[5],node[8];
cx node[14],node[16];
cx node[2],node[3];
cx node[8],node[5];
cx node[14],node[11];
cx node[3],node[5];
cx node[8],node[9];
rz(0.44831092670007067*pi) node[11];
rz(0.45359327926128756*pi) node[5];
rz(0.4485735323561677*pi) node[9];
cx node[14],node[11];
cx node[3],node[5];
cx node[8],node[9];
sx node[14];
cx node[3],node[2];
cx node[8],node[11];
rz(3.536761409240354*pi) node[14];
rz(0.44797574638991833*pi) node[2];
cx node[11],node[8];
sx node[14];
cx node[3],node[2];
cx node[8],node[11];
rz(1.5*pi) node[14];
cx node[3],node[5];
cx node[14],node[13];
cx node[5],node[3];
cx node[13],node[14];
cx node[3],node[5];
cx node[14],node[13];
cx node[3],node[2];
cx node[5],node[8];
cx node[12],node[13];
cx node[2],node[3];
cx node[8],node[5];
rz(0.99896638536209*pi) node[13];
cx node[3],node[2];
cx node[5],node[8];
cx node[12],node[13];
cx node[1],node[2];
cx node[8],node[9];
cx node[14],node[13];
rz(0.4480346337188621*pi) node[2];
rz(0.4479830675172991*pi) node[9];
rz(0.9989434670502835*pi) node[13];
cx node[1],node[2];
cx node[8],node[9];
cx node[14],node[13];
cx node[1],node[2];
cx node[9],node[8];
rz(0.494901685685891*pi) node[13];
cx node[16],node[14];
cx node[2],node[1];
cx node[8],node[9];
cx node[12],node[13];
cx node[14],node[16];
cx node[1],node[2];
cx node[9],node[8];
cx node[13],node[12];
cx node[16],node[14];
cx node[0],node[1];
cx node[2],node[3];
cx node[11],node[14];
cx node[12],node[13];
cx node[1],node[0];
rz(0.44845798586748487*pi) node[3];
rz(0.44852196615460826*pi) node[14];
cx node[0],node[1];
cx node[2],node[3];
cx node[11],node[14];
cx node[1],node[4];
cx node[2],node[3];
cx node[3],node[2];
rz(0.44853692671926026*pi) node[4];
cx node[1],node[4];
cx node[2],node[3];
cx node[1],node[0];
cx node[3],node[5];
rz(3.052304888541813*pi) node[4];
rz(0.44813299147369534*pi) node[0];
cx node[5],node[3];
cx node[1],node[0];
cx node[3],node[5];
cx node[1],node[2];
cx node[5],node[8];
rz(0.4483593098027683*pi) node[2];
rz(0.4484831323484926*pi) node[8];
cx node[1],node[2];
cx node[5],node[8];
cx node[1],node[2];
cx node[8],node[5];
cx node[2],node[1];
cx node[5],node[8];
cx node[1],node[2];
cx node[8],node[5];
cx node[4],node[1];
cx node[5],node[3];
cx node[9],node[8];
cx node[1],node[4];
cx node[3],node[5];
cx node[8],node[9];
cx node[4],node[1];
cx node[5],node[3];
cx node[9],node[8];
cx node[1],node[0];
cx node[2],node[3];
cx node[11],node[8];
rz(0.44703163926749667*pi) node[0];
rz(0.448526740802901*pi) node[3];
cx node[8],node[11];
cx node[1],node[0];
cx node[2],node[3];
cx node[11],node[8];
rz(0.7637829803485747*pi) node[0];
cx node[1],node[4];
cx node[3],node[2];
cx node[8],node[5];
cx node[11],node[14];
cx node[2],node[3];
rz(0.4485878563010459*pi) node[4];
rz(0.44892112675188045*pi) node[5];
rz(0.44862096052921174*pi) node[14];
cx node[1],node[4];
cx node[3],node[2];
cx node[8],node[5];
cx node[11],node[14];
cx node[1],node[2];
sx node[8];
rz(0.44846944502338815*pi) node[2];
rz(3.536761409240354*pi) node[8];
cx node[1],node[2];
sx node[8];
cx node[0],node[1];
rz(1.5*pi) node[8];
cx node[1],node[0];
cx node[11],node[8];
cx node[0],node[1];
cx node[8],node[11];
cx node[1],node[4];
cx node[11],node[8];
rz(0.4507685973312938*pi) node[4];
cx node[8],node[5];
cx node[14],node[11];
cx node[1],node[4];
rz(0.44351940798334155*pi) node[5];
cx node[11],node[14];
cx node[1],node[2];
rz(3.053288879892973*pi) node[4];
cx node[8],node[5];
cx node[14],node[11];
rz(0.4500208874086482*pi) node[2];
sx node[8];
cx node[13],node[14];
cx node[1],node[2];
rz(3.536761409240354*pi) node[8];
rz(0.9983603233387939*pi) node[14];
cx node[0],node[1];
sx node[8];
cx node[13],node[14];
cx node[1],node[0];
rz(1.5*pi) node[8];
cx node[16],node[14];
cx node[0],node[1];
cx node[11],node[8];
rz(0.9988750304247542*pi) node[14];
cx node[1],node[2];
cx node[8],node[11];
cx node[16],node[14];
cx node[2],node[1];
cx node[11],node[8];
cx node[13],node[14];
cx node[1],node[2];
cx node[9],node[8];
cx node[14],node[13];
cx node[4],node[1];
rz(0.44847867601008673*pi) node[8];
cx node[13],node[14];
rz(0.44865438306726446*pi) node[1];
cx node[9],node[8];
cx node[14],node[11];
cx node[12],node[13];
cx node[4],node[1];
cx node[8],node[5];
rz(0.9991108980504162*pi) node[11];
rz(0.9989622473335693*pi) node[13];
rz(3.060411254582267*pi) node[1];
cx node[5],node[8];
cx node[14],node[11];
cx node[12],node[13];
cx node[0],node[1];
cx node[8],node[5];
cx node[14],node[11];
rz(0.4936223664223297*pi) node[13];
cx node[1],node[0];
cx node[3],node[5];
cx node[9],node[8];
cx node[11],node[14];
cx node[12],node[13];
cx node[0],node[1];
rz(0.4485292872819926*pi) node[5];
rz(0.4484605323465729*pi) node[8];
cx node[14],node[11];
cx node[13],node[12];
cx node[3],node[5];
cx node[9],node[8];
cx node[12],node[13];
cx node[16],node[14];
cx node[5],node[3];
sx node[9];
rz(0.998984210715717*pi) node[14];
cx node[3],node[5];
rz(3.536761409240354*pi) node[9];
cx node[16],node[14];
cx node[5],node[3];
sx node[9];
cx node[13],node[14];
cx node[2],node[3];
cx node[5],node[8];
rz(1.5*pi) node[9];
rz(0.9994199769499001*pi) node[14];
rz(0.44857194080674034*pi) node[3];
rz(0.44854392953675415*pi) node[8];
cx node[13],node[14];
cx node[2],node[3];
cx node[5],node[8];
cx node[14],node[13];
cx node[3],node[2];
sx node[5];
cx node[13],node[14];
cx node[2],node[3];
rz(3.536761409240354*pi) node[5];
cx node[14],node[13];
cx node[3],node[2];
sx node[5];
cx node[12],node[13];
cx node[16],node[14];
cx node[1],node[2];
rz(1.5*pi) node[5];
rz(0.999801948813321*pi) node[13];
cx node[14],node[16];
rz(0.4483176112076812*pi) node[2];
cx node[8],node[5];
cx node[12],node[13];
cx node[16],node[14];
cx node[1],node[2];
cx node[5],node[8];
rz(0.597041534922341*pi) node[13];
cx node[1],node[2];
cx node[8],node[5];
cx node[12],node[13];
cx node[2],node[1];
cx node[3],node[5];
cx node[9],node[8];
cx node[13],node[12];
cx node[1],node[2];
rz(0.4484404788237448*pi) node[5];
cx node[8],node[9];
cx node[12],node[13];
cx node[4],node[1];
cx node[3],node[5];
cx node[9],node[8];
rz(0.4486282816565925*pi) node[1];
sx node[3];
cx node[11],node[8];
cx node[4],node[1];
rz(3.536761409240354*pi) node[3];
rz(0.9986910473105386*pi) node[8];
cx node[0],node[1];
sx node[3];
cx node[11],node[8];
rz(0.448488861926446*pi) node[1];
rz(1.5*pi) node[3];
cx node[8],node[11];
cx node[0],node[1];
cx node[5],node[3];
cx node[11],node[8];
rz(3.051095820270133*pi) node[1];
cx node[3],node[5];
cx node[8],node[11];
cx node[4],node[1];
cx node[5],node[3];
cx node[8],node[9];
cx node[14],node[11];
cx node[1],node[4];
cx node[2],node[3];
rz(0.9992006614383193*pi) node[9];
rz(0.9989864388849181*pi) node[11];
cx node[4],node[1];
rz(0.45135715231084816*pi) node[3];
cx node[8],node[9];
cx node[14],node[11];
cx node[2],node[3];
cx node[8],node[5];
cx node[14],node[11];
sx node[2];
rz(0.9989555628259588*pi) node[5];
cx node[11],node[14];
rz(3.536761409240354*pi) node[2];
cx node[8],node[5];
cx node[14],node[11];
sx node[2];
cx node[8],node[5];
cx node[16],node[14];
rz(1.5*pi) node[2];
cx node[5],node[8];
rz(0.998989303673893*pi) node[14];
cx node[3],node[2];
cx node[8],node[5];
cx node[16],node[14];
cx node[2],node[3];
cx node[9],node[8];
cx node[13],node[14];
cx node[3],node[2];
cx node[8],node[9];
rz(0.9989682952214061*pi) node[14];
cx node[1],node[2];
cx node[5],node[3];
cx node[9],node[8];
cx node[13],node[14];
rz(0.4490347633812526*pi) node[2];
rz(1.0020832757675997*pi) node[3];
cx node[11],node[8];
cx node[14],node[13];
cx node[1],node[2];
cx node[5],node[3];
rz(0.9989603374742515*pi) node[8];
cx node[13],node[14];
sx node[1];
cx node[5],node[3];
cx node[11],node[8];
cx node[14],node[13];
rz(3.536761409240354*pi) node[1];
cx node[3],node[5];
cx node[8],node[11];
cx node[12],node[13];
cx node[16],node[14];
sx node[1];
cx node[5],node[3];
cx node[11],node[8];
rz(0.9989065431034874*pi) node[13];
cx node[14],node[16];
rz(1.5*pi) node[1];
cx node[8],node[11];
cx node[12],node[13];
cx node[16],node[14];
cx node[2],node[1];
cx node[8],node[9];
cx node[14],node[11];
rz(0.5044151405782205*pi) node[13];
cx node[1],node[2];
rz(0.9989787994476504*pi) node[9];
rz(0.9989447402898293*pi) node[11];
cx node[12],node[13];
cx node[2],node[1];
cx node[8],node[9];
cx node[14],node[11];
cx node[13],node[12];
cx node[0],node[1];
cx node[3],node[2];
cx node[8],node[5];
cx node[14],node[11];
cx node[12],node[13];
rz(0.4488011239247882*pi) node[1];
rz(0.9994333459651212*pi) node[2];
rz(0.9983641430574277*pi) node[5];
cx node[11],node[14];
cx node[0],node[1];
cx node[3],node[2];
cx node[8],node[5];
cx node[14],node[11];
sx node[0];
cx node[4],node[1];
cx node[3],node[2];
cx node[8],node[5];
cx node[16],node[14];
rz(3.536761409240354*pi) node[0];
rz(0.44868239433724355*pi) node[1];
cx node[2],node[3];
cx node[5],node[8];
rz(0.9990013994495701*pi) node[14];
sx node[0];
cx node[4],node[1];
cx node[3],node[2];
cx node[8],node[5];
cx node[16],node[14];
rz(1.5*pi) node[0];
rz(3.049439590270341*pi) node[1];
cx node[5],node[3];
sx node[4];
cx node[9],node[8];
cx node[13],node[14];
sx node[1];
rz(0.9990201797328542*pi) node[3];
rz(3.536761409240354*pi) node[4];
cx node[8],node[9];
rz(0.9989526980369821*pi) node[14];
rz(3.536761409240354*pi) node[1];
cx node[5],node[3];
sx node[4];
cx node[9],node[8];
cx node[13],node[14];
sx node[1];
cx node[5],node[3];
rz(1.5*pi) node[4];
cx node[11],node[8];
cx node[14],node[13];
rz(1.5*pi) node[1];
cx node[3],node[5];
rz(0.9989943966320727*pi) node[8];
cx node[13],node[14];
cx node[0],node[1];
cx node[5],node[3];
cx node[11],node[8];
cx node[14],node[13];
cx node[1],node[0];
cx node[8],node[11];
cx node[12],node[13];
cx node[16],node[14];
cx node[0],node[1];
cx node[11],node[8];
rz(0.9989383740921038*pi) node[13];
cx node[14],node[16];
cx node[2],node[1];
cx node[8],node[11];
cx node[12],node[13];
cx node[16],node[14];
rz(0.9990341853678455*pi) node[1];
cx node[8],node[9];
cx node[14],node[11];
rz(0.4953822062900741*pi) node[13];
cx node[2],node[1];
rz(0.9986503036451087*pi) node[9];
rz(0.999081613540886*pi) node[11];
cx node[12],node[13];
cx node[1],node[2];
cx node[8],node[9];
cx node[14],node[11];
cx node[13],node[12];
cx node[2],node[1];
cx node[8],node[5];
cx node[14],node[11];
cx node[12],node[13];
cx node[1],node[2];
rz(0.9989351909932438*pi) node[5];
cx node[11],node[14];
cx node[1],node[4];
cx node[3],node[2];
cx node[8],node[5];
cx node[14],node[11];
rz(0.998988030434349*pi) node[2];
rz(0.999202252987752*pi) node[4];
cx node[8],node[5];
cx node[16],node[14];
cx node[1],node[4];
cx node[3],node[2];
cx node[5],node[8];
rz(0.9984946501107643*pi) node[14];
cx node[1],node[0];
cx node[3],node[2];
cx node[8],node[5];
cx node[16],node[14];
rz(0.999421886809218*pi) node[0];
cx node[2],node[3];
cx node[9],node[8];
cx node[13],node[14];
cx node[1],node[0];
cx node[3],node[2];
cx node[8],node[9];
rz(0.9989441036700573*pi) node[14];
sx node[1];
cx node[5],node[3];
cx node[9],node[8];
cx node[13],node[14];
rz(3.4315607876313265*pi) node[1];
rz(0.9989803909970796*pi) node[3];
cx node[11],node[8];
cx node[14],node[13];
sx node[1];
cx node[5],node[3];
rz(0.9980038162662694*pi) node[8];
cx node[13],node[14];
rz(1.1210545633153104*pi) node[1];
cx node[5],node[3];
cx node[11],node[8];
cx node[14],node[13];
cx node[4],node[1];
cx node[3],node[5];
cx node[8],node[11];
cx node[12],node[13];
cx node[16],node[14];
cx node[1],node[4];
cx node[5],node[3];
cx node[11],node[8];
rz(0.9989523797270969*pi) node[13];
cx node[14],node[16];
cx node[4],node[1];
cx node[8],node[11];
cx node[12],node[13];
cx node[16],node[14];
cx node[2],node[1];
cx node[8],node[9];
cx node[14],node[11];
rz(0.5005309641920628*pi) node[13];
rz(0.9989476050788042*pi) node[1];
rz(0.9990663346663506*pi) node[9];
rz(0.9958501315763488*pi) node[11];
cx node[12],node[13];
cx node[2],node[1];
cx node[8],node[9];
cx node[14],node[11];
cx node[13],node[12];
cx node[2],node[1];
cx node[8],node[5];
cx node[14],node[11];
cx node[12],node[13];
cx node[1],node[2];
rz(0.9989300980350642*pi) node[5];
cx node[11],node[14];
cx node[2],node[1];
cx node[8],node[5];
cx node[14],node[11];
cx node[1],node[0];
cx node[3],node[2];
cx node[8],node[5];
cx node[16],node[14];
rz(0.9990469177632946*pi) node[0];
rz(0.9989320078943802*pi) node[2];
cx node[5],node[8];
rz(0.9992608220068089*pi) node[14];
cx node[1],node[0];
cx node[3],node[2];
cx node[8],node[5];
cx node[16],node[14];
sx node[1];
cx node[3],node[2];
cx node[9],node[8];
cx node[13],node[14];
rz(3.4315607876313265*pi) node[1];
cx node[2],node[3];
cx node[8],node[9];
rz(0.9992003431284342*pi) node[14];
sx node[1];
cx node[3],node[2];
cx node[9],node[8];
cx node[13],node[14];
rz(1.5*pi) node[1];
cx node[5],node[3];
cx node[11],node[8];
cx node[14],node[13];
cx node[4],node[1];
rz(0.9989616107137955*pi) node[3];
rz(0.9992967910239461*pi) node[8];
cx node[13],node[14];
rz(2.8808760708229473*pi) node[1];
cx node[5],node[3];
cx node[11],node[8];
cx node[14],node[13];
cx node[4],node[1];
cx node[5],node[3];
cx node[8],node[11];
cx node[12],node[13];
cx node[16],node[14];
rz(3.6181249664468282*pi) node[1];
cx node[3],node[5];
cx node[11],node[8];
rz(0.9998761150168018*pi) node[13];
cx node[14],node[16];
cx node[0],node[1];
cx node[5],node[3];
cx node[8],node[11];
cx node[12],node[13];
cx node[16],node[14];
cx node[1],node[0];
cx node[8],node[9];
cx node[14],node[11];
rz(0.6775612355623812*pi) node[13];
cx node[0],node[1];
rz(0.999292334685542*pi) node[9];
rz(0.9990010811396814*pi) node[11];
cx node[12],node[13];
cx node[2],node[1];
cx node[8],node[9];
cx node[14],node[11];
cx node[13],node[12];
rz(0.9990911628374732*pi) node[1];
cx node[8],node[5];
cx node[14],node[11];
cx node[12],node[13];
cx node[2],node[1];
rz(0.998900813525534*pi) node[5];
cx node[11],node[14];
sx node[2];
cx node[8],node[5];
cx node[14],node[11];
rz(3.4315607876313265*pi) node[2];
cx node[8],node[5];
cx node[16],node[14];
sx node[2];
cx node[5],node[8];
rz(0.9990615600180579*pi) node[14];
rz(1.5*pi) node[2];
cx node[8],node[5];
cx node[16],node[14];
cx node[1],node[2];
cx node[9],node[8];
cx node[13],node[14];
cx node[2],node[1];
cx node[8],node[9];
rz(0.9989211853582507*pi) node[14];
cx node[1],node[2];
cx node[9],node[8];
cx node[13],node[14];
cx node[4],node[1];
cx node[3],node[2];
cx node[11],node[8];
cx node[14],node[13];
rz(2.8808867023731457*pi) node[1];
rz(0.9987168304113201*pi) node[2];
rz(0.9989854839552592*pi) node[8];
cx node[13],node[14];
cx node[4],node[1];
cx node[3],node[2];
cx node[11],node[8];
cx node[14],node[13];
cx node[0],node[1];
sx node[3];
cx node[8],node[11];
cx node[12],node[13];
cx node[16],node[14];
rz(2.8808910950495754*pi) node[1];
rz(3.4315607876313265*pi) node[3];
cx node[11],node[8];
rz(0.9975830105967329*pi) node[13];
cx node[14],node[16];
cx node[0],node[1];
sx node[3];
cx node[8],node[11];
cx node[12],node[13];
cx node[16],node[14];
rz(3.620290906067366*pi) node[1];
rz(1.5*pi) node[3];
cx node[8],node[9];
cx node[14],node[11];
rz(0.49992722583093796*pi) node[13];
cx node[4],node[1];
cx node[3],node[2];
rz(0.9989883487442359*pi) node[9];
rz(0.9989587459248206*pi) node[11];
cx node[12],node[13];
cx node[1],node[4];
cx node[2],node[3];
cx node[8],node[9];
cx node[14],node[11];
cx node[13],node[12];
cx node[4],node[1];
cx node[3],node[2];
cx node[14],node[11];
cx node[12],node[13];
cx node[1],node[2];
cx node[5],node[3];
cx node[11],node[14];
rz(2.881003140129512*pi) node[2];
rz(1.0020310729462665*pi) node[3];
cx node[14],node[11];
cx node[1],node[2];
cx node[5],node[3];
cx node[16],node[14];
cx node[2],node[1];
sx node[5];
rz(0.9989940783221876*pi) node[14];
cx node[1],node[2];
rz(3.4315607876313265*pi) node[5];
cx node[16],node[14];
cx node[2],node[1];
sx node[5];
cx node[13],node[14];
cx node[0],node[1];
rz(1.5*pi) node[5];
rz(0.9980420134526113*pi) node[14];
rz(2.880904273078863*pi) node[1];
cx node[5],node[3];
cx node[13],node[14];
cx node[0],node[1];
cx node[3],node[5];
cx node[14],node[13];
cx node[4],node[1];
cx node[5],node[3];
cx node[13],node[14];
rz(2.8808874981478616*pi) node[1];
cx node[2],node[3];
cx node[8],node[5];
cx node[14],node[13];
cx node[4],node[1];
rz(2.8808589457510707*pi) node[3];
rz(0.9989994895902523*pi) node[5];
cx node[12],node[13];
cx node[16],node[14];
rz(3.6205365776375222*pi) node[1];
cx node[2],node[3];
cx node[8],node[5];
rz(0.9988804416928208*pi) node[13];
cx node[14],node[16];
cx node[0],node[1];
cx node[3],node[2];
sx node[8];
cx node[12],node[13];
cx node[16],node[14];
cx node[1],node[0];
cx node[2],node[3];
rz(3.4315607876313265*pi) node[8];
rz(0.49555708574154345*pi) node[13];
cx node[0],node[1];
cx node[3],node[2];
sx node[8];
cx node[12],node[13];
cx node[1],node[2];
rz(1.5*pi) node[8];
cx node[13],node[12];
rz(2.880883264626375*pi) node[2];
cx node[8],node[5];
cx node[12],node[13];
cx node[1],node[2];
cx node[5],node[8];
cx node[1],node[2];
cx node[8],node[5];
cx node[2],node[1];
cx node[3],node[5];
cx node[9],node[8];
cx node[1],node[2];
rz(2.8809396054762297*pi) node[5];
cx node[8],node[9];
cx node[4],node[1];
cx node[3],node[5];
cx node[9],node[8];
rz(2.8807995809572975*pi) node[1];
cx node[5],node[3];
cx node[11],node[8];
cx node[4],node[1];
cx node[3],node[5];
rz(0.9989571543753879*pi) node[8];
cx node[0],node[1];
cx node[5],node[3];
cx node[11],node[8];
rz(2.880726210528532*pi) node[1];
cx node[2],node[3];
cx node[8],node[11];
cx node[0],node[1];
rz(2.8808828508235234*pi) node[3];
cx node[11],node[8];
rz(3.600675377641176*pi) node[1];
cx node[2],node[3];
cx node[8],node[11];
cx node[4],node[1];
cx node[3],node[2];
cx node[8],node[9];
cx node[14],node[11];
cx node[1],node[4];
cx node[2],node[3];
rz(0.9989482416985762*pi) node[9];
rz(0.9989310529647213*pi) node[11];
cx node[4],node[1];
cx node[3],node[2];
cx node[8],node[9];
cx node[14],node[11];
cx node[1],node[2];
sx node[8];
cx node[14],node[11];
rz(2.8808822778657284*pi) node[2];
rz(3.4315607876313265*pi) node[8];
cx node[11],node[14];
cx node[1],node[2];
sx node[8];
cx node[14],node[11];
cx node[2],node[1];
rz(1.5*pi) node[8];
cx node[16],node[14];
cx node[1],node[2];
cx node[5],node[8];
rz(0.9990870248089525*pi) node[14];
cx node[2],node[1];
rz(2.880841725186228*pi) node[8];
cx node[16],node[14];
cx node[0],node[1];
cx node[5],node[8];
cx node[13],node[14];
rz(2.8808863204012822*pi) node[1];
cx node[8],node[5];
rz(0.9988963571871281*pi) node[14];
cx node[0],node[1];
cx node[5],node[8];
cx node[13],node[14];
cx node[4],node[1];
cx node[8],node[5];
cx node[14],node[13];
rz(2.8808981933600375*pi) node[1];
cx node[3],node[5];
cx node[9],node[8];
cx node[13],node[14];
cx node[4],node[1];
rz(2.880887880119725*pi) node[5];
cx node[8],node[9];
cx node[14],node[13];
rz(3.618463870982648*pi) node[1];
cx node[3],node[5];
cx node[9],node[8];
cx node[12],node[13];
cx node[16],node[14];
cx node[0],node[1];
cx node[5],node[3];
cx node[11],node[8];
rz(0.9989819825465123*pi) node[13];
cx node[14],node[16];
cx node[1],node[0];
cx node[3],node[5];
rz(0.9990119036758127*pi) node[8];
cx node[12],node[13];
cx node[16],node[14];
cx node[0],node[1];
cx node[5],node[3];
cx node[11],node[8];
rz(0.5012728490437914*pi) node[13];
cx node[2],node[3];
sx node[11];
cx node[12],node[13];
rz(2.8808909040636435*pi) node[3];
rz(3.4315607876313265*pi) node[11];
cx node[13],node[12];
cx node[2],node[3];
sx node[11];
cx node[12],node[13];
cx node[3],node[2];
rz(1.5*pi) node[11];
cx node[2],node[3];
cx node[8],node[11];
cx node[3],node[2];
cx node[11],node[8];
cx node[1],node[2];
cx node[8],node[11];
rz(2.880879954203559*pi) node[2];
cx node[9],node[8];
cx node[14],node[11];
cx node[1],node[2];
rz(2.880888803218395*pi) node[8];
rz(0.9972220471858009*pi) node[11];
cx node[1],node[2];
cx node[9],node[8];
cx node[14],node[11];
cx node[2],node[1];
cx node[5],node[8];
sx node[14];
cx node[1],node[2];
rz(2.8808843150489998*pi) node[8];
rz(3.4315607876313265*pi) node[14];
cx node[4],node[1];
cx node[5],node[8];
sx node[14];
rz(2.8808893443452015*pi) node[1];
cx node[8],node[5];
rz(1.5*pi) node[14];
cx node[4],node[1];
cx node[5],node[8];
cx node[14],node[11];
cx node[0],node[1];
cx node[8],node[5];
cx node[11],node[14];
rz(2.880892113641211*pi) node[1];
cx node[3],node[5];
cx node[9],node[8];
cx node[14],node[11];
cx node[0],node[1];
rz(2.880881322936069*pi) node[5];
cx node[8],node[9];
cx node[16],node[14];
rz(3.6201986280313614*pi) node[1];
cx node[3],node[5];
cx node[9],node[8];
rz(0.9986471205462468*pi) node[14];
cx node[4],node[1];
cx node[5],node[3];
cx node[8],node[11];
cx node[16],node[14];
cx node[1],node[4];
cx node[3],node[5];
rz(2.880288152463166*pi) node[11];
cx node[13],node[14];
sx node[16];
cx node[4],node[1];
cx node[5],node[3];
cx node[8],node[11];
rz(0.998790359995029*pi) node[14];
rz(3.4315607876313265*pi) node[16];
cx node[2],node[3];
cx node[8],node[11];
cx node[13],node[14];
sx node[16];
rz(2.8808645480050674*pi) node[3];
cx node[11],node[8];
sx node[13];
rz(1.5*pi) node[16];
cx node[2],node[3];
cx node[8],node[11];
rz(3.4315607876313265*pi) node[13];
cx node[3],node[2];
cx node[9],node[8];
sx node[13];
cx node[2],node[3];
rz(2.8810024080167738*pi) node[8];
rz(1.5*pi) node[13];
cx node[3],node[2];
cx node[9],node[8];
cx node[14],node[13];
cx node[1],node[2];
cx node[5],node[8];
cx node[13],node[14];
rz(2.8809772933667537*pi) node[2];
rz(2.8809474040684413*pi) node[8];
cx node[14],node[13];
cx node[1],node[2];
cx node[5],node[8];
cx node[12],node[13];
cx node[16],node[14];
cx node[2],node[1];
cx node[8],node[5];
rz(0.9988632529589658*pi) node[13];
cx node[14],node[16];
cx node[1],node[2];
cx node[5],node[8];
cx node[12],node[13];
cx node[16],node[14];
cx node[2],node[1];
cx node[8],node[5];
cx node[11],node[14];
sx node[12];
rz(0.5022890533554332*pi) node[13];
cx node[0],node[1];
cx node[3],node[5];
cx node[9],node[8];
rz(3.4315607876313265*pi) node[12];
sx node[13];
rz(2.8807970663091966*pi) node[14];
rz(2.880890967725621*pi) node[1];
rz(2.881071544924053*pi) node[5];
cx node[8],node[9];
cx node[11],node[14];
sx node[12];
rz(3.4315607876313265*pi) node[13];
cx node[0],node[1];
cx node[3],node[5];
cx node[9],node[8];
cx node[14],node[11];
rz(1.5*pi) node[12];
sx node[13];
cx node[4],node[1];
cx node[5],node[3];
cx node[11],node[14];
rz(1.5*pi) node[13];
rz(2.88088937617619*pi) node[1];
cx node[3],node[5];
cx node[14],node[11];
cx node[12],node[13];
cx node[4],node[1];
cx node[5],node[3];
cx node[8],node[11];
cx node[13],node[12];
cx node[14],node[16];
rz(3.61920983020092*pi) node[1];
cx node[2],node[3];
rz(2.8808763573018448*pi) node[11];
cx node[12],node[13];
rz(2.880873683498801*pi) node[16];
cx node[0],node[1];
rz(2.8814851567901605*pi) node[3];
cx node[8],node[11];
cx node[14],node[16];
cx node[1],node[0];
cx node[2],node[3];
cx node[8],node[11];
cx node[14],node[13];
cx node[0],node[1];
cx node[3],node[2];
cx node[11],node[8];
rz(2.8808414068763417*pi) node[13];
cx node[2],node[3];
cx node[8],node[11];
cx node[14],node[13];
cx node[3],node[2];
cx node[9],node[8];
cx node[14],node[13];
cx node[1],node[2];
rz(2.8808926865990063*pi) node[8];
cx node[13],node[14];
rz(2.88083017053736*pi) node[2];
cx node[9],node[8];
cx node[14],node[13];
cx node[1],node[2];
cx node[5],node[8];
cx node[13],node[12];
cx node[16],node[14];
cx node[1],node[2];
rz(2.8808675401179977*pi) node[8];
rz(2.880799198985434*pi) node[12];
cx node[14],node[16];
cx node[2],node[1];
cx node[5],node[8];
cx node[13],node[12];
cx node[16],node[14];
cx node[1],node[2];
cx node[8],node[5];
cx node[11],node[14];
sx node[13];
cx node[4],node[1];
cx node[5],node[8];
rz(3.965799972995537*pi) node[13];
rz(2.880882532513637*pi) node[14];
rz(2.8808417570172167*pi) node[1];
cx node[8],node[5];
cx node[11],node[14];
sx node[13];
cx node[4],node[1];
cx node[3],node[5];
cx node[9],node[8];
cx node[14],node[11];
rz(0.5*pi) node[13];
cx node[0],node[1];
rz(2.880823263212829*pi) node[5];
cx node[8],node[9];
cx node[11],node[14];
cx node[12],node[13];
rz(2.8807120139076083*pi) node[1];
cx node[3],node[5];
cx node[9],node[8];
cx node[14],node[11];
cx node[13],node[12];
cx node[0],node[1];
cx node[5],node[3];
cx node[8],node[11];
cx node[12],node[13];
cx node[14],node[16];
rz(3.5852118833703672*pi) node[1];
cx node[3],node[5];
rz(2.8808839967391133*pi) node[11];
rz(2.8808903311058485*pi) node[16];
cx node[4],node[1];
cx node[5],node[3];
cx node[8],node[11];
cx node[14],node[16];
cx node[1],node[4];
cx node[2],node[3];
cx node[8],node[11];
cx node[14],node[13];
cx node[4],node[1];
rz(2.880880049696525*pi) node[3];
cx node[11],node[8];
rz(2.880871264343666*pi) node[13];
cx node[2],node[3];
cx node[8],node[11];
cx node[14],node[13];
cx node[3],node[2];
cx node[9],node[8];
sx node[14];
cx node[2],node[3];
rz(2.8808937051906423*pi) node[8];
rz(3.965799972995537*pi) node[14];
cx node[3],node[2];
cx node[9],node[8];
sx node[14];
cx node[1],node[2];
cx node[5],node[8];
rz(0.5*pi) node[14];
rz(2.880868431385679*pi) node[2];
rz(2.8808240908185336*pi) node[8];
cx node[16],node[14];
cx node[1],node[2];
cx node[5],node[8];
cx node[14],node[16];
cx node[2],node[1];
cx node[8],node[5];
cx node[16],node[14];
cx node[1],node[2];
cx node[5],node[8];
cx node[11],node[14];
cx node[2],node[1];
cx node[8],node[5];
rz(2.88089329138779*pi) node[14];
cx node[0],node[1];
cx node[3],node[5];
cx node[9],node[8];
cx node[11],node[14];
rz(2.880895392233039*pi) node[1];
rz(2.880883041809455*pi) node[5];
cx node[8],node[9];
cx node[14],node[11];
cx node[0],node[1];
cx node[3],node[5];
cx node[9],node[8];
cx node[11],node[14];
cx node[4],node[1];
cx node[5],node[3];
cx node[14],node[11];
rz(2.881152363804155*pi) node[1];
cx node[3],node[5];
cx node[8],node[11];
cx node[14],node[13];
cx node[4],node[1];
cx node[5],node[3];
rz(2.8808876254718156*pi) node[11];
rz(2.8808627336387165*pi) node[13];
rz(3.6193257586614678*pi) node[1];
cx node[2],node[3];
cx node[8],node[11];
cx node[14],node[13];
cx node[0],node[1];
rz(2.8808881984296106*pi) node[3];
cx node[8],node[11];
sx node[14];
cx node[1],node[0];
cx node[2],node[3];
cx node[11],node[8];
rz(3.965799972995537*pi) node[14];
cx node[0],node[1];
cx node[3],node[2];
cx node[8],node[11];
sx node[14];
cx node[2],node[3];
cx node[9],node[8];
rz(0.5*pi) node[14];
cx node[3],node[2];
rz(2.88089927561365*pi) node[8];
cx node[13],node[14];
cx node[1],node[2];
cx node[9],node[8];
cx node[14],node[13];
rz(2.8808814184290354*pi) node[2];
cx node[5],node[8];
cx node[13],node[14];
cx node[1],node[2];
rz(2.8808825006826484*pi) node[8];
cx node[11],node[14];
cx node[1],node[2];
cx node[5],node[8];
rz(2.8809346398420055*pi) node[14];
cx node[2],node[1];
cx node[8],node[5];
cx node[11],node[14];
cx node[1],node[2];
cx node[5],node[8];
sx node[11];
cx node[4],node[1];
cx node[8],node[5];
rz(3.965799972995537*pi) node[11];
rz(2.8810642556276593*pi) node[1];
cx node[3],node[5];
sx node[11];
cx node[4],node[1];
rz(2.880888484908508*pi) node[5];
rz(0.5*pi) node[11];
cx node[0],node[1];
cx node[3],node[5];
cx node[14],node[11];
rz(2.8809032226562388*pi) node[1];
cx node[5],node[3];
cx node[11],node[14];
cx node[0],node[1];
cx node[3],node[5];
cx node[14],node[11];
rz(3.6201650463383688*pi) node[1];
cx node[5],node[3];
cx node[11],node[8];
cx node[2],node[3];
cx node[8],node[11];
rz(2.8808935142047107*pi) node[3];
cx node[11],node[8];
cx node[2],node[3];
cx node[9],node[8];
cx node[3],node[2];
rz(2.8802981155626037*pi) node[8];
cx node[2],node[3];
cx node[9],node[8];
cx node[3],node[2];
cx node[11],node[8];
sx node[9];
cx node[2],node[1];
rz(2.880880368006411*pi) node[8];
rz(3.965799972995537*pi) node[9];
cx node[1],node[2];
cx node[11],node[8];
sx node[9];
cx node[2],node[1];
cx node[5],node[8];
rz(0.5*pi) node[9];
sx node[11];
cx node[4],node[1];
rz(2.880890203781894*pi) node[8];
rz(3.965799972995537*pi) node[11];
rz(2.8808635294134315*pi) node[1];
cx node[5],node[8];
sx node[11];
cx node[4],node[1];
sx node[5];
rz(0.5*pi) node[11];
cx node[0],node[1];
rz(3.965799972995537*pi) node[5];
rz(2.880900135050343*pi) node[1];
sx node[5];
cx node[0],node[1];
rz(0.5*pi) node[5];
cx node[2],node[1];
cx node[8],node[5];
rz(2.880883710260216*pi) node[1];
cx node[5],node[8];
cx node[2],node[1];
cx node[8],node[5];
rz(3.619067354695864*pi) node[1];
cx node[3],node[5];
rz(2.8808779806822646*pi) node[5];
cx node[3],node[5];
sx node[3];
rz(3.965799972995537*pi) node[3];
sx node[3];
rz(0.5*pi) node[3];
cx node[5],node[3];
cx node[3],node[5];
cx node[5],node[3];
cx node[3],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[1];
cx node[4],node[1];
rz(2.8812216916973656*pi) node[1];
cx node[4],node[1];
cx node[0],node[1];
sx node[4];
rz(2.8809480406882138*pi) node[1];
rz(3.965799972995537*pi) node[4];
cx node[0],node[1];
sx node[4];
sx node[0];
cx node[1],node[2];
rz(0.5*pi) node[4];
rz(3.965799972995537*pi) node[0];
cx node[2],node[1];
sx node[0];
cx node[1],node[2];
rz(0.5*pi) node[0];
cx node[3],node[2];
rz(2.880920506883059*pi) node[2];
cx node[3],node[2];
cx node[1],node[2];
sx node[3];
rz(2.880906533079055*pi) node[2];
rz(3.965799972995537*pi) node[3];
cx node[1],node[2];
sx node[3];
sx node[1];
rz(3.618872198904645*pi) node[2];
rz(0.5*pi) node[3];
rz(3.965799972995537*pi) node[1];
sx node[2];
sx node[1];
rz(3.965799972995537*pi) node[2];
rz(0.5*pi) node[1];
sx node[2];
rz(0.5*pi) node[2];
barrier node[2],node[1],node[3],node[0],node[4],node[5],node[8],node[11],node[9],node[14],node[13],node[16],node[12];
measure node[2] -> meas[0];
measure node[1] -> meas[1];
measure node[3] -> meas[2];
measure node[0] -> meas[3];
measure node[4] -> meas[4];
measure node[5] -> meas[5];
measure node[8] -> meas[6];
measure node[11] -> meas[7];
measure node[9] -> meas[8];
measure node[14] -> meas[9];
measure node[13] -> meas[10];
measure node[16] -> meas[11];
measure node[12] -> meas[12];
