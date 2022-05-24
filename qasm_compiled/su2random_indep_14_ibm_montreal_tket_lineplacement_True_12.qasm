OPENQASM 2.0;
include "qelib1.inc";

qreg node[26];
creg meas[14];
rz(3.5971967832159946*pi) node[1];
sx node[4];
sx node[6];
sx node[7];
sx node[10];
sx node[12];
sx node[13];
sx node[15];
sx node[17];
sx node[18];
sx node[21];
sx node[23];
sx node[24];
sx node[25];
sx node[1];
rz(3.1191871007121637*pi) node[4];
rz(3.2620692148166395*pi) node[6];
rz(3.281399578965094*pi) node[7];
rz(3.047325021539668*pi) node[10];
rz(3.145132797366012*pi) node[12];
rz(3.0958382015745936*pi) node[13];
rz(3.2169342003080033*pi) node[15];
rz(3.193608980242866*pi) node[17];
rz(3.014729821814144*pi) node[18];
rz(3.02993745255055*pi) node[21];
rz(3.082498445399612*pi) node[23];
rz(3.132290031148725*pi) node[24];
rz(3.1042918246022806*pi) node[25];
rz(2.506439885965802*pi) node[1];
sx node[4];
sx node[6];
sx node[7];
sx node[10];
sx node[12];
sx node[13];
sx node[15];
sx node[17];
sx node[18];
sx node[21];
sx node[23];
sx node[24];
sx node[25];
sx node[1];
rz(1.022928553107511*pi) node[4];
rz(1.1298422663211583*pi) node[6];
rz(1.2278545339676834*pi) node[7];
rz(1.2601009647340153*pi) node[10];
rz(1.1909749691177942*pi) node[12];
rz(1.15015416765155*pi) node[13];
rz(1.1982678878779078*pi) node[15];
rz(1.0897552901003245*pi) node[17];
rz(1.2364413314855522*pi) node[18];
rz(1.068089269866217*pi) node[21];
rz(1.2103477034951986*pi) node[23];
rz(1.2566620147518606*pi) node[24];
rz(1.3082904840935698*pi) node[25];
rz(1.5204010282488098*pi) node[1];
cx node[23],node[24];
cx node[25],node[24];
cx node[24],node[25];
cx node[25],node[24];
cx node[23],node[24];
cx node[23],node[21];
cx node[25],node[24];
cx node[23],node[21];
cx node[25],node[24];
cx node[21],node[23];
cx node[24],node[25];
cx node[23],node[21];
cx node[25],node[24];
cx node[21],node[18];
cx node[24],node[23];
cx node[21],node[18];
cx node[24],node[23];
cx node[18],node[21];
cx node[23],node[24];
cx node[21],node[18];
cx node[24],node[23];
cx node[18],node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
cx node[18],node[15];
cx node[21],node[23];
cx node[24],node[25];
cx node[15],node[18];
cx node[23],node[21];
cx node[25],node[24];
cx node[18],node[15];
cx node[24],node[23];
cx node[15],node[12];
cx node[21],node[18];
cx node[24],node[23];
cx node[15],node[12];
cx node[21],node[18];
cx node[23],node[24];
cx node[12],node[15];
cx node[18],node[21];
cx node[24],node[23];
cx node[15],node[12];
cx node[21],node[18];
cx node[25],node[24];
cx node[12],node[10];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
cx node[12],node[13];
cx node[18],node[15];
cx node[23],node[21];
cx node[24],node[25];
cx node[12],node[10];
cx node[18],node[15];
cx node[21],node[23];
cx node[25],node[24];
cx node[10],node[12];
cx node[15],node[18];
cx node[23],node[21];
cx node[12],node[10];
cx node[18],node[15];
cx node[24],node[23];
cx node[10],node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[23],node[24];
cx node[10],node[7];
cx node[15],node[12];
cx node[18],node[17];
cx node[24],node[23];
cx node[7],node[10];
cx node[12],node[15];
cx node[17],node[18];
cx node[23],node[24];
cx node[10],node[7];
cx node[15],node[12];
cx node[21],node[18];
cx node[25],node[24];
cx node[7],node[4];
cx node[12],node[13];
cx node[21],node[18];
cx node[25],node[24];
cx node[7],node[6];
cx node[12],node[10];
cx node[18],node[21];
cx node[24],node[25];
cx node[7],node[4];
cx node[12],node[10];
cx node[21],node[18];
cx node[25],node[24];
cx node[4],node[1];
cx node[10],node[12];
cx node[18],node[17];
cx node[23],node[21];
cx node[7],node[4];
cx node[12],node[10];
cx node[18],node[15];
cx node[23],node[21];
cx node[4],node[1];
sx node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[21],node[23];
rz(3.307392980721585*pi) node[7];
cx node[12],node[13];
cx node[15],node[18];
cx node[23],node[21];
sx node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[24],node[23];
rz(1.296121232947558*pi) node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[24],node[23];
cx node[4],node[7];
cx node[12],node[15];
cx node[18],node[17];
cx node[23],node[24];
cx node[7],node[4];
cx node[15],node[12];
cx node[17],node[18];
cx node[24],node[23];
cx node[4],node[7];
cx node[12],node[15];
cx node[21],node[18];
cx node[25],node[24];
cx node[1],node[4];
cx node[10],node[7];
cx node[12],node[13];
cx node[21],node[18];
cx node[25],node[24];
cx node[4],node[1];
cx node[10],node[7];
cx node[18],node[21];
cx node[24],node[25];
cx node[1],node[4];
cx node[7],node[10];
cx node[21],node[18];
cx node[25],node[24];
cx node[10],node[7];
cx node[18],node[17];
cx node[23],node[21];
cx node[7],node[6];
cx node[12],node[10];
cx node[18],node[15];
cx node[23],node[21];
cx node[7],node[4];
cx node[12],node[10];
cx node[18],node[15];
cx node[21],node[23];
sx node[7];
cx node[10],node[12];
cx node[15],node[18];
cx node[23],node[21];
rz(3.255845833189598*pi) node[7];
cx node[12],node[10];
cx node[18],node[15];
cx node[24],node[23];
sx node[7];
cx node[13],node[12];
cx node[17],node[18];
cx node[23],node[24];
rz(1.3116872325510143*pi) node[7];
cx node[12],node[13];
cx node[18],node[17];
cx node[24],node[23];
cx node[7],node[4];
cx node[13],node[12];
cx node[17],node[18];
cx node[23],node[24];
cx node[4],node[7];
cx node[15],node[12];
cx node[21],node[18];
cx node[25],node[24];
cx node[7],node[4];
cx node[12],node[15];
cx node[21],node[18];
cx node[25],node[24];
cx node[1],node[4];
cx node[6],node[7];
cx node[15],node[12];
cx node[18],node[21];
cx node[24],node[25];
cx node[1],node[4];
cx node[7],node[6];
cx node[12],node[15];
cx node[21],node[18];
cx node[25],node[24];
cx node[4],node[1];
cx node[6],node[7];
cx node[12],node[13];
cx node[18],node[17];
cx node[23],node[21];
cx node[1],node[4];
cx node[10],node[7];
cx node[18],node[15];
cx node[23],node[21];
cx node[10],node[7];
cx node[18],node[15];
cx node[21],node[23];
cx node[7],node[10];
cx node[15],node[18];
cx node[23],node[21];
cx node[10],node[7];
cx node[18],node[15];
cx node[24],node[23];
cx node[7],node[6];
cx node[12],node[10];
cx node[17],node[18];
cx node[23],node[24];
sx node[7];
cx node[12],node[10];
cx node[18],node[17];
cx node[24],node[23];
rz(3.043549707134585*pi) node[7];
cx node[10],node[12];
cx node[17],node[18];
cx node[23],node[24];
sx node[7];
cx node[12],node[10];
cx node[21],node[18];
cx node[25],node[24];
rz(1.199476230403048*pi) node[7];
cx node[13],node[12];
cx node[21],node[18];
cx node[25],node[24];
cx node[4],node[7];
cx node[12],node[13];
cx node[18],node[21];
cx node[24],node[25];
cx node[7],node[4];
cx node[13],node[12];
cx node[21],node[18];
cx node[25],node[24];
cx node[4],node[7];
cx node[15],node[12];
cx node[18],node[17];
cx node[23],node[21];
cx node[7],node[4];
cx node[12],node[15];
cx node[23],node[21];
cx node[1],node[4];
cx node[6],node[7];
cx node[15],node[12];
cx node[21],node[23];
cx node[1],node[4];
cx node[7],node[6];
cx node[12],node[15];
cx node[23],node[21];
cx node[4],node[1];
cx node[6],node[7];
cx node[12],node[13];
cx node[18],node[15];
cx node[24],node[23];
cx node[1],node[4];
cx node[10],node[7];
cx node[18],node[15];
cx node[23],node[24];
sx node[10];
cx node[15],node[18];
cx node[24],node[23];
rz(3.222729751166322*pi) node[10];
cx node[18],node[15];
cx node[23],node[24];
sx node[10];
cx node[17],node[18];
cx node[25],node[24];
rz(1.212389470429136*pi) node[10];
cx node[18],node[17];
cx node[25],node[24];
cx node[10],node[7];
cx node[17],node[18];
cx node[24],node[25];
cx node[7],node[10];
cx node[21],node[18];
cx node[25],node[24];
cx node[10],node[7];
cx node[21],node[18];
cx node[6],node[7];
cx node[12],node[10];
cx node[18],node[21];
cx node[4],node[7];
sx node[12];
cx node[21],node[18];
cx node[7],node[4];
rz(3.2178275019894906*pi) node[12];
cx node[18],node[17];
cx node[23],node[21];
cx node[4],node[7];
sx node[12];
cx node[23],node[21];
cx node[7],node[4];
rz(1.0156359774854549*pi) node[12];
cx node[21],node[23];
cx node[1],node[4];
cx node[6],node[7];
cx node[12],node[10];
cx node[23],node[21];
cx node[1],node[4];
cx node[7],node[6];
cx node[10],node[12];
cx node[24],node[23];
cx node[4],node[1];
cx node[6],node[7];
cx node[12],node[10];
cx node[23],node[24];
cx node[1],node[4];
cx node[7],node[10];
cx node[13],node[12];
cx node[24],node[23];
cx node[10],node[7];
cx node[12],node[13];
cx node[23],node[24];
cx node[7],node[10];
cx node[13],node[12];
cx node[25],node[24];
cx node[10],node[7];
cx node[15],node[12];
cx node[25],node[24];
cx node[6],node[7];
cx node[12],node[15];
cx node[24],node[25];
cx node[4],node[7];
cx node[15],node[12];
cx node[25],node[24];
cx node[7],node[4];
cx node[12],node[15];
cx node[4],node[7];
cx node[12],node[13];
cx node[18],node[15];
cx node[7],node[4];
sx node[12];
cx node[18],node[15];
cx node[1],node[4];
cx node[6],node[7];
rz(3.1862142787135466*pi) node[12];
cx node[15],node[18];
cx node[1],node[4];
cx node[7],node[6];
sx node[12];
cx node[18],node[15];
cx node[4],node[1];
cx node[6],node[7];
rz(1.0439671641841175*pi) node[12];
cx node[17],node[18];
cx node[1],node[4];
cx node[10],node[12];
cx node[18],node[17];
cx node[12],node[10];
cx node[17],node[18];
cx node[10],node[12];
cx node[21],node[18];
cx node[12],node[10];
cx node[21],node[18];
cx node[7],node[10];
cx node[13],node[12];
cx node[18],node[21];
cx node[10],node[7];
cx node[12],node[13];
cx node[21],node[18];
cx node[7],node[10];
cx node[13],node[12];
cx node[18],node[17];
cx node[23],node[21];
cx node[10],node[7];
cx node[15],node[12];
cx node[23],node[21];
cx node[6],node[7];
sx node[15];
cx node[21],node[23];
cx node[4],node[7];
rz(3.233851259857169*pi) node[15];
cx node[23],node[21];
cx node[7],node[4];
sx node[15];
cx node[24],node[23];
cx node[4],node[7];
rz(1.1318662460986555*pi) node[15];
cx node[23],node[24];
cx node[7],node[4];
cx node[12],node[15];
cx node[24],node[23];
cx node[1],node[4];
cx node[6],node[7];
cx node[15],node[12];
cx node[23],node[24];
cx node[1],node[4];
cx node[7],node[6];
cx node[12],node[15];
cx node[25],node[24];
cx node[4],node[1];
cx node[6],node[7];
cx node[13],node[12];
cx node[18],node[15];
cx node[25],node[24];
cx node[1],node[4];
cx node[10],node[12];
sx node[18];
cx node[24],node[25];
cx node[12],node[10];
rz(3.147681593751454*pi) node[18];
cx node[25],node[24];
cx node[10],node[12];
sx node[18];
cx node[12],node[10];
rz(1.12094981491818*pi) node[18];
cx node[7],node[10];
cx node[13],node[12];
cx node[18],node[15];
cx node[10],node[7];
cx node[12],node[13];
cx node[15],node[18];
cx node[7],node[10];
cx node[13],node[12];
cx node[18],node[15];
cx node[10],node[7];
cx node[12],node[15];
cx node[17],node[18];
cx node[6],node[7];
cx node[12],node[15];
cx node[18],node[17];
cx node[4],node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[7],node[4];
cx node[12],node[15];
cx node[21],node[18];
cx node[4],node[7];
cx node[13],node[12];
cx node[21],node[18];
cx node[7],node[4];
cx node[10],node[12];
cx node[18],node[21];
cx node[1],node[4];
cx node[6],node[7];
cx node[12],node[10];
cx node[21],node[18];
cx node[1],node[4];
cx node[7],node[6];
cx node[10],node[12];
cx node[18],node[17];
cx node[23],node[21];
cx node[4],node[1];
cx node[6],node[7];
cx node[12],node[10];
sx node[18];
cx node[23],node[21];
cx node[1],node[4];
cx node[7],node[10];
cx node[13],node[12];
rz(3.2012234110866187*pi) node[18];
cx node[21],node[23];
cx node[10],node[7];
cx node[12],node[13];
sx node[18];
cx node[23],node[21];
cx node[7],node[10];
cx node[13],node[12];
rz(1.2372328281161415*pi) node[18];
cx node[24],node[23];
cx node[10],node[7];
cx node[15],node[18];
cx node[23],node[24];
cx node[6],node[7];
cx node[18],node[15];
cx node[24],node[23];
cx node[4],node[7];
cx node[15],node[18];
cx node[23],node[24];
cx node[7],node[4];
cx node[18],node[15];
cx node[25],node[24];
cx node[4],node[7];
cx node[12],node[15];
cx node[17],node[18];
cx node[25],node[24];
cx node[7],node[4];
cx node[12],node[15];
cx node[18],node[17];
cx node[24],node[25];
cx node[1],node[4];
cx node[6],node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[25],node[24];
cx node[1],node[4];
cx node[7],node[6];
cx node[12],node[15];
cx node[21],node[18];
cx node[4],node[1];
cx node[6],node[7];
cx node[13],node[12];
sx node[21];
cx node[1],node[4];
cx node[10],node[12];
rz(3.314603568629637*pi) node[21];
cx node[12],node[10];
sx node[21];
cx node[10],node[12];
rz(1.0875956307338412*pi) node[21];
cx node[12],node[10];
cx node[21],node[18];
cx node[7],node[10];
cx node[13],node[12];
cx node[18],node[21];
cx node[10],node[7];
cx node[12],node[13];
cx node[21],node[18];
cx node[7],node[10];
cx node[13],node[12];
cx node[17],node[18];
cx node[23],node[21];
cx node[10],node[7];
cx node[15],node[18];
sx node[23];
cx node[6],node[7];
cx node[18],node[15];
rz(3.036280130038426*pi) node[23];
cx node[4],node[7];
cx node[15],node[18];
sx node[23];
cx node[7],node[4];
cx node[18],node[15];
rz(1.310408031062111*pi) node[23];
cx node[4],node[7];
cx node[12],node[15];
cx node[17],node[18];
cx node[23],node[21];
cx node[7],node[4];
cx node[12],node[15];
cx node[18],node[17];
cx node[21],node[23];
cx node[1],node[4];
cx node[6],node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[23],node[21];
cx node[1],node[4];
cx node[7],node[6];
cx node[12],node[15];
cx node[18],node[21];
cx node[24],node[23];
cx node[4],node[1];
cx node[6],node[7];
cx node[13],node[12];
cx node[21],node[18];
sx node[24];
cx node[1],node[4];
cx node[10],node[12];
cx node[18],node[21];
rz(3.296363695954062*pi) node[24];
cx node[12],node[10];
cx node[21],node[18];
sx node[24];
cx node[10],node[12];
cx node[17],node[18];
rz(1.1936844165027929*pi) node[24];
cx node[12],node[10];
cx node[15],node[18];
cx node[23],node[24];
cx node[7],node[10];
cx node[13],node[12];
cx node[18],node[15];
cx node[24],node[23];
cx node[10],node[7];
cx node[12],node[13];
cx node[15],node[18];
cx node[23],node[24];
cx node[7],node[10];
cx node[13],node[12];
cx node[18],node[15];
cx node[21],node[23];
cx node[25],node[24];
cx node[10],node[7];
cx node[12],node[15];
cx node[17],node[18];
cx node[23],node[21];
rz(2.048339032696194*pi) node[24];
sx node[25];
cx node[6],node[7];
cx node[12],node[15];
cx node[18],node[17];
cx node[21],node[23];
sx node[24];
rz(3.1768105579713803*pi) node[25];
cx node[4],node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[23],node[21];
rz(0.7547948334040813*pi) node[24];
sx node[25];
cx node[7],node[4];
cx node[12],node[15];
cx node[18],node[21];
sx node[24];
rz(1.2013136105590665*pi) node[25];
cx node[4],node[7];
cx node[13],node[12];
cx node[21],node[18];
rz(1.0*pi) node[24];
cx node[7],node[4];
cx node[10],node[12];
cx node[18],node[21];
cx node[25],node[24];
cx node[1],node[4];
cx node[6],node[7];
cx node[12],node[10];
cx node[21],node[18];
cx node[24],node[25];
cx node[1],node[4];
cx node[7],node[6];
cx node[10],node[12];
cx node[17],node[18];
cx node[25],node[24];
cx node[4],node[1];
cx node[6],node[7];
cx node[12],node[10];
cx node[15],node[18];
cx node[23],node[24];
cx node[1],node[4];
cx node[7],node[10];
cx node[13],node[12];
cx node[18],node[15];
cx node[23],node[24];
cx node[10],node[7];
cx node[12],node[13];
cx node[15],node[18];
cx node[24],node[23];
cx node[7],node[10];
cx node[13],node[12];
cx node[18],node[15];
cx node[23],node[24];
cx node[10],node[7];
cx node[12],node[15];
cx node[17],node[18];
cx node[21],node[23];
cx node[24],node[25];
cx node[6],node[7];
cx node[12],node[15];
cx node[18],node[17];
cx node[23],node[21];
sx node[24];
cx node[4],node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[21],node[23];
rz(3.0424730239445688*pi) node[24];
cx node[7],node[4];
cx node[12],node[15];
cx node[23],node[21];
sx node[24];
cx node[4],node[7];
cx node[13],node[12];
cx node[18],node[21];
rz(1.148400592122366*pi) node[24];
cx node[7],node[4];
cx node[10],node[12];
cx node[21],node[18];
cx node[25],node[24];
cx node[1],node[4];
cx node[6],node[7];
cx node[12],node[10];
cx node[18],node[21];
cx node[24],node[25];
cx node[1],node[4];
cx node[7],node[6];
cx node[10],node[12];
cx node[21],node[18];
cx node[25],node[24];
cx node[4],node[1];
cx node[6],node[7];
cx node[12],node[10];
cx node[17],node[18];
cx node[23],node[24];
cx node[1],node[4];
cx node[7],node[10];
cx node[13],node[12];
cx node[15],node[18];
sx node[23];
cx node[10],node[7];
cx node[12],node[13];
cx node[18],node[15];
rz(3.070827670718463*pi) node[23];
cx node[7],node[10];
cx node[13],node[12];
cx node[15],node[18];
sx node[23];
cx node[10],node[7];
cx node[18],node[15];
rz(1.1678231801941443*pi) node[23];
cx node[6],node[7];
cx node[12],node[15];
cx node[17],node[18];
cx node[23],node[24];
cx node[4],node[7];
cx node[12],node[15];
cx node[18],node[17];
cx node[24],node[23];
cx node[7],node[4];
cx node[15],node[12];
cx node[17],node[18];
cx node[23],node[24];
cx node[4],node[7];
cx node[12],node[15];
cx node[21],node[23];
cx node[25],node[24];
cx node[7],node[4];
cx node[13],node[12];
sx node[21];
cx node[25],node[24];
cx node[1],node[4];
cx node[6],node[7];
cx node[10],node[12];
rz(3.024721104727329*pi) node[21];
cx node[24],node[25];
cx node[1],node[4];
cx node[7],node[6];
cx node[12],node[10];
sx node[21];
cx node[25],node[24];
cx node[4],node[1];
cx node[6],node[7];
cx node[10],node[12];
rz(1.0182567176347521*pi) node[21];
cx node[1],node[4];
cx node[12],node[10];
cx node[23],node[21];
cx node[7],node[10];
cx node[13],node[12];
cx node[21],node[23];
cx node[10],node[7];
cx node[12],node[13];
cx node[23],node[21];
cx node[7],node[10];
cx node[13],node[12];
cx node[18],node[21];
cx node[24],node[23];
cx node[10],node[7];
sx node[18];
cx node[23],node[24];
cx node[6],node[7];
rz(3.050642278469236*pi) node[18];
cx node[24],node[23];
cx node[4],node[7];
sx node[18];
cx node[23],node[24];
cx node[7],node[4];
rz(1.1672754771053833*pi) node[18];
cx node[25],node[24];
cx node[4],node[7];
cx node[21],node[18];
cx node[25],node[24];
cx node[7],node[4];
cx node[18],node[21];
cx node[24],node[25];
cx node[1],node[4];
cx node[6],node[7];
cx node[21],node[18];
cx node[25],node[24];
cx node[1],node[4];
cx node[7],node[6];
cx node[17],node[18];
cx node[23],node[21];
cx node[4],node[1];
cx node[6],node[7];
cx node[15],node[18];
sx node[17];
cx node[23],node[21];
cx node[1],node[4];
sx node[15];
rz(3.1589688782310255*pi) node[17];
cx node[21],node[23];
rz(3.249456666224535*pi) node[15];
sx node[17];
cx node[23],node[21];
sx node[15];
rz(1.3080178262112625*pi) node[17];
cx node[24],node[23];
rz(1.2923029180599506*pi) node[15];
cx node[23],node[24];
cx node[18],node[15];
cx node[24],node[23];
cx node[15],node[18];
cx node[23],node[24];
cx node[18],node[15];
cx node[25],node[24];
cx node[12],node[15];
cx node[17],node[18];
cx node[25],node[24];
sx node[12];
cx node[18],node[17];
cx node[24],node[25];
rz(3.0260634382075087*pi) node[12];
cx node[17],node[18];
cx node[25],node[24];
sx node[12];
cx node[21],node[18];
rz(1.232376036774156*pi) node[12];
cx node[21],node[18];
cx node[12],node[15];
cx node[18],node[21];
cx node[15],node[12];
cx node[21],node[18];
cx node[12],node[15];
cx node[18],node[17];
cx node[23],node[21];
cx node[13],node[12];
cx node[18],node[15];
cx node[23],node[21];
cx node[10],node[12];
sx node[13];
cx node[18],node[15];
cx node[21],node[23];
sx node[10];
rz(3.143309849380233*pi) node[13];
cx node[15],node[18];
cx node[23],node[21];
rz(3.290546098953026*pi) node[10];
sx node[13];
cx node[18],node[15];
cx node[24],node[23];
sx node[10];
rz(1.229413332494413*pi) node[13];
cx node[17],node[18];
cx node[23],node[24];
rz(1.1174988296392288*pi) node[10];
cx node[18],node[17];
cx node[24],node[23];
cx node[12],node[10];
cx node[17],node[18];
cx node[23],node[24];
cx node[10],node[12];
cx node[21],node[18];
cx node[25],node[24];
cx node[12],node[10];
cx node[21],node[18];
cx node[25],node[24];
cx node[7],node[10];
cx node[13],node[12];
cx node[18],node[21];
cx node[24],node[25];
sx node[7];
cx node[12],node[13];
cx node[21],node[18];
cx node[25],node[24];
rz(3.1993796297187886*pi) node[7];
cx node[13],node[12];
cx node[18],node[17];
cx node[23],node[21];
sx node[7];
cx node[15],node[12];
cx node[23],node[21];
rz(1.1299946571791688*pi) node[7];
cx node[12],node[15];
cx node[21],node[23];
cx node[10],node[7];
cx node[15],node[12];
cx node[23],node[21];
cx node[7],node[10];
cx node[12],node[15];
cx node[24],node[23];
cx node[10],node[7];
cx node[12],node[13];
cx node[18],node[15];
cx node[23],node[24];
cx node[6],node[7];
cx node[12],node[10];
cx node[18],node[15];
cx node[24],node[23];
cx node[4],node[7];
sx node[6];
cx node[12],node[10];
cx node[15],node[18];
cx node[23],node[24];
sx node[4];
rz(3.218999808652416*pi) node[6];
cx node[10],node[12];
cx node[18],node[15];
cx node[25],node[24];
rz(3.1408528631152635*pi) node[4];
sx node[6];
cx node[12],node[10];
cx node[17],node[18];
cx node[25],node[24];
sx node[4];
rz(1.1743992841891653*pi) node[6];
cx node[13],node[12];
cx node[18],node[17];
cx node[24],node[25];
rz(1.1580518433644242*pi) node[4];
cx node[12],node[13];
cx node[17],node[18];
cx node[25],node[24];
cx node[7],node[4];
cx node[13],node[12];
cx node[21],node[18];
cx node[4],node[7];
cx node[15],node[12];
cx node[21],node[18];
cx node[7],node[4];
cx node[12],node[15];
cx node[18],node[21];
cx node[1],node[4];
cx node[6],node[7];
cx node[15],node[12];
cx node[21],node[18];
sx node[1];
rz(2.048213405333414*pi) node[4];
cx node[7],node[6];
cx node[12],node[15];
cx node[18],node[17];
cx node[23],node[21];
rz(3.0034255908345413*pi) node[1];
sx node[4];
cx node[6],node[7];
cx node[12],node[13];
cx node[18],node[15];
cx node[23],node[21];
sx node[1];
rz(0.8496878647012329*pi) node[4];
cx node[10],node[7];
cx node[18],node[15];
cx node[21],node[23];
rz(1.0787110702329419*pi) node[1];
sx node[4];
cx node[10],node[7];
cx node[15],node[18];
cx node[23],node[21];
rz(1.0*pi) node[4];
cx node[7],node[10];
cx node[18],node[15];
cx node[24],node[23];
cx node[1],node[4];
cx node[10],node[7];
cx node[17],node[18];
cx node[23],node[24];
cx node[4],node[1];
cx node[7],node[6];
cx node[12],node[10];
cx node[18],node[17];
cx node[24],node[23];
cx node[1],node[4];
cx node[12],node[10];
cx node[17],node[18];
cx node[23],node[24];
cx node[7],node[4];
cx node[10],node[12];
cx node[21],node[18];
cx node[25],node[24];
cx node[7],node[4];
cx node[12],node[10];
cx node[21],node[18];
cx node[25],node[24];
cx node[4],node[7];
cx node[13],node[12];
cx node[18],node[21];
cx node[24],node[25];
cx node[7],node[4];
cx node[12],node[13];
cx node[21],node[18];
cx node[25],node[24];
cx node[4],node[1];
cx node[6],node[7];
cx node[13],node[12];
cx node[18],node[17];
cx node[23],node[21];
sx node[4];
cx node[7],node[6];
cx node[15],node[12];
cx node[23],node[21];
rz(3.0324815790116517*pi) node[4];
cx node[6],node[7];
cx node[12],node[15];
cx node[21],node[23];
sx node[4];
cx node[10],node[7];
cx node[15],node[12];
cx node[23],node[21];
rz(1.1028652679177662*pi) node[4];
cx node[10],node[7];
cx node[12],node[15];
cx node[24],node[23];
cx node[1],node[4];
cx node[7],node[10];
cx node[12],node[13];
cx node[18],node[15];
cx node[23],node[24];
cx node[4],node[1];
cx node[10],node[7];
cx node[18],node[15];
cx node[24],node[23];
cx node[1],node[4];
cx node[7],node[6];
cx node[12],node[10];
cx node[15],node[18];
cx node[23],node[24];
cx node[7],node[4];
cx node[12],node[10];
cx node[18],node[15];
cx node[25],node[24];
sx node[7];
cx node[10],node[12];
cx node[17],node[18];
cx node[25],node[24];
rz(3.2275525088152768*pi) node[7];
cx node[12],node[10];
cx node[18],node[17];
cx node[24],node[25];
sx node[7];
cx node[13],node[12];
cx node[17],node[18];
cx node[25],node[24];
rz(1.1486587923696427*pi) node[7];
cx node[12],node[13];
cx node[21],node[18];
cx node[6],node[7];
cx node[13],node[12];
cx node[21],node[18];
cx node[7],node[6];
cx node[15],node[12];
cx node[18],node[21];
cx node[6],node[7];
cx node[12],node[15];
cx node[21],node[18];
cx node[10],node[7];
cx node[15],node[12];
cx node[18],node[17];
cx node[23],node[21];
cx node[10],node[7];
cx node[12],node[15];
cx node[23],node[21];
cx node[7],node[10];
cx node[12],node[13];
cx node[18],node[15];
cx node[21],node[23];
cx node[10],node[7];
cx node[18],node[15];
cx node[23],node[21];
cx node[7],node[4];
cx node[12],node[10];
cx node[15],node[18];
cx node[24],node[23];
sx node[7];
cx node[12],node[10];
cx node[18],node[15];
cx node[23],node[24];
rz(3.044205304542367*pi) node[7];
cx node[10],node[12];
cx node[17],node[18];
cx node[24],node[23];
sx node[7];
cx node[12],node[10];
cx node[18],node[17];
cx node[23],node[24];
rz(1.2434888396896158*pi) node[7];
cx node[13],node[12];
cx node[17],node[18];
cx node[25],node[24];
cx node[4],node[7];
cx node[12],node[13];
cx node[21],node[18];
cx node[25],node[24];
cx node[7],node[4];
cx node[13],node[12];
cx node[21],node[18];
cx node[24],node[25];
cx node[4],node[7];
cx node[15],node[12];
cx node[18],node[21];
cx node[25],node[24];
cx node[10],node[7];
cx node[12],node[15];
cx node[21],node[18];
sx node[10];
cx node[15],node[12];
cx node[18],node[17];
cx node[23],node[21];
rz(3.1900587427583043*pi) node[10];
cx node[12],node[15];
cx node[23],node[21];
sx node[10];
cx node[12],node[13];
cx node[18],node[15];
cx node[21],node[23];
rz(1.2230600581521163*pi) node[10];
cx node[18],node[15];
cx node[23],node[21];
cx node[7],node[10];
cx node[15],node[18];
cx node[24],node[23];
cx node[10],node[7];
cx node[18],node[15];
cx node[23],node[24];
cx node[7],node[10];
cx node[17],node[18];
cx node[24],node[23];
cx node[12],node[10];
cx node[18],node[17];
cx node[23],node[24];
sx node[12];
cx node[17],node[18];
cx node[25],node[24];
rz(3.232986800234469*pi) node[12];
cx node[21],node[18];
cx node[25],node[24];
sx node[12];
cx node[21],node[18];
cx node[24],node[25];
rz(1.0644026716095123*pi) node[12];
cx node[18],node[21];
cx node[25],node[24];
cx node[13],node[12];
cx node[21],node[18];
cx node[12],node[13];
cx node[18],node[17];
cx node[23],node[21];
cx node[13],node[12];
cx node[23],node[21];
cx node[15],node[12];
cx node[21],node[23];
cx node[15],node[12];
cx node[23],node[21];
cx node[12],node[15];
cx node[24],node[23];
cx node[15],node[12];
cx node[23],node[24];
cx node[12],node[10];
cx node[18],node[15];
cx node[24],node[23];
sx node[12];
cx node[18],node[15];
cx node[23],node[24];
rz(3.055757715692339*pi) node[12];
cx node[15],node[18];
cx node[25],node[24];
sx node[12];
cx node[18],node[15];
cx node[25],node[24];
rz(1.1467013043442704*pi) node[12];
cx node[17],node[18];
cx node[24],node[25];
cx node[10],node[12];
cx node[18],node[17];
cx node[25],node[24];
cx node[12],node[10];
cx node[17],node[18];
cx node[10],node[12];
cx node[21],node[18];
cx node[15],node[12];
cx node[21],node[18];
sx node[15];
cx node[18],node[21];
rz(3.258010143063518*pi) node[15];
cx node[21],node[18];
sx node[15];
cx node[18],node[17];
cx node[23],node[21];
rz(1.2683642670976543*pi) node[15];
cx node[23],node[21];
cx node[12],node[15];
cx node[21],node[23];
cx node[15],node[12];
cx node[23],node[21];
cx node[12],node[15];
cx node[24],node[23];
cx node[18],node[15];
cx node[23],node[24];
sx node[18];
cx node[24],node[23];
rz(3.149579653321076*pi) node[18];
cx node[23],node[24];
sx node[18];
cx node[25],node[24];
rz(1.0437369051786498*pi) node[18];
cx node[25],node[24];
cx node[17],node[18];
cx node[24],node[25];
cx node[18],node[17];
cx node[25],node[24];
cx node[17],node[18];
cx node[21],node[18];
cx node[21],node[18];
cx node[18],node[21];
cx node[21],node[18];
cx node[18],node[15];
cx node[23],node[21];
sx node[18];
cx node[23],node[21];
rz(3.036543690624186*pi) node[18];
cx node[21],node[23];
sx node[18];
cx node[23],node[21];
rz(1.3180054546086446*pi) node[18];
cx node[24],node[23];
cx node[15],node[18];
cx node[25],node[24];
cx node[18],node[15];
cx node[24],node[23];
cx node[15],node[18];
cx node[25],node[24];
cx node[21],node[18];
cx node[24],node[23];
sx node[21];
rz(3.239105206444146*pi) node[21];
sx node[21];
rz(1.1509336862811221*pi) node[21];
cx node[18],node[21];
cx node[21],node[18];
cx node[18],node[21];
cx node[21],node[23];
cx node[23],node[21];
cx node[21],node[23];
cx node[24],node[23];
sx node[24];
rz(3.045017892386015*pi) node[24];
sx node[24];
rz(1.2306693960376895*pi) node[24];
cx node[25],node[24];
cx node[24],node[25];
cx node[25],node[24];
cx node[24],node[23];
cx node[21],node[23];
sx node[24];
sx node[21];
rz(3.791982660117263*pi) node[23];
rz(3.2061980025512833*pi) node[24];
rz(3.090817073204569*pi) node[21];
sx node[23];
sx node[24];
sx node[21];
rz(3.5*pi) node[23];
rz(1.0617148406488846*pi) node[24];
rz(1.2559119111388708*pi) node[21];
sx node[23];
rz(1.6329032073979755*pi) node[23];
barrier node[1],node[6],node[4],node[7],node[13],node[10],node[12],node[17],node[15],node[18],node[25],node[24],node[21],node[23];
measure node[1] -> meas[0];
measure node[6] -> meas[1];
measure node[4] -> meas[2];
measure node[7] -> meas[3];
measure node[13] -> meas[4];
measure node[10] -> meas[5];
measure node[12] -> meas[6];
measure node[17] -> meas[7];
measure node[15] -> meas[8];
measure node[18] -> meas[9];
measure node[25] -> meas[10];
measure node[24] -> meas[11];
measure node[21] -> meas[12];
measure node[23] -> meas[13];
