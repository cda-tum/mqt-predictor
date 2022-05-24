OPENQASM 2.0;
include "qelib1.inc";

qreg node[80];
creg meas[68];
rz(2.5*pi) node[0];
rz(2.5*pi) node[1];
rz(2.5*pi) node[2];
rz(2.5*pi) node[3];
rz(2.5*pi) node[4];
rz(0.5*pi) node[5];
rz(2.5*pi) node[6];
rz(2.5*pi) node[7];
rz(2.5*pi) node[8];
rz(2.5*pi) node[9];
rz(2.5*pi) node[10];
rz(2.5*pi) node[11];
rz(2.5*pi) node[12];
rz(2.5*pi) node[13];
rz(2.5*pi) node[14];
rz(2.5*pi) node[15];
rz(2.5*pi) node[16];
rz(2.5*pi) node[17];
rz(2.5*pi) node[18];
rz(2.5*pi) node[19];
rz(2.5*pi) node[20];
rz(2.5*pi) node[21];
rz(2.5*pi) node[22];
rz(2.5*pi) node[23];
rz(2.5*pi) node[24];
rz(2.5*pi) node[25];
rz(2.5*pi) node[26];
rz(2.5*pi) node[27];
rz(2.5*pi) node[28];
rz(2.5*pi) node[29];
rz(2.5*pi) node[30];
rz(2.5*pi) node[31];
rz(2.5*pi) node[32];
rz(2.5*pi) node[33];
rz(2.5*pi) node[34];
rz(2.5*pi) node[35];
rz(2.5*pi) node[36];
rz(2.5*pi) node[37];
rz(2.5*pi) node[38];
rz(2.5*pi) node[39];
rz(2.5*pi) node[40];
rz(2.5*pi) node[41];
rz(2.5*pi) node[42];
rz(2.5*pi) node[46];
rz(2.5*pi) node[47];
rz(2.5*pi) node[48];
rz(2.5*pi) node[49];
rz(2.5*pi) node[50];
rz(2.5*pi) node[51];
rz(2.5*pi) node[52];
rz(2.5*pi) node[53];
rz(2.5*pi) node[54];
rz(2.5*pi) node[56];
rz(2.5*pi) node[57];
rz(2.5*pi) node[58];
rz(2.5*pi) node[59];
rz(2.5*pi) node[60];
rz(2.5*pi) node[61];
rz(2.5*pi) node[62];
rz(2.5*pi) node[63];
rz(2.5*pi) node[64];
rz(2.5*pi) node[65];
rz(2.5*pi) node[70];
rz(2.5*pi) node[71];
rz(2.5*pi) node[72];
rz(2.5*pi) node[73];
rz(2.5*pi) node[78];
rz(2.5*pi) node[79];
rx(2.9601069137172544*pi) node[0];
rx(2.9604165655745343*pi) node[1];
rx(2.9607191509523405*pi) node[2];
rx(2.961014860836605*pi) node[3];
rx(2.9613040135372146*pi) node[4];
rx(1.0*pi) node[5];
rx(2.959465137324731*pi) node[6];
rx(2.959789877070616*pi) node[7];
rx(2.9518293287890227*pi) node[8];
rx(2.9512680529667144*pi) node[9];
rx(2.9506866917905885*pi) node[10];
rx(2.9543710650611885*pi) node[11];
rx(2.953894873471458*pi) node[12];
rx(2.9534034985001556*pi) node[13];
rx(2.9528960488796017*pi) node[14];
rx(2.9523716333421137*pi) node[15];
rx(2.9481341011513034*pi) node[16];
rx(2.9474315275705187*pi) node[17];
rx(2.9466996058182273*pi) node[18];
rx(2.9459362350491816*pi) node[19];
rx(2.9451390916012112*pi) node[20];
rx(2.950084003852088*pi) node[21];
rx(2.94945868408068*pi) node[22];
rx(2.9488092364198994*pi) node[23];
rx(2.9263184784537883*pi) node[24];
rx(2.9242609870114737*pi) node[25];
rx(2.9220208811874553*pi) node[26];
rx(2.919569385768022*pi) node[27];
rx(2.9331371855116632*pi) node[28];
rx(2.9316111760863093*pi) node[29];
rx(2.92997563622912*pi) node[30];
rx(2.9282168467839997*pi) node[31];
rx(2.8766241300087065*pi) node[32];
rx(2.8661397663015395*pi) node[33];
rx(2.852416376685532*pi) node[34];
rx(2.833333333333333*pi) node[35];
rx(2.9025088989672407*pi) node[36];
rx(2.897583626436342*pi) node[37];
rx(2.8918265465108672*pi) node[38];
rx(2.884973270999353*pi) node[39];
rx(2.9587913707886457*pi) node[40];
rx(2.9580828129820005*pi) node[41];
rx(2.957336439960877*pi) node[42];
rx(2.958441611885707*pi) node[46];
rx(2.959132408000703*pi) node[47];
rx(2.9548327735510984*pi) node[48];
rx(2.9552807310539246*pi) node[49];
rx(2.9557156378514176*pi) node[50];
rx(2.9561380987323607*pi) node[51];
rx(2.956548686654549*pi) node[52];
rx(2.956947974575778*pi) node[53];
rx(2.957714560274675*pi) node[54];
rx(2.937167052332727*pi) node[56];
rx(2.9383566400393732*pi) node[57];
rx(2.9394811333742945*pi) node[58];
rx(2.9405462619154425*pi) node[59];
rx(2.9415570231280306*pi) node[60];
rx(2.9425179370124424*pi) node[61];
rx(2.9434329506112666*pi) node[62];
rx(2.944305628995228*pi) node[63];
rx(2.9168710092008645*pi) node[64];
rx(2.9138813472568605*pi) node[65];
rx(2.9359057812397125*pi) node[70];
rx(2.9345653783089922*pi) node[71];
rx(2.8040867245816834*pi) node[72];
rx(2.75*pi) node[73];
rx(2.9105438044382463*pi) node[78];
rx(2.9067852649641654*pi) node[79];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[28];
rz(0.5*pi) node[29];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[32];
rz(0.5*pi) node[33];
rz(0.5*pi) node[34];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[42];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[50];
rz(0.5*pi) node[51];
rz(0.5*pi) node[52];
rz(0.5*pi) node[53];
rz(0.5*pi) node[54];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[58];
rz(0.5*pi) node[59];
rz(0.5*pi) node[60];
rz(0.5*pi) node[61];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[28];
rz(0.5*pi) node[29];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[32];
rz(0.5*pi) node[33];
rz(0.5*pi) node[34];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[42];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[50];
rz(0.5*pi) node[51];
rz(0.5*pi) node[52];
rz(0.5*pi) node[53];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[58];
rz(0.5*pi) node[59];
rz(0.5*pi) node[60];
rz(0.5*pi) node[61];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[4];
rx(0.5*pi) node[6];
rx(0.5*pi) node[7];
rx(0.5*pi) node[8];
rx(0.5*pi) node[9];
rx(0.5*pi) node[10];
rx(0.5*pi) node[11];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rx(0.5*pi) node[16];
rx(0.5*pi) node[17];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rx(0.5*pi) node[20];
rx(0.5*pi) node[21];
rx(0.5*pi) node[22];
rx(0.5*pi) node[23];
rx(0.5*pi) node[24];
rx(0.5*pi) node[25];
rx(0.5*pi) node[26];
rx(0.5*pi) node[27];
rx(0.5*pi) node[28];
rx(0.5*pi) node[29];
rx(0.5*pi) node[30];
rx(0.5*pi) node[31];
rx(0.5*pi) node[32];
rx(0.5*pi) node[33];
rx(0.5*pi) node[34];
rx(0.5*pi) node[35];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[39];
rx(0.5*pi) node[42];
rx(0.5*pi) node[48];
rx(0.5*pi) node[49];
rx(0.5*pi) node[50];
rx(0.5*pi) node[51];
rx(0.5*pi) node[52];
rx(0.5*pi) node[53];
rx(0.5*pi) node[56];
rx(0.5*pi) node[57];
rx(0.5*pi) node[58];
rx(0.5*pi) node[59];
rx(0.5*pi) node[60];
rx(0.5*pi) node[61];
rx(0.5*pi) node[62];
rx(0.5*pi) node[63];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rx(0.5*pi) node[70];
rx(0.5*pi) node[71];
rx(0.5*pi) node[72];
rx(0.5*pi) node[73];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[28];
rz(0.5*pi) node[29];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[32];
rz(0.5*pi) node[33];
rz(0.5*pi) node[34];
rz(0.5*pi) node[35];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[42];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[50];
rz(0.5*pi) node[51];
rz(0.5*pi) node[52];
rz(0.5*pi) node[53];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[58];
rz(0.5*pi) node[59];
rz(0.5*pi) node[60];
rz(0.5*pi) node[61];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
cz node[5],node[4];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rx(0.5*pi) node[4];
rx(0.5*pi) node[5];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(2.5*pi) node[4];
rx(0.9613040135372146*pi) node[4];
rz(0.5*pi) node[4];
cz node[4],node[3];
rz(0.5*pi) node[3];
cz node[4],node[5];
rx(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[3];
rx(0.5*pi) node[4];
rx(0.5*pi) node[5];
rz(2.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rx(0.9610148608366053*pi) node[3];
rz(0.5*pi) node[5];
rz(0.5*pi) node[3];
rx(0.5*pi) node[5];
cz node[3],node[2];
rz(0.5*pi) node[5];
rz(0.5*pi) node[2];
cz node[3],node[4];
rx(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[4];
rz(2.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rx(0.9607191509523405*pi) node[2];
rz(0.5*pi) node[4];
rz(0.5*pi) node[2];
rx(0.5*pi) node[4];
cz node[2],node[1];
rz(0.5*pi) node[4];
rz(0.5*pi) node[1];
cz node[2],node[3];
cz node[47],node[4];
rx(0.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[47];
rz(0.5*pi) node[1];
rx(0.5*pi) node[2];
rx(0.5*pi) node[3];
rx(0.5*pi) node[4];
rx(0.5*pi) node[47];
rz(2.5*pi) node[1];
rz(0.5*pi) node[2];
rz(0.5*pi) node[3];
rz(0.5*pi) node[4];
rz(0.5*pi) node[47];
rx(0.9604165655745343*pi) node[1];
cz node[4],node[47];
rz(0.5*pi) node[1];
rz(0.5*pi) node[4];
rz(0.5*pi) node[47];
cz node[1],node[0];
rx(0.5*pi) node[4];
rx(0.5*pi) node[47];
rz(0.5*pi) node[0];
cz node[1],node[2];
rz(0.5*pi) node[4];
rz(0.5*pi) node[47];
rx(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
cz node[47],node[4];
rz(0.5*pi) node[0];
rx(0.5*pi) node[1];
rx(0.5*pi) node[2];
rz(0.5*pi) node[4];
rz(0.5*pi) node[47];
rz(2.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[2];
rx(0.5*pi) node[4];
rx(0.5*pi) node[47];
rx(0.9601069137172547*pi) node[0];
rz(0.5*pi) node[4];
rz(0.5*pi) node[47];
rz(0.5*pi) node[0];
cz node[4],node[5];
cz node[40],node[47];
cz node[0],node[7];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[40];
rz(0.5*pi) node[47];
cz node[0],node[1];
rx(0.5*pi) node[4];
rx(0.5*pi) node[5];
rz(0.5*pi) node[7];
rx(0.5*pi) node[40];
rx(0.5*pi) node[47];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rx(0.5*pi) node[7];
rz(0.5*pi) node[40];
rz(0.5*pi) node[47];
rx(0.5*pi) node[0];
rx(0.5*pi) node[1];
cz node[5],node[4];
rz(0.5*pi) node[7];
cz node[47],node[40];
rz(0.5*pi) node[0];
rz(0.5*pi) node[1];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(2.5*pi) node[7];
rz(0.5*pi) node[40];
rz(0.5*pi) node[47];
rx(0.5*pi) node[4];
rx(0.5*pi) node[5];
rx(0.9597898770706157*pi) node[7];
rx(0.5*pi) node[40];
rx(0.5*pi) node[47];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[7];
rz(0.5*pi) node[40];
rz(0.5*pi) node[47];
cz node[4],node[5];
cz node[7],node[6];
cz node[40],node[47];
cz node[7],node[0];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[40];
rz(0.5*pi) node[47];
rz(0.5*pi) node[0];
rx(0.5*pi) node[4];
rx(0.5*pi) node[5];
rx(0.5*pi) node[6];
rz(0.5*pi) node[7];
rx(0.5*pi) node[40];
rx(0.5*pi) node[47];
rx(0.5*pi) node[0];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rx(0.5*pi) node[7];
rz(0.5*pi) node[40];
rz(0.5*pi) node[47];
rz(0.5*pi) node[0];
cz node[47],node[4];
rz(0.5*pi) node[5];
rz(2.5*pi) node[6];
rz(0.5*pi) node[7];
cz node[41],node[40];
rz(0.5*pi) node[4];
rx(0.5*pi) node[5];
rx(0.9594651373247309*pi) node[6];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[47];
rx(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rx(0.5*pi) node[40];
rx(0.5*pi) node[41];
rx(0.5*pi) node[47];
rz(0.5*pi) node[4];
cz node[6],node[5];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[47];
cz node[4],node[47];
rz(0.5*pi) node[5];
cz node[6],node[7];
cz node[40],node[41];
rz(0.5*pi) node[4];
rx(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[47];
rx(0.5*pi) node[4];
rz(0.5*pi) node[5];
rx(0.5*pi) node[6];
rx(0.5*pi) node[7];
rx(0.5*pi) node[40];
rx(0.5*pi) node[41];
rx(0.5*pi) node[47];
rz(0.5*pi) node[4];
rz(2.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[7];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[47];
cz node[47],node[4];
rx(0.959132408000703*pi) node[5];
cz node[41],node[40];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[47];
rx(0.5*pi) node[4];
rx(0.5*pi) node[40];
rx(0.5*pi) node[41];
rx(0.5*pi) node[47];
rz(0.5*pi) node[4];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[47];
rz(0.5*pi) node[4];
rz(0.5*pi) node[40];
cz node[54],node[41];
cz node[46],node[47];
rx(0.5*pi) node[4];
rx(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[54];
rz(0.5*pi) node[4];
rz(0.5*pi) node[40];
rx(0.5*pi) node[41];
rx(0.5*pi) node[46];
rx(0.5*pi) node[47];
rx(0.5*pi) node[54];
cz node[5],node[4];
rz(0.5*pi) node[41];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[54];
rz(0.5*pi) node[4];
cz node[5],node[6];
cz node[41],node[54];
cz node[47],node[46];
rx(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[41];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[54];
rz(0.5*pi) node[4];
rx(0.5*pi) node[5];
rx(0.5*pi) node[6];
rx(0.5*pi) node[41];
rx(0.5*pi) node[46];
rx(0.5*pi) node[47];
rx(0.5*pi) node[54];
rz(2.5*pi) node[4];
rz(0.5*pi) node[5];
rz(0.5*pi) node[6];
rz(0.5*pi) node[41];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[54];
rx(0.9587913707886456*pi) node[4];
cz node[54],node[41];
cz node[46],node[47];
rz(0.5*pi) node[4];
rz(0.5*pi) node[41];
rz(0.5*pi) node[47];
rx(0.5*pi) node[41];
rx(0.5*pi) node[47];
rz(0.5*pi) node[41];
rz(0.5*pi) node[47];
rz(0.5*pi) node[41];
rz(0.5*pi) node[47];
rx(0.5*pi) node[41];
rx(0.5*pi) node[47];
rz(0.5*pi) node[41];
rz(0.5*pi) node[47];
cz node[4],node[47];
cz node[4],node[5];
rz(0.5*pi) node[47];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rx(0.5*pi) node[47];
rx(0.5*pi) node[4];
rx(0.5*pi) node[5];
rz(0.5*pi) node[47];
rz(0.5*pi) node[4];
rz(0.5*pi) node[5];
rz(2.5*pi) node[47];
rx(0.9584416118857069*pi) node[47];
rz(0.5*pi) node[47];
cz node[47],node[40];
cz node[47],node[4];
rz(0.5*pi) node[40];
rz(0.5*pi) node[4];
rx(0.5*pi) node[40];
rz(0.5*pi) node[47];
rx(0.5*pi) node[4];
rz(0.5*pi) node[40];
rx(0.5*pi) node[47];
rz(0.5*pi) node[4];
rz(2.5*pi) node[40];
rz(0.5*pi) node[47];
rx(0.9580828129820006*pi) node[40];
rz(0.5*pi) node[40];
cz node[40],node[41];
cz node[40],node[47];
rz(0.5*pi) node[41];
rz(0.5*pi) node[40];
rx(0.5*pi) node[41];
rz(0.5*pi) node[47];
rx(0.5*pi) node[40];
rz(0.5*pi) node[41];
rx(0.5*pi) node[47];
rz(0.5*pi) node[40];
rz(2.5*pi) node[41];
rz(0.5*pi) node[47];
rx(0.9577145602746746*pi) node[41];
rz(0.5*pi) node[41];
cz node[41],node[42];
cz node[41],node[40];
rz(0.5*pi) node[42];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rx(0.5*pi) node[42];
rx(0.5*pi) node[40];
rx(0.5*pi) node[41];
rz(0.5*pi) node[42];
rz(0.5*pi) node[40];
rz(0.5*pi) node[41];
rz(2.5*pi) node[42];
rx(0.9573364399608768*pi) node[42];
rz(0.5*pi) node[42];
cz node[42],node[53];
cz node[42],node[41];
rz(0.5*pi) node[53];
rz(0.5*pi) node[41];
rz(0.5*pi) node[42];
rx(0.5*pi) node[53];
rx(0.5*pi) node[41];
rx(0.5*pi) node[42];
rz(0.5*pi) node[53];
rz(0.5*pi) node[41];
rz(0.5*pi) node[42];
rz(2.5*pi) node[53];
rx(0.9569479745757781*pi) node[53];
rz(0.5*pi) node[53];
cz node[53],node[52];
cz node[53],node[42];
rz(0.5*pi) node[52];
rz(0.5*pi) node[42];
rx(0.5*pi) node[52];
rz(0.5*pi) node[53];
rx(0.5*pi) node[42];
rz(0.5*pi) node[52];
rx(0.5*pi) node[53];
rz(0.5*pi) node[42];
rz(2.5*pi) node[52];
rz(0.5*pi) node[53];
rx(0.9565486866545492*pi) node[52];
rz(0.5*pi) node[52];
cz node[52],node[51];
rz(0.5*pi) node[51];
cz node[52],node[53];
rx(0.5*pi) node[51];
rz(0.5*pi) node[52];
rz(0.5*pi) node[53];
rz(0.5*pi) node[51];
rx(0.5*pi) node[52];
rx(0.5*pi) node[53];
rz(2.5*pi) node[51];
rz(0.5*pi) node[52];
rz(0.5*pi) node[53];
rx(0.9561380987323607*pi) node[51];
rz(0.5*pi) node[51];
cz node[51],node[50];
rz(0.5*pi) node[50];
cz node[51],node[52];
rx(0.5*pi) node[50];
rz(0.5*pi) node[51];
rz(0.5*pi) node[52];
rz(0.5*pi) node[50];
rx(0.5*pi) node[51];
rx(0.5*pi) node[52];
rz(2.5*pi) node[50];
rz(0.5*pi) node[51];
rz(0.5*pi) node[52];
rx(0.9557156378514176*pi) node[50];
rz(0.5*pi) node[50];
cz node[50],node[49];
rz(0.5*pi) node[49];
cz node[50],node[51];
rx(0.5*pi) node[49];
rz(0.5*pi) node[50];
rz(0.5*pi) node[51];
rz(0.5*pi) node[49];
rx(0.5*pi) node[50];
rx(0.5*pi) node[51];
rz(2.5*pi) node[49];
rz(0.5*pi) node[50];
rz(0.5*pi) node[51];
rx(0.9552807310539246*pi) node[49];
rz(0.5*pi) node[49];
cz node[49],node[48];
rz(0.5*pi) node[48];
cz node[49],node[50];
rx(0.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[50];
rz(0.5*pi) node[48];
rx(0.5*pi) node[49];
rx(0.5*pi) node[50];
rz(2.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[50];
rx(0.9548327735510982*pi) node[48];
rz(0.5*pi) node[48];
cz node[48],node[11];
rz(0.5*pi) node[11];
cz node[48],node[49];
rx(0.5*pi) node[11];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[11];
rx(0.5*pi) node[48];
rx(0.5*pi) node[49];
rz(2.5*pi) node[11];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rx(0.9543710650611886*pi) node[11];
rz(0.5*pi) node[11];
cz node[11],node[12];
cz node[11],node[48];
rz(0.5*pi) node[12];
rz(0.5*pi) node[11];
rx(0.5*pi) node[12];
rz(0.5*pi) node[48];
rx(0.5*pi) node[11];
rz(0.5*pi) node[12];
rx(0.5*pi) node[48];
rz(0.5*pi) node[11];
rz(2.5*pi) node[12];
rz(0.5*pi) node[48];
rx(0.9538948734714576*pi) node[12];
rz(0.5*pi) node[12];
cz node[12],node[13];
cz node[12],node[11];
rz(0.5*pi) node[13];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rx(0.5*pi) node[13];
rx(0.5*pi) node[11];
rx(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[11];
rz(0.5*pi) node[12];
rz(2.5*pi) node[13];
rx(0.9534034985001558*pi) node[13];
rz(0.5*pi) node[13];
cz node[13],node[14];
cz node[13],node[12];
rz(0.5*pi) node[14];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rx(0.5*pi) node[14];
rx(0.5*pi) node[12];
rx(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(2.5*pi) node[14];
rx(0.9528960488796016*pi) node[14];
rz(0.5*pi) node[14];
cz node[14],node[15];
cz node[14],node[13];
rz(0.5*pi) node[15];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rx(0.5*pi) node[15];
rx(0.5*pi) node[13];
rx(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[13];
rz(0.5*pi) node[14];
rz(2.5*pi) node[15];
rx(0.9523716333421137*pi) node[15];
rz(0.5*pi) node[15];
cz node[15],node[8];
rz(0.5*pi) node[8];
cz node[15],node[14];
rx(0.5*pi) node[8];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rz(0.5*pi) node[8];
rx(0.5*pi) node[14];
rx(0.5*pi) node[15];
rz(2.5*pi) node[8];
rz(0.5*pi) node[14];
rz(0.5*pi) node[15];
rx(0.9518293287890225*pi) node[8];
rz(0.5*pi) node[8];
cz node[8],node[9];
cz node[8],node[15];
rz(0.5*pi) node[9];
rz(0.5*pi) node[8];
rx(0.5*pi) node[9];
rz(0.5*pi) node[15];
rx(0.5*pi) node[8];
rz(0.5*pi) node[9];
rx(0.5*pi) node[15];
rz(0.5*pi) node[8];
rz(2.5*pi) node[9];
rz(0.5*pi) node[15];
rx(0.9512680529667146*pi) node[9];
rz(0.5*pi) node[9];
cz node[9],node[10];
cz node[9],node[8];
rz(0.5*pi) node[10];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rx(0.5*pi) node[10];
rx(0.5*pi) node[8];
rx(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(0.5*pi) node[8];
rz(0.5*pi) node[9];
rz(2.5*pi) node[10];
rx(0.9506866917905885*pi) node[10];
rz(0.5*pi) node[10];
cz node[10],node[21];
cz node[10],node[9];
rz(0.5*pi) node[21];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rx(0.5*pi) node[21];
rx(0.5*pi) node[9];
rx(0.5*pi) node[10];
rz(0.5*pi) node[21];
rz(0.5*pi) node[9];
rz(0.5*pi) node[10];
rz(2.5*pi) node[21];
rx(0.950084003852088*pi) node[21];
rz(0.5*pi) node[21];
cz node[21],node[22];
cz node[21],node[10];
rz(0.5*pi) node[22];
rz(0.5*pi) node[10];
rz(0.5*pi) node[21];
rx(0.5*pi) node[22];
rx(0.5*pi) node[10];
rx(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(0.5*pi) node[10];
rz(0.5*pi) node[21];
rz(2.5*pi) node[22];
rx(0.94945868408068*pi) node[22];
rz(0.5*pi) node[22];
cz node[22],node[23];
cz node[22],node[21];
rz(0.5*pi) node[23];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rx(0.5*pi) node[23];
rx(0.5*pi) node[21];
rx(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[21];
rz(0.5*pi) node[22];
rz(2.5*pi) node[23];
rx(0.9488092364198992*pi) node[23];
rz(0.5*pi) node[23];
cz node[23],node[16];
rz(0.5*pi) node[16];
cz node[23],node[22];
rx(0.5*pi) node[16];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rz(0.5*pi) node[16];
rx(0.5*pi) node[22];
rx(0.5*pi) node[23];
rz(2.5*pi) node[16];
rz(0.5*pi) node[22];
rz(0.5*pi) node[23];
rx(0.9481341011513034*pi) node[16];
rz(0.5*pi) node[16];
cz node[16],node[17];
cz node[16],node[23];
rz(0.5*pi) node[17];
rz(0.5*pi) node[16];
rx(0.5*pi) node[17];
rz(0.5*pi) node[23];
rx(0.5*pi) node[16];
rz(0.5*pi) node[17];
rx(0.5*pi) node[23];
rz(0.5*pi) node[16];
rz(2.5*pi) node[17];
rz(0.5*pi) node[23];
rx(0.9474315275705185*pi) node[17];
rz(0.5*pi) node[17];
cz node[17],node[18];
cz node[17],node[16];
rz(0.5*pi) node[18];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rx(0.5*pi) node[18];
rx(0.5*pi) node[16];
rx(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[16];
rz(0.5*pi) node[17];
rz(2.5*pi) node[18];
rx(0.9466996058182275*pi) node[18];
rz(0.5*pi) node[18];
cz node[18],node[19];
cz node[18],node[17];
rz(0.5*pi) node[19];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rx(0.5*pi) node[19];
rx(0.5*pi) node[17];
rx(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(2.5*pi) node[19];
rx(0.9459362350491816*pi) node[19];
rz(0.5*pi) node[19];
cz node[19],node[20];
cz node[19],node[18];
rz(0.5*pi) node[20];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rx(0.5*pi) node[20];
rx(0.5*pi) node[18];
rx(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(0.5*pi) node[18];
rz(0.5*pi) node[19];
rz(2.5*pi) node[20];
rx(0.9451390916012115*pi) node[20];
rz(0.5*pi) node[20];
cz node[20],node[63];
cz node[20],node[19];
rz(0.5*pi) node[63];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rx(0.5*pi) node[63];
rx(0.5*pi) node[19];
rx(0.5*pi) node[20];
rz(0.5*pi) node[63];
rz(0.5*pi) node[19];
rz(0.5*pi) node[20];
rz(2.5*pi) node[63];
rx(0.9443056289952279*pi) node[63];
rz(0.5*pi) node[63];
cz node[63],node[62];
cz node[63],node[20];
rz(0.5*pi) node[62];
rz(0.5*pi) node[20];
rx(0.5*pi) node[62];
rz(0.5*pi) node[63];
rx(0.5*pi) node[20];
rz(0.5*pi) node[62];
rx(0.5*pi) node[63];
rz(0.5*pi) node[20];
rz(2.5*pi) node[62];
rz(0.5*pi) node[63];
rx(0.9434329506112664*pi) node[62];
rz(0.5*pi) node[62];
cz node[62],node[61];
rz(0.5*pi) node[61];
cz node[62],node[63];
rx(0.5*pi) node[61];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rz(0.5*pi) node[61];
rx(0.5*pi) node[62];
rx(0.5*pi) node[63];
rz(2.5*pi) node[61];
rz(0.5*pi) node[62];
rz(0.5*pi) node[63];
rx(0.9425179370124424*pi) node[61];
rz(0.5*pi) node[61];
cz node[61],node[60];
rz(0.5*pi) node[60];
cz node[61],node[62];
rx(0.5*pi) node[60];
rz(0.5*pi) node[61];
rz(0.5*pi) node[62];
rz(0.5*pi) node[60];
rx(0.5*pi) node[61];
rx(0.5*pi) node[62];
rz(2.5*pi) node[60];
rz(0.5*pi) node[61];
rz(0.5*pi) node[62];
rx(0.9415570231280308*pi) node[60];
rz(0.5*pi) node[60];
cz node[60],node[59];
rz(0.5*pi) node[59];
cz node[60],node[61];
rx(0.5*pi) node[59];
rz(0.5*pi) node[60];
rz(0.5*pi) node[61];
rz(0.5*pi) node[59];
rx(0.5*pi) node[60];
rx(0.5*pi) node[61];
rz(2.5*pi) node[59];
rz(0.5*pi) node[60];
rz(0.5*pi) node[61];
rx(0.9405462619154428*pi) node[59];
rz(0.5*pi) node[59];
cz node[59],node[58];
rz(0.5*pi) node[58];
cz node[59],node[60];
rx(0.5*pi) node[58];
rz(0.5*pi) node[59];
rz(0.5*pi) node[60];
rz(0.5*pi) node[58];
rx(0.5*pi) node[59];
rx(0.5*pi) node[60];
rz(2.5*pi) node[58];
rz(0.5*pi) node[59];
rz(0.5*pi) node[60];
rx(0.9394811333742946*pi) node[58];
rz(0.5*pi) node[58];
cz node[58],node[57];
rz(0.5*pi) node[57];
cz node[58],node[59];
rx(0.5*pi) node[57];
rz(0.5*pi) node[58];
rz(0.5*pi) node[59];
rz(0.5*pi) node[57];
rx(0.5*pi) node[58];
rx(0.5*pi) node[59];
rz(2.5*pi) node[57];
rz(0.5*pi) node[58];
rz(0.5*pi) node[59];
rx(0.9383566400393731*pi) node[57];
rz(0.5*pi) node[57];
cz node[57],node[56];
rz(0.5*pi) node[56];
cz node[57],node[58];
rx(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[58];
rz(0.5*pi) node[56];
rx(0.5*pi) node[57];
rx(0.5*pi) node[58];
rz(2.5*pi) node[56];
rz(0.5*pi) node[57];
rz(0.5*pi) node[58];
rx(0.9371670523327271*pi) node[56];
rz(0.5*pi) node[56];
cz node[56],node[57];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rx(0.5*pi) node[56];
rx(0.5*pi) node[57];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
cz node[57],node[56];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rx(0.5*pi) node[56];
rx(0.5*pi) node[57];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
cz node[56],node[57];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rx(0.5*pi) node[56];
rx(0.5*pi) node[57];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
cz node[57],node[70];
cz node[57],node[56];
rz(0.5*pi) node[70];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rx(0.5*pi) node[70];
rx(0.5*pi) node[56];
rx(0.5*pi) node[57];
rz(0.5*pi) node[70];
rz(0.5*pi) node[56];
rz(0.5*pi) node[57];
rz(2.5*pi) node[70];
rx(0.9359057812397125*pi) node[70];
rz(0.5*pi) node[70];
cz node[70],node[71];
cz node[70],node[57];
rz(0.5*pi) node[71];
rz(0.5*pi) node[57];
rz(0.5*pi) node[70];
rx(0.5*pi) node[71];
rx(0.5*pi) node[57];
rx(0.5*pi) node[70];
rz(0.5*pi) node[71];
rz(0.5*pi) node[57];
rz(0.5*pi) node[70];
rz(2.5*pi) node[71];
rx(0.9345653783089924*pi) node[71];
rz(0.5*pi) node[71];
cz node[71],node[28];
rz(0.5*pi) node[28];
cz node[71],node[70];
rx(0.5*pi) node[28];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
rz(0.5*pi) node[28];
rx(0.5*pi) node[70];
rx(0.5*pi) node[71];
rz(2.5*pi) node[28];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
rx(0.933137185511663*pi) node[28];
rz(0.5*pi) node[28];
cz node[28],node[29];
cz node[28],node[71];
rz(0.5*pi) node[29];
rz(0.5*pi) node[28];
rx(0.5*pi) node[29];
rz(0.5*pi) node[71];
rx(0.5*pi) node[28];
rz(0.5*pi) node[29];
rx(0.5*pi) node[71];
rz(0.5*pi) node[28];
rz(2.5*pi) node[29];
rz(0.5*pi) node[71];
rx(0.9316111760863093*pi) node[29];
rz(0.5*pi) node[29];
cz node[29],node[30];
cz node[29],node[28];
rz(0.5*pi) node[30];
rz(0.5*pi) node[28];
rz(0.5*pi) node[29];
rx(0.5*pi) node[30];
rx(0.5*pi) node[28];
rx(0.5*pi) node[29];
rz(0.5*pi) node[30];
rz(0.5*pi) node[28];
rz(0.5*pi) node[29];
rz(2.5*pi) node[30];
rx(0.9299756362291198*pi) node[30];
rz(0.5*pi) node[30];
cz node[30],node[31];
cz node[30],node[29];
rz(0.5*pi) node[31];
rz(0.5*pi) node[29];
rz(0.5*pi) node[30];
rx(0.5*pi) node[31];
rx(0.5*pi) node[29];
rx(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[29];
rz(0.5*pi) node[30];
rz(2.5*pi) node[31];
rx(0.9282168467839998*pi) node[31];
rz(0.5*pi) node[31];
cz node[31],node[24];
rz(0.5*pi) node[24];
cz node[31],node[30];
rx(0.5*pi) node[24];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rz(0.5*pi) node[24];
rx(0.5*pi) node[30];
rx(0.5*pi) node[31];
rz(2.5*pi) node[24];
rz(0.5*pi) node[30];
rz(0.5*pi) node[31];
rx(0.9263184784537883*pi) node[24];
rz(0.5*pi) node[24];
cz node[24],node[25];
cz node[24],node[31];
rz(0.5*pi) node[25];
rz(0.5*pi) node[24];
rx(0.5*pi) node[25];
rz(0.5*pi) node[31];
rx(0.5*pi) node[24];
rz(0.5*pi) node[25];
rx(0.5*pi) node[31];
rz(0.5*pi) node[24];
rz(2.5*pi) node[25];
rz(0.5*pi) node[31];
rx(0.9242609870114735*pi) node[25];
rz(0.5*pi) node[25];
cz node[25],node[26];
cz node[25],node[24];
rz(0.5*pi) node[26];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rx(0.5*pi) node[26];
rx(0.5*pi) node[24];
rx(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
rz(2.5*pi) node[26];
rx(0.9220208811874551*pi) node[26];
rz(0.5*pi) node[26];
cz node[26],node[27];
cz node[26],node[25];
rz(0.5*pi) node[27];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rx(0.5*pi) node[27];
rx(0.5*pi) node[25];
rx(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(0.5*pi) node[25];
rz(0.5*pi) node[26];
rz(2.5*pi) node[27];
rx(0.919569385768022*pi) node[27];
rz(0.5*pi) node[27];
cz node[27],node[64];
cz node[27],node[26];
rz(0.5*pi) node[64];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rx(0.5*pi) node[64];
rx(0.5*pi) node[26];
rx(0.5*pi) node[27];
rz(0.5*pi) node[64];
rz(0.5*pi) node[26];
rz(0.5*pi) node[27];
rz(2.5*pi) node[64];
rx(0.9168710092008648*pi) node[64];
rz(0.5*pi) node[64];
cz node[64],node[65];
cz node[64],node[27];
rz(0.5*pi) node[65];
rz(0.5*pi) node[27];
rz(0.5*pi) node[64];
rx(0.5*pi) node[65];
rx(0.5*pi) node[27];
rx(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(0.5*pi) node[27];
rz(0.5*pi) node[64];
rz(2.5*pi) node[65];
rx(0.9138813472568608*pi) node[65];
rz(0.5*pi) node[65];
cz node[65],node[78];
cz node[65],node[64];
rz(0.5*pi) node[78];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rx(0.5*pi) node[78];
rx(0.5*pi) node[64];
rx(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(0.5*pi) node[64];
rz(0.5*pi) node[65];
rz(2.5*pi) node[78];
rx(0.9105438044382466*pi) node[78];
rz(0.5*pi) node[78];
cz node[78],node[79];
cz node[78],node[65];
rz(0.5*pi) node[79];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rx(0.5*pi) node[79];
rx(0.5*pi) node[65];
rx(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[65];
rz(0.5*pi) node[78];
rz(2.5*pi) node[79];
rx(0.9067852649641656*pi) node[79];
rz(0.5*pi) node[79];
cz node[79],node[36];
rz(0.5*pi) node[36];
cz node[79],node[78];
rx(0.5*pi) node[36];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[36];
rx(0.5*pi) node[78];
rx(0.5*pi) node[79];
rz(2.5*pi) node[36];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rx(0.9025088989672407*pi) node[36];
rz(0.5*pi) node[36];
cz node[36],node[37];
cz node[36],node[79];
rz(0.5*pi) node[37];
rz(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[79];
rx(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[79];
rz(0.5*pi) node[36];
rz(2.5*pi) node[37];
rz(0.5*pi) node[79];
rx(0.8975836264363417*pi) node[37];
rz(0.5*pi) node[37];
cz node[37],node[38];
cz node[37],node[36];
rz(0.5*pi) node[38];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rx(0.5*pi) node[38];
rx(0.5*pi) node[36];
rx(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(0.5*pi) node[36];
rz(0.5*pi) node[37];
rz(2.5*pi) node[38];
rx(0.8918265465108672*pi) node[38];
rz(0.5*pi) node[38];
cz node[38],node[39];
cz node[38],node[37];
rz(0.5*pi) node[39];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rx(0.5*pi) node[39];
rx(0.5*pi) node[37];
rx(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[37];
rz(0.5*pi) node[38];
rz(2.5*pi) node[39];
rx(0.884973270999353*pi) node[39];
rz(0.5*pi) node[39];
cz node[39],node[32];
rz(0.5*pi) node[32];
cz node[39],node[38];
rx(0.5*pi) node[32];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rz(0.5*pi) node[32];
rx(0.5*pi) node[38];
rx(0.5*pi) node[39];
rz(2.5*pi) node[32];
rz(0.5*pi) node[38];
rz(0.5*pi) node[39];
rx(0.8766241300087068*pi) node[32];
rz(0.5*pi) node[32];
cz node[32],node[33];
cz node[32],node[39];
rz(0.5*pi) node[33];
rz(0.5*pi) node[32];
rx(0.5*pi) node[33];
rz(0.5*pi) node[39];
rx(0.5*pi) node[32];
rz(0.5*pi) node[33];
rx(0.5*pi) node[39];
rz(0.5*pi) node[32];
rz(2.5*pi) node[33];
rz(0.5*pi) node[39];
rx(0.8661397663015394*pi) node[33];
rz(0.5*pi) node[33];
cz node[33],node[34];
cz node[33],node[32];
rz(0.5*pi) node[34];
rz(0.5*pi) node[32];
rz(0.5*pi) node[33];
rx(0.5*pi) node[34];
rx(0.5*pi) node[32];
rx(0.5*pi) node[33];
rz(0.5*pi) node[34];
rz(0.5*pi) node[32];
rz(0.5*pi) node[33];
rz(2.5*pi) node[34];
rx(0.8524163766855318*pi) node[34];
rz(0.5*pi) node[34];
cz node[34],node[35];
cz node[34],node[33];
rz(0.5*pi) node[35];
rz(0.5*pi) node[33];
rz(0.5*pi) node[34];
rx(0.5*pi) node[35];
rx(0.5*pi) node[33];
rx(0.5*pi) node[34];
rz(0.5*pi) node[35];
rz(0.5*pi) node[33];
rz(0.5*pi) node[34];
rz(2.5*pi) node[35];
rx(0.8333333333333341*pi) node[35];
rz(0.5*pi) node[35];
cz node[35],node[72];
cz node[35],node[34];
rz(0.5*pi) node[72];
rz(0.5*pi) node[34];
rz(0.5*pi) node[35];
rx(0.5*pi) node[72];
rx(0.5*pi) node[34];
rx(0.5*pi) node[35];
rz(0.5*pi) node[72];
rz(0.5*pi) node[34];
rz(0.5*pi) node[35];
rz(2.5*pi) node[72];
rx(0.8040867245816836*pi) node[72];
rz(0.5*pi) node[72];
cz node[72],node[73];
cz node[72],node[35];
rz(0.5*pi) node[73];
rz(0.5*pi) node[35];
rz(0.5*pi) node[72];
rx(0.5*pi) node[73];
rx(0.5*pi) node[35];
rx(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[35];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rx(0.75*pi) node[73];
rz(0.5*pi) node[73];
cz node[73],node[72];
rz(0.5*pi) node[72];
rx(0.5*pi) node[72];
rz(0.5*pi) node[72];
barrier node[73],node[72],node[35],node[34],node[33],node[32],node[39],node[38],node[37],node[36],node[79],node[78],node[65],node[64],node[27],node[26],node[25],node[24],node[31],node[30],node[29],node[28],node[71],node[70],node[57],node[56],node[58],node[59],node[60],node[61],node[62],node[63],node[20],node[19],node[18],node[17],node[16],node[23],node[22],node[21],node[10],node[9],node[8],node[15],node[14],node[13],node[12],node[11],node[48],node[49],node[50],node[51],node[52],node[53],node[42],node[41],node[40],node[47],node[4],node[5],node[6],node[7],node[0],node[1],node[2],node[3],node[54],node[46];
measure node[73] -> meas[0];
measure node[72] -> meas[1];
measure node[35] -> meas[2];
measure node[34] -> meas[3];
measure node[33] -> meas[4];
measure node[32] -> meas[5];
measure node[39] -> meas[6];
measure node[38] -> meas[7];
measure node[37] -> meas[8];
measure node[36] -> meas[9];
measure node[79] -> meas[10];
measure node[78] -> meas[11];
measure node[65] -> meas[12];
measure node[64] -> meas[13];
measure node[27] -> meas[14];
measure node[26] -> meas[15];
measure node[25] -> meas[16];
measure node[24] -> meas[17];
measure node[31] -> meas[18];
measure node[30] -> meas[19];
measure node[29] -> meas[20];
measure node[28] -> meas[21];
measure node[71] -> meas[22];
measure node[70] -> meas[23];
measure node[57] -> meas[24];
measure node[56] -> meas[25];
measure node[58] -> meas[26];
measure node[59] -> meas[27];
measure node[60] -> meas[28];
measure node[61] -> meas[29];
measure node[62] -> meas[30];
measure node[63] -> meas[31];
measure node[20] -> meas[32];
measure node[19] -> meas[33];
measure node[18] -> meas[34];
measure node[17] -> meas[35];
measure node[16] -> meas[36];
measure node[23] -> meas[37];
measure node[22] -> meas[38];
measure node[21] -> meas[39];
measure node[10] -> meas[40];
measure node[9] -> meas[41];
measure node[8] -> meas[42];
measure node[15] -> meas[43];
measure node[14] -> meas[44];
measure node[13] -> meas[45];
measure node[12] -> meas[46];
measure node[11] -> meas[47];
measure node[48] -> meas[48];
measure node[49] -> meas[49];
measure node[50] -> meas[50];
measure node[51] -> meas[51];
measure node[52] -> meas[52];
measure node[53] -> meas[53];
measure node[42] -> meas[54];
measure node[41] -> meas[55];
measure node[40] -> meas[56];
measure node[47] -> meas[57];
measure node[4] -> meas[58];
measure node[5] -> meas[59];
measure node[6] -> meas[60];
measure node[7] -> meas[61];
measure node[0] -> meas[62];
measure node[1] -> meas[63];
measure node[2] -> meas[64];
measure node[3] -> meas[65];
measure node[54] -> meas[66];
measure node[46] -> meas[67];
