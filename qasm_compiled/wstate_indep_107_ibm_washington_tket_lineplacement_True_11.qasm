OPENQASM 2.0;
include "qelib1.inc";

qreg node[127];
creg meas[107];
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
rz(0.5*pi) node[43];
rz(0.5*pi) node[44];
rz(0.5*pi) node[45];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[50];
rz(0.5*pi) node[51];
rz(0.5*pi) node[52];
rz(0.5*pi) node[53];
rz(0.5*pi) node[54];
rz(0.5*pi) node[55];
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
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[68];
rz(0.5*pi) node[69];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[80];
rz(0.5*pi) node[81];
rz(0.5*pi) node[82];
rz(0.5*pi) node[83];
rz(0.5*pi) node[84];
rz(0.5*pi) node[85];
rz(0.5*pi) node[86];
rz(0.5*pi) node[87];
rz(0.5*pi) node[89];
rz(0.5*pi) node[91];
rz(0.5*pi) node[92];
rz(0.5*pi) node[93];
rz(0.5*pi) node[98];
rz(0.5*pi) node[99];
rz(0.5*pi) node[100];
rz(0.5*pi) node[101];
rz(0.5*pi) node[102];
rz(0.5*pi) node[104];
rz(0.5*pi) node[105];
rz(0.5*pi) node[106];
x node[107];
rz(0.5*pi) node[108];
rz(0.5*pi) node[111];
rz(0.5*pi) node[112];
rz(0.5*pi) node[122];
rz(0.5*pi) node[123];
rz(0.5*pi) node[124];
rz(0.5*pi) node[125];
rz(0.5*pi) node[126];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
sx node[7];
sx node[8];
sx node[9];
sx node[10];
sx node[11];
sx node[12];
sx node[13];
sx node[14];
sx node[15];
sx node[16];
sx node[17];
sx node[18];
sx node[19];
sx node[20];
sx node[21];
sx node[22];
sx node[23];
sx node[24];
sx node[25];
sx node[26];
sx node[27];
sx node[28];
sx node[29];
sx node[30];
sx node[31];
sx node[32];
sx node[33];
sx node[34];
sx node[35];
sx node[36];
sx node[37];
sx node[38];
sx node[39];
sx node[40];
sx node[41];
sx node[42];
sx node[43];
sx node[44];
sx node[45];
sx node[46];
sx node[47];
sx node[48];
sx node[49];
sx node[50];
sx node[51];
sx node[52];
sx node[53];
sx node[54];
sx node[55];
sx node[56];
sx node[57];
sx node[58];
sx node[59];
sx node[60];
sx node[61];
sx node[62];
sx node[63];
sx node[64];
sx node[65];
sx node[66];
sx node[67];
sx node[68];
sx node[69];
sx node[70];
sx node[71];
sx node[72];
sx node[73];
sx node[74];
sx node[77];
sx node[78];
sx node[79];
sx node[80];
sx node[81];
sx node[82];
sx node[83];
sx node[84];
sx node[85];
sx node[86];
sx node[87];
sx node[89];
sx node[91];
sx node[92];
sx node[93];
sx node[98];
sx node[99];
sx node[100];
sx node[101];
sx node[102];
sx node[104];
sx node[105];
sx node[106];
sx node[108];
sx node[111];
sx node[112];
sx node[122];
sx node[123];
sx node[124];
sx node[125];
sx node[126];
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
rz(0.5*pi) node[43];
rz(0.5*pi) node[44];
rz(0.5*pi) node[45];
rz(0.5*pi) node[46];
rz(0.5*pi) node[47];
rz(0.5*pi) node[48];
rz(0.5*pi) node[49];
rz(0.5*pi) node[50];
rz(0.5*pi) node[51];
rz(0.5*pi) node[52];
rz(0.5*pi) node[53];
rz(0.5*pi) node[54];
rz(0.5*pi) node[55];
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
rz(0.5*pi) node[66];
rz(0.5*pi) node[67];
rz(0.5*pi) node[68];
rz(0.5*pi) node[69];
rz(0.5*pi) node[70];
rz(0.5*pi) node[71];
rz(0.5*pi) node[72];
rz(0.5*pi) node[73];
rz(0.5*pi) node[74];
rz(0.5*pi) node[77];
rz(0.5*pi) node[78];
rz(0.5*pi) node[79];
rz(0.5*pi) node[80];
rz(0.5*pi) node[81];
rz(0.5*pi) node[82];
rz(0.5*pi) node[83];
rz(0.5*pi) node[84];
rz(0.5*pi) node[85];
rz(0.5*pi) node[86];
rz(0.5*pi) node[87];
rz(0.5*pi) node[89];
rz(0.5*pi) node[91];
rz(0.5*pi) node[92];
rz(0.5*pi) node[93];
rz(0.5*pi) node[98];
rz(0.5*pi) node[99];
rz(0.5*pi) node[100];
rz(0.5*pi) node[101];
rz(0.5*pi) node[102];
rz(0.5*pi) node[104];
rz(0.5*pi) node[105];
rz(0.5*pi) node[106];
rz(0.5*pi) node[108];
rz(0.5*pi) node[111];
rz(0.5*pi) node[112];
rz(0.5*pi) node[122];
rz(0.5*pi) node[123];
rz(0.5*pi) node[124];
rz(0.5*pi) node[125];
rz(0.5*pi) node[126];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
sx node[5];
sx node[6];
sx node[7];
sx node[8];
sx node[9];
sx node[10];
sx node[11];
sx node[12];
sx node[13];
sx node[14];
sx node[15];
sx node[16];
sx node[17];
sx node[18];
sx node[19];
sx node[20];
sx node[21];
sx node[22];
sx node[23];
sx node[24];
sx node[25];
sx node[26];
sx node[27];
sx node[28];
sx node[29];
sx node[30];
sx node[31];
sx node[32];
sx node[33];
sx node[34];
sx node[35];
sx node[36];
sx node[37];
sx node[38];
sx node[39];
sx node[40];
sx node[41];
sx node[42];
sx node[43];
sx node[44];
sx node[45];
sx node[46];
sx node[47];
sx node[48];
sx node[49];
sx node[50];
sx node[51];
sx node[52];
sx node[53];
sx node[54];
sx node[55];
sx node[56];
sx node[57];
sx node[58];
sx node[59];
sx node[60];
sx node[61];
sx node[62];
sx node[63];
sx node[64];
sx node[65];
sx node[66];
sx node[67];
sx node[68];
sx node[69];
sx node[70];
sx node[71];
sx node[72];
sx node[73];
sx node[74];
sx node[77];
sx node[78];
sx node[79];
sx node[80];
sx node[81];
sx node[82];
sx node[83];
sx node[84];
sx node[85];
sx node[86];
sx node[87];
sx node[89];
sx node[91];
sx node[92];
sx node[93];
sx node[98];
sx node[99];
sx node[100];
sx node[101];
sx node[102];
sx node[104];
sx node[105];
sx node[106];
sx node[108];
sx node[111];
sx node[112];
sx node[122];
sx node[123];
sx node[124];
sx node[125];
sx node[126];
rz(0.46316256130066447*pi) node[0];
rz(0.4634068004763334*pi) node[1];
rz(0.4636462331727209*pi) node[2];
rz(0.4638810503757588*pi) node[3];
rz(0.46411134757841255*pi) node[4];
rz(0.44593623504918156*pi) node[5];
rz(0.43590578123971224*pi) node[6];
rz(0.4371670523327271*pi) node[7];
rz(0.43948113337429473*pi) node[8];
rz(0.41687100920086506*pi) node[9];
rz(0.41956938576802205*pi) node[10];
rz(0.4242609870114735*pi) node[11];
rz(0.4263184784537881*pi) node[12];
rz(0.4220208811874552*pi) node[13];
rz(0.4629133883217601*pi) node[14];
rz(0.4643373157666145*pi) node[15];
rz(0.4415570231280308*pi) node[16];
rz(0.42997563622911983*pi) node[17];
rz(0.46265905872269897*pi) node[18];
rz(0.46239944517952736*pi) node[19];
rz(0.462134324875325*pi) node[20];
rz(0.46186350682416*pi) node[21];
rz(0.4645590504333301*pi) node[22];
rz(0.4647767107335026*pi) node[23];
rz(0.4649904239910867*pi) node[24];
rz(0.44513909160121146*pi) node[25];
rz(0.4434329506112663*pi) node[26];
rz(0.44054626191544266*pi) node[27];
rz(0.43835664003937325*pi) node[28];
rz(0.43456537830899256*pi) node[29];
rz(0.4316111760863093*pi) node[30];
rz(0.4282168467839995*pi) node[31];
rz(0.41388134725686065*pi) node[32];
rz(0.4615868318710892*pi) node[33];
rz(0.4652002538680591*pi) node[34];
rz(0.43313718551166325*pi) node[35];
rz(0.41054380443824645*pi) node[36];
rz(0.44743152757051863*pi) node[37];
rz(0.44669960581822754*pi) node[38];
rz(0.46130401353721473*pi) node[39];
rz(0.4610148608366049*pi) node[40];
rz(0.46071915095234073*pi) node[41];
rz(0.4604165655745345*pi) node[42];
rz(0.46540639135035156*pi) node[43];
rz(0.46560890009994194*pi) node[44];
rz(0.4658078756097952*pi) node[45];
rz(0.4443056289952281*pi) node[46];
rz(0.44251793701244235*pi) node[47];
rz(0.4025088989672406*pi) node[48];
rz(0.3975836264363418*pi) node[49];
rz(0.39182654651086735*pi) node[50];
rz(0.40678526496416534*pi) node[51];
rz(0.44813410115130337*pi) node[52];
rz(0.46010691371725465*pi) node[53];
rz(0.4660034452038664*pi) node[54];
rz(0.38497327099935286*pi) node[55];
rz(0.44880923641989934*pi) node[56];
rz(0.44945868408068*pi) node[57];
rz(0.4506866917905885*pi) node[58];
rz(0.450084003852088*pi) node[59];
rz(0.4597898770706158*pi) node[60];
rz(0.45946513732473093*pi) node[61];
rz(0.4591324080007031*pi) node[62];
rz(0.4587913707886456*pi) node[63];
rz(0.4661957043751215*pi) node[64];
rz(0.4663847486165258*pi) node[65];
rz(0.46657064159005757*pi) node[66];
rz(0.36613976630153944*pi) node[67];
rz(0.37662413000870665*pi) node[68];
rz(0.35241637668553194*pi) node[69];
rz(0.33333333333333337*pi) node[70];
rz(0.4512680529667147*pi) node[71];
rz(0.458441611885707*pi) node[72];
rz(0.4667534787886811*pi) node[73];
rz(0.3040867245816834*pi) node[74];
rz(0.4518293287890226*pi) node[77];
rz(0.4523716333421135*pi) node[78];
rz(0.45340349850015604*pi) node[79];
rz(0.45289604887960166*pi) node[80];
rz(0.4580828129820004*pi) node[81];
rz(0.4577145602746747*pi) node[82];
rz(0.45733643996087703*pi) node[83];
rz(0.45694797457577796*pi) node[84];
rz(0.46693335570536354*pi) node[85];
rz(0.46711033600208196*pi) node[86];
rz(0.4672845151718017*pi) node[87];
rz(0.25*pi) node[89];
rz(0.4538948734714576*pi) node[91];
rz(0.45654868665454895*pi) node[92];
rz(0.46745595687650043*pi) node[93];
rz(0.45437106506118863*pi) node[98];
rz(0.4548327735510983*pi) node[99];
rz(0.4552807310539244*pi) node[100];
rz(0.45571563785141755*pi) node[101];
rz(0.4561380987323609*pi) node[102];
rz(0.46795455748221837*pi) node[104];
rz(0.46779091436973164*pi) node[105];
rz(0.46762472477815464*pi) node[106];
rz(0.46917970040315116*pi) node[108];
rz(0.4681157177775932*pi) node[111];
rz(0.4690342009541768*pi) node[112];
rz(0.46827449074882155*pi) node[122];
rz(0.4684309082268927*pi) node[123];
rz(0.4685850338737829*pi) node[124];
rz(0.46873689952048125*pi) node[125];
rz(0.46888660065995347*pi) node[126];
cx node[107],node[108];
rz(1.5308202995968487*pi) node[108];
sx node[108];
rz(0.5*pi) node[108];
sx node[108];
rz(1.5*pi) node[108];
cx node[108],node[112];
cx node[108],node[107];
rz(1.5309657990458234*pi) node[112];
sx node[112];
rz(0.5*pi) node[112];
sx node[112];
rz(1.5*pi) node[112];
cx node[112],node[126];
cx node[112],node[108];
rz(1.5311133993400468*pi) node[126];
sx node[126];
rz(0.5*pi) node[126];
sx node[126];
rz(1.5*pi) node[126];
cx node[126],node[125];
cx node[126],node[112];
rz(1.531263100479519*pi) node[125];
sx node[125];
rz(0.5*pi) node[125];
sx node[125];
rz(1.5*pi) node[125];
cx node[125],node[124];
rz(1.5314149661262173*pi) node[124];
cx node[125],node[126];
sx node[124];
rz(0.5*pi) node[124];
sx node[124];
rz(1.5*pi) node[124];
cx node[124],node[123];
rz(1.5315690917731075*pi) node[123];
cx node[124],node[125];
sx node[123];
rz(0.5*pi) node[123];
sx node[123];
rz(1.5*pi) node[123];
cx node[123],node[122];
rz(1.5317255092511783*pi) node[122];
cx node[123],node[124];
sx node[122];
rz(0.5*pi) node[122];
sx node[122];
rz(1.5*pi) node[122];
cx node[122],node[111];
rz(1.5318842822224068*pi) node[111];
cx node[122],node[123];
sx node[111];
rz(0.5*pi) node[111];
sx node[111];
rz(1.5*pi) node[111];
cx node[111],node[104];
rz(1.5320454425177816*pi) node[104];
cx node[111],node[122];
sx node[104];
rz(0.5*pi) node[104];
sx node[104];
rz(1.5*pi) node[104];
cx node[104],node[105];
cx node[104],node[111];
rz(1.5322090856302686*pi) node[105];
sx node[105];
rz(0.5*pi) node[105];
sx node[105];
rz(1.5*pi) node[105];
cx node[105],node[106];
cx node[105],node[104];
rz(1.5323752752218454*pi) node[106];
sx node[106];
rz(0.5*pi) node[106];
sx node[106];
rz(1.5*pi) node[106];
cx node[106],node[93];
rz(1.5325440431234998*pi) node[93];
cx node[106],node[105];
sx node[93];
rz(0.5*pi) node[93];
sx node[93];
rz(1.5*pi) node[93];
cx node[93],node[87];
rz(1.5327154848281985*pi) node[87];
cx node[93],node[106];
sx node[87];
rz(0.5*pi) node[87];
sx node[87];
rz(1.5*pi) node[87];
cx node[87],node[86];
rz(1.5328896639979182*pi) node[86];
cx node[87],node[93];
sx node[86];
rz(0.5*pi) node[86];
sx node[86];
rz(1.5*pi) node[86];
cx node[86],node[85];
rz(1.5330666442946363*pi) node[85];
cx node[86],node[87];
sx node[85];
rz(0.5*pi) node[85];
sx node[85];
rz(1.5*pi) node[85];
cx node[85],node[73];
rz(1.5332465212113189*pi) node[73];
cx node[85],node[86];
sx node[73];
rz(0.5*pi) node[73];
sx node[73];
rz(1.5*pi) node[73];
cx node[73],node[66];
rz(1.533429358409943*pi) node[66];
cx node[73],node[85];
sx node[66];
rz(0.5*pi) node[66];
sx node[66];
rz(1.5*pi) node[66];
cx node[66],node[65];
rz(1.5336152513834742*pi) node[65];
cx node[66],node[73];
sx node[65];
rz(0.5*pi) node[65];
sx node[65];
rz(1.5*pi) node[65];
cx node[65],node[64];
rz(1.5338042956248787*pi) node[64];
cx node[65],node[66];
sx node[64];
rz(0.5*pi) node[64];
sx node[64];
rz(1.5*pi) node[64];
cx node[64],node[54];
rz(1.5339965547961336*pi) node[54];
cx node[64],node[65];
sx node[54];
rz(0.5*pi) node[54];
sx node[54];
rz(1.5*pi) node[54];
cx node[54],node[45];
rz(1.534192124390205*pi) node[45];
cx node[54],node[64];
sx node[45];
rz(0.5*pi) node[45];
sx node[45];
rz(1.5*pi) node[45];
cx node[45],node[44];
rz(1.5343910999000585*pi) node[44];
cx node[45],node[54];
sx node[44];
rz(0.5*pi) node[44];
sx node[44];
rz(1.5*pi) node[44];
cx node[44],node[43];
rz(1.5345936086496486*pi) node[43];
cx node[44],node[45];
sx node[43];
cx node[46],node[45];
rz(0.5*pi) node[43];
cx node[45],node[46];
sx node[43];
cx node[46],node[45];
rz(1.5*pi) node[43];
cx node[47],node[46];
cx node[43],node[34];
cx node[46],node[47];
rz(1.5347997461319411*pi) node[34];
cx node[43],node[44];
cx node[47],node[46];
sx node[34];
cx node[45],node[44];
cx node[48],node[47];
rz(0.5*pi) node[34];
cx node[44],node[45];
cx node[47],node[48];
sx node[34];
cx node[45],node[44];
cx node[48],node[47];
rz(1.5*pi) node[34];
cx node[46],node[45];
cx node[49],node[48];
cx node[34],node[24];
cx node[45],node[46];
cx node[48],node[49];
rz(1.5350095760089137*pi) node[24];
cx node[34],node[43];
cx node[46],node[45];
cx node[49],node[48];
sx node[24];
cx node[50],node[49];
rz(0.5*pi) node[24];
cx node[49],node[50];
sx node[24];
cx node[50],node[49];
rz(1.5*pi) node[24];
cx node[24],node[23];
rz(1.5352232892664976*pi) node[23];
cx node[24],node[34];
sx node[23];
rz(0.5*pi) node[23];
sx node[23];
rz(1.5*pi) node[23];
cx node[23],node[22];
rz(1.5354409495666699*pi) node[22];
cx node[23],node[24];
sx node[22];
cx node[25],node[24];
rz(0.5*pi) node[22];
cx node[24],node[25];
sx node[22];
cx node[25],node[24];
rz(1.5*pi) node[22];
cx node[24],node[34];
cx node[26],node[25];
cx node[22],node[15];
cx node[34],node[24];
cx node[25],node[26];
rz(1.5356626842333856*pi) node[15];
cx node[22],node[23];
cx node[24],node[34];
cx node[26],node[25];
sx node[15];
cx node[16],node[26];
cx node[25],node[24];
cx node[34],node[43];
rz(0.5*pi) node[15];
cx node[26],node[16];
cx node[24],node[25];
cx node[43],node[34];
sx node[15];
cx node[16],node[26];
cx node[25],node[24];
cx node[34],node[43];
cx node[8],node[16];
rz(1.5*pi) node[15];
cx node[24],node[34];
cx node[26],node[25];
cx node[15],node[4];
cx node[16],node[8];
cx node[34],node[24];
cx node[25],node[26];
rz(1.5358886524215873*pi) node[4];
cx node[8],node[16];
cx node[15],node[22];
cx node[24],node[34];
cx node[26],node[25];
sx node[4];
cx node[7],node[8];
cx node[25],node[24];
cx node[27],node[26];
rz(0.5*pi) node[4];
cx node[8],node[7];
cx node[24],node[25];
cx node[26],node[27];
sx node[4];
cx node[7],node[8];
cx node[25],node[24];
cx node[27],node[26];
rz(1.5*pi) node[4];
cx node[6],node[7];
cx node[26],node[25];
cx node[28],node[27];
cx node[4],node[3];
cx node[7],node[6];
cx node[25],node[26];
cx node[27],node[28];
rz(1.5361189496242413*pi) node[3];
cx node[4],node[15];
cx node[6],node[7];
cx node[26],node[25];
cx node[28],node[27];
sx node[3];
cx node[16],node[26];
cx node[29],node[28];
rz(0.5*pi) node[3];
cx node[26],node[16];
cx node[28],node[29];
sx node[3];
cx node[16],node[26];
cx node[29],node[28];
rz(1.5*pi) node[3];
cx node[8],node[16];
cx node[30],node[29];
cx node[3],node[2];
cx node[16],node[8];
cx node[29],node[30];
rz(1.536353766827279*pi) node[2];
cx node[3],node[4];
cx node[8],node[16];
cx node[30],node[29];
sx node[2];
cx node[5],node[4];
cx node[7],node[8];
cx node[17],node[30];
rz(0.5*pi) node[2];
cx node[4],node[5];
cx node[8],node[7];
cx node[30],node[17];
sx node[2];
cx node[5],node[4];
cx node[7],node[8];
cx node[17],node[30];
rz(1.5*pi) node[2];
cx node[4],node[15];
cx node[12],node[17];
cx node[2],node[1];
cx node[15],node[4];
cx node[17],node[12];
rz(1.5365931995236666*pi) node[1];
cx node[2],node[3];
cx node[4],node[15];
cx node[12],node[17];
sx node[1];
cx node[11],node[12];
cx node[15],node[22];
rz(0.5*pi) node[1];
cx node[12],node[11];
cx node[22],node[15];
sx node[1];
cx node[11],node[12];
cx node[15],node[22];
rz(1.5*pi) node[1];
cx node[10],node[11];
cx node[1],node[0];
cx node[11],node[10];
rz(1.5368374386993355*pi) node[0];
cx node[1],node[2];
cx node[10],node[11];
sx node[0];
cx node[9],node[10];
rz(0.5*pi) node[0];
cx node[10],node[9];
sx node[0];
cx node[9],node[10];
rz(1.5*pi) node[0];
cx node[0],node[14];
cx node[0],node[1];
rz(1.5370866116782402*pi) node[14];
sx node[14];
rz(0.5*pi) node[14];
sx node[14];
rz(1.5*pi) node[14];
cx node[14],node[18];
cx node[14],node[0];
rz(1.537340941277301*pi) node[18];
sx node[18];
rz(0.5*pi) node[18];
sx node[18];
rz(1.5*pi) node[18];
cx node[18],node[19];
cx node[18],node[14];
rz(1.5376005548204725*pi) node[19];
sx node[19];
rz(0.5*pi) node[19];
sx node[19];
rz(1.5*pi) node[19];
cx node[19],node[20];
cx node[19],node[18];
rz(1.5378656751246749*pi) node[20];
sx node[20];
rz(0.5*pi) node[20];
sx node[20];
rz(1.5*pi) node[20];
cx node[20],node[21];
cx node[20],node[19];
rz(1.5381364931758403*pi) node[21];
sx node[21];
rz(0.5*pi) node[21];
sx node[21];
rz(1.5*pi) node[21];
cx node[21],node[20];
cx node[20],node[21];
cx node[21],node[20];
cx node[20],node[33];
cx node[20],node[21];
rz(1.538413168128911*pi) node[33];
cx node[22],node[21];
sx node[33];
cx node[21],node[22];
rz(0.5*pi) node[33];
cx node[22],node[21];
sx node[33];
rz(1.5*pi) node[33];
cx node[33],node[39];
cx node[33],node[20];
rz(1.5386959864627854*pi) node[39];
cx node[21],node[20];
sx node[39];
cx node[20],node[21];
rz(0.5*pi) node[39];
cx node[21],node[20];
sx node[39];
rz(1.5*pi) node[39];
cx node[39],node[40];
cx node[39],node[33];
rz(1.5389851391633949*pi) node[40];
cx node[20],node[33];
sx node[40];
cx node[33],node[20];
rz(0.5*pi) node[40];
cx node[20],node[33];
sx node[40];
rz(1.5*pi) node[40];
cx node[40],node[41];
cx node[40],node[39];
rz(1.5392808490476595*pi) node[41];
cx node[33],node[39];
sx node[41];
cx node[39],node[33];
rz(0.5*pi) node[41];
cx node[33],node[39];
sx node[41];
rz(1.5*pi) node[41];
cx node[41],node[42];
cx node[41],node[40];
rz(1.5395834344254657*pi) node[42];
sx node[42];
rz(0.5*pi) node[42];
sx node[42];
rz(1.5*pi) node[42];
cx node[42],node[41];
cx node[41],node[42];
cx node[42],node[41];
cx node[41],node[53];
cx node[41],node[42];
rz(1.5398930862827456*pi) node[53];
cx node[43],node[42];
sx node[53];
cx node[42],node[43];
rz(0.5*pi) node[53];
cx node[43],node[42];
sx node[53];
cx node[44],node[43];
rz(1.5*pi) node[53];
cx node[43],node[44];
cx node[53],node[60];
cx node[53],node[41];
cx node[44],node[43];
rz(1.5402101229293843*pi) node[60];
cx node[42],node[41];
cx node[45],node[44];
sx node[60];
cx node[41],node[42];
cx node[44],node[45];
rz(0.5*pi) node[60];
cx node[42],node[41];
cx node[45],node[44];
sx node[60];
cx node[41],node[40];
cx node[43],node[42];
rz(1.5*pi) node[60];
cx node[40],node[41];
cx node[42],node[43];
cx node[60],node[61];
cx node[41],node[40];
cx node[43],node[42];
cx node[60],node[53];
rz(1.540534862675269*pi) node[61];
cx node[34],node[43];
cx node[42],node[41];
sx node[61];
cx node[43],node[34];
cx node[41],node[42];
rz(0.5*pi) node[61];
cx node[34],node[43];
cx node[42],node[41];
sx node[61];
cx node[24],node[34];
cx node[43],node[42];
rz(1.5*pi) node[61];
cx node[34],node[24];
cx node[42],node[43];
cx node[61],node[62];
cx node[24],node[34];
cx node[43],node[42];
cx node[61],node[60];
rz(1.5408675919992971*pi) node[62];
cx node[25],node[24];
cx node[44],node[43];
sx node[62];
cx node[24],node[25];
cx node[43],node[44];
rz(0.5*pi) node[62];
cx node[25],node[24];
cx node[44],node[43];
sx node[62];
cx node[26],node[25];
rz(1.5*pi) node[62];
cx node[25],node[26];
cx node[62],node[63];
cx node[26],node[25];
cx node[62],node[61];
rz(1.5412086292113545*pi) node[63];
cx node[27],node[26];
sx node[63];
cx node[26],node[27];
rz(0.5*pi) node[63];
cx node[27],node[26];
sx node[63];
cx node[28],node[27];
rz(1.5*pi) node[63];
cx node[27],node[28];
cx node[63],node[62];
cx node[28],node[27];
cx node[62],node[63];
cx node[35],node[28];
cx node[63],node[62];
cx node[28],node[35];
cx node[62],node[72];
cx node[35],node[28];
cx node[62],node[63];
rz(1.5415583881142931*pi) node[72];
cx node[47],node[35];
sx node[72];
cx node[35],node[47];
rz(0.5*pi) node[72];
cx node[47],node[35];
sx node[72];
cx node[48],node[47];
rz(1.5*pi) node[72];
cx node[47],node[48];
cx node[72],node[81];
cx node[48],node[47];
cx node[72],node[62];
rz(1.5419171870179995*pi) node[81];
cx node[49],node[48];
sx node[81];
cx node[48],node[49];
rz(0.5*pi) node[81];
cx node[49],node[48];
sx node[81];
cx node[55],node[49];
rz(1.5*pi) node[81];
cx node[49],node[55];
cx node[81],node[82];
cx node[55],node[49];
cx node[81],node[72];
rz(1.5422854397253254*pi) node[82];
cx node[68],node[55];
sx node[82];
cx node[55],node[68];
rz(0.5*pi) node[82];
cx node[68],node[55];
sx node[82];
cx node[67],node[68];
rz(1.5*pi) node[82];
cx node[68],node[67];
cx node[82],node[83];
cx node[67],node[68];
cx node[82],node[81];
rz(1.5426635600391232*pi) node[83];
sx node[83];
rz(0.5*pi) node[83];
sx node[83];
rz(1.5*pi) node[83];
cx node[83],node[84];
cx node[83],node[82];
rz(1.543052025424222*pi) node[84];
sx node[84];
rz(0.5*pi) node[84];
sx node[84];
rz(1.5*pi) node[84];
cx node[84],node[83];
cx node[83],node[84];
cx node[84],node[83];
cx node[83],node[92];
cx node[83],node[84];
rz(1.5434513133454508*pi) node[92];
sx node[92];
rz(0.5*pi) node[92];
sx node[92];
rz(1.5*pi) node[92];
cx node[92],node[102];
cx node[92],node[83];
rz(1.5438619012676393*pi) node[102];
sx node[102];
rz(0.5*pi) node[102];
sx node[102];
rz(1.5*pi) node[102];
cx node[102],node[101];
cx node[102],node[92];
rz(1.5442843621485824*pi) node[101];
sx node[101];
rz(0.5*pi) node[101];
sx node[101];
rz(1.5*pi) node[101];
cx node[101],node[100];
rz(1.5447192689460754*pi) node[100];
cx node[101],node[102];
sx node[100];
rz(0.5*pi) node[100];
sx node[100];
rz(1.5*pi) node[100];
cx node[100],node[99];
rz(1.545167226448902*pi) node[99];
cx node[100],node[101];
sx node[99];
rz(0.5*pi) node[99];
sx node[99];
rz(1.5*pi) node[99];
cx node[99],node[98];
rz(1.5456289349388115*pi) node[98];
cx node[99],node[100];
sx node[98];
rz(0.5*pi) node[98];
sx node[98];
rz(1.5*pi) node[98];
cx node[98],node[91];
rz(1.5461051265285424*pi) node[91];
cx node[98],node[99];
sx node[91];
rz(0.5*pi) node[91];
sx node[91];
rz(1.5*pi) node[91];
cx node[91],node[79];
rz(1.5465965014998442*pi) node[79];
cx node[91],node[98];
sx node[79];
rz(0.5*pi) node[79];
sx node[79];
rz(1.5*pi) node[79];
cx node[79],node[80];
cx node[79],node[91];
rz(1.5471039511203986*pi) node[80];
sx node[80];
rz(0.5*pi) node[80];
sx node[80];
rz(1.5*pi) node[80];
cx node[80],node[79];
cx node[79],node[80];
cx node[80],node[79];
cx node[79],node[78];
rz(1.5476283666578863*pi) node[78];
cx node[79],node[80];
sx node[78];
rz(0.5*pi) node[78];
sx node[78];
rz(1.5*pi) node[78];
cx node[78],node[77];
rz(1.5481706712109777*pi) node[77];
cx node[78],node[79];
sx node[77];
rz(0.5*pi) node[77];
sx node[77];
rz(1.5*pi) node[77];
cx node[77],node[71];
rz(1.5487319470332856*pi) node[71];
cx node[77],node[78];
sx node[71];
rz(0.5*pi) node[71];
sx node[71];
rz(1.5*pi) node[71];
cx node[71],node[58];
rz(1.5493133082094115*pi) node[58];
cx node[71],node[77];
sx node[58];
rz(0.5*pi) node[58];
sx node[58];
rz(1.5*pi) node[58];
cx node[58],node[59];
cx node[58],node[71];
rz(1.5499159961479119*pi) node[59];
sx node[59];
rz(0.5*pi) node[59];
sx node[59];
rz(1.5*pi) node[59];
cx node[59],node[58];
cx node[58],node[59];
cx node[59],node[58];
cx node[58],node[57];
rz(1.55054131591932*pi) node[57];
cx node[58],node[59];
sx node[57];
rz(0.5*pi) node[57];
sx node[57];
rz(1.5*pi) node[57];
cx node[57],node[56];
rz(1.5511907635801008*pi) node[56];
cx node[57],node[58];
sx node[56];
rz(0.5*pi) node[56];
sx node[56];
rz(1.5*pi) node[56];
cx node[56],node[52];
rz(1.5518658988486966*pi) node[52];
cx node[56],node[57];
sx node[52];
rz(0.5*pi) node[52];
sx node[52];
rz(1.5*pi) node[52];
cx node[52],node[37];
rz(1.5525684724294817*pi) node[37];
cx node[52],node[56];
sx node[37];
rz(0.5*pi) node[37];
sx node[37];
rz(1.5*pi) node[37];
cx node[37],node[38];
cx node[37],node[52];
rz(1.5533003941817725*pi) node[38];
sx node[38];
rz(0.5*pi) node[38];
sx node[38];
rz(1.5*pi) node[38];
cx node[38],node[39];
cx node[38],node[37];
rz(1.5540637649508184*pi) node[39];
sx node[39];
rz(0.5*pi) node[39];
sx node[39];
rz(1.5*pi) node[39];
cx node[39],node[40];
cx node[39],node[38];
rz(1.5548609083987888*pi) node[40];
sx node[40];
rz(0.5*pi) node[40];
sx node[40];
rz(1.5*pi) node[40];
cx node[40],node[41];
cx node[40],node[39];
rz(1.5556943710047721*pi) node[41];
sx node[41];
rz(0.5*pi) node[41];
sx node[41];
rz(1.5*pi) node[41];
cx node[41],node[42];
cx node[41],node[40];
rz(1.5565670493887336*pi) node[42];
sx node[42];
rz(0.5*pi) node[42];
sx node[42];
rz(1.5*pi) node[42];
cx node[42],node[43];
cx node[42],node[41];
rz(1.5574820629875576*pi) node[43];
sx node[43];
rz(0.5*pi) node[43];
sx node[43];
rz(1.5*pi) node[43];
cx node[43],node[34];
rz(1.558442976871969*pi) node[34];
cx node[43],node[42];
sx node[34];
rz(0.5*pi) node[34];
sx node[34];
rz(1.5*pi) node[34];
cx node[34],node[24];
rz(1.5594537380845572*pi) node[24];
cx node[34],node[43];
sx node[24];
rz(0.5*pi) node[24];
sx node[24];
rz(1.5*pi) node[24];
cx node[24],node[25];
cx node[24],node[34];
rz(1.5605188666257055*pi) node[25];
sx node[25];
rz(0.5*pi) node[25];
sx node[25];
rz(1.5*pi) node[25];
cx node[25],node[26];
cx node[25],node[24];
rz(1.561643359960627*pi) node[26];
sx node[26];
rz(0.5*pi) node[26];
sx node[26];
rz(1.5*pi) node[26];
cx node[26],node[16];
rz(1.5628329476672729*pi) node[16];
cx node[26],node[25];
sx node[16];
rz(0.5*pi) node[16];
sx node[16];
rz(1.5*pi) node[16];
cx node[16],node[8];
rz(1.5640942187602875*pi) node[8];
cx node[16],node[26];
sx node[8];
cx node[27],node[26];
rz(0.5*pi) node[8];
cx node[26],node[27];
sx node[8];
cx node[27],node[26];
rz(1.5*pi) node[8];
cx node[28],node[27];
cx node[8],node[16];
cx node[27],node[28];
cx node[16],node[8];
cx node[28],node[27];
cx node[8],node[16];
cx node[29],node[28];
cx node[16],node[26];
cx node[28],node[29];
cx node[16],node[8];
rz(1.5654346216910078*pi) node[26];
cx node[29],node[28];
sx node[26];
cx node[30],node[29];
rz(0.5*pi) node[26];
cx node[29],node[30];
sx node[26];
cx node[30],node[29];
rz(1.5*pi) node[26];
cx node[31],node[30];
cx node[26],node[27];
cx node[30],node[31];
cx node[26],node[16];
rz(1.566862814488337*pi) node[27];
cx node[31],node[30];
sx node[27];
cx node[32],node[31];
rz(0.5*pi) node[27];
cx node[31],node[32];
sx node[27];
cx node[32],node[31];
rz(1.5*pi) node[27];
cx node[36],node[32];
cx node[27],node[28];
cx node[32],node[36];
cx node[27],node[26];
rz(1.5683888239136907*pi) node[28];
cx node[36],node[32];
sx node[28];
cx node[51],node[36];
rz(0.5*pi) node[28];
cx node[36],node[51];
sx node[28];
cx node[51],node[36];
rz(1.5*pi) node[28];
cx node[28],node[29];
cx node[28],node[27];
rz(1.5700243637708802*pi) node[29];
sx node[29];
rz(0.5*pi) node[29];
sx node[29];
rz(1.5*pi) node[29];
cx node[29],node[30];
cx node[29],node[28];
rz(1.571783153216*pi) node[30];
cx node[35],node[28];
sx node[30];
cx node[28],node[35];
rz(0.5*pi) node[30];
cx node[35],node[28];
sx node[30];
rz(1.5*pi) node[30];
cx node[47],node[35];
cx node[30],node[17];
cx node[35],node[47];
rz(1.5736815215462117*pi) node[17];
cx node[30],node[29];
cx node[47],node[35];
sx node[17];
cx node[28],node[29];
cx node[48],node[47];
rz(0.5*pi) node[17];
cx node[29],node[28];
cx node[47],node[48];
sx node[17];
cx node[28],node[29];
cx node[48],node[47];
rz(1.5*pi) node[17];
cx node[35],node[28];
cx node[49],node[48];
cx node[17],node[12];
cx node[28],node[35];
cx node[48],node[49];
rz(1.5757390129885265*pi) node[12];
cx node[17],node[30];
cx node[35],node[28];
cx node[49],node[48];
sx node[12];
cx node[31],node[30];
cx node[47],node[35];
cx node[55],node[49];
rz(0.5*pi) node[12];
cx node[30],node[31];
cx node[35],node[47];
cx node[49],node[55];
sx node[12];
cx node[31],node[30];
cx node[47],node[35];
cx node[55],node[49];
rz(1.5*pi) node[12];
cx node[32],node[31];
cx node[48],node[47];
cx node[68],node[55];
cx node[12],node[13];
cx node[31],node[32];
cx node[47],node[48];
cx node[55],node[68];
cx node[12],node[17];
rz(1.577979118812545*pi) node[13];
cx node[32],node[31];
cx node[48],node[47];
cx node[68],node[55];
sx node[13];
cx node[30],node[17];
cx node[36],node[32];
cx node[49],node[48];
cx node[69],node[68];
rz(0.5*pi) node[13];
cx node[17],node[30];
cx node[32],node[36];
cx node[48],node[49];
cx node[68],node[69];
sx node[13];
cx node[30],node[17];
cx node[36],node[32];
cx node[49],node[48];
cx node[69],node[68];
rz(1.5*pi) node[13];
cx node[31],node[30];
cx node[55],node[49];
cx node[70],node[69];
cx node[13],node[12];
cx node[30],node[31];
cx node[49],node[55];
cx node[69],node[70];
cx node[12],node[13];
cx node[31],node[30];
cx node[55],node[49];
cx node[70],node[69];
cx node[13],node[12];
cx node[32],node[31];
cx node[68],node[55];
cx node[74],node[70];
cx node[12],node[11];
cx node[31],node[32];
cx node[55],node[68];
cx node[70],node[74];
rz(1.580430614231978*pi) node[11];
cx node[12],node[13];
cx node[32],node[31];
cx node[68],node[55];
cx node[74],node[70];
sx node[11];
cx node[69],node[68];
cx node[89],node[74];
rz(0.5*pi) node[11];
cx node[68],node[69];
cx node[74],node[89];
sx node[11];
cx node[69],node[68];
cx node[89],node[74];
rz(1.5*pi) node[11];
cx node[70],node[69];
cx node[11],node[10];
cx node[69],node[70];
rz(1.583128990799135*pi) node[10];
cx node[11],node[12];
cx node[70],node[69];
sx node[10];
cx node[17],node[12];
cx node[74],node[70];
rz(0.5*pi) node[10];
cx node[12],node[17];
cx node[70],node[74];
sx node[10];
cx node[17],node[12];
cx node[74],node[70];
rz(1.5*pi) node[10];
cx node[30],node[17];
cx node[10],node[11];
cx node[17],node[30];
cx node[11],node[10];
cx node[30],node[17];
cx node[10],node[11];
cx node[31],node[30];
cx node[11],node[12];
cx node[30],node[31];
cx node[11],node[10];
rz(1.5861186527431395*pi) node[12];
cx node[31],node[30];
sx node[12];
rz(0.5*pi) node[12];
sx node[12];
rz(1.5*pi) node[12];
cx node[12],node[17];
cx node[12],node[11];
rz(1.5894561955617534*pi) node[17];
sx node[17];
rz(0.5*pi) node[17];
sx node[17];
rz(1.5*pi) node[17];
cx node[17],node[30];
cx node[17],node[12];
rz(1.5932147350358346*pi) node[30];
sx node[30];
rz(0.5*pi) node[30];
sx node[30];
rz(1.5*pi) node[30];
cx node[30],node[29];
cx node[30],node[17];
rz(1.5974911010327593*pi) node[29];
sx node[29];
rz(0.5*pi) node[29];
sx node[29];
rz(1.5*pi) node[29];
cx node[29],node[28];
rz(1.6024163735636583*pi) node[28];
cx node[29],node[30];
sx node[28];
rz(0.5*pi) node[28];
sx node[28];
rz(1.5*pi) node[28];
cx node[28],node[35];
cx node[28],node[29];
rz(1.6081734534891328*pi) node[35];
sx node[35];
rz(0.5*pi) node[35];
sx node[35];
rz(1.5*pi) node[35];
cx node[35],node[47];
cx node[35],node[28];
rz(1.615026729000647*pi) node[47];
sx node[47];
rz(0.5*pi) node[47];
sx node[47];
rz(1.5*pi) node[47];
cx node[47],node[48];
cx node[47],node[35];
rz(1.6233758699912932*pi) node[48];
sx node[48];
rz(0.5*pi) node[48];
sx node[48];
rz(1.5*pi) node[48];
cx node[48],node[49];
cx node[48],node[47];
rz(1.6338602336984605*pi) node[49];
sx node[49];
rz(0.5*pi) node[49];
sx node[49];
rz(1.5*pi) node[49];
cx node[49],node[55];
cx node[49],node[48];
rz(1.647583623314468*pi) node[55];
sx node[55];
rz(0.5*pi) node[55];
sx node[55];
rz(1.5*pi) node[55];
cx node[55],node[68];
cx node[55],node[49];
rz(1.6666666666666665*pi) node[68];
sx node[68];
rz(0.5*pi) node[68];
sx node[68];
rz(1.5*pi) node[68];
cx node[68],node[69];
cx node[68],node[55];
rz(1.6959132754183164*pi) node[69];
sx node[69];
rz(0.5*pi) node[69];
sx node[69];
rz(1.5*pi) node[69];
cx node[69],node[70];
cx node[69],node[68];
rz(1.75*pi) node[70];
sx node[70];
rz(0.5*pi) node[70];
sx node[70];
rz(1.5*pi) node[70];
cx node[70],node[69];
barrier node[70],node[69],node[68],node[55],node[49],node[48],node[47],node[35],node[28],node[29],node[30],node[17],node[12],node[11],node[10],node[31],node[13],node[32],node[36],node[74],node[89],node[27],node[26],node[16],node[8],node[51],node[25],node[24],node[34],node[43],node[42],node[41],node[40],node[39],node[38],node[37],node[52],node[56],node[57],node[58],node[59],node[71],node[77],node[78],node[79],node[80],node[91],node[98],node[99],node[100],node[101],node[102],node[92],node[83],node[84],node[82],node[81],node[72],node[62],node[63],node[61],node[60],node[53],node[67],node[45],node[44],node[33],node[20],node[21],node[22],node[19],node[18],node[14],node[0],node[1],node[2],node[3],node[5],node[4],node[15],node[23],node[6],node[9],node[7],node[46],node[50],node[54],node[64],node[65],node[66],node[73],node[85],node[86],node[87],node[93],node[106],node[105],node[104],node[111],node[122],node[123],node[124],node[125],node[126],node[112],node[108],node[107];
measure node[70] -> meas[0];
measure node[69] -> meas[1];
measure node[68] -> meas[2];
measure node[55] -> meas[3];
measure node[49] -> meas[4];
measure node[48] -> meas[5];
measure node[47] -> meas[6];
measure node[35] -> meas[7];
measure node[28] -> meas[8];
measure node[29] -> meas[9];
measure node[30] -> meas[10];
measure node[17] -> meas[11];
measure node[12] -> meas[12];
measure node[11] -> meas[13];
measure node[10] -> meas[14];
measure node[31] -> meas[15];
measure node[13] -> meas[16];
measure node[32] -> meas[17];
measure node[36] -> meas[18];
measure node[74] -> meas[19];
measure node[89] -> meas[20];
measure node[27] -> meas[21];
measure node[26] -> meas[22];
measure node[16] -> meas[23];
measure node[8] -> meas[24];
measure node[51] -> meas[25];
measure node[25] -> meas[26];
measure node[24] -> meas[27];
measure node[34] -> meas[28];
measure node[43] -> meas[29];
measure node[42] -> meas[30];
measure node[41] -> meas[31];
measure node[40] -> meas[32];
measure node[39] -> meas[33];
measure node[38] -> meas[34];
measure node[37] -> meas[35];
measure node[52] -> meas[36];
measure node[56] -> meas[37];
measure node[57] -> meas[38];
measure node[58] -> meas[39];
measure node[59] -> meas[40];
measure node[71] -> meas[41];
measure node[77] -> meas[42];
measure node[78] -> meas[43];
measure node[79] -> meas[44];
measure node[80] -> meas[45];
measure node[91] -> meas[46];
measure node[98] -> meas[47];
measure node[99] -> meas[48];
measure node[100] -> meas[49];
measure node[101] -> meas[50];
measure node[102] -> meas[51];
measure node[92] -> meas[52];
measure node[83] -> meas[53];
measure node[84] -> meas[54];
measure node[82] -> meas[55];
measure node[81] -> meas[56];
measure node[72] -> meas[57];
measure node[62] -> meas[58];
measure node[63] -> meas[59];
measure node[61] -> meas[60];
measure node[60] -> meas[61];
measure node[53] -> meas[62];
measure node[67] -> meas[63];
measure node[45] -> meas[64];
measure node[44] -> meas[65];
measure node[33] -> meas[66];
measure node[20] -> meas[67];
measure node[21] -> meas[68];
measure node[22] -> meas[69];
measure node[19] -> meas[70];
measure node[18] -> meas[71];
measure node[14] -> meas[72];
measure node[0] -> meas[73];
measure node[1] -> meas[74];
measure node[2] -> meas[75];
measure node[3] -> meas[76];
measure node[5] -> meas[77];
measure node[4] -> meas[78];
measure node[15] -> meas[79];
measure node[23] -> meas[80];
measure node[6] -> meas[81];
measure node[9] -> meas[82];
measure node[7] -> meas[83];
measure node[46] -> meas[84];
measure node[50] -> meas[85];
measure node[54] -> meas[86];
measure node[64] -> meas[87];
measure node[65] -> meas[88];
measure node[66] -> meas[89];
measure node[73] -> meas[90];
measure node[85] -> meas[91];
measure node[86] -> meas[92];
measure node[87] -> meas[93];
measure node[93] -> meas[94];
measure node[106] -> meas[95];
measure node[105] -> meas[96];
measure node[104] -> meas[97];
measure node[111] -> meas[98];
measure node[122] -> meas[99];
measure node[123] -> meas[100];
measure node[124] -> meas[101];
measure node[125] -> meas[102];
measure node[126] -> meas[103];
measure node[112] -> meas[104];
measure node[108] -> meas[105];
measure node[107] -> meas[106];
