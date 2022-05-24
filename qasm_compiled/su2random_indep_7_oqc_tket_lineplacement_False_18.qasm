OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }

qreg node[8];
creg meas[7];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
rz(1.624632278820009*pi) node[5];
sx node[6];
sx node[7];
rz(3.0152084421083045*pi) node[0];
rz(1.2699874406157656*pi) node[1];
rz(1.1519980763433342*pi) node[2];
rz(1.3012872050481916*pi) node[3];
rz(1.3072761227961691*pi) node[4];
sx node[5];
rz(1.2387778979374797*pi) node[7];
sx node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
rz(2.634950137211045*pi) node[5];
sx node[7];
rz(1.1494359077595746*pi) node[0];
rz(1.1083301584663179*pi) node[1];
rz(1.1016414746307506*pi) node[2];
rz(1.0136057422820741*pi) node[3];
rz(1.3005337114349136*pi) node[4];
sx node[5];
rz(1.0925507479996694*pi) node[7];
x node[0];
sx node[1];
sx node[2];
sx node[3];
sx node[4];
rz(3.749424891367654*pi) node[5];
sx node[7];
rz(3.5*pi) node[0];
sx node[5];
ecr node[0],node[1];
x node[0];
sx node[1];
rz(3.5*pi) node[0];
ecr node[0],node[7];
x node[0];
sx node[7];
rz(3.5*pi) node[0];
ecr node[0],node[1];
sx node[0];
x node[1];
rz(3.5*pi) node[1];
ecr node[1],node[0];
x node[0];
sx node[1];
rz(3.5*pi) node[0];
ecr node[0],node[1];
x node[0];
x node[1];
rz(3.5*pi) node[0];
rz(3.5*pi) node[1];
ecr node[0],node[7];
ecr node[1],node[2];
x node[0];
sx node[1];
x node[2];
x node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[2];
rz(3.5*pi) node[7];
ecr node[2],node[1];
x node[1];
sx node[2];
rz(3.5*pi) node[1];
ecr node[1],node[2];
sx node[1];
x node[2];
rz(3.5*pi) node[2];
ecr node[2],node[1];
sx node[1];
x node[2];
ecr node[0],node[1];
rz(3.5*pi) node[2];
x node[0];
sx node[1];
ecr node[2],node[3];
rz(3.5*pi) node[0];
sx node[2];
x node[3];
ecr node[0],node[1];
rz(3.5*pi) node[3];
sx node[0];
x node[1];
ecr node[3],node[2];
rz(3.5*pi) node[1];
x node[2];
sx node[3];
ecr node[1],node[0];
rz(3.5*pi) node[2];
x node[0];
sx node[1];
ecr node[2],node[3];
rz(3.5*pi) node[0];
sx node[2];
x node[3];
ecr node[0],node[1];
rz(3.5*pi) node[3];
sx node[0];
x node[1];
ecr node[3],node[2];
ecr node[7],node[0];
rz(3.5*pi) node[1];
sx node[2];
x node[3];
sx node[0];
ecr node[1],node[2];
rz(3.5*pi) node[3];
x node[7];
x node[1];
sx node[2];
ecr node[3],node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[1];
sx node[3];
x node[4];
x node[0];
ecr node[1],node[2];
rz(3.5*pi) node[4];
sx node[7];
rz(3.5*pi) node[0];
sx node[1];
x node[2];
ecr node[4],node[3];
ecr node[0],node[7];
rz(3.5*pi) node[2];
x node[3];
sx node[4];
sx node[0];
ecr node[2],node[1];
rz(3.5*pi) node[3];
x node[7];
x node[1];
sx node[2];
ecr node[3],node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[1];
sx node[3];
x node[4];
x node[0];
ecr node[1],node[2];
rz(3.5*pi) node[4];
x node[7];
rz(3.5*pi) node[0];
sx node[1];
x node[2];
ecr node[4],node[3];
rz(3.5*pi) node[7];
ecr node[0],node[1];
rz(3.5*pi) node[2];
sx node[3];
x node[4];
x node[0];
sx node[1];
ecr node[2],node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[0];
x node[2];
sx node[3];
ecr node[4],node[5];
ecr node[0],node[1];
rz(3.5*pi) node[2];
sx node[4];
x node[5];
sx node[0];
x node[1];
ecr node[2],node[3];
rz(3.072346110734723*pi) node[4];
rz(3.5*pi) node[5];
rz(3.5*pi) node[1];
sx node[2];
x node[3];
sx node[4];
ecr node[1],node[0];
rz(3.5*pi) node[3];
rz(1.1079806223802988*pi) node[4];
x node[0];
sx node[1];
ecr node[3],node[2];
sx node[4];
rz(3.5*pi) node[0];
x node[2];
sx node[3];
ecr node[5],node[4];
ecr node[0],node[1];
rz(3.5*pi) node[2];
x node[4];
sx node[5];
sx node[0];
x node[1];
ecr node[2],node[3];
rz(3.5*pi) node[4];
ecr node[7],node[0];
rz(3.5*pi) node[1];
sx node[2];
x node[3];
ecr node[4],node[5];
sx node[0];
ecr node[1],node[2];
rz(3.5*pi) node[3];
sx node[4];
x node[5];
x node[7];
sx node[1];
x node[2];
rz(3.5*pi) node[5];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[2];
ecr node[5],node[4];
x node[0];
ecr node[2],node[1];
sx node[4];
x node[5];
sx node[7];
rz(3.5*pi) node[0];
x node[1];
sx node[2];
ecr node[3],node[4];
rz(3.5*pi) node[5];
ecr node[0],node[7];
rz(3.5*pi) node[1];
sx node[3];
x node[4];
sx node[0];
ecr node[1],node[2];
rz(1.0048656101173643*pi) node[3];
rz(3.5*pi) node[4];
x node[7];
sx node[1];
x node[2];
sx node[3];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[2];
rz(1.16864768874303*pi) node[3];
x node[0];
ecr node[2],node[1];
sx node[3];
x node[7];
rz(3.5*pi) node[0];
sx node[1];
x node[2];
ecr node[4],node[3];
rz(3.5*pi) node[7];
ecr node[0],node[1];
rz(3.5*pi) node[2];
x node[3];
sx node[4];
x node[0];
sx node[1];
rz(3.5*pi) node[3];
rz(3.5*pi) node[0];
ecr node[3],node[4];
ecr node[0],node[1];
sx node[3];
x node[4];
sx node[0];
x node[1];
rz(3.5*pi) node[4];
rz(3.5*pi) node[1];
ecr node[4],node[3];
ecr node[1],node[0];
sx node[3];
sx node[4];
x node[0];
sx node[1];
ecr node[2],node[3];
ecr node[5],node[4];
rz(3.5*pi) node[0];
sx node[2];
sx node[3];
sx node[4];
x node[5];
ecr node[0],node[1];
rz(1.2864911493129307*pi) node[2];
rz(3.5*pi) node[5];
sx node[0];
x node[1];
sx node[2];
ecr node[5],node[4];
ecr node[7],node[0];
rz(3.5*pi) node[1];
rz(1.1442502386431834*pi) node[2];
x node[4];
sx node[5];
sx node[0];
x node[2];
rz(3.5*pi) node[4];
x node[7];
rz(3.5*pi) node[2];
ecr node[4],node[5];
rz(3.5*pi) node[7];
ecr node[7],node[0];
ecr node[2],node[3];
sx node[4];
x node[5];
x node[0];
sx node[2];
x node[3];
rz(3.5*pi) node[5];
sx node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[3];
ecr node[5],node[4];
ecr node[0],node[7];
ecr node[3],node[2];
x node[4];
x node[5];
sx node[0];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
x node[7];
rz(3.5*pi) node[2];
rz(3.5*pi) node[7];
ecr node[7],node[0];
ecr node[2],node[3];
x node[0];
sx node[2];
sx node[3];
x node[7];
rz(3.5*pi) node[0];
ecr node[1],node[2];
ecr node[4],node[3];
rz(3.5*pi) node[7];
sx node[1];
x node[2];
sx node[3];
x node[4];
rz(1.3127099781308171*pi) node[1];
rz(3.5*pi) node[2];
rz(3.5*pi) node[4];
sx node[1];
ecr node[4],node[3];
rz(1.1254217887063627*pi) node[1];
x node[3];
sx node[4];
sx node[1];
rz(3.5*pi) node[3];
ecr node[2],node[1];
ecr node[3],node[4];
x node[1];
sx node[2];
sx node[3];
x node[4];
rz(3.5*pi) node[1];
rz(3.5*pi) node[4];
ecr node[1],node[2];
ecr node[4],node[3];
sx node[1];
x node[2];
x node[3];
sx node[4];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
ecr node[2],node[1];
sx node[4];
x node[5];
sx node[1];
sx node[2];
rz(3.5*pi) node[5];
ecr node[0],node[1];
ecr node[3],node[2];
ecr node[5],node[4];
sx node[0];
sx node[1];
x node[2];
sx node[3];
x node[4];
sx node[5];
rz(1.261088250592497*pi) node[0];
rz(3.5*pi) node[2];
rz(3.5*pi) node[4];
sx node[0];
ecr node[2],node[3];
ecr node[4],node[5];
rz(1.1800471742743825*pi) node[0];
sx node[2];
x node[3];
sx node[4];
x node[5];
x node[0];
rz(3.5*pi) node[3];
rz(3.5*pi) node[5];
rz(3.5*pi) node[0];
ecr node[3],node[2];
ecr node[5],node[4];
ecr node[0],node[1];
x node[2];
sx node[3];
x node[4];
x node[5];
sx node[0];
x node[1];
rz(3.5*pi) node[2];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
rz(3.5*pi) node[1];
ecr node[2],node[3];
ecr node[1],node[0];
x node[2];
sx node[3];
x node[0];
sx node[1];
rz(3.5*pi) node[2];
ecr node[4],node[3];
rz(3.5*pi) node[0];
sx node[3];
x node[4];
ecr node[0],node[1];
rz(3.5*pi) node[4];
sx node[0];
sx node[1];
ecr node[4],node[3];
ecr node[7],node[0];
ecr node[2],node[1];
x node[3];
sx node[4];
rz(2.0734650145480433*pi) node[0];
sx node[1];
x node[2];
rz(3.5*pi) node[3];
sx node[7];
sx node[0];
rz(3.5*pi) node[2];
ecr node[3],node[4];
rz(1.029603809040526*pi) node[7];
rz(0.8987062216241251*pi) node[0];
ecr node[2],node[1];
sx node[3];
x node[4];
sx node[7];
sx node[0];
x node[1];
sx node[2];
rz(3.5*pi) node[4];
rz(1.2228861813767886*pi) node[7];
rz(1.0*pi) node[0];
rz(3.5*pi) node[1];
ecr node[4],node[3];
x node[7];
sx node[0];
ecr node[1],node[2];
x node[3];
sx node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[1];
x node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
x node[0];
rz(3.5*pi) node[2];
sx node[4];
x node[5];
sx node[7];
rz(3.5*pi) node[0];
ecr node[2],node[1];
rz(3.5*pi) node[5];
ecr node[0],node[7];
x node[1];
sx node[2];
ecr node[5],node[4];
sx node[0];
rz(3.5*pi) node[1];
ecr node[3],node[2];
x node[4];
sx node[5];
x node[7];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[2];
ecr node[4],node[5];
sx node[0];
ecr node[2],node[3];
sx node[4];
x node[5];
sx node[7];
ecr node[1],node[0];
sx node[2];
x node[3];
rz(3.5*pi) node[5];
x node[0];
sx node[1];
rz(3.5*pi) node[3];
ecr node[5],node[4];
rz(3.5*pi) node[0];
ecr node[3],node[2];
x node[4];
x node[5];
ecr node[0],node[1];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
sx node[0];
x node[1];
rz(3.5*pi) node[2];
rz(3.5*pi) node[1];
ecr node[2],node[3];
ecr node[1],node[0];
x node[2];
sx node[3];
x node[0];
sx node[1];
rz(3.5*pi) node[2];
ecr node[4],node[3];
rz(3.5*pi) node[0];
sx node[3];
x node[4];
ecr node[0],node[1];
rz(3.5*pi) node[4];
x node[0];
sx node[1];
ecr node[4],node[3];
rz(3.5*pi) node[0];
ecr node[2],node[1];
x node[3];
sx node[4];
ecr node[0],node[7];
sx node[1];
x node[2];
rz(3.5*pi) node[3];
sx node[0];
rz(3.5*pi) node[2];
ecr node[3],node[4];
x node[7];
rz(3.2240153475008384*pi) node[0];
ecr node[2],node[1];
sx node[3];
x node[4];
rz(3.5*pi) node[7];
sx node[0];
x node[1];
sx node[2];
rz(3.5*pi) node[4];
rz(1.070927199853676*pi) node[0];
rz(3.5*pi) node[1];
ecr node[4],node[3];
sx node[0];
ecr node[1],node[2];
x node[3];
sx node[4];
ecr node[7],node[0];
sx node[1];
x node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
x node[0];
rz(3.5*pi) node[2];
sx node[4];
x node[5];
sx node[7];
rz(3.5*pi) node[0];
ecr node[2],node[1];
rz(3.5*pi) node[5];
ecr node[0],node[7];
x node[1];
sx node[2];
ecr node[5],node[4];
sx node[0];
rz(3.5*pi) node[1];
ecr node[3],node[2];
x node[4];
sx node[5];
x node[7];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[2];
ecr node[4],node[5];
sx node[0];
ecr node[2],node[3];
sx node[4];
x node[5];
x node[7];
ecr node[1],node[0];
sx node[2];
x node[3];
rz(3.5*pi) node[5];
rz(3.5*pi) node[7];
x node[0];
sx node[1];
rz(3.5*pi) node[3];
ecr node[5],node[4];
rz(3.5*pi) node[0];
rz(1.3068276846454143*pi) node[1];
ecr node[3],node[2];
x node[4];
x node[5];
sx node[1];
x node[2];
sx node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
rz(1.2319545976679482*pi) node[1];
rz(3.5*pi) node[2];
sx node[1];
ecr node[2],node[3];
ecr node[0],node[1];
x node[2];
sx node[3];
sx node[0];
x node[1];
rz(3.5*pi) node[2];
ecr node[4],node[3];
rz(3.5*pi) node[1];
sx node[3];
x node[4];
ecr node[1],node[0];
rz(3.5*pi) node[4];
x node[0];
sx node[1];
ecr node[4],node[3];
rz(3.5*pi) node[0];
x node[3];
sx node[4];
ecr node[0],node[1];
rz(3.5*pi) node[3];
sx node[0];
sx node[1];
ecr node[3],node[4];
ecr node[7],node[0];
ecr node[2],node[1];
sx node[3];
x node[4];
sx node[0];
sx node[1];
sx node[2];
rz(3.5*pi) node[4];
x node[7];
rz(1.2198586373732292*pi) node[2];
ecr node[4],node[3];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[2];
x node[3];
sx node[4];
x node[0];
rz(1.2089818357735838*pi) node[2];
rz(3.5*pi) node[3];
ecr node[5],node[4];
sx node[7];
rz(3.5*pi) node[0];
x node[2];
sx node[4];
x node[5];
ecr node[0],node[7];
rz(3.5*pi) node[2];
rz(3.5*pi) node[5];
sx node[0];
ecr node[2],node[1];
ecr node[5],node[4];
x node[7];
x node[1];
sx node[2];
x node[4];
sx node[5];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[1];
rz(3.5*pi) node[4];
x node[0];
ecr node[1],node[2];
ecr node[4],node[5];
x node[7];
rz(3.5*pi) node[0];
sx node[1];
x node[2];
sx node[4];
x node[5];
rz(3.5*pi) node[7];
rz(3.5*pi) node[2];
rz(3.5*pi) node[5];
ecr node[2],node[1];
ecr node[5],node[4];
sx node[1];
sx node[2];
x node[4];
x node[5];
ecr node[0],node[1];
ecr node[3],node[2];
rz(3.5*pi) node[4];
rz(3.5*pi) node[5];
x node[0];
sx node[1];
x node[2];
sx node[3];
rz(3.5*pi) node[0];
rz(3.5*pi) node[2];
rz(1.0910023932203043*pi) node[3];
ecr node[0],node[1];
sx node[3];
sx node[0];
x node[1];
rz(1.1534159686327479*pi) node[3];
rz(3.5*pi) node[1];
sx node[3];
ecr node[1],node[0];
ecr node[2],node[3];
x node[0];
sx node[1];
sx node[2];
x node[3];
rz(3.5*pi) node[0];
rz(3.5*pi) node[3];
ecr node[0],node[1];
ecr node[3],node[2];
sx node[0];
x node[1];
x node[2];
sx node[3];
ecr node[7],node[0];
rz(3.5*pi) node[1];
rz(3.5*pi) node[2];
sx node[0];
ecr node[2],node[3];
x node[7];
sx node[2];
sx node[3];
rz(3.5*pi) node[7];
ecr node[7],node[0];
ecr node[1],node[2];
ecr node[4],node[3];
x node[0];
sx node[1];
x node[2];
sx node[3];
sx node[4];
sx node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[2];
rz(1.1250895495795588*pi) node[4];
ecr node[0],node[7];
ecr node[2],node[1];
sx node[4];
sx node[0];
x node[1];
sx node[2];
rz(1.2826660066782642*pi) node[4];
x node[7];
rz(3.5*pi) node[1];
x node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
ecr node[1],node[2];
rz(3.5*pi) node[4];
x node[0];
sx node[1];
x node[2];
ecr node[4],node[3];
x node[7];
rz(3.5*pi) node[0];
rz(3.5*pi) node[2];
x node[3];
sx node[4];
rz(3.5*pi) node[7];
ecr node[2],node[1];
rz(3.5*pi) node[3];
sx node[1];
x node[2];
ecr node[3],node[4];
ecr node[0],node[1];
rz(3.5*pi) node[2];
sx node[3];
x node[4];
x node[0];
sx node[1];
rz(3.5*pi) node[4];
rz(3.5*pi) node[0];
ecr node[4],node[3];
ecr node[0],node[1];
sx node[3];
sx node[4];
sx node[0];
x node[1];
ecr node[2],node[3];
ecr node[5],node[4];
rz(3.5*pi) node[1];
x node[2];
sx node[3];
rz(2.0910898584108306*pi) node[4];
sx node[5];
ecr node[1],node[0];
rz(3.5*pi) node[2];
sx node[4];
rz(1.1374328525713358*pi) node[5];
x node[0];
sx node[1];
ecr node[2],node[3];
rz(0.8571692133646709*pi) node[4];
sx node[5];
rz(3.5*pi) node[0];
sx node[2];
x node[3];
sx node[4];
rz(1.2804724695905945*pi) node[5];
ecr node[0],node[1];
rz(3.5*pi) node[3];
rz(1.0*pi) node[4];
x node[5];
sx node[0];
x node[1];
ecr node[3],node[2];
sx node[4];
rz(3.5*pi) node[5];
ecr node[7],node[0];
rz(3.5*pi) node[1];
x node[2];
sx node[3];
ecr node[5],node[4];
sx node[0];
rz(3.5*pi) node[2];
x node[4];
sx node[5];
x node[7];
ecr node[2],node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
sx node[2];
x node[3];
ecr node[4],node[5];
x node[0];
ecr node[1],node[2];
rz(3.5*pi) node[3];
sx node[4];
x node[5];
sx node[7];
rz(3.5*pi) node[0];
sx node[1];
x node[2];
rz(3.5*pi) node[5];
ecr node[0],node[7];
rz(3.5*pi) node[2];
ecr node[5],node[4];
sx node[0];
ecr node[2],node[1];
sx node[4];
sx node[5];
x node[7];
x node[1];
sx node[2];
ecr node[3],node[4];
rz(3.5*pi) node[7];
ecr node[7],node[0];
rz(3.5*pi) node[1];
sx node[3];
x node[4];
x node[0];
ecr node[1],node[2];
rz(3.5*pi) node[4];
x node[7];
rz(3.5*pi) node[0];
sx node[1];
x node[2];
ecr node[4],node[3];
rz(3.5*pi) node[7];
rz(3.5*pi) node[2];
x node[3];
sx node[4];
ecr node[2],node[1];
rz(3.5*pi) node[3];
sx node[1];
x node[2];
ecr node[3],node[4];
ecr node[0],node[1];
rz(3.5*pi) node[2];
sx node[3];
x node[4];
x node[0];
sx node[1];
rz(3.5*pi) node[4];
rz(3.5*pi) node[0];
ecr node[4],node[3];
ecr node[0],node[1];
sx node[3];
x node[4];
sx node[0];
x node[1];
ecr node[2],node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[1];
x node[2];
sx node[3];
ecr node[4],node[5];
ecr node[1],node[0];
rz(3.5*pi) node[2];
sx node[4];
x node[5];
x node[0];
sx node[1];
ecr node[2],node[3];
rz(3.074338412312348*pi) node[4];
rz(3.5*pi) node[5];
rz(3.5*pi) node[0];
sx node[2];
x node[3];
sx node[4];
ecr node[0],node[1];
rz(3.5*pi) node[3];
rz(1.2520972504487566*pi) node[4];
sx node[0];
x node[1];
ecr node[3],node[2];
sx node[4];
ecr node[7],node[0];
rz(3.5*pi) node[1];
x node[2];
sx node[3];
ecr node[5],node[4];
sx node[0];
rz(3.5*pi) node[2];
x node[4];
sx node[5];
x node[7];
ecr node[2],node[3];
rz(3.5*pi) node[4];
rz(3.5*pi) node[7];
sx node[2];
x node[3];
ecr node[4],node[5];
ecr node[1],node[2];
rz(3.5*pi) node[3];
sx node[4];
x node[5];
sx node[1];
x node[2];
rz(3.5*pi) node[5];
rz(3.5*pi) node[2];
ecr node[5],node[4];
ecr node[2],node[1];
sx node[4];
sx node[5];
x node[1];
sx node[2];
ecr node[3],node[4];
rz(3.5*pi) node[1];
sx node[3];
sx node[4];
ecr node[1],node[2];
rz(3.251823542143825*pi) node[3];
sx node[1];
x node[2];
sx node[3];
rz(3.5*pi) node[2];
rz(1.1840140634844647*pi) node[3];
ecr node[2],node[1];
sx node[3];
x node[1];
x node[2];
rz(3.5*pi) node[1];
rz(3.5*pi) node[2];
ecr node[1],node[0];
ecr node[2],node[3];
x node[0];
sx node[1];
x node[2];
x node[3];
rz(3.5*pi) node[0];
rz(3.5*pi) node[2];
rz(3.5*pi) node[3];
ecr node[0],node[1];
ecr node[3],node[4];
sx node[0];
x node[1];
sx node[3];
sx node[4];
rz(3.5*pi) node[1];
ecr node[2],node[3];
ecr node[1],node[0];
sx node[2];
x node[3];
sx node[0];
x node[1];
rz(3.0708330979020224*pi) node[2];
rz(3.5*pi) node[3];
ecr node[7],node[0];
rz(3.5*pi) node[1];
sx node[2];
ecr node[3],node[4];
sx node[0];
rz(1.1018163954625053*pi) node[2];
sx node[3];
sx node[4];
x node[7];
ecr node[1],node[0];
sx node[2];
rz(3.5*pi) node[7];
x node[0];
x node[1];
ecr node[7],node[6];
rz(3.5*pi) node[0];
rz(3.5*pi) node[1];
x node[6];
sx node[7];
rz(3.5*pi) node[6];
ecr node[6],node[7];
sx node[6];
x node[7];
rz(3.5*pi) node[7];
ecr node[7],node[6];
x node[6];
rz(3.5*pi) node[6];
ecr node[6],node[5];
x node[5];
sx node[6];
rz(3.5*pi) node[5];
ecr node[5],node[6];
sx node[5];
x node[6];
rz(3.5*pi) node[6];
ecr node[6],node[5];
x node[5];
rz(3.5*pi) node[5];
ecr node[5],node[4];
x node[4];
sx node[5];
rz(3.5*pi) node[4];
rz(3.2577209999122054*pi) node[5];
ecr node[4],node[3];
sx node[5];
x node[3];
sx node[4];
rz(1.125370722251322*pi) node[5];
rz(3.5*pi) node[3];
ecr node[3],node[4];
sx node[3];
x node[4];
rz(3.5*pi) node[4];
ecr node[4],node[3];
x node[3];
rz(3.5*pi) node[3];
ecr node[3],node[2];
x node[2];
sx node[3];
rz(3.5*pi) node[2];
ecr node[2],node[3];
sx node[2];
x node[3];
rz(3.5*pi) node[3];
ecr node[3],node[2];
sx node[2];
ecr node[1],node[2];
sx node[1];
x node[2];
rz(3.100246844427821*pi) node[1];
rz(3.5*pi) node[2];
sx node[1];
rz(1.0756759886512781*pi) node[1];
sx node[1];
ecr node[2],node[1];
x node[1];
sx node[2];
rz(3.5*pi) node[1];
ecr node[1],node[2];
sx node[1];
x node[2];
rz(3.5*pi) node[2];
ecr node[2],node[1];
sx node[1];
ecr node[0],node[1];
sx node[0];
rz(3.5790043418634783*pi) node[1];
rz(3.002461916694121*pi) node[0];
sx node[1];
sx node[0];
rz(3.5*pi) node[1];
rz(1.0745970550103663*pi) node[0];
sx node[1];
rz(1.6837496243634187*pi) node[1];
barrier node[6],node[4],node[3],node[5],node[2],node[0],node[1];
measure node[6] -> meas[0];
measure node[4] -> meas[1];
measure node[3] -> meas[2];
measure node[5] -> meas[3];
measure node[2] -> meas[4];
measure node[0] -> meas[5];
measure node[1] -> meas[6];
