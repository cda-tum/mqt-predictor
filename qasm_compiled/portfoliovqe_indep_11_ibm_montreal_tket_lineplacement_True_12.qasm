OPENQASM 2.0;
include "qelib1.inc";

qreg node[26];
creg meas[11];
rz(0.5*pi) node[7];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(0.5*pi) node[15];
rz(0.5*pi) node[17];
rz(0.5*pi) node[18];
rz(0.5*pi) node[21];
sx node[23];
rz(0.5*pi) node[24];
rz(0.5*pi) node[25];
sx node[7];
sx node[10];
sx node[12];
sx node[13];
sx node[15];
sx node[17];
sx node[18];
sx node[21];
rz(3.6375659580456494*pi) node[23];
sx node[24];
sx node[25];
rz(0.5*pi) node[7];
rz(0.5*pi) node[10];
rz(0.5*pi) node[12];
rz(0.5*pi) node[13];
rz(2.5*pi) node[15];
rz(0.5*pi) node[17];
rz(2.5*pi) node[18];
rz(2.5*pi) node[21];
sx node[23];
rz(0.5*pi) node[24];
rz(2.5*pi) node[25];
sx node[7];
sx node[10];
sx node[12];
sx node[13];
sx node[15];
sx node[17];
sx node[18];
sx node[21];
rz(1.0*pi) node[23];
sx node[24];
sx node[25];
rz(0.7130988587328129*pi) node[7];
rz(0.6588503961119794*pi) node[10];
rz(1.4567085480523512*pi) node[12];
rz(1.4694585838779188*pi) node[13];
rz(3.6300116258489643*pi) node[15];
rz(1.254430266516229*pi) node[17];
rz(0.8411801109332693*pi) node[18];
rz(0.7270367410233866*pi) node[21];
rz(1.0097085255035614*pi) node[24];
rz(3.5100041885718243*pi) node[25];
cx node[23],node[24];
sx node[24];
rz(2.5*pi) node[24];
sx node[24];
rz(1.5*pi) node[24];
cx node[25],node[24];
cx node[24],node[25];
cx node[25],node[24];
cx node[23],node[24];
cx node[23],node[21];
cx node[25],node[24];
cx node[23],node[21];
sx node[24];
cx node[21],node[23];
rz(2.5*pi) node[24];
cx node[23],node[21];
sx node[24];
cx node[21],node[18];
rz(1.5*pi) node[24];
cx node[21],node[18];
cx node[25],node[24];
cx node[18],node[21];
cx node[24],node[25];
cx node[21],node[18];
cx node[25],node[24];
cx node[18],node[15];
cx node[24],node[23];
cx node[18],node[17];
cx node[24],node[23];
cx node[18],node[15];
cx node[23],node[24];
cx node[15],node[18];
cx node[24],node[23];
cx node[18],node[15];
cx node[23],node[21];
cx node[25],node[24];
cx node[15],node[12];
cx node[23],node[21];
sx node[24];
cx node[15],node[12];
cx node[21],node[23];
rz(2.5*pi) node[24];
cx node[12],node[15];
cx node[23],node[21];
sx node[24];
cx node[15],node[12];
cx node[21],node[18];
rz(1.5*pi) node[24];
cx node[12],node[10];
cx node[21],node[18];
cx node[25],node[24];
cx node[12],node[13];
cx node[18],node[21];
cx node[24],node[25];
cx node[12],node[10];
cx node[21],node[18];
cx node[25],node[24];
cx node[10],node[7];
cx node[18],node[17];
cx node[24],node[23];
cx node[12],node[10];
cx node[18],node[15];
cx node[24],node[23];
cx node[10],node[7];
sx node[12];
cx node[18],node[15];
cx node[23],node[24];
rz(1.4411706459857054*pi) node[12];
cx node[15],node[18];
cx node[24],node[23];
sx node[12];
cx node[18],node[15];
cx node[23],node[21];
cx node[25],node[24];
rz(1.0*pi) node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
cx node[10],node[12];
cx node[18],node[17];
cx node[21],node[23];
rz(2.5*pi) node[24];
cx node[12],node[10];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
cx node[10],node[12];
cx node[21],node[18];
rz(1.5*pi) node[24];
cx node[7],node[10];
cx node[15],node[12];
cx node[21],node[18];
cx node[25],node[24];
cx node[10],node[7];
cx node[12],node[15];
cx node[18],node[21];
cx node[24],node[25];
cx node[7],node[10];
cx node[15],node[12];
cx node[21],node[18];
cx node[25],node[24];
cx node[12],node[15];
cx node[18],node[17];
cx node[24],node[23];
cx node[12],node[13];
cx node[18],node[15];
cx node[23],node[24];
cx node[12],node[10];
cx node[18],node[15];
cx node[24],node[23];
rz(0.5*pi) node[12];
cx node[15],node[18];
cx node[23],node[24];
sx node[12];
cx node[18],node[15];
cx node[23],node[21];
cx node[25],node[24];
rz(2.5*pi) node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
sx node[12];
cx node[18],node[17];
cx node[21],node[23];
rz(2.5*pi) node[24];
rz(3.633400007382842*pi) node[12];
cx node[17],node[18];
cx node[23],node[21];
sx node[24];
cx node[12],node[10];
cx node[21],node[18];
rz(1.5*pi) node[24];
cx node[10],node[12];
cx node[21],node[18];
cx node[25],node[24];
cx node[12],node[10];
cx node[18],node[21];
cx node[24],node[25];
cx node[7],node[10];
cx node[13],node[12];
cx node[21],node[18];
cx node[25],node[24];
sx node[10];
cx node[12],node[13];
cx node[18],node[17];
cx node[24],node[23];
rz(2.5*pi) node[10];
cx node[13],node[12];
cx node[23],node[24];
sx node[10];
cx node[15],node[12];
cx node[24],node[23];
rz(1.5*pi) node[10];
cx node[12],node[15];
cx node[23],node[24];
cx node[7],node[10];
cx node[15],node[12];
cx node[23],node[21];
cx node[25],node[24];
cx node[10],node[7];
cx node[12],node[15];
cx node[23],node[21];
sx node[24];
cx node[7],node[10];
cx node[12],node[13];
cx node[18],node[15];
cx node[21],node[23];
rz(2.5*pi) node[24];
rz(0.5*pi) node[12];
cx node[18],node[15];
cx node[23],node[21];
sx node[24];
sx node[12];
cx node[15],node[18];
rz(1.5*pi) node[24];
rz(0.5*pi) node[12];
cx node[18],node[15];
cx node[25],node[24];
sx node[12];
cx node[17],node[18];
cx node[24],node[25];
rz(3.8246775703038596*pi) node[12];
cx node[18],node[17];
cx node[25],node[24];
cx node[10],node[12];
cx node[17],node[18];
cx node[24],node[23];
cx node[12],node[10];
cx node[21],node[18];
cx node[23],node[24];
cx node[10],node[12];
cx node[21],node[18];
cx node[24],node[23];
cx node[12],node[10];
cx node[18],node[21];
cx node[23],node[24];
cx node[7],node[10];
cx node[13],node[12];
cx node[21],node[18];
cx node[25],node[24];
sx node[10];
cx node[12],node[13];
cx node[18],node[17];
cx node[23],node[21];
sx node[24];
rz(2.5*pi) node[10];
cx node[13],node[12];
cx node[23],node[21];
rz(2.5*pi) node[24];
sx node[10];
cx node[15],node[12];
cx node[21],node[23];
sx node[24];
rz(1.5*pi) node[10];
rz(0.5*pi) node[15];
cx node[23],node[21];
rz(1.5*pi) node[24];
cx node[7],node[10];
sx node[15];
cx node[25],node[24];
cx node[10],node[7];
rz(0.5*pi) node[15];
cx node[24],node[25];
cx node[7],node[10];
sx node[15];
cx node[25],node[24];
rz(3.6194012786208423*pi) node[15];
cx node[24],node[23];
cx node[12],node[15];
cx node[23],node[24];
cx node[15],node[12];
cx node[24],node[23];
cx node[12],node[15];
cx node[23],node[24];
cx node[13],node[12];
cx node[18],node[15];
cx node[25],node[24];
cx node[10],node[12];
rz(0.5*pi) node[18];
sx node[24];
cx node[12],node[10];
sx node[18];
rz(2.5*pi) node[24];
cx node[10],node[12];
rz(2.5*pi) node[18];
sx node[24];
cx node[12],node[10];
sx node[18];
rz(1.5*pi) node[24];
cx node[7],node[10];
cx node[13],node[12];
rz(0.41423612454915626*pi) node[18];
cx node[25],node[24];
sx node[10];
cx node[12],node[13];
cx node[18],node[15];
cx node[24],node[25];
rz(2.5*pi) node[10];
cx node[13],node[12];
cx node[15],node[18];
cx node[25],node[24];
sx node[10];
cx node[18],node[15];
rz(1.5*pi) node[10];
cx node[12],node[15];
cx node[17],node[18];
cx node[7],node[10];
cx node[12],node[15];
cx node[18],node[17];
cx node[10],node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[7],node[10];
cx node[12],node[15];
cx node[21],node[18];
cx node[13],node[12];
cx node[21],node[18];
cx node[10],node[12];
cx node[18],node[21];
cx node[12],node[10];
cx node[21],node[18];
cx node[10],node[12];
cx node[18],node[17];
cx node[23],node[21];
cx node[12],node[10];
rz(0.5*pi) node[18];
cx node[23],node[21];
cx node[7],node[10];
cx node[13],node[12];
sx node[18];
cx node[21],node[23];
sx node[10];
cx node[12],node[13];
rz(2.5*pi) node[18];
cx node[23],node[21];
rz(2.5*pi) node[10];
cx node[13],node[12];
sx node[18];
cx node[24],node[23];
sx node[10];
rz(0.675761313833976*pi) node[18];
cx node[23],node[24];
rz(1.5*pi) node[10];
cx node[15],node[18];
cx node[24],node[23];
cx node[7],node[10];
cx node[18],node[15];
cx node[23],node[24];
cx node[10],node[7];
cx node[15],node[18];
cx node[25],node[24];
cx node[7],node[10];
cx node[18],node[15];
sx node[24];
cx node[12],node[15];
cx node[17],node[18];
rz(2.5*pi) node[24];
cx node[12],node[15];
cx node[18],node[17];
sx node[24];
cx node[15],node[12];
cx node[17],node[18];
rz(1.5*pi) node[24];
cx node[12],node[15];
cx node[21],node[18];
cx node[25],node[24];
cx node[13],node[12];
rz(0.5*pi) node[21];
cx node[24],node[25];
cx node[10],node[12];
sx node[21];
cx node[25],node[24];
cx node[12],node[10];
rz(0.5*pi) node[21];
cx node[10],node[12];
sx node[21];
cx node[12],node[10];
rz(3.9563244167105247*pi) node[21];
cx node[7],node[10];
cx node[13],node[12];
cx node[21],node[18];
sx node[10];
cx node[12],node[13];
cx node[18],node[21];
rz(2.5*pi) node[10];
cx node[13],node[12];
cx node[21],node[18];
sx node[10];
cx node[17],node[18];
cx node[23],node[21];
rz(1.5*pi) node[10];
cx node[15],node[18];
rz(0.5*pi) node[23];
cx node[7],node[10];
cx node[18],node[15];
sx node[23];
cx node[10],node[7];
cx node[15],node[18];
rz(0.5*pi) node[23];
cx node[7],node[10];
cx node[18],node[15];
sx node[23];
cx node[12],node[15];
cx node[17],node[18];
rz(1.1208143622016724*pi) node[23];
cx node[12],node[15];
cx node[18],node[17];
cx node[23],node[21];
cx node[15],node[12];
cx node[17],node[18];
cx node[21],node[23];
cx node[12],node[15];
cx node[23],node[21];
cx node[13],node[12];
cx node[18],node[21];
cx node[24],node[23];
cx node[10],node[12];
cx node[21],node[18];
rz(0.5*pi) node[24];
cx node[12],node[10];
cx node[18],node[21];
sx node[24];
cx node[10],node[12];
cx node[21],node[18];
rz(2.5*pi) node[24];
cx node[12],node[10];
cx node[17],node[18];
sx node[24];
cx node[7],node[10];
cx node[13],node[12];
cx node[15],node[18];
rz(1.0170590915546467*pi) node[24];
sx node[10];
cx node[12],node[13];
cx node[18],node[15];
cx node[23],node[24];
rz(2.5*pi) node[10];
cx node[13],node[12];
cx node[15],node[18];
cx node[24],node[23];
sx node[10];
cx node[18],node[15];
cx node[23],node[24];
rz(1.5*pi) node[10];
cx node[12],node[15];
cx node[17],node[18];
cx node[21],node[23];
cx node[25],node[24];
cx node[7],node[10];
cx node[12],node[15];
cx node[18],node[17];
cx node[23],node[21];
rz(2.033554813373288*pi) node[24];
rz(0.5*pi) node[25];
cx node[10],node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[21],node[23];
sx node[25];
cx node[7],node[10];
cx node[12],node[15];
cx node[23],node[21];
rz(2.5*pi) node[25];
cx node[13],node[12];
cx node[18],node[21];
sx node[25];
cx node[10],node[12];
cx node[21],node[18];
rz(0.24114907419384868*pi) node[25];
cx node[12],node[10];
cx node[18],node[21];
cx node[25],node[24];
cx node[10],node[12];
cx node[21],node[18];
cx node[24],node[25];
cx node[12],node[10];
cx node[17],node[18];
cx node[25],node[24];
cx node[7],node[10];
cx node[13],node[12];
cx node[15],node[18];
cx node[23],node[24];
sx node[10];
cx node[12],node[13];
cx node[18],node[15];
cx node[23],node[24];
rz(2.5*pi) node[10];
cx node[13],node[12];
cx node[15],node[18];
cx node[24],node[23];
sx node[10];
cx node[18],node[15];
cx node[23],node[24];
rz(1.5*pi) node[10];
cx node[12],node[15];
cx node[17],node[18];
cx node[21],node[23];
cx node[24],node[25];
cx node[7],node[10];
cx node[12],node[15];
cx node[18],node[17];
cx node[23],node[21];
sx node[24];
cx node[10],node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[21],node[23];
rz(1.0438445593933734*pi) node[24];
cx node[7],node[10];
cx node[12],node[15];
cx node[23],node[21];
sx node[24];
cx node[13],node[12];
cx node[18],node[21];
rz(1.0*pi) node[24];
cx node[10],node[12];
cx node[21],node[18];
cx node[25],node[24];
cx node[12],node[10];
cx node[18],node[21];
cx node[24],node[25];
cx node[10],node[12];
cx node[21],node[18];
cx node[25],node[24];
cx node[12],node[10];
cx node[17],node[18];
cx node[23],node[24];
cx node[7],node[10];
cx node[13],node[12];
cx node[15],node[18];
rz(0.5*pi) node[23];
sx node[10];
cx node[12],node[13];
cx node[18],node[15];
sx node[23];
rz(2.5*pi) node[10];
cx node[13],node[12];
cx node[15],node[18];
rz(2.5*pi) node[23];
sx node[10];
cx node[18],node[15];
sx node[23];
rz(1.5*pi) node[10];
cx node[12],node[15];
cx node[17],node[18];
rz(1.2429765845424572*pi) node[23];
cx node[7],node[10];
cx node[12],node[15];
cx node[18],node[17];
cx node[23],node[24];
cx node[10],node[7];
cx node[15],node[12];
cx node[17],node[18];
cx node[24],node[23];
cx node[7],node[10];
cx node[12],node[15];
cx node[23],node[24];
cx node[13],node[12];
cx node[21],node[23];
cx node[25],node[24];
cx node[10],node[12];
rz(0.5*pi) node[21];
sx node[24];
cx node[12],node[10];
sx node[21];
rz(2.5*pi) node[24];
cx node[10],node[12];
rz(2.5*pi) node[21];
sx node[24];
cx node[12],node[10];
sx node[21];
rz(1.5*pi) node[24];
cx node[7],node[10];
cx node[13],node[12];
rz(0.4950886411967861*pi) node[21];
cx node[25],node[24];
sx node[10];
cx node[12],node[13];
cx node[23],node[21];
cx node[24],node[25];
rz(2.5*pi) node[10];
cx node[13],node[12];
cx node[21],node[23];
cx node[25],node[24];
sx node[10];
cx node[23],node[21];
rz(1.5*pi) node[10];
cx node[18],node[21];
cx node[24],node[23];
cx node[7],node[10];
rz(0.5*pi) node[18];
cx node[23],node[24];
cx node[10],node[7];
sx node[18];
cx node[24],node[23];
cx node[7],node[10];
rz(0.5*pi) node[18];
cx node[23],node[24];
sx node[18];
cx node[25],node[24];
rz(0.4421416484417906*pi) node[18];
sx node[24];
cx node[21],node[18];
rz(2.5*pi) node[24];
cx node[18],node[21];
sx node[24];
cx node[21],node[18];
rz(1.5*pi) node[24];
cx node[17],node[18];
cx node[23],node[21];
cx node[25],node[24];
cx node[15],node[18];
rz(0.5*pi) node[17];
cx node[23],node[21];
cx node[24],node[25];
rz(0.5*pi) node[15];
sx node[17];
cx node[21],node[23];
cx node[25],node[24];
sx node[15];
rz(0.5*pi) node[17];
cx node[23],node[21];
rz(0.5*pi) node[15];
sx node[17];
cx node[24],node[23];
sx node[15];
rz(0.6468709171761519*pi) node[17];
cx node[23],node[24];
rz(0.3088677073346512*pi) node[15];
cx node[24],node[23];
cx node[18],node[15];
cx node[23],node[24];
cx node[15],node[18];
cx node[25],node[24];
cx node[18],node[15];
sx node[24];
cx node[12],node[15];
cx node[17],node[18];
rz(2.5*pi) node[24];
rz(0.5*pi) node[12];
cx node[18],node[17];
sx node[24];
sx node[12];
cx node[17],node[18];
rz(1.5*pi) node[24];
rz(0.5*pi) node[12];
cx node[21],node[18];
cx node[25],node[24];
sx node[12];
cx node[21],node[18];
cx node[24],node[25];
rz(0.9484271453547624*pi) node[12];
cx node[18],node[21];
cx node[25],node[24];
cx node[12],node[15];
cx node[21],node[18];
cx node[15],node[12];
cx node[18],node[17];
cx node[23],node[21];
cx node[12],node[15];
cx node[23],node[21];
cx node[13],node[12];
cx node[18],node[15];
cx node[21],node[23];
cx node[10],node[12];
rz(0.5*pi) node[13];
cx node[18],node[15];
cx node[23],node[21];
rz(0.5*pi) node[10];
sx node[13];
cx node[15],node[18];
cx node[24],node[23];
sx node[10];
rz(2.5*pi) node[13];
cx node[18],node[15];
cx node[23],node[24];
rz(2.5*pi) node[10];
sx node[13];
cx node[17],node[18];
cx node[24],node[23];
sx node[10];
rz(1.4287118092467295*pi) node[13];
cx node[18],node[17];
cx node[23],node[24];
rz(0.5856448765274083*pi) node[10];
cx node[17],node[18];
cx node[25],node[24];
cx node[12],node[10];
cx node[21],node[18];
sx node[24];
cx node[10],node[12];
cx node[21],node[18];
rz(2.5*pi) node[24];
cx node[12],node[10];
cx node[18],node[21];
sx node[24];
cx node[7],node[10];
cx node[13],node[12];
cx node[21],node[18];
rz(1.5*pi) node[24];
rz(0.5*pi) node[7];
rz(2.286333861523252*pi) node[10];
cx node[12],node[13];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
sx node[7];
cx node[13],node[12];
cx node[23],node[21];
cx node[24],node[25];
rz(0.5*pi) node[7];
cx node[15],node[12];
cx node[21],node[23];
cx node[25],node[24];
sx node[7];
cx node[12],node[15];
cx node[23],node[21];
rz(1.2974479348019492*pi) node[7];
cx node[15],node[12];
cx node[24],node[23];
cx node[7],node[10];
cx node[12],node[15];
cx node[23],node[24];
cx node[10],node[7];
cx node[12],node[13];
cx node[18],node[15];
cx node[24],node[23];
cx node[7],node[10];
cx node[18],node[15];
cx node[23],node[24];
cx node[12],node[10];
cx node[15],node[18];
cx node[25],node[24];
cx node[12],node[10];
cx node[18],node[15];
sx node[24];
cx node[10],node[12];
cx node[17],node[18];
rz(2.5*pi) node[24];
cx node[12],node[10];
cx node[18],node[17];
sx node[24];
cx node[10],node[7];
cx node[13],node[12];
cx node[17],node[18];
rz(1.5*pi) node[24];
sx node[10];
cx node[12],node[13];
cx node[21],node[18];
cx node[25],node[24];
rz(3.590802001961054*pi) node[10];
cx node[13],node[12];
cx node[21],node[18];
cx node[24],node[25];
sx node[10];
cx node[15],node[12];
cx node[18],node[21];
cx node[25],node[24];
rz(1.0*pi) node[10];
cx node[12],node[15];
cx node[21],node[18];
cx node[7],node[10];
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
cx node[12],node[10];
cx node[18],node[15];
cx node[23],node[21];
sx node[12];
cx node[15],node[18];
cx node[24],node[23];
rz(2.7588221708545984*pi) node[12];
cx node[18],node[15];
cx node[23],node[24];
sx node[12];
cx node[17],node[18];
cx node[24],node[23];
rz(1.0*pi) node[12];
cx node[18],node[17];
cx node[23],node[24];
cx node[13],node[12];
cx node[17],node[18];
cx node[25],node[24];
cx node[12],node[13];
cx node[21],node[18];
sx node[24];
cx node[13],node[12];
cx node[21],node[18];
rz(2.5*pi) node[24];
cx node[15],node[12];
cx node[18],node[21];
sx node[24];
cx node[15],node[12];
cx node[21],node[18];
rz(1.5*pi) node[24];
cx node[12],node[15];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
cx node[15],node[12];
cx node[23],node[21];
cx node[24],node[25];
cx node[12],node[10];
cx node[18],node[15];
cx node[21],node[23];
cx node[25],node[24];
sx node[12];
cx node[18],node[15];
cx node[23],node[21];
rz(0.9283373090419567*pi) node[12];
cx node[15],node[18];
cx node[24],node[23];
sx node[12];
cx node[18],node[15];
cx node[23],node[24];
rz(1.0*pi) node[12];
cx node[17],node[18];
cx node[24],node[23];
cx node[10],node[12];
cx node[18],node[17];
cx node[23],node[24];
cx node[12],node[10];
cx node[17],node[18];
cx node[25],node[24];
cx node[10],node[12];
cx node[21],node[18];
sx node[24];
cx node[15],node[12];
cx node[21],node[18];
rz(2.5*pi) node[24];
sx node[15];
cx node[18],node[21];
sx node[24];
rz(3.865103678155841*pi) node[15];
cx node[21],node[18];
rz(1.5*pi) node[24];
sx node[15];
cx node[18],node[17];
cx node[23],node[21];
cx node[25],node[24];
rz(1.0*pi) node[15];
cx node[23],node[21];
cx node[24],node[25];
cx node[12],node[15];
cx node[21],node[23];
cx node[25],node[24];
cx node[15],node[12];
cx node[23],node[21];
cx node[12],node[15];
cx node[24],node[23];
cx node[18],node[15];
cx node[23],node[24];
sx node[18];
cx node[24],node[23];
rz(3.223790567355353*pi) node[18];
cx node[23],node[24];
sx node[18];
cx node[25],node[24];
rz(1.0*pi) node[18];
sx node[24];
cx node[17],node[18];
rz(2.5*pi) node[24];
cx node[18],node[17];
sx node[24];
cx node[17],node[18];
rz(1.5*pi) node[24];
cx node[21],node[18];
cx node[25],node[24];
cx node[21],node[18];
cx node[24],node[25];
cx node[18],node[21];
cx node[25],node[24];
cx node[21],node[18];
cx node[18],node[15];
cx node[23],node[21];
sx node[18];
cx node[23],node[21];
rz(1.9418857151861553*pi) node[18];
cx node[21],node[23];
sx node[18];
cx node[23],node[21];
rz(1.0*pi) node[18];
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
sx node[23];
rz(3.1580118610844403*pi) node[21];
rz(2.5*pi) node[23];
sx node[21];
sx node[23];
rz(1.0*pi) node[21];
rz(1.5*pi) node[23];
cx node[18],node[21];
cx node[21],node[18];
cx node[18],node[21];
cx node[21],node[23];
cx node[23],node[21];
cx node[21],node[23];
cx node[24],node[23];
sx node[24];
rz(3.6912691839945153*pi) node[24];
sx node[24];
rz(1.0*pi) node[24];
cx node[25],node[24];
cx node[24],node[25];
cx node[25],node[24];
cx node[24],node[23];
cx node[21],node[23];
sx node[24];
sx node[21];
rz(0.03905839503359687*pi) node[23];
rz(2.556098759056278*pi) node[24];
rz(2.481957351236124*pi) node[21];
sx node[23];
sx node[24];
sx node[21];
rz(0.5*pi) node[23];
rz(1.0*pi) node[24];
rz(1.0*pi) node[21];
sx node[23];
rz(1.5*pi) node[23];
barrier node[7],node[13],node[10],node[12],node[17],node[15],node[18],node[25],node[24],node[21],node[23];
measure node[7] -> meas[0];
measure node[13] -> meas[1];
measure node[10] -> meas[2];
measure node[12] -> meas[3];
measure node[17] -> meas[4];
measure node[15] -> meas[5];
measure node[18] -> meas[6];
measure node[25] -> meas[7];
measure node[24] -> meas[8];
measure node[21] -> meas[9];
measure node[23] -> meas[10];
