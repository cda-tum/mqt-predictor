OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[9];
sx q[62];
rz(-2.3481303) q[62];
sx q[62];
rz(-2.8939574) q[62];
sx q[63];
rz(-2.6966178) q[63];
sx q[63];
rz(-2.4993623) q[63];
sx q[72];
rz(-2.8620597) q[72];
sx q[72];
rz(-2.6911055) q[72];
sx q[81];
rz(-2.1775173) q[81];
sx q[81];
rz(-2.9620301) q[81];
cx q[81],q[72];
sx q[82];
rz(-2.2099121) q[82];
sx q[82];
rz(-3.011925) q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
rz(-pi) q[81];
sx q[81];
rz(pi/2) q[81];
rz(0.65944553) q[82];
sx q[82];
rz(pi/2) q[82];
rz(0.77376992) q[83];
sx q[83];
rz(-2.9684283) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[82],q[83];
sx q[82];
rz(-pi/2) q[82];
sx q[82];
rz(pi/2) q[83];
cx q[82],q[83];
rz(pi/2) q[82];
sx q[82];
rz(1.7375717) q[82];
sx q[82];
cx q[81],q[82];
sx q[81];
rz(-pi/2) q[81];
sx q[81];
rz(pi/2) q[82];
cx q[81],q[82];
rz(-pi) q[81];
sx q[81];
rz(-pi) q[81];
cx q[72],q[81];
rz(-pi) q[81];
sx q[81];
rz(pi/2) q[81];
rz(-pi/2) q[82];
sx q[82];
rz(-pi) q[82];
sx q[83];
rz(2.5721449) q[83];
sx q[83];
rz(pi/2) q[83];
rz(0.83011342) q[92];
sx q[92];
rz(-3.0345969) q[92];
sx q[92];
rz(-pi/2) q[92];
cx q[83],q[92];
sx q[83];
rz(-pi/2) q[83];
sx q[83];
rz(pi/2) q[92];
cx q[83],q[92];
rz(-pi/2) q[83];
sx q[83];
rz(1.6870658) q[83];
sx q[83];
cx q[82],q[83];
sx q[82];
rz(-pi/2) q[82];
sx q[82];
rz(pi/2) q[83];
cx q[82],q[83];
rz(-pi) q[82];
sx q[82];
rz(-pi/2) q[82];
cx q[81],q[82];
sx q[81];
rz(-pi/2) q[81];
sx q[81];
rz(pi/2) q[82];
cx q[81],q[82];
rz(-pi) q[81];
sx q[81];
rz(-pi) q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(-pi/2) q[82];
sx q[82];
rz(-pi) q[82];
rz(-pi/2) q[83];
sx q[83];
rz(-pi) q[83];
sx q[92];
rz(-2.6314933) q[92];
sx q[92];
rz(pi/2) q[92];
rz(0.32666598) q[102];
sx q[102];
rz(-2.9952147) q[102];
sx q[102];
rz(pi/2) q[102];
cx q[92],q[102];
rz(pi/2) q[102];
sx q[92];
rz(-pi/2) q[92];
sx q[92];
cx q[92],q[102];
rz(-pi) q[102];
sx q[102];
rz(-0.16819634) q[102];
rz(pi/2) q[92];
sx q[92];
rz(1.6201727) q[92];
sx q[92];
cx q[83],q[92];
sx q[83];
rz(-pi/2) q[83];
sx q[83];
rz(pi/2) q[92];
cx q[83],q[92];
rz(-pi) q[83];
sx q[83];
rz(-pi/2) q[83];
cx q[82],q[83];
sx q[82];
rz(-pi/2) q[82];
sx q[82];
rz(pi/2) q[83];
cx q[82],q[83];
rz(-pi) q[82];
sx q[82];
rz(-pi) q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
rz(-pi) q[62];
sx q[62];
rz(pi/2) q[62];
x q[81];
rz(-pi) q[83];
sx q[83];
rz(-pi/2) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
rz(-pi) q[92];
sx q[92];
rz(-pi/2) q[92];
sx q[103];
rz(-2.4252294) q[103];
sx q[103];
rz(-2.8336541) q[103];
cx q[102],q[103];
cx q[102],q[92];
rz(pi/2) q[103];
cx q[92],q[102];
cx q[102],q[92];
rz(-pi) q[102];
sx q[102];
rz(pi/2) q[102];
cx q[102],q[103];
sx q[102];
rz(-pi/2) q[102];
sx q[102];
rz(pi/2) q[103];
cx q[102],q[103];
rz(-pi) q[102];
sx q[102];
rz(-pi/2) q[102];
rz(-pi) q[103];
sx q[103];
rz(-pi/2) q[103];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
rz(pi/2) q[82];
sx q[82];
rz(-pi) q[82];
cx q[81],q[82];
sx q[81];
rz(-pi/2) q[81];
sx q[81];
rz(pi/2) q[82];
cx q[81],q[82];
rz(pi/2) q[81];
sx q[81];
cx q[81],q[72];
sx q[81];
rz(-3.013546) q[81];
sx q[81];
rz(-2.7623643) q[81];
rz(pi/2) q[82];
sx q[82];
rz(-pi) q[83];
sx q[83];
rz(pi/2) q[83];
rz(-pi) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[92],q[102];
rz(pi/2) q[102];
sx q[92];
rz(-pi/2) q[92];
sx q[92];
cx q[92],q[102];
rz(-pi) q[102];
sx q[102];
rz(-pi/2) q[102];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
rz(-pi) q[92];
sx q[92];
rz(-pi/2) q[92];
cx q[83],q[92];
sx q[83];
rz(-pi/2) q[83];
sx q[83];
rz(pi/2) q[92];
cx q[83],q[92];
rz(-pi) q[83];
sx q[83];
rz(-pi) q[83];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[72];
cx q[62],q[72];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[72];
cx q[62],q[72];
rz(-pi) q[62];
sx q[62];
rz(-pi) q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-pi) q[62];
sx q[62];
rz(pi/2) q[62];
rz(-pi) q[72];
sx q[72];
rz(-pi/2) q[72];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(-pi) q[81];
sx q[81];
rz(pi/2) q[81];
x q[82];
x q[83];
rz(-pi) q[92];
sx q[92];
rz(-pi/2) q[92];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[103],q[102];
cx q[102],q[103];
cx q[103],q[102];
rz(pi/2) q[102];
sx q[102];
rz(-pi) q[102];
rz(pi/2) q[92];
sx q[92];
rz(-pi) q[92];
cx q[83],q[92];
sx q[83];
rz(-pi/2) q[83];
sx q[83];
rz(pi/2) q[92];
cx q[83],q[92];
rz(pi/2) q[83];
sx q[83];
rz(-1.9560108) q[83];
sx q[83];
cx q[82],q[83];
sx q[82];
rz(-pi/2) q[82];
sx q[82];
rz(pi/2) q[83];
cx q[82],q[83];
rz(-0.30475657) q[82];
sx q[82];
rz(-1.4497251) q[82];
sx q[82];
rz(-2.1735899) q[82];
cx q[81],q[82];
sx q[81];
rz(-pi/2) q[81];
sx q[81];
rz(pi/2) q[82];
cx q[81],q[82];
x q[81];
rz(-pi/2) q[81];
rz(-pi/2) q[82];
sx q[82];
rz(-pi) q[82];
rz(pi/2) q[83];
sx q[83];
rz(-pi) q[83];
rz(pi/2) q[92];
sx q[92];
x q[92];
cx q[92],q[102];
rz(pi/2) q[102];
sx q[92];
rz(-pi/2) q[92];
sx q[92];
cx q[92],q[102];
rz(pi/2) q[102];
sx q[102];
cx q[103],q[102];
rz(-1.5620713) q[103];
sx q[103];
rz(-pi) q[103];
rz(-pi/2) q[92];
sx q[92];
rz(-1.563751) q[92];
sx q[92];
cx q[83],q[92];
sx q[83];
rz(-pi/2) q[83];
sx q[83];
rz(pi/2) q[92];
cx q[83],q[92];
rz(-2.7280249) q[83];
sx q[83];
rz(-1.5736277) q[83];
sx q[83];
rz(0.64522567) q[83];
cx q[82],q[83];
sx q[82];
rz(-pi/2) q[82];
sx q[82];
rz(pi/2) q[83];
cx q[82],q[83];
rz(-pi) q[82];
sx q[82];
rz(-pi/2) q[82];
cx q[81],q[82];
sx q[81];
rz(-pi/2) q[81];
sx q[81];
rz(pi/2) q[82];
cx q[81],q[82];
rz(-pi) q[81];
sx q[81];
rz(-pi) q[81];
rz(-pi) q[82];
sx q[82];
rz(-pi/2) q[82];
rz(-pi) q[83];
sx q[83];
rz(-pi/2) q[83];
rz(pi/2) q[92];
sx q[92];
rz(-pi) q[92];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
rz(-pi) q[102];
x q[102];
cx q[102],q[103];
sx q[102];
rz(-pi/2) q[102];
sx q[102];
rz(pi/2) q[103];
cx q[102],q[103];
rz(3.1368201) q[102];
sx q[102];
rz(-1.570838) q[102];
sx q[102];
rz(-2.7800459) q[102];
rz(pi/2) q[103];
sx q[103];
rz(-pi) q[103];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[72];
cx q[62],q[72];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[72];
cx q[62],q[72];
rz(-pi) q[62];
sx q[62];
rz(-pi) q[62];
cx q[63],q[62];
rz(-pi) q[72];
sx q[72];
rz(-pi/2) q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(1.5662585) q[72];
sx q[72];
rz(-pi/2) q[72];
rz(-pi) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[92],q[102];
rz(pi/2) q[102];
sx q[92];
rz(-pi/2) q[92];
sx q[92];
cx q[92],q[102];
rz(-pi) q[102];
sx q[102];
rz(-pi/2) q[102];
cx q[103],q[102];
cx q[102],q[103];
cx q[103],q[102];
rz(-pi/2) q[102];
rz(-pi) q[92];
sx q[92];
rz(-pi) q[92];
cx q[83],q[92];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[83],q[92];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
sx q[92];
rz(pi/2) q[92];
cx q[92],q[102];
rz(pi/2) q[102];
sx q[92];
rz(-pi/2) q[92];
sx q[92];
cx q[92],q[102];
rz(-pi/2) q[102];
sx q[102];
rz(-1.7112861) q[102];
sx q[102];
rz(0.4401768) q[102];
cx q[103],q[102];
rz(-pi) q[92];
sx q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
rz(pi/2) q[81];
cx q[72],q[81];
sx q[72];
rz(-pi/2) q[72];
sx q[72];
rz(pi/2) q[81];
cx q[72],q[81];
sx q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
sx q[63];
rz(-2.3360938) q[63];
sx q[63];
rz(-2.5753157) q[63];
cx q[72],q[62];
sx q[62];
rz(-2.7137204) q[62];
sx q[62];
rz(-2.196435) q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
sx q[72];
rz(-3.0322729) q[72];
sx q[72];
rz(-2.8088201) q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
x q[72];
rz(1.9430012) q[81];
sx q[81];
rz(-1.5724466) q[81];
sx q[81];
rz(-1.3123493) q[81];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[103],q[102];
cx q[102],q[103];
cx q[103],q[102];
cx q[83],q[92];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[83],q[92];
cx q[102],q[92];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(pi/2) q[81];
sx q[81];
rz(-pi) q[81];
cx q[72],q[81];
sx q[72];
rz(-pi/2) q[72];
sx q[72];
rz(pi/2) q[81];
cx q[72],q[81];
rz(pi/2) q[72];
sx q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[63],q[62];
cx q[62],q[63];
x q[62];
sx q[63];
rz(-2.2902865) q[63];
sx q[63];
rz(-3.0593655) q[63];
x q[72];
rz(pi/2) q[81];
sx q[81];
cx q[82],q[83];
cx q[92],q[102];
cx q[102],q[92];
cx q[103],q[102];
cx q[83],q[92];
cx q[92],q[83];
cx q[102],q[92];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[81],q[82];
cx q[82],q[81];
rz(-pi) q[81];
sx q[81];
cx q[72],q[81];
sx q[72];
rz(-pi/2) q[72];
sx q[72];
rz(pi/2) q[81];
cx q[72],q[81];
rz(pi/2) q[72];
sx q[72];
rz(2.9116803) q[72];
sx q[72];
cx q[62],q[72];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[72];
cx q[62],q[72];
rz(2.9968409) q[62];
sx q[62];
rz(-1.6045446) q[62];
sx q[62];
rz(-1.2761282) q[62];
cx q[62],q[63];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[63];
sx q[63];
rz(-pi) q[63];
rz(-pi/2) q[72];
sx q[72];
rz(-pi) q[72];
rz(pi/2) q[81];
sx q[81];
x q[82];
rz(pi/2) q[83];
sx q[83];
rz(-pi) q[83];
cx q[82],q[83];
sx q[82];
rz(-pi/2) q[82];
sx q[82];
rz(pi/2) q[83];
cx q[82],q[83];
rz(-pi) q[82];
sx q[82];
rz(-pi/2) q[82];
cx q[81],q[82];
sx q[81];
rz(-pi/2) q[81];
sx q[81];
rz(pi/2) q[82];
cx q[81],q[82];
rz(pi/2) q[81];
sx q[81];
rz(0.16625492) q[81];
sx q[81];
cx q[72],q[81];
sx q[72];
rz(-pi/2) q[72];
sx q[72];
rz(pi/2) q[81];
cx q[72],q[81];
rz(0.097597029) q[72];
sx q[72];
rz(-2.0968969) q[72];
sx q[72];
rz(1.5737256) q[72];
cx q[62],q[72];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[72];
cx q[62],q[72];
sx q[62];
cx q[62],q[63];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[63];
cx q[62],q[63];
rz(-pi) q[63];
sx q[63];
rz(-pi/2) q[63];
rz(-pi/2) q[72];
sx q[72];
rz(-pi) q[72];
rz(-pi/2) q[81];
sx q[81];
rz(-pi) q[81];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[83];
sx q[83];
cx q[92],q[102];
cx q[102],q[92];
cx q[103],q[102];
cx q[92],q[102];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[102],q[92];
cx q[92],q[102];
cx q[102],q[92];
cx q[103],q[102];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[83],q[92];
cx q[102],q[92];
cx q[103],q[102];
cx q[102],q[103];
cx q[103],q[102];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[102],q[92];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[92];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[81],q[82];
cx q[82],q[81];
sx q[81];
rz(-2.3888871) q[81];
sx q[81];
rz(-1.2226192) q[81];
cx q[72],q[81];
sx q[72];
rz(-pi/2) q[72];
sx q[72];
rz(pi/2) q[81];
cx q[72],q[81];
rz(-pi) q[72];
sx q[72];
rz(-pi/2) q[72];
cx q[62],q[72];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[72];
cx q[62],q[72];
sx q[62];
cx q[62],q[63];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[63];
cx q[62],q[63];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[63];
sx q[63];
rz(-pi) q[72];
sx q[72];
rz(-pi/2) q[72];
rz(-pi) q[81];
sx q[81];
rz(-pi/2) q[81];
rz(-pi) q[82];
x q[82];
cx q[92],q[102];
cx q[102],q[92];
cx q[92],q[102];
cx q[103],q[102];
cx q[92],q[83];
cx q[83],q[92];
cx q[92],q[83];
rz(1.5747252) q[83];
sx q[83];
rz(-pi) q[83];
cx q[82],q[83];
sx q[82];
rz(-pi/2) q[82];
sx q[82];
rz(pi/2) q[83];
cx q[82],q[83];
rz(-2.9517262) q[82];
sx q[82];
rz(-1.5700548) q[82];
sx q[82];
rz(-0.97382735) q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
cx q[72],q[81];
cx q[72],q[62];
cx q[62],q[72];
cx q[62],q[63];
rz(-pi) q[62];
sx q[62];
rz(pi/2) q[62];
rz(-pi) q[72];
sx q[72];
rz(pi/2) q[72];
rz(-pi) q[81];
sx q[81];
rz(pi/2) q[81];
rz(-pi) q[82];
sx q[82];
rz(pi/2) q[82];
rz(pi/2) q[83];
sx q[83];
rz(-pi) q[83];
cx q[92],q[102];
cx q[102],q[103];
cx q[103],q[102];
cx q[102],q[103];
rz(1.3669141) q[102];
sx q[102];
rz(-pi) q[102];
cx q[83],q[92];
cx q[92],q[83];
sx q[83];
rz(-3.1168271) q[83];
sx q[83];
rz(-0.89371485) q[83];
cx q[82],q[83];
sx q[82];
rz(-pi/2) q[82];
sx q[82];
rz(pi/2) q[83];
cx q[82],q[83];
rz(-pi) q[82];
sx q[82];
rz(-pi/2) q[82];
cx q[81],q[82];
sx q[81];
rz(-pi/2) q[81];
sx q[81];
rz(pi/2) q[82];
cx q[81],q[82];
rz(-pi) q[81];
sx q[81];
rz(-pi/2) q[81];
cx q[72],q[81];
sx q[72];
rz(-pi/2) q[72];
sx q[72];
rz(pi/2) q[81];
cx q[72],q[81];
rz(-pi) q[72];
sx q[72];
rz(-pi/2) q[72];
cx q[62],q[72];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[72];
cx q[62],q[72];
rz(-pi) q[62];
sx q[62];
rz(-pi) q[62];
rz(-pi) q[72];
sx q[72];
rz(-pi/2) q[72];
rz(-pi) q[81];
sx q[81];
rz(-pi/2) q[81];
rz(-pi) q[82];
sx q[82];
rz(-pi/2) q[82];
rz(-pi) q[83];
sx q[83];
rz(-pi/2) q[83];
rz(-pi) q[92];
x q[92];
cx q[92],q[102];
rz(pi/2) q[102];
sx q[92];
rz(-pi/2) q[92];
sx q[92];
cx q[92],q[102];
rz(pi/2) q[102];
sx q[102];
rz(-pi) q[102];
cx q[103],q[102];
sx q[102];
rz(-2.7162728) q[102];
sx q[102];
rz(-2.6650323) q[102];
sx q[103];
rz(-3.0771157) q[103];
sx q[103];
rz(-1.5592898) q[103];
rz(-2.2329244) q[92];
sx q[92];
rz(-1.7324384) q[92];
sx q[92];
rz(-1.2265614) q[92];
cx q[92],q[83];
cx q[83],q[92];
cx q[102],q[92];
cx q[83],q[82];
cx q[82],q[83];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
cx q[72],q[81];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[63],q[62];
rz(-pi) q[62];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[72];
cx q[62],q[72];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[72];
cx q[62],q[72];
rz(-pi) q[62];
sx q[62];
rz(-pi) q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-pi) q[62];
sx q[62];
rz(pi/2) q[62];
rz(-pi) q[72];
sx q[72];
rz(-pi/2) q[72];
cx q[92],q[102];
rz(-1.5089947) q[102];
sx q[102];
rz(-pi/2) q[102];
cx q[102],q[103];
sx q[102];
rz(-pi/2) q[102];
sx q[102];
rz(pi/2) q[103];
cx q[102],q[103];
sx q[102];
rz(-pi/2) q[102];
rz(1.4799337) q[103];
sx q[103];
rz(-1.5764112) q[103];
sx q[103];
rz(2.3955101) q[103];
cx q[92],q[83];
cx q[83],q[92];
cx q[83],q[82];
cx q[82],q[83];
cx q[82],q[81];
cx q[81],q[82];
sx q[82];
rz(-pi/2) q[82];
rz(pi/2) q[83];
sx q[83];
rz(pi/2) q[83];
rz(-1.6376169) q[92];
sx q[92];
rz(pi/2) q[92];
cx q[92],q[102];
rz(pi/2) q[102];
sx q[92];
rz(-pi/2) q[92];
sx q[92];
cx q[92],q[102];
rz(-1.9337788) q[102];
sx q[102];
rz(-1.5470398) q[102];
sx q[102];
rz(2.5856625) q[102];
rz(-pi) q[92];
sx q[92];
rz(-pi/2) q[92];
cx q[83],q[92];
sx q[83];
rz(-pi/2) q[83];
sx q[83];
rz(pi/2) q[92];
cx q[83],q[92];
rz(-pi) q[83];
sx q[83];
rz(pi/2) q[83];
cx q[82],q[83];
sx q[82];
rz(-pi/2) q[82];
sx q[82];
rz(pi/2) q[83];
cx q[82],q[83];
sx q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[72];
cx q[72],q[81];
rz(pi/2) q[72];
cx q[62],q[72];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[72];
cx q[62],q[72];
rz(-pi) q[62];
sx q[62];
rz(-pi) q[62];
cx q[63],q[62];
sx q[72];
rz(2.8471093) q[72];
sx q[72];
rz(-pi/2) q[72];
cx q[82],q[81];
cx q[81],q[82];
rz(-pi/2) q[81];
cx q[72],q[81];
sx q[72];
rz(-pi/2) q[72];
sx q[72];
rz(pi/2) q[81];
cx q[72],q[81];
rz(-pi) q[72];
sx q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
sx q[63];
rz(-2.8906389) q[63];
sx q[63];
rz(-2.2019744) q[63];
cx q[72],q[62];
sx q[62];
rz(-2.5908445) q[62];
sx q[62];
rz(-2.5843295) q[62];
sx q[72];
rz(-2.4348029) q[72];
sx q[72];
rz(-2.6327055) q[72];
rz(-1.7057983) q[81];
sx q[81];
rz(-1.6115958) q[81];
sx q[81];
rz(2.2433171) q[81];
sx q[82];
rz(-3.1106537) q[82];
sx q[82];
rz(-2.6154099) q[82];
rz(pi/2) q[83];
sx q[83];
rz(-2.0030053) q[83];
sx q[83];
rz(0.35193369) q[83];
rz(-3.0254755) q[92];
sx q[92];
rz(-2.9598914) q[92];
barrier q[55],q[52],q[119],q[116],q[61],q[6],q[125],q[70],q[15],q[79],q[24],q[88],q[21],q[85],q[30],q[94],q[39],q[63],q[48],q[45],q[112],q[57],q[109],q[54],q[121],q[118],q[62],q[8],q[92],q[17],q[103],q[14],q[78],q[23],q[87],q[32],q[96],q[41],q[38],q[105],q[50],q[47],q[114],q[111],q[56],q[1],q[120],q[65],q[10],q[74],q[7],q[71],q[16],q[80],q[25],q[89],q[34],q[98],q[43],q[40],q[107],q[104],q[49],q[113],q[58],q[3],q[122],q[67],q[0],q[12],q[64],q[76],q[9],q[73],q[18],q[102],q[27],q[91],q[36],q[33],q[100],q[97],q[42],q[106],q[51],q[115],q[60],q[5],q[124],q[69],q[2],q[66],q[11],q[75],q[20],q[84],q[29],q[26],q[93],q[90],q[35],q[82],q[99],q[44],q[108],q[53],q[117],q[72],q[126],q[59],q[4],q[123],q[68],q[13],q[77],q[22],q[19],q[86],q[31],q[83],q[28],q[95],q[81],q[37],q[101],q[46],q[110];
measure q[103] -> meas[0];
measure q[102] -> meas[1];
measure q[92] -> meas[2];
measure q[83] -> meas[3];
measure q[81] -> meas[4];
measure q[82] -> meas[5];
measure q[63] -> meas[6];
measure q[72] -> meas[7];
measure q[62] -> meas[8];
