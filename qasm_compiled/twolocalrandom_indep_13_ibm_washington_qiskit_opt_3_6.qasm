OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[13];
rz(-pi) q[44];
sx q[44];
rz(2.6600127) q[44];
sx q[44];
rz(-pi) q[45];
sx q[45];
rz(2.6430642) q[45];
sx q[45];
cx q[45],q[44];
rz(-pi) q[53];
sx q[53];
rz(2.4107058) q[53];
sx q[53];
rz(-pi) q[54];
sx q[54];
rz(2.8515143) q[54];
sx q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[45],q[44];
rz(-pi) q[44];
sx q[44];
rz(pi/2) q[44];
rz(-pi) q[45];
sx q[45];
rz(pi/2) q[45];
sx q[54];
rz(pi/2) q[54];
sx q[60];
rz(0.41704554) q[60];
sx q[60];
sx q[61];
rz(0.77003865) q[61];
sx q[61];
rz(-pi) q[61];
sx q[62];
rz(0.94203542) q[62];
sx q[62];
rz(-pi) q[63];
sx q[63];
rz(2.5435971) q[63];
sx q[63];
sx q[64];
rz(-2.8623538) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
sx q[54];
rz(-pi/2) q[54];
sx q[54];
rz(pi/2) q[64];
cx q[54],q[64];
rz(-pi) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[45],q[54];
sx q[45];
rz(-pi/2) q[45];
sx q[45];
rz(pi/2) q[54];
cx q[45],q[54];
rz(-pi) q[45];
sx q[45];
rz(-pi/2) q[45];
cx q[44],q[45];
sx q[44];
rz(-pi/2) q[44];
sx q[44];
rz(pi/2) q[45];
cx q[44],q[45];
rz(-pi) q[44];
sx q[44];
rz(-pi) q[44];
rz(-pi) q[45];
sx q[45];
rz(-pi/2) q[45];
rz(-pi) q[54];
sx q[54];
rz(-pi/2) q[54];
rz(-pi) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[63],q[64];
cx q[64],q[63];
rz(2.1498248) q[63];
sx q[63];
cx q[62],q[63];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[63];
cx q[62],q[63];
rz(pi/2) q[62];
sx q[62];
rz(2.4528718) q[62];
sx q[62];
cx q[61],q[62];
sx q[61];
rz(-pi/2) q[61];
sx q[61];
rz(pi/2) q[62];
cx q[61],q[62];
rz(pi/2) q[61];
sx q[61];
rz(-1.3564334) q[61];
sx q[61];
cx q[60],q[61];
sx q[60];
rz(-pi/2) q[60];
sx q[60];
rz(pi/2) q[61];
cx q[60],q[61];
rz(-pi/2) q[60];
sx q[60];
rz(-3.0529086) q[60];
cx q[60],q[53];
rz(-pi/2) q[61];
sx q[61];
rz(-pi) q[61];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[63];
sx q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[54],q[45];
cx q[45],q[54];
cx q[45],q[44];
rz(-pi) q[44];
sx q[44];
rz(pi/2) q[44];
rz(-pi) q[45];
sx q[45];
rz(pi/2) q[45];
rz(-pi) q[54];
sx q[54];
rz(pi/2) q[54];
rz(pi/2) q[64];
sx q[64];
rz(-pi) q[64];
cx q[63],q[64];
sx q[63];
rz(-pi/2) q[63];
sx q[63];
rz(pi/2) q[64];
cx q[63],q[64];
rz(-pi) q[63];
sx q[63];
rz(-pi/2) q[63];
cx q[62],q[63];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[63];
cx q[62],q[63];
rz(pi/2) q[62];
sx q[62];
cx q[62],q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[61],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
x q[60];
rz(-pi) q[61];
sx q[61];
rz(pi/2) q[61];
rz(-pi) q[62];
sx q[62];
rz(-pi/2) q[62];
rz(pi/2) q[63];
sx q[63];
x q[63];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
sx q[54];
rz(-pi/2) q[54];
sx q[54];
rz(pi/2) q[64];
cx q[54],q[64];
rz(-pi) q[54];
sx q[54];
rz(-pi/2) q[54];
cx q[45],q[54];
sx q[45];
rz(-pi/2) q[45];
sx q[45];
rz(pi/2) q[54];
cx q[45],q[54];
rz(-pi) q[45];
sx q[45];
rz(-pi/2) q[45];
cx q[44],q[45];
sx q[44];
rz(-pi/2) q[44];
sx q[44];
rz(pi/2) q[45];
cx q[44],q[45];
rz(-pi) q[44];
sx q[44];
rz(-pi) q[44];
rz(-pi/2) q[45];
sx q[45];
rz(-pi) q[45];
rz(-pi/2) q[54];
sx q[54];
rz(-pi) q[54];
x q[64];
cx q[63],q[64];
sx q[63];
rz(-pi/2) q[63];
sx q[63];
rz(pi/2) q[64];
cx q[63],q[64];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
sx q[54];
rz(-pi/2) q[54];
sx q[54];
rz(pi/2) q[64];
cx q[54],q[64];
rz(-pi) q[54];
sx q[54];
rz(-pi/2) q[54];
cx q[45],q[54];
sx q[45];
rz(-pi/2) q[45];
sx q[45];
rz(pi/2) q[54];
cx q[45],q[54];
rz(-pi) q[45];
sx q[45];
rz(-pi) q[45];
cx q[44],q[45];
sx q[44];
rz(-pi/2) q[44];
rz(-pi/2) q[54];
sx q[54];
rz(-pi) q[54];
x q[64];
sx q[72];
rz(-2.5811469) q[72];
sx q[72];
rz(-pi/2) q[72];
cx q[62],q[72];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[72];
cx q[62],q[72];
sx q[62];
rz(pi/2) q[62];
cx q[61],q[62];
sx q[61];
rz(-pi/2) q[61];
sx q[61];
rz(pi/2) q[62];
cx q[61],q[62];
rz(-pi) q[61];
sx q[61];
rz(-pi) q[61];
rz(-pi/2) q[62];
sx q[62];
rz(-pi) q[62];
rz(-pi/2) q[72];
sx q[72];
rz(-pi) q[72];
rz(-pi) q[80];
sx q[80];
rz(3.0982425) q[80];
sx q[80];
sx q[81];
rz(-2.2089036) q[81];
sx q[81];
rz(-pi/2) q[81];
cx q[72],q[81];
sx q[72];
rz(-pi/2) q[72];
sx q[72];
rz(pi/2) q[81];
cx q[72],q[81];
sx q[72];
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
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
rz(pi/2) q[61];
sx q[61];
rz(-pi) q[61];
cx q[60],q[61];
sx q[60];
rz(-pi/2) q[60];
sx q[60];
rz(pi/2) q[61];
cx q[60],q[61];
rz(pi/2) q[60];
sx q[60];
cx q[60],q[53];
rz(-pi) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[61];
sx q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[61];
cx q[60],q[61];
sx q[60];
rz(-pi/2) q[60];
sx q[60];
rz(pi/2) q[61];
cx q[60],q[61];
rz(-pi) q[60];
sx q[60];
rz(-pi) q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
x q[60];
rz(-pi) q[61];
sx q[61];
rz(-pi/2) q[61];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
x q[63];
cx q[63],q[64];
sx q[63];
rz(-pi/2) q[63];
sx q[63];
rz(pi/2) q[64];
cx q[63],q[64];
rz(pi/2) q[63];
sx q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[61];
sx q[61];
rz(-pi) q[61];
cx q[60],q[61];
sx q[60];
rz(-pi/2) q[60];
sx q[60];
rz(pi/2) q[61];
cx q[60],q[61];
rz(pi/2) q[60];
sx q[60];
cx q[60],q[53];
rz(pi/2) q[61];
sx q[61];
rz(-pi) q[62];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
sx q[54];
rz(-pi/2) q[54];
sx q[54];
rz(pi/2) q[64];
cx q[54],q[64];
rz(-pi) q[54];
sx q[54];
rz(-pi) q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(-pi/2) q[45];
cx q[44],q[45];
sx q[44];
rz(-pi/2) q[44];
sx q[44];
rz(pi/2) q[45];
cx q[44],q[45];
x q[44];
rz(pi/2) q[44];
sx q[45];
rz(3.1279016) q[45];
sx q[45];
rz(-pi/2) q[45];
x q[64];
rz(-pi/2) q[72];
sx q[72];
rz(-pi) q[72];
sx q[81];
rz(-pi/2) q[81];
rz(-pi) q[82];
sx q[82];
rz(2.9295778) q[82];
sx q[82];
cx q[81],q[82];
cx q[81],q[80];
sx q[81];
rz(2.3299676) q[81];
sx q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
rz(pi/2) q[81];
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
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[61],q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
x q[63];
cx q[63],q[64];
sx q[63];
rz(-pi/2) q[63];
sx q[63];
rz(pi/2) q[64];
cx q[63],q[64];
rz(pi/2) q[63];
sx q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[64];
sx q[64];
cx q[64],q[54];
cx q[54],q[64];
rz(-pi/2) q[54];
cx q[45],q[54];
sx q[45];
rz(-pi/2) q[45];
sx q[45];
rz(pi/2) q[54];
cx q[45],q[54];
sx q[45];
rz(-pi/2) q[45];
cx q[44],q[45];
sx q[44];
rz(-pi/2) q[44];
sx q[44];
rz(pi/2) q[45];
cx q[44],q[45];
x q[44];
rz(-pi/2) q[44];
rz(-pi/2) q[45];
sx q[45];
rz(-pi) q[45];
sx q[54];
rz(0.013691025) q[54];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
rz(-pi) q[54];
sx q[54];
rz(pi/2) q[54];
rz(pi/2) q[64];
sx q[64];
rz(-pi) q[64];
rz(-pi) q[72];
sx q[72];
rz(-pi/2) q[72];
rz(-pi) q[81];
sx q[81];
rz(-3*pi/2) q[81];
cx q[81],q[80];
sx q[81];
rz(2.7808029) q[81];
sx q[81];
cx q[82],q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[80];
rz(-pi) q[81];
sx q[81];
rz(3.0466549) q[81];
sx q[81];
cx q[82],q[81];
cx q[72],q[81];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
rz(1.5670228) q[72];
sx q[72];
rz(-pi/2) q[72];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
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
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[53];
rz(1.5681934) q[60];
sx q[60];
rz(-pi/2) q[60];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[61];
cx q[60],q[61];
sx q[60];
rz(-pi/2) q[60];
sx q[60];
rz(pi/2) q[61];
cx q[60],q[61];
sx q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
x q[60];
rz(1.9733453) q[61];
sx q[61];
rz(-1.5718161) q[61];
sx q[61];
rz(-1.5684015) q[61];
x q[63];
cx q[63],q[64];
sx q[63];
rz(-pi/2) q[63];
sx q[63];
rz(pi/2) q[64];
cx q[63],q[64];
rz(pi/2) q[63];
sx q[63];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
rz(pi/2) q[61];
sx q[61];
rz(-pi) q[61];
cx q[60],q[61];
sx q[60];
rz(-pi/2) q[60];
sx q[60];
rz(pi/2) q[61];
cx q[60],q[61];
rz(pi/2) q[60];
sx q[60];
cx q[60],q[53];
rz(-pi) q[60];
sx q[60];
rz(2.91783) q[60];
sx q[60];
rz(pi/2) q[61];
sx q[61];
x q[62];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
sx q[54];
rz(-pi/2) q[54];
sx q[54];
rz(pi/2) q[64];
cx q[54],q[64];
rz(-pi) q[54];
sx q[54];
rz(-pi/2) q[54];
cx q[45],q[54];
sx q[45];
rz(-pi/2) q[45];
sx q[45];
rz(pi/2) q[54];
cx q[45],q[54];
rz(-pi) q[45];
sx q[45];
rz(-pi) q[45];
rz(-pi) q[54];
sx q[54];
rz(-pi/2) q[54];
rz(-pi) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[63],q[64];
cx q[64],q[63];
rz(pi/2) q[63];
sx q[63];
rz(-pi) q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
rz(pi/2) q[45];
cx q[44],q[45];
sx q[44];
rz(-pi/2) q[44];
sx q[44];
rz(pi/2) q[45];
cx q[44],q[45];
rz(-pi) q[44];
sx q[44];
rz(-pi) q[44];
rz(-pi/2) q[45];
sx q[45];
rz(-pi) q[45];
rz(pi/2) q[54];
cx q[45],q[54];
sx q[45];
rz(-pi/2) q[45];
sx q[45];
rz(pi/2) q[54];
cx q[45],q[54];
rz(-pi) q[45];
sx q[45];
rz(-pi) q[45];
cx q[44],q[45];
rz(-pi) q[45];
sx q[45];
rz(pi/2) q[45];
rz(-pi/2) q[54];
sx q[54];
rz(-pi) q[54];
rz(pi/2) q[64];
sx q[64];
rz(-pi) q[64];
rz(2.0951147) q[81];
sx q[81];
rz(-1.5726855) q[81];
sx q[81];
rz(-1.5675297) q[81];
cx q[72],q[81];
rz(pi/2) q[72];
sx q[72];
rz(-pi) q[72];
cx q[62],q[72];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[72];
cx q[62],q[72];
rz(pi/2) q[62];
sx q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
x q[62];
cx q[62],q[63];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[63];
cx q[62],q[63];
rz(pi/2) q[62];
sx q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[60],q[53];
sx q[60];
rz(0.98774213) q[60];
sx q[60];
cx q[61],q[62];
rz(pi/2) q[61];
sx q[61];
rz(-pi) q[61];
cx q[60],q[61];
sx q[60];
rz(-pi/2) q[60];
sx q[60];
rz(pi/2) q[61];
cx q[60],q[61];
rz(pi/2) q[60];
sx q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[63];
sx q[63];
x q[63];
cx q[63],q[64];
sx q[63];
rz(-pi/2) q[63];
sx q[63];
rz(pi/2) q[64];
cx q[63],q[64];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
sx q[54];
rz(-pi/2) q[54];
sx q[54];
rz(pi/2) q[64];
cx q[54],q[64];
rz(-pi) q[54];
sx q[54];
rz(-pi/2) q[54];
cx q[45],q[54];
sx q[45];
rz(-pi/2) q[45];
sx q[45];
rz(pi/2) q[54];
cx q[45],q[54];
rz(-pi) q[45];
sx q[45];
rz(-pi) q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(-1.1196204) q[44];
sx q[44];
rz(pi/2) q[44];
rz(0.73451958) q[45];
sx q[45];
rz(-pi/2) q[45];
rz(-pi) q[54];
sx q[54];
rz(-pi/2) q[54];
rz(-pi) q[64];
sx q[64];
rz(-pi/2) q[64];
rz(pi/2) q[72];
sx q[72];
cx q[80],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[80],q[81];
cx q[82],q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
x q[72];
cx q[82],q[81];
cx q[81],q[80];
cx q[80],q[81];
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
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[61],q[60];
rz(-pi) q[61];
sx q[61];
rz(2.6315861) q[61];
sx q[61];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[53],q[60];
x q[60];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
rz(pi/2) q[61];
sx q[61];
rz(-pi) q[61];
cx q[60],q[61];
sx q[60];
rz(-pi/2) q[60];
sx q[60];
rz(pi/2) q[61];
cx q[60],q[61];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[61];
sx q[61];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[63],q[62];
rz(-pi) q[63];
sx q[63];
rz(2.1885559) q[63];
sx q[63];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
x q[60];
rz(pi/2) q[61];
sx q[61];
rz(-pi) q[61];
cx q[60],q[61];
sx q[60];
rz(-pi/2) q[60];
sx q[60];
rz(pi/2) q[61];
cx q[60],q[61];
rz(pi/2) q[60];
sx q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
rz(-pi) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[61];
sx q[61];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(-pi) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[64],q[54];
cx q[54],q[64];
rz(pi/2) q[54];
cx q[45],q[54];
sx q[45];
rz(-pi/2) q[45];
sx q[45];
rz(pi/2) q[54];
cx q[45],q[54];
sx q[45];
rz(pi/2) q[45];
cx q[44],q[45];
sx q[44];
rz(-pi/2) q[44];
sx q[44];
rz(pi/2) q[45];
cx q[44],q[45];
rz(-3.0403078) q[44];
sx q[44];
rz(-pi) q[44];
rz(2.033531) q[45];
sx q[45];
rz(-1.7837976) q[45];
sx q[45];
rz(1.9716122) q[45];
rz(1.8128497) q[54];
sx q[54];
rz(-1.830233) q[54];
sx q[54];
rz(-0.76624807) q[54];
sx q[64];
rz(-2.6121033) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[63],q[64];
sx q[63];
rz(-pi/2) q[63];
sx q[63];
rz(pi/2) q[64];
cx q[63],q[64];
rz(-pi) q[63];
sx q[63];
rz(-pi) q[63];
rz(-pi) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[54],q[64];
cx q[64],q[54];
rz(pi/2) q[54];
sx q[54];
rz(-pi) q[54];
cx q[45],q[54];
sx q[45];
rz(-pi/2) q[45];
sx q[45];
rz(pi/2) q[54];
cx q[45],q[54];
rz(pi/2) q[45];
sx q[45];
cx q[45],q[44];
rz(-pi) q[45];
sx q[45];
rz(2.958066) q[45];
sx q[45];
rz(pi/2) q[54];
sx q[54];
x q[54];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(pi/2) q[64];
rz(pi/2) q[81];
sx q[81];
cx q[82],q[81];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[80],q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[82],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[62],q[72];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[61];
cx q[60],q[61];
sx q[60];
rz(-pi/2) q[60];
sx q[60];
rz(pi/2) q[61];
cx q[60],q[61];
rz(-pi) q[60];
sx q[60];
rz(-pi) q[60];
cx q[53],q[60];
x q[60];
rz(-pi) q[61];
sx q[61];
rz(-pi/2) q[61];
cx q[72],q[81];
cx q[81],q[72];
cx q[80],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[72],q[62];
cx q[63],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[61];
sx q[61];
rz(-pi) q[61];
cx q[60],q[61];
sx q[60];
rz(-pi/2) q[60];
sx q[60];
rz(pi/2) q[61];
cx q[60],q[61];
rz(pi/2) q[60];
sx q[60];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(-pi) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[61];
sx q[61];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[62],q[61];
rz(-pi) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[63],q[64];
sx q[63];
rz(-pi/2) q[63];
sx q[63];
rz(pi/2) q[64];
cx q[63],q[64];
rz(-pi) q[63];
sx q[63];
rz(-pi) q[63];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
rz(pi/2) q[61];
cx q[60],q[61];
sx q[60];
rz(-pi/2) q[60];
sx q[60];
rz(pi/2) q[61];
cx q[60],q[61];
rz(-pi) q[60];
sx q[60];
rz(-pi) q[60];
cx q[53],q[60];
rz(-pi) q[61];
sx q[61];
rz(-pi/2) q[61];
x q[62];
x q[64];
cx q[54],q[64];
sx q[54];
rz(-pi/2) q[54];
sx q[54];
rz(pi/2) q[64];
cx q[54],q[64];
rz(pi/2) q[54];
sx q[54];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[45],q[44];
rz(-pi) q[45];
sx q[45];
rz(2.4538617) q[45];
sx q[45];
cx q[54],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[72],q[81];
cx q[81],q[72];
rz(pi/2) q[72];
sx q[72];
rz(-pi) q[72];
cx q[62],q[72];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[72];
cx q[62],q[72];
rz(pi/2) q[62];
sx q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[62],q[61];
rz(-pi) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[63],q[64];
sx q[63];
rz(-pi/2) q[63];
sx q[63];
rz(pi/2) q[64];
cx q[63],q[64];
rz(-pi) q[63];
sx q[63];
rz(-pi) q[63];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[41],q[53];
cx q[53],q[41];
cx q[41],q[53];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
rz(pi/2) q[43];
sx q[43];
rz(pi/2) q[43];
cx q[61],q[62];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(-pi) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[45];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(-pi/2) q[44];
cx q[43],q[44];
sx q[43];
rz(-pi/2) q[43];
sx q[43];
rz(pi/2) q[44];
cx q[43],q[44];
rz(-pi) q[43];
sx q[43];
rz(-2.9660356) q[44];
sx q[44];
rz(-pi) q[44];
rz(-pi) q[54];
sx q[54];
rz(2.8053673) q[54];
sx q[54];
cx q[64],q[54];
cx q[45],q[54];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[54],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[44];
cx q[44],q[43];
cx q[43],q[44];
rz(-pi) q[45];
sx q[45];
rz(pi/2) q[45];
rz(pi/2) q[72];
sx q[72];
cx q[80],q[81];
cx q[82],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[72],q[81];
cx q[81],q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[63],q[62];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[62];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[64],q[54];
rz(-pi) q[64];
sx q[64];
rz(2.6597232) q[64];
sx q[64];
cx q[63],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
rz(pi/2) q[54];
cx q[45],q[54];
sx q[45];
rz(-pi/2) q[45];
sx q[45];
rz(pi/2) q[54];
cx q[45],q[54];
rz(-pi) q[45];
sx q[45];
rz(-pi) q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[43],q[44];
rz(-pi/2) q[54];
sx q[54];
rz(-pi) q[54];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
rz(pi/2) q[64];
sx q[64];
rz(-pi) q[64];
cx q[82],q[81];
cx q[80],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[80],q[81];
cx q[72],q[81];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
x q[72];
cx q[81],q[82];
cx q[82],q[81];
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
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
rz(-pi/2) q[62];
cx q[61],q[62];
sx q[61];
rz(-pi/2) q[61];
sx q[61];
rz(pi/2) q[62];
cx q[61],q[62];
rz(-pi) q[61];
sx q[61];
cx q[60],q[61];
rz(-pi) q[60];
sx q[60];
rz(3.0925353) q[60];
sx q[60];
rz(-2.8985092) q[62];
sx q[62];
rz(-pi) q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
rz(-pi) q[62];
sx q[62];
rz(2.5775149) q[62];
sx q[62];
x q[63];
cx q[63],q[64];
sx q[63];
rz(-pi/2) q[63];
sx q[63];
rz(pi/2) q[64];
cx q[63],q[64];
rz(pi/2) q[63];
sx q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
sx q[54];
rz(-pi/2) q[54];
sx q[54];
rz(pi/2) q[64];
cx q[54],q[64];
rz(-pi) q[54];
sx q[54];
rz(-pi) q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[45],q[44];
cx q[44],q[45];
cx q[43],q[44];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
rz(-pi) q[44];
sx q[44];
rz(pi/2) q[44];
cx q[53],q[41];
cx q[41],q[53];
cx q[53],q[41];
rz(-pi) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
rz(pi/2) q[81];
sx q[81];
cx q[80],q[81];
cx q[72],q[81];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[82],q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[80],q[81];
rz(-pi) q[80];
sx q[80];
rz(3.0172003) q[80];
sx q[80];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[62],q[72];
rz(-pi) q[62];
sx q[62];
rz(3.0941874) q[62];
sx q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[63];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[72],q[81];
cx q[81],q[72];
rz(-pi) q[72];
sx q[72];
rz(2.446595) q[72];
sx q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[72],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[64];
cx q[54],q[64];
rz(pi/2) q[63];
sx q[63];
rz(-pi) q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[82],q[81];
rz(-pi) q[81];
sx q[81];
rz(2.728573) q[81];
sx q[81];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
x q[62];
cx q[62],q[63];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[63];
cx q[62],q[63];
rz(pi/2) q[62];
sx q[62];
cx q[62],q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(-pi) q[62];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[63];
sx q[63];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[54],q[45];
rz(pi/2) q[64];
rz(-pi) q[82];
sx q[82];
rz(2.3219705) q[82];
sx q[82];
cx q[81],q[82];
cx q[81],q[72];
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
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[60],q[61];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[53],q[60];
cx q[60],q[53];
cx q[53],q[60];
rz(-pi) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[63],q[64];
sx q[63];
rz(-pi/2) q[63];
sx q[63];
rz(pi/2) q[64];
cx q[63],q[64];
rz(-pi) q[63];
sx q[63];
rz(-pi) q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
x q[62];
x q[63];
rz(-pi) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[45];
rz(pi/2) q[64];
sx q[64];
rz(-pi) q[64];
cx q[63],q[64];
sx q[63];
rz(-pi/2) q[63];
sx q[63];
rz(pi/2) q[64];
cx q[63],q[64];
rz(-pi) q[63];
sx q[63];
rz(-pi/2) q[63];
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
rz(pi/2) q[64];
sx q[64];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[54],q[45];
cx q[45],q[54];
rz(pi/2) q[45];
cx q[44],q[45];
sx q[44];
rz(-pi/2) q[44];
sx q[44];
rz(pi/2) q[45];
cx q[44],q[45];
rz(-pi) q[44];
sx q[44];
rz(-pi) q[44];
rz(-pi) q[45];
sx q[45];
rz(-pi/2) q[45];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[54],q[64];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[45],q[54];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[54];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
rz(-pi) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[63],q[64];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[61],q[60];
cx q[63],q[64];
rz(pi/2) q[63];
sx q[63];
rz(-pi) q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
rz(pi/2) q[54];
cx q[45],q[54];
sx q[45];
rz(-pi/2) q[45];
sx q[45];
rz(pi/2) q[54];
cx q[45],q[54];
rz(-pi) q[45];
sx q[45];
rz(-pi) q[45];
cx q[44],q[45];
rz(-pi) q[54];
sx q[54];
rz(-pi/2) q[54];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
cx q[54],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
rz(-pi) q[54];
sx q[54];
rz(pi/2) q[54];
rz(pi/2) q[64];
sx q[64];
rz(-pi) q[64];
rz(-pi) q[72];
sx q[72];
rz(-pi/2) q[72];
cx q[81],q[80];
x q[80];
rz(-pi) q[81];
sx q[81];
rz(2.7875716) q[81];
sx q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[53];
cx q[62],q[72];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[60],q[53];
x q[62];
cx q[62],q[63];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[63];
cx q[62],q[63];
rz(pi/2) q[62];
sx q[62];
cx q[62],q[72];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[60],q[53];
cx q[62],q[72];
rz(pi/2) q[63];
sx q[63];
x q[63];
cx q[63],q[64];
sx q[63];
rz(-pi/2) q[63];
sx q[63];
rz(pi/2) q[64];
cx q[63],q[64];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[54],q[64];
sx q[54];
rz(-pi/2) q[54];
sx q[54];
rz(pi/2) q[64];
cx q[54],q[64];
rz(-pi) q[54];
sx q[54];
rz(-pi) q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[44],q[45];
rz(-pi) q[45];
sx q[45];
rz(pi/2) q[45];
rz(-pi) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[64],q[54];
cx q[54],q[64];
cx q[64],q[54];
rz(-pi) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[72],q[62];
cx q[62],q[72];
sx q[72];
rz(-pi/2) q[72];
rz(-1.5676558) q[81];
sx q[81];
cx q[80],q[81];
sx q[80];
rz(-pi/2) q[80];
sx q[80];
rz(pi/2) q[81];
cx q[80],q[81];
rz(-0.26992298) q[80];
sx q[80];
rz(-1.5716338) q[80];
sx q[80];
rz(1.5677695) q[80];
rz(-pi/2) q[81];
sx q[81];
rz(-pi/2) q[81];
cx q[72],q[81];
sx q[72];
rz(-pi/2) q[72];
sx q[72];
rz(pi/2) q[81];
cx q[72],q[81];
sx q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[61],q[62];
rz(-pi) q[61];
sx q[61];
rz(2.7445891) q[61];
sx q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[53];
cx q[53],q[60];
cx q[60],q[53];
cx q[61],q[62];
rz(-pi) q[61];
sx q[61];
rz(3.1339204) q[61];
sx q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[72];
cx q[62],q[61];
cx q[62],q[63];
rz(-pi) q[62];
sx q[62];
rz(2.8487149) q[62];
sx q[62];
cx q[62],q[72];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(pi/2) q[63];
sx q[63];
rz(-pi) q[63];
cx q[72],q[62];
cx q[62],q[72];
x q[62];
cx q[62],q[63];
sx q[62];
rz(-pi/2) q[62];
sx q[62];
rz(pi/2) q[63];
cx q[62],q[63];
rz(pi/2) q[62];
sx q[62];
cx q[62],q[61];
rz(pi/2) q[63];
sx q[63];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[62],q[63];
rz(-pi) q[62];
sx q[62];
rz(3.0565285) q[62];
sx q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
rz(pi/2) q[64];
cx q[54],q[64];
sx q[54];
rz(-pi/2) q[54];
sx q[54];
rz(pi/2) q[64];
cx q[54],q[64];
rz(-pi) q[54];
sx q[54];
rz(-pi/2) q[54];
cx q[45],q[54];
sx q[45];
rz(-pi/2) q[45];
sx q[45];
rz(pi/2) q[54];
cx q[45],q[54];
rz(-pi) q[45];
sx q[45];
rz(-pi) q[45];
cx q[45],q[44];
cx q[44],q[45];
rz(-pi/2) q[54];
sx q[54];
rz(-pi) q[54];
rz(-pi) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[63],q[64];
rz(-pi) q[63];
sx q[63];
rz(2.7045853) q[63];
sx q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(pi/2) q[64];
cx q[54],q[64];
sx q[54];
rz(-pi/2) q[54];
sx q[54];
rz(pi/2) q[64];
cx q[54],q[64];
rz(-pi) q[54];
sx q[54];
rz(-pi) q[54];
cx q[54],q[45];
cx q[45],q[54];
cx q[44],q[45];
rz(-pi) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[54],q[64];
rz(-pi) q[54];
sx q[54];
rz(2.6223059) q[54];
sx q[54];
cx q[63],q[64];
cx q[54],q[64];
rz(-pi) q[63];
sx q[63];
rz(3.0464917) q[63];
sx q[63];
cx q[64],q[54];
cx q[54],q[64];
cx q[54],q[45];
cx q[45],q[54];
cx q[44],q[45];
rz(-pi) q[44];
sx q[44];
rz(2.2315466) q[44];
sx q[44];
rz(-pi) q[45];
sx q[45];
rz(2.8835129) q[45];
sx q[45];
rz(-pi) q[54];
sx q[54];
rz(2.2253467) q[54];
sx q[54];
rz(pi/2) q[81];
sx q[81];
rz(-2.0215093) q[81];
sx q[81];
barrier q[15],q[12],q[79],q[76],q[21],q[88],q[85],q[30],q[94],q[39],q[103],q[48],q[112],q[82],q[109],q[80],q[118],q[53],q[8],q[5],q[63],q[17],q[69],q[14],q[44],q[78],q[23],q[87],q[32],q[96],q[41],q[105],q[38],q[102],q[47],q[111],q[56],q[1],q[120],q[65],q[10],q[7],q[74],q[71],q[16],q[45],q[25],q[89],q[34],q[98],q[31],q[43],q[95],q[107],q[40],q[104],q[49],q[113],q[58],q[3],q[122],q[0],q[67],q[60],q[9],q[73],q[18],q[54],q[27],q[91],q[24],q[36],q[100],q[33],q[97],q[42],q[106],q[51],q[115],q[62],q[57],q[124],q[2],q[121],q[66],q[11],q[75],q[20],q[84],q[29],q[93],q[26],q[90],q[35],q[99],q[81],q[108],q[64],q[50],q[117],q[72],q[114],q[59],q[126],q[4],q[123],q[68],q[13],q[77],q[22],q[86],q[19],q[83],q[28],q[92],q[37],q[101],q[46],q[110],q[55],q[52],q[119],q[116],q[61],q[6],q[125],q[70];
measure q[82] -> meas[0];
measure q[80] -> meas[1];
measure q[81] -> meas[2];
measure q[53] -> meas[3];
measure q[60] -> meas[4];
measure q[72] -> meas[5];
measure q[61] -> meas[6];
measure q[62] -> meas[7];
measure q[64] -> meas[8];
measure q[63] -> meas[9];
measure q[44] -> meas[10];
measure q[54] -> meas[11];
measure q[45] -> meas[12];
