OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
creg meas[15];
rz(-pi) q[60];
sx q[60];
rz(2.5838711) q[60];
sx q[60];
sx q[61];
rz(2.4647933) q[61];
sx q[61];
rz(-pi) q[61];
rz(-pi) q[62];
sx q[62];
rz(2.772837) q[62];
sx q[62];
rz(-pi) q[63];
sx q[63];
rz(2.9336188) q[63];
sx q[63];
cx q[63],q[62];
sx q[64];
rz(1.8374758) q[64];
sx q[64];
rz(-pi) q[64];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[62],q[61];
cx q[63],q[64];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[62],q[61];
sx q[65];
rz(0.29566112) q[65];
sx q[65];
rz(-pi) q[65];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[61],q[60];
cx q[65],q[64];
cx q[64],q[65];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[60],q[61];
cx q[61],q[60];
sx q[66];
rz(1.8705657) q[66];
sx q[66];
rz(-pi) q[66];
rz(-pi) q[72];
sx q[72];
rz(0.069242875) q[72];
sx q[72];
cx q[62],q[72];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
sx q[73];
rz(0.027961041) q[73];
sx q[73];
rz(-pi) q[73];
sx q[81];
rz(0.96355437) q[81];
sx q[81];
rz(-pi) q[81];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[65],q[64];
cx q[66],q[65];
cx q[65],q[66];
cx q[72],q[81];
cx q[73],q[66];
cx q[66],q[73];
cx q[81],q[72];
cx q[72],q[81];
cx q[62],q[72];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[62],q[61];
cx q[62],q[72];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[62],q[72];
cx q[65],q[64];
cx q[64],q[65];
cx q[64],q[63];
cx q[63],q[64];
cx q[66],q[65];
cx q[65],q[66];
cx q[65],q[64];
cx q[64],q[65];
cx q[72],q[62];
cx q[62],q[72];
cx q[61],q[62];
cx q[81],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[81],q[72];
cx q[62],q[72];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[61],q[62];
cx q[61],q[60];
cx q[62],q[72];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[72],q[62];
cx q[62],q[72];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[81],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[63],q[62];
cx q[61],q[62];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[81],q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[81],q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
cx q[72],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
rz(-pi) q[85];
sx q[85];
rz(1.4566548) q[85];
sx q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
sx q[86];
rz(1.0288256) q[86];
sx q[86];
rz(-pi) q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[85],q[73];
cx q[73],q[85];
cx q[73],q[66];
cx q[66],q[73];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[63],q[62];
cx q[62],q[72];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[64],q[63];
cx q[72],q[62];
cx q[63],q[62];
cx q[62],q[63];
cx q[81],q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[62],q[61];
cx q[61],q[62];
cx q[61],q[60];
cx q[60],q[61];
cx q[81],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
rz(-pi) q[87];
sx q[87];
rz(0.80080864) q[87];
sx q[87];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
rz(-pi) q[93];
sx q[93];
rz(1.2080602) q[93];
sx q[93];
cx q[87],q[93];
cx q[87],q[86];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
rz(-pi) q[106];
sx q[106];
rz(0.59477871) q[106];
sx q[106];
cx q[93],q[106];
cx q[106],q[93];
cx q[93],q[106];
cx q[87],q[93];
cx q[106],q[93];
rz(-pi) q[87];
sx q[87];
rz(0.91561184) q[87];
sx q[87];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[93],q[106];
cx q[106],q[93];
cx q[87],q[93];
cx q[87],q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[86],q[85];
cx q[85],q[86];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[66],q[73];
cx q[72],q[62];
cx q[72],q[81];
cx q[73],q[66];
cx q[66],q[73];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[65];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[81],q[72];
cx q[72],q[81];
cx q[72],q[62];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[62];
cx q[72],q[81];
cx q[81],q[72];
cx q[72],q[81];
cx q[72],q[62];
cx q[63],q[62];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[61],q[60];
cx q[60],q[61];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[93],q[106];
rz(-pi) q[93];
sx q[93];
rz(0.59540381) q[93];
sx q[93];
cx q[87],q[93];
cx q[87],q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[93],q[87];
cx q[87],q[93];
cx q[93],q[87];
cx q[93],q[106];
sx q[93];
rz(2.0734798) q[93];
sx q[93];
rz(-pi) q[93];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[86],q[87];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[93],q[87];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[93],q[106];
cx q[106],q[93];
cx q[93],q[106];
cx q[87],q[93];
sx q[87];
rz(2.3976384) q[87];
sx q[87];
rz(-pi) q[87];
cx q[86],q[87];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[106],q[93];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[65],q[66];
rz(-pi) q[65];
sx q[65];
rz(0.047391981) q[65];
sx q[65];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[66];
rz(-pi) q[73];
sx q[73];
rz(2.6843261) q[73];
sx q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[73],q[66];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[72],q[62];
rz(-pi) q[72];
sx q[72];
rz(1.7810571) q[72];
sx q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
rz(-pi) q[73];
sx q[73];
rz(1.7176667) q[73];
sx q[73];
cx q[81],q[72];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[63],q[62];
sx q[63];
rz(2.0604767) q[63];
sx q[63];
rz(-pi) q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[63],q[62];
cx q[61],q[62];
sx q[61];
rz(0.35248862) q[61];
sx q[61];
rz(-pi) q[61];
sx q[63];
rz(1.2725369) q[63];
sx q[63];
rz(-pi) q[63];
cx q[72],q[62];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[60],q[61];
sx q[60];
rz(0.58772752) q[60];
sx q[60];
rz(-pi) q[60];
sx q[61];
rz(1.3727141) q[61];
sx q[61];
rz(-pi) q[61];
sx q[72];
rz(2.1340034) q[72];
sx q[72];
rz(-pi) q[72];
sx q[81];
rz(2.218528) q[81];
sx q[81];
rz(-pi) q[81];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[73],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[72];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[81];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[62],q[63];
cx q[62],q[72];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[61],q[62];
sx q[61];
rz(1.2675111) q[61];
sx q[61];
rz(-pi) q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[87],q[93];
cx q[93],q[106];
cx q[106],q[93];
cx q[93],q[106];
cx q[93],q[87];
cx q[87],q[93];
cx q[93],q[87];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[87],q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[85],q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[85],q[86];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[87],q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[93],q[87];
cx q[87],q[93];
cx q[93],q[87];
cx q[86],q[87];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[86],q[87];
cx q[85],q[86];
cx q[86],q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[93],q[106];
cx q[106],q[93];
cx q[93],q[106];
cx q[93],q[87];
cx q[87],q[93];
cx q[93],q[87];
cx q[106],q[93];
cx q[87],q[86];
cx q[93],q[106];
cx q[106],q[93];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[93],q[87];
cx q[106],q[93];
cx q[86],q[87];
cx q[93],q[106];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[86],q[87];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[73],q[85];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[65],q[66];
cx q[66],q[65];
cx q[64],q[65];
cx q[65],q[64];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[72];
cx q[62],q[61];
cx q[62],q[63];
rz(-pi) q[62];
sx q[62];
rz(2.9820713) q[62];
sx q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[62],q[72];
cx q[73],q[85];
cx q[66],q[73];
cx q[73],q[66];
cx q[65],q[66];
cx q[66],q[65];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[62],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[62],q[63];
rz(-pi) q[62];
sx q[62];
rz(0.51273759) q[62];
sx q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[84],q[85];
cx q[93],q[87];
cx q[93],q[106];
cx q[106],q[93];
cx q[93],q[106];
cx q[93],q[87];
cx q[87],q[93];
cx q[93],q[87];
cx q[86],q[87];
cx q[87],q[86];
cx q[85],q[86];
cx q[86],q[85];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[65],q[66];
cx q[66],q[65];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[63],q[64];
rz(-pi) q[63];
sx q[63];
rz(2.6729016) q[63];
sx q[63];
cx q[72],q[62];
cx q[62],q[72];
cx q[72],q[62];
cx q[81],q[72];
cx q[72],q[81];
cx q[81],q[72];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[83],q[82];
cx q[82],q[83];
cx q[83],q[82];
cx q[84],q[85];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[84],q[85];
cx q[73],q[85];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[93],q[106];
cx q[106],q[93];
cx q[87],q[93];
cx q[93],q[87];
cx q[106],q[93];
cx q[86],q[87];
cx q[87],q[86];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[84],q[85];
cx q[73],q[85];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[73],q[85];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[84],q[85];
cx q[84],q[83];
cx q[85],q[73];
cx q[73],q[85];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[65],q[66];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[83];
cx q[85],q[73];
rz(-pi) q[85];
sx q[85];
rz(1.4097191) q[85];
sx q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[84],q[85];
sx q[84];
rz(1.7972735) q[84];
sx q[84];
rz(-pi) q[84];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[93],q[106];
cx q[106],q[93];
cx q[87],q[93];
cx q[93],q[87];
cx q[106],q[93];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[73];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[65],q[66];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[65],q[66];
cx q[85],q[86];
rz(-pi) q[85];
sx q[85];
rz(3.024521) q[85];
sx q[85];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[73],q[66];
cx q[66],q[73];
cx q[73],q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[84],q[85];
cx q[73],q[85];
rz(-pi) q[84];
sx q[84];
rz(1.4388053) q[84];
sx q[84];
cx q[85],q[73];
cx q[73],q[85];
cx q[66],q[73];
rz(-pi) q[66];
sx q[66];
rz(2.0734606) q[66];
sx q[66];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[85],q[73];
cx q[73],q[85];
cx q[85],q[73];
cx q[93],q[106];
cx q[106],q[93];
cx q[87],q[93];
cx q[93],q[87];
cx q[106],q[93];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[85];
cx q[73],q[85];
rz(-pi) q[73];
sx q[73];
rz(0.25494905) q[73];
sx q[73];
sx q[86];
rz(2.8772994) q[86];
sx q[86];
rz(-pi) q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[87],q[93];
cx q[93],q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[93],q[106];
cx q[106],q[93];
cx q[93],q[87];
cx q[86],q[87];
sx q[86];
rz(2.0198669) q[86];
sx q[86];
rz(-pi) q[86];
sx q[93];
rz(1.6900985) q[93];
sx q[93];
rz(-pi) q[93];
cx q[93],q[87];
cx q[87],q[93];
cx q[93],q[87];
cx q[106],q[93];
sx q[106];
rz(1.5505524) q[106];
sx q[106];
rz(-pi) q[106];
sx q[93];
rz(0.1290059) q[93];
sx q[93];
rz(-pi) q[93];
barrier q[123],q[68],q[13],q[77],q[22],q[19],q[73],q[81],q[28],q[92],q[37],q[101],q[46],q[110],q[55],q[119],q[52],q[116],q[63],q[6],q[125],q[70],q[15],q[12],q[79],q[76],q[21],q[88],q[87],q[30],q[94],q[39],q[103],q[48],q[112],q[45],q[109],q[54],q[118],q[60],q[8],q[5],q[65],q[17],q[69],q[14],q[66],q[78],q[23],q[106],q[32],q[96],q[41],q[105],q[38],q[102],q[47],q[111],q[56],q[1],q[120],q[64],q[10],q[126],q[7],q[74],q[71],q[16],q[80],q[25],q[89],q[34],q[98],q[31],q[43],q[95],q[107],q[40],q[104],q[49],q[113],q[58],q[3],q[122],q[117],q[0],q[67],q[61],q[9],q[85],q[18],q[72],q[27],q[91],q[24],q[36],q[100],q[33],q[97],q[42],q[93],q[51],q[115],q[83],q[57],q[124],q[2],q[121],q[84],q[11],q[75],q[20],q[82],q[29],q[86],q[26],q[90],q[35],q[99],q[44],q[108],q[53],q[50],q[62],q[114],q[59],q[4];
measure q[60] -> meas[0];
measure q[62] -> meas[1];
measure q[61] -> meas[2];
measure q[63] -> meas[3];
measure q[64] -> meas[4];
measure q[83] -> meas[5];
measure q[65] -> meas[6];
measure q[84] -> meas[7];
measure q[66] -> meas[8];
measure q[85] -> meas[9];
measure q[73] -> meas[10];
measure q[87] -> meas[11];
measure q[86] -> meas[12];
measure q[106] -> meas[13];
measure q[93] -> meas[14];
