OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg meas[5];
x q[12];
rz(pi/2) q[15];
sx q[15];
rz(-pi/2) q[15];
cx q[12],q[15];
rz(-pi/4) q[15];
rz(-pi) q[18];
sx q[18];
rz(1.6817907) q[18];
sx q[18];
rz(-pi) q[21];
sx q[21];
rz(1.4039086) q[21];
sx q[21];
cx q[18],q[21];
rz(-pi) q[21];
sx q[21];
rz(1.8249778) q[21];
sx q[21];
cx q[18],q[21];
x q[18];
cx q[18],q[15];
rz(pi/4) q[15];
cx q[12],q[15];
rz(pi/4) q[12];
rz(-pi/4) q[15];
x q[15];
rz(pi/4) q[18];
sx q[18];
rz(-pi) q[18];
cx q[15],q[18];
sx q[15];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[18];
cx q[15],q[18];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[12];
rz(-pi/4) q[12];
cx q[15],q[12];
x q[15];
rz(pi/2) q[18];
sx q[18];
rz(-pi/4) q[18];
sx q[18];
rz(2.4133232) q[18];
sx q[21];
rz(-pi) q[21];
sx q[23];
rz(-pi/2) q[23];
cx q[23],q[21];
rz(-pi/2) q[21];
sx q[23];
rz(-pi) q[23];
cx q[23],q[21];
rz(pi/2) q[21];
sx q[23];
cx q[23],q[21];
rz(-pi) q[21];
sx q[21];
rz(-7*pi/8) q[21];
cx q[18],q[21];
x q[18];
rz(0.17976045) q[21];
cx q[18],q[21];
rz(-1.5136676) q[18];
rz(-0.17976045) q[21];
sx q[21];
rz(-pi/4) q[23];
cx q[23],q[21];
rz(-pi/4) q[21];
cx q[18],q[21];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[21];
cx q[23],q[21];
rz(pi/4) q[21];
cx q[18],q[21];
sx q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[21];
cx q[18],q[21];
sx q[18];
rz(3*pi/4) q[18];
rz(-pi) q[21];
sx q[21];
rz(-3.3771619) q[21];
cx q[21],q[23];
rz(-pi/4) q[23];
cx q[21],q[23];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[18];
rz(pi/2) q[18];
sx q[21];
rz(-pi) q[21];
cx q[21],q[18];
rz(1.4563044) q[18];
sx q[21];
cx q[21],q[18];
x q[18];
rz(-3.6914215) q[18];
rz(1.4563044) q[21];
sx q[21];
cx q[23],q[21];
rz(-pi/4) q[21];
cx q[18],q[21];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[21];
cx q[23],q[21];
rz(pi/4) q[21];
cx q[18],q[21];
sx q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[21];
cx q[18],q[21];
sx q[18];
rz(pi/4) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
sx q[21];
rz(-pi/4) q[21];
rz(pi/4) q[23];
cx q[21],q[23];
rz(-pi/4) q[23];
cx q[21],q[23];
cx q[21],q[18];
sx q[18];
rz(2.9126088) q[18];
sx q[18];
rz(-pi) q[18];
cx q[21],q[18];
sx q[18];
rz(1.7997801) q[18];
sx q[18];
cx q[15],q[18];
rz(-pi/4) q[18];
cx q[21],q[18];
rz(pi/4) q[18];
cx q[15],q[18];
rz(pi/4) q[15];
rz(-pi/4) q[18];
x q[18];
sx q[21];
rz(-pi) q[21];
cx q[18],q[21];
sx q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[21];
cx q[18],q[21];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[15];
rz(-pi/4) q[15];
rz(-pi/2) q[21];
sx q[21];
rz(-3*pi/4) q[21];
sx q[21];
rz(-pi/2) q[21];
cx q[18],q[21];
rz(-pi) q[21];
sx q[21];
rz(2.9126088) q[21];
sx q[21];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
sx q[21];
rz(1.3418125) q[21];
sx q[21];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(-pi/4) q[21];
cx q[18],q[21];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
rz(pi/4) q[21];
cx q[18],q[21];
rz(pi/4) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(-2.3039136) q[15];
sx q[15];
rz(-pi/4) q[21];
cx q[18],q[21];
rz(0.73311726) q[18];
sx q[18];
rz(-pi/2) q[18];
cx q[18],q[15];
rz(-pi/2) q[15];
sx q[18];
rz(-pi) q[18];
cx q[18],q[15];
rz(pi/4) q[15];
sx q[18];
cx q[18],q[15];
rz(-1.6230772) q[15];
sx q[15];
rz(-pi/2) q[15];
cx q[12],q[15];
rz(-pi/4) q[15];
rz(0.65792017) q[18];
cx q[18],q[15];
rz(pi/4) q[15];
cx q[12],q[15];
rz(pi/4) q[12];
x q[15];
rz(-3*pi/4) q[15];
sx q[18];
rz(-pi) q[18];
cx q[15],q[18];
sx q[15];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[18];
cx q[15],q[18];
rz(pi/2) q[15];
sx q[15];
rz(-2.1764356) q[15];
cx q[15],q[12];
rz(-pi/4) q[12];
cx q[15],q[12];
rz(pi/2) q[18];
sx q[18];
rz(-3*pi/4) q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
rz(-pi/4) q[15];
cx q[18],q[15];
rz(pi/4) q[15];
cx q[12],q[15];
rz(pi/4) q[12];
rz(-pi/4) q[15];
x q[15];
rz(pi/4) q[18];
sx q[18];
rz(-pi) q[18];
cx q[15],q[18];
sx q[15];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[18];
cx q[15],q[18];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[12];
rz(-pi/4) q[12];
cx q[15],q[12];
x q[15];
rz(-pi/2) q[18];
sx q[18];
rz(pi/4) q[18];
rz(-pi/4) q[21];
cx q[21],q[18];
rz(pi/2) q[18];
sx q[21];
rz(-pi) q[21];
cx q[21],q[18];
rz(1.3418125) q[18];
sx q[21];
cx q[21],q[18];
rz(-2.9126088) q[18];
sx q[18];
rz(-pi) q[18];
cx q[15],q[18];
rz(-pi/4) q[18];
cx q[21],q[18];
rz(pi/4) q[18];
cx q[15],q[18];
rz(pi/4) q[15];
rz(-pi/4) q[18];
x q[18];
rz(pi/4) q[21];
sx q[21];
rz(-pi) q[21];
cx q[18],q[21];
sx q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[21];
cx q[18],q[21];
rz(pi/2) q[18];
sx q[18];
rz(pi) q[18];
cx q[18],q[15];
rz(-pi/4) q[15];
cx q[18],q[15];
sx q[18];
rz(pi/2) q[21];
sx q[21];
rz(-pi/4) q[21];
cx q[21],q[18];
rz(pi/2) q[18];
sx q[21];
rz(-pi) q[21];
cx q[21],q[18];
rz(1.3418125) q[18];
sx q[21];
cx q[21],q[18];
rz(0.22898382) q[18];
sx q[18];
cx q[15],q[18];
rz(-pi/4) q[18];
rz(-3*pi/4) q[21];
cx q[21],q[18];
rz(pi/4) q[18];
cx q[15],q[18];
rz(pi/4) q[15];
rz(-pi/4) q[18];
x q[18];
sx q[21];
rz(-pi) q[21];
cx q[18],q[21];
sx q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[21];
cx q[18],q[21];
rz(pi/2) q[18];
sx q[18];
rz(pi/4) q[18];
cx q[18],q[15];
rz(-pi/4) q[15];
cx q[18],q[15];
x q[15];
rz(-pi/2) q[21];
sx q[21];
rz(-3*pi/4) q[21];
sx q[21];
rz(-pi/2) q[21];
cx q[18],q[21];
rz(-pi) q[21];
sx q[21];
rz(3.0271007) q[21];
sx q[21];
cx q[18],q[21];
sx q[21];
rz(1.4563044) q[21];
sx q[21];
cx q[23],q[21];
rz(-pi/4) q[21];
cx q[18],q[21];
rz(pi/4) q[21];
cx q[23],q[21];
rz(-pi/4) q[21];
cx q[18],q[21];
rz(3*pi/4) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/4) q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[18],q[21];
rz(-pi/4) q[21];
cx q[18],q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[18],q[21];
sx q[21];
rz(3.0271007) q[21];
sx q[21];
rz(-pi) q[21];
cx q[18],q[21];
sx q[21];
rz(1.6852882) q[21];
sx q[21];
cx q[23],q[21];
rz(-pi/4) q[21];
cx q[18],q[21];
rz(pi/4) q[21];
cx q[23],q[21];
rz(-pi/4) q[21];
cx q[18],q[21];
sx q[18];
rz(-pi) q[18];
rz(3*pi/4) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/4) q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[18];
rz(-pi/2) q[18];
sx q[21];
rz(-pi) q[21];
cx q[21],q[18];
rz(pi/4) q[18];
sx q[21];
cx q[21],q[18];
rz(3*pi/4) q[21];
cx q[21],q[23];
sx q[23];
rz(2.8473403) q[23];
sx q[23];
rz(-pi) q[23];
cx q[21],q[23];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
rz(-pi/4) q[15];
cx q[18],q[15];
rz(pi/4) q[15];
cx q[12],q[15];
rz(pi/4) q[12];
rz(-pi/4) q[15];
cx q[18],q[15];
rz(pi/4) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
x q[12];
cx q[18],q[15];
rz(-pi/4) q[15];
rz(pi/4) q[18];
cx q[18],q[15];
rz(2.2780452) q[15];
sx q[15];
rz(-pi) q[15];
cx q[12],q[15];
sx q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[15];
cx q[12],q[15];
rz(pi/2) q[12];
sx q[12];
rz(0.078149328) q[12];
rz(pi/2) q[15];
sx q[15];
rz(-pi/4) q[15];
x q[18];
rz(-pi) q[23];
sx q[23];
rz(2.8473403) q[23];
sx q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
sx q[21];
rz(-3*pi/8) q[21];
sx q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
x q[18];
cx q[21],q[23];
sx q[23];
rz(1.8249778) q[23];
sx q[23];
rz(-pi) q[23];
cx q[21],q[23];
rz(1.6996667) q[21];
sx q[21];
rz(-0.71441802) q[21];
sx q[21];
rz(0.16987933) q[21];
cx q[18],q[21];
sx q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[21];
cx q[18],q[21];
rz(pi/2) q[18];
sx q[18];
rz(0.078149328) q[18];
rz(pi/2) q[21];
sx q[21];
rz(-pi/4) q[21];
sx q[23];
rz(-1.7376842) q[23];
sx q[23];
cx q[23],q[21];
rz(pi/4) q[21];
cx q[18],q[21];
rz(-pi/4) q[21];
cx q[23],q[21];
rz(pi/4) q[21];
sx q[21];
rz(-3*pi/8) q[21];
sx q[21];
cx q[18],q[21];
cx q[21],q[18];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
sx q[21];
rz(-1.403517) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-pi/4) q[23];
sx q[23];
rz(-1.6137664) q[23];
sx q[23];
cx q[21],q[23];
sx q[21];
rz(-0.77883164) q[21];
sx q[21];
rz(1.3183038) q[23];
cx q[21],q[23];
rz(-pi/2) q[21];
sx q[21];
rz(-3.1104109) q[21];
sx q[21];
cx q[21],q[18];
rz(pi/4) q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[12],q[15];
rz(-pi/4) q[15];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(pi/2) q[15];
sx q[15];
rz(-pi/2) q[15];
cx q[21],q[18];
rz(-pi/4) q[18];
sx q[18];
rz(-0.037344913) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[15];
rz(pi/2) q[15];
sx q[18];
rz(-pi) q[18];
cx q[18],q[15];
rz(1.3910359) q[15];
sx q[18];
cx q[18],q[15];
rz(1.5334514) q[15];
rz(-1.3910359) q[18];
sx q[18];
rz(-pi) q[18];
cx q[21],q[18];
cx q[18],q[21];
cx q[21],q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[15],q[12];
rz(-pi/4) q[12];
rz(pi/4) q[15];
cx q[15],q[12];
x q[15];
rz(1.4051241) q[23];
sx q[23];
rz(-1.5505973) q[23];
sx q[23];
rz(2.4764077) q[23];
cx q[23],q[21];
rz(-pi/4) q[21];
cx q[18],q[21];
rz(-3*pi/4) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[21];
cx q[23],q[21];
rz(pi/4) q[21];
cx q[18],q[21];
sx q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[21];
cx q[18],q[21];
sx q[18];
rz(3*pi/4) q[18];
rz(-pi) q[21];
sx q[21];
rz(-3.3771619) q[21];
cx q[21],q[23];
rz(-pi/4) q[23];
cx q[21],q[23];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[18];
rz(pi/2) q[18];
sx q[21];
rz(-pi) q[21];
cx q[21],q[18];
rz(1.4563044) q[18];
sx q[21];
cx q[21],q[18];
x q[18];
rz(-3.6914215) q[18];
rz(1.4563044) q[21];
sx q[21];
cx q[23],q[21];
rz(-pi/4) q[21];
cx q[18],q[21];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[21];
cx q[23],q[21];
rz(pi/4) q[21];
cx q[18],q[21];
sx q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[21];
cx q[18],q[21];
sx q[18];
rz(pi/4) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
sx q[21];
rz(-pi/4) q[21];
rz(pi/4) q[23];
cx q[21],q[23];
rz(-pi/4) q[23];
cx q[21],q[23];
cx q[21],q[18];
sx q[18];
rz(2.9126088) q[18];
sx q[18];
rz(-pi) q[18];
cx q[21],q[18];
sx q[18];
rz(1.7997801) q[18];
sx q[18];
cx q[15],q[18];
rz(-pi/4) q[18];
cx q[21],q[18];
rz(pi/4) q[18];
cx q[15],q[18];
rz(pi/4) q[15];
rz(-pi/4) q[18];
x q[18];
sx q[21];
rz(-pi) q[21];
cx q[18],q[21];
sx q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[21];
cx q[18],q[21];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[15];
rz(-pi/4) q[15];
rz(-pi/2) q[21];
sx q[21];
rz(-3*pi/4) q[21];
sx q[21];
rz(-pi/2) q[21];
cx q[18],q[21];
rz(-pi) q[21];
sx q[21];
rz(2.9126088) q[21];
sx q[21];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
sx q[21];
rz(1.3418125) q[21];
sx q[21];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(-pi/4) q[21];
cx q[18],q[21];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[15];
rz(pi/4) q[21];
cx q[18],q[21];
rz(pi/4) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(-2.3039136) q[15];
sx q[15];
rz(-pi/4) q[21];
cx q[18],q[21];
rz(0.73311726) q[18];
sx q[18];
rz(-pi/2) q[18];
cx q[18],q[15];
rz(-pi/2) q[15];
sx q[18];
rz(-pi) q[18];
cx q[18],q[15];
rz(pi/4) q[15];
sx q[18];
cx q[18],q[15];
rz(-1.6230772) q[15];
sx q[15];
rz(-pi/2) q[15];
cx q[12],q[15];
rz(-pi/4) q[15];
rz(0.052280907) q[18];
cx q[18],q[15];
rz(pi/4) q[15];
cx q[12],q[15];
rz(pi/4) q[12];
rz(-pi/4) q[15];
cx q[18],q[15];
rz(-pi/4) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[18],q[15];
rz(-pi/4) q[15];
cx q[18],q[15];
x q[15];
x q[18];
rz(3*pi/4) q[21];
sx q[21];
rz(pi/2) q[21];
barrier q[17],q[14],q[20],q[21],q[3],q[26],q[0],q[6],q[9],q[12],q[15],q[18],q[24],q[23],q[1],q[4],q[10],q[7],q[13],q[16],q[22],q[19],q[25],q[5],q[2],q[8],q[11];
measure q[23] -> meas[0];
measure q[18] -> meas[1];
measure q[21] -> meas[2];
measure q[12] -> meas[3];
measure q[15] -> meas[4];
