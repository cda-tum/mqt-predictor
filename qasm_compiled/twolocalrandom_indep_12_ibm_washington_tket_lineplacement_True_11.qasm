OPENQASM 2.0;
include "qelib1.inc";

qreg node[126];
creg meas[12];
sx node[92];
rz(0.5*pi) node[101];
sx node[102];
sx node[103];
sx node[104];
sx node[105];
sx node[111];
sx node[121];
sx node[122];
sx node[123];
sx node[124];
sx node[125];
rz(3.0422031179230355*pi) node[92];
sx node[101];
rz(3.281143925979425*pi) node[102];
rz(3.075445910815289*pi) node[103];
rz(3.2320746846077935*pi) node[104];
rz(3.2567423474563597*pi) node[105];
rz(3.2032972991270494*pi) node[111];
rz(3.295538137894619*pi) node[121];
rz(3.28845184595117*pi) node[122];
rz(3.0405669339460726*pi) node[123];
rz(3.0677217244932242*pi) node[124];
rz(3.2221677680947733*pi) node[125];
sx node[92];
rz(3.5*pi) node[101];
sx node[102];
sx node[103];
sx node[104];
sx node[105];
sx node[111];
sx node[121];
sx node[122];
sx node[123];
sx node[124];
sx node[125];
rz(1.0*pi) node[92];
sx node[101];
rz(1.0*pi) node[102];
rz(1.0*pi) node[103];
rz(1.0*pi) node[104];
rz(1.0*pi) node[105];
rz(1.0*pi) node[111];
rz(1.0*pi) node[121];
rz(1.0*pi) node[122];
rz(1.0*pi) node[123];
rz(1.0*pi) node[124];
rz(1.0*pi) node[125];
rz(0.5864806126230769*pi) node[101];
cx node[123],node[124];
cx node[125],node[124];
cx node[124],node[125];
cx node[125],node[124];
cx node[123],node[124];
cx node[123],node[122];
cx node[125],node[124];
cx node[123],node[122];
cx node[125],node[124];
cx node[122],node[123];
cx node[124],node[125];
cx node[123],node[122];
cx node[125],node[124];
cx node[122],node[111];
cx node[124],node[123];
cx node[122],node[121];
cx node[124],node[123];
cx node[122],node[111];
cx node[123],node[124];
cx node[111],node[122];
cx node[124],node[123];
cx node[122],node[111];
cx node[125],node[124];
cx node[111],node[104];
cx node[123],node[122];
cx node[125],node[124];
cx node[111],node[104];
cx node[123],node[122];
cx node[124],node[125];
cx node[104],node[111];
cx node[122],node[123];
cx node[125],node[124];
cx node[111],node[104];
cx node[123],node[122];
cx node[104],node[103];
cx node[122],node[121];
cx node[124],node[123];
cx node[104],node[105];
cx node[122],node[111];
cx node[124],node[123];
cx node[104],node[103];
cx node[122],node[111];
cx node[123],node[124];
cx node[103],node[104];
cx node[111],node[122];
cx node[124],node[123];
cx node[104],node[103];
cx node[122],node[111];
cx node[125],node[124];
cx node[103],node[102];
cx node[111],node[104];
cx node[121],node[122];
cx node[125],node[124];
cx node[103],node[102];
cx node[104],node[111];
cx node[122],node[121];
cx node[124],node[125];
cx node[102],node[103];
cx node[111],node[104];
cx node[121],node[122];
cx node[125],node[124];
cx node[103],node[102];
cx node[104],node[111];
cx node[123],node[122];
cx node[102],node[92];
cx node[104],node[105];
cx node[123],node[122];
cx node[102],node[101];
cx node[104],node[103];
cx node[122],node[123];
sx node[102];
cx node[104],node[103];
cx node[123],node[122];
rz(3.046523597018987*pi) node[102];
cx node[103],node[104];
cx node[122],node[121];
cx node[124],node[123];
sx node[102];
cx node[104],node[103];
cx node[122],node[111];
cx node[124],node[123];
rz(1.0*pi) node[102];
cx node[105],node[104];
cx node[122],node[111];
cx node[123],node[124];
cx node[92],node[102];
cx node[104],node[105];
cx node[111],node[122];
cx node[124],node[123];
cx node[102],node[92];
cx node[105],node[104];
cx node[122],node[111];
cx node[125],node[124];
cx node[92],node[102];
cx node[111],node[104];
cx node[121],node[122];
cx node[125],node[124];
cx node[103],node[102];
cx node[104],node[111];
cx node[122],node[121];
cx node[124],node[125];
cx node[103],node[102];
cx node[111],node[104];
cx node[121],node[122];
cx node[125],node[124];
cx node[102],node[103];
cx node[104],node[111];
cx node[123],node[122];
cx node[103],node[102];
cx node[104],node[105];
cx node[123],node[122];
cx node[102],node[101];
cx node[104],node[103];
cx node[122],node[123];
sx node[102];
cx node[104],node[103];
cx node[123],node[122];
rz(3.0393974610461783*pi) node[102];
cx node[103],node[104];
cx node[122],node[121];
cx node[124],node[123];
sx node[102];
cx node[104],node[103];
cx node[122],node[111];
cx node[124],node[123];
rz(1.0*pi) node[102];
cx node[105],node[104];
cx node[122],node[111];
cx node[123],node[124];
cx node[92],node[102];
cx node[104],node[105];
cx node[111],node[122];
cx node[124],node[123];
cx node[101],node[102];
cx node[105],node[104];
cx node[122],node[111];
cx node[125],node[124];
cx node[102],node[101];
cx node[111],node[104];
cx node[121],node[122];
cx node[125],node[124];
cx node[101],node[102];
cx node[104],node[111];
cx node[122],node[121];
cx node[124],node[125];
cx node[103],node[102];
cx node[111],node[104];
cx node[121],node[122];
cx node[125],node[124];
sx node[103];
cx node[104],node[111];
cx node[123],node[122];
rz(3.2332133140547796*pi) node[103];
cx node[104],node[105];
cx node[123],node[122];
sx node[103];
cx node[122],node[123];
rz(1.0*pi) node[103];
cx node[123],node[122];
cx node[103],node[102];
cx node[122],node[121];
cx node[124],node[123];
cx node[102],node[103];
cx node[122],node[111];
cx node[124],node[123];
cx node[103],node[102];
cx node[122],node[111];
cx node[123],node[124];
cx node[92],node[102];
cx node[104],node[103];
cx node[111],node[122];
cx node[124],node[123];
cx node[101],node[102];
sx node[104];
cx node[122],node[111];
cx node[125],node[124];
cx node[92],node[102];
rz(3.2806666275610574*pi) node[104];
cx node[121],node[122];
cx node[125],node[124];
cx node[102],node[92];
sx node[104];
cx node[122],node[121];
cx node[124],node[125];
cx node[92],node[102];
rz(1.0*pi) node[104];
cx node[121],node[122];
cx node[125],node[124];
cx node[104],node[103];
cx node[123],node[122];
cx node[103],node[104];
cx node[123],node[122];
cx node[104],node[103];
cx node[122],node[123];
cx node[102],node[103];
cx node[105],node[104];
cx node[123],node[122];
cx node[103],node[102];
cx node[104],node[105];
cx node[122],node[121];
cx node[124],node[123];
cx node[102],node[103];
cx node[105],node[104];
cx node[124],node[123];
cx node[103],node[102];
cx node[111],node[104];
cx node[123],node[124];
cx node[101],node[102];
cx node[104],node[111];
cx node[124],node[123];
cx node[92],node[102];
cx node[111],node[104];
cx node[125],node[124];
cx node[101],node[102];
cx node[104],node[111];
cx node[125],node[124];
cx node[102],node[101];
cx node[104],node[105];
cx node[122],node[111];
cx node[124],node[125];
cx node[101],node[102];
sx node[104];
cx node[122],node[111];
cx node[125],node[124];
rz(3.2217038443566812*pi) node[104];
cx node[111],node[122];
sx node[104];
cx node[122],node[111];
rz(1.0*pi) node[104];
cx node[121],node[122];
cx node[103],node[104];
cx node[122],node[121];
cx node[104],node[103];
cx node[121],node[122];
cx node[103],node[104];
cx node[123],node[122];
cx node[104],node[103];
cx node[123],node[122];
cx node[102],node[103];
cx node[105],node[104];
cx node[122],node[123];
cx node[103],node[102];
cx node[104],node[105];
cx node[123],node[122];
cx node[102],node[103];
cx node[105],node[104];
cx node[122],node[121];
cx node[124],node[123];
cx node[103],node[102];
cx node[111],node[104];
cx node[124],node[123];
cx node[92],node[102];
sx node[111];
cx node[123],node[124];
cx node[101],node[102];
rz(3.2090688798966*pi) node[111];
cx node[124],node[123];
cx node[92],node[102];
sx node[111];
cx node[125],node[124];
cx node[102],node[92];
rz(1.0*pi) node[111];
cx node[125],node[124];
cx node[92],node[102];
cx node[104],node[111];
cx node[124],node[125];
cx node[111],node[104];
cx node[125],node[124];
cx node[104],node[111];
cx node[105],node[104];
cx node[122],node[111];
cx node[103],node[104];
sx node[122];
cx node[104],node[103];
rz(3.161394477173841*pi) node[122];
cx node[103],node[104];
sx node[122];
cx node[104],node[103];
rz(1.0*pi) node[122];
cx node[102],node[103];
cx node[105],node[104];
cx node[122],node[111];
cx node[103],node[102];
cx node[104],node[105];
cx node[111],node[122];
cx node[102],node[103];
cx node[105],node[104];
cx node[122],node[111];
cx node[103],node[102];
cx node[104],node[111];
cx node[121],node[122];
cx node[101],node[102];
cx node[104],node[111];
cx node[122],node[121];
cx node[92],node[102];
cx node[111],node[104];
cx node[121],node[122];
cx node[101],node[102];
cx node[104],node[111];
cx node[123],node[122];
cx node[102],node[101];
cx node[105],node[104];
cx node[123],node[122];
cx node[101],node[102];
cx node[103],node[104];
cx node[122],node[123];
cx node[104],node[103];
cx node[123],node[122];
cx node[103],node[104];
cx node[122],node[121];
cx node[124],node[123];
cx node[104],node[103];
sx node[122];
cx node[124],node[123];
cx node[102],node[103];
cx node[105],node[104];
rz(3.27570016704673*pi) node[122];
cx node[123],node[124];
cx node[103],node[102];
cx node[104],node[105];
sx node[122];
cx node[124],node[123];
cx node[102],node[103];
cx node[105],node[104];
rz(1.0*pi) node[122];
cx node[125],node[124];
cx node[103],node[102];
cx node[111],node[122];
cx node[125],node[124];
cx node[92],node[102];
cx node[122],node[111];
cx node[124],node[125];
cx node[101],node[102];
cx node[111],node[122];
cx node[125],node[124];
cx node[92],node[102];
cx node[122],node[111];
cx node[102],node[92];
cx node[104],node[111];
cx node[121],node[122];
cx node[92],node[102];
cx node[104],node[111];
cx node[122],node[121];
cx node[111],node[104];
cx node[121],node[122];
cx node[104],node[111];
cx node[123],node[122];
cx node[105],node[104];
sx node[123];
cx node[103],node[104];
rz(3.2072203472653458*pi) node[123];
cx node[104],node[103];
sx node[123];
cx node[103],node[104];
rz(1.0*pi) node[123];
cx node[104],node[103];
cx node[123],node[122];
cx node[102],node[103];
cx node[105],node[104];
cx node[122],node[123];
cx node[103],node[102];
cx node[104],node[105];
cx node[123],node[122];
cx node[102],node[103];
cx node[105],node[104];
cx node[121],node[122];
cx node[124],node[123];
cx node[103],node[102];
cx node[111],node[122];
sx node[124];
cx node[101],node[102];
cx node[122],node[111];
rz(3.0196705595924427*pi) node[124];
cx node[92],node[102];
cx node[111],node[122];
sx node[124];
cx node[101],node[102];
cx node[122],node[111];
rz(1.0*pi) node[124];
cx node[102],node[101];
cx node[104],node[111];
cx node[121],node[122];
cx node[124],node[123];
cx node[101],node[102];
cx node[104],node[111];
cx node[122],node[121];
cx node[123],node[124];
cx node[111],node[104];
cx node[121],node[122];
cx node[124],node[123];
cx node[104],node[111];
cx node[122],node[123];
cx node[125],node[124];
cx node[105],node[104];
cx node[123],node[122];
rz(0.047379034919996066*pi) node[124];
sx node[125];
cx node[103],node[104];
cx node[122],node[123];
rz(3.1102523787738274*pi) node[125];
cx node[104],node[103];
cx node[123],node[122];
sx node[125];
cx node[103],node[104];
cx node[121],node[122];
rz(1.0*pi) node[125];
cx node[104],node[103];
cx node[111],node[122];
cx node[125],node[124];
cx node[102],node[103];
cx node[105],node[104];
cx node[122],node[111];
cx node[124],node[125];
cx node[103],node[102];
cx node[104],node[105];
cx node[111],node[122];
cx node[125],node[124];
cx node[102],node[103];
cx node[105],node[104];
cx node[122],node[111];
cx node[123],node[124];
cx node[103],node[102];
cx node[104],node[111];
cx node[121],node[122];
cx node[124],node[123];
cx node[92],node[102];
cx node[104],node[111];
cx node[122],node[121];
cx node[123],node[124];
cx node[101],node[102];
cx node[111],node[104];
cx node[121],node[122];
cx node[124],node[123];
cx node[92],node[102];
cx node[104],node[111];
cx node[122],node[123];
cx node[124],node[125];
cx node[102],node[92];
cx node[105],node[104];
cx node[123],node[122];
sx node[124];
cx node[92],node[102];
cx node[103],node[104];
cx node[122],node[123];
rz(3.1666121268945124*pi) node[124];
cx node[104],node[103];
cx node[123],node[122];
sx node[124];
cx node[103],node[104];
cx node[121],node[122];
rz(1.0*pi) node[124];
cx node[104],node[103];
cx node[111],node[122];
cx node[125],node[124];
cx node[102],node[103];
cx node[105],node[104];
cx node[122],node[111];
cx node[124],node[125];
cx node[103],node[102];
cx node[104],node[105];
cx node[111],node[122];
cx node[125],node[124];
cx node[102],node[103];
cx node[105],node[104];
cx node[122],node[111];
cx node[123],node[124];
cx node[103],node[102];
cx node[104],node[111];
cx node[121],node[122];
sx node[123];
cx node[101],node[102];
cx node[104],node[111];
cx node[122],node[121];
rz(3.0015444490883776*pi) node[123];
cx node[92],node[102];
cx node[111],node[104];
cx node[121],node[122];
sx node[123];
cx node[101],node[102];
cx node[104],node[111];
rz(1.0*pi) node[123];
cx node[102],node[101];
cx node[105],node[104];
cx node[124],node[123];
cx node[101],node[102];
cx node[103],node[104];
cx node[123],node[124];
cx node[104],node[103];
cx node[124],node[123];
cx node[103],node[104];
cx node[122],node[123];
cx node[125],node[124];
cx node[104],node[103];
sx node[122];
cx node[125],node[124];
cx node[102],node[103];
cx node[105],node[104];
rz(3.1587455773486255*pi) node[122];
cx node[124],node[125];
cx node[103],node[102];
cx node[104],node[105];
sx node[122];
cx node[125],node[124];
cx node[102],node[103];
cx node[105],node[104];
rz(1.0*pi) node[122];
cx node[103],node[102];
cx node[123],node[122];
cx node[92],node[102];
cx node[122],node[123];
cx node[101],node[102];
cx node[123],node[122];
cx node[92],node[102];
cx node[121],node[122];
cx node[124],node[123];
cx node[102],node[92];
cx node[111],node[122];
sx node[121];
cx node[124],node[123];
cx node[92],node[102];
sx node[111];
rz(3.215607342750273*pi) node[121];
cx node[123],node[124];
rz(3.2344196156355447*pi) node[111];
sx node[121];
cx node[124],node[123];
sx node[111];
rz(1.0*pi) node[121];
cx node[125],node[124];
rz(1.0*pi) node[111];
cx node[125],node[124];
cx node[122],node[111];
cx node[124],node[125];
cx node[111],node[122];
cx node[125],node[124];
cx node[122],node[111];
cx node[104],node[111];
cx node[121],node[122];
sx node[104];
cx node[122],node[121];
rz(3.010800386027634*pi) node[104];
cx node[121],node[122];
sx node[104];
cx node[123],node[122];
rz(1.0*pi) node[104];
cx node[123],node[122];
cx node[104],node[111];
cx node[122],node[123];
cx node[111],node[104];
cx node[123],node[122];
cx node[104],node[111];
cx node[122],node[121];
cx node[124],node[123];
cx node[105],node[104];
cx node[122],node[111];
cx node[124],node[123];
cx node[103],node[104];
sx node[105];
cx node[122],node[111];
cx node[123],node[124];
sx node[103];
rz(3.232456030045526*pi) node[105];
cx node[111],node[122];
cx node[124],node[123];
rz(3.2569700425196775*pi) node[103];
sx node[105];
cx node[122],node[111];
cx node[125],node[124];
sx node[103];
rz(1.0*pi) node[105];
cx node[121],node[122];
cx node[125],node[124];
rz(1.0*pi) node[103];
cx node[122],node[121];
cx node[124],node[125];
cx node[104],node[103];
cx node[121],node[122];
cx node[125],node[124];
cx node[103],node[104];
cx node[123],node[122];
cx node[104],node[103];
cx node[123],node[122];
cx node[102],node[103];
cx node[105],node[104];
cx node[122],node[123];
sx node[102];
cx node[104],node[105];
cx node[123],node[122];
rz(3.1558766868766672*pi) node[102];
cx node[105],node[104];
cx node[122],node[121];
cx node[124],node[123];
sx node[102];
cx node[111],node[104];
cx node[124],node[123];
rz(1.0*pi) node[102];
cx node[104],node[111];
cx node[123],node[124];
cx node[103],node[102];
cx node[111],node[104];
cx node[124],node[123];
cx node[102],node[103];
cx node[104],node[111];
cx node[125],node[124];
cx node[103],node[102];
cx node[104],node[105];
cx node[122],node[111];
cx node[125],node[124];
cx node[101],node[102];
cx node[104],node[103];
cx node[122],node[111];
cx node[124],node[125];
cx node[92],node[102];
sx node[101];
cx node[104],node[103];
cx node[111],node[122];
cx node[125],node[124];
sx node[92];
rz(3.2272102109069585*pi) node[101];
rz(0.22345155773655667*pi) node[102];
cx node[103],node[104];
cx node[122],node[111];
rz(3.2046409908652262*pi) node[92];
sx node[101];
cx node[104],node[103];
cx node[121],node[122];
sx node[92];
rz(1.0*pi) node[101];
cx node[105],node[104];
cx node[122],node[121];
rz(1.0*pi) node[92];
cx node[101],node[102];
cx node[104],node[105];
cx node[121],node[122];
cx node[102],node[101];
cx node[105],node[104];
cx node[123],node[122];
cx node[101],node[102];
cx node[111],node[104];
cx node[123],node[122];
cx node[103],node[102];
cx node[104],node[111];
cx node[122],node[123];
cx node[103],node[102];
cx node[111],node[104];
cx node[123],node[122];
cx node[102],node[103];
cx node[104],node[111];
cx node[122],node[121];
cx node[124],node[123];
cx node[103],node[102];
cx node[104],node[105];
cx node[122],node[111];
cx node[124],node[123];
cx node[102],node[92];
cx node[104],node[103];
cx node[122],node[111];
cx node[123],node[124];
cx node[102],node[101];
cx node[104],node[103];
cx node[111],node[122];
cx node[124],node[123];
sx node[102];
cx node[103],node[104];
cx node[122],node[111];
cx node[125],node[124];
rz(3.030064283036931*pi) node[102];
cx node[104],node[103];
cx node[121],node[122];
cx node[125],node[124];
sx node[102];
cx node[105],node[104];
cx node[122],node[121];
cx node[124],node[125];
rz(1.0*pi) node[102];
cx node[104],node[105];
cx node[121],node[122];
cx node[125],node[124];
cx node[92],node[102];
cx node[105],node[104];
cx node[123],node[122];
cx node[102],node[92];
cx node[111],node[104];
cx node[123],node[122];
cx node[92],node[102];
cx node[104],node[111];
cx node[122],node[123];
cx node[103],node[102];
cx node[111],node[104];
cx node[123],node[122];
cx node[103],node[102];
cx node[104],node[111];
cx node[122],node[121];
cx node[124],node[123];
cx node[102],node[103];
cx node[104],node[105];
cx node[122],node[111];
cx node[124],node[123];
cx node[103],node[102];
cx node[122],node[111];
cx node[123],node[124];
cx node[102],node[101];
cx node[104],node[103];
cx node[111],node[122];
cx node[124],node[123];
sx node[102];
cx node[104],node[103];
cx node[122],node[111];
cx node[125],node[124];
rz(3.10574696667459*pi) node[102];
cx node[103],node[104];
cx node[121],node[122];
cx node[125],node[124];
sx node[102];
cx node[104],node[103];
cx node[122],node[121];
cx node[124],node[125];
rz(1.0*pi) node[102];
cx node[105],node[104];
cx node[121],node[122];
cx node[125],node[124];
cx node[101],node[102];
cx node[104],node[105];
cx node[123],node[122];
cx node[102],node[101];
cx node[105],node[104];
cx node[123],node[122];
cx node[101],node[102];
cx node[111],node[104];
cx node[122],node[123];
cx node[103],node[102];
cx node[104],node[111];
cx node[123],node[122];
sx node[103];
cx node[111],node[104];
cx node[122],node[121];
cx node[124],node[123];
rz(3.160392098884689*pi) node[103];
cx node[104],node[111];
cx node[124],node[123];
sx node[103];
cx node[104],node[105];
cx node[122],node[111];
cx node[123],node[124];
rz(1.0*pi) node[103];
cx node[122],node[111];
cx node[124],node[123];
cx node[102],node[103];
cx node[111],node[122];
cx node[125],node[124];
cx node[103],node[102];
cx node[122],node[111];
cx node[125],node[124];
cx node[102],node[103];
cx node[121],node[122];
cx node[124],node[125];
cx node[104],node[103];
cx node[122],node[121];
cx node[125],node[124];
sx node[104];
cx node[121],node[122];
rz(3.1537100570107763*pi) node[104];
cx node[123],node[122];
sx node[104];
cx node[123],node[122];
rz(1.0*pi) node[104];
cx node[122],node[123];
cx node[105],node[104];
cx node[123],node[122];
cx node[104],node[105];
cx node[122],node[121];
cx node[124],node[123];
cx node[105],node[104];
cx node[124],node[123];
cx node[111],node[104];
cx node[123],node[124];
cx node[111],node[104];
cx node[124],node[123];
cx node[104],node[111];
cx node[125],node[124];
cx node[111],node[104];
cx node[125],node[124];
cx node[104],node[103];
cx node[122],node[111];
cx node[124],node[125];
sx node[104];
cx node[122],node[111];
cx node[125],node[124];
rz(3.0190660216983485*pi) node[104];
cx node[111],node[122];
sx node[104];
cx node[122],node[111];
rz(1.0*pi) node[104];
cx node[121],node[122];
cx node[103],node[104];
cx node[122],node[121];
cx node[104],node[103];
cx node[121],node[122];
cx node[103],node[104];
cx node[123],node[122];
cx node[111],node[104];
cx node[123],node[122];
sx node[111];
cx node[122],node[123];
rz(3.0905305211179526*pi) node[111];
cx node[123],node[122];
sx node[111];
cx node[122],node[121];
cx node[124],node[123];
rz(1.0*pi) node[111];
cx node[124],node[123];
cx node[104],node[111];
cx node[123],node[124];
cx node[111],node[104];
cx node[124],node[123];
cx node[104],node[111];
cx node[125],node[124];
cx node[122],node[111];
cx node[125],node[124];
sx node[122];
cx node[124],node[125];
rz(3.0967892797662744*pi) node[122];
cx node[125],node[124];
sx node[122];
rz(1.0*pi) node[122];
cx node[121],node[122];
cx node[122],node[121];
cx node[121],node[122];
cx node[123],node[122];
cx node[123],node[122];
cx node[122],node[123];
cx node[123],node[122];
cx node[122],node[111];
cx node[124],node[123];
sx node[122];
cx node[125],node[124];
rz(3.099774525118433*pi) node[122];
cx node[124],node[123];
sx node[122];
cx node[125],node[124];
rz(1.0*pi) node[122];
cx node[124],node[123];
cx node[111],node[122];
cx node[122],node[111];
cx node[111],node[122];
cx node[122],node[123];
cx node[123],node[122];
cx node[122],node[123];
cx node[124],node[123];
sx node[124];
rz(3.037270108062495*pi) node[124];
sx node[124];
rz(1.0*pi) node[124];
cx node[125],node[124];
cx node[124],node[125];
cx node[125],node[124];
cx node[124],node[123];
cx node[122],node[123];
sx node[124];
sx node[122];
rz(3.7189641429969553*pi) node[123];
rz(3.10404632248258*pi) node[124];
rz(3.065682037876354*pi) node[122];
sx node[123];
sx node[124];
sx node[122];
rz(3.5*pi) node[123];
rz(1.0*pi) node[124];
rz(1.0*pi) node[122];
sx node[123];
rz(1.5*pi) node[123];
barrier node[92],node[101],node[102],node[105],node[103],node[104],node[121],node[111],node[125],node[124],node[122],node[123];
measure node[92] -> meas[0];
measure node[101] -> meas[1];
measure node[102] -> meas[2];
measure node[105] -> meas[3];
measure node[103] -> meas[4];
measure node[104] -> meas[5];
measure node[121] -> meas[6];
measure node[111] -> meas[7];
measure node[125] -> meas[8];
measure node[124] -> meas[9];
measure node[122] -> meas[10];
measure node[123] -> meas[11];
