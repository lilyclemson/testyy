IMPORT * FROM $;
IMPORT Std.Str AS Str;
IMPORT ML.Mat;
IMPORT ML;
IMPORT ML.Types AS Types;
IMPORT ML.Utils;


lMatrix:={UNSIGNED id;REAL x;REAL y;};

dDocumentMatrix:=DATASET([
{1,2,2},
{2,4,2},
{3,6,2},
{4,9,2}
],lMatrix);

dCentroidMatrix:=DATASET([
{5,3,2},
{6,7,2}
],lMatrix);

//transform the input datasets to right format
ML.ToField(dDocumentMatrix,dDocuments);
ML.ToField(dCentroidMatrix,dCentroids);

//Inputs:
d01 := dDocuments;
d02 := dCentroids;
n := 10;
nConverge := 0;

//Step 2: run Standard Kmeans on the d02 to get t group of centroids

K := COUNT(d02)/MAX(d02,number);
t:= IF(K/10<1, 1, K/10);
nt := n;
tConverge := nConverge;

//K;
//t;
//temporary solution to get dt is that dt = the first t cendroids of d02 
temp := t * 2;
tempDt := d02[1..temp];
Types.NumericField transDt(Types.NumericField L, INTEGER C) := TRANSFORM
SELF.id := ROUNDUP(C/2);
SELF := L;
END;
dt := project(tempDt, transDt(LEFT, COUNTER));

//run Kmeans on d02
KmeansDt :=ML.Cluster.KMeans(d02,dt,nt,tConverge);
PreGroups := KmeansDt.Allegiances();//the assignment of each centroid to a group

//run Kmeans on d01
KmeansD02 := ML.Cluster.KMeans(d01,d02,n,nConverge);
ub := KmeansD02.Allegiances();
ubValue := SET(DEDUP(KmeansD02.Allegiances(),y), y); // all the initial ub are in this SET. 


//all the distances from each data point to each centroid
dDistances:= ML.Cluster.Distances(d01,d02);
joinlb := JOIN(ub,dDistances, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y,RIGHT ONLY);
// joinlb;

//Join PreGroups and dDistances
lBound := RECORD
TYPEOF(Types.NumericField.id) xid;
TYPEOF(Types.NumericField.id) cid;
TYPEOF(Types.NumericField.id) gid;
TYPEOF(Types.NumericField.value) lb;
END;


lBound transLbound(joinlb L, PreGroups R) := TRANSFORM
SELF.xid := L.x;
SELF.cid := L.y;
SELF.gid := R.y;
SELF.lb := L.value;
END;


jDistG := JOIN(joinlb, PreGroups, LEFT.y = RIGHT.x, transLbound(LEFT,RIGHT));
// jDistG;
//initialize lb

jDistGv := DEDUP(SORT(GROUP(jDistG,xid,gid),lb),gid);
// jDistGv ;

//iteration
//get C'

//Define the RECORD Structure of V
lV := RECORD
TYPEOF(Types.NumericField.id) id := ub.y;
TYPEOF(Types.NumericField.number) member := COUNT(GROUP);
SET OF TYPEOF(Types.NumericField.id) Vouts :=[];
SET OF TYPEOF(Types.NumericField.id) Vins :=[];
TYPEOF(Types.NumericField.value) Vin := 0;
TYPEOF(Types.NumericField.value) Vout := 0;
END;
			 		
// ub;
V := TABLE(ub,lV, ub.y);
// V;

lV transLv(V L, ub R) := TRANSFORM
SELF.Vins := L.Vins + [R.x];
SELF.Vin := L.Vin + R.value;
SELF := L; 
END;
Vini := JOIN(V,ub, LEFT.id = RIGHT.y, transLv(LEFT, RIGHT));
	
lV transLvfinal(V L, V R) := TRANSFORM
SELF.Vins := L.Vins + R.Vins;
SELF.Vin := L.Vin + R.Vin;
SELF := L; 
END;		
Vf := ROLLUP(Vini, LEFT.id = RIGHT.id, transLvfinal(LEFT,RIGHT));


//update C --> C'
x := TABLE(Vf,{Vins});

x[1];