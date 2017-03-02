IMPORT * FROM $;
IMPORT Std.Str AS Str;
IMPORT TS;
IMPORT ML.Mat;
IMPORT ML;
IMPORT ML.Cluster.DF as DF;
IMPORT ML.Types AS Types;
IMPORT ML.Utils;
IMPORT ML.MAT AS Mat;

lMatrix:={UNSIGNED id;REAL x;REAL y;};

dDocumentMatrix:=DATASET([
{1,1,1},
{2,1.5,2},
{3,3,4},
{4,5,7},
{5,3.5,5},
{6,4.5,5},
{7,3.5,4.5}
],lMatrix);

dCentroidMatrix:=DATASET([
{1,1,1},
{2,5,7}
],lMatrix);

ML.ToField(dDocumentMatrix,dDocuments);
ML.ToField(dCentroidMatrix,dCentroids);

//***************************************Initialization*********************************
//initialization c := 0;
c := 0;

//Inputs:
d01 := dDocuments;
d02 := dCentroids;
n := 10;
nConverge := 0;

// internal record structure of d02
lIterations:=RECORD 
TYPEOF(Types.NumericField.id) id;
TYPEOF(Types.NumericField.number) number;
SET OF TYPEOF(Types.NumericField.value) values;
END;

//make sure all ids are different
iOffsetC:=IF(MAX(d01,id)>MIN(d02,id),MAX(d01,id),0);

//transfrom the d02 to internal data structure
d02Prep:=PROJECT(d02,TRANSFORM(lIterations,SELF.id:=LEFT.id+iOffset;SELF.values:=[LEFT.value];SELF:=LEFT;));


dCentroid0 := PROJECT(d02Prep,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[1];SELF:=LEFT;));
OUTPUT(dCentroid0,NAMED('Initial_Centroids')); 

//transform dCentroid to the format align with ub0, lbs0, Vin0, Vout0
dCentroid0Trans := PROJECT(dCentroid0, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value ; SELF.x := LEFT.id; SELF.y := LEFT.number));

// get Gt 
//running Kmeans on dCentroid
//inputs

K := COUNT(DEDUP(d02,id));
// K; 
t:= IF(K/10<1, 1, K/10);
// t; 

//temporary solution to get dt is that dt = the first t cendroids of dCentroid 
temp := t * 2;
tempDt := dCentroid0[1..temp];
//transform ids of the centroids of dt starting from 1....
Types.NumericField transDt(Types.NumericField L, INTEGER c) := TRANSFORM
SELF.id := ROUNDUP(c/2);
SELF := L;
END;
dt := project(tempDt, transDt(LEFT, COUNTER));
// dt;

nt := n;
tConverge := nConverge;

//run Kmeans on dCentroid to get Gt
KmeansDt :=ML.Cluster.KMeans(dCentroid0,dt,nt,tConverge);

Gt := TABLE(KmeansDt.Allegiances(), {x,y},y,x);//the assignment of each centroid to a group
OUTPUT(Gt, NAMED('Gt'));


//initialize ub and lb for each data point
//get ub0
dDistances0 := ML.Cluster.Distances(d01,dCentroid0);
dClosest0 := ML.Cluster.Closest(dDistances0);
// dClosest0;
ub0 := dClosest0; 
OUTPUT(ub0, NAMED('ub0'));

//make sure all ids are different
iOffsetCub:=IF(MAX(dCentroidTrans,id)>MIN(d02,id),MAX(d01,id),0);

//get lbs0
//lb of each group
lbs0Temp := JOIN(dClosest0,dDistances0, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y,RIGHT ONLY);
lbs0 := SORT(JOIN(lbs0Temp, Gt, LEFT.y = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.y := RIGHT.y; SELF := LEFT;)), x);
OUTPUT(lbs0Temp,NAMED('lbs0Temp')) ;
OUTPUT(lbs0,NAMED('lbs0')) ;


//Define the RECORD Structure of V
lV := RECORD
TYPEOF(Types.NumericField.id) id;
TYPEOF(Types.NumericField.id) xid;
END;
			 		
//initialize Vcount0
//Get V
lVcount := RECORD
TYPEOF(Types.NumericField.id) id := dClosest0.y;
TYPEOF(Types.NumericField.number) c := COUNT(group);
TYPEOF(Types.NumericField.value) inNum := 0;
TYPEOF(Types.NumericField.value) outNum := 0;
END;

Vcount0 := TABLE(dClosest0, lVcount, y);
OUTPUT(Vcount0, NAMED('Vcount0')); 	

Vin0 := PROJECT(Vcount0, TRANSFORM(Mat.Types.Element,SELF.y := LEFT.c; SELF.x := LEFT.id; SELF.value := LEFT.innum;));
OUTPUT(Vin0, NAME('Vin0'));
Vout0 := PROJECT(Vcount0, TRANSFORM(Mat.Types.Element,SELF.y := LEFT.c; SELF.x := LEFT.id; SELF.value := LEFT.outnum;));
OUTPUT(Vout0, NAME('Vout0'));


result := dCentroid0Trans + ub0 + lbs0 +Vin0 + Vout0;
OUTPUT(result, NAMED('result'));