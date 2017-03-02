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
iOffset:=IF(MAX(d01,id)>MIN(d02,id),MAX(d01,id),0);

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

//get lbs0
//lb of each group
lbs0Temp := JOIN(dClosest0,dDistances0, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y,RIGHT ONLY);
lbs0 := SORT(JOIN(lbs0Temp, Gt, LEFT.y = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.y := RIGHT.y; SELF := LEFT;)), x);
OUTPUT(lbs0Temp,NAMED('lbs0Temp')) ;
OUTPUT(lbs0,NAMED('lbs0')) ;


Vin0temp := TABLE(dClosest0, {y; UNSIGNED c := COUNT(GROUP);},y );
Vin0 := PROJECT(Vin0temp, TRANSFORM(Mat.Types.Element, SELF.x := LEFT.y; SELF.y := LEFT.c; SELF.value := 0; ));
OUTPUT(Vin0temp, NAMED('Vin0temp'));
OUTPUT(Vin0, NAMED('Vin0'));

Vout0 := Vin0;

lInput:=RECORD 
TYPEOF(Types.NumericField.id) id;
TYPEOF(Types.NumericField.id) x;
TYPEOF(Types.NumericField.id) y;
SET OF TYPEOF(Types.NumericField.value) values;
END;

lInput transFormat(Mat.Types.Element input , UNSIGNED c) := TRANSFORM
SELF.id := c;
SELF.values := [input.value];
SELF := input;
END;

// result := dCentroid0Trans + ub0 + lbs0 +Vin0 + Vout0 + dClosest0;
// OUTPUT(result);

iCentroid0 := PROJECT(dCentroid0Trans, transFormat(LEFT, 1));
iUb0 := PROJECT(ub0, transFormat(LEFT, 2));
iLbs0 := PROJECT(lbs0, transFormat(LEFT, 3));
iVin0 := PROJECT(Vin0, transFormat(LEFT, 4));
iVout0 := PROJECT(Vout0, transFormat(LEFT, 5));
// iClosest0 := PROJECT(dClosest0, transFormat(LEFT, 6));

OUTPUT(iCentroid0, NAMED('iCentroid0'));
OUTPUT(iUb0, NAMED('iUb0'));
OUTPUT(iLbs0, NAMED('iLbs0'));
OUTPUT(iVin0, NAMED('iVin0'));
OUTPUT(iVout0, NAMED('iVout0'));
// OUTPUT(iClosest0, NAMED('iClosest'));

result := iCentroid0 + iUb0 + iLbs0 + iVin0 + iVout0 ;
OUTPUT(result, NAMED('result'));

