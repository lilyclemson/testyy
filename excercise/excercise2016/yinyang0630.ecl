IMPORT * FROM $;
IMPORT Std.Str AS Str;
IMPORT TS;
IMPORT ML.Mat;
IMPORT ML;
IMPORT ML.Cluster.DF as DF;
IMPORT ML.Types AS Types;
IMPORT ML.Utils;
IMPORT excercise.Cluster as Cluster;

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


ML.ToField(dDocumentMatrix,dDocuments);
ML.ToField(dCentroidMatrix,dCentroids);

//Inputs:
d01 := dDocuments;
d02 := dCentroids;
n := 10;
nConverge := 0;

//run Standard Kmeans on the d02 to get t group of centroids

K := COUNT(d02)/MAX(d02,number);
t:= IF(K/10<1, 1, K/10);
nt := n;
tConverge := nConverge;

K;//********************result 1 
t;//********************result 2
//temporary solution to get dt is that dt = the first t cendroids of d02 
temp := t * 2;
tempDt := d02[1..temp];
tempDt;//****************result 
//transform ids of the centroids of dt starting from 1....
Types.NumericField transDt(Types.NumericField L, INTEGER C) := TRANSFORM
SELF.id := ROUNDUP(C/2);
SELF := L;
END;
dt := project(tempDt, transDt(LEFT, COUNTER));

dt;//****************result 4


//run Kmeans on d02
KmeansDt :=ML.Cluster.KMeans(d02,dt,nt,tConverge);
PreGroups := KmeansDt.Allegiances();//the assignment of each centroid to a group
PreGroups;//**************result 5****** Gt



//run Kmeans on d01
KmeansD02 := ML.Cluster.KMeans(d01,d02,n,nConverge);
ub := KmeansD02.Allegiances();
ub; //***************result 6 ********dClosest


ubValue := SET(DEDUP(KmeansD02.Allegiances(),y), y); // all the initial ub are in this SET. 
ubValue; //***************result 7

//all the distances from each data point to each centroid
dDistances:= ML.Cluster.Distances(d01,d02);
joinlb := JOIN(ub,dDistances, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y,RIGHT ONLY);
joinlb;//***************result 8 ==  dDistances - dClosest

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

//join result 8 and result 5 to get the dataset for lb
jDistG := JOIN(joinlb, PreGroups, LEFT.y = RIGHT.x, transLbound(LEFT,RIGHT));
jDistG;//***************result 9 

//initialize lb

jDistGv := DEDUP(SORT(GROUP(jDistG,xid,gid),lb),gid);
jDistGv ;//***************result 10 == lb of each group

//iteration
//get C'

//Define the RECORD Structure of V
lV := RECORD
TYPEOF(Types.NumericField.id) id := ub.y;
TYPEOF(Types.NumericField.number) member := COUNT(GROUP);
SET OF TYPEOF(Types.NumericField.id) Vouts :=[];
SET OF TYPEOF(Types.NumericField.id) Vins :=[];
// TYPEOF(Types.NumericField.value) Vin := 0;
// TYPEOF(Types.NumericField.value) Vout := 0;
END;
			 		
V := TABLE(ub,lV, ub.y);
V;//***************result 11

lV transLv(V L, ub R) := TRANSFORM
SELF.Vins := L.Vins + [R.x];
SELF.Vin := L.Vin + R.value;
SELF := L; 
END;
Vini := JOIN(V,ub, LEFT.id = RIGHT.y, transLv(LEFT, RIGHT));
Vini;//***************result 12

lV transLvfinal(V L, V R) := TRANSFORM
SELF.Vins := L.Vins + R.Vins;
SELF.Vin := L.Vin + R.Vin;
SELF := L; 
END;		
Vf := ROLLUP(Vini, LEFT.id = RIGHT.id, transLvfinal(LEFT,RIGHT));

Vf;//***************result 13


// Join closest to the document set and replace the id with the centriod id
dClustered := SORT(DISTRIBUTE(JOIN(d01,ub,LEFT.id=RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;),HASH),id),RECORD,LOCAL);

d02Prep;//***************result 14
dCentroid;//***************result 15
dClustered;//***************result 16
// Now roll up on centroid ID, summing up the values for each axis
dRolled:=ROLLUP(dClustered,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;),id,number,LOCAL);
// Join to cluster counts to calculate the new average on each axis
dJoined:=JOIN(dRolled,Vf,LEFT.id=RIGHT.id,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value/RIGHT.member;SELF:=LEFT;),LOOKUP);
dJoined;//***************result 17
// Find any centroids with no document allegiance and pass those through also
dPass:=JOIN(dCentroids,TABLE(dJoined,{id},id,LOCAL),LEFT.id=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);
// Now join to the existing centroid dataset to add the new values to
// the end of the values set.
dAdded:=JOIN(d02Prep,dJoined+dPass,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=RIGHT;),RIGHT OUTER);
// If the centroids have converged, simply pass the input dataset through
// to the next iteration.  Otherwise perform an iteration.
dAdded;//***************result 18
// Check the distance delta for the last two iterations.  If the highest
// value is below the convergence threshold, then set bConverged to TRUE

//update C --> C'
// x := TABLE(Vf,{Vins});
// Function to pull iteration N from a table of type lIterations
Types.NumericField dResult(UNSIGNED n=n,DATASET(lIterations) d):=PROJECT(d,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[n+1];SELF:=LEFT;));

    // Determine the delta along each axis between any two iterations
Types.NumericField tGetDelta(Types.NumericField L,Types.NumericField R):=TRANSFORM
      SELF.id:=IF(L.id=0,R.id,L.id);
      SELF.number:=IF(L.number=0,R.number,L.number);
      SELF.value:=R.value-L.value;
END;
dDelta(UNSIGNED n01=n-1,UNSIGNED n02=n,DATASET(lIterations) d):=JOIN(dResult(n01,d),dResult(n02,d),LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,tGetDelta(LEFT,RIGHT));
    
    // Determine the distance delta between two iterations, using the distance
    // method specified by the user for this module
fDist :=DF.Euclidean;
dDistanceDelta(UNSIGNED n01=n-1,UNSIGNED n02=n,DATASET(lIterations) d):=FUNCTION
      iMax01:=MAX(dResult(n01,d),id);
      dDistancell:=ML.Cluster.Distances(dResult(n01,d),PROJECT(dResult(n02,d),TRANSFORM(Types.NumericField,SELF.id:=LEFT.id+iMax01;SELF:=LEFT;)),fDist);
RETURN PROJECT(dDistancell(x=y-iMax01),TRANSFORM({Types.NumericField AND NOT [number];},SELF.id:=LEFT.x;SELF:=LEFT;));
END;
bConverged:=IF(c=1,FALSE,MAX(dDistanceDelta(c-1,c-2,d02Prep),value)<=nConverge);
bConverged;//***************result 19
result := IF(bConverged,d02Prep,dAdded);
result ;//***************result 20
// update c
c := 2;