IMPORT * FROM $;
IMPORT Std.Str AS Str;
IMPORT TS;
IMPORT ML.Mat;
IMPORT ML;
IMPORT ML.Cluster.DF as DF;
IMPORT ML.Types AS Types;
IMPORT ML.Utils;

lMatrix:={UNSIGNED id;REAL x;REAL y;};

// dDocumentMatrix:=DATASET([
// {1,2,2},
// {2,4,2},
// {3,6,2},
// {4,9,2}
// ],lMatrix);

// dCentroidMatrix:=DATASET([
// {1,3,2},
// {2,7,2}
// ],lMatrix);
dDocumentMatrix:=DATASET([
{1,2.4639,7.8579},
{2,0.5573,9.4681},
{3,4.6054,8.4723},
{4,1.24,7.3835},
{5,7.8253,4.8205},
{6,3.0965,3.4085},
{7,8.8631,1.4446},
{8,5.8085,9.1887},
{9,1.3813,0.515},
{10,2.7123,9.2429},
{11,6.786,4.9368},
{12,9.0227,5.8075},
{13,8.55,0.074},
{14,1.7074,3.9685},
{15,5.7943,3.4692},
{16,8.3931,8.5849},
{17,4.7333,5.3947},
{18,1.069,3.2497},
{19,9.3669,7.7855},
{20,2.3341,8.5196},
{21,0.5004,2.2394},
{22,6.5147,1.8744},
{23,5.1284,2.0043},
{24,3.555,1.3365},
{25,1.9224,8.0774},
{26,6.6664,9.9721},
{27,2.5007,5.2815},
{28,8.7526,6.6125},
{29,0.0898,3.9292},
{30,1.2544,9.5753},
{31,1.5462,8.4605},
{32,3.723,4.1098},
{33,9.8581,8.0831},
{34,4.0208,2.7462},
{35,4.6232,1.3271},
{36,1.5694,2.168},
{37,1.8174,4.779},
{38,9.2858,3.3175},
{39,7.1321,2.2322},
{40,2.9921,3.2818},
{41,7.0561,9.2796},
{42,1.4107,2.6271},
{43,5.1149,8.3582},
{44,6.8967,7.6558},
{45,0.0982,8.2855},
{46,1.065,4.9598},
{47,0.3701,3.7443},
{48,3.1341,8.8177},
{49,3.1314,7.3348},
{50,9.6476,3.3575},
{51,6.1636,5.3563},
{52,8.9044,7.8936},
{53,9.7695,9.6457},
{54,2.3383,2.229},
{55,5.9883,9.3733},
{56,9.3741,4.4313},
{57,8.4276,2.9337},
{58,8.2181,1.0951},
{59,3.2603,6.9417},
{60,3.0235,0.8046},
{61,1.0006,9.4768},
{62,8.5635,9.2097},
{63,5.903,7.6075},
{64,4.3534,7.5549},
{65,8.2062,3.453},
{66,9.0327,8.9012},
{67,8.077,8.6283},
{68,4.7475,5.5387},
{69,2.4441,7.106},
{70,8.1469,1.1593},
{71,5.0788,5.315},
{72,5.1421,9.8605},
{73,7.7034,2.019},
{74,3.5393,2.2992},
{75,2.804,1.3503},
{76,4.7581,2.2302},
{77,2.6552,1.7776},
{78,7.4403,5.5851},
{79,2.6909,9.7426},
{80,7.2932,5.4318},
{81,5.7443,4.3915},
{82,3.3988,9.8385},
{83,2.5105,3.6425},
{84,4.3386,4.9175},
{85,6.5916,5.7468},
{86,2.7913,7.4308},
{87,9.3152,5.4451},
{88,9.3501,3.9941},
{89,1.7224,4.6733},
{90,6.6617,1.6269},
{91,3.0622,1.9185},
{92,0.6733,2.4744},
{93,1.355,1.0267},
{94,3.75,9.499},
{95,7.2441,0.5949},
{96,3.3434,4.9163},
{97,8.7538,5.3958},
{98,7.4316,2.6315},
{99,3.6239,5.3696},
{100,3.2393,3.0533}
],lMatrix);

dCentroidMatrix:=DATASET([
{1,1,1},
{2,2,2},
{3,3,3},
{4,4,4}
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

d02Prep; // *********************************result 1

//iteration 1:
c1 := 1;

// set the current centroids to the results of the most recent iteration
dCentroid1 := PROJECT(d02Prep,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c1];SELF:=LEFT;));
dCentroid1; //*********************************result 2

//running Kmeans on d01
KmeansD01 := ML.Cluster.KMeans(d01,dCentroid1,1,nConverge);

//*********************helper functions for calculate C'*****************************************
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
      dDistance:=ML.Cluster.Distances(dResult(n01,d),PROJECT(dResult(n02,d),TRANSFORM(Types.NumericField,SELF.id:=LEFT.id+iMax01;SELF:=LEFT;)),fDist);
RETURN PROJECT(dDistance(x=y-iMax01),TRANSFORM({Types.NumericField AND NOT [number];},SELF.id:=LEFT.x;SELF:=LEFT;));
END;

//*****************************************************************************************************

firstIte := KmeansD01.AllResults();

//set the result of Standard Kmeans to the result of first iteration
firstResult := PROJECT(firstIte,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[2];SELF:=LEFT;));
firstResult; // result 3

// Now join to the existing centroid dataset to add the new values to
// the end of the values set.
dAdded1:=JOIN(d02Prep,firstResult,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=RIGHT;),RIGHT OUTER);
dAdded1;// **********************************************************************************************result4

//calculate the deltaC
deltaC1 := dDistanceDelta(c1,c1-1,dAdded1);
deltaC1;// **********************************************************************************************result5

// Check the distance delta for the last two iterations.  If the highest
// value is below the convergence threshold, then set bConverged to TRUE
bConverged1:= IF(MAX(deltaC1,value)<=nConverge
,TRUE,FALSE);
bConverged1; //******************************************************************************************result 6


// If the centroids have converged, simply pass the input dataset through
// to the next iteration.  Otherwise perform an iteration.
result1 :=IF(bConverged1,d02Prep,dAdded1);
result1; //******************************************************************************************result 7

//Since result is FALSE, go to Iteration 2:
c2 := 2;

// set the current centroids to the results of the most recent iteration
dCentroid2 := IF(c2 = 2,firstResult, PROJECT(dAdded1,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c1];SELF:=LEFT;)));
dCentroid2; // **********************************************************************************************result 8


//step 1: get Gt 
//running Kmeans on dCentroid
//inputs

K := COUNT(DEDUP(d02,id));
K; // **********************************************************************************************result 9
t:= IF(K/10<1, 1, K/10);
t; // **********************************************************************************************result 10


//temporary solution to get dt is that dt = the first t cendroids of dCentroid 
temp := t * 2;
tempDt := dCentroid2[1..temp];
//transform ids of the centroids of dt starting from 1....
Types.NumericField transDt(Types.NumericField L, INTEGER c) := TRANSFORM
SELF.id := ROUNDUP(c/2);
SELF := L;
END;
dt := project(tempDt, transDt(LEFT, COUNTER));
dt; // **********************************************************************************************result 11

nt := n;
tConverge := nConverge;

//run Kmeans on dCentroid
KmeansDt :=ML.Cluster.KMeans(dCentroid2,dt,nt,tConverge);

Gt := TABLE(KmeansDt.Allegiances(), {x,y});//the assignment of each centroid to a group
Gt;// **********************************************************************************************result 12

//Step 2: get ub and lb for each data point
//get ub
dClosest := KmeansD01.Allegiances();
dClosest;
ub1 := dClosest; // **********************************************************************************************result13

//get lb
//all the distances from each data point to each centroid
dDistances:= ML.Cluster.Distances(d01,dCentroid2);
joinlb := JOIN(dClosest,dDistances, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y,RIGHT ONLY);

//Join Gt and dDistances to get the lb of each group
lBound := RECORD
TYPEOF(Types.NumericField.id) xid;
TYPEOF(Types.NumericField.id) cid;
TYPEOF(Types.NumericField.id) gid;
TYPEOF(Types.NumericField.value) lb;
END;


lBound transLbound(joinlb L, Gt R) := TRANSFORM
SELF.xid := L.x;
SELF.cid := L.y;
SELF.gid := R.y;
SELF.lb := L.value;
END;

//join result 8 and result 5 to get the dataset for lb
jDistG := JOIN(joinlb, Gt, LEFT.y = RIGHT.x, transLbound(LEFT,RIGHT));
// jDistG;

//lb of each group

// jDistGv := DEDUP(SORT(GROUP(jDistG,xid,gid),lb),gid);
// jDistGv ;//***************result 11 == lb of each group

re := SORT(jDistG, xid,gid,lb);
// re;

lbs1 := DEDUP(re,xid,gid );
lbs1 ;// **********************************************************************************************result 14


//*********************now Gt (deltaG will be initailized and updated in iteration ), ub, lb are initialized ************************

//*********************************Iteration Part******************************************************	
//update c
//step 1: get C' and deltaC --> c' according to equation 1 --> c' = (c* V + Vin - Vout)/Vnew
//Define the RECORD Structure of V
lV := RECORD
TYPEOF(Types.NumericField.id) id;
TYPEOF(Types.NumericField.id) xid;
END;
			 		
//initialize V, Vin and Vout
//V
V1 := TABLE(dClosest, {x; y;});
V1; // **********************************************************************************************result 15
//Vin
// lV transV(ub1 L) := TRANSFORM
// SELF.id := L.y;
// SELF.xid := 0;
// END;
Vin2 := TABLE(dCentroid2, {id; TYPEOF(Types.NumericField.id) xid := 0; }, id);
Vin2; // **********************************************************************************************result 16
		
//Vout
// lV transVout(ub1 L) := TRANSFORM
// SELF.id := L.y;
// SELF.xid := 0;
// END;		
Vout2 := TABLE(dCentroid2, {id; TYPEOF(Types.NumericField.id) xid := 0; }, id);		
Vout2; // **********************************************************************************************result 17
	
//Get Vcount
//calculate the number of documents allied to each centroid
// dClusterCounts:=TABLE(ub1,{y;UNSIGNED c:=COUNT(GROUP);},y,FEW);
// dClusterCounts; //************************result 14
lVcount := RECORD
TYPEOF(Types.NumericField.id) id := dClosest.y;
TYPEOF(Types.NumericField.number) c := COUNT(group);
TYPEOF(Types.NumericField.number) inNum := 0;
TYPEOF(Types.NumericField.number) outNum := 0;
END;

// lVcount transVcount(dClosest L) := TRANSFORM
// SELF.id := L.y;
// SELF.c := COUNT(group);
// SELF.inNum :=  0;
// SELF.outNum := 0;
// END; 
// Vcount:=TABLE(dClosest,{y;UNSIGNED c:=COUNT(group);},y,FEW);
Vcount1 := TABLE(dClosest, lVcount, y);
Vcount1; // **********************************************************************************************result 18

//Groupfilter
//get deltaC1g which is deltaC1 grouped by Gt
//RECORD structure of lGroup
lGroup := RECORD
TYPEOF(Types.NumericField.id) id;
TYPEOF(Types.NumericField.id) cid;
TYPEOF(Types.NumericField.value) value;
END;
//group deltaC1 by Gt: first JOIN(deltaC with Gt) then group by Gt
deltaC1g :=JOIN(deltaC1, Gt, LEFT.id = RIGHT.x, TRANSFORM(lGroup,SELF.id := RIGHT.y; SELF.cid := LEFT.id; SELF := LEFT));
deltaC1g; // **********************************************************************************************result 19

//get deltaG
//RECORD structure of deltaG
ldeltaG := RECORD
deltaC1g.id;
TYPEOF(Types.NumericField.value) value := MAX(GROUP, deltaC1g.value);
END;

deltaG1 := TABLE(deltaC1g, ldeltaG, id );
deltaG1; // **********************************************************************************************result 20


//**************************Now we got deltaC and deltaG *****************************
//apply Lemma1 to each data point to see if they have to change centroid
lGroupFilter := RECORD
TYPEOF(Types.NumericField.id) id;
TYPEOF(Types.NumericField.id) gid;
TYPEOF(Types.NumericField.id) cid;
TYPEOF(Types.NumericField.value) ubUpdate;
TYPEOF(Types.NumericField.value) lbUpdate;
BOOLEAN pass;
END;

//ubUpdate;
lUbUpdate := RECORD
TYPEOF(Types.NumericField.id) id;
TYPEOF(Types.NumericField.id) cid;
TYPEOF(Types.NumericField.value) ubValue;
TYPEOF(Types.NumericField.value) ubUpdate;
END;
ub2 := JOIN(ub1,deltaC1, LEFT.y = RIGHT.id, TRANSFORM(lUbUpdate, SELF.id := LEFT.x; SELF.cid := LEFT.y ; SELF.ubValue := LEFT.value; SELF.ubUpdate:= LEFT.value+ RIGHT.value) );
//lbUpdate
// lGroupFilter transLemma1(Types.NumericField d, lBound lbs, deltaC dc, deltaG dg) := TRANSFORM
// SELF.id := d.id ;
// SELF.left := 

// END;
ub2;// **********************************************************************************************result 21

//lbUpdate;
lLbUpdate := RECORD
TYPEOF(Types.NumericField.id) xid;
TYPEOF(Types.NumericField.id) gid;
TYPEOF(Types.NumericField.value) lb;
TYPEOF(Types.NumericField.value) lbUpdate;
END;
lbs2 := JOIN(lbs1,deltaG1, LEFT.gid = RIGHT.id, TRANSFORM(lLbUpdate, SELF.lbUpdate := LEFT.lb - RIGHT.value; SELF := LEFT));
lbs2;// **********************************************************************************************result 22
lGroupFilter transLemma1(ub2 L, lbs2 R) := TRANSFORM
SELF.id := L.id;
SELF.gid := R.gid;
SELF.cid := L.cid;
SELF.ubUpdate :=  L.ubUpdate;
SELF.lbUpdate := R.lbUpdate;
SELF.pass := IF(L.ubUpdate < R.lbUpdate, FALSE, 
																	IF(L.ubValue < R.lbUpdate, FALSE,TRUE));
END;

groupFilter2 := JOIN(ub2, lbs2, LEFT.id = RIGHT.xid, transLemma1(LEFT, RIGHT));
groupFilter2; // **********************************************************************************************result 23
// groupFilter;
pGroupFilter2 := groupFilter2(pass = TRUE);
pGroupFilter2; // **********************************************************************************************result 24
countGF2 := COUNT(pGroupFilter2);
countGF2; // **********************************************************************************************result 25

converge2 := IF(countGF2 = 0, TRUE, FALSE);
converge2; // **********************************************************************************************result 26

lLocalFilter := RECORD
TYPEOF(Types.NumericField.id) id;
TYPEOF(Types.NumericField.id) gid;
TYPEOF(Types.NumericField.value) cid;
TYPEOF(Types.NumericField.value) lb ;
TYPEOF(Types.NumericField.value) deltaC;
TYPEOF(Types.NumericField.value)  p ;
BOOLEAN pass;
END;

lLocalFilter transLocalFilter(pGroupFilter2 L, Gt R) := TRANSFORM
SELF.cid := R.x;
SELF.lb := 0;
SELF.deltaC := 0;
SELF := L;
SELF.p := lbs1(xid = L.id).lb;
// SELF.pass := TRUE;
END;

localFilterC2 := JOIN(pGroupFilter2, Gt, LEFT.cid <> RIGHT.x AND LEFT.gid = RIGHT.y, transLocalFilter(LEFT,RIGHT));
localFilterC2; // **********************************************************************************************result 27
localFilterDeltaC2 := SORT(JOIN(localFilterC2, deltaC1, LEFT.cid = RIGHT.id, TRANSFORM(lLocalFilter, SELF.deltaC := RIGHT.value; SELF := LEFT;) ),id);
localFilterDeltaClbs2 := JOIN(localFilterDeltaC2, lbs1, LEFT.id = RIGHT.xid AND LEFT.gid = RIGHT.gid, TRANSFORM(lLocalFilter, SELF.lb := RIGHT.lb; SELF := LEFT;));
localFilterDeltaClbs2; // **********************************************************************************************result 28

lLocalFilter transLemma2(localFilterDeltaClbs2 L) := TRANSFORM
SELF.pass := IF((L.lb - L.deltaC ) > L.p, FALSE,TRUE);
SELF := L;
END;
localFilter2 := PROJECT(localFilterDeltaClbs2, transLemma2(LEFT));
rest2 := localFilter2(pass = TRUE);
rest2; // **********************************************************************************************result 29
// COUNT(rest2);


// lresult := RECORD
// rest.id;
// rest.cid;
// ML.Cluster.Distances(d01(id = rest.id), dCentroid1(id = rest.cid)).value;
// END;


TYPEOF(Types.NumericField.value) distancePair(TYPEOF(Types.NumericField.id) L, TYPEOF(Types.NumericField.id) R) := FUNCTION
distanceValue := ML.Cluster.Distances(d01(id = L), dCentroid2(id = R));

RETURN distanceValue[1].value;
END;

resultFinal2 := PROJECT(localFilterC2, TRANSFORM(Types.NumericField, SELF.id := LEFT.id; SELF.number := LEFT.cid; SELF.value :=distancePair(LEFT.id, LEFT.cid); ));
resultFinal2; // **********************************************************************************************result30

// a := ML.Cluster.Distances(d01(id = 1), dCentroid1(id = 101));
// a;
// b := a[1].value;
// b;
