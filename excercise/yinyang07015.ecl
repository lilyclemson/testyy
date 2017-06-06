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
// {3,5,2},
// {4,8,2}
// ],lMatrix);

// dCentroidMatrix:=DATASET([
// {1,0,2},
// {2,9,2}
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

//Since result is FALSE

// set the current centroids to the results of the most recent iteration
dCentroid2 := firstResult;
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
getUb := ML.Cluster.Distances(d01,dCentroid1);
dClosest := ML.Cluster.Closest(getUb);
dClosest;
ub1 := dClosest; // **********************************************************************************************result13

//get lb
//all the distances from each data point to each centroid
dDistances:= ML.Cluster.Distances(d01,dCentroid1);
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
//Get V
lVcount := RECORD
TYPEOF(Types.NumericField.id) id := dClosest.y;
TYPEOF(Types.NumericField.number) c := COUNT(group);
TYPEOF(Types.NumericField.number) inNum := 0;
TYPEOF(Types.NumericField.number) outNum := 0;
END;

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
COUNT(pGroupFilter2); // **********************************************************************************************result 24
pGroupFilter2;
pGroupFilter3 := groupFilter2(pass = FALSE);

//localFilter
//function that calculate the distance between two data points.
TYPEOF(Types.NumericField.value) distancePair(TYPEOF(Types.NumericField.id) L, TYPEOF(Types.NumericField.id) R) := FUNCTION
distanceValue := ML.Cluster.Distances(d01(id = L), dCentroid2(id = R));
RETURN distanceValue[1].value;
END;   
   
 
changedSet := JOIN(d01,pGroupFilter3, LEFT.id = RIGHT.id,TRANSFORM(LEFT), LEFT ONLY);
changedSet;
getUbP := ML.Cluster.Distances(changedSet,dCentroid2);
dClosest1 := ML.Cluster.Closest(getUbP);
dClosest1;

lLocalFilter := RECORD
TYPEOF(Types.NumericField.id) Cvin;
TYPEOF(Types.NumericField.id) Cvout;
TYPEOF(Types.NumericField.id) xid;
TYPEOF(Types.NumericField.value) nUb;
END;
lLocalFilter transLemma2(dClosest1 l, ub1 r) := TRANSFORM
SELF.Cvin := l.y;
SELF.Cvout := r.y;
SELF.xid := l.x;
SELF.nUb := l.value;
END;
//changeC is the dataset that change their best centroid.		
changeC := JOIN(dClosest1, ub1, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, transLemma2(LEFT,RIGHT));

//changeC is empty then it's converged. Or update the ub and lb.
convergedend := IF(COUNT(changeC) =0, TRUE, FALSE);
convergedend;
// dAdded1;

//the IDs of data points who changed their best centroid
changeId := SET(changeC, xid);
changeId;

//update the Ub of data points who do not change their best centroid		
unchangedUb := PROJECT(ub1(x NOT IN changeID), TRANSFORM(RECORDOF(ub1), SELF.x := LEFT.x; SELF.y := LEFT.y; SELF.value :=distancePair(LEFT.x, LEFT.y); ));
unchangedUb;
//update the Ub of of data points who change their best centroid 		
changeUb := TABLE(changeC, {TYPEOF(TYPEOF(ub1.x)) x := xid; TYPEOF(TYPEOF(ub1.y)) y := Cvin;TYPEOF(TYPEOF(ub1.value)) value :=nUb;});
//new Ub
ub3 := changeUb + unchangedUb;
ub3;

//update lbs		
joinlb1 := JOIN(dClosest1,getUbP, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY);
//new lbs for data points who change their best centroid
changelbs3 := DEDUP(joinlb1,x );
changelbs3 ;// **********************************************************************************************result 14
idlbs := SET(changelbs3, x);
unchangedLbs3Temp := TABLE(lbs2, {TYPEOF(ub1.x) xid := xid;TYPEOF(TYPEOF(ub1.x)) cid := gid; TYPEOF(TYPEOF(ub1.value)) lbupdate :=lbupdate;});	
//new lbs for data points who do not change their best centroid
unchangedLbs3 := PROJECT(unchangedLbs3Temp(xid NOT IN idlbs), TRANSFORM(RECORDOF(unchangedLbs3Temp),  SELF := LEFT; ));
unchangedLbs3;
//new lbs
lbs3 := changelbs3 + unchangedLbs3;
lbs3;

//iteration 2
//update c
//step 1: get C' and deltaC --> c' according to equation 1 --> c' = (c* V + Vin - Vout)/Vnew	 		

changeC;
Vcount1;
//calculate the Vin
dClusterCountsVin:=TABLE(SORT(changeC, cvin),{cvin;UNSIGNED c:=COUNT(GROUP);},cvin,FEW);
dClusterCountsVin;

dClusteredVin := JOIN(d01, SORT(changeC, cvin), LEFT.id = RIGHT.xid,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.cvin;SELF:=LEFT;));
dClusteredVin;
dRolledVin:=ROLLUP(SORT(dClusteredVin,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
dRolledVin;

//calculate the Vout
dClusterCountsVout:=TABLE(SORT(changeC, cvout),{cvout;UNSIGNED c:=COUNT(GROUP);},cvout,FEW);
dClusterCountsVout;

dClusteredVout := JOIN(d01,SORT(changeC,cvout), LEFT.id = RIGHT.xid,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.cvout;SELF:=LEFT;));
dClusteredVout;
dRolledVout:=ROLLUP(SORT(dClusteredVout,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
dRolledVout;

//calculate the new Vcount = old Vcount + |Vin| - |Vout| 
Vcount2 :=JOIN(Vcount1,dClusterCountsVin, LEFT.id = RIGHT.cvin, TRANSFORM(RECORDOF(LEFT), SELF.inNum := RIGHT.c;  SELF := LEFT;), LEFT OUTER );
Vcount2;
Vcount3 :=JOIN(Vcount2,dClusterCountsVout, LEFT.id = RIGHT.cvout, TRANSFORM(RECORDOF(LEFT), SELF.outNum := RIGHT.c; SELF := LEFT;), LEFT OUTER);
Vcount3;
Vcount4 := TABLE(Vcount3, {id; TYPEOF(Types.NumericField.number) c := (c + inNum - outNum); inNum; outNum;});
Vcount4;

//let the old |c*V| to multiply old Vcount
sumC :=JOIN(dCentroid2, Vcount1, LEFT.id = RIGHT.id, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value * RIGHT.c; SELF := LEFT), LEFT OUTER);
sumC ;

//Add Vin
plusC := JOIN(sumC, dRolledVin, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value + RIGHT.value; SELF := LEFT), LEFT OUTER);
plusC;

//Minus Vout
minusC:= JOIN(plusC, dRolledVout, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value - RIGHT.value; SELF := LEFT), LEFT OUTER);
minusC;

//get the new C
dAdded2Temp:=JOIN(minusC,Vcount4,LEFT.id=RIGHT.id,TRANSFORM(RECORDOF(LEFT),SELF.value:=(LEFT.value/RIGHT.c);SELF:=LEFT;), LEFT OUTER);
dAdded2Temp;

//add to the dAdded1
dAdded2 := JOIN(dAdded1,dAdded2Temp,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=RIGHT;), LEFT OUTER);
dAdded2;

c2:= 2;
//calculate the deltaC
deltaC2 := dDistanceDelta(c2,c2-1,dAdded2);
deltaC2;// **********************************************************************************************result5

// Check the distance delta for the last two iterations.  If the highest
// value is below the convergence threshold, then set bConverged to TRUE
bConverged2:= IF(MAX(deltaC2,value)<=nConverge,TRUE,FALSE);
bConverged2; //******************************************************************************************result 6


// If the centroids have converged, simply pass the input dataset through
// to the next iteration.  Otherwise perform an iteration.
result2 :=IF(bConverged2,dAdded1,dAdded2);
result2; //******************************************************************************************result 7

//Since result is FALSE

// set the current centroids to the results of the most recent iteration
dCentroid3 := PROJECT(dAdded2,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c2+1];SELF:=LEFT;));
dCentroid3; // **********************************************************************************************result 8

/* //group filter
   groupFilter3 := JOIN(ub3, lbs3, LEFT.x = RIGHT.x, transLemma1(LEFT, RIGHT));
   groupFilter3; // **********************************************************************************************result 23
   // groupFilter;
   pGroupFilter3 := groupFilter3(pass = TRUE);
   COUNT(pGroupFilter3); // **********************************************************************************************result 24
   pGroupFilter3;
   pGroupFilter4 := groupFilter3(pass = FALSE);
   
   //localFilter
   changedSet1 := JOIN(d01,pGroupFilter4, LEFT.id = RIGHT.id,TRANSFORM(LEFT), LEFT ONLY);
   changedSet1;
   getUbP1 := ML.Cluster.Distances(changedSet1,dCentroid3);
   dClosest2 := ML.Cluster.Closest(getUbP1);
   dClosest2;
*/
