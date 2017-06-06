IMPORT * FROM $;
IMPORT Std.Str AS Str;
IMPORT TS;
IMPORT ML.Mat;
IMPORT ML;
IMPORT ML.Cluster.DF as DF;
IMPORT ML.Types AS Types;
IMPORT ML.Utils;

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
nConverge := 2;

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

// d02Prep; 

dCentroid0 := PROJECT(d02Prep,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[1];SELF:=LEFT;));
OUTPUT(dCentroid0,NAMED('Initial_Centroids')); 

//running Kmeans on d01
KmeansD01 := ML.Cluster.KMeans(d01,dCentroid0,1,nConverge);

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

//*********************************************************************************************************************************************************************


//iteration 1:
c1 := 1;

//set the result of Standard Kmeans to the result of first iteration
firstIte := KmeansD01.AllResults();
firstResult := PROJECT(firstIte,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[2];SELF:=LEFT;)); 


// Now join to the existing centroid dataset to add the new values to the end of the values set.
dAdded1:=JOIN(d02Prep,firstResult,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=RIGHT;),RIGHT OUTER);
// OUTPUT(dAdded1, NAMED('iteration1_result'));

//calculate the deltaC
deltaC1 := dDistanceDelta(c1,c1-1,dAdded1);
OUTPUT(deltaC1,NAMED('deltaC1'));

// Check the distance delta for the last two iterations.  If the highest
// value is below the convergence threshold, then set bConverged to TRUE
bConverged1:= IF(MAX(deltaC1,value)<=nConverge,TRUE,FALSE);
OUTPUT(bConverged1, NAMED('iteration1_converge'));

// If the centroids have converged, simply pass the input dataset through
// to the next iteration.  Otherwise perform an iteration.
result1 :=IF(bConverged1,d02Prep,dAdded1);
OUTPUT(result1, NAMED('iteration1_result'));


//Since result is FALSE
// set the current centroids to the results of the most recent iteration
dCentroid1 := firstResult;
OUTPUT(dCentroid1,NAMED('C1')); 

//step 1: get Gt 
//running Kmeans on dCentroid
//inputs

K := COUNT(DEDUP(d02,id));
// K; 
t:= IF(K/10<1, 1, K/10);
// t; 

//temporary solution to get dt is that dt = the first t cendroids of dCentroid 
temp := t * 2;
tempDt := dCentroid1[1..temp];
//transform ids of the centroids of dt starting from 1....
Types.NumericField transDt(Types.NumericField L, INTEGER c) := TRANSFORM
SELF.id := ROUNDUP(c/2);
SELF := L;
END;
dt := project(tempDt, transDt(LEFT, COUNTER));
// dt;

nt := n;
tConverge := nConverge;

//run Kmeans on dCentroid
KmeansDt :=ML.Cluster.KMeans(dCentroid1,dt,nt,tConverge);

Gt := TABLE(KmeansDt.Allegiances(), {x,y});//the assignment of each centroid to a group
// Gt;
//Step 2: get ub and lb for each data point
//get ub
dDistances1 := ML.Cluster.Distances(d01,dCentroid0);
dClosest1 := ML.Cluster.Closest(dDistances1);
// dClosest1;
ub1 := dClosest1; 
OUTPUT(ub1, NAMED('ub1'));

//get lb
//all the distances from each data point to each centroid
// joinlb1 := JOIN(dClosest1,dDistances1, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y,RIGHT ONLY);
lbs1 := JOIN(dClosest1,dDistances1, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y,RIGHT ONLY);

//Join Gt and dDistances to get the lb of each group
lBound := RECORD
TYPEOF(Types.NumericField.id) xid;
TYPEOF(Types.NumericField.id) cid;
TYPEOF(Types.NumericField.id) gid;
TYPEOF(Types.NumericField.value) lb;
END;


lBound transLbound(joinlb1 L, Gt R) := TRANSFORM
SELF.xid := L.x;
SELF.cid := L.y;
SELF.gid := R.y;
SELF.lb := L.value;
END;

//join result 8 and result 5 to get the dataset for lb
lbs1G := JOIN(lbs1, Gt, LEFT.y = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.y := RIGHT.y; SELF := LEFT;));

//lb of each group
re := SORT(jDistG, xid,gid,lb);
// re;
lbs1 := DEDUP(re,xid,gid );
OUTPUT(lbs1,NAMED('lbs1')) ;

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
TYPEOF(Types.NumericField.id) id := dClosest1.y;
TYPEOF(Types.NumericField.number) c := COUNT(group);
TYPEOF(Types.NumericField.number) inNum := 0;
TYPEOF(Types.NumericField.number) outNum := 0;
END;

Vcount1 := TABLE(dClosest1, lVcount, y);
OUTPUT(Vcount1, NAMED('Vcount1')); 

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
// deltaC1g; 

//get deltaG
//RECORD structure of deltaG
ldeltaG := RECORD
deltaC1g.id;
TYPEOF(Types.NumericField.value) value := MAX(GROUP, deltaC1g.value);
END;

deltaG1 := TABLE(deltaC1g, ldeltaG, id );
OUTPUT(deltaG1,NAMED('deltaG1')); // **********************************************************************************************result 20


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
ubUpdate1 := JOIN(ub1,deltaC1, LEFT.y = RIGHT.id, TRANSFORM(lUbUpdate, SELF.id := LEFT.x; SELF.cid := LEFT.y ; SELF.ubValue := LEFT.value; SELF.ubUpdate:= LEFT.value+ RIGHT.value) );
OUTPUT(ubUpdate1, NAMED('ubUpdate1'));

//lbUpdate;
lLbUpdate := RECORD
TYPEOF(Types.NumericField.id) xid;
TYPEOF(Types.NumericField.id) gid;
TYPEOF(Types.NumericField.value) lb;
TYPEOF(Types.NumericField.value) lbUpdate;
END;
lbsUpdate1 := JOIN(lbs1,deltaG1, LEFT.gid = RIGHT.id, TRANSFORM(lLbUpdate, SELF.lbUpdate := LEFT.lb - RIGHT.value; SELF := LEFT));
OUTPUT(lbsUpdate1,NAMED('lbsUpdate1'));

lGroupFilter transLemma1(ubUpdate1 L, lbsUpdate1 R) := TRANSFORM
SELF.id := L.id;
SELF.gid := R.gid;
SELF.cid := L.cid;
SELF.ubUpdate :=  L.ubUpdate;
SELF.lbUpdate := R.lbUpdate;
SELF.pass := IF(L.ubUpdate < R.lbUpdate, FALSE, 
																	IF(L.ubValue < R.lbUpdate, FALSE,TRUE));
END;

groupFilter1 := JOIN(ubUpdate1, lbsUpdate1, LEFT.id = RIGHT.xid, transLemma1(LEFT, RIGHT));
// OUTPUT(groupFilter1,NAMED('groupFilter1')); 


pGroupFilter1 := groupFilter1(pass = TRUE);
groupFilter1_Converge := IF(COUNT(pGroupFilter1)=0, TRUE, FALSE);
OUTPUT(groupFilter1_Converge, NAMED('groupFilter1_converge'));
OUTPUT(pGroupFilter1,NAMED('groupFilter1_result')); 

unpGroupFilter1 := groupFilter1(pass = FALSE);

//localFilter
changedSetTemp := JOIN(d01,unpGroupFilter1, LEFT.id = RIGHT.id,TRANSFORM(LEFT), LEFT ONLY);
// changedSetTemp;
getUbP := ML.Cluster.Distances(changedSetTemp,dCentroid1);
dClosest2 := ML.Cluster.Closest(getUbP);
// OUTPUT(dClosest2,NAMED('distacesOfDataPointsPassLfilter')); 

lLocalFilter := RECORD
TYPEOF(Types.NumericField.id) Cvin;
TYPEOF(Types.NumericField.id) Cvout;
TYPEOF(Types.NumericField.id) xid;
TYPEOF(Types.NumericField.value) nUb;
END;
lLocalFilter transLemma2(dClosest2 l, ub1 r) := TRANSFORM
SELF.Cvin := l.y;
SELF.Cvout := r.y;
SELF.xid := l.x;
SELF.nUb := l.value;
END;
//the dataset that change their best centroid = result of local fitler		
localFilter1 := JOIN(dClosest2, dClosest1, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, transLemma2(LEFT,RIGHT));
 

//localFilter1 is empty then it's converged. Or update the ub and lb.
localFilter1_converge := IF(COUNT(localFilter1) =0, TRUE, FALSE);
OUTPUT(localFilter1_converge,NAMED('localFilter1_converge'));
OUTPUT(localFilter1,NAMED('localFilter1_result'));
// dAdded1;

//the IDs of data points who changed their best centroid
changeId := SET(localFilter1, xid);
// OUTPUT(changeId,NAMED('changeID'));

//update the Ub of data points who do not change their best centroid		
unChangedUbSet := ub1(x NOT IN changeID);
unChangedUbD01 := JOIN(unChangedUbSet, d01, LEFT.x = RIGHT.id, TRANSFORM({LEFT.x; LEFT.y; RIGHT.number; TYPEOF(RIGHT.value) xValue;},SELF.x := LEFT.x; SELF.y := LEFT.y; SELF.number := RIGHT.number; SELF.xValue := RIGHT.value;));
// unchangedUb := PROJECT(ub1(x NOT IN changeID), TRANSFORM(RECORDOF(ub1), SELF.x := LEFT.x; SELF.y := LEFT.y; SELF.value :=ML.Cluster.Distances(d01(id = LEFT.x), dCentroid1(id = LEFT.y))[1].value; ));
unChangedUbD01CTemp := JOIN(unchangedUbD01, dCentroid1, LEFT.y = RIGHT.id AND LEFT.number = RIGHT.number);
unChangedUbD01C := SORT(unChangedUbD01CTemp, x, number);
unChangedUbTemp := PROJECT(unChangedUbD01C, TRANSFORM(literations, SELF.id := LEFT.x; SELF.number := LEFT.number; SELF.values :=[LEFT.xvalue] + [LEFT.value]; ));
unChangedUbTemp1 := dDistanceDelta(0,1,unChangedUbTemp);
unChangedUb := JOIN(unChangedUbTemp1, ub1, LEFT.id = RIGHT.x, TRANSFORM(RECORDOF(ub1), SELF.x := RIGHT.x; SELF.y := RIGHT.y; SELF.value := LEFT.value; ));
// OUTPUT(unChangedUbSet, NAMED('unChangedUbSet'));
// OUTPUT(unChangedUbD01, NAMED('unChangedUbD01'));
// OUTPUT(unChangedUbD01C, NAMED('unChangedUbD01C'));
// OUTPUT(unChangedUbTemp, NAMED('unChangedUbTemp'));
// OUTPUT(unChangedUbTemp1, NAMED('unChangedUbTemp1'));
OUTPUT(unChangedUb, NAMED('unChangedUb'));

//update the Ub of of data points who change their best centroid 		
changedUb := TABLE(localFilter1, {TYPEOF(TYPEOF(ub1.x)) x := xid; TYPEOF(TYPEOF(ub1.y)) y := Cvin;TYPEOF(TYPEOF(ub1.value)) value :=nUb;});
OUTPUT(changedUb, NAMED('changedUb'));
//new Ub
ub2 := changedUb + unchangedUb;
OUTPUT(ub2, NAMED('ub2'));

//update lbs		
joinlb2 := JOIN(dClosest2,getUbP, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY);
//new lbs for data points who change their best centroid
changelbs2 := DEDUP(joinlb2,x );
// OUTPUT(changelbs2,NAMED('changelbs2')) ;
idlbs := SET(changelbs2, x);
unchangedLbs2Temp := TABLE(lbsUpdate1, {TYPEOF(ub1.x) xid := xid;TYPEOF(TYPEOF(ub1.x)) cid := gid; TYPEOF(TYPEOF(ub1.value)) lbupdate :=lbupdate;});	
//new lbs for data points who do not change their best centroid
unchangedLbs2 := PROJECT(unchangedLbs2Temp(xid NOT IN idlbs), TRANSFORM(RECORDOF(unchangedLbs2Temp),  SELF := LEFT; ));
// OUTPUT(unchangedLbs2,NAMED('unchangedLbs2'));
//new lbs
lbs2 := changelbs2 + unchangedLbs2;
OUTPUT(lbs2,NAMED('lbs2'));

//iteration 2

c2:= 2;

//update c
//step 1: get C' and deltaC --> c' according to equation 1 --> c' = (c* V + Vin - Vout)/Vnew	 		

// localFilter1;
// Vcount1;
//calculate the Vin
dClusterCountsVin:=TABLE(SORT(localFilter1, cvin),{cvin;UNSIGNED c:=COUNT(GROUP);},cvin,FEW);
// dClusterCountsVin;

dClusteredVin := JOIN(d01, SORT(localFilter1, cvin), LEFT.id = RIGHT.xid,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.cvin;SELF:=LEFT;));
// dClusteredVin;
dRolledVin:=ROLLUP(SORT(dClusteredVin,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
// dRolledVin;

//calculate the Vout
dClusterCountsVout:=TABLE(SORT(localFilter1, cvout),{cvout;UNSIGNED c:=COUNT(GROUP);},cvout,FEW);
// dClusterCountsVout;

dClusteredVout := JOIN(d01,SORT(localFilter1,cvout), LEFT.id = RIGHT.xid,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.cvout;SELF:=LEFT;));
// dClusteredVout;
dRolledVout:=ROLLUP(SORT(dClusteredVout,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
// dRolledVout;

//calculate the new Vcount = old Vcount + |Vin| - |Vout| 
Vcount2Vin :=JOIN(Vcount1,dClusterCountsVin, LEFT.id = RIGHT.cvin, TRANSFORM(RECORDOF(LEFT), SELF.inNum := RIGHT.c;  SELF := LEFT;), LEFT OUTER );
// Vcount2;
Vcount2VinVout :=JOIN(Vcount2Vin,dClusterCountsVout, LEFT.id = RIGHT.cvout, TRANSFORM(RECORDOF(LEFT), SELF.outNum := RIGHT.c; SELF := LEFT;), LEFT OUTER);
// Vcount3;
Vcount2 := TABLE(Vcount2VinVout, {id; TYPEOF(Types.NumericField.number) c := (c + inNum - outNum); inNum; outNum;});
OUTPUT(Vcount2, NAMED('Vcount2'));

//let the old |c*V| to multiply old Vcount
CV1 :=JOIN(dCentroid1, Vcount1, LEFT.id = RIGHT.id, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value * RIGHT.c; SELF := LEFT), LEFT OUTER);
// CV1 ;

//Add Vin
CV1Vin := JOIN(CV1, dRolledVin, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value + RIGHT.value; SELF := LEFT), LEFT OUTER);
// CV1Vin;

//Minus Vout
CV1VinVout:= JOIN(CV1Vin, dRolledVout, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value - RIGHT.value; SELF := LEFT), LEFT OUTER);
// OUTPUT(CV1VinVout , NAMED('CV1VinVout'));

//get the new C
dAdded2Temp:=JOIN(CV1VinVout,Vcount2,LEFT.id=RIGHT.id,TRANSFORM(RECORDOF(LEFT),SELF.value:=(LEFT.value/RIGHT.c);SELF:=LEFT;), LEFT OUTER);
OUTPUT(dAdded2Temp, NAMED('C2'));

//add to the dAdded1
dAdded2 := JOIN(dAdded1,dAdded2Temp,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=RIGHT;), LEFT OUTER);
// dAdded2;

//calculate the deltaC
deltaC2 := dDistanceDelta(c2,c2-1,dAdded2);
OUTPUT(deltaC2, NAMED('deltaC2'));// **********************************************************************************************result5

// Check the distance delta for the last two iterations.  If the highest
// value is below the convergence threshold, then set bConverged to TRUE
bConverged2:= IF(MAX(deltaC2,value)<=nConverge,TRUE,FALSE);
OUTPUT(bConverged2, NAMED('iteration2_converge'));

//Since result is FALSE

result2 :=IF(bConverged2,dAdded1,dAdded2);
OUTPUT(result2, NAMED('iteration2_result')); 

// set the current centroids to the results of the most recent iteration
dCentroid2 := PROJECT(dAdded2,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c2+1];SELF:=LEFT;));
dCentroid2; 
/*
//groupfilter2
groupFilter2 := JOIN(ub2, lbs2, LEFT.x = RIGHT.x, transLemma1(LEFT, RIGHT));
OUTPUT(groupFilter2,NAMED('groupFilter2')); 
// groupFilter;
unpGroupFilter2 := groupFilter2(pass = TRUE);
COUNT(unpGroupFilter2); // **********************************************************************************************result 24
unpGroupFilter2;
pGroupFilter3 := groupFilter2(pass = FALSE);
// localFilter
changedSet1 := JOIN(d01,pGroupFilter3, LEFT.id = RIGHT.id,TRANSFORM(LEFT), LEFT ONLY);
changedSet1;
getUbP1 := ML.Cluster.Distances(changedSet1,dCentroid2);
dClosest2 := ML.Cluster.Closest(getUbP1);
dClosest2;
*/