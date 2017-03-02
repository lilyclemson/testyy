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
{1,2,2},
{2,4,2},
{3,5,2},
{4,8,2}
],lMatrix);

dCentroidMatrix:=DATASET([
{1,0,2},
{2,9,2}
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
/* // lV transV(ub1 L) := TRANSFORM
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
*/
	
//Get V
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
// ub2;
// lbs2;

groupFilter2 := JOIN(ub2, lbs2, LEFT.id = RIGHT.xid, transLemma1(LEFT, RIGHT));
groupFilter2; // **********************************************************************************************result 23
// groupFilter;
   pGroupFilter2 := groupFilter2(pass = TRUE);
   COUNT(pGroupFilter2); // **********************************************************************************************result 24
   
//Local filter
//Join deltaC1 and lbs2 to get the right part of Lemma2  


    localFilterC2 := JOIN(pGroupFilter2, Gt, LEFT.gid = RIGHT.y AND LEFT.cid <> RIGHT.x  );//need to sort Gt first if t != 1?
    localFilterC2;
   	 
   	localFilterTem1 := TABLE(localFilterC2,{id; cid; x; BOOLEAN passL := TRUE;});
   	localFilterTem1;	 
		
   	localFilterTem2 := SORT(JOIN(localFilterTem1, deltaC1, LEFT.x = RIGHT.id), id);		 
   	
		localFilterTem3 := JOIN(localFilterTem2, lbs1, LEFT.id = RIGHT.xid);
		
		localFilterTemX := TABLE(localFilterTem3, {id; cid; x; TYPEOF(Types.NumericField.value) value1 := value; lb; });
		localFilterTemX;
		
		localFilterTem4 := JOIN(localFilterTemX, ub1, LEFT.id = RIGHT.x AND LEFT.cid = right.y  );
   	localFilterTem4;
//apply lemma2		
		localFilterTem5 := TABLE(localFilterTem4, {id; cid; x; value1; lb; value;TYPEOF(Types.NumericField.value) diff := lb - value1; BOOLEAN passl := IF((lb - value1) > value, FALSE, TRUE);});
		localFilterTem5;
//result of local filter		
		localFilterTem6 := localFilterTem5(passl = TRUE);
		localFilterTem6;
		
		
//calculate all the distances from each passed centroids to each data points

  
   TYPEOF(Types.NumericField.value) distancePair(TYPEOF(Types.NumericField.id) L, TYPEOF(Types.NumericField.id) R) := FUNCTION
   distanceValue := ML.Cluster.Distances(d01(id = L), dCentroid2(id = R));
   RETURN distanceValue[1].value;
   END;   
   
   resultFinal2 := PROJECT(localFilterTem6, TRANSFORM(RECORDOF(ub1), SELF.x := LEFT.id; SELF.y := LEFT.x; SELF.value :=distancePair(LEFT.id, LEFT.x); ));
   resultFinal2; // **********************************************************************************************result30
   
	 
   re1 := SORT(resultFinal2, x, value);
   re1;
   
   re2 := DEDUP(re1,x);
   re2 ;// **********************************************************************************************result 14
	 
	 
	 
   re3 := TABLE(re2, { TYPEOF(x) nx := x; TYPEOF(y) ny := y; TYPEOF(value) nvalue := value;});
	 re3;
	 
   result3 := JOIN(ub1, re3, LEFT.x = RIGHT.nx); 
   result3;

   result4 := TABLE(result3, {x; y; value; nx; ny; nvalue; BOOLEAN change :=IF(value > nvalue, TRUE, FALSE);});
	 result4;
	 
	 
	 // KmeansD01.Allegiances();
	 