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

// dDocumentMatrix:=DATASET([
// {1,2.4639,7.8579},
// {2,0.5573,9.4681},
// {3,4.6054,8.4723},
// {4,1.24,7.3835},
// {5,7.8253,4.8205},
// {6,3.0965,3.4085},
// {7,8.8631,1.4446},
// {8,5.8085,9.1887},
// {9,1.3813,0.515},
// {10,2.7123,9.2429},
// {11,6.786,4.9368},
// {12,9.0227,5.8075},
// {13,8.55,0.074},
// {14,1.7074,3.9685},
// {15,5.7943,3.4692},
// {16,8.3931,8.5849},
// {17,4.7333,5.3947},
// {18,1.069,3.2497},
// {19,9.3669,7.7855},
// {20,2.3341,8.5196},
// {21,0.5004,2.2394},
// {22,6.5147,1.8744},
// {23,5.1284,2.0043},
// {24,3.555,1.3365},
// {25,1.9224,8.0774},
// {26,6.6664,9.9721},
// {27,2.5007,5.2815},
// {28,8.7526,6.6125},
// {29,0.0898,3.9292},
// {30,1.2544,9.5753},
// {31,1.5462,8.4605},
// {32,3.723,4.1098},
// {33,9.8581,8.0831},
// {34,4.0208,2.7462},
// {35,4.6232,1.3271},
// {36,1.5694,2.168},
// {37,1.8174,4.779},
// {38,9.2858,3.3175},
// {39,7.1321,2.2322},
// {40,2.9921,3.2818},
// {41,7.0561,9.2796},
// {42,1.4107,2.6271},
// {43,5.1149,8.3582},
// {44,6.8967,7.6558},
// {45,0.0982,8.2855},
// {46,1.065,4.9598},
// {47,0.3701,3.7443},
// {48,3.1341,8.8177},
// {49,3.1314,7.3348},
// {50,9.6476,3.3575},
// {51,6.1636,5.3563},
// {52,8.9044,7.8936},
// {53,9.7695,9.6457},
// {54,2.3383,2.229},
// {55,5.9883,9.3733},
// {56,9.3741,4.4313},
// {57,8.4276,2.9337},
// {58,8.2181,1.0951},
// {59,3.2603,6.9417},
// {60,3.0235,0.8046},
// {61,1.0006,9.4768},
// {62,8.5635,9.2097},
// {63,5.903,7.6075},
// {64,4.3534,7.5549},
// {65,8.2062,3.453},
// {66,9.0327,8.9012},
// {67,8.077,8.6283},
// {68,4.7475,5.5387},
// {69,2.4441,7.106},
// {70,8.1469,1.1593},
// {71,5.0788,5.315},
// {72,5.1421,9.8605},
// {73,7.7034,2.019},
// {74,3.5393,2.2992},
// {75,2.804,1.3503},
// {76,4.7581,2.2302},
// {77,2.6552,1.7776},
// {78,7.4403,5.5851},
// {79,2.6909,9.7426},
// {80,7.2932,5.4318},
// {81,5.7443,4.3915},
// {82,3.3988,9.8385},
// {83,2.5105,3.6425},
// {84,4.3386,4.9175},
// {85,6.5916,5.7468},
// {86,2.7913,7.4308},
// {87,9.3152,5.4451},
// {88,9.3501,3.9941},
// {89,1.7224,4.6733},
// {90,6.6617,1.6269},
// {91,3.0622,1.9185},
// {92,0.6733,2.4744},
// {93,1.355,1.0267},
// {94,3.75,9.499},
// {95,7.2441,0.5949},
// {96,3.3434,4.9163},
// {97,8.7538,5.3958},
// {98,7.4316,2.6315},
// {99,3.6239,5.3696},
// {100,3.2393,3.0533}
// ],lMatrix);

// dCentroidMatrix:=DATASET([
// {1,1,1},
// {2,2,2},
// {3,3,3},
// {4,4,4}
// ],lMatrix);



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

// d02Prep; 

dCentroid0 := PROJECT(d02Prep,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[1];SELF:=LEFT;));
OUTPUT(dCentroid0,NAMED('Initial_Centroids')); 


//step 1: get Gt 
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
TYPEOF(Types.NumericField.number) inNum := 0;
TYPEOF(Types.NumericField.number) outNum := 0;
END;

Vcount0 := TABLE(dClosest0, lVcount, y);
OUTPUT(Vcount0, NAMED('Vcount0')); 	


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




//set the result of Standard Kmeans to the result of first iteration
firstIte := KmeansD01.AllResults();
// OUTPUT(firstIte,NAMED('firstIte'));
firstResult := PROJECT(firstIte,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[2];SELF:=LEFT;)); 
OUTPUT(firstResult,NAMED('firstResult'));

//*********************now Gt (deltaG will be initailized and updated in iteration ), ub, lb are initialized ************************

//********************************************start iterations*************************************************

//iteration 1:
c1 := 1;

// Now join to the existing centroid dataset to add the new values to the end of the values set.
dAdded1:=JOIN(d02Prep,firstResult,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=RIGHT;),RIGHT OUTER);

//calculate the deltaC
deltaC1 := dDistanceDelta(c1,c1-1,dAdded1);
OUTPUT(deltaC1,NAMED('deltaC1'));

// Check the distance delta for the last two iterations.  If the highest
// value is below the convergence threshold, then set bConverged to TRUE
bConverged1:= IF(MAX(deltaC1,value)<=nConverge,TRUE,FALSE);
OUTPUT(bConverged1, NAMED('iteration1_converge'));

// If the centroids have converged, simply pass the input dataset through
// to the next iteration.  Otherwise perform an iteration.
// result1 :=IF(bConverged1,d02Prep,dAdded1);
// OUTPUT(result1, NAMED('iteration1_result'));


//Since not converged
// set the current centroids to the results of the most recent iteration
dCentroid1 := PROJECT(dAdded1,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c1+1];SELF:=LEFT;));
OUTPUT(dCentroid1,NAMED('dCentroid1'));

//get deltaG1
//group deltaC1g by Gt: first JOIN(deltaC with Gt) then group by Gt
deltaC1g :=JOIN(deltaC1, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
OUTPUT(deltaC1g, NAMED('delatC1g'));
deltaC1Gt := SORT(deltaC1g, y,value);
output(deltaC1Gt,NAMED('deltaC1Gt'));
deltaG1 := DEDUP(deltaC1Gt,y, RIGHT);
OUTPUT(deltaG1,NAMED('deltaG1')); 

//update ub0 and lbs0
ub1 := JOIN(ub0, deltaC1Gt, LEFT.y = RIGHT.x, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;));
OUTPUT(ub1, NAMED('ub1'));
lbs1 := JOIN(lbs0, deltaG1, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := LEFT.value - RIGHT.value; SELF := LEFT));
OUTPUT(lbs1, NAMED('lbs1'));

//Groupfilter
// groupFilterTemp := JOIN(ub1, lbs1,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
// OUTPUT(groupFilterTemp,NAMED('groupFilterTemp'));
// groupFilterTemp1:= groupFilterTemp(value <0);
// lbs1Temp := JOIN(lbs1, groupFilterTemp1, LEFT.x = RIGHT.x, TRANSFORM(LEFT));
// groupFilter1 := JOIN(ub0, lbs1Temp, LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT;));

//groupfilter1 testing on one time comparison
groupFilterTemp := JOIN(ub0, lbs1,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
OUTPUT(groupFilterTemp,NAMED('groupFilterTemp'));
groupFilter1:= groupFilterTemp(value <0);
OUTPUT(groupFilter1,NAMED('groupFilter1_comparison'));


pGroupFilter1 := groupFilter1(value < 0);
OUTPUT(pGroupFilter1,NAMED('groupFilter1'));
groupFilter1_Converge := IF(COUNT(pGroupFilter1)=0, TRUE, FALSE);
OUTPUT(groupFilter1_Converge, NAMED('groupFilter1_converge'));
OUTPUT(pGroupFilter1,NAMED('groupFilter1_result')); 

unpGroupFilter1 := groupFilter1(value >=0);

//localFilter
changedSetTemp := JOIN(d01,pGroupFilter1, LEFT.id = RIGHT.x,TRANSFORM(LEFT));
// changedSetTemp;
dDistances1 := ML.Cluster.Distances(changedSetTemp,dCentroid1);
dClosest1 := ML.Cluster.Closest(dDistances1);
OUTPUT(dClosest1,NAMED('distacesOfLocalfilter')); 

//the dataset that change their best centroid = result of local fitler		
localFilter1 := JOIN(dClosest1, dClosest0, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, TRANSFORM(LEFT));

//localFilter1 is empty then it's converged. Or update the ub and lb.
localFilter1_converge := IF(COUNT(localFilter1) =0, TRUE, FALSE);
OUTPUT(localFilter1_converge,NAMED('localFilter1_converge'));
OUTPUT(localFilter1,NAMED('localFilter1_result'));
// dAdded1;


//update the Ub of data points who do not change their best centroid		
unChangedUbSet := JOIN(ub0, localFilter1, LEFT.x != RIGHT.x, TRANSFORM(LEFT), ALL);
unChangedUbD01 := JOIN(unChangedUbSet, d01, LEFT.x = RIGHT.id, TRANSFORM({LEFT.x; LEFT.y; RIGHT.number; TYPEOF(RIGHT.value) xValue;},SELF.x := LEFT.x; SELF.y := LEFT.y; SELF.number := RIGHT.number; SELF.xValue := RIGHT.value;));
unChangedUbD01CTemp := JOIN(unchangedUbD01, dCentroid1, LEFT.y = RIGHT.id AND LEFT.number = RIGHT.number);
unChangedUbD01C := SORT(unChangedUbD01CTemp, x, number);
unChangedUbTemp := PROJECT(unChangedUbD01C, TRANSFORM(literations, SELF.id := LEFT.x; SELF.number := LEFT.number; SELF.values :=[LEFT.xvalue] + [LEFT.value]; ));
unChangedUbTemp1 := dDistanceDelta(0,1,unChangedUbTemp);
unChangedUb := JOIN(unChangedUbTemp1, ub0, LEFT.id = RIGHT.x, TRANSFORM(RECORDOF(ub0), SELF.x := RIGHT.x; SELF.y := RIGHT.y; SELF.value := LEFT.value; ));
OUTPUT(unChangedUb, NAMED('unChangedUb'));

//update the Ub of of data points who change their best centroid 		
changedUb := LocalFilter1;
OUTPUT(changedUb, NAMED('changedUb'));
//new Ub
ub1_result := changedUb + unchangedUb;
OUTPUT(ub1_result, NAMED('ub1_result'));

//update lbs	
changelbs1Temp := JOIN(localFilter1,dDistances1, LEFT.x = RIGHT.x, TRANSFORM(RIGHT));
changelbs1Temp1 := SORT(JOIN(localFilter1,changelbs1Temp, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
//new lbs for data points who change their best centroid
changelbs1 := DEDUP(changelbs1Temp,x );
OUTPUT(changelbs1,NAMED('changelbs1')) ;
//new lbs
lbs1_result := JOIN(lbs1, changelbs1,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value); SELF := LEFT;), LEFT OUTER );
OUTPUT(lbs1_result,NAMED('lbs1_result'));

//update Vcount
//calculate the Vin
VinSet1 := SORT(localFilter1,y);
dClusterCountsVin1:=TABLE(VinSet1, {y; INTEGER c := COUNT(GROUP)},y);
OUTPUT(dClusterCountsVin1,NAMED('dClusterCountsVin1'));

dClusteredVin1 := JOIN(d01, VinSet1, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
OUTPUT(dClusteredVin1,NAMED('dClusteredVin1'));
// dClusteredVin;
dRolledVin1:=ROLLUP(SORT(dClusteredVin1,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
// dRolledVin1;


VoutSet1 := JOIN(dClosest0, localFilter1, LEFT.x = RIGHT.x, TRANSFORM(LEFT));
// OUTPUT(VoutSet1,NAMED('VoutSet1'));
dClusterCountsVout1:=TABLE(VoutSet1, {y; INTEGER c := COUNT(GROUP)},y);
OUTPUT(dClusterCountsVout1,NAMED('dClusterCountsVout1'));
dClusteredVout1 := JOIN(d01,VoutSet1, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
OUTPUT(dClusteredVout1,NAMED('dClusteredVout1'));
dRolledVout1:=ROLLUP(SORT(dClusteredVout1,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
OUTPUT(dRolledVout1,NAMED('dRolledVout1'));

//calculate the new Vcount = old Vcount + |Vin| - |Vout| 
Vcount1Vin :=JOIN(Vcount0,dClusterCountsVin1, LEFT.id = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.inNum := RIGHT.c;  SELF := LEFT;), LEFT OUTER );
OUTPUT(Vcount1Vin,NAMED('Vcount1Vin'));
Vcount1VinVout :=JOIN(Vcount1Vin,dClusterCountsVout1, LEFT.id = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.outNum := RIGHT.c; SELF := LEFT;), LEFT OUTER);
OUTPUT(Vcount1VinVout,NAMED('Vcount1VinVout'));
Vcount1 := TABLE(Vcount1VinVout, {id; TYPEOF(Types.NumericField.number) c := (c + inNum - outNum); inNum; outNum;});
OUTPUT(Vcount1, NAMED('Vcount1'));


//let the old |c*V| to multiply old Vcount
CV1 :=JOIN(dCentroid1, Vcount0, LEFT.id = RIGHT.id, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value * RIGHT.c; SELF := LEFT), LEFT OUTER);
// CV1 ;

//Add Vin
CV1Vin := JOIN(CV1, dRolledVin1, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value + RIGHT.value; SELF := LEFT), LEFT OUTER);
// CV1Vin;

//Minus Vout
CV1VinVout:= JOIN(CV1Vin, dRolledVout1, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value - RIGHT.value; SELF := LEFT), LEFT OUTER);
OUTPUT(CV1VinVout , NAMED('CV1VinVout'));

//get the new C
dAdded2Temp:=JOIN(CV1VinVout,Vcount1,LEFT.id=RIGHT.id,TRANSFORM(RECORDOF(LEFT),SELF.value:=(LEFT.value/RIGHT.c);SELF:=LEFT;), LEFT OUTER);
OUTPUT(dAdded2Temp, NAMED('C2'));

//add to the dAdded1
dAdded2 := JOIN(dAdded1,dAdded2Temp,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=RIGHT;), LEFT OUTER);
// dAdded2;

//iteration 2
c2 := 2;
//update c
//step 1: get C' and deltaC --> c' according to equation 1 --> c' = (c* V + Vin - Vout)/Vnew
//calculate the deltaC
deltaC2 := dDistanceDelta(c2,c2-1,dAdded2);
OUTPUT(deltaC2, NAMED('deltaC2'));// **********************************************************************************************result5

// Check the distance delta for the last two iterations.  If the highest
// value is below the convergence threshold, then set bConverged to TRUE
bConverged2:= IF(MAX(deltaC2,value)<=nConverge,TRUE,FALSE);
OUTPUT(bConverged2, NAMED('iteration2_converge'));

//Since result is FALSE
// result2 :=IF(bConverged2,dAdded1,dAdded2);
// OUTPUT(result2, NAMED('iteration2_result')); 
// set the current centroids to the results of the most recent iteration
dCentroid2 := PROJECT(dAdded2,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c2+1];SELF:=LEFT;));
// dCentroid2; 

//get deltaC2 and deltaC2Gt and deltaG2
//get lbs0
//get deltaG
//group deltaC1g by Gt: first JOIN(deltaC with Gt) then group by Gt
deltaC2g :=JOIN(deltaC2, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
OUTPUT(deltaC2g, NAMED('delatC2g'));
deltaC2Gt := SORT(deltaC2g, y,value);
output(deltaC2Gt,NAMED('deltaC2Gt'));
deltaG2 := DEDUP(deltaC2Gt,y, RIGHT);
OUTPUT(deltaG2,NAMED('deltaG2')); 

//update ub2 and lbs2
ub2 := JOIN(ub1_result, deltaC2Gt, LEFT.y = RIGHT.x, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;));
OUTPUT(ub2, NAMED('ub2'));
lbs2 := JOIN(lbs1_result, deltaG2, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := LEFT.value - RIGHT.value; SELF := LEFT;));
OUTPUT(lbs2, NAMED('lbs2'));

//Groupfilter
// groupFilterTempIt2 := JOIN(ub2, lbs2,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
// OUTPUT(groupFilterTempIt2,NAMED('groupFilterTempIt2'));
// groupFilterTempIt21:= groupFilterTempIt2(value <0);
// lbs2Temp := JOIN(lbs2, groupFilterTempIt21, LEFT.x = RIGHT.x, TRANSFORM(LEFT));
// groupFilter2 := JOIN(ub1, lbs2Temp, LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT;));

//groupfilter2 test for comparison
groupFilterTempIt2 := JOIN(ub1, lbs2,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
// OUTPUT(groupFilterTempIt2,NAMED('groupFilterTempIt2'));
groupFilter2:= groupFilterTempIt2(value <0);

pGroupFilter2 := groupFilter2(value < 0);
OUTPUT(pGroupFilter2,NAMED('groupFilter2'));
groupFilter2_Converge := IF(COUNT(pGroupFilter2)=0, TRUE, FALSE);
OUTPUT(groupFilter2_Converge, NAMED('groupFilter2_converge'));
OUTPUT(pGroupFilter2,NAMED('groupFilter2_result')); 

unpGroupFilter2 := groupFilter2(value >=0);

//localFilter
changedSetTempIt2 := JOIN(d01,pGroupFilter2, LEFT.id = RIGHT.x,TRANSFORM(LEFT));
// changedSetTemp;
dDistances2 := ML.Cluster.Distances(changedSetTempIt2,dCentroid2);
OUTPUT(dDistances2,NAMED('dDistances2')); 
dClosest2 := ML.Cluster.Closest(dDistances2);
OUTPUT(dClosest2,NAMED('distacesOfLocalfilter2')); 

//the dataset that change their best centroid = result of local fitler		
localFilter2 := JOIN(dClosest2, dClosest1, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, TRANSFORM(LEFT));

//localFilter2 is empty then it's converged. Or update the ub and lb.
localFilter2_converge := IF(COUNT(localFilter2) =0, TRUE, FALSE);
OUTPUT(localFilter2_converge,NAMED('localFilter2_converge'));
OUTPUT(localFilter2,NAMED('localFilter2_result'));

// If the centroids have converged, simply pass the input dataset through
// to the next iteration.  Otherwise perform an iteration.
localFilter2_result :=IF(localFilter2_Converge,dAdded2,dAdded1);
OUTPUT(localFilter2_result, NAMED('iteration2_result'));