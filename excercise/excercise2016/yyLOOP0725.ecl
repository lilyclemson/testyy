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
// c := 0;

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
ub0_ini := dClosest0; 
OUTPUT(ub0_ini, NAMED('ub0_ini'));

//get lbs0_ini
//lb of each group
lbs0_iniTemp := JOIN(dClosest0,dDistances0, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y,RIGHT ONLY);
lbs0_ini := SORT(JOIN(lbs0_iniTemp, Gt, LEFT.y = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.y := RIGHT.y; SELF := LEFT;)), x);
OUTPUT(lbs0_iniTemp,NAMED('lbs0_iniTemp')) ;
OUTPUT(lbs0_ini,NAMED('lbs0_ini')) ;

//initialize Vcount0
//Get V
lVcount := RECORD
TYPEOF(Types.NumericField.id)  id := dClosest0.y;
TYPEOF(Types.NumericField.number) c := COUNT(group);
TYPEOF(Types.NumericField.value)   inNum:= 0;
TYPEOF(Types.NumericField.number) outNum := 0;
END;


Vin0temp := TABLE(dClosest0, {y; UNSIGNED c := COUNT(GROUP);},y );
OUTPUT(Vin0temp, NAMED('Vin0temp'));
// Vin0_ini := PROJECT(Vin0temp, TRANSFORM(Mat.Types.Element, SELF.x := LEFT.y; SELF.y := LEFT.c; SELF.value := 0; ));
// OUTPUT(Vin0_ini, NAMED('Vin0_ini'));
V0_ini := PROJECT(Vin0temp, TRANSFORM(Mat.Types.Element, SELF.x := LEFT.y; SELF.y := LEFT.c; SELF.value := 0; ));
OUTPUT(V0_ini, NAMED('V0_ini'));

// Vout0_ini := Vin0_ini;



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
// Now join to the existing centroid dataset to add the new values to the end of the values set.
dAdded1:=JOIN(d02Prep,firstResult,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=RIGHT;),RIGHT OUTER);

//data preparation

//transform dCentroid to the format align with ub_ini, lbs0_ini, Vin0_ini, Vout0_ini
dCentroid0Trans := PROJECT(dCentroid0, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value ; SELF.x := LEFT.id; SELF.y := LEFT.number));


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

iCentroid0Temp := PROJECT(dCentroid0Trans, transFormat(LEFT, 1));
iCentroid0 := JOIN(iCentroid0temp, firstResult, LEFT.x = RIGHT.id AND LEFT.y = RIGHT.number, TRANSFORM(lInput, SELF.values := LEFT.values + [RIGHT.value]; SELF := LEFT;));

iUb0 := PROJECT(ub0_ini, transFormat(LEFT, 2));
iLbs0 := PROJECT(lbs0_ini, transFormat(LEFT, 3));
iV0 := PROJECT(V0_ini, transFormat(LEFT, 4));
// iVout0 := PROJECT(Vout0_ini, transFormat(LEFT, 5));
// iClosest0 := PROJECT(dClosest0, transFormat(LEFT, 6));

// OUTPUT(iCentroid0, NAMED('iCentroid0'));
// OUTPUT(iUb0, NAMED('iUb0'));
// OUTPUT(iLbs0, NAMED('iLbs0'));
// OUTPUT(iVin0, NAMED('iVin0'));
// OUTPUT(iVout0, NAMED('iVout0'));
// OUTPUT(iClosest0, NAMED('iClosest'));

input0 := iCentroid0 + iUb0 + iLbs0 + iV0 ;
OUTPUT(input0, NAMED('input0'));

// d := input0;
// c:=1;
//********************************************start iterations*************************************************
lInput yyfIterate(DATASET(lInput) d,UNSIGNED c):=FUNCTION
			
			dAddedx := PROJECT(d(id = 1), TRANSFORM(lIterations , SELF.id := LEFT. x; SELF.number:= LEFT.y; SELF.values := LEFT.values;));
			iUb := TABLE(d(id = 2), {x;y;values;});
			iLbs := TABLE(d(id = 3), {x;y;values;});
			iV := TABLE(d(id = 4), {x;y;values;});
			// iVout := TABLE(d(id = 5), {x;y;values;});
			
			ub0 := PROJECT(d(id = 2), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c]; SELF := LEFT;));
			lbs0 := PROJECT(d(id = 3), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c]; SELF := LEFT;));
			V0 := PROJECT(d(id = 4), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c]; SELF := LEFT;));
			
			//calculate the deltaC
			deltac := dDistanceDelta(c,c-1,dAddedx);
			OUTPUT(deltac,NAMED('deltac'));
			
			bConverged := IF(MAX(deltac,value)<=nConverge,TRUE,FALSE);
			OUTPUT(bConverged, NAMED('iteration1_converge'));
			
			dCentroid1 := PROJECT(dAddedx,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c+1];SELF:=LEFT;));
			OUTPUT(dCentroid1,NAMED('dCentroid1'));

			//get deltaG1
			//group deltacg by Gt: first JOIN(deltaC with Gt) then group by Gt
			deltacg :=JOIN(deltac, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
			OUTPUT(deltacg, NAMED('delatcg'));
			deltacGt := SORT(deltacg, y,value);
			output(deltacGt,NAMED('deltacGt'));
			deltaG1 := DEDUP(deltacGt,y, RIGHT);
			OUTPUT(deltaG1,NAMED('deltaG1')); 

			//update ub0 and lbs0
			ub1_temp := JOIN(ub0, deltacGt, LEFT.y = RIGHT.x, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;));
			OUTPUT(ub1_temp, NAMED('ub1_temp'));
			lbs1_temp := JOIN(lbs0, deltaG1, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := LEFT.value - RIGHT.value; SELF := LEFT));
			OUTPUT(lbs1_temp, NAMED('lbs1_temp'));
			
			//groupfilter1 testing on one time comparison
			groupFilterTemp := JOIN(ub0, lbs1_temp,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
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
			localFilter1 := JOIN(dClosest1, ub0, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, TRANSFORM(LEFT));

			//localFilter1 is empty then it's converged. Or update the ub and lb.
			localFilter1_converge := IF(COUNT(localFilter1) =0, TRUE, FALSE);
			OUTPUT(localFilter1_converge,NAMED('localFilter1_converge'));
			OUTPUT(localFilter1,NAMED('localFilter1_result'));
			// dAddedx;

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
			ub1:= SORT(changedUb + unchangedUb, x);
			OUTPUT(ub1, NAMED('ub1'));

			//update lbs	
			changelbs1Temp := JOIN(localFilter1,dDistances1, LEFT.x = RIGHT.x, TRANSFORM(RIGHT));
			changelbs1Temp1 := SORT(JOIN(localFilter1,changelbs1Temp, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			//new lbs for data points who change their best centroid
			changelbs1 := DEDUP(changelbs1Temp,x );
			OUTPUT(changelbs1,NAMED('changelbs1')) ;
			//new lbs
			lbs1 := JOIN(lbs1_temp, changelbs1,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value); SELF := LEFT;), LEFT OUTER );
			OUTPUT(lbs1,NAMED('lbs1'));
			
			//update Vin and Vout
			//calculate the Vin
			VinSet1 := SORT(localFilter1,y);
			dClusterCountsVin1:=TABLE(VinSet1, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVin1,NAMED('dClusterCountsVin1'));

			dClusteredVin1 := JOIN(d01, VinSet1, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVin1,NAMED('dClusteredVin1'));
			dRolledVin1:=ROLLUP(SORT(dClusteredVin1,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVin1, NAMED('dRolledVin1'));


			VoutSet1 := JOIN(ub1_temp, localFilter1, LEFT.x = RIGHT.x, TRANSFORM(LEFT));
			OUTPUT(VoutSet1,NAMED('VoutSet1'));
			dClusterCountsVout1:=TABLE(VoutSet1, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVout1,NAMED('dClusterCountsVout1'));
			dClusteredVout1 := JOIN(d01,VoutSet1, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVout1,NAMED('dClusteredVout1'));
			dRolledVout1:=ROLLUP(SORT(dClusteredVout1,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVout1,NAMED('dRolledVout1'));
			


			//calculate the new Vcount = old Vcount + |Vin| - |Vout| 
			// Vin1 :=JOIN(Vin0,dClusterCountsVin1, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value +RIGHT.c; SELF.y := LEFT.y + RIGHT.c; SELF := LEFT;), LEFT OUTER );
			// OUTPUT(Vin1,NAMED('Vin1'));
			// Vout1:=JOIN(Vout0,dClusterCountsVout1, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value +RIGHT.c; SELF.y := LEFT.y - RIGHT.c;SELF := LEFT;), LEFT OUTER);
			// OUTPUT(Vout1,NAMED('Vout'));
			
			// Vin1 :=JOIN(Vin0,dClusterCountsVin1, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value +RIGHT.c;  SELF := LEFT;), LEFT OUTER );
			// OUTPUT(Vin1,NAMED('Vin1'));
			// Vout1:=JOIN(Vout0,dClusterCountsVout1, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value +RIGHT.c; SELF := LEFT;), LEFT OUTER);
			// OUTPUT(Vout1,NAMED('Vout1'));
			
			V1Temp :=JOIN(V0,dClusterCountsVin1, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y +RIGHT.c; SELF.value := LEFT.value + RIGHT.c; SELF := LEFT;), LEFT OUTER );
			OUTPUT(V1Temp,NAMED('V1Temp'));
			V1:=JOIN(V1Temp,dClusterCountsVout1, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y -RIGHT.c;SELF.value := LEFT.value -RIGHT.c; SELF := LEFT;), LEFT OUTER);
			OUTPUT(V1,NAMED('V1'));
			

			//let the old |c*V| to multiply old Vcount
			CV1 :=JOIN(dCentroid1, V0, LEFT.id = RIGHT.x, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value * RIGHT.y; SELF := LEFT), LEFT OUTER);
			OUTPUT( CV1, NAMED('CV1')) ;

			//Add Vin
			CV1Vin := JOIN(CV1, dRolledVin1, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value + RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV1Vin, NAMED('CV1Vin'));

			//Minus Vout
			CV1VinVout:= JOIN(CV1Vin, dRolledVout1, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value - RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV1VinVout , NAMED('CV1VinVout'));
		
			//get the new C
			newC:=JOIN(CV1VinVout,V1,LEFT.id=RIGHT.x,TRANSFORM(RECORDOF(LEFT),SELF.value:=(LEFT.value/RIGHT.y);SELF:=LEFT;), LEFT OUTER);
			OUTPUT(newC, NAMED('C2'));
			
			dAddedx1Temp := JOIN(dAddedx, newC, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
			OUTPUT(dAddedx1Temp, NAMED('dAddedx1Temp'));
			dAddedx1 := PROJECT(dAddedx1Temp, TRANSFORM(lInput,SELF.id := 1;SELF.values:=LEFT.values;SELF.y := LEFT.number; SELF.x:=LEFT.id;));
			dAddedUb1 := JOIN(iUb, ub1, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 2;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedUb1, NAMED('dAddedUb1'));
			dAddedLbs1 := JOIN(iLbs, lbs1, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 3;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedLbs1, NAMED('dAddedLbs1'));
			dAddedV1 := JOIN(iV, V1, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 4;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedLbs1, NAMED('dAddedVin1'));
			

			outputSet := dAddedx1+ dAddedUb1 + dAddedLbs1 +dAddedV1 ;
			SORT(outputSet,id);
			

			RETURN IF(bConverged, SORT(d,id), SORT(outputSet,id));
			// RETURN SORT(outputSet,id);
END;

yyResult :=LOOP(input0,n,yyfIterate(ROWS(LEFT),COUNTER));

OUTPUT(yyResult, NAMED('yyResult'));
			
			

