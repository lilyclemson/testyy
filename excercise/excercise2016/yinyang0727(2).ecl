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
// c := 0;

//Inputs:
d01 := dDocuments;
d02 := dCentroids;
n := 30;
nConverge := 0.3;

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
lbs0_iniTemp1 := SORT(JOIN(lbs0_iniTemp, Gt, LEFT.y = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.y := RIGHT.y; SELF := LEFT;)),x, value);
lbs0_ini := DEDUP(lbs0_iniTemp1,x);
// lbs0_ini := DEDUP(lbs0_iniTemp,x, RIGHT);
OUTPUT(lbs0_iniTemp,NAMED('lbs0_iniTemp')) ;
OUTPUT(lbs0_ini,NAMED('lbs0_ini')) ;
// OUTPUT(lbs0_iniTemp1,NAMED('lbs0_iniTemp1')) ;

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

input0 := iCentroid0 + iUb0 + iLbs0 + iV0 ;
OUTPUT(input0, NAMED('input0'));


d := input0;
c:=1;
//********************************************start iterations*************************************************
// lInput yyfIterate(DATASET(lInput) d,UNSIGNED c):=FUNCTION
			
			dAddedx := PROJECT(d(id = 1), TRANSFORM(lIterations , SELF.id := LEFT. x; SELF.number:= LEFT.y; SELF.values := LEFT.values;));
			iUb := TABLE(d(id = 2), {x;y;values;});
			iLbs := TABLE(d(id = 3), {x;y;values;});
			iV := TABLE(d(id = 4), {x;y;values;});
			
			ub0 := PROJECT(d(id = 2), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c]; SELF := LEFT;));
			OUTPUT(ub0, NAMED('ub0'));
			lbs0 := PROJECT(d(id = 3), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c]; SELF := LEFT;));
			OUTPUT(lbs0, NAMED('lbs0'));
			V0 := PROJECT(d(id = 4), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c]; SELF := LEFT;));
			OUTPUT(V0, NAMED('V0'));
			
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
			ub1_temp := JOIN(ub0, deltac, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;));
			OUTPUT(ub1_temp, NAMED('ub1_temp'));
			lbs1_temp := JOIN(lbs0, deltaG1, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := LEFT.value - RIGHT.value; SELF := LEFT));
			OUTPUT(lbs1_temp, NAMED('lbs1_temp'));
			
			
		
			// groupfilter1 testing on one time comparison
			groupFilterTemp := JOIN(ub0, lbs1_temp,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
			OUTPUT(groupFilterTemp,NAMED('groupFilterTemp'));
			groupFilter1:= groupFilterTemp(value <0);
			OUTPUT(groupFilter1,NAMED('groupFilter1_comparison'));
			
			pGroupFilter1 := groupFilter1;
			OUTPUT(pGroupFilter1,NAMED('groupFilter1'));
			groupFilter1_Converge := IF(COUNT(pGroupFilter1)=0, TRUE, FALSE);
			OUTPUT(groupFilter1_Converge, NAMED('groupFilter1_converge'));
			OUTPUT(pGroupFilter1,NAMED('groupFilter1_result')); 

			unpGroupFilter1 := groupFilterTemp(value >=0);

			//localFilter
			changedSetTemp := JOIN(d01,pGroupFilter1, LEFT.id = RIGHT.x,TRANSFORM(LEFT));
			// changedSetTemp;
			dDistances1 := ML.Cluster.Distances(changedSetTemp,dCentroid1);
			OUTPUT(dDistances1,NAMED('dDistances1')); 
			dClosest1 := ML.Cluster.Closest(dDistances1);
			OUTPUT(dClosest1,NAMED('dclosest1')); 

			//the dataset that change their best centroid = result of local fitler		
			localFilter1 := JOIN(dClosest1, ub0, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, TRANSFORM(LEFT));

			//localFilter1 is empty then it's converged. Or update the ub and lb.
			localFilter1_converge := IF(COUNT(localFilter1) =0, TRUE, FALSE);
			OUTPUT(localFilter1_converge,NAMED('localFilter1_converge'));
			OUTPUT(localFilter1,NAMED('localFilter1_result'));
			// dAddedx;
			tempR := RECORD
			d01.id;
			d01.number;
			d01.value;
			TYPEOF(Types.NumericField.id) cid:=0 ;
			END;
			//update the Ub of data points who do not change their best centroid		
			unChangedUbSet := JOIN(ub0, localFilter1, LEFT.x = RIGHT.x, TRANSFORM(LEFT), LEFT ONLY);
			OUTPUT(unChangedUbSet, NAMED('unChangedUbSet'));	
			unChangedUbD01 := JOIN(unChangedUbSet, d01, LEFT.x = RIGHT.id, TRANSFORM(tempR, SELF.cid := LEFT.y; SELF := RIGHT; ));
			OUTPUT(unChangedUbD01, NAMED('unChangedUbD01'));	
			unChangedUbD01CTemp := SORT(JOIN(unChangedUbD01, dCentroid1, LEFT.cid = RIGHT.id AND LEFT.number = RIGHT.number , TRANSFORM(Mat.Types.Element, SELF.value := (LEFT.value-RIGHT.value)*(LEFT.value-RIGHT.value); SELF.y := RIGHT.id;SELF.x := LEFT.id;)),x);
			OUTPUT(unChangedUbD01CTemp, NAMED('unChangedUbD01CTemp'));		
			

			unChangedUbRolled:=ROLLUP(unChangedUbD01CTemp,TRANSFORM(Mat.Types.Element,SELF.value:=SQRT(LEFT.value+RIGHT.value);SELF := LEFT;),x);
			OUTPUT(unChangedUbRolled, NAMED('unChangedUbRolled'));		
			
			//update the Ub of of data points who change their best centroid 		
			changedUb := LocalFilter1;
			OUTPUT(changedUb, NAMED('changedUb'));
			unchangedUb:= unChangedUbRolled;
			//new Ub
			ub1:= SORT(changedUb + unchangedUb, x);
			OUTPUT(ub1, NAMED('ub1'));
			
			
			//update lbs	
			changelbs1Temp := JOIN(localFilter1,dDistances1, LEFT.x = RIGHT.x , TRANSFORM(RIGHT));
			OUTPUT(changelbs1temp,NAMED('changelbs1temp'));
			changelbs1Temp1 := SORT(JOIN(localFilter1,changelbs1Temp, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			OUTPUT(changelbs1temp1,NAMED('changelbs1temp1'));
			//new lbs for data points who change their best centroid
			changelbs1 := DEDUP(changelbs1Temp1,x );
			OUTPUT(changelbs1,NAMED('changelbs1')) ;
			//new lbs
			lbs1 := JOIN(lbs1_temp, changelbs1,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value); SELF := LEFT;), LEFT OUTER );
			OUTPUT(lbs1,NAMED('lbs1'));
			
			
			/*
lbs0_iniTemp := JOIN(dClosest0,dDistances0, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y,RIGHT ONLY);
lbs0_iniTemp1 := SORT(JOIN(lbs0_iniTemp, Gt, LEFT.y = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.y := RIGHT.y; SELF := LEFT;)),x, value);
lbs0_ini := DEDUP(lbs0_iniTemp1,x);
// lbs0_ini := DEDUP(lbs0_iniTemp,x, RIGHT);
OUTPUT(lbs0_iniTemp,NAMED('lbs0_iniTemp')) ;
OUTPUT(lbs0_ini,NAMED('lbs0_ini')) ;
*/
			
			//update Vin and Vout
			//calculate the Vin
			VinSet1 := SORT(localFilter1,y);
			OUTPUT(VinSet1, NAMED('VinSet1'));
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
			

			outputSet1 := dAddedx1+ dAddedUb1 + dAddedLbs1 +dAddedV1 ;
			SORT(outputSet1,id);
			
//iteration2		

d2 := outputSet1;
c2:=2;
//********************************************start iterations*************************************************
// lInput yyfIterate(DATASET(lInput) d,UNSIGNED c):=FUNCTION
			
			dAddedx_input2 := PROJECT(d2(id = 1), TRANSFORM(lIterations , SELF.id := LEFT. x; SELF.number:= LEFT.y; SELF.values := LEFT.values;));
			iUb_input2 := TABLE(d2(id = 2), {x;y;values;});
			iLbs_input2 := TABLE(d2(id = 3), {x;y;values;});
			iV_input2 := TABLE(d2(id = 4), {x;y;values;});
			
			ub_input2 := PROJECT(d2(id = 2), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c2]; SELF := LEFT;));
			OUTPUT(ub_input2, NAMED('ub_input2'));
			lbs_input2 := PROJECT(d2(id = 3), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c2]; SELF := LEFT;));
			OUTPUT(lbs_input2, NAMED('lbs_input2'));
			V_input2 := PROJECT(d2(id = 4), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c2]; SELF := LEFT;));
			OUTPUT(V_input2, NAMED('V_input2'));
			
			//calculate the deltaC
			deltac2 := dDistanceDelta(c2,c2-1,dAddedx_input2);
			OUTPUT(deltac2,NAMED('deltac2'));
			
			bConverged2 := IF(MAX(deltac2,value)<=nConverge,TRUE,FALSE);
			OUTPUT(bConverged2, NAMED('iteration2_converge'));
			
			dCentroid2 := PROJECT(dAddedx_input2,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c2+1];SELF:=LEFT;));
			OUTPUT(dCentroid2,NAMED('dCentroid2'));

			//get deltaG1
			//group deltacg by Gt: first JOIN(deltac2 with Gt) then group by Gt
			deltacg2 :=JOIN(deltac2, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
			OUTPUT(deltacg2, NAMED('delatcg2'));
			deltacGt2 := SORT(deltacg2, y,value);
			output(deltacGt2,NAMED('deltacGt2'));
			deltaG2 := DEDUP(deltacGt2,y, RIGHT);
			OUTPUT(deltaG2,NAMED('deltaG2')); 

			//update ub_input2 and lbs_input2
			ub2_temp := SORT(JOIN(ub_input2, deltac2, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;)),x);
			OUTPUT(ub2_temp, NAMED('ub2_temp'));
			lbs2_temp := JOIN(lbs_input2, deltaG2, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := LEFT.value - RIGHT.value; SELF := LEFT));
			OUTPUT(lbs2_temp, NAMED('lbs2_temp'));
			
			
			// groupfilter1 testing on one time comparison
			groupFilterTemp2 := JOIN(ub_input2, lbs2_temp,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
			OUTPUT(groupFilterTemp2,NAMED('groupFilterTemp2'));
			groupFilter2:= groupFilterTemp2(value <0);
			OUTPUT(groupFilter2,NAMED('groupFilter2_comparison'));
			
			pGroupFilter2 := groupFilter2;
			OUTPUT(pGroupFilter2,NAMED('groupFilter2'));
			groupFilter2_Converge := IF(COUNT(pGroupFilter2)=0, TRUE, FALSE);
			OUTPUT(groupFilter2_Converge, NAMED('groupFilter2_Converge'));
			OUTPUT(pGroupFilter2,NAMED('groupFilter2_result')); 

			unpGroupFilter2 := groupFilterTemp2(value >=0);

			//localFilter
			changedSetTemp2 := JOIN(d01,pGroupFilter2, LEFT.id = RIGHT.x,TRANSFORM(LEFT));
			// changedSetTemp2;
			dDistances2 := ML.Cluster.Distances(changedSetTemp2,dCentroid2);
			dClosest2 := ML.Cluster.Closest(dDistances2);
			OUTPUT(dClosest2,NAMED('distancesOfLocalfilter2')); 

			//the dataset that change their best centroid = result of local fitler		
			localFilter2 := JOIN(dClosest2, ub_input2, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, TRANSFORM(LEFT));

			//localFilter2 is empty then it's converged. Or update the ub and lb.
			localFilter2_converge := IF(COUNT(localFilter2) =0, TRUE, FALSE);
			OUTPUT(localFilter2_converge,NAMED('localFilter2_converge'));
			OUTPUT(localFilter2,NAMED('localFilter2_result'));

			//update the Ub of data points who do not change their best centroid		
			unChangedUbSet2 := JOIN(ub_input2, localFilter2, LEFT.x = RIGHT.x, TRANSFORM(LEFT), LEFT ONLY);
			OUTPUT(unChangedUbSet2, NAMED('unChangedUbSet2'));	
			unChangedUbD012 := JOIN(unChangedUbSet2, d01, LEFT.x = RIGHT.id, TRANSFORM(tempR, SELF.cid := LEFT.y; SELF := RIGHT; ));
			OUTPUT(unChangedUbD012, NAMED('unChangedUbD012'));	
			unChangedUbD01CTemp2 := SORT(JOIN(unChangedUbD012, dCentroid2, LEFT.cid = RIGHT.id AND LEFT.number = RIGHT.number , TRANSFORM(Mat.Types.Element, SELF.value := (LEFT.value-RIGHT.value)*(LEFT.value-RIGHT.value); SELF.y := RIGHT.id;SELF.x := LEFT.id;)),x);
			OUTPUT(unChangedUbD01CTemp2, NAMED('unChangedUbD01CTemp2'));		
			

			unChangedUbRolled2:=ROLLUP(unChangedUbD01CTemp2,TRANSFORM(Mat.Types.Element,SELF.value:=SQRT(LEFT.value+RIGHT.value);SELF := LEFT;),x);
			OUTPUT(unChangedUbRolled2, NAMED('unChangedUbRolled2'));		
			
			//update the Ub of of data points who change their best centroid 		
			changedUb2 := localFilter2;
			OUTPUT(changedUb2, NAMED('changedUb2'));
			unchangedUb2:= unChangedUbRolled2;
			//new Ub
			ub2:= SORT(changedUb2 + unchangedUb2, x);
			OUTPUT(ub2, NAMED('ub2'));
			
			
			//update lbs	
			changelbsTemp2 := JOIN(localFilter2,dDistances2, LEFT.x = RIGHT.x, TRANSFORM(RIGHT));
			changelbsTemp22 := SORT(JOIN(localFilter2,changelbsTemp2, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			//new lbs for data points who change their best centroid
			changelbs2 := DEDUP(changelbsTemp2,x );
			OUTPUT(changelbs2,NAMED('changelbs2')) ;
			//new lbs
			lbs2 := JOIN(lbs2_temp, changelbs2,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value); SELF := LEFT;), LEFT OUTER );
			OUTPUT(lbs2,NAMED('lbs2'));
			
			//update Vin and Vout
			//calculate the Vin
			VinSet2 := SORT(localFilter2,y);
			OUTPUT(VinSet2, NAMED('VinSet2'));
			dClusterCountsVin2:=TABLE(VinSet2, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVin2,NAMED('dClusterCountsVin2'));
			dClusteredVin2 := JOIN(d01, VinSet2, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVin2,NAMED('dClusteredVin2'));
			dRolledVin2:=ROLLUP(SORT(dClusteredVin2,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVin2, NAMED('dRolledVin2'));


			VoutSet2 := JOIN(ub2_temp, localFilter2, LEFT.x = RIGHT.x, TRANSFORM(LEFT));
			OUTPUT(VoutSet2,NAMED('VoutSet2'));
			dClusterCountsVout2:=TABLE(VoutSet2, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVout2,NAMED('dClusterCountsVout2'));
			dClusteredVout2 := JOIN(d01,VoutSet2, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVout2,NAMED('dClusteredVout2'));
			dRolledVout2:=ROLLUP(SORT(dClusteredVout2,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVout2,NAMED('dRolledVout2'));

			
			V2Temp :=JOIN(V_input2,dClusterCountsVin2, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y +RIGHT.c; SELF.value := LEFT.value + RIGHT.c; SELF := LEFT;), LEFT OUTER );
			OUTPUT(V2Temp,NAMED('V2Temp'));
			V2:=JOIN(V2Temp,dClusterCountsVout2, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y -RIGHT.c;SELF.value := LEFT.value -RIGHT.c; SELF := LEFT;), LEFT OUTER);
			OUTPUT(V2,NAMED('V2'));
			

			//let the old |c*V| to multiply old Vcount
			CV2 :=JOIN(dCentroid2, V_input2, LEFT.id = RIGHT.x, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value * RIGHT.y; SELF := LEFT), LEFT OUTER);
			OUTPUT( CV2, NAMED('CV2')) ;

			//Add Vin
			CV2Vin := JOIN(CV2, dRolledVin2, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value + RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV2Vin, NAMED('CV2Vin'));

			//Minus Vout
			CV2VinVout:= JOIN(CV2Vin, dRolledVout2, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value - RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV2VinVout , NAMED('CV2VinVout'));
		
			//get the new C
			newC2:=JOIN(CV2VinVout,V2,LEFT.id=RIGHT.x,TRANSFORM(RECORDOF(LEFT),SELF.value:=(LEFT.value/RIGHT.y);SELF:=LEFT;), LEFT OUTER);
			OUTPUT(newC2, NAMED('C3'));
			
			dAddedx2Temp := JOIN(dAddedx_input2, newC2, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
			OUTPUT(dAddedx2Temp, NAMED('dAddedx2Temp'));
			dAddedx2 := PROJECT(dAddedx2Temp, TRANSFORM(lInput,SELF.id := 1;SELF.values:=LEFT.values;SELF.y := LEFT.number; SELF.x:=LEFT.id;));
			dAddedUb2 := JOIN(iUb_input2, ub2, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 2;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedUb2, NAMED('dAddedUb2'));
			dAddedLbs2 := JOIN(iLbs_input2, lbs2, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 3;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedLbs2, NAMED('dAddedLbs2'));
			dAddedV2 := JOIN(iV_input2, V2, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 4;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedV2, NAMED('dAddedV2'));
			

			outputSet2 := dAddedx2+ dAddedUb2 + dAddedLbs2 +dAddedV2 ;
			SORT(outputSet2,id);
			
//iteration3

d3 := outputSet2;
c3:=3;
//********************************************start iterations*************************************************
// lInput yyfIterate(DATASET(lInput) d,UNSIGNED c):=FUNCTION
			
			dAddedx_input3 := PROJECT(d3(id = 1), TRANSFORM(lIterations , SELF.id := LEFT. x; SELF.number:= LEFT.y; SELF.values := LEFT.values;));
			iUb_input3 := TABLE(d3(id = 2), {x;y;values;});
			iLbs_input3 := TABLE(d3(id = 3), {x;y;values;});
			iV_input3 := TABLE(d3(id = 4), {x;y;values;});
			
			ub_input3 := PROJECT(d3(id = 2), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c3]; SELF := LEFT;));
			OUTPUT(ub_input3, NAMED('ub_input3'));
			lbs_input3 := PROJECT(d3(id = 3), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c3]; SELF := LEFT;));
			OUTPUT(lbs_input3, NAMED('lbs_input3'));
			V_input3 := PROJECT(d3(id = 4), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c3]; SELF := LEFT;));
			OUTPUT(V_input3, NAMED('V_input3'));
			
			//calculate the deltaC
			deltac3 := dDistanceDelta(c3,c3-1,dAddedx_input3);
			OUTPUT(deltac3,NAMED('deltac3'));
			
			bConverged3 := IF(MAX(deltac3,value)<=nConverge,TRUE,FALSE);
			OUTPUT(bConverged3, NAMED('iteration3_converge'));
			
			dCentroid3 := PROJECT(dAddedx_input3,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c3+1];SELF:=LEFT;));
			OUTPUT(dCentroid3,NAMED('dCentroid3'));

			//get deltaG1
			//group deltacg by Gt: first JOIN(deltac3 with Gt) then group by Gt
			deltacg3 :=JOIN(deltac3, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
			OUTPUT(deltacg3, NAMED('delatcg3'));
			deltacGt3 := SORT(deltacg3, y,value);
			output(deltacGt3,NAMED('deltacGt3'));
			deltaG3 := DEDUP(deltacGt3,y, RIGHT);
			OUTPUT(deltaG3,NAMED('deltaG3')); 

			//update ub_input3 and lbs_input3
			ub3_temp := SORT(JOIN(ub_input3, deltac3, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;)),x);
			OUTPUT(ub3_temp, NAMED('ub3_temp'));
			lbs3_temp := JOIN(lbs_input3, deltaG3, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := LEFT.value - RIGHT.value; SELF := LEFT));
			OUTPUT(lbs3_temp, NAMED('lbs3_temp'));
			
			
			// groupfilter1 testing on one time comparison
			groupFilterTemp3 := JOIN(ub_input3, lbs3_temp,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
			OUTPUT(groupFilterTemp3,NAMED('groupFilterTemp3'));
			groupFilter3:= groupFilterTemp3(value <0);
			OUTPUT(groupFilter3,NAMED('groupFilter3_comparison'));
			
			pGroupFilter3 := groupFilter3;
			OUTPUT(pGroupFilter3,NAMED('groupFilter3'));
			groupFilter3_Converge := IF(COUNT(pGroupFilter3)=0, TRUE, FALSE);
			OUTPUT(groupFilter3_Converge, NAMED('groupFilter3_Converge'));
			OUTPUT(pGroupFilter3,NAMED('groupFilter3_result')); 

			unpGroupFilter3 := groupFilterTemp3(value >=0);

			//localFilter
			changedSetTemp3 := JOIN(d01,pGroupFilter3, LEFT.id = RIGHT.x,TRANSFORM(LEFT));
			// changedSetTemp3;
			dDistances3 := ML.Cluster.Distances(changedSetTemp3,dCentroid3);
			dClosest3 := ML.Cluster.Closest(dDistances3);
			OUTPUT(dClosest3,NAMED('distancesOfLocalfilter3')); 

			//the dataset that change their best centroid = result of local fitler		
			localFilter3 := JOIN(dClosest3, ub_input3, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, TRANSFORM(LEFT));

			//localFilter3 is empty then it's converged. Or update the ub and lb.
			localFilter3_converge := IF(COUNT(localFilter3) =0, TRUE, FALSE);
			OUTPUT(localFilter3_converge,NAMED('localFilter3_converge'));
			OUTPUT(localFilter3,NAMED('localFilter3_result'));

			//update the Ub of data points who do not change their best centroid		
			unChangedUbSet3 := JOIN(ub_input3, localFilter3, LEFT.x = RIGHT.x, TRANSFORM(LEFT), LEFT ONLY);
			OUTPUT(unChangedUbSet3, NAMED('unChangedUbSet3'));	
			unChangedUbD013 := JOIN(unChangedUbSet3, d01, LEFT.x = RIGHT.id, TRANSFORM(tempR, SELF.cid := LEFT.y; SELF := RIGHT; ));
			OUTPUT(unChangedUbD013, NAMED('unChangedUbD013'));	
			unChangedUbD01CTemp3 := SORT(JOIN(unChangedUbD013, dCentroid3, LEFT.cid = RIGHT.id AND LEFT.number = RIGHT.number , TRANSFORM(Mat.Types.Element, SELF.value := (LEFT.value-RIGHT.value)*(LEFT.value-RIGHT.value); SELF.y := RIGHT.id;SELF.x := LEFT.id;)),x);
			OUTPUT(unChangedUbD01CTemp3, NAMED('unChangedUbD01CTemp3'));		
			

			unChangedUbRolled3:=ROLLUP(unChangedUbD01CTemp3,TRANSFORM(Mat.Types.Element,SELF.value:=SQRT(LEFT.value+RIGHT.value);SELF := LEFT;),x);
			OUTPUT(unChangedUbRolled3, NAMED('unChangedUbRolled3'));		
			
			//update the Ub of of data points who change their best centroid 		
			changedUb3 := localFilter3;
			OUTPUT(changedUb3, NAMED('changedUb3'));
			unchangedUb3:= unChangedUbRolled3;
			//new Ub
			ub3:= SORT(changedUb3 + unchangedUb3, x);
			OUTPUT(ub3, NAMED('ub3'));
			
			
			//update lbs	
			changelbsTemp3 := JOIN(localFilter3,dDistances3, LEFT.x = RIGHT.x, TRANSFORM(RIGHT));
			changelbsTemp33 := SORT(JOIN(localFilter3,changelbsTemp3, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			//new lbs for data points who change their best centroid
			changelbs3 := DEDUP(changelbsTemp3,x );
			OUTPUT(changelbs3,NAMED('changelbs3')) ;
			//new lbs
			lbs3 := JOIN(lbs3_temp, changelbs3,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value); SELF := LEFT;), LEFT OUTER );
			OUTPUT(lbs3,NAMED('lbs3'));
			
			//update Vin and Vout
			//calculate the Vin
			VinSet3 := SORT(localFilter3,y);
			OUTPUT(VinSet3, NAMED('VinSet3'));
			dClusterCountsVin3:=TABLE(VinSet3, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVin3,NAMED('dClusterCountsVin3'));
			dClusteredVin3 := JOIN(d01, VinSet3, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVin3,NAMED('dClusteredVin3'));
			dRolledVin3:=ROLLUP(SORT(dClusteredVin3,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVin3, NAMED('dRolledVin3'));


			VoutSet3 := JOIN(ub3_temp, localFilter3, LEFT.x = RIGHT.x, TRANSFORM(LEFT));
			OUTPUT(VoutSet3,NAMED('VoutSet3'));
			dClusterCountsVout3:=TABLE(VoutSet3, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVout3,NAMED('dClusterCountsVout3'));
			dClusteredVout3 := JOIN(d01,VoutSet3, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVout3,NAMED('dClusteredVout3'));
			dRolledVout3:=ROLLUP(SORT(dClusteredVout3,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVout3,NAMED('dRolledVout3'));

			
			V3Temp :=JOIN(V_input3,dClusterCountsVin3, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y +RIGHT.c; SELF.value := LEFT.value + RIGHT.c; SELF := LEFT;), LEFT OUTER );
			OUTPUT(V3Temp,NAMED('V3Temp'));
			V3:=JOIN(V3Temp,dClusterCountsVout3, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y -RIGHT.c;SELF.value := LEFT.value -RIGHT.c; SELF := LEFT;), LEFT OUTER);
			OUTPUT(V3,NAMED('V3'));
			

			//let the old |c*V| to multiply old Vcount
			CV3 :=JOIN(dCentroid3, V_input3, LEFT.id = RIGHT.x, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value * RIGHT.y; SELF := LEFT), LEFT OUTER);
			OUTPUT( CV3, NAMED('CV3')) ;

			//Add Vin
			CV3Vin := JOIN(CV3, dRolledVin3, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value + RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV3Vin, NAMED('CV3Vin'));

			//Minus Vout
			CV3VinVout:= JOIN(CV3Vin, dRolledVout3, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value - RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV3VinVout , NAMED('CV3VinVout'));
		
			//get the new C
			newC3:=JOIN(CV3VinVout,V3,LEFT.id=RIGHT.x,TRANSFORM(RECORDOF(LEFT),SELF.value:=(LEFT.value/RIGHT.y);SELF:=LEFT;), LEFT OUTER);
			OUTPUT(newC3, NAMED('C4'));
			
			dAddedx3Temp := JOIN(dAddedx_input3, newC3, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
			OUTPUT(dAddedx3Temp, NAMED('dAddedx3Temp'));
			dAddedx3 := PROJECT(dAddedx3Temp, TRANSFORM(lInput,SELF.id := 1;SELF.values:=LEFT.values;SELF.y := LEFT.number; SELF.x:=LEFT.id;));
			dAddedUb3 := JOIN(iUb_input3, ub3, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 3;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedUb3, NAMED('dAddedUb3'));
			dAddedLbs3 := JOIN(iLbs_input3, lbs3, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 3;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedLbs3, NAMED('dAddedLbs3'));
			dAddedV3 := JOIN(iV_input3, V3, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 4;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedV3, NAMED('dAddedV3'));
			

			outputSet3 := dAddedx3+ dAddedUb3 + dAddedLbs3 +dAddedV3 ;
			SORT(outputSet3,id);		
			