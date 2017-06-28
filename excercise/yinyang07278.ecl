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
			changelbsTemp1 := JOIN(localFilter1,dDistances1, LEFT.x = RIGHT.x , TRANSFORM(RIGHT));
			OUTPUT(changelbstemp1,NAMED('changelbs1temp1'));
			changelbsTemp11 := SORT(JOIN(localFilter1,changelbsTemp1, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			OUTPUT(changelbstemp11,NAMED('changelbstemp11'));
			//new lbs for data points who change their best centroid
			changelbs1 := DEDUP(changelbsTemp11,x );
			OUTPUT(changelbs1,NAMED('changelbs1')) ;
			//new lbs
			lbs1 := JOIN(lbs1_temp, changelbs1,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value); SELF := LEFT;), LEFT OUTER );
			OUTPUT(lbs1,NAMED('lbs1'));

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
			OUTPUT(dDistances2,NAMED('dDistances2'));
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
			OUTPUT(changelbstemp2,NAMED('changelbstemp2'));
			changelbsTemp22 := SORT(JOIN(localFilter2,changelbsTemp2, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			OUTPUT(changelbstemp22,NAMED('changelbstemp22'));
			//new lbs for data points who change their best centroid
			changelbs2 := DEDUP(changelbsTemp22,x );
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
			OUTPUT(dDistances3,NAMED('dDistances3'));
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
			OUTPUT(changelbsTemp3, NAMED('changelbsTemp3'));
			changelbsTemp33 := SORT(JOIN(localFilter3,changelbsTemp3, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			OUTPUT(changelbsTemp33, NAMED('changelbsTemp33'));
			//new lbs for data points who change their best centroid
			changelbs3 := DEDUP(changelbsTemp33,x );
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
			dAddedUb3 := JOIN(iUb_input3, ub3, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 2;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedUb3, NAMED('dAddedUb3'));
			dAddedLbs3 := JOIN(iLbs_input3, lbs3, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 3;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedLbs3, NAMED('dAddedLbs3'));
			dAddedV3 := JOIN(iV_input3, V3, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 4;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedV3, NAMED('dAddedV3'));
			

			outputSet3 := dAddedx3+ dAddedUb3 + dAddedLbs3 +dAddedV3 ;
			SORT(outputSet3,id);		
			
			
//iteration4(actually it's 5 so in the spreadsheet it's iteration 5)
d4 := outputSet3;
c4:=4;
//********************************************start iterations*************************************************
// lInput yyfIterate(DATASET(lInput) d,UNSIGNED c):=FUNCTION
			
			dAddedx_input4 := PROJECT(d4(id = 1), TRANSFORM(lIterations , SELF.id := LEFT. x; SELF.number:= LEFT.y; SELF.values := LEFT.values;));
			iUb_input4 := TABLE(d4(id = 2), {x;y;values;});
			iLbs_input4 := TABLE(d4(id = 3), {x;y;values;});
			iV_input4 := TABLE(d4(id = 4), {x;y;values;});
			
			ub_input4 := PROJECT(d4(id = 2), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c4]; SELF := LEFT;));
			OUTPUT(ub_input4, NAMED('ub_input4'));
			lbs_input4 := PROJECT(d4(id = 3), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c4]; SELF := LEFT;));
			OUTPUT(lbs_input4, NAMED('lbs_input4'));
			V_input4 := PROJECT(d4(id = 4), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c4]; SELF := LEFT;));
			OUTPUT(V_input4, NAMED('V_input4'));
			
			//calculate the deltaC
			deltac4 := dDistanceDelta(c4,c4-1,dAddedx_input4);
			OUTPUT(deltac4,NAMED('deltac4'));
			
			bConverged4 := IF(MAX(deltac4,value)<=nConverge,TRUE,FALSE);
			OUTPUT(bConverged4, NAMED('iteration4_converge'));
			
			dCentroid4 := PROJECT(dAddedx_input4,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c4+1];SELF:=LEFT;));
			OUTPUT(dCentroid4,NAMED('dCentroid4'));

			//get deltaG1
			//group deltacg by Gt: first JOIN(deltac4 with Gt) then group by Gt
			deltacg4 :=JOIN(deltac4, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
			OUTPUT(deltacg4, NAMED('delatcg4'));
			deltacGt4 := SORT(deltacg4, y,value);
			output(deltacGt4,NAMED('deltacGt4'));
			deltaG4 := DEDUP(deltacGt4,y, RIGHT);
			OUTPUT(deltaG4,NAMED('deltaG4')); 

			//update ub_input4 and lbs_input4
			ub4_temp := SORT(JOIN(ub_input4, deltac4, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;)),x);
			OUTPUT(ub4_temp, NAMED('ub4_temp'));
			lbs4_temp := JOIN(lbs_input4, deltaG4, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := LEFT.value - RIGHT.value; SELF := LEFT));
			OUTPUT(lbs4_temp, NAMED('lbs4_temp'));
			
			
			// groupfilter1 testing on one time comparison
			groupFilterTemp4 := JOIN(ub_input4, lbs4_temp,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
			OUTPUT(groupFilterTemp4,NAMED('groupFilterTemp4'));
			groupFilter4:= groupFilterTemp4(value <0);
			OUTPUT(groupFilter4,NAMED('groupFilter4_comparison'));
			
			pGroupFilter4 := groupFilter4;
			OUTPUT(pGroupFilter4,NAMED('groupFilter4'));
			groupFilter4_Converge := IF(COUNT(pGroupFilter4)=0, TRUE, FALSE);
			OUTPUT(groupFilter4_Converge, NAMED('groupFilter4_Converge'));
			OUTPUT(pGroupFilter4,NAMED('groupFilter4_result')); 

			unpGroupFilter4 := groupFilterTemp4(value >=0);

			//localFilter
			changedSetTemp4 := JOIN(d01,pGroupFilter4, LEFT.id = RIGHT.x,TRANSFORM(LEFT));
			// changedSetTemp4;
			dDistances4 := ML.Cluster.Distances(changedSetTemp4,dCentroid4);
			OUTPUT(dDistances4,NAMED('dDistances4'));
			dClosest4 := ML.Cluster.Closest(dDistances4);
			OUTPUT(dClosest4,NAMED('distancesOfLocalfilter4')); 

			//the dataset that change their best centroid = result of local fitler		
			localFilter4 := JOIN(dClosest4, ub_input4, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, TRANSFORM(LEFT));

			//localFilter4 is empty then it's converged. Or update the ub and lb.
			localFilter4_converge := IF(COUNT(localFilter4) =0, TRUE, FALSE);
			OUTPUT(localFilter4_converge,NAMED('localFilter4_converge'));
			OUTPUT(localFilter4,NAMED('localFilter4_result'));

			//update the Ub of data points who do not change their best centroid		
			unChangedUbSet4 := JOIN(ub_input4, localFilter4, LEFT.x = RIGHT.x, TRANSFORM(LEFT), LEFT ONLY);
			OUTPUT(unChangedUbSet4, NAMED('unChangedUbSet4'));	
			unChangedUbD014 := JOIN(unChangedUbSet4, d01, LEFT.x = RIGHT.id, TRANSFORM(tempR, SELF.cid := LEFT.y; SELF := RIGHT; ));
			OUTPUT(unChangedUbD014, NAMED('unChangedUbD014'));	
			unChangedUbD01CTemp4 := SORT(JOIN(unChangedUbD014, dCentroid4, LEFT.cid = RIGHT.id AND LEFT.number = RIGHT.number , TRANSFORM(Mat.Types.Element, SELF.value := (LEFT.value-RIGHT.value)*(LEFT.value-RIGHT.value); SELF.y := RIGHT.id;SELF.x := LEFT.id;)),x);
			OUTPUT(unChangedUbD01CTemp4, NAMED('unChangedUbD01CTemp4'));		
			

			unChangedUbRolled4:=ROLLUP(unChangedUbD01CTemp4,TRANSFORM(Mat.Types.Element,SELF.value:=SQRT(LEFT.value+RIGHT.value);SELF := LEFT;),x);
			OUTPUT(unChangedUbRolled4, NAMED('unChangedUbRolled4'));		
			
			//update the Ub of of data points who change their best centroid 		
			changedUb4 := localFilter4;
			OUTPUT(changedUb4, NAMED('changedUb4'));
			unchangedUb4:= unChangedUbRolled4;
			//new Ub
			ub4:= SORT(changedUb4 + unchangedUb4, x);
			OUTPUT(ub4, NAMED('ub4'));
			
			
			//update lbs	
			changelbsTemp4 := JOIN(localFilter4,dDistances4, LEFT.x = RIGHT.x, TRANSFORM(RIGHT));
			OUTPUT(changelbsTemp4, NAMED('changelbsTemp4'));
			changelbsTemp44 := SORT(JOIN(localFilter4,changelbsTemp4, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			OUTPUT(changelbsTemp44, NAMED('changelbsTemp44'));
			//new lbs for data points who change their best centroid
			changelbs4 := DEDUP(changelbsTemp44,x );
			OUTPUT(changelbs4,NAMED('changelbs4')) ;
			//new lbs
			lbs4 := JOIN(lbs4_temp, changelbs4,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value); SELF := LEFT;), LEFT OUTER );
			OUTPUT(lbs4,NAMED('lbs4'));
			
			//update Vin and Vout
			//calculate the Vin
			VinSet4 := SORT(localFilter4,y);
			OUTPUT(VinSet4, NAMED('VinSet4'));
			dClusterCountsVin4:=TABLE(VinSet4, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVin4,NAMED('dClusterCountsVin4'));
			dClusteredVin4 := JOIN(d01, VinSet4, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVin4,NAMED('dClusteredVin4'));
			dRolledVin4:=ROLLUP(SORT(dClusteredVin4,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVin4, NAMED('dRolledVin4'));


			VoutSet4 := JOIN(ub4_temp, localFilter4, LEFT.x = RIGHT.x, TRANSFORM(LEFT));
			OUTPUT(VoutSet4,NAMED('VoutSet4'));
			dClusterCountsVout4:=TABLE(VoutSet4, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVout4,NAMED('dClusterCountsVout4'));
			dClusteredVout4 := JOIN(d01,VoutSet4, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVout4,NAMED('dClusteredVout4'));
			dRolledVout4:=ROLLUP(SORT(dClusteredVout4,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVout4,NAMED('dRolledVout4'));

			
			V4Temp :=JOIN(V_input4,dClusterCountsVin4, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y +RIGHT.c; SELF.value := LEFT.value + RIGHT.c; SELF := LEFT;), LEFT OUTER );
			OUTPUT(V4Temp,NAMED('V4Temp'));
			V4:=JOIN(V4Temp,dClusterCountsVout4, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y -RIGHT.c;SELF.value := LEFT.value -RIGHT.c; SELF := LEFT;), LEFT OUTER);
			OUTPUT(V4,NAMED('V4'));
			

			//let the old |c*V| to multiply old Vcount
			CV4 :=JOIN(dCentroid4, V_input4, LEFT.id = RIGHT.x, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value * RIGHT.y; SELF := LEFT), LEFT OUTER);
			OUTPUT( CV4, NAMED('CV4')) ;

			//Add Vin
			CV4Vin := JOIN(CV4, dRolledVin4, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value + RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV4Vin, NAMED('CV4Vin'));

			//Minus Vout
			CV4VinVout:= JOIN(CV4Vin, dRolledVout4, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value - RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV4VinVout , NAMED('CV4VinVout'));
		
			//get the new C
			newC4:=JOIN(CV4VinVout,V4,LEFT.id=RIGHT.x,TRANSFORM(RECORDOF(LEFT),SELF.value:=(LEFT.value/RIGHT.y);SELF:=LEFT;), LEFT OUTER);
			OUTPUT(newC4, NAMED('C5'));
			
			dAddedx4Temp := JOIN(dAddedx_input4, newC4, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
			OUTPUT(dAddedx4Temp, NAMED('dAddedx4Temp'));
			dAddedx4 := PROJECT(dAddedx4Temp, TRANSFORM(lInput,SELF.id := 1;SELF.values:=LEFT.values;SELF.y := LEFT.number; SELF.x:=LEFT.id;));
			dAddedUb4 := JOIN(iUb_input4, ub4, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 2;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedUb4, NAMED('dAddedUb4'));
			dAddedLbs4 := JOIN(iLbs_input4, lbs4, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 3;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedLbs4, NAMED('dAddedLbs4'));
			dAddedV4 := JOIN(iV_input4, V4, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 4;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedV4, NAMED('dAddedV4'));
			

			outputSet4 := dAddedx4+ dAddedUb4 + dAddedLbs4 +dAddedV4 ;
			SORT(outputSet4,id);		
			
			
			

d5 := outputSet4;
c5:=5;
//********************************************start iterations*************************************************
// lInput yyfIterate(DATASET(lInput) d,UNSIGNED c):=FUNCTION
			
			dAddedx_input5 := PROJECT(d5(id = 1), TRANSFORM(lIterations , SELF.id := LEFT. x; SELF.number:= LEFT.y; SELF.values := LEFT.values;));
			iUb_input5 := TABLE(d5(id = 2), {x;y;values;});
			iLbs_input5 := TABLE(d5(id = 3), {x;y;values;});
			iV_input5 := TABLE(d5(id = 4), {x;y;values;});
			
			ub_input5 := PROJECT(d5(id = 2), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c5]; SELF := LEFT;));
			OUTPUT(ub_input5, NAMED('ub_input5'));
			lbs_input5 := PROJECT(d5(id = 3), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c5]; SELF := LEFT;));
			OUTPUT(lbs_input5, NAMED('lbs_input5'));
			V_input5 := PROJECT(d5(id = 4), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c5]; SELF := LEFT;));
			OUTPUT(V_input5, NAMED('V_input5'));
			
			//calculate the deltaC
			deltac5 := dDistanceDelta(c5,c5-1,dAddedx_input5);
			OUTPUT(deltac5,NAMED('deltac5'));
			
			bConverged5 := IF(MAX(deltac5,value)<=nConverge,TRUE,FALSE);
			OUTPUT(bConverged5, NAMED('iteration5_converge'));
			
			dCentroid5 := PROJECT(dAddedx_input5,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c5+1];SELF:=LEFT;));
			OUTPUT(dCentroid5,NAMED('dCentroid5'));

			//get deltaG1
			//group deltacg by Gt: first JOIN(deltac5 with Gt) then group by Gt
			deltacg5 :=JOIN(deltac5, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
			OUTPUT(deltacg5, NAMED('delatcg5'));
			deltacGt5 := SORT(deltacg5, y,value);
			output(deltacGt5,NAMED('deltacGt5'));
			deltaG5 := DEDUP(deltacGt5,y, RIGHT);
			OUTPUT(deltaG5,NAMED('deltaG5')); 

			//update ub_input5 and lbs_input5
			ub5_temp := SORT(JOIN(ub_input5, deltac5, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;)),x);
			OUTPUT(ub5_temp, NAMED('ub5_temp'));
			lbs5_temp := JOIN(lbs_input5, deltaG5, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := LEFT.value - RIGHT.value; SELF := LEFT));
			OUTPUT(lbs5_temp, NAMED('lbs5_temp'));
			
			
			// groupfilter1 testing on one time comparison
			groupFilterTemp5 := JOIN(ub_input5, lbs5_temp,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
			OUTPUT(groupFilterTemp5,NAMED('groupFilterTemp5'));
			groupFilter5:= groupFilterTemp5(value <0);
			OUTPUT(groupFilter5,NAMED('groupFilter5_comparison'));
			
			pGroupFilter5 := groupFilter5;
			OUTPUT(pGroupFilter5,NAMED('groupFilter5'));
			groupFilter5_Converge := IF(COUNT(pGroupFilter5)=0, TRUE, FALSE);
			OUTPUT(groupFilter5_Converge, NAMED('groupFilter5_Converge'));
			OUTPUT(pGroupFilter5,NAMED('groupFilter5_result')); 

			unpGroupFilter5 := groupFilterTemp5(value >=0);

			//localFilter
			changedSetTemp5 := JOIN(d01,pGroupFilter5, LEFT.id = RIGHT.x,TRANSFORM(LEFT));
			// changedSetTemp5;
			dDistances5 := ML.Cluster.Distances(changedSetTemp5,dCentroid5);
			OUTPUT(dDistances5,NAMED('dDistances5'));
			dClosest5 := ML.Cluster.Closest(dDistances5);
			OUTPUT(dClosest5,NAMED('distancesOfLocalfilter5')); 

			//the dataset that change their best centroid = result of local fitler		
			localFilter5 := JOIN(dClosest5, ub_input5, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, TRANSFORM(LEFT));

			//localFilter5 is empty then it's converged. Or update the ub and lb.
			localFilter5_converge := IF(COUNT(localFilter5) =0, TRUE, FALSE);
			OUTPUT(localFilter5_converge,NAMED('localFilter5_converge'));
			OUTPUT(localFilter5,NAMED('localFilter5_result'));

			//update the Ub of data points who do not change their best centroid		
			unChangedUbSet5 := JOIN(ub_input5, localFilter5, LEFT.x = RIGHT.x, TRANSFORM(LEFT), LEFT ONLY);
			OUTPUT(unChangedUbSet5, NAMED('unChangedUbSet5'));	
			unChangedUbD015 := JOIN(unChangedUbSet5, d01, LEFT.x = RIGHT.id, TRANSFORM(tempR, SELF.cid := LEFT.y; SELF := RIGHT; ));
			OUTPUT(unChangedUbD015, NAMED('unChangedUbD015'));	
			unChangedUbD01CTemp5 := SORT(JOIN(unChangedUbD015, dCentroid5, LEFT.cid = RIGHT.id AND LEFT.number = RIGHT.number , TRANSFORM(Mat.Types.Element, SELF.value := (LEFT.value-RIGHT.value)*(LEFT.value-RIGHT.value); SELF.y := RIGHT.id;SELF.x := LEFT.id;)),x);
			OUTPUT(unChangedUbD01CTemp5, NAMED('unChangedUbD01CTemp5'));		
			

			unChangedUbRolled5:=ROLLUP(unChangedUbD01CTemp5,TRANSFORM(Mat.Types.Element,SELF.value:=SQRT(LEFT.value+RIGHT.value);SELF := LEFT;),x);
			OUTPUT(unChangedUbRolled5, NAMED('unChangedUbRolled5'));		
			
			//update the Ub of of data points who change their best centroid 		
			changedUb5 := localFilter5;
			OUTPUT(changedUb5, NAMED('changedUb5'));
			unchangedUb5:= unChangedUbRolled5;
			//new Ub
			ub5:= SORT(changedUb5 + unchangedUb5, x);
			OUTPUT(ub5, NAMED('ub5'));
			
			
			//update lbs	
			changelbsTemp5 := JOIN(localFilter5,dDistances5, LEFT.x = RIGHT.x, TRANSFORM(RIGHT));
			OUTPUT(changelbsTemp5, NAMED('changelbsTemp5'));
			changelbsTemp55 := SORT(JOIN(localFilter5,changelbsTemp5, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			OUTPUT(changelbsTemp55, NAMED('changelbsTemp55'));
			//new lbs for data points who change their best centroid
			changelbs5 := DEDUP(changelbsTemp55,x );
			OUTPUT(changelbs5,NAMED('changelbs5')) ;
			//new lbs
			lbs5 := JOIN(lbs5_temp, changelbs5,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value); SELF := LEFT;), LEFT OUTER );
			OUTPUT(lbs5,NAMED('lbs5'));
			
			//update Vin and Vout
			//calculate the Vin
			VinSet5 := SORT(localFilter5,y);
			OUTPUT(VinSet5, NAMED('VinSet5'));
			dClusterCountsVin5:=TABLE(VinSet5, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVin5,NAMED('dClusterCountsVin5'));
			dClusteredVin5 := JOIN(d01, VinSet5, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVin5,NAMED('dClusteredVin5'));
			dRolledVin5:=ROLLUP(SORT(dClusteredVin5,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVin5, NAMED('dRolledVin5'));


			VoutSet5 := JOIN(ub5_temp, localFilter5, LEFT.x = RIGHT.x, TRANSFORM(LEFT));
			OUTPUT(VoutSet5,NAMED('VoutSet5'));
			dClusterCountsVout5:=TABLE(VoutSet5, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVout5,NAMED('dClusterCountsVout5'));
			dClusteredVout5 := JOIN(d01,VoutSet5, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVout5,NAMED('dClusteredVout5'));
			dRolledVout5:=ROLLUP(SORT(dClusteredVout5,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVout5,NAMED('dRolledVout5'));

			
			V5Temp :=JOIN(V_input5,dClusterCountsVin5, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y +RIGHT.c; SELF.value := LEFT.value + RIGHT.c; SELF := LEFT;), LEFT OUTER );
			OUTPUT(V5Temp,NAMED('V5Temp'));
			V5:=JOIN(V5Temp,dClusterCountsVout5, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y -RIGHT.c;SELF.value := LEFT.value -RIGHT.c; SELF := LEFT;), LEFT OUTER);
			OUTPUT(V5,NAMED('V5'));
			

			//let the old |c*V| to multiply old Vcount
			CV5 :=JOIN(dCentroid5, V_input5, LEFT.id = RIGHT.x, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value * RIGHT.y; SELF := LEFT), LEFT OUTER);
			OUTPUT( CV5, NAMED('CV5')) ;

			//Add Vin
			CV5Vin := JOIN(CV5, dRolledVin5, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value + RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV5Vin, NAMED('CV5Vin'));

			//Minus Vout
			CV5VinVout:= JOIN(CV5Vin, dRolledVout5, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value - RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV5VinVout , NAMED('CV5VinVout'));
		
			//get the new C
			newC5:=JOIN(CV5VinVout,V5,LEFT.id=RIGHT.x,TRANSFORM(RECORDOF(LEFT),SELF.value:=(LEFT.value/RIGHT.y);SELF:=LEFT;), LEFT OUTER);
			OUTPUT(newC5, NAMED('C6'));
			
			dAddedx5Temp := JOIN(dAddedx_input5, newC5, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
			OUTPUT(dAddedx5Temp, NAMED('dAddedx5Temp'));
			dAddedx5 := PROJECT(dAddedx5Temp, TRANSFORM(lInput,SELF.id := 1;SELF.values:=LEFT.values;SELF.y := LEFT.number; SELF.x:=LEFT.id;));
			dAddedUb5 := JOIN(iUb_input5, ub5, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 2;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedUb5, NAMED('dAddedUb5'));
			dAddedLbs5 := JOIN(iLbs_input5, lbs5, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 3;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedLbs5, NAMED('dAddedLbs5'));
			dAddedV5 := JOIN(iV_input5, V5, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 4;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedV5, NAMED('dAddedV5'));
			

			outputSet5 := dAddedx5+ dAddedUb5 + dAddedLbs5 +dAddedV5 ;
			SORT(outputSet5,id);		
//iteration6

d6 := outputSet5;
c6:=6;
//********************************************start iterations*************************************************
// lInput yyfIterate(DATASET(lInput) d,UNSIGNED c):=FUNCTION
			
			dAddedx_input6 := PROJECT(d6(id = 1), TRANSFORM(lIterations , SELF.id := LEFT. x; SELF.number:= LEFT.y; SELF.values := LEFT.values;));
			iUb_input6 := TABLE(d6(id = 2), {x;y;values;});
			iLbs_input6 := TABLE(d6(id = 3), {x;y;values;});
			iV_input6 := TABLE(d6(id = 4), {x;y;values;});
			
			ub_input6 := PROJECT(d6(id = 2), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c6]; SELF := LEFT;));
			OUTPUT(ub_input6, NAMED('ub_input6'));
			lbs_input6 := PROJECT(d6(id = 3), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c6]; SELF := LEFT;));
			OUTPUT(lbs_input6, NAMED('lbs_input6'));
			V_input6 := PROJECT(d6(id = 4), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c6]; SELF := LEFT;));
			OUTPUT(V_input6, NAMED('V_input6'));
			
			//calculate the deltaC
			deltac6 := dDistanceDelta(c6,c6-1,dAddedx_input6);
			OUTPUT(deltac6,NAMED('deltac6'));
			
			bConverged6 := IF(MAX(deltac6,value)<=nConverge,TRUE,FALSE);
			OUTPUT(bConverged6, NAMED('iteration6_converge'));
			
			dCentroid6 := PROJECT(dAddedx_input6,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c6+1];SELF:=LEFT;));
			OUTPUT(dCentroid6,NAMED('dCentroid6'));

			//get deltaG1
			//group deltacg by Gt: first JOIN(deltac6 with Gt) then group by Gt
			deltacg6 :=JOIN(deltac6, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
			OUTPUT(deltacg6, NAMED('delatcg6'));
			deltacGt6 := SORT(deltacg6, y,value);
			output(deltacGt6,NAMED('deltacGt6'));
			deltaG6 := DEDUP(deltacGt6,y, RIGHT);
			OUTPUT(deltaG6,NAMED('deltaG6')); 

			//update ub_input6 and lbs_input6
			ub6_temp := SORT(JOIN(ub_input6, deltac6, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;)),x);
			OUTPUT(ub6_temp, NAMED('ub6_temp'));
			lbs6_temp := JOIN(lbs_input6, deltaG6, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := LEFT.value - RIGHT.value; SELF := LEFT));
			OUTPUT(lbs6_temp, NAMED('lbs6_temp'));
			
			
			// groupfilter1 testing on one time comparison
			groupFilterTemp6 := JOIN(ub_input6, lbs6_temp,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
			OUTPUT(groupFilterTemp6,NAMED('groupFilterTemp6'));
			groupFilter6:= groupFilterTemp6(value <0);
			OUTPUT(groupFilter6,NAMED('groupFilter6_comparison'));
			
			pGroupFilter6 := groupFilter6;
			OUTPUT(pGroupFilter6,NAMED('groupFilter6'));
			groupFilter6_Converge := IF(COUNT(pGroupFilter6)=0, TRUE, FALSE);
			OUTPUT(groupFilter6_Converge, NAMED('groupFilter6_Converge'));
			OUTPUT(pGroupFilter6,NAMED('groupFilter6_result')); 

			unpGroupFilter6 := groupFilterTemp6(value >=0);

			//localFilter
			changedSetTemp6 := JOIN(d01,pGroupFilter6, LEFT.id = RIGHT.x,TRANSFORM(LEFT));
			// changedSetTemp6;
			dDistances6 := ML.Cluster.Distances(changedSetTemp6,dCentroid6);
			OUTPUT(dDistances6,NAMED('dDistances6'));
			dClosest6 := ML.Cluster.Closest(dDistances6);
			OUTPUT(dClosest6,NAMED('distancesOfLocalfilter6')); 

			//the dataset that change their best centroid = result of local fitler		
			localFilter6 := JOIN(dClosest6, ub_input6, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, TRANSFORM(LEFT));

			//localFilter6 is empty then it's converged. Or update the ub and lb.
			localFilter6_converge := IF(COUNT(localFilter6) =0, TRUE, FALSE);
			OUTPUT(localFilter6_converge,NAMED('localFilter6_converge'));
			OUTPUT(localFilter6,NAMED('localFilter6_result'));

			//update the Ub of data points who do not change their best centroid		
			unChangedUbSet6 := JOIN(ub_input6, localFilter6, LEFT.x = RIGHT.x, TRANSFORM(LEFT), LEFT ONLY);
			OUTPUT(unChangedUbSet6, NAMED('unChangedUbSet6'));	
			unChangedUbD016 := JOIN(unChangedUbSet6, d01, LEFT.x = RIGHT.id, TRANSFORM(tempR, SELF.cid := LEFT.y; SELF := RIGHT; ));
			OUTPUT(unChangedUbD016, NAMED('unChangedUbD016'));	
			unChangedUbD01CTemp6 := SORT(JOIN(unChangedUbD016, dCentroid6, LEFT.cid = RIGHT.id AND LEFT.number = RIGHT.number , TRANSFORM(Mat.Types.Element, SELF.value := (LEFT.value-RIGHT.value)*(LEFT.value-RIGHT.value); SELF.y := RIGHT.id;SELF.x := LEFT.id;)),x);
			OUTPUT(unChangedUbD01CTemp6, NAMED('unChangedUbD01CTemp6'));		
			

			unChangedUbRolled6:=ROLLUP(unChangedUbD01CTemp6,TRANSFORM(Mat.Types.Element,SELF.value:=SQRT(LEFT.value+RIGHT.value);SELF := LEFT;),x);
			OUTPUT(unChangedUbRolled6, NAMED('unChangedUbRolled6'));		
			
			//update the Ub of of data points who change their best centroid 		
			changedUb6 := localFilter6;
			OUTPUT(changedUb6, NAMED('changedUb6'));
			unchangedUb6:= unChangedUbRolled6;
			//new Ub
			ub6:= SORT(changedUb6 + unchangedUb6, x);
			OUTPUT(ub6, NAMED('ub6'));
			
			
			//update lbs	
			changelbsTemp6 := JOIN(localFilter6,dDistances6, LEFT.x = RIGHT.x, TRANSFORM(RIGHT));
			OUTPUT(changelbsTemp6, NAMED('changelbsTemp6'));
			changelbsTemp66 := SORT(JOIN(localFilter6,changelbsTemp6, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			OUTPUT(changelbsTemp66, NAMED('changelbsTemp66'));
			//new lbs for data points who change their best centroid
			changelbs6 := DEDUP(changelbsTemp66,x );
			OUTPUT(changelbs6,NAMED('changelbs6')) ;
			//new lbs
			lbs6 := JOIN(lbs6_temp, changelbs6,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value); SELF := LEFT;), LEFT OUTER );
			OUTPUT(lbs6,NAMED('lbs6'));
			
			//update Vin and Vout
			//calculate the Vin
			VinSet6 := SORT(localFilter6,y);
			OUTPUT(VinSet6, NAMED('VinSet6'));
			dClusterCountsVin6:=TABLE(VinSet6, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVin6,NAMED('dClusterCountsVin6'));
			dClusteredVin6 := JOIN(d01, VinSet6, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVin6,NAMED('dClusteredVin6'));
			dRolledVin6:=ROLLUP(SORT(dClusteredVin6,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVin6, NAMED('dRolledVin6'));


			VoutSet6 := JOIN(ub6_temp, localFilter6, LEFT.x = RIGHT.x, TRANSFORM(LEFT));
			OUTPUT(VoutSet6,NAMED('VoutSet6'));
			dClusterCountsVout6:=TABLE(VoutSet6, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVout6,NAMED('dClusterCountsVout6'));
			dClusteredVout6 := JOIN(d01,VoutSet6, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVout6,NAMED('dClusteredVout6'));
			dRolledVout6:=ROLLUP(SORT(dClusteredVout6,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVout6,NAMED('dRolledVout6'));

			
			V6Temp :=JOIN(V_input6,dClusterCountsVin6, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y +RIGHT.c; SELF.value := LEFT.value + RIGHT.c; SELF := LEFT;), LEFT OUTER );
			OUTPUT(V6Temp,NAMED('V6Temp'));
			V6:=JOIN(V6Temp,dClusterCountsVout6, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y -RIGHT.c;SELF.value := LEFT.value -RIGHT.c; SELF := LEFT;), LEFT OUTER);
			OUTPUT(V6,NAMED('V6'));
			

			//let the old |c*V| to multiply old Vcount
			CV6 :=JOIN(dCentroid6, V_input6, LEFT.id = RIGHT.x, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value * RIGHT.y; SELF := LEFT), LEFT OUTER);
			OUTPUT( CV6, NAMED('CV6')) ;

			//Add Vin
			CV6Vin := JOIN(CV6, dRolledVin6, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value + RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV6Vin, NAMED('CV6Vin'));

			//Minus Vout
			CV6VinVout:= JOIN(CV6Vin, dRolledVout6, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value - RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV6VinVout , NAMED('CV6VinVout'));
		
			//get the new C
			newC6:=JOIN(CV6VinVout,V6,LEFT.id=RIGHT.x,TRANSFORM(RECORDOF(LEFT),SELF.value:=(LEFT.value/RIGHT.y);SELF:=LEFT;), LEFT OUTER);
			OUTPUT(newC6, NAMED('C7'));
			
			dAddedx6Temp := JOIN(dAddedx_input6, newC6, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
			OUTPUT(dAddedx6Temp, NAMED('dAddedx6Temp'));
			dAddedx6 := PROJECT(dAddedx6Temp, TRANSFORM(lInput,SELF.id := 1;SELF.values:=LEFT.values;SELF.y := LEFT.number; SELF.x:=LEFT.id;));
			dAddedUb6 := JOIN(iUb_input6, ub6, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 2;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedUb6, NAMED('dAddedUb6'));
			dAddedLbs6 := JOIN(iLbs_input6, lbs6, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 3;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedLbs6, NAMED('dAddedLbs6'));
			dAddedV6 := JOIN(iV_input6, V6, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 4;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedV6, NAMED('dAddedV6'));
			

			outputSet6 := dAddedx6+ dAddedUb6 + dAddedLbs6 +dAddedV6 ;
			SORT(outputSet6,id);		

//iteration7

d7 := outputSet6;
c7:=7;
//********************************************start iterations*************************************************
// lInput yyfIterate(DATASET(lInput) d,UNSIGNED c):=FUNCTION
			
			dAddedx_input7 := PROJECT(d7(id = 1), TRANSFORM(lIterations , SELF.id := LEFT. x; SELF.number:= LEFT.y; SELF.values := LEFT.values;));
			iUb_input7 := TABLE(d7(id = 2), {x;y;values;});
			iLbs_input7 := TABLE(d7(id = 3), {x;y;values;});
			iV_input7 := TABLE(d7(id = 4), {x;y;values;});
			
			ub_input7 := PROJECT(d7(id = 2), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c7]; SELF := LEFT;));
			OUTPUT(ub_input7, NAMED('ub_input7'));
			lbs_input7 := PROJECT(d7(id = 3), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c7]; SELF := LEFT;));
			OUTPUT(lbs_input7, NAMED('lbs_input7'));
			V_input7 := PROJECT(d7(id = 4), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c7]; SELF := LEFT;));
			OUTPUT(V_input7, NAMED('V_input7'));
			
			//calculate the deltaC
			deltac7 := dDistanceDelta(c7,c7-1,dAddedx_input7);
			OUTPUT(deltac7,NAMED('deltac7'));
			
			bConverged7 := IF(MAX(deltac7,value)<=nConverge,TRUE,FALSE);
			OUTPUT(bConverged7, NAMED('iteration7_converge'));
			
			dCentroid7 := PROJECT(dAddedx_input7,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c7+1];SELF:=LEFT;));
			OUTPUT(dCentroid7,NAMED('dCentroid7'));

			//get deltaG1
			//group deltacg by Gt: first JOIN(deltac7 with Gt) then group by Gt
			deltacg7 :=JOIN(deltac7, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
			OUTPUT(deltacg7, NAMED('delatcg7'));
			deltacGt7 := SORT(deltacg7, y,value);
			output(deltacGt7,NAMED('deltacGt7'));
			deltaG7 := DEDUP(deltacGt7,y, RIGHT);
			OUTPUT(deltaG7,NAMED('deltaG7')); 

			//update ub_input7 and lbs_input7
			ub7_temp := SORT(JOIN(ub_input7, deltac7, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;)),x);
			OUTPUT(ub7_temp, NAMED('ub7_temp'));
			lbs7_temp := JOIN(lbs_input7, deltaG7, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := LEFT.value - RIGHT.value; SELF := LEFT));
			OUTPUT(lbs7_temp, NAMED('lbs7_temp'));
			
			
			// groupfilter1 testing on one time comparison
			groupFilterTemp7 := JOIN(ub_input7, lbs7_temp,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
			OUTPUT(groupFilterTemp7,NAMED('groupFilterTemp7'));
			groupFilter7:= groupFilterTemp7(value <0);
			OUTPUT(groupFilter7,NAMED('groupFilter7_comparison'));
			
			pGroupFilter7 := groupFilter7;
			OUTPUT(pGroupFilter7,NAMED('groupFilter7'));
			groupFilter7_Converge := IF(COUNT(pGroupFilter7)=0, TRUE, FALSE);
			OUTPUT(groupFilter7_Converge, NAMED('groupFilter7_Converge'));
			OUTPUT(pGroupFilter7,NAMED('groupFilter7_result')); 

			unpGroupFilter7 := groupFilterTemp7(value >=0);

			//localFilter
			changedSetTemp7 := JOIN(d01,pGroupFilter7, LEFT.id = RIGHT.x,TRANSFORM(LEFT));
			// changedSetTemp7;
			dDistances7 := ML.Cluster.Distances(changedSetTemp7,dCentroid7);
			OUTPUT(dDistances7,NAMED('dDistances7'));
			dClosest7 := ML.Cluster.Closest(dDistances7);
			OUTPUT(dClosest7,NAMED('distancesOfLocalfilter7')); 

			//the dataset that change their best centroid = result of local fitler		
			localFilter7 := JOIN(dClosest7, ub_input7, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, TRANSFORM(LEFT));

			//localFilter7 is empty then it's converged. Or update the ub and lb.
			localFilter7_converge := IF(COUNT(localFilter7) =0, TRUE, FALSE);
			OUTPUT(localFilter7_converge,NAMED('localFilter7_converge'));
			OUTPUT(localFilter7,NAMED('localFilter7_result'));

			//update the Ub of data points who do not change their best centroid		
			unChangedUbSet7 := JOIN(ub_input7, localFilter7, LEFT.x = RIGHT.x, TRANSFORM(LEFT), LEFT ONLY);
			OUTPUT(unChangedUbSet7, NAMED('unChangedUbSet7'));	
			unChangedUbD017 := JOIN(unChangedUbSet7, d01, LEFT.x = RIGHT.id, TRANSFORM(tempR, SELF.cid := LEFT.y; SELF := RIGHT; ));
			OUTPUT(unChangedUbD017, NAMED('unChangedUbD017'));	
			unChangedUbD01CTemp7 := SORT(JOIN(unChangedUbD017, dCentroid7, LEFT.cid = RIGHT.id AND LEFT.number = RIGHT.number , TRANSFORM(Mat.Types.Element, SELF.value := (LEFT.value-RIGHT.value)*(LEFT.value-RIGHT.value); SELF.y := RIGHT.id;SELF.x := LEFT.id;)),x);
			OUTPUT(unChangedUbD01CTemp7, NAMED('unChangedUbD01CTemp7'));		
			

			unChangedUbRolled7:=ROLLUP(unChangedUbD01CTemp7,TRANSFORM(Mat.Types.Element,SELF.value:=SQRT(LEFT.value+RIGHT.value);SELF := LEFT;),x);
			OUTPUT(unChangedUbRolled7, NAMED('unChangedUbRolled7'));		
			
			//update the Ub of of data points who change their best centroid 		
			changedUb7 := localFilter7;
			OUTPUT(changedUb7, NAMED('changedUb7'));
			unchangedUb7:= unChangedUbRolled7;
			//new Ub
			ub7:= SORT(changedUb7 + unchangedUb7, x);
			OUTPUT(ub7, NAMED('ub7'));
			
			
			//update lbs	
			changelbsTemp7 := JOIN(localFilter7,dDistances7, LEFT.x = RIGHT.x, TRANSFORM(RIGHT));
			OUTPUT(changelbsTemp7, NAMED('changelbsTemp7'));
			changelbsTemp77 := SORT(JOIN(localFilter7,changelbsTemp7, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			OUTPUT(changelbsTemp77, NAMED('changelbsTemp77'));
			//new lbs for data points who change their best centroid
			changelbs7 := DEDUP(changelbsTemp77,x );
			OUTPUT(changelbs7,NAMED('changelbs7')) ;
			//new lbs
			lbs7 := JOIN(lbs7_temp, changelbs7,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value); SELF := LEFT;), LEFT OUTER );
			OUTPUT(lbs7,NAMED('lbs7'));
			
			//update Vin and Vout
			//calculate the Vin
			VinSet7 := SORT(localFilter7,y);
			OUTPUT(VinSet7, NAMED('VinSet7'));
			dClusterCountsVin7:=TABLE(VinSet7, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVin7,NAMED('dClusterCountsVin7'));
			dClusteredVin7 := JOIN(d01, VinSet7, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVin7,NAMED('dClusteredVin7'));
			dRolledVin7:=ROLLUP(SORT(dClusteredVin7,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVin7, NAMED('dRolledVin7'));


			VoutSet7 := JOIN(ub7_temp, localFilter7, LEFT.x = RIGHT.x, TRANSFORM(LEFT));
			OUTPUT(VoutSet7,NAMED('VoutSet7'));
			dClusterCountsVout7:=TABLE(VoutSet7, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVout7,NAMED('dClusterCountsVout7'));
			dClusteredVout7 := JOIN(d01,VoutSet7, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVout7,NAMED('dClusteredVout7'));
			dRolledVout7:=ROLLUP(SORT(dClusteredVout7,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVout7,NAMED('dRolledVout7'));

			
			V7Temp :=JOIN(V_input7,dClusterCountsVin7, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y +RIGHT.c; SELF.value := LEFT.value + RIGHT.c; SELF := LEFT;), LEFT OUTER );
			OUTPUT(V7Temp,NAMED('V7Temp'));
			V7:=JOIN(V7Temp,dClusterCountsVout7, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y -RIGHT.c;SELF.value := LEFT.value -RIGHT.c; SELF := LEFT;), LEFT OUTER);
			OUTPUT(V7,NAMED('V7'));
			

			//let the old |c*V| to multiply old Vcount
			CV7 :=JOIN(dCentroid7, V_input7, LEFT.id = RIGHT.x, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value * RIGHT.y; SELF := LEFT), LEFT OUTER);
			OUTPUT( CV7, NAMED('CV7')) ;

			//Add Vin
			CV7Vin := JOIN(CV7, dRolledVin7, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value + RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV7Vin, NAMED('CV7Vin'));

			//Minus Vout
			CV7VinVout:= JOIN(CV7Vin, dRolledVout7, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value - RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV7VinVout , NAMED('CV7VinVout'));
		
			//get the new C
			newC7:=JOIN(CV7VinVout,V7,LEFT.id=RIGHT.x,TRANSFORM(RECORDOF(LEFT),SELF.value:=(LEFT.value/RIGHT.y);SELF:=LEFT;), LEFT OUTER);
			OUTPUT(newC7, NAMED('C8'));
			
			dAddedx7Temp := JOIN(dAddedx_input7, newC7, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
			OUTPUT(dAddedx7Temp, NAMED('dAddedx7Temp'));
			dAddedx7 := PROJECT(dAddedx7Temp, TRANSFORM(lInput,SELF.id := 1;SELF.values:=LEFT.values;SELF.y := LEFT.number; SELF.x:=LEFT.id;));
			dAddedUb7 := JOIN(iUb_input7, ub7, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 2;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedUb7, NAMED('dAddedUb7'));
			dAddedLbs7 := JOIN(iLbs_input7, lbs7, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 3;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedLbs7, NAMED('dAddedLbs7'));
			dAddedV7 := JOIN(iV_input7, V7, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 4;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedV7, NAMED('dAddedV7'));
			

			outputSet7 := dAddedx7+ dAddedUb7 + dAddedLbs7 +dAddedV7 ;
			SORT(outputSet7,id);		


//iteration8

d8 := outputSet7;
c8:=8;
//********************************************start iterations*************************************************
// lInput yyfIterate(DATASET(lInput) d,UNSIGNED c):=FUNCTION
			
			dAddedx_input8 := PROJECT(d8(id = 1), TRANSFORM(lIterations , SELF.id := LEFT. x; SELF.number:= LEFT.y; SELF.values := LEFT.values;));
			iUb_input8 := TABLE(d8(id = 2), {x;y;values;});
			iLbs_input8 := TABLE(d8(id = 3), {x;y;values;});
			iV_input8 := TABLE(d8(id = 4), {x;y;values;});
			
			ub_input8 := PROJECT(d8(id = 2), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c8]; SELF := LEFT;));
			OUTPUT(ub_input8, NAMED('ub_input8'));
			lbs_input8 := PROJECT(d8(id = 3), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c8]; SELF := LEFT;));
			OUTPUT(lbs_input8, NAMED('lbs_input8'));
			V_input8 := PROJECT(d8(id = 4), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c8]; SELF := LEFT;));
			OUTPUT(V_input8, NAMED('V_input8'));
			
			//calculate the deltaC
			deltac8 := dDistanceDelta(c8,c8-1,dAddedx_input8);
			OUTPUT(deltac8,NAMED('deltac8'));
			
			bConverged8 := IF(MAX(deltac8,value)<=nConverge,TRUE,FALSE);
			OUTPUT(bConverged8, NAMED('iteration8_converge'));
			
			dCentroid8 := PROJECT(dAddedx_input8,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c8+1];SELF:=LEFT;));
			OUTPUT(dCentroid8,NAMED('dCentroid8'));

			//get deltaG1
			//group deltacg by Gt: first JOIN(deltac8 with Gt) then group by Gt
			deltacg8 :=JOIN(deltac8, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
			OUTPUT(deltacg8, NAMED('delatcg8'));
			deltacGt8 := SORT(deltacg8, y,value);
			output(deltacGt8,NAMED('deltacGt8'));
			deltaG8 := DEDUP(deltacGt8,y, RIGHT);
			OUTPUT(deltaG8,NAMED('deltaG8')); 

			//update ub_input8 and lbs_input8
			ub8_temp := SORT(JOIN(ub_input8, deltac8, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;)),x);
			OUTPUT(ub8_temp, NAMED('ub8_temp'));
			lbs8_temp := JOIN(lbs_input8, deltaG8, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := LEFT.value - RIGHT.value; SELF := LEFT));
			OUTPUT(lbs8_temp, NAMED('lbs8_temp'));
			
			
			// groupfilter1 testing on one time comparison
			groupFilterTemp8 := JOIN(ub_input8, lbs8_temp,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
			OUTPUT(groupFilterTemp8,NAMED('groupFilterTemp8'));
			groupFilter8:= groupFilterTemp8(value <0);
			OUTPUT(groupFilter8,NAMED('groupFilter8_comparison'));
			
			pGroupFilter8 := groupFilter8;
			OUTPUT(pGroupFilter8,NAMED('groupFilter8'));
			groupFilter8_Converge := IF(COUNT(pGroupFilter8)=0, TRUE, FALSE);
			OUTPUT(groupFilter8_Converge, NAMED('groupFilter8_Converge'));
			OUTPUT(pGroupFilter8,NAMED('groupFilter8_result')); 

			unpGroupFilter8 := groupFilterTemp8(value >=0);

			//localFilter
			changedSetTemp8 := JOIN(d01,pGroupFilter8, LEFT.id = RIGHT.x,TRANSFORM(LEFT));
			// changedSetTemp8;
			dDistances8 := ML.Cluster.Distances(changedSetTemp8,dCentroid8);
			OUTPUT(dDistances8,NAMED('dDistances8'));
			dClosest8 := ML.Cluster.Closest(dDistances8);
			OUTPUT(dClosest8,NAMED('distancesOfLocalfilter8')); 

			//the dataset that change their best centroid = result of local fitler		
			localFilter8 := JOIN(dClosest8, ub_input8, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, TRANSFORM(LEFT));

			//localFilter8 is empty then it's converged. Or update the ub and lb.
			localFilter8_converge := IF(COUNT(localFilter8) =0, TRUE, FALSE);
			OUTPUT(localFilter8_converge,NAMED('localFilter8_converge'));
			OUTPUT(localFilter8,NAMED('localFilter8_result'));

			//update the Ub of data points who do not change their best centroid		
			unChangedUbSet8 := JOIN(ub_input8, localFilter8, LEFT.x = RIGHT.x, TRANSFORM(LEFT), LEFT ONLY);
			OUTPUT(unChangedUbSet8, NAMED('unChangedUbSet8'));	
			unChangedUbD018 := JOIN(unChangedUbSet8, d01, LEFT.x = RIGHT.id, TRANSFORM(tempR, SELF.cid := LEFT.y; SELF := RIGHT; ));
			OUTPUT(unChangedUbD018, NAMED('unChangedUbD018'));	
			unChangedUbD01CTemp8 := SORT(JOIN(unChangedUbD018, dCentroid8, LEFT.cid = RIGHT.id AND LEFT.number = RIGHT.number , TRANSFORM(Mat.Types.Element, SELF.value := (LEFT.value-RIGHT.value)*(LEFT.value-RIGHT.value); SELF.y := RIGHT.id;SELF.x := LEFT.id;)),x);
			OUTPUT(unChangedUbD01CTemp8, NAMED('unChangedUbD01CTemp8'));		
			

			unChangedUbRolled8:=ROLLUP(unChangedUbD01CTemp8,TRANSFORM(Mat.Types.Element,SELF.value:=SQRT(LEFT.value+RIGHT.value);SELF := LEFT;),x);
			OUTPUT(unChangedUbRolled8, NAMED('unChangedUbRolled8'));		
			
			//update the Ub of of data points who change their best centroid 		
			changedUb8 := localFilter8;
			OUTPUT(changedUb8, NAMED('changedUb8'));
			unchangedUb8:= unChangedUbRolled8;
			//new Ub
			ub8:= SORT(changedUb8 + unchangedUb8, x);
			OUTPUT(ub8, NAMED('ub8'));
			
			
			//update lbs	
			changelbsTemp8 := JOIN(localFilter8,dDistances8, LEFT.x = RIGHT.x, TRANSFORM(RIGHT));
			OUTPUT(changelbsTemp8, NAMED('changelbsTemp8'));
			changelbsTemp88 := SORT(JOIN(localFilter8,changelbsTemp8, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			OUTPUT(changelbsTemp88, NAMED('changelbsTemp88'));
			//new lbs for data points who change their best centroid
			changelbs8 := DEDUP(changelbsTemp88,x );
			OUTPUT(changelbs8,NAMED('changelbs8')) ;
			//new lbs
			lbs8 := JOIN(lbs8_temp, changelbs8,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value); SELF := LEFT;), LEFT OUTER );
			OUTPUT(lbs8,NAMED('lbs8'));
			
			//update Vin and Vout
			//calculate the Vin
			VinSet8 := SORT(localFilter8,y);
			OUTPUT(VinSet8, NAMED('VinSet8'));
			dClusterCountsVin8:=TABLE(VinSet8, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVin8,NAMED('dClusterCountsVin8'));
			dClusteredVin8 := JOIN(d01, VinSet8, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVin8,NAMED('dClusteredVin8'));
			dRolledVin8:=ROLLUP(SORT(dClusteredVin8,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVin8, NAMED('dRolledVin8'));


			VoutSet8 := JOIN(ub8_temp, localFilter8, LEFT.x = RIGHT.x, TRANSFORM(LEFT));
			OUTPUT(VoutSet8,NAMED('VoutSet8'));
			dClusterCountsVout8:=TABLE(VoutSet8, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVout8,NAMED('dClusterCountsVout8'));
			dClusteredVout8 := JOIN(d01,VoutSet8, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVout8,NAMED('dClusteredVout8'));
			dRolledVout8:=ROLLUP(SORT(dClusteredVout8,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVout8,NAMED('dRolledVout8'));

			
			V8Temp :=JOIN(V_input8,dClusterCountsVin8, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y +RIGHT.c; SELF.value := LEFT.value + RIGHT.c; SELF := LEFT;), LEFT OUTER );
			OUTPUT(V8Temp,NAMED('V8Temp'));
			V8:=JOIN(V8Temp,dClusterCountsVout8, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y -RIGHT.c;SELF.value := LEFT.value -RIGHT.c; SELF := LEFT;), LEFT OUTER);
			OUTPUT(V8,NAMED('V8'));
			

			//let the old |c*V| to multiply old Vcount
			CV8 :=JOIN(dCentroid8, V_input8, LEFT.id = RIGHT.x, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value * RIGHT.y; SELF := LEFT), LEFT OUTER);
			OUTPUT( CV8, NAMED('CV8')) ;

			//Add Vin
			CV8Vin := JOIN(CV8, dRolledVin8, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value + RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV8Vin, NAMED('CV8Vin'));

			//Minus Vout
			CV8VinVout:= JOIN(CV8Vin, dRolledVout8, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value - RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV8VinVout , NAMED('CV8VinVout'));
		
			//get the new C
			newC8:=JOIN(CV8VinVout,V8,LEFT.id=RIGHT.x,TRANSFORM(RECORDOF(LEFT),SELF.value:=(LEFT.value/RIGHT.y);SELF:=LEFT;), LEFT OUTER);
			OUTPUT(newC8, NAMED('C9'));
			
			dAddedx8Temp := JOIN(dAddedx_input8, newC8, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
			OUTPUT(dAddedx8Temp, NAMED('dAddedx8Temp'));
			dAddedx8 := PROJECT(dAddedx8Temp, TRANSFORM(lInput,SELF.id := 1;SELF.values:=LEFT.values;SELF.y := LEFT.number; SELF.x:=LEFT.id;));
			dAddedUb8 := JOIN(iUb_input8, ub8, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 2;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedUb8, NAMED('dAddedUb8'));
			dAddedLbs8 := JOIN(iLbs_input8, lbs8, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 3;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedLbs8, NAMED('dAddedLbs8'));
			dAddedV8 := JOIN(iV_input8, V8, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 4;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedV8, NAMED('dAddedV8'));
			

			outputSet8 := dAddedx8+ dAddedUb8 + dAddedLbs8 +dAddedV8 ;
			SORT(outputSet8,id);		

//iteration9

d9 := outputSet8;
c9:=9;
//********************************************start iterations*************************************************
// lInput yyfIterate(DATASET(lInput) d,UNSIGNED c):=FUNCTION
			
			dAddedx_input9 := PROJECT(d9(id = 1), TRANSFORM(lIterations , SELF.id := LEFT. x; SELF.number:= LEFT.y; SELF.values := LEFT.values;));
			iUb_input9 := TABLE(d9(id = 2), {x;y;values;});
			iLbs_input9 := TABLE(d9(id = 3), {x;y;values;});
			iV_input9 := TABLE(d9(id = 4), {x;y;values;});
			
			ub_input9 := PROJECT(d9(id = 2), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c9]; SELF := LEFT;));
			OUTPUT(ub_input9, NAMED('ub_input9'));
			lbs_input9 := PROJECT(d9(id = 3), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c9]; SELF := LEFT;));
			OUTPUT(lbs_input9, NAMED('lbs_input9'));
			V_input9 := PROJECT(d9(id = 4), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c9]; SELF := LEFT;));
			OUTPUT(V_input9, NAMED('V_input9'));
			
			//calculate the deltaC
			deltac9 := dDistanceDelta(c9,c9-1,dAddedx_input9);
			OUTPUT(deltac9,NAMED('deltac9'));
			
			bConverged9 := IF(MAX(deltac9,value)<=nConverge,TRUE,FALSE);
			OUTPUT(bConverged9, NAMED('iteration9_converge'));
			
			dCentroid9 := PROJECT(dAddedx_input9,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c9+1];SELF:=LEFT;));
			OUTPUT(dCentroid9,NAMED('dCentroid9'));

			//get deltaG1
			//group deltacg by Gt: first JOIN(deltac9 with Gt) then group by Gt
			deltacg9 :=JOIN(deltac9, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
			OUTPUT(deltacg9, NAMED('delatcg9'));
			deltacGt9 := SORT(deltacg9, y,value);
			output(deltacGt9,NAMED('deltacGt9'));
			deltaG9 := DEDUP(deltacGt9,y, RIGHT);
			OUTPUT(deltaG9,NAMED('deltaG9')); 

			//update ub_input9 and lbs_input9
			ub9_temp := SORT(JOIN(ub_input9, deltac9, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;)),x);
			OUTPUT(ub9_temp, NAMED('ub9_temp'));
			lbs9_temp := JOIN(lbs_input9, deltaG9, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := LEFT.value - RIGHT.value; SELF := LEFT));
			OUTPUT(lbs9_temp, NAMED('lbs9_temp'));
			
			
			// groupfilter1 testing on one time comparison
			groupFilterTemp9 := JOIN(ub_input9, lbs9_temp,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
			OUTPUT(groupFilterTemp9,NAMED('groupFilterTemp9'));
			groupFilter9:= groupFilterTemp9(value <0);
			OUTPUT(groupFilter9,NAMED('groupFilter9_comparison'));
			
			pGroupFilter9 := groupFilter9;
			OUTPUT(pGroupFilter9,NAMED('groupFilter9'));
			groupFilter9_Converge := IF(COUNT(pGroupFilter9)=0, TRUE, FALSE);
			OUTPUT(groupFilter9_Converge, NAMED('groupFilter9_Converge'));
			OUTPUT(pGroupFilter9,NAMED('groupFilter9_result')); 

			unpGroupFilter9 := groupFilterTemp9(value >=0);

			//localFilter
			changedSetTemp9 := JOIN(d01,pGroupFilter9, LEFT.id = RIGHT.x,TRANSFORM(LEFT));
			// changedSetTemp9;
			dDistances9 := ML.Cluster.Distances(changedSetTemp9,dCentroid9);
			OUTPUT(dDistances9,NAMED('dDistances9'));
			dClosest9 := ML.Cluster.Closest(dDistances9);
			OUTPUT(dClosest9,NAMED('distancesOfLocalfilter9')); 

			//the dataset that change their best centroid = result of local fitler		
			localFilter9 := JOIN(dClosest9, ub_input9, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, TRANSFORM(LEFT));

			//localFilter9 is empty then it's converged. Or update the ub and lb.
			localFilter9_converge := IF(COUNT(localFilter9) =0, TRUE, FALSE);
			OUTPUT(localFilter9_converge,NAMED('localFilter9_converge'));
			OUTPUT(localFilter9,NAMED('localFilter9_result'));

			//update the Ub of data points who do not change their best centroid		
			unChangedUbSet9 := JOIN(ub_input9, localFilter9, LEFT.x = RIGHT.x, TRANSFORM(LEFT), LEFT ONLY);
			OUTPUT(unChangedUbSet9, NAMED('unChangedUbSet9'));	
			unChangedUbD019 := JOIN(unChangedUbSet9, d01, LEFT.x = RIGHT.id, TRANSFORM(tempR, SELF.cid := LEFT.y; SELF := RIGHT; ));
			OUTPUT(unChangedUbD019, NAMED('unChangedUbD019'));	
			unChangedUbD01CTemp9 := SORT(JOIN(unChangedUbD019, dCentroid9, LEFT.cid = RIGHT.id AND LEFT.number = RIGHT.number , TRANSFORM(Mat.Types.Element, SELF.value := (LEFT.value-RIGHT.value)*(LEFT.value-RIGHT.value); SELF.y := RIGHT.id;SELF.x := LEFT.id;)),x);
			OUTPUT(unChangedUbD01CTemp9, NAMED('unChangedUbD01CTemp9'));		
			

			unChangedUbRolled9:=ROLLUP(unChangedUbD01CTemp9,TRANSFORM(Mat.Types.Element,SELF.value:=SQRT(LEFT.value+RIGHT.value);SELF := LEFT;),x);
			OUTPUT(unChangedUbRolled9, NAMED('unChangedUbRolled9'));		
			
			//update the Ub of of data points who change their best centroid 		
			changedUb9 := localFilter9;
			OUTPUT(changedUb9, NAMED('changedUb9'));
			unchangedUb9:= unChangedUbRolled9;
			//new Ub
			ub9:= SORT(changedUb9 + unchangedUb9, x);
			OUTPUT(ub9, NAMED('ub9'));
			
			
			//update lbs	
			changelbsTemp9 := JOIN(localFilter9,dDistances9, LEFT.x = RIGHT.x, TRANSFORM(RIGHT));
			OUTPUT(changelbsTemp9, NAMED('changelbsTemp9'));
			changelbsTemp99 := SORT(JOIN(localFilter9,changelbsTemp9, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			OUTPUT(changelbsTemp99, NAMED('changelbsTemp99'));
			//new lbs for data points who change their best centroid
			changelbs9 := DEDUP(changelbsTemp99,x );
			OUTPUT(changelbs9,NAMED('changelbs9')) ;
			//new lbs
			lbs9 := JOIN(lbs9_temp, changelbs9,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value); SELF := LEFT;), LEFT OUTER );
			OUTPUT(lbs9,NAMED('lbs9'));
			
			//update Vin and Vout
			//calculate the Vin
			VinSet9 := SORT(localFilter9,y);
			OUTPUT(VinSet9, NAMED('VinSet9'));
			dClusterCountsVin9:=TABLE(VinSet9, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVin9,NAMED('dClusterCountsVin9'));
			dClusteredVin9 := JOIN(d01, VinSet9, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVin9,NAMED('dClusteredVin9'));
			dRolledVin9:=ROLLUP(SORT(dClusteredVin9,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVin9, NAMED('dRolledVin9'));


			VoutSet9 := JOIN(ub9_temp, localFilter9, LEFT.x = RIGHT.x, TRANSFORM(LEFT));
			OUTPUT(VoutSet9,NAMED('VoutSet9'));
			dClusterCountsVout9:=TABLE(VoutSet9, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVout9,NAMED('dClusterCountsVout9'));
			dClusteredVout9 := JOIN(d01,VoutSet9, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVout9,NAMED('dClusteredVout9'));
			dRolledVout9:=ROLLUP(SORT(dClusteredVout9,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVout9,NAMED('dRolledVout9'));

			
			V9Temp :=JOIN(V_input9,dClusterCountsVin9, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y +RIGHT.c; SELF.value := LEFT.value + RIGHT.c; SELF := LEFT;), LEFT OUTER );
			OUTPUT(V9Temp,NAMED('V9Temp'));
			V9:=JOIN(V9Temp,dClusterCountsVout9, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y -RIGHT.c;SELF.value := LEFT.value -RIGHT.c; SELF := LEFT;), LEFT OUTER);
			OUTPUT(V9,NAMED('V9'));
			

			//let the old |c*V| to multiply old Vcount
			CV9 :=JOIN(dCentroid9, V_input9, LEFT.id = RIGHT.x, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value * RIGHT.y; SELF := LEFT), LEFT OUTER);
			OUTPUT( CV9, NAMED('CV9')) ;

			//Add Vin
			CV9Vin := JOIN(CV9, dRolledVin9, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value + RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV9Vin, NAMED('CV9Vin'));

			//Minus Vout
			CV9VinVout:= JOIN(CV9Vin, dRolledVout9, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value - RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV9VinVout , NAMED('CV9VinVout'));
		
			//get the new C
			newC9:=JOIN(CV9VinVout,V9,LEFT.id=RIGHT.x,TRANSFORM(RECORDOF(LEFT),SELF.value:=(LEFT.value/RIGHT.y);SELF:=LEFT;), LEFT OUTER);
			OUTPUT(newC9, NAMED('C10'));
			
			dAddedx9Temp := JOIN(dAddedx_input9, newC9, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
			OUTPUT(dAddedx9Temp, NAMED('dAddedx9Temp'));
			dAddedx9 := PROJECT(dAddedx9Temp, TRANSFORM(lInput,SELF.id := 1;SELF.values:=LEFT.values;SELF.y := LEFT.number; SELF.x:=LEFT.id;));
			dAddedUb9 := JOIN(iUb_input9, ub9, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 2;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedUb9, NAMED('dAddedUb9'));
			dAddedLbs9 := JOIN(iLbs_input9, lbs9, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 3;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedLbs9, NAMED('dAddedLbs9'));
			dAddedV9 := JOIN(iV_input9, V9, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 4;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedV9, NAMED('dAddedV9'));
			

			outputSet9 := dAddedx9+ dAddedUb9 + dAddedLbs9 +dAddedV9 ;
			SORT(outputSet9,id);		
			
			
//iteration10

d10 := outputSet9;
c10:=10;
//********************************************start iterations*************************************************
// lInput yyfIterate(DATASET(lInput) d,UNSIGNED c):=FUNCTION
			
			dAddedx_input10 := PROJECT(d10(id = 1), TRANSFORM(lIterations , SELF.id := LEFT. x; SELF.number:= LEFT.y; SELF.values := LEFT.values;));
			iUb_input10 := TABLE(d10(id = 2), {x;y;values;});
			iLbs_input10 := TABLE(d10(id = 3), {x;y;values;});
			iV_input10 := TABLE(d10(id = 4), {x;y;values;});
			
			ub_input10 := PROJECT(d10(id = 2), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c10]; SELF := LEFT;));
			OUTPUT(ub_input10, NAMED('ub_input10'));
			lbs_input10 := PROJECT(d10(id = 3), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c10]; SELF := LEFT;));
			OUTPUT(lbs_input10, NAMED('lbs_input10'));
			V_input10 := PROJECT(d10(id = 4), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c10]; SELF := LEFT;));
			OUTPUT(V_input10, NAMED('V_input10'));
			
			//calculate the deltaC
			deltac10 := dDistanceDelta(c10,c10-1,dAddedx_input10);
			OUTPUT(deltac10,NAMED('deltac10'));
			
			bConverged10 := IF(MAX(deltac10,value)<=nConverge,TRUE,FALSE);
			OUTPUT(bConverged10, NAMED('iteration10_converge'));
			
			dCentroid10 := PROJECT(dAddedx_input10,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c10+1];SELF:=LEFT;));
			OUTPUT(dCentroid10,NAMED('dCentroid10'));

			//get deltaG1
			//group deltacg by Gt: first JOIN(deltac10 with Gt) then group by Gt
			deltacg10 :=JOIN(deltac10, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
			OUTPUT(deltacg10, NAMED('delatcg10'));
			deltacGt10 := SORT(deltacg10, y,value);
			output(deltacGt10,NAMED('deltacGt10'));
			deltaG10 := DEDUP(deltacGt10,y, RIGHT);
			OUTPUT(deltaG10,NAMED('deltaG10')); 

			//update ub_input10 and lbs_input10
			ub10_temp := SORT(JOIN(ub_input10, deltac10, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;)),x);
			OUTPUT(ub10_temp, NAMED('ub10_temp'));
			lbs10_temp := JOIN(lbs_input10, deltaG10, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := LEFT.value - RIGHT.value; SELF := LEFT));
			OUTPUT(lbs10_temp, NAMED('lbs10_temp'));
			
			
			// groupfilter1 testing on one time comparison
			groupFilterTemp10 := JOIN(ub_input10, lbs10_temp,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
			OUTPUT(groupFilterTemp10,NAMED('groupFilterTemp10'));
			groupFilter10:= groupFilterTemp10(value <0);
			OUTPUT(groupFilter10,NAMED('groupFilter10_comparison'));
			
			pGroupFilter10 := groupFilter10;
			OUTPUT(pGroupFilter10,NAMED('groupFilter10'));
			groupFilter10_Converge := IF(COUNT(pGroupFilter10)=0, TRUE, FALSE);
			OUTPUT(groupFilter10_Converge, NAMED('groupFilter10_Converge'));
			OUTPUT(pGroupFilter10,NAMED('groupFilter10_result')); 

			unpGroupFilter10 := groupFilterTemp10(value >=0);

			//localFilter
			changedSetTemp10 := JOIN(d01,pGroupFilter10, LEFT.id = RIGHT.x,TRANSFORM(LEFT));
			// changedSetTemp10;
			dDistances10 := ML.Cluster.Distances(changedSetTemp10,dCentroid10);
			OUTPUT(dDistances10,NAMED('dDistances10'));
			dClosest10 := ML.Cluster.Closest(dDistances10);
			OUTPUT(dClosest10,NAMED('distancesOfLocalfilter10')); 

			//the dataset that change their best centroid = result of local fitler		
			localFilter10 := JOIN(dClosest10, ub_input10, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, TRANSFORM(LEFT));

			//localFilter10 is empty then it's converged. Or update the ub and lb.
			localFilter10_converge := IF(COUNT(localFilter10) =0, TRUE, FALSE);
			OUTPUT(localFilter10_converge,NAMED('localFilter10_converge'));
			OUTPUT(localFilter10,NAMED('localFilter10_result'));

			//update the Ub of data points who do not change their best centroid		
			unChangedUbSet10 := JOIN(ub_input10, localFilter10, LEFT.x = RIGHT.x, TRANSFORM(LEFT), LEFT ONLY);
			OUTPUT(unChangedUbSet10, NAMED('unChangedUbSet10'));	
			unChangedUbD0110 := JOIN(unChangedUbSet10, d01, LEFT.x = RIGHT.id, TRANSFORM(tempR, SELF.cid := LEFT.y; SELF := RIGHT; ));
			OUTPUT(unChangedUbD0110, NAMED('unChangedUbD0110'));	
			unChangedUbD01CTemp10 := SORT(JOIN(unChangedUbD0110, dCentroid10, LEFT.cid = RIGHT.id AND LEFT.number = RIGHT.number , TRANSFORM(Mat.Types.Element, SELF.value := (LEFT.value-RIGHT.value)*(LEFT.value-RIGHT.value); SELF.y := RIGHT.id;SELF.x := LEFT.id;)),x);
			OUTPUT(unChangedUbD01CTemp10, NAMED('unChangedUbD01CTemp10'));		
			

			unChangedUbRolled10:=ROLLUP(unChangedUbD01CTemp10,TRANSFORM(Mat.Types.Element,SELF.value:=SQRT(LEFT.value+RIGHT.value);SELF := LEFT;),x);
			OUTPUT(unChangedUbRolled10, NAMED('unChangedUbRolled10'));		
			
			//update the Ub of of data points who change their best centroid 		
			changedUb10 := localFilter10;
			OUTPUT(changedUb10, NAMED('changedUb10'));
			unchangedUb10:= unChangedUbRolled10;
			//new Ub
			ub10:= SORT(changedUb10 + unchangedUb10, x);
			OUTPUT(ub10, NAMED('ub10'));
			
			
			//update lbs	
			changelbsTemp10 := JOIN(localFilter10,dDistances10, LEFT.x = RIGHT.x, TRANSFORM(RIGHT));
			OUTPUT(changelbsTemp10, NAMED('changelbsTemp10'));
			changelbsTemp1010 := SORT(JOIN(localFilter10,changelbsTemp10, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			OUTPUT(changelbsTemp1010, NAMED('changelbsTemp1010'));
			//new lbs for data points who change their best centroid
			changelbs10 := DEDUP(changelbsTemp1010,x );
			OUTPUT(changelbs10,NAMED('changelbs10')) ;
			//new lbs
			lbs10 := JOIN(lbs10_temp, changelbs10,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value); SELF := LEFT;), LEFT OUTER );
			OUTPUT(lbs10,NAMED('lbs10'));
			
			//update Vin and Vout
			//calculate the Vin
			VinSet10 := SORT(localFilter10,y);
			OUTPUT(VinSet10, NAMED('VinSet10'));
			dClusterCountsVin10:=TABLE(VinSet10, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVin10,NAMED('dClusterCountsVin10'));
			dClusteredVin10 := JOIN(d01, VinSet10, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVin10,NAMED('dClusteredVin10'));
			dRolledVin10:=ROLLUP(SORT(dClusteredVin10,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVin10, NAMED('dRolledVin10'));


			VoutSet10 := JOIN(ub10_temp, localFilter10, LEFT.x = RIGHT.x, TRANSFORM(LEFT));
			OUTPUT(VoutSet10,NAMED('VoutSet10'));
			dClusterCountsVout10:=TABLE(VoutSet10, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVout10,NAMED('dClusterCountsVout10'));
			dClusteredVout10 := JOIN(d01,VoutSet10, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVout10,NAMED('dClusteredVout10'));
			dRolledVout10:=ROLLUP(SORT(dClusteredVout10,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVout10,NAMED('dRolledVout10'));

			
			V10Temp :=JOIN(V_input10,dClusterCountsVin10, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y +RIGHT.c; SELF.value := LEFT.value + RIGHT.c; SELF := LEFT;), LEFT OUTER );
			OUTPUT(V10Temp,NAMED('V10Temp'));
			V10:=JOIN(V10Temp,dClusterCountsVout10, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y -RIGHT.c;SELF.value := LEFT.value -RIGHT.c; SELF := LEFT;), LEFT OUTER);
			OUTPUT(V10,NAMED('V10'));
			

			//let the old |c*V| to multiply old Vcount
			CV10 :=JOIN(dCentroid10, V_input10, LEFT.id = RIGHT.x, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value * RIGHT.y; SELF := LEFT), LEFT OUTER);
			OUTPUT( CV10, NAMED('CV10')) ;

			//Add Vin
			CV10Vin := JOIN(CV10, dRolledVin10, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value + RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV10Vin, NAMED('CV10Vin'));

			//Minus Vout
			CV10VinVout:= JOIN(CV10Vin, dRolledVout10, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value - RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV10VinVout , NAMED('CV10VinVout'));
		
			//get the new C
			newC10:=JOIN(CV10VinVout,V10,LEFT.id=RIGHT.x,TRANSFORM(RECORDOF(LEFT),SELF.value:=(LEFT.value/RIGHT.y);SELF:=LEFT;), LEFT OUTER);
			OUTPUT(newC10, NAMED('C11'));
			
			dAddedx10Temp := JOIN(dAddedx_input10, newC10, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
			OUTPUT(dAddedx10Temp, NAMED('dAddedx10Temp'));
			dAddedx10 := PROJECT(dAddedx10Temp, TRANSFORM(lInput,SELF.id := 1;SELF.values:=LEFT.values;SELF.y := LEFT.number; SELF.x:=LEFT.id;));
			dAddedUb10 := JOIN(iUb_input10, ub10, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 2;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedUb10, NAMED('dAddedUb10'));
			dAddedLbs10 := JOIN(iLbs_input10, lbs10, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 3;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedLbs10, NAMED('dAddedLbs10'));
			dAddedV10 := JOIN(iV_input10, V10, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 4;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedV10, NAMED('dAddedV10'));
			

			outputSet10 := dAddedx10+ dAddedUb10 + dAddedLbs10 +dAddedV10 ;
			SORT(outputSet10,id);		

//iteration11

d11 := outputSet10;
c11:=11;
//********************************************start iterations*************************************************
// lInput yyfIterate(DATASET(lInput) d,UNSIGNED c):=FUNCTION
			
			dAddedx_input11 := PROJECT(d11(id = 1), TRANSFORM(lIterations , SELF.id := LEFT. x; SELF.number:= LEFT.y; SELF.values := LEFT.values;));
			iUb_input11 := TABLE(d11(id = 2), {x;y;values;});
			iLbs_input11 := TABLE(d11(id = 3), {x;y;values;});
			iV_input11 := TABLE(d11(id = 4), {x;y;values;});
			
			ub_input11 := PROJECT(d11(id = 2), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c11]; SELF := LEFT;));
			OUTPUT(ub_input11, NAMED('ub_input11'));
			lbs_input11 := PROJECT(d11(id = 3), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c11]; SELF := LEFT;));
			OUTPUT(lbs_input11, NAMED('lbs_input11'));
			V_input11 := PROJECT(d11(id = 4), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c11]; SELF := LEFT;));
			OUTPUT(V_input11, NAMED('V_input11'));
			
			//calculate the deltaC
			deltac11 := dDistanceDelta(c11,c11-1,dAddedx_input11);
			OUTPUT(deltac11,NAMED('deltac11'));
			
			bConverged11 := IF(MAX(deltac11,value)<=nConverge,TRUE,FALSE);
			OUTPUT(bConverged11, NAMED('iteration11_converge'));
			
			dCentroid11 := PROJECT(dAddedx_input11,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c11+1];SELF:=LEFT;));
			OUTPUT(dCentroid11,NAMED('dCentroid11'));

			//get deltaG1
			//group deltacg by Gt: first JOIN(deltac11 with Gt) then group by Gt
			deltacg11 :=JOIN(deltac11, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
			OUTPUT(deltacg11, NAMED('delatcg11'));
			deltacGt11 := SORT(deltacg11, y,value);
			output(deltacGt11,NAMED('deltacGt11'));
			deltaG11 := DEDUP(deltacGt11,y, RIGHT);
			OUTPUT(deltaG11,NAMED('deltaG11')); 

			//update ub_input11 and lbs_input11
			ub11_temp := SORT(JOIN(ub_input11, deltac11, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;)),x);
			OUTPUT(ub11_temp, NAMED('ub11_temp'));
			lbs11_temp := JOIN(lbs_input11, deltaG11, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := LEFT.value - RIGHT.value; SELF := LEFT));
			OUTPUT(lbs11_temp, NAMED('lbs11_temp'));
			
			
			// groupfilter1 testing on one time comparison
			groupFilterTemp11 := JOIN(ub_input11, lbs11_temp,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
			OUTPUT(groupFilterTemp11,NAMED('groupFilterTemp11'));
			groupFilter11:= groupFilterTemp11(value <0);
			OUTPUT(groupFilter11,NAMED('groupFilter11_comparison'));
			
			pGroupFilter11 := groupFilter11;
			OUTPUT(pGroupFilter11,NAMED('groupFilter11'));
			groupFilter11_Converge := IF(COUNT(pGroupFilter11)=0, TRUE, FALSE);
			OUTPUT(groupFilter11_Converge, NAMED('groupFilter11_Converge'));
			OUTPUT(pGroupFilter11,NAMED('groupFilter11_result')); 

			unpGroupFilter11 := groupFilterTemp11(value >=0);

			//localFilter
			changedSetTemp11 := JOIN(d01,pGroupFilter11, LEFT.id = RIGHT.x,TRANSFORM(LEFT));
			// changedSetTemp11;
			dDistances11 := ML.Cluster.Distances(changedSetTemp11,dCentroid11);
			OUTPUT(dDistances11,NAMED('dDistances11'));
			dClosest11 := ML.Cluster.Closest(dDistances11);
			OUTPUT(dClosest11,NAMED('distancesOfLocalfilter11')); 

			//the dataset that change their best centroid = result of local fitler		
			localFilter11 := JOIN(dClosest11, ub_input11, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, TRANSFORM(LEFT));

			//localFilter11 is empty then it's converged. Or update the ub and lb.
			localFilter11_converge := IF(COUNT(localFilter11) =0, TRUE, FALSE);
			OUTPUT(localFilter11_converge,NAMED('localFilter11_converge'));
			OUTPUT(localFilter11,NAMED('localFilter11_result'));

			//update the Ub of data points who do not change their best centroid		
			unChangedUbSet11 := JOIN(ub_input11, localFilter11, LEFT.x = RIGHT.x, TRANSFORM(LEFT), LEFT ONLY);
			OUTPUT(unChangedUbSet11, NAMED('unChangedUbSet11'));	
			unChangedUbD0111 := JOIN(unChangedUbSet11, d01, LEFT.x = RIGHT.id, TRANSFORM(tempR, SELF.cid := LEFT.y; SELF := RIGHT; ));
			OUTPUT(unChangedUbD0111, NAMED('unChangedUbD0111'));	
			unChangedUbD01CTemp11 := SORT(JOIN(unChangedUbD0111, dCentroid11, LEFT.cid = RIGHT.id AND LEFT.number = RIGHT.number , TRANSFORM(Mat.Types.Element, SELF.value := (LEFT.value-RIGHT.value)*(LEFT.value-RIGHT.value); SELF.y := RIGHT.id;SELF.x := LEFT.id;)),x);
			OUTPUT(unChangedUbD01CTemp11, NAMED('unChangedUbD01CTemp11'));		
			

			unChangedUbRolled11:=ROLLUP(unChangedUbD01CTemp11,TRANSFORM(Mat.Types.Element,SELF.value:=SQRT(LEFT.value+RIGHT.value);SELF := LEFT;),x);
			OUTPUT(unChangedUbRolled11, NAMED('unChangedUbRolled11'));		
			
			//update the Ub of of data points who change their best centroid 		
			changedUb11 := localFilter11;
			OUTPUT(changedUb11, NAMED('changedUb11'));
			unchangedUb11:= unChangedUbRolled11;
			//new Ub
			ub11:= SORT(changedUb11 + unchangedUb11, x);
			OUTPUT(ub11, NAMED('ub11'));
			
			
			//update lbs	
			changelbsTemp111 := JOIN(localFilter11,dDistances11, LEFT.x = RIGHT.x, TRANSFORM(RIGHT));
			OUTPUT(changelbsTemp111, NAMED('changelbsTemp111'));
			changelbsTemp11111 := SORT(JOIN(localFilter11,changelbsTemp11, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			OUTPUT(changelbsTemp11111, NAMED('changelbsTemp11111'));
			//new lbs for data points who change their best centroid
			changelbs11 := DEDUP(changelbsTemp11111,x );
			OUTPUT(changelbs11,NAMED('changelbs11')) ;
			//new lbs
			lbs11 := JOIN(lbs11_temp, changelbs11,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value); SELF := LEFT;), LEFT OUTER );
			OUTPUT(lbs11,NAMED('lbs11'));
			
			//update Vin and Vout
			//calculate the Vin
			VinSet11 := SORT(localFilter11,y);
			OUTPUT(VinSet11, NAMED('VinSet11'));
			dClusterCountsVin11:=TABLE(VinSet11, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVin11,NAMED('dClusterCountsVin11'));
			dClusteredVin11 := JOIN(d01, VinSet11, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVin11,NAMED('dClusteredVin11'));
			dRolledVin11:=ROLLUP(SORT(dClusteredVin11,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVin11, NAMED('dRolledVin11'));


			VoutSet11 := JOIN(ub11_temp, localFilter11, LEFT.x = RIGHT.x, TRANSFORM(LEFT));
			OUTPUT(VoutSet11,NAMED('VoutSet11'));
			dClusterCountsVout11:=TABLE(VoutSet11, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVout11,NAMED('dClusterCountsVout11'));
			dClusteredVout11 := JOIN(d01,VoutSet11, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVout11,NAMED('dClusteredVout11'));
			dRolledVout11:=ROLLUP(SORT(dClusteredVout11,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVout11,NAMED('dRolledVout11'));

			
			V11Temp :=JOIN(V_input11,dClusterCountsVin11, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y +RIGHT.c; SELF.value := LEFT.value + RIGHT.c; SELF := LEFT;), LEFT OUTER );
			OUTPUT(V11Temp,NAMED('V11Temp'));
			V11:=JOIN(V11Temp,dClusterCountsVout11, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y -RIGHT.c;SELF.value := LEFT.value -RIGHT.c; SELF := LEFT;), LEFT OUTER);
			OUTPUT(V11,NAMED('V11'));
			

			//let the old |c*V| to multiply old Vcount
			CV11 :=JOIN(dCentroid11, V_input11, LEFT.id = RIGHT.x, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value * RIGHT.y; SELF := LEFT), LEFT OUTER);
			OUTPUT( CV11, NAMED('CV11')) ;

			//Add Vin
			CV11Vin := JOIN(CV11, dRolledVin11, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value + RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV11Vin, NAMED('CV11Vin'));

			//Minus Vout
			CV11VinVout:= JOIN(CV11Vin, dRolledVout11, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value - RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV11VinVout , NAMED('CV11VinVout'));
		
			//get the new C
			newC11:=JOIN(CV11VinVout,V11,LEFT.id=RIGHT.x,TRANSFORM(RECORDOF(LEFT),SELF.value:=(LEFT.value/RIGHT.y);SELF:=LEFT;), LEFT OUTER);
			OUTPUT(newC11, NAMED('C12'));
			
			dAddedx11Temp := JOIN(dAddedx_input11, newC11, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
			OUTPUT(dAddedx11Temp, NAMED('dAddedx11Temp'));
			dAddedx11 := PROJECT(dAddedx11Temp, TRANSFORM(lInput,SELF.id := 1;SELF.values:=LEFT.values;SELF.y := LEFT.number; SELF.x:=LEFT.id;));
			dAddedUb11 := JOIN(iUb_input11, ub11, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 2;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedUb11, NAMED('dAddedUb11'));
			dAddedLbs11 := JOIN(iLbs_input11, lbs11, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 3;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedLbs11, NAMED('dAddedLbs11'));
			dAddedV11 := JOIN(iV_input11, V11, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 4;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedV11, NAMED('dAddedV11'));
			

			outputSet11 := dAddedx11+ dAddedUb11 + dAddedLbs11 +dAddedV11 ;
			SORT(outputSet11,id);		
			
//iteration12

d12 := outputSet11;
c12:=12;
//********************************************start iterations*************************************************
// lInput yyfIterate(DATASET(lInput) d,UNSIGNED c):=FUNCTION
			
			dAddedx_input12 := PROJECT(d12(id = 1), TRANSFORM(lIterations , SELF.id := LEFT. x; SELF.number:= LEFT.y; SELF.values := LEFT.values;));
			iUb_input12 := TABLE(d12(id = 2), {x;y;values;});
			iLbs_input12 := TABLE(d12(id = 3), {x;y;values;});
			iV_input12 := TABLE(d12(id = 4), {x;y;values;});
			
			ub_input12 := PROJECT(d12(id = 2), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c12]; SELF := LEFT;));
			OUTPUT(ub_input12, NAMED('ub_input12'));
			lbs_input12 := PROJECT(d12(id = 3), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c12]; SELF := LEFT;));
			OUTPUT(lbs_input12, NAMED('lbs_input12'));
			V_input12 := PROJECT(d12(id = 4), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c12]; SELF := LEFT;));
			OUTPUT(V_input12, NAMED('V_input12'));
			
			//calculate the deltaC
			deltac12 := dDistanceDelta(c12,c12-1,dAddedx_input12);
			OUTPUT(deltac12,NAMED('deltac12'));
			
			bConverged12 := IF(MAX(deltac12,value)<=nConverge,TRUE,FALSE);
			OUTPUT(bConverged12, NAMED('iteration12_converge'));
			
			dCentroid12 := PROJECT(dAddedx_input12,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c12+1];SELF:=LEFT;));
			OUTPUT(dCentroid12,NAMED('dCentroid12'));

			//get deltaG1
			//group deltacg by Gt: first JOIN(deltac12 with Gt) then group by Gt
			deltacg12 :=JOIN(deltac12, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
			OUTPUT(deltacg12, NAMED('delatcg12'));
			deltacGt12 := SORT(deltacg12, y,value);
			output(deltacGt12,NAMED('deltacGt12'));
			deltaG12 := DEDUP(deltacGt12,y, RIGHT);
			OUTPUT(deltaG12,NAMED('deltaG12')); 

			//update ub_input12 and lbs_input12
			ub12_temp := SORT(JOIN(ub_input12, deltac12, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;)),x);
			OUTPUT(ub12_temp, NAMED('ub12_temp'));
			lbs12_temp := JOIN(lbs_input12, deltaG12, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := LEFT.value - RIGHT.value; SELF := LEFT));
			OUTPUT(lbs12_temp, NAMED('lbs12_temp'));
			
			
			// groupfilter1 testing on one time comparison
			groupFilterTemp12 := JOIN(ub_input12, lbs12_temp,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
			OUTPUT(groupFilterTemp12,NAMED('groupFilterTemp12'));
			groupFilter12:= groupFilterTemp12(value <0);
			OUTPUT(groupFilter12,NAMED('groupFilter12_comparison'));
			
			pGroupFilter12 := groupFilter12;
			OUTPUT(pGroupFilter12,NAMED('groupFilter12'));
			groupFilter12_Converge := IF(COUNT(pGroupFilter12)=0, TRUE, FALSE);
			OUTPUT(groupFilter12_Converge, NAMED('groupFilter12_Converge'));
			OUTPUT(pGroupFilter12,NAMED('groupFilter12_result')); 

			unpGroupFilter12 := groupFilterTemp12(value >=0);

			//localFilter
			changedSetTemp12 := JOIN(d01,pGroupFilter12, LEFT.id = RIGHT.x,TRANSFORM(LEFT));
			// changedSetTemp12;
			dDistances12 := ML.Cluster.Distances(changedSetTemp12,dCentroid12);
			OUTPUT(dDistances12,NAMED('dDistances12'));
			dClosest12 := ML.Cluster.Closest(dDistances12);
			OUTPUT(dClosest12,NAMED('distancesOfLocalfilter12')); 

			//the dataset that change their best centroid = result of local fitler		
			localFilter12 := JOIN(dClosest12, ub_input12, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, TRANSFORM(LEFT));

			//localFilter12 is empty then it's converged. Or update the ub and lb.
			localFilter12_converge := IF(COUNT(localFilter12) =0, TRUE, FALSE);
			OUTPUT(localFilter12_converge,NAMED('localFilter12_converge'));
			OUTPUT(localFilter12,NAMED('localFilter12_result'));

			//update the Ub of data points who do not change their best centroid		
			unChangedUbSet12 := JOIN(ub_input12, localFilter12, LEFT.x = RIGHT.x, TRANSFORM(LEFT), LEFT ONLY);
			OUTPUT(unChangedUbSet12, NAMED('unChangedUbSet12'));	
			unChangedUbD0112 := JOIN(unChangedUbSet12, d01, LEFT.x = RIGHT.id, TRANSFORM(tempR, SELF.cid := LEFT.y; SELF := RIGHT; ));
			OUTPUT(unChangedUbD0112, NAMED('unChangedUbD0112'));	
			unChangedUbD01CTemp12 := SORT(JOIN(unChangedUbD0112, dCentroid12, LEFT.cid = RIGHT.id AND LEFT.number = RIGHT.number , TRANSFORM(Mat.Types.Element, SELF.value := (LEFT.value-RIGHT.value)*(LEFT.value-RIGHT.value); SELF.y := RIGHT.id;SELF.x := LEFT.id;)),x);
			OUTPUT(unChangedUbD01CTemp12, NAMED('unChangedUbD01CTemp12'));		
			

			unChangedUbRolled12:=ROLLUP(unChangedUbD01CTemp12,TRANSFORM(Mat.Types.Element,SELF.value:=SQRT(LEFT.value+RIGHT.value);SELF := LEFT;),x);
			OUTPUT(unChangedUbRolled12, NAMED('unChangedUbRolled12'));		
			
			//update the Ub of of data points who change their best centroid 		
			changedUb12 := localFilter12;
			OUTPUT(changedUb12, NAMED('changedUb12'));
			unchangedUb12:= unChangedUbRolled12;
			//new Ub
			ub12:= SORT(changedUb12 + unchangedUb12, x);
			OUTPUT(ub12, NAMED('ub12'));
			
			
			//update lbs	
			changelbsTemp12 := JOIN(localFilter12,dDistances12, LEFT.x = RIGHT.x, TRANSFORM(RIGHT));
			OUTPUT(changelbsTemp12, NAMED('changelbsTemp12'));
			changelbsTemp1212 := SORT(JOIN(localFilter12,changelbsTemp12, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			OUTPUT(changelbsTemp1212, NAMED('changelbsTemp1212'));
			//new lbs for data points who change their best centroid
			changelbs12 := DEDUP(changelbsTemp1212,x );
			OUTPUT(changelbs12,NAMED('changelbs12')) ;
			//new lbs
			lbs12 := JOIN(lbs12_temp, changelbs12,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value); SELF := LEFT;), LEFT OUTER );
			OUTPUT(lbs12,NAMED('lbs12'));
			
			//update Vin and Vout
			//calculate the Vin
			VinSet12 := SORT(localFilter12,y);
			OUTPUT(VinSet12, NAMED('VinSet12'));
			dClusterCountsVin12:=TABLE(VinSet12, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVin12,NAMED('dClusterCountsVin12'));
			dClusteredVin12 := JOIN(d01, VinSet12, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVin12,NAMED('dClusteredVin12'));
			dRolledVin12:=ROLLUP(SORT(dClusteredVin12,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVin12, NAMED('dRolledVin12'));


			VoutSet12 := JOIN(ub12_temp, localFilter12, LEFT.x = RIGHT.x, TRANSFORM(LEFT));
			OUTPUT(VoutSet12,NAMED('VoutSet12'));
			dClusterCountsVout12:=TABLE(VoutSet12, {y; INTEGER c := COUNT(GROUP)},y);
			OUTPUT(dClusterCountsVout12,NAMED('dClusterCountsVout12'));
			dClusteredVout12 := JOIN(d01,VoutSet12, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			OUTPUT(dClusteredVout12,NAMED('dClusteredVout12'));
			dRolledVout12:=ROLLUP(SORT(dClusteredVout12,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));
			OUTPUT(dRolledVout12,NAMED('dRolledVout12'));

			
			V12Temp :=JOIN(V_input12,dClusterCountsVin12, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y +RIGHT.c; SELF.value := LEFT.value + RIGHT.c; SELF := LEFT;), LEFT OUTER );
			OUTPUT(V12Temp,NAMED('V12Temp'));
			V12:=JOIN(V12Temp,dClusterCountsVout12, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y -RIGHT.c;SELF.value := LEFT.value -RIGHT.c; SELF := LEFT;), LEFT OUTER);
			OUTPUT(V12,NAMED('V12'));
			

			//let the old |c*V| to multiply old Vcount
			CV12 :=JOIN(dCentroid12, V_input12, LEFT.id = RIGHT.x, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value * RIGHT.y; SELF := LEFT), LEFT OUTER);
			OUTPUT( CV12, NAMED('CV12')) ;

			//Add Vin
			CV12Vin := JOIN(CV12, dRolledVin12, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value + RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV12Vin, NAMED('CV12Vin'));

			//Minus Vout
			CV12VinVout:= JOIN(CV12Vin, dRolledVout12, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value - RIGHT.value; SELF := LEFT), LEFT OUTER);
			OUTPUT(CV12VinVout , NAMED('CV12VinVout'));
		
			//get the new C
			newC12:=JOIN(CV12VinVout,V12,LEFT.id=RIGHT.x,TRANSFORM(RECORDOF(LEFT),SELF.value:=(LEFT.value/RIGHT.y);SELF:=LEFT;), LEFT OUTER);
			OUTPUT(newC12, NAMED('C13'));
			
			dAddedx12Temp := JOIN(dAddedx_input12, newC12, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
			OUTPUT(dAddedx12Temp, NAMED('dAddedx12Temp'));
			dAddedx12 := PROJECT(dAddedx12Temp, TRANSFORM(lInput,SELF.id := 1;SELF.values:=LEFT.values;SELF.y := LEFT.number; SELF.x:=LEFT.id;));
			dAddedUb12 := JOIN(iUb_input12, ub12, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 2;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedUb12, NAMED('dAddedUb12'));
			dAddedLbs12 := JOIN(iLbs_input12, lbs12, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 3;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedLbs12, NAMED('dAddedLbs12'));
			dAddedV12 := JOIN(iV_input12, V12, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 4;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			OUTPUT(dAddedV12, NAMED('dAddedV12'));
			

			outputSet12 := dAddedx12+ dAddedUb12 + dAddedLbs12 +dAddedV12 ;
			SORT(outputSet12,id);		


			