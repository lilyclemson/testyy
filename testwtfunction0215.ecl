// IMPORT * FROM $;
// IMPORT Std.Str AS Str;
IMPORT TS;
IMPORT ML.Mat;
IMPORT ML;
IMPORT ML.Cluster.DF as DF;
IMPORT ML.Types AS Types;
IMPORT ML.Utils;
IMPORT ML.MAT AS Mat;

// IMPORT excercise.irisset as irisset;
// IMPORT excercise.uscensus as uscensus;

lMatrix:={UNSIGNED id;REAL x;REAL y;};
// dDocumentMatrix := irisset.input;
// dCentroidMatrix := irisset.input[1..4];
// dDocumentMatrix := uscensus.input;
// dCentroidMatrix := uscensus.input[1..4];
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

K := COUNT(d02)/2;
//K; 
//t:= IF(K/10<1, 1, K/10);
t:=2;
//t; 


//temporary solution to get dt is that dt = the first t cendroids of dCentroid 
temp := t * 2;
tempDt := dCentroid0[1..temp];
groupDs:=PROJECT(tempDt,TRANSFORM(Types.NumericField,SELF.id:=LEFT.id + k;SELF:=LEFT;));
nt := n;
tConverge := nConverge;
KmeansDt :=ML.Cluster.KMeans(dCentroid0,groupDs,nt,tConverge);
Gt := TABLE(KmeansDt.Allegiances(), {x,y},y,x);//the assignment of each centroid to a group
OUTPUT(Gt, NAMED('Gt'));

//initialize ub and lb for each data point
//get ub0
dDistances0 := ML.Cluster.Distances(d01,dCentroid0);
OUTPUT(dDistances0, NAMED('dDistances0'));
dClosest0 := ML.Cluster.Closest(dDistances0);
OUTPUT(dClosest0, NAMED('dClosest0'));

//*******add a data structure to keep track of  second closest center and best center
dTrack0 := DEDUP(SORT(DISTRIBUTE(dDistances0,x),x,value,LOCAL),x,KEEP 2,LOCAL);
OUTPUT(dTrack0, NAMED('dTrack0'));

// dClosest0;
ub0_ini := dClosest0; 
OUTPUT(ub0_ini, NAMED('ub0_ini'));

//get lbs0_ini
//lb of each group: every group should have at least two centroids**
lbs0_iniTemp := JOIN(dClosest0,dDistances0, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y,RIGHT ONLY);
lbs0_iniTemp1 := SORT(JOIN(lbs0_iniTemp, Gt, LEFT.y = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.y := RIGHT.y; SELF := LEFT;)),x, y, value);
//********************correct the lbs0_ini
lbs0_ini := DEDUP(lbs0_iniTemp1,LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y);
// lbs0_ini := DEDUP(lbs0_iniTemp,x, RIGHT);
OUTPUT(lbs0_iniTemp,NAMED('lbs0_iniTemp'));
OUTPUT(lbs0_iniTemp1,NAMED('lbs0_iniTemp1')) ;
OUTPUT(lbs0_ini,NAMED('lbs0_ini'));


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
ClusterPair:=RECORD
		Types.t_RecordID    id;
		Types.t_RecordID    clusterid;
		Types.t_FieldNumber number;
		Types.t_FieldReal   value01 := 0;
		Types.t_FieldReal   value02 := 0;
		Types.t_FieldReal   value03 := 0;
  END;
c_model := ENUM ( Dense = 1, SJoins = 2, Background = 4 );

MappedDistances(DATASET(Types.NumericField) d01,DATASET(Types.NumericField) d02 ,DF.Default Control = DF.Euclidean, DATASET(ClusterPair) dMap = DATASET([],ClusterPair)) := FUNCTION	
			// If we are in dense model then fatten up the records; otherwise zeroes not needed
   		df1 := IF( Control.Pmodel & c_model.dense > 0, Utils.Fat(d01), d01(value<>0) );
   		df2 := IF( Control.Pmodel & c_model.dense > 0, Utils.Fat(d02), d02(value<>0) );
   		// Construct the summary records used by SJoins and Background processing models
   		si1 := Control.SummaryID1(df1); // Summaries of each document by ID
   		si2 := Control.SummaryID2(df2); // May be used by any summary joins features
			//Slim down the si1 by filtering out the data points not in the dMap table
			si1_mapped := JOIN(dMap,si1, LEFT.id = RIGHT.id, TRANSFORM(Types.NumericField, SELF.number:=LEFT.clusterid; SELF := RIGHT; ));
			// Construct the 'background' matrix from the summary matrix
			bck := JOIN(si1_mapped,si2,LEFT.id<>RIGHT.id AND LEFT.number = RIGHT.id,TRANSFORM(Mat.Types.Element,SELF.x := LEFT.id, SELF.y := RIGHT.Id, SELF.value := Control.BackGround(LEFT,RIGHT)),ALL);
   		// Create up to two 'aggregate' numbers that the models may use
   		ex1 := Control.EV1(d01); 
   		ex2 := Control.EV2(d01);
   		//Mapped dataset of d01 and d02 based on the relation specified in the dMap table.
			df2_mapped := JOIN(dMap, df2, LEFT.clusterid = RIGHT.id, TRANSFORM(ClusterPair, SELF.clusterid := LEFT.id; SELF.value01:=RIGHT.value; SELF.value02:=0;SELF.value03:=0;SELF := RIGHT; ));//dRelation   
			ClusterPair Take2_mapped(df1 le,ClusterPair ri) := TRANSFORM
						SELF.clusterid := ri.id;
						SELF.id := le.id;
						SELF.number := le.number;
						SELF.value01 := Control.IV1(le.value,ri.value01);
						SELF.value02 := Control.IV2(le.value,ri.value01);
			END;
 			J := JOIN(df1,df2_mapped,LEFT.number=RIGHT.number AND LEFT.id<>RIGHT.id AND Control.JoinFilter(LEFT.value,RIGHT.value01,ex1) AND LEFT.id = RIGHT.clusterid ,Take2_mapped(LEFT,RIGHT),HASH); // numbers will be evenly distribute by definition
   		// Take all of the values computed for each matching ID and combine them
   		JG := GROUP(J,id,clusterid, ALL);  
   		ClusterPair  roll(ClusterPair le, DATASET(ClusterPair) gd) := TRANSFORM
   				SELF.Value01 := Control.Comb1(gd,ex1,ex2);
   				SELF.Value02 := Control.Comb2(gd,ex1,ex2); // These are really scratchpad
   				SELF.Value03 := Control.Comb3(gd,ex1,ex2);
   				SELF := le;
   		END;			  		
   		rld := ROLLUP(JG,GROUP,roll(LEFT,ROWS(LEFT)));
   		// In the SJoins processing model the si1/si2 data is now "passed to" the result - 01 first
   		J1 := JOIN(rld,si1,LEFT.id=RIGHT.id,TRANSFORM(ClusterPair,SELF.value01 := Control.Join11(LEFT,RIGHT),SELF.value02 := Control.Join12(LEFT,RIGHT),SELF.value03 := Control.Join13(LEFT,RIGHT), SELF := LEFT),LOOKUP);
   		J2 := JOIN(J1,si2,LEFT.clusterid=RIGHT.id,TRANSFORM(ClusterPair,SELF.value01 := Control.Join21(LEFT,RIGHT), SELF := LEFT),LOOKUP);
   		// Select either the 'normal' or 'post joined' version of the scores
   		Pro := IF ( Control.PModel & c_model.SJoins > 0, J2, rld );
   		ProAsDist := PROJECT(Pro,TRANSFORM(Mat.Types.Element,SELF.x := LEFT.id,SELF.y := LEFT.clusterid,SELF.Value := LEFT.Value01, SELF := LEFT));
   		// Now blend the scores that were computed with the background model
   		Mat.Types.Element blend(bck le,pro ri) := TRANSFORM
   		  SELF.value := Control.BackFront(le,ri);
   		  SELF := le;
   		END;   		
			BF := JOIN(bck,pro,LEFT.y=RIGHT.ClusterID AND LEFT.x=RIGHT.id,blend(LEFT,RIGHT),LEFT OUTER);
			
			// Either select the background blended version - or slim the scores down to a cluster distance
			RETURN IF(Control.PModel & c_model.Background>0, BF,ProAsDist); 	
			
  END; 	

//*********************************************************************************************************************************************************************

//set the result of Standard Kmeans to the result of first iteration
firstIte := KmeansD01.AllResults();
// OUTPUT(firstIte,NAMED('firfostIte'));
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
			
			dAddedx0 := PROJECT(d(id = 1), TRANSFORM(lIterations , SELF.id := LEFT. x; SELF.number:= LEFT.y; SELF.values := LEFT.values;));
			iUb1 := TABLE(d(id = 2), {x;y;values;});
			iLbs1 := TABLE(d(id = 3), {x;y;values;});
			iV1 := TABLE(d(id = 4), {x;y;values;});
			
			ub0 := PROJECT(d(id = 2), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c]; SELF := LEFT;));
			OUTPUT(ub0, NAMED('ub0'));
			lbs0 := PROJECT(d(id = 3), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c]; SELF := LEFT;));
			OUTPUT(lbs0, NAMED('lbs0'));
			V0 := PROJECT(d(id = 4), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c]; SELF := LEFT;));
			OUTPUT(V0, NAMED('V0'));
			
			//*********add lb0 which is the min(lbs0) of each data point
			lb0 := DEDUP(SORT(lbs0, x, y, value), x);
			OUTPUT(lb0, NAMED('lb0_globallb'));
			//lbG := MIN(lb0, value);
			
			//calculate the deltac0
			deltac0 := dDistanceDelta(c,c-1,dAddedx0);
			OUTPUT(deltac0,NAMED('deltac0'));
			
			bConverged0 := IF(MAX(deltac0,value)<=nConverge,TRUE,FALSE);
			OUTPUT(bConverged0, NAMED('iteration1_converge'));
			
			dCentroid1 := PROJECT(dAddedx0,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c+1];SELF:=LEFT;));
			OUTPUT(dCentroid1,NAMED('dCentroid1'));

			//get deltaG0
			//group deltac0g by Gt: first JOIN(deltac0 with Gt) then group by Gt
			deltac0g :=JOIN(deltac0, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
			OUTPUT(deltac0g, NAMED('delatc0g'));
			deltac0Gt := SORT(deltac0g, y,-value);
			output(deltac0Gt,NAMED('deltac0Gt'));
			deltaG0 := DEDUP(deltac0Gt,y);
			OUTPUT(deltaG0,NAMED('deltaG0')); 

			//update ub0 and lbs0
			ub1_temp := JOIN(ub0, deltac0, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;));
			OUTPUT(ub1_temp, NAMED('ub1_temp'));
			lbs1_temp := JOIN(lbs0, deltaG0, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := ABS(LEFT.value - RIGHT.value); SELF := LEFT));
			OUTPUT(lbs1_temp, NAMED('lbs1_temp'));
			
			
		
			// groupfilter1
			//calculate the 'b(x) for all x
			dMap1 := PROJECT(ub0, TRANSFORM(ClusterPair, SELF.id := LEFT.x; SELF.clusterid := LEFT.y; SELF.number := 0; SELF.value01 := LEFT.value; SELF.value02 := 0; SELF.value03 := 0;));		
			dMappedDistances1 := SORT(MappedDistances(d01,dCentroid1,fDist,dMap1), x, value);	
			OUTPUT(dMappedDistances1, NAMED('dMappedDistances1'));
			ub1_changed_temp := ML.Cluster.Closest(dMappedDistances1);
			OUTPUT(ub1_changed_temp);
      groupFilter1 := JOIN(lbs1_temp, ub1_temp,LEFT.x = RIGHT.x AND (LEFT.value < RIGHT.value), TRANSFORM(Mat.Types.Element, SELF := LEFT));	
      OUTPUT(groupFilter1, NAMED('groupFilter1'));			
//			groupFilter2:= JOIN(groupFilter1, ub1_changed_temp,LEFT.x = RIGHT.x AND (LEFT.value < RIGHT.value), TRANSFORM(Mat.Types.Element,SELF := LEFT));
 //     OUTPUT(groupFilter2, NAMED('groupFilter2'));
			
			dMap2_temp := JOIN(groupFilter1, Gt, LEFT.y = RIGHT.y, TRANSFORM(Mat.Types.Element, SELF.x := LEFT.x, SELF.y := RIGHT.x, SELF.value := LEFT.value ));
			dMap2 := PROJECT(dMap2_temp, TRANSFORM(ClusterPair, SELF.id := LEFT.x; SELF.clusterid := LEFT.y; SELF.number := 0; SELF.value01 := LEFT.value; SELF.value02 := 0; SELF.value03 := 0;));		
      OUTPUT(dMap2, NAMED('dMap2'));			
			dMappedDistances2 := SORT(MappedDistances(d01,dCentroid1,fDist,dMap2), x, value);	
			OUTPUT(dMappedDistances2, NAMED('dMappedDistances2'));
			ub1_changed_final := ML.Cluster.Closest(dMappedDistances2);
			OUTPUT(ub1_changed_final, NAMED('ub1_changed_final'));
			ub1_changed := JOIN(ub1_changed_temp, ub1_changed_final, LEFT.x = RIGHT.x AND LEFT.value > RIGHT.value, TRANSFORM(RIGHT));
			

 //     ub1_changed := JOIN(dMap2, ub1_changed_temp, LEFT.x = RIGHT.x AND LEFT.value < RIGHT.value, TRANSFORM(RIGHT));
			OUTPUT(ub1_changed, NAMED('ub1_changed'));
			ub1_unchanged := JOIN(ub1_changed_temp, ub1_changed, LEFT.x = RIGHT.x, TRANSFORM(LEFT),LEFT ONLY);
			OUTPUT(ub1_unchanged, NAMED('ub1_unchanged'));
			ub1 := SORT(ub1_changed + ub1_unchanged, x, y, value);
      OUTPUT(ub1, NAMED('ub1'));
							
			lbs1_changed_temp := JOIN(dMappedDistances1, ub1_changed, LEFT.x = RIGHT.x, TRANSFORM(LEFT), LEFT ONLY);
			lbs1_changed_temp1 := JOIN(lbs1_changed_temp, Gt, LEFT.y = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.x := LEFT.x, SELF.y := RIGHT.y, SELF.value := LEFT.value ));
			OUTPUT(lbs1_changed_temp, NAMED('lbs1_changed_temp'));
			lbs1_changed:= DEDUP(SORT(lbs1_changed_temp1,x,y,value),x,y);
			lbs1_unchanged:= JOIN(lbs1_temp, lbs1_changed, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, LEFT ONLY);
      lbs1 := lbs1_unchanged + lbs1_changed;			
			OUTPUT(lbs1, NAMED('lbs1'));
			
      dClusterCounts0:=TABLE(ub1,{y;UNSIGNED c:=COUNT(GROUP);},y,FEW);
			OUTPUT(dClusterCounts0, NAMED('dClusterCounts0'));
      // Join closest to the document set and replace the id with the centriod id
       dClustered0:=SORT(DISTRIBUTE(JOIN(d01,ub1,LEFT.id=RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;),HASH),id),RECORD,LOCAL);
      // Now roll up on centroid ID, summing up the values for each axis
      dRolled0:=ROLLUP( dClustered0,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;),id,number,LOCAL);
      // Join to cluster counts to calculate the new average on each axis
      dJoined0:=JOIN(dRolled0,dClusterCounts0,LEFT.id=RIGHT.y,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value/RIGHT.c;SELF:=LEFT;),LOOKUP);
      // Find any centroids with no document allegiance and pass those through also
		  dPass0:=JOIN(dCentroid1,TABLE(dJoined0,{id},id,LOCAL),LEFT.id=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);		
      dCentroidout0 := dPass0 + dJoined0;
      OUTPUT(dCentroidout0, NAMED('ddCentroidout0'));  
			//Now join to the existing input datasets to add the new values to the end of each values set.
			dAddedx1 := JOIN(dAddedx0, dCentroidOut0, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
						
	
//itr2
c2 := 2;

			
			//calculate the deltac1
			deltac1 := dDistanceDelta(c2,c2-2,dAddedx1);
			OUTPUT(deltac1,NAMED('deltac1'));
			
			bConverged1 := IF(MAX(deltac1,value)<=nConverge,TRUE,FALSE);
			OUTPUT(bConverged1, NAMED('iteration2_converge'));
			
			dCentroid2 := dCentroidOut0;
			OUTPUT(dCentroid2,NAMED('dCentroid2'));

			//get deltaG1
			//group deltac1g by Gt: first JOIN(deltac1 with Gt) then group by Gt
			deltac1g :=JOIN(deltac1, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
			OUTPUT(deltac1g, NAMED('delatcg'));
			deltac1Gt := SORT(deltac1g, y,-value);
			output(deltac1Gt,NAMED('deltac1Gt'));
			deltaG1 := DEDUP(deltac1Gt,y);
			OUTPUT(deltaG1,NAMED('deltaG1')); 

			//update ub21 and lbs1
			ub2_temp := JOIN(ub1, deltac1, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;));
			OUTPUT(ub2_temp, NAMED('ub2_temp'));
			lbs2_temp := JOIN(lbs1, deltaG1, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := ABS(LEFT.value - RIGHT.value); SELF := LEFT));
			OUTPUT(lbs2_temp, NAMED('lbs2_temp'));
			
			
		
			// groupfilter2
			//calculate the 'b(x) for all x
			dMap3 := PROJECT(ub1, TRANSFORM(ClusterPair, SELF.id := LEFT.x; SELF.clusterid := LEFT.y; SELF.number := 0; SELF.value01 := LEFT.value; SELF.value02 := 0; SELF.value03 := 0;));		
			dMappedDistances3 := SORT(MappedDistances(d02,dCentroid2,fDist,dMap2), x, value);	
			OUTPUT(dMappedDistances3, NAMED('dMappedDistances3'));
			ub2_changed_temp := ML.Cluster.Closest(dMappedDistances3);
			OUTPUT(ub2_changed_temp);
      groupFilter2 := JOIN(lbs2_temp, ub2_temp,LEFT.x = RIGHT.x AND (LEFT.value < RIGHT.value), TRANSFORM(Mat.Types.Element, SELF := LEFT));	
      OUTPUT(groupFilter2, NAMED('groupFilter2'));			
//			groupFilter2:= JOIN(groupFilter2, ub2_changed_temp,LEFT.x = RIGHT.x AND (LEFT.value < RIGHT.value), TRANSFORM(Mat.Types.Element,SELF := LEFT));
 //     OUTPUT(groupFilter2, NAMED('groupFilter2'));
			
			dMap4_temp := JOIN(groupFilter2, Gt, LEFT.y = RIGHT.y, TRANSFORM(Mat.Types.Element, SELF.x := LEFT.x, SELF.y := RIGHT.x, SELF.value := LEFT.value ));
			dMap4 := PROJECT(dMap4_temp, TRANSFORM(ClusterPair, SELF.id := LEFT.x; SELF.clusterid := LEFT.y; SELF.number := 0; SELF.value01 := LEFT.value; SELF.value02 := 0; SELF.value03 := 0;));		
      OUTPUT(dMap4, NAMED('dMap4'));			
			dMappedDistances4 := SORT(MappedDistances(d02,dCentroid2,fDist,dMap4), x, value);	
			OUTPUT(dMappedDistances4, NAMED('dMappedDistances4'));
			ub2_changed_final := ML.Cluster.Closest(dMappedDistances4);
			OUTPUT(ub2_changed_final, NAMED('ub2_changed_final'));
			ub2_changed := JOIN(ub2_changed_temp, ub2_changed_final, LEFT.x = RIGHT.x AND LEFT.value > RIGHT.value, TRANSFORM(RIGHT));
			

 //     ub2_changed := JOIN(dMap3, ub2_changed_temp, LEFT.x = RIGHT.x AND LEFT.value < RIGHT.value, TRANSFORM(RIGHT));
			OUTPUT(ub2_changed, NAMED('ub2_changed'));
			ub2_unchanged := JOIN(ub2_changed_temp, ub2_changed, LEFT.x = RIGHT.x, TRANSFORM(LEFT),LEFT ONLY);
			OUTPUT(ub2_unchanged, NAMED('ub2_unchanged'));
			ub2 := SORT(ub2_changed + ub2_unchanged, x, y, value);
      OUTPUT(ub2, NAMED('ub2'));
							
			lbs2_changed_temp := JOIN(dMappedDistances2, ub2_changed, LEFT.x = RIGHT.x, TRANSFORM(LEFT), LEFT ONLY);
			lbs2_changed_temp2 := JOIN(lbs2_changed_temp, Gt, LEFT.y = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.x := LEFT.x, SELF.y := RIGHT.y, SELF.value := LEFT.value ));
			OUTPUT(lbs2_changed_temp, NAMED('lbs2_changed_temp'));
			lbs2_changed:= DEDUP(SORT(lbs2_changed_temp2,x,y,value),x,y);
			lbs2_unchanged:= JOIN(lbs2_temp, lbs2_changed, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, LEFT ONLY);
      lbs2 := lbs2_unchanged + lbs2_changed;			
			OUTPUT(lbs2, NAMED('lbs2'));
			
      dClusterCounts1:=TABLE(ub2,{y;UNSIGNED c:=COUNT(GROUP);},y,FEW);
			OUTPUT(dClusterCounts1, NAMED('dClusterCounts1'));
      // Join closest to the document set and replace the id with the centriod id
       dClustered1:=SORT(DISTRIBUTE(JOIN(d02,ub2,LEFT.id=RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;),HASH),id),RECORD,LOCAL);
      // Now roll up on centroid ID, summing up the values for each axis
      dRolled1:=ROLLUP( dClustered1,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;),id,number,LOCAL);
      // Join to cluster counts to calculate the new average on each axis
      dJoined1:=JOIN(dRolled1,dClusterCounts1,LEFT.id=RIGHT.y,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value/RIGHT.c;SELF:=LEFT;),LOOKUP);
      // Find any centroids with no document allegiance and pass those through also
		  dPass1:=JOIN(dCentroid2,TABLE(dJoined1,{id},id,LOCAL),LEFT.id=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);		
      dCentroidout1 := dPass1 + dJoined1;
      OUTPUT(dCentroidout1, NAMED('dCentroid3'));  
			//Now join to the existing input datasets to add the new values to the end of each values set.
			dAddedx3 := JOIN(dAddedx1, dCentroidOut1, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));

