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
IMPORT excercise.relation20network as network;

lMatrix:={UNSIGNED id;REAL x;REAL y;};
// dDocumentMatrix := irisset.input;
// dCentroidMatrix := irisset.input[1..4];

// dDocumentMatrix := network.input;
// dCentroidMatrix := network.input[1..4];
 
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
// internal record structure of d02
lIterations:=RECORD 
TYPEOF(Types.NumericField.id) id;
TYPEOF(Types.NumericField.number) number;
SET OF TYPEOF(Types.NumericField.value) values;
END;

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

//Inputs:
d01 := dDocuments;
d02 := dCentroids;
n := 30;
nConverge := 0.3;
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

//***simple version of Distances function
  EuclideanDist(DATASET(ClusterPair) d) :=FUNCTION
  ClusterPair Euclidean(ClusterPair d):=TRANSFORM
      SELF.value03:= (d.value01-d.value02) * (d.value01-d.value02);
      SELF := d;      
  END;
  input := PROJECT(d, Euclidean(LEFT));
  dRolled:=ROLLUP(input,TRANSFORM(ClusterPair, SELF.value03:=LEFT.value03+RIGHT.value03, SELF := LEFT;),id,clusterid);
  result:=PROJECT(dRolled,TRANSFORM(Mat.Types.Element,SELF.y := LEFT.clusterid, SELF.x := LEFT.id, SELF.value:=SQRT(LEFT.value03);));
  RETURN result;
  END;
  Distances(DATASET(Types.NumericField) d01,DATASET(Types.NumericField) d02 , DATASET(ClusterPair) dMap = DATASET([],ClusterPair)) := FUNCTION	
			j01 := JOIN(d01,dMap, LEFT.id = RIGHT.id, TRANSFORM(ClusterPair, SELF.id := LEFT.id, SELF.number:=LEFT.number, SELF.value01 := LEFT.value, SELF := RIGHT; ));
			// Construct the 'background' matrix from the summary matrix
			j02 := JOIN(j01,d02,LEFT.clusterid = RIGHT.id AND LEFT.number = RIGHT.number,TRANSFORM(ClusterPair,SELF.value02 := RIGHT.value, SELF := LEFT;));
            rst:= EuclideanDist(j02);
			RETURN 	rst;
  END; 	
//*********************************************************************************************************************************************************************

//make sure all ids are different
maxID1 := MAX(d01,id);
minID2 := MIN(d02,id);
//k := COUNT(d02)/2;



iOffset:=IF(maxID1>minID2,maxID1,0);
d02Prep:=PROJECT(d02,TRANSFORM(lIterations,SELF.id:=LEFT.id+iOffset;SELF.values:=[LEFT.value];SELF:=LEFT;));
dCentroid0 := PROJECT(d02Prep,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[1];SELF:=LEFT;));
OUTPUT(dCentroid0,NAMED('Initial_Centroids')); 

//number of groups
t:=1;
//temporary solution to get dt is that dt = the first t cendroids of dCentroid 
//temp := t * 2;
//tempDt := dCentroid0[1..temp];
//groupDs:=PROJECT(tempDt,TRANSFORM(Types.NumericField,SELF.id:=LEFT.id + k;SELF:=LEFT;));
//KmeansDt :=ML.Cluster.KMeans(dCentroid0,groupDs,5,nConverge);
//Gt := TABLE(KmeansDt.Allegiances(), {x,y},y,x);//the assignment of each centroid to a group
//gid := minID2 + iOffset + k;
//Gt := PROJECT(dCentroid0,TRANSFORM({Mat.Types.Element.x , Mat.Types.Element.y},SELF.x:=LEFT.id ;SELF.y:=1;));
//Gt := DEDUP(Gt1, x);
//OUTPUT(Gt, NAMED('Gt'));

//initialize ub and lb for each data point
//get ub0
dDistances0 := ML.Cluster.Distances(d01,dCentroid0);
OUTPUT(dDistances0, NAMED('dDistances0'));
//dClosest0 := ML.Cluster.Closest(dDistances0);
groupclose := GROUP(SORT(dDistances0, x), x);
OUTPUT(groupclose, NAMED('groupclose'));
dClosest0 :=TOPN( groupclose,1, value);
OUTPUT(dClosest0, NAMED('dClosest0'));

ub0_ini := dClosest0; 
OUTPUT(ub0_ini, NAMED('ub0_ini'));
dClusterCounts_ini:=TABLE(dClosest0,{y;UNSIGNED c:=COUNT(GROUP);},y,FEW);
dClustered_ini:=SORT(DISTRIBUTE(JOIN(d01,dClosest0,LEFT.id=RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;),HASH),id),RECORD,LOCAL);
dRolled_ini:=ROLLUP(dClustered_ini,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;),id,number,LOCAL);
dJoined_ini:=JOIN(dRolled_ini,dClusterCounts_ini,LEFT.id=RIGHT.y,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value/RIGHT.c;SELF:=LEFT;),LOOKUP);		
dPass_ini:=JOIN(dCentroid0,TABLE(dJoined_ini,{id},id,LOCAL),LEFT.id=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);
dCentroid1 := dJoined_ini + dPass_ini;
OUTPUT(dCentroid1, NAMED('dCentroid1') );

//get lbs0_ini
//lb of each group: every group should have at least two centroids**
lbs0_iniTemp := JOIN(dDistances0, dClosest0, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y,LEFT ONLY);
lbs0_ini_Temp1 := GROUP(SORT(lbs0_iniTemp, x), x);
lbs0_ini := TOPN( lbs0_ini_Temp1,1, value);
OUTPUT(lbs0_iniTemp,NAMED('lbs0_iniTemp'));
OUTPUT(lbs0_iniTemp,NAMED('lbs0_ini_Temp1'));
OUTPUT(lbs0_ini,NAMED('lbs0_ini'));


//*********************now Gt (deltaG will be initailized and updated in iteration ), ub, lb are initialized ************************
// Now join to the existing centroid dataset to add the new values to the end of the values set.
dAdded1:=JOIN(d02Prep,dCentroid1,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=RIGHT;),RIGHT OUTER);

//data preparation

//transform dCentroid to the format align with ub_ini, lbs0_ini, Vin0_ini, Vout0_ini
dCentroid0Trans := PROJECT(dCentroid0, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value ; SELF.x := LEFT.id; SELF.y := LEFT.number));


iCentroid0Temp := PROJECT(dCentroid0Trans, transFormat(LEFT, 1));
iCentroid0 := JOIN(iCentroid0temp, dCentroid1, LEFT.x = RIGHT.id AND LEFT.y = RIGHT.number, TRANSFORM(lInput, SELF.values := LEFT.values + [RIGHT.value]; SELF := LEFT;));

iUb0 := PROJECT(ub0_ini, transFormat(LEFT, 2));
iLbs0 := PROJECT(lbs0_ini, transFormat(LEFT, 3));

input0 := iCentroid0 + iUb0 + iLbs0 ;
OUTPUT(input0, NAMED('input0'));


d := input0;
c:=1;

//********************************************start iterations*************************************************
	    OUTPUT(c, NAMED('IterationNo')); 
		dAddedx1 := dAdded1;
			
			dCentroidIn := dCentroid1;
			ub0 := ub0_ini;
			lbs0 := lbs0_ini;
           
			//calculate the deltaC
			deltac1 := dDistanceDelta(c,c-1,dAddedx1);
			OUTPUT(deltac1,NAMED('deltac1'));
		
			//get deltaG1
			//group deltacg by Gt: first JOIN(deltaC with Gt) then group by Gt
			deltaG1 := MAX(deltac1, value);
			OUTPUT(deltaG1,NAMED('deltaG1')); 
       
			//update ub0 and lbs0
			ub1_temp := JOIN(ub0, deltac1, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;));
			OUTPUT(ub1_temp, NAMED('ub1_temp'));
			lbs1_temp := SORT(PROJECT(lbs0,TRANSFORM(Mat.Types.Element, SELF.value := ABS(LEFT.value - deltaG1); SELF := LEFT)),x);
			OUTPUT(lbs1_temp, NAMED('lbs1_temp'));
			
//            dMapa1 := PROJECT(ub0, TRANSFORM(ClusterPair, SELF.id := LEFT.x; SELF.clusterid := LEFT.y; SELF.number := 0; SELF.value01 := LEFT.value; SELF.value02 := 0; SELF.value03 := 0;));		
//			ub1_changed_temp := MappedDistances(d01,dCentroidIn,fDist,dMapa1);	
//			action8 :=OUTPUT(ub1_changed_temp);			
						
			// groupfilter1	
            groupFilter1 := JOIN(lbs1_temp, ub1_temp,LEFT.x = RIGHT.x AND (LEFT.value < RIGHT.value), TRANSFORM(Mat.Types.Element, SELF := RIGHT));	
            OUTPUT(groupFilter1, NAMED('groupFilter1'));	
            
//            dMapa1 := PROJECT(groupFilter1, TRANSFORM(ClusterPair, SELF.id := LEFT.x; SELF.clusterid := LEFT.y; SELF.number := 0; SELF.value01 := LEFT.value; SELF.value02 := 0; SELF.value03 := 0;));		
//			ub1_changed_temp := MappedDistances(d01,dCentroidIn,fDist,dMapa1);	
//			action8 :=OUTPUT(ub1_changed_temp);		
			
			changeSet1 := JOIN(d01, groupFilter1, LEFT.id = RIGHT.x, TRANSFORM(LEFT));		
            OUTPUT(changeSet1, NAMED('changeSet1'));		
            	
			dMappedDistancesb1 := ML.Cluster.Distances(changeSet1,dCentroidIn,fDist);	
			OUTPUT(CHOOSEN(dMappedDistancesb1,1000), ALL,NAMED('dMappedDistancesb1'));	
				
				//old best c all    groupfilter
			ub1_changed_old := JOIN(dMappedDistancesb1, groupFilter1, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, TRANSFORM(LEFT));
			OUTPUT(ub1_changed_old, NAMED('ub1_changed_old'));
				//new best c all	groupfilter
			ub1_changed_final := ML.Cluster.Closest(dMappedDistancesb1);
			OUTPUT(ub1_changed_final, NAMED('ub1_changed_final'));
			
//			ub1_changed := JOIN(ub1_changed_temp, ub1_changed_final, LEFT.x = RIGHT.x AND LEFT.value > RIGHT.value, TRANSFORM(Mat.Types.Element, SELF := RIGHT;));
//			action13 :=OUTPUT(ub1_changed, NAMED('ub1_changed'));

			ub1_changed:= JOIN(ub1_changed_old, ub1_changed_final, LEFT.x = RIGHT.x AND LEFT.value > RIGHT.value, TRANSFORM(Mat.Types.Element, SELF := RIGHT;));
			OUTPUT(ub1_changed, NAMED('ub1_changed'));
			

//			ub1_unchanged := JOIN(ub1_changed_temp, ub1_changed, LEFT.x = RIGHT.x, TRANSFORM(LEFT),LEFT ONLY);
//			action14 :=OUTPUT(ub1_unchanged, NAMED('ub1_unchanged'));
			
			ub1_unchanged_temp := JOIN(ub1_temp, ub1_changed, LEFT.x = RIGHT.x, TRANSFORM(LEFT),LEFT ONLY);
			OUTPUT(ub1_unchanged_temp, NAMED('ub1_unchanged_temp'));
			
			dMapa1 := PROJECT(ub1_unchanged_temp, TRANSFORM(ClusterPair, SELF.id := LEFT.x; SELF.clusterid := LEFT.y; SELF.number := 0; SELF.value01 := LEFT.value; SELF.value02 := 0; SELF.value03 := 0;));		
			ub1_unchanged:= MappedDistances(d01,dCentroidIn,fDist,dMapa1);	
			OUTPUT(ub1_unchanged);	
			
			ub1 := SORT(ub1_changed + ub1_unchanged, x, y, value);
            OUTPUT(ub1, NAMED('ub1'));	
            
			//updating lb
		            					
			lbs1_changed_temp := JOIN(dMappedDistancesb1, ub1_changed, LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF := LEFT;));
			OUTPUT(lbs1_changed_temp, NAMED('lbs1_changed_temp'));
			
			lbs1_changed_temp1 := JOIN(lbs1_changed_temp, ub1_changed, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, TRANSFORM(LEFT), LEFT ONLY);
			OUTPUT(lbs1_changed_temp1, NAMED('lbs1_changed_temp1'));
			
			lbs1_changed_temp2:= GROUP(SORT(lbs1_changed_temp1, x), x);
			OUTPUT(lbs1_changed_temp2, NAMED('lbs1_changed_temp2'));
			
			lbs1_changed := TOPN( lbs1_changed_temp2,1, value);
			OUTPUT(lbs1_changed, NAMED('lbs1_changed'));

//			lbs1_changed := SecondClosest(ub1_changed, lbs1_changed_temp);
			
			lbs1_unchanged:= JOIN(lbs1_temp, lbs1_changed, LEFT.x = RIGHT.x, LEFT ONLY);
            OUTPUT(lbs1_unchanged, NAMED('lbs1_unchanged'));
//           
//           lbs1:= JOIN(lbs1_temp, lbs1_changed,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value), SELF := LEFT;), LEFT OUTER );
		   lbs1 :=SORT( lbs1_unchanged + lbs1_changed, x);			
		   OUTPUT(lbs1, NAMED('lbs1'));
           dClusterCounts1:=TABLE(ub1,{y;UNSIGNED c:=COUNT(GROUP);},y,FEW);
		   OUTPUT(dClusterCounts1, NAMED('dClusterCounts1'));
			
	        // Join closest to the document set and replace the id with the centriod id
	        dClustered1:=SORT(DISTRIBUTE(JOIN(d01,ub1,LEFT.id=RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;),HASH),id),RECORD,LOCAL);
	        // Now roll up on centroid ID, summing up the values for each axis
	        dRolled1:=ROLLUP(dClustered1,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;),id,number,LOCAL);
	        // Join to cluster counts to calculate the new average on each axis
	        dJoined1:=JOIN(dRolled1,dClusterCounts1,LEFT.id=RIGHT.y,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value/RIGHT.c;SELF:=LEFT;),LOOKUP);
	        // Find any centroids with no document allegiance and pass those through also
		    dPass1:=JOIN(dCentroidIn,TABLE(dJoined1,{id},id,LOCAL),LEFT.id=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);
			dCentroid2 := SORT(dPass1 + dJoined1, id);
            OUTPUT(dCentroid2, NAMED('dCentroid2'));  
   
//itr 2

c2:=2;

//********************************************start iterations*************************************************
	    OUTPUT(c2); 
		dAddedx2 := JOIN(dAddedx1, dCentroid2, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
		
			//calculate the deltaC
			deltac2 := dDistanceDelta(c2,c2-1,dAddedx2);
			OUTPUT(deltac2,NAMED('deltac2'));
		
			//get deltaG2
			//group deltacg by Gt: first JOIN(deltaC with Gt) then group by Gt
			deltaG2 := MAX(deltac2, value);
			OUTPUT(deltaG2,NAMED('deltaG2')); 
       
			//update ub0 and lbs0
			ub2_temp := JOIN(ub1, deltac2, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;));
			OUTPUT(ub2_temp, NAMED('ub2_temp'));
			lbs2_temp := PROJECT(lbs1,TRANSFORM(Mat.Types.Element, SELF.value := ABS(LEFT.value - deltaG2); SELF := LEFT));
			OUTPUT(lbs2_temp, NAMED('lbs2_temp'));
		
						
			// groupfilter2	
            groupFilter2 := JOIN(lbs2_temp, ub2_temp,LEFT.x = RIGHT.x AND (LEFT.value < RIGHT.value), TRANSFORM(Mat.Types.Element, SELF := RIGHT));	
            OUTPUT(groupFilter2, NAMED('groupFilter2'));	

			
			changeSet2 := JOIN(d01, groupFilter2, LEFT.id = RIGHT.x, TRANSFORM(LEFT));		
            OUTPUT(changeSet2, NAMED('changeSet2'));		
            	
			dMappedDistancesb2 := ML.Cluster.Distances(changeSet2,dCentroid2,fDist);	
			OUTPUT(dMappedDistancesb2, NAMED('dMappedDistancesb2'));	
				
				//old best c all    groupfilter
			ub2_changed_old := JOIN(dMappedDistancesb2, groupFilter2, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, TRANSFORM(LEFT));
			OUTPUT(ub2_changed_old, NAMED('ub2_changed_old'));
				//new best c all	groupfilter
			ub2_changed_final := ML.Cluster.Closest(dMappedDistancesb2);
			OUTPUT(ub2_changed_final, NAMED('ub2_changed_final'));
			
			ub2_changed:= JOIN(ub2_changed_old, ub2_changed_final, LEFT.x = RIGHT.x AND LEFT.value > RIGHT.value, TRANSFORM(Mat.Types.Element, SELF := RIGHT;));
			OUTPUT(ub2_changed, NAMED('ub2_changed'));
					
			ub2_unchanged_temp := JOIN(ub2_temp, ub2_changed, LEFT.x = RIGHT.x, TRANSFORM(LEFT),LEFT ONLY);
			OUTPUT(ub2_unchanged_temp, NAMED('ub2_unchanged_temp'));
			
			dMapa2 := PROJECT(ub2_unchanged_temp, TRANSFORM(ClusterPair, SELF.id := LEFT.x; SELF.clusterid := LEFT.y; SELF.number := 0; SELF.value01 := LEFT.value; SELF.value02 := 0; SELF.value03 := 0;));		
			ub2_unchanged:= MappedDistances(d01,dCentroid2,fDist,dMapa2);	
			OUTPUT(ub2_unchanged);	
			
			ub2 := SORT(ub2_changed + ub2_unchanged, x, y, value);
            OUTPUT(ub2, NAMED('ub2'));	
            
			//updating lb
		            					
			lbs2_changed_temp := JOIN(dMappedDistancesb2, ub2_changed, LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF := LEFT;));
			OUTPUT(lbs2_changed_temp, NAMED('lbs2_changed_temp'));
			
			lbs2_changed_temp1 := JOIN(lbs2_changed_temp, ub2_changed, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, TRANSFORM(LEFT), LEFT ONLY);
			OUTPUT(lbs2_changed_temp1, NAMED('lbs2_changed_temp1'));
			
			lbs2_changed_temp2:= GROUP(SORT(lbs2_changed_temp1, x), x);
			OUTPUT(lbs2_changed_temp2, NAMED('lbs2_changed_temp2'));
			
			lbs2_changed := TOPN( lbs2_changed_temp2,1, value);
			OUTPUT(lbs2_changed, NAMED('lbs2_changed'));			
			lbs2_unchanged:= JOIN(lbs2_temp, lbs2_changed, LEFT.x = RIGHT.x, LEFT ONLY);
            OUTPUT(lbs2_unchanged, NAMED('lbs2_unchanged'));
		   lbs2 := SORT(lbs2_unchanged + lbs2_changed,x);			
		   OUTPUT(lbs2, NAMED('lbs2'));
           dClusterCounts2:=TABLE(ub2,{y;UNSIGNED c:=COUNT(GROUP);},y,FEW);
		   OUTPUT(dClusterCounts2, NAMED('dClusterCounts2'));
			
	        // Join closest to the document set and replace the id with the centriod id
	        dClustered2:=SORT(DISTRIBUTE(JOIN(d01,ub2,LEFT.id=RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;),HASH),id),RECORD,LOCAL);
	        // Now roll up on centroid ID, summing up the values for each axis
	        dRolled2:=ROLLUP(dClustered2,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;),id,number,LOCAL);
	        // Join to cluster counts to calculate the new average on each axis
	        dJoined2:=JOIN(dRolled2,dClusterCounts2,LEFT.id=RIGHT.y,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value/RIGHT.c;SELF:=LEFT;),LOOKUP);
	        // Find any centroids with no document allegiance and pass those through also
		    dPass2:=JOIN(dCentroidIn,TABLE(dJoined2,{id},id,LOCAL),LEFT.id=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);
			dCentroid3 := SORT(dPass2 + dJoined2, id);
            OUTPUT(dCentroid3, NAMED('dCentroid3'));  
            
//itr3

c3:=3;

//********************************************start iterations*************************************************
	    OUTPUT(c3); 
		dAddedx3 := JOIN(dAddedx2, dCentroid3, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
		
			//calculate the deltaC
			deltac3 := dDistanceDelta(c3,c3-1,dAddedx3);
			OUTPUT(deltac3,NAMED('deltac3'));

			deltaG3 := MAX(deltac3, value);
			OUTPUT(deltaG3,NAMED('deltaG3')); 

			ub3_temp := JOIN(ub2, deltac3, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;));
			OUTPUT(ub3_temp, NAMED('ub3_temp'));
			lbs3_temp := SORT(PROJECT(lbs2,TRANSFORM(Mat.Types.Element, SELF.value := ABS(LEFT.value - deltaG3); SELF := LEFT)),x);
			OUTPUT(lbs3_temp, NAMED('lbs3_temp'));

            groupFilter3 := JOIN(lbs3_temp, ub3_temp,LEFT.x = RIGHT.x AND (LEFT.value < RIGHT.value), TRANSFORM(Mat.Types.Element, SELF := RIGHT));	
            OUTPUT(groupFilter3, NAMED('groupFilter3'));	

			
			changeSet3 := JOIN(d01, groupFilter3, LEFT.id = RIGHT.x, TRANSFORM(LEFT));		
            OUTPUT(changeSet3, NAMED('changeSet3'));		
            	
			dMappedDistancesb3 := ML.Cluster.Distances(changeSet3,dCentroid3,fDist);	
			OUTPUT(dMappedDistancesb3, NAMED('dMappedDistancesb3'));	
				
				//old best c all    groupfilter
			ub3_changed_old := JOIN(dMappedDistancesb3, groupFilter3, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, TRANSFORM(LEFT));
			OUTPUT(ub3_changed_old, NAMED('ub3_changed_old'));
				//new best c all	groupfilter
			ub3_changed_final := ML.Cluster.Closest(dMappedDistancesb3);
			OUTPUT(ub3_changed_final, NAMED('ub3_changed_final'));
			
			ub3_changed:= JOIN(ub3_changed_old, ub3_changed_final, LEFT.x = RIGHT.x AND LEFT.value > RIGHT.value, TRANSFORM(Mat.Types.Element, SELF := RIGHT;));
			OUTPUT(ub3_changed, NAMED('ub3_changed'));
					
			ub3_unchanged_temp := JOIN(ub3_temp, ub3_changed, LEFT.x = RIGHT.x, TRANSFORM(LEFT),LEFT ONLY);
			OUTPUT(ub3_unchanged_temp, NAMED('ub3_unchanged_temp'));
			
			dMapa3 := PROJECT(ub3_unchanged_temp, TRANSFORM(ClusterPair, SELF.id := LEFT.x; SELF.clusterid := LEFT.y; SELF.number := 0; SELF.value01 := LEFT.value; SELF.value02 := 0; SELF.value03 := 0;));		
			ub3_unchanged:= MappedDistances(d01,dCentroid3,fDist,dMapa3);	
			OUTPUT(ub3_unchanged);	
			
			ub3 := SORT(ub3_changed + ub3_unchanged, x, y, value);
            OUTPUT(ub3, NAMED('ub3'));	
            
			//updating lb
		            					
			lbs3_changed_temp := JOIN(dMappedDistancesb3, ub3_changed, LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF := LEFT;));
			OUTPUT(lbs3_changed_temp, NAMED('lbs3_changed_temp'));
			
			lbs3_changed_temp1 := JOIN(lbs3_changed_temp, ub3_changed, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, TRANSFORM(LEFT), LEFT ONLY);
			OUTPUT(lbs3_changed_temp1, NAMED('lbs3_changed_temp1'));
			
			lbs3_changed_temp2:= GROUP(SORT(lbs3_changed_temp1, x), x);
			OUTPUT(lbs3_changed_temp2, NAMED('lbs3_changed_temp2'));
			
			lbs3_changed := TOPN( lbs3_changed_temp2,1, value);
			OUTPUT(lbs3_changed, NAMED('lbs3_changed'));			
			lbs3_unchanged:= JOIN(lbs3_temp, lbs3_changed, LEFT.x = RIGHT.x, LEFT ONLY);
            OUTPUT(lbs3_unchanged, NAMED('lbs3_unchanged'));
		   lbs3 := SORT(lbs3_unchanged + lbs3_changed, x);			
		   OUTPUT(lbs3, NAMED('lbs3'));
           dClusterCounts3:=TABLE(ub3,{y;UNSIGNED c:=COUNT(GROUP);},y,FEW);
		   OUTPUT(dClusterCounts3, NAMED('dClusterCounts3'));
			
	        // Join closest to the document set and replace the id with the centriod id
	        dClustered3:=SORT(DISTRIBUTE(JOIN(d01,ub3,LEFT.id=RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;),HASH),id),RECORD,LOCAL);
	        // Now roll up on centroid ID, summing up the values for each axis
	        dRolled3:=ROLLUP(dClustered3,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;),id,number,LOCAL);
	        // Join to cluster counts to calculate the new average on each axis
	        dJoined3:=JOIN(dRolled3,dClusterCounts3,LEFT.id=RIGHT.y,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value/RIGHT.c;SELF:=LEFT;),LOOKUP);
	        // Find any centroids with no document allegiance and pass those through also
		    dPass3:=JOIN(dCentroidIn,TABLE(dJoined3,{id},id,LOCAL),LEFT.id=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);
			dCentroid4 := SORT(dPass3 + dJoined3, id);
            OUTPUT(dCentroid4, NAMED('dCentroid4'));  

//itr4

c4:=4;

//********************************************start iterations*************************************************
	    OUTPUT(c4); 
		dAddedx4 := JOIN(dAddedx3, dCentroid4, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
		
			//calculate the deltaC
			deltac4 := dDistanceDelta(c4,c4-1,dAddedx4);
			OUTPUT(deltac4,NAMED('deltac4'));

			deltaG4 := MAX(deltac4, value);
			OUTPUT(deltaG4,NAMED('deltaG4')); 

			ub4_temp := JOIN(ub3, deltac4, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;));
			OUTPUT(ub4_temp, NAMED('ub4_temp'));
			lbs4_temp := SORT(PROJECT(lbs3,TRANSFORM(Mat.Types.Element, SELF.value := ABS(LEFT.value - deltaG4); SELF := LEFT)),x);
			OUTPUT(lbs4_temp, NAMED('lbs4_temp'));

            groupFilter4 := JOIN(lbs4_temp, ub4_temp,LEFT.x = RIGHT.x AND (LEFT.value < RIGHT.value), TRANSFORM(Mat.Types.Element, SELF := RIGHT));	
            OUTPUT(groupFilter4, NAMED('groupFilter4'));	

			
			changeSet4 := JOIN(d01, groupFilter4, LEFT.id = RIGHT.x, TRANSFORM(LEFT));		
            OUTPUT(changeSet4, NAMED('changeSet4'));		
            	
			dMappedDistancesb4 := ML.Cluster.Distances(changeSet4,dCentroid4,fDist);	
			OUTPUT(dMappedDistancesb4, NAMED('dMappedDistancesb4'));	
				
				//old best c all    groupfilter
			ub4_changed_old := JOIN(dMappedDistancesb4, groupFilter4, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, TRANSFORM(LEFT));
			OUTPUT(ub4_changed_old, NAMED('ub4_changed_old'));
				//new best c all	groupfilter
			ub4_changed_final := ML.Cluster.Closest(dMappedDistancesb4);
			OUTPUT(ub4_changed_final, NAMED('ub4_changed_final'));
			
			ub4_changed:= JOIN(ub4_changed_old, ub4_changed_final, LEFT.x = RIGHT.x AND LEFT.value > RIGHT.value, TRANSFORM(Mat.Types.Element, SELF := RIGHT;));
			OUTPUT(ub4_changed, NAMED('ub4_changed'));
					
			ub4_unchanged_temp := JOIN(ub4_temp, ub4_changed, LEFT.x = RIGHT.x, TRANSFORM(LEFT),LEFT ONLY);
			OUTPUT(ub4_unchanged_temp, NAMED('ub4_unchanged_temp'));
			
			dMapa4 := PROJECT(ub4_unchanged_temp, TRANSFORM(ClusterPair, SELF.id := LEFT.x; SELF.clusterid := LEFT.y; SELF.number := 0; SELF.value01 := LEFT.value; SELF.value02 := 0; SELF.value03 := 0;));		
			ub4_unchanged:= MappedDistances(d01,dCentroid4,fDist,dMapa4);	
			OUTPUT(ub4_unchanged);	
			
			ub4 := SORT(ub4_changed + ub4_unchanged, x, y, value);
            OUTPUT(ub4, NAMED('ub4'));	
            
			//updating lb
		            					
			lbs4_changed_temp := JOIN(dMappedDistancesb4, ub4_changed, LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF := LEFT;));
			OUTPUT(lbs4_changed_temp, NAMED('lbs4_changed_temp'));
			
			lbs4_changed_temp1 := JOIN(lbs4_changed_temp, ub4_changed, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, TRANSFORM(LEFT), LEFT ONLY);
			OUTPUT(lbs4_changed_temp1, NAMED('lbs4_changed_temp1'));
			
			lbs4_changed_temp2:= GROUP(SORT(lbs4_changed_temp1, x), x);
			OUTPUT(lbs4_changed_temp2, NAMED('lbs4_changed_temp2'));
			
			lbs4_changed := TOPN( lbs4_changed_temp2,1, value);
			OUTPUT(lbs4_changed, NAMED('lbs4_changed'));			
			lbs4_unchanged:= JOIN(lbs4_temp, lbs4_changed, LEFT.x = RIGHT.x, LEFT ONLY);
            OUTPUT(lbs4_unchanged, NAMED('lbs4_unchanged'));
		   lbs4 := SORT(lbs4_unchanged + lbs4_changed, x);			
		   OUTPUT(lbs4, NAMED('lbs4'));
           dClusterCounts4:=TABLE(ub4,{y;UNSIGNED c:=COUNT(GROUP);},y,FEW);
		   OUTPUT(dClusterCounts4, NAMED('dClusterCounts4'));
			
	        // Join closest to the document set and replace the id with the centriod id
	        dClustered4:=SORT(DISTRIBUTE(JOIN(d01,ub4,LEFT.id=RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;),HASH),id),RECORD,LOCAL);
	        // Now roll up on centroid ID, summing up the values for each axis
	        dRolled4:=ROLLUP(dClustered4,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;),id,number,LOCAL);
	        // Join to cluster counts to calculate the new average on each axis
	        dJoined4:=JOIN(dRolled4,dClusterCounts4,LEFT.id=RIGHT.y,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value/RIGHT.c;SELF:=LEFT;),LOOKUP);
	        // Find any centroids with no document allegiance and pass those through also
		    dPass4:=JOIN(dCentroidIn,TABLE(dJoined4,{id},id,LOCAL),LEFT.id=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);
			dCentroid5 := SORT(dPass4 + dJoined4, id);
            OUTPUT(dCentroid5, NAMED('dCentroid5'));
//itr5

c5:=5;

//********************************************start iterations*************************************************
	    OUTPUT(c5); 
		dAddedx5 := JOIN(dAddedx4, dCentroid5, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
		
			//calculate the deltaC
			deltac5 := dDistanceDelta(c5,c5-1,dAddedx5);
			OUTPUT(deltac5,NAMED('deltac5'));

			deltaG5 := MAX(deltac5, value);
			OUTPUT(deltaG5,NAMED('deltaG5')); 

			ub5_temp := JOIN(ub4, deltac5, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;));
			OUTPUT(ub5_temp, NAMED('ub5_temp'));
			lbs5_temp := SORT(PROJECT(lbs4,TRANSFORM(Mat.Types.Element, SELF.value := ABS(LEFT.value - deltaG5); SELF := LEFT)),x);
			OUTPUT(lbs5_temp, NAMED('lbs5_temp'));

            groupFilter5 := JOIN(lbs5_temp, ub5_temp,LEFT.x = RIGHT.x AND (LEFT.value < RIGHT.value), TRANSFORM(Mat.Types.Element, SELF := RIGHT));	
            OUTPUT(groupFilter5, NAMED('groupFilter5'));	

			
			changeSet5 := JOIN(d01, groupFilter5, LEFT.id = RIGHT.x, TRANSFORM(LEFT));		
            OUTPUT(changeSet5, NAMED('changeSet5'));		
            	
			dMappedDistancesb5 := ML.Cluster.Distances(changeSet5,dCentroid5,fDist);	
			OUTPUT(dMappedDistancesb5, NAMED('dMappedDistancesb5'));	
				
				//old best c all    groupfilter
			ub5_changed_old := JOIN(dMappedDistancesb5, groupFilter5, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, TRANSFORM(LEFT));
			OUTPUT(ub5_changed_old, NAMED('ub5_changed_old'));
				//new best c all	groupfilter
			ub5_changed_final := ML.Cluster.Closest(dMappedDistancesb5);
			OUTPUT(ub5_changed_final, NAMED('ub5_changed_final'));
			
			ub5_changed:= JOIN(ub5_changed_old, ub5_changed_final, LEFT.x = RIGHT.x AND LEFT.value > RIGHT.value, TRANSFORM(Mat.Types.Element, SELF := RIGHT;));
			OUTPUT(ub5_changed, NAMED('ub5_changed'));
					
			ub5_unchanged_temp := JOIN(ub5_temp, ub5_changed, LEFT.x = RIGHT.x, TRANSFORM(LEFT),LEFT ONLY);
			OUTPUT(ub5_unchanged_temp, NAMED('ub5_unchanged_temp'));
			
			dMapa5 := PROJECT(ub5_unchanged_temp, TRANSFORM(ClusterPair, SELF.id := LEFT.x; SELF.clusterid := LEFT.y; SELF.number := 0; SELF.value01 := LEFT.value; SELF.value02 := 0; SELF.value03 := 0;));		
			ub5_unchanged:= MappedDistances(d01,dCentroid5,fDist,dMapa5);	
			OUTPUT(ub5_unchanged);	
			
			ub5 := SORT(ub5_changed + ub5_unchanged, x, y, value);
            OUTPUT(ub5, NAMED('ub5'));	
            
			//updating lb
		            					
			lbs5_changed_temp := JOIN(dMappedDistancesb5, ub5_changed, LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF := LEFT;));
			OUTPUT(lbs5_changed_temp, NAMED('lbs5_changed_temp'));
			
			lbs5_changed_temp1 := JOIN(lbs5_changed_temp, ub5_changed, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, TRANSFORM(LEFT), LEFT ONLY);
			OUTPUT(lbs5_changed_temp1, NAMED('lbs5_changed_temp1'));
			
			lbs5_changed_temp2:= GROUP(SORT(lbs5_changed_temp1, x), x);
			OUTPUT(lbs5_changed_temp2, NAMED('lbs5_changed_temp2'));
			
			lbs5_changed := TOPN( lbs5_changed_temp2,1, value);
			OUTPUT(lbs5_changed, NAMED('lbs5_changed'));			
			lbs5_unchanged:= JOIN(lbs5_temp, lbs5_changed, LEFT.x = RIGHT.x, LEFT ONLY);
            OUTPUT(lbs5_unchanged, NAMED('lbs5_unchanged'));
		   lbs5 := SORT(lbs5_unchanged + lbs5_changed, x);			
		   OUTPUT(lbs5, NAMED('lbs5'));
           dClusterCounts5:=TABLE(ub5,{y;UNSIGNED c:=COUNT(GROUP);},y,FEW);
		   OUTPUT(dClusterCounts5, NAMED('dClusterCounts5'));
			
	        // Join closest to the document set and replace the id with the centriod id
	        dClustered5:=SORT(DISTRIBUTE(JOIN(d01,ub5,LEFT.id=RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;),HASH),id),RECORD,LOCAL);
	        // Now roll up on centroid ID, summing up the values for each axis
	        dRolled5:=ROLLUP(dClustered5,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;),id,number,LOCAL);
	        // Join to cluster counts to calculate the new average on each axis
	        dJoined5:=JOIN(dRolled5,dClusterCounts5,LEFT.id=RIGHT.y,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value/RIGHT.c;SELF:=LEFT;),LOOKUP);
	        // Find any centroids with no document allegiance and pass those through also
		    dPass5:=JOIN(dCentroidIn,TABLE(dJoined5,{id},id,LOCAL),LEFT.id=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);
			dCentroid6 := SORT(dPass5 + dJoined5, id);
            OUTPUT(dCentroid6, NAMED('dCentroid6'));
   