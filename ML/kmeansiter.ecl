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
//
///**
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
{101,1,1},
{102,2,2},
{103,3,3},
{104,4,4}
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

//make sure all ids are different
//iOffset:=IF(MAX(d01,id)>MIN(d02,id),MAX(d01,id),0);
//transfrom the d02 to internal data structure
//d02Prep:=PROJECT(d02,TRANSFORM(lIterations,SELF.id:=LEFT.id+iOffset;SELF.values:=[LEFT.value];SELF:=LEFT;));
//dCentroid0 := PROJECT(d02Prep,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[1];SELF:=LEFT;));
//OUTPUT(dCentroid0,NAMED('Initial_Centroids')); 

d02Prep:=PROJECT(d02,TRANSFORM(lIterations,SELF.id:=LEFT.id;SELF.values:=[LEFT.value];SELF:=LEFT;));
dCentroid0 := dCentroids;
OUTPUT(dCentroid0,NAMED('Initial_Centroids')); 
c1:=1;
//********************************************start iterations*************************************************
			bConverged1:=False;
			OUTPUT(bConverged1, NAMED('bConverged1')); 
			dDistances1 := ML.Cluster.Distances(d01,dCentroid0);
			OUTPUT(dDistances1, NAMED('dDistances1'));
			dClosest1:=ML.Cluster.Closest(dDistances1);
			OUTPUT(dClosest1, NAMED('dClosest1'));
			dClusterCounts1:=TABLE(dClosest1,{y;UNSIGNED c:=COUNT(GROUP);},y,FEW);
			OUTPUT(dClusterCounts1, NAMED('dClusterCounts1'));
			dClustered1:=SORT(DISTRIBUTE(JOIN(d01,dClosest1,LEFT.id=RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;),HASH),id),RECORD,LOCAL);
		    OUTPUT(dClustered1, NAMED('dClustered1'));
		    dRolled1:=ROLLUP(dClustered1,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;),id,number,LOCAL);
		    OUTPUT(dRolled1, NAMED('dRolled1'));
            dJoined1:=JOIN(dRolled1,dClusterCounts1,LEFT.id=RIGHT.y,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value/RIGHT.c;SELF:=LEFT;),LOOKUP);		    
            OUTPUT(dJoined1, NAMED('dJoined1'));
            dPass1:=JOIN(dCentroid0,TABLE(dJoined1,{id},id,LOCAL),LEFT.id=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);
            OUTPUT(dPass1, NAMED('dPass1'));
            dCentroid1 := dJoined1+dPass1;
            OUTPUT(dCentroid1, NAMED('dCentroid1'));
            dResult1:=JOIN(d02Prep,dCentroid1,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=RIGHT;),RIGHT OUTER);
            OUTPUT(dResult1, NAMED('dResult1'));
            dDelta1 := dDistanceDelta(c1 ,c1 - 1,dResult1);
            OUTPUT(dDelta1, NAMED('dDelta1'));
             
c2 := 2;
			bConverged2:= MAX(dDelta1,value)<=nConverge;
			OUTPUT(bConverged2, NAMED('bConverged2'));       
			dDistances2 := ML.Cluster.Distances(d01,dCentroid1);
			OUTPUT(dDistances2, NAMED('dDistances2'));
			dClosest2:=ML.Cluster.Closest(dDistances2);
			OUTPUT(dClosest2, NAMED('dClosest2'));
			dClusterCounts2:=TABLE(dClosest2,{y;UNSIGNED c:=COUNT(GROUP);},y,FEW);
			OUTPUT(dClusterCounts2, NAMED('dClusterCounts2'));
			dClustered2:=SORT(DISTRIBUTE(JOIN(d01,dClosest2,LEFT.id=RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;),HASH),id),RECORD,LOCAL);
		    OUTPUT(dClustered2, NAMED('dClustered2'));
		    dRolled2:=ROLLUP(dClustered2,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;),id,number,LOCAL);
		    OUTPUT(dRolled2, NAMED('dRolled2'));
            dJoined2:=JOIN(dRolled2,dClusterCounts2,LEFT.id=RIGHT.y,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value/RIGHT.c;SELF:=LEFT;),LOOKUP);		    
            OUTPUT(dJoined2, NAMED('dJoined2'));
            dPass2:=JOIN(dCentroid1,TABLE(dJoined2,{id},id,LOCAL),LEFT.id=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);
            OUTPUT(dPass2, NAMED('dPass2'));
            dCentroid2 := dJoined2+dPass2;
            OUTPUT(dCentroid2, NAMED('dCentroid2'));
            dResult2:=JOIN(dResult1,dCentroid2,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=RIGHT;),RIGHT OUTER);
            OUTPUT(dResult2, NAMED('dResult2'));
            dDelta2 := dDistanceDelta(c2 ,c2 - 1,dResult2);
            OUTPUT(dDelta2, NAMED('dDelta2'));    
c3 := 3;
			bConverged3:= MAX(dDelta2,value)<=nConverge;
			OUTPUT(bConverged3, NAMED('bConverged3'));       
			dDistances3 := ML.Cluster.Distances(d01,dCentroid2);
			OUTPUT(dDistances3, NAMED('dDistances3'));
			dClosest3:=ML.Cluster.Closest(dDistances3);
			OUTPUT(dClosest3, NAMED('dClosest3'));
			dClusterCounts3:=TABLE(dClosest3,{y;UNSIGNED c:=COUNT(GROUP);},y,FEW);
			OUTPUT(dClusterCounts3, NAMED('dClusterCounts3'));
			dClustered3:=SORT(DISTRIBUTE(JOIN(d01,dClosest3,LEFT.id=RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;),HASH),id),RECORD,LOCAL);
		    OUTPUT(dClustered3, NAMED('dClustered3'));
		    dRolled3:=ROLLUP(dClustered3,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;),id,number,LOCAL);
		    OUTPUT(dRolled3, NAMED('dRolled3'));
            dJoined3:=JOIN(dRolled3,dClusterCounts3,LEFT.id=RIGHT.y,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value/RIGHT.c;SELF:=LEFT;),LOOKUP);		    
            OUTPUT(dJoined3, NAMED('dJoined3'));
            dPass3:=JOIN(dCentroid1,TABLE(dJoined3,{id},id,LOCAL),LEFT.id=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);
            OUTPUT(dPass3, NAMED('dPass3'));
            dCentroid3 := dJoined3+dPass3;
            OUTPUT(dCentroid3, NAMED('dCentroid3'));
            dResult3:=JOIN(dResult2,dCentroid3,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=RIGHT;),RIGHT OUTER);
            OUTPUT(dResult3, NAMED('dResult3'));
            dDelta3 := dDistanceDelta(c3 ,c3 - 1,dResult3);
            OUTPUT(dDelta3, NAMED('dDelta3'));  

c4 := 4;
			bConverged4:= MAX(dDelta3,value)<=nConverge;
			OUTPUT(bConverged4, NAMED('bConverged4'));       
			dDistances4 := ML.Cluster.Distances(d01,dCentroid3);
			OUTPUT(dDistances4, NAMED('dDistances4'));
			dClosest4:=ML.Cluster.Closest(dDistances4);
			OUTPUT(dClosest4, NAMED('dClosest4'));
			dClusterCounts4:=TABLE(dClosest4,{y;UNSIGNED c:=COUNT(GROUP);},y,FEW);
			OUTPUT(dClusterCounts4, NAMED('dClusterCounts4'));
			dClustered4:=SORT(DISTRIBUTE(JOIN(d01,dClosest4,LEFT.id=RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;),HASH),id),RECORD,LOCAL);
		    OUTPUT(dClustered4, NAMED('dClustered4'));
		    dRolled4:=ROLLUP(dClustered4,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;),id,number,LOCAL);
		    OUTPUT(dRolled4, NAMED('dRolled4'));
            dJoined4:=JOIN(dRolled4,dClusterCounts4,LEFT.id=RIGHT.y,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value/RIGHT.c;SELF:=LEFT;),LOOKUP);		    
            OUTPUT(dJoined4, NAMED('dJoined4'));
            dPass4:=JOIN(dCentroid3,TABLE(dJoined4,{id},id,LOCAL),LEFT.id=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);
            OUTPUT(dPass4, NAMED('dPass4'));
            dCentroid4 := dJoined4+dPass4;
            OUTPUT(dCentroid4, NAMED('dCentroid4'));
            dResult4:=JOIN(dResult3,dCentroid4,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=RIGHT;),RIGHT OUTER);
            OUTPUT(dResult4, NAMED('dResult4'));
            dDelta4 := dDistanceDelta(c4 ,c4 - 1,dResult4);
            OUTPUT(dDelta4, NAMED('dDelta4'));  
c5 := 5;
			//***
			bConverged5:= MAX(dDelta4,value)<=nConverge;
			OUTPUT(bConverged5, NAMED('bConverged5'));       
			dDistances5 := ML.Cluster.Distances(d01,dCentroid3);
			OUTPUT(dDistances5, NAMED('dDistances5'));
			dClosest5:=ML.Cluster.Closest(dDistances5);
			OUTPUT(dClosest5, NAMED('dClosest5'));
			dClusterCounts5:=TABLE(dClosest5,{y;UNSIGNED c:=COUNT(GROUP);},y,FEW);
			OUTPUT(dClusterCounts5, NAMED('dClusterCounts5'));
			dClustered5:=SORT(DISTRIBUTE(JOIN(d01,dClosest5,LEFT.id=RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;),HASH),id),RECORD,LOCAL);
		    OUTPUT(dClustered5, NAMED('dClustered5'));
		    dRolled5:=ROLLUP(dClustered5,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;),id,number,LOCAL);
		    OUTPUT(dRolled5, NAMED('dRolled5'));
            dJoined5:=JOIN(dRolled5,dClusterCounts5,LEFT.id=RIGHT.y,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value/RIGHT.c;SELF:=LEFT;),LOOKUP);		    
            OUTPUT(dJoined5, NAMED('dJoined5'));
            dPass5:=JOIN(dCentroid4,TABLE(dJoined5,{id},id,LOCAL),LEFT.id=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);
            OUTPUT(dPass5, NAMED('dPass5'));
            dCentroid5 := dJoined5+dPass5;
            OUTPUT(dCentroid5, NAMED('dCentroid5'));
            //***
            dResult5:=JOIN(dResult4,dCentroid5,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=RIGHT;),RIGHT OUTER);
            OUTPUT(dResult5, NAMED('dResult5'));
            dDelta5 := dDistanceDelta(c5 ,c5 - 1,dResult5);
            OUTPUT(dDelta5, NAMED('dDelta5')); 
			
