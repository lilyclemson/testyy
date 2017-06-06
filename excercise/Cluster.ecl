<<<<<<< HEAD
﻿//-----------------------------------------------------------------------------
// Module used to cluster perform clustering on data in the NumericField
// format.  Includes functions for calculating distance using many different
// algorithms, determining centroid allegiance based on those distances, and
// performing K-Means calculations.
//-----------------------------------------------------------------------------

IMPORT * FROM $;
IMPORT Std.Str AS Str;
IMPORT ML.Mat;

EXPORT Cluster := MODULE
// Working structure for cluster distance logic
  SHARED ClusterPair:=RECORD
		Types.t_RecordID    id;
		Types.t_RecordID    clusterid;
		Types.t_FieldNumber number;
		Types.t_FieldReal   value01 := 0;
		Types.t_FieldReal   value02 := 0;
		Types.t_FieldReal   value03 := 0;
  END;
	
	 SHARED lIterations:=RECORD
    TYPEOF(Types.NumericField.id) id;
    TYPEOF(Types.NumericField.number) number;
    SET OF TYPEOF(Types.NumericField.value) values;
   END;
  
	// Compute the 'N^2' distance metric
	// The DF module contains the different distance functions
	// Note that this is a very flexible interface - it allows for a variety of 'speeds' of computation

	SHARED c_model := ENUM ( Dense = 1, SJoins = 2, Background = 4 );
  EXPORT DF := MODULE
	  // Each nested module is a 'control' interface for pairsB
		// Dense model - sparse vectors will be made dense - very simple handling of nulls - usually slow
		// Summary Joins - dimensions that are both non-null score directly; each combined score gets passed by the data from a join the summarizes either side
		// Background - a 'background' N^2 matrix is constructed from the summary joins - which is then merged with the dimension matched data
		// -- note Background REQUIRES that d02 fit in memory
		// 0 ==> score constructed only from co-populated dimensions and any EV1/2 stats computed
		// NAMING: A leading Q implies a 'quick' version of the result that probably shaves a corner or two
		//         A leading W implies a 'wide' version and is probably simple, unrestricted and painful
		//         No leading letter implies our 'best shot' at the 'correct' result
    EXPORT Default := MODULE,VIRTUAL
		  EXPORT UNSIGNED1 PModel := c_model.dense; // The process model for this distance metric
			EXPORT REAL8 EV1(DATASET(Types.NumericField) d) := 0; // An 'exotic' value which will be passed in at Comb time
			EXPORT REAL8 EV2(DATASET(Types.NumericField) d) := 0; // An 'exotic' value which will be passed in at Comb time
			EXPORT BOOLEAN JoinFilter(Types.t_FieldReal x,Types.t_FieldReal y,REAL8 ex1) := x<>0 OR y<>0; // If false - join value will not be computed
			EXPORT IV1(Types.t_FieldReal x,Types.t_FieldReal y) := x;
			EXPORT IV2(Types.t_FieldReal x,Types.t_FieldReal y) := y;
			EXPORT Types.t_FieldReal Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := 0; // The value1 - eventual result
			EXPORT Types.t_FieldReal Comb2(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := 0; // Scratchpad
			EXPORT Types.t_FieldReal Comb3(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := 0; // Scratchpad
// These can be though of as a 'turbo' or summary-join interface
// They will usually be used to prevent the need for dense computation
			EXPORT SummaryID1(DATASET(Types.NumericField) d) := d; // Used to create some form of summary by ID for dataset 1
			EXPORT SummaryID2(DATASET(Types.NumericField) d) := SummaryID1(d); // Used to create some form of summary by ID for dataset 2
			EXPORT Types.t_FieldReal Join11(ClusterPair im,Types.NumericField ri) := 0; // join 1 result 1
			EXPORT Types.t_FieldReal Join12(ClusterPair im,Types.NumericField ri) := 0;
			EXPORT Types.t_FieldReal Join13(ClusterPair im,Types.NumericField ri) := 0;
			EXPORT Types.t_FieldReal Join21(ClusterPair im,Types.NumericField ri) := 0;  // join 2 result 1
// This is the 'background' interface			
      EXPORT Types.t_FieldReal Background(Types.NumericField va1,Types.NumericField va2) := 0;// Compute background value from SI1/SI2 value
      EXPORT Types.t_FieldReal BackFront(Mat.Types.Element Back,ClusterPair Fro) := IF ( Fro.id>0, Fro.value01, Back.value );// Compute value given a background and a clusterpair <by default take front if possible>
	  END;

// These models compute a 'proper' Euclidean result but only for those vectors that have at least one dimension along
// which they both have a non-zero value. For very sparse vectors this will produce a result MUCH smaller than
// N^2 (and is correspondingly faster)		
    EXPORT QEuclideanSquared := MODULE(Default),VIRTUAL
		  EXPORT UNSIGNED1 PModel := c_model.SJoins;
			EXPORT SummaryID1(DATASET(Types.NumericField) d) := PROJECT(TABLE( d, { id, val := SUM(GROUP,value*value); }, id ),TRANSFORM(Types.NumericField,SELF.value:=LEFT.val,SELF.number:=0,SELF.id:=LEFT.id));
			EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := SUM(D,(Value01-Value02)*(Value01-Value02));
			EXPORT Comb2(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := SUM(D,Value01*Value01);
			EXPORT Comb3(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := SUM(D,Value02*Value02); // sum of all the contributing rhs
			EXPORT Join11(ClusterPair im,Types.NumericField ri) := im.value01 + ( ri.value-im.value02 ); // add in all of the lhs^2 that did not match
			EXPORT Join13(ClusterPair im,Types.NumericField ri) := im.value03; // keep the rhs^2
			EXPORT Join21(ClusterPair im,Types.NumericField ri) := im.value01 + ( ri.value-im.value03 ); // add in all of the rhs^2 that did not match
    END;
    EXPORT QEuclidean := MODULE(QEuclideanSquared)
			EXPORT Join21(ClusterPair im,Types.NumericField ri) := SQRT(im.value01 + ( ri.value-im.value03 )); // add in all of the rhs^2 that did not match
    END;

// These models compute a proper Euclidean result (the full N^2) - they do require that D02 be able to fit in memory
// However given this is N^2 - if N does not fit in memory - you are probably dead anyway
    EXPORT EuclideanSquared := MODULE(QEuclideanSquared),VIRTUAL // QEuclidean with a background added
		  EXPORT UNSIGNED1 PModel := c_model.Background; // We avoid the SJoins through cleverness in BackFront
      EXPORT Types.t_FieldReal Background(Types.NumericField va1,Types.NumericField va2) := va1.value+va2.value;
      EXPORT Types.t_FieldReal BackFront(Mat.Types.Element Back,ClusterPair Fro) := IF ( Fro.id>0, Back.value+Fro.value01-Fro.value02-Fro.value03, Back.value );
    END;
    EXPORT Euclidean := MODULE(EuclideanSquared)
      EXPORT Types.t_FieldReal BackFront(Mat.Types.Element Back,ClusterPair Fro) := SQRT(IF ( Fro.id>0, Back.value+Fro.value01-Fro.value02-Fro.value03, Back.value ));
    END;
    EXPORT Manhattan:=MODULE(Default),VIRTUAL
      EXPORT UNSIGNED1 PModel := c_model.Background;
      EXPORT SummaryID1(DATASET(Types.NumericField) d) := PROJECT(TABLE(d,{id,val:=SUM(GROUP,value);},id),TRANSFORM(Types.NumericField,SELF.value:=LEFT.val,SELF.number:=0,SELF.id:=LEFT.id));
      EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2):=SUM(d,ABS(Value01-Value02));
      EXPORT Comb2(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2):=SUM(D,Value01);
      EXPORT Comb3(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2):=SUM(D,Value02);
      EXPORT Types.t_FieldReal Background(Types.NumericField va1,Types.NumericField va2):=va1.value+va2.value;
      EXPORT Types.t_FieldReal BackFront(Mat.Types.Element Back,ClusterPair Fro):=IF(Fro.id>0,Back.value+Fro.value01-Fro.value02-Fro.value03,Back.value);
    END;
    EXPORT Cosine := MODULE(Default),VIRTUAL
      EXPORT UNSIGNED1 PModel := c_model.Background;
      EXPORT SummaryID1(DATASET(Types.NumericField) d) := PROJECT(TABLE(d,{id,val:=SQRT(SUM(GROUP,(value*value)));},id),TRANSFORM(Types.NumericField,SELF.value:=LEFT.val,SELF.number:=0,SELF.id:=LEFT.id));
      EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := SUM(d,Value01*Value02);
      EXPORT Types.t_FieldReal Background(Types.NumericField va1,Types.NumericField va2):=va1.value*va2.value;
      EXPORT Types.t_FieldReal BackFront(Mat.Types.Element Back,ClusterPair Fro):=IF(Fro.id>0,1.0-(Fro.Value01/Back.value),1.0);
    END;
    EXPORT Tanimoto := MODULE(Cosine),VIRTUAL
      EXPORT Types.t_FieldReal BackFront(Mat.Types.Element Back,ClusterPair Fro):=IF(Fro.id>0,1.0-(Fro.Value01/(Back.value-Fro.Value01)),1.0);
    END;
// These compute full Euclidean	the 'simple' way and have no obvious restrictions
// Expect to wait a while
    EXPORT WEuclideanSquared := MODULE(Default),VIRTUAL
			EXPORT IV1(Types.t_FieldReal x,Types.t_FieldReal y) := (x-y)*(x-y);
			EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := SUM(D,Value01); 
    END;
    EXPORT WEuclidean := MODULE(WEuclideanSquared)
			EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := SQRT( SUM(D,Value01) );
    END;
    EXPORT WManhattan := MODULE(Default),VIRTUAL
			EXPORT IV1(Types.t_FieldReal x,Types.t_FieldReal y) := ABS(x-y);
			EXPORT IV2(Types.t_FieldReal x,Types.t_FieldReal y) := 0;
			EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := SUM(D,Value01);
    END;
		EXPORT Maximum := MODULE(WManhattan)
			EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := MAX(D,Value01);
		END;
    EXPORT WCosine := MODULE(Default),VIRTUAL
			EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := 1-SUM(D,Value01*Value02)/( SQRT(SUM(D,Value01*Value01))*SQRT(SUM(D,Value02*Value02)));
    END;
    EXPORT WTanimoto := MODULE(Default),VIRTUAL
			EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := 1-SUM(D,Value01*Value02)/( SQRT(SUM(D,Value01*Value01))*SQRT(SUM(D,Value02*Value02))-SUM(D,Value01*Value02));
    END;
		// Now for some quick and dirty functions
		// This attempts to approximate the missing values - it will have far few intermediates if the matrices were sparse
		EXPORT MissingAppx := MODULE(Default),VIRTUAL
		  EXPORT UNSIGNED1 Pmodel := 0;
			EXPORT REAL8 EV1(DATASET(Types.NumericField) d) := AVE(d,value); // Average value
			EXPORT REAL8 EV2(DATASET(Types.NumericField) d) := MAX(TABLE(d,{UNSIGNED C := COUNT(GROUP)},id),C);
			EXPORT BOOLEAN JoinFilter(Types.t_FieldReal x,Types.t_FieldReal y,REAL8 ex1) := (x<>0 OR y<>0) AND ABS(x-y)<ex1; // Only produce record if closer
			EXPORT Types.t_FieldReal IV1(Types.t_FieldReal x,Types.t_FieldReal y) := ABS(x-y);
			EXPORT Types.t_FieldReal Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := SUM(d,value01) + (ev2-COUNT(d))*ev1;
		END;

		// Co-occurences - only counts number of fields with exact matches
		// For this metric missing values are 'infinity'
		EXPORT CoOccur := MODULE(Default),VIRTUAL
		  EXPORT UNSIGNED1 Pmodel := 0;
			EXPORT REAL8 EV1(DATASET(Types.NumericField) d) := MAX(d,number);
			EXPORT BOOLEAN JoinFilter(Types.t_FieldReal x,Types.t_FieldReal y,REAL8 ex1) := x<>0 AND x=y;
			EXPORT Types.t_FieldReal IV1(Types.t_FieldReal x,Types.t_FieldReal y) := 1;
			EXPORT Types.t_FieldReal Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := ev1 - COUNT(d);
		END;
	END;

// This is the 'distance computation engine'. It extremely configurable - see the 'Control' parameter
	EXPORT Distances(DATASET(Types.NumericField) d01,DATASET(Types.NumericField) d02,DF.Default Control = DF.WEuclidean) := FUNCTION
		// If we are in dense model then fatten up the records; otherwise zeroes not needed
		df1 := IF( Control.Pmodel & c_model.dense > 0, Utils.Fat(d01), d01(value<>0) );
		df2 := IF( Control.Pmodel & c_model.dense > 0, Utils.Fat(d02), d02(value<>0) );
		// Construct the summary records used by SJoins and Background processing models
		si1 := Control.SummaryID1(df1); // Summaries of each document by ID
		si2 := Control.SummaryID2(df2); // May be used by any summary joins features
		// Construct the 'background' matrix from the summary matrix
		bck := JOIN(si1,si2,LEFT.id<>RIGHT.id,TRANSFORM(Mat.Types.Element,SELF.x := LEFT.id, SELF.y := RIGHT.Id, SELF.value := Control.BackGround(LEFT,RIGHT)),ALL);
		// Create up to two 'aggregate' numbers that the models may use
		ex1 := Control.EV1(d01); 
		ex2 := Control.EV2(d01);
		// This is the principle N^2 join (although some join filters can improve on that)
		ClusterPair Take2(df1 le,df2 ri) := TRANSFORM
		  SELF.clusterid := ri.id;
			SELF.id := le.id;
			SELF.number := le.number;
			SELF.value01 := Control.IV1(le.value,ri.value);
			SELF.value02 := Control.IV2(le.value,ri.value);
		END;
		J := JOIN(df1,df2,LEFT.number=RIGHT.number AND LEFT.id<>RIGHT.id AND Control.JoinFilter(LEFT.value,RIGHT.value,ex1),Take2(LEFT,RIGHT),HASH); // numbers will be evenly distribute by definition
		// Take all of the values computed for each matching ID and combine them
		JG := GROUP(J,clusterid,id,ALL);
		ClusterPair roll(ClusterPair le, DATASET(ClusterPair) gd) := TRANSFORM
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
    Result:=IF(Control.PModel & c_model.Background>0,BF,ProAsDist);
    
    // If the d02 IDs were adjusted to avoid intersection, revert them back
    // to their original numbers before returning the results.
		RETURN Result;
	END;
	SHARED tempR := RECORD
		TYPEOF(Types.NumericField.id) id;
		TYPEOF(Types.NumericField.number) number;
		TYPEOF(Types.NumericField.value) value;
		TYPEOF(Types.NumericField.id) cid:=0 ;
	END;
	
	EXPORT yyDistances(DATASET(Types.NumericField) d01,DATASET(Types.NumericField) d02, DATASET(Mat.Types.Element) dMap ,DF.Default Control = DF.WEuclidean) := FUNCTION
			// If we are in dense model then fatten up the records; otherwise zeroes not needed
			df1 := IF( Control.Pmodel & c_model.dense > 0, Utils.Fat(d01), d01(value<>0) );
			df2 := IF( Control.Pmodel & c_model.dense > 0, Utils.Fat(d02), d02(value<>0) );
			//Mapped dataset
			dNew := JOIN(dMap, d01, LEFT.x = RIGHT.id, TRANSFORM(tempR, SELF.cid := LEFT.y; SELF := RIGHT; ));//dRelation

			// This is the principle N^2 join (although some join filters can improve on that)
			ClusterPair Take2(dNew le,df2 ri) := TRANSFORM
				SELF.clusterid := ri.id;
				SELF.id := le.id;
				SELF.number := le.number;
				SELF.value01 := Control.IV1(le.value,ri.value);
				SELF.value02 := Control.IV2(le.value,ri.value);
			END;
			// df1 : replace the id of  data points with the cid and df2 is just the cendroid set
			J := JOIN(dNew,df2,LEFT.number=RIGHT.number AND LEFT.cid = RIGHT.id ,Take2(LEFT,RIGHT),HASH); // numbers will be evenly distribute by definition

			// Take all of the values computed for each matching ID and combine them
			JG := GROUP(J,id,clusterid, ALL);
			// Create up to two 'aggregate' numbers that the models may use
			ex1 := Control.EV1(d01); 
			ex2 := Control.EV2(d01);
			ClusterPair  roll(ClusterPair le, DATASET(ClusterPair) gd) := TRANSFORM
				SELF.Value01 := Control.Comb1(gd,ex1,ex2);
				SELF.Value02 := Control.Comb2(gd,ex1,ex2); // These are really scratchpad
				SELF.Value03 := Control.Comb3(gd,ex1,ex2);
				SELF := le;
			END;
			
			rld := ROLLUP(JG,GROUP,roll(LEFT,ROWS(LEFT)));
			Result := PROJECT(rld,TRANSFORM(Mat.Types.Element,SELF.x := LEFT.id,SELF.y := LEFT.clusterid,SELF.Value := LEFT.Value01, SELF := LEFT));
    
			// If the d02 IDs were adjusted to avoid intersection, revert them back
			// to their original numbers before returning the results.
			RETURN Result;
	END; 
  
  //---------------------------------------------------------------------------
  // Closest takes a set of distances and returns a collapsed set containing
  // only the row for each id with the closest centroid
  //---------------------------------------------------------------------------
  EXPORT Closest(DATASET(Mat.Types.Element) dDistances):=DEDUP(SORT(DISTRIBUTE(dDistances,x),x,value,LOCAL),x,LOCAL);

  //---------------------------------------------------------------------------
  // Suite of functions to perform KMeans clustering.  User passes in the
  // following parameters:
  //   d01      : The Document dataset
  //   d02      : The Centroid dataset
  //   n        : The number of iterations to perform
  //   nConverge: [OPTIONAL] If the maximum distance moved by a centroid in
  //              any one iteration is below the threshold, stop iterating.
  //              Default is 0.
  //   fDist    : [OPTIONAL] The distance calculation to use when determining
  //              centroid allegiance.  Default is simple Euclidean.
  //---------------------------------------------------------------------------
  EXPORT KMeans(DATASET(Types.NumericField) d01,DATASET(Types.NumericField) d02,UNSIGNED n=1,REAL nConverge=0.0,DF.Default fDist=DF.WEuclidean):=MODULE
    SHARED iOffset:=IF(MAX(d01,id)>MIN(d02,id),MAX(d01,id),0);
    
    // For the internal storage of all iterations, we convert the VALUE field
    // in NumericField to a SET OF VALUES, where values[1] is the initial
    // location of the centroids, values[2] is after the first iteration, etc.
    // SHARED lIterations:=RECORD
      // TYPEOF(Types.NumericField.id) id;
      // TYPEOF(Types.NumericField.number) number;
      // SET OF TYPEOF(Types.NumericField.value) values;
    // END;
    
    // Function to pull iteration N from a table of type lIterations
    SHARED Types.NumericField dResult(UNSIGNED n=n,DATASET(lIterations) d):=PROJECT(d,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[n+1];SELF:=LEFT;));

    // Determine the delta along each axis between any two iterations
    Types.NumericField tGetDelta(Types.NumericField L,Types.NumericField R):=TRANSFORM
      SELF.id:=IF(L.id=0,R.id,L.id);
      SELF.number:=IF(L.number=0,R.number,L.number);
      SELF.value:=R.value-L.value;
    END;
    SHARED dDelta(UNSIGNED n01=n-1,UNSIGNED n02=n,DATASET(lIterations) d):=JOIN(dResult(n01,d),dResult(n02,d),LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,tGetDelta(LEFT,RIGHT));
    
    // Determine the distance delta between two iterations, using the distance
    // method specified by the user for this module
    SHARED dDistanceDelta(UNSIGNED n01=n-1,UNSIGNED n02=n,DATASET(lIterations) d):=FUNCTION
      iMax01:=MAX(dResult(n01,d),id);
      dDistances:=Distances(dResult(n01,d),PROJECT(dResult(n02,d),TRANSFORM(Types.NumericField,SELF.id:=LEFT.id+iMax01;SELF:=LEFT;)),fDist);
      RETURN PROJECT(dDistances(x=y-iMax01),TRANSFORM({Types.NumericField AND NOT [number];},SELF.id:=LEFT.x;SELF:=LEFT;));
    END;

    // Convert the input centroid dataset to our internal structure, then
    // iterate as many times as requested by the user.
    // NOTE: Values will stop being added once convergence is determined
    // to have been reached.
    d02Prep:=PROJECT(d02,TRANSFORM(lIterations,SELF.id:=LEFT.id+iOffset;SELF.values:=[LEFT.value];SELF:=LEFT;));
    fIterate(DATASET(lIterations) d,UNSIGNED c):=FUNCTION
      // Check the distance delta for the last two iterations.  If the highest
      // value is below the convergence threshold, then set bConverged to TRUE
      bConverged:=IF(c=1,FALSE,MAX(dDistanceDelta(c-1,c-2,d),value)<=nConverge);
      // set the current centroids to the results of the most recent iteration
      dCentroids:=PROJECT(d,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c];SELF:=LEFT;));
      // get all document-to-centroid distances, and determine centroid allegiance
      dDistances:=Distances(d01,dCentroids,fDist);
      dClosest:=Closest(dDistances);
      // Get a count of the number of documents allied to each centroid
      dClusterCounts:=TABLE(dClosest,{y;UNSIGNED c:=COUNT(GROUP);},y,FEW);
      // Join closest to the document set and replace the id with the centriod id
      dClustered:=SORT(DISTRIBUTE(JOIN(d01,dClosest,LEFT.id=RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;),HASH),id),RECORD,LOCAL);
      // Now roll up on centroid ID, summing up the values for each axis
      dRolled:=ROLLUP(dClustered,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;),id,number,LOCAL);
      // Join to cluster counts to calculate the new average on each axis
      dJoined:=JOIN(dRolled,dClusterCounts,LEFT.id=RIGHT.y,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value/RIGHT.c;SELF:=LEFT;),LOOKUP);
      // Find any centroids with no document allegiance and pass those through also
		  dPass:=JOIN(dCentroids,TABLE(dJoined,{id},id,LOCAL),LEFT.id=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);
      // Now join to the existing centroid dataset to add the new values to
      // the end of the values set.
      dAdded:=JOIN(d,dJoined+dPass,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=RIGHT;),RIGHT OUTER);
      // If the centroids have converged, simply pass the input dataset through
      // to the next iteration.  Otherwise perform an iteration.
      RETURN IF(bConverged,d,dAdded);
    END;
    dIterationResults:=LOOP(d02Prep,n,fIterate(ROWS(LEFT),COUNTER));
    SHARED dIterations:=IF(iOffset>0,PROJECT(dIterationResults,TRANSFORM(lIterations,SELF.id:=LEFT.id-iOffset;SELF:=LEFT;)),dIterationResults):INDEPENDENT;

    // Show the fully traced result set
    EXPORT lIterations AllResults:=dIterations;
    
    // The number of iterations upon which convergence was reached is simply
    // one less than the number of values in any of the dIterations rows
    EXPORT UNSIGNED Convergence:=COUNT(dIterations[1].values)-1;
    
    // Specific-instance exports for the SHARED attributes at the top of
    // the KMeans module (with d assumed to be the iterated results). 
    EXPORT Types.NumericField Result(UNSIGNED n=Convergence,DATASET(lIterations) d=dIterations):=dResult(MIN(Convergence,n),d);
    EXPORT Types.NumericField Delta(UNSIGNED n01=Convergence-1,UNSIGNED n02=Convergence,DATASET(lIterations) d=dIterations):=dDelta(MIN(Convergence-1,n01),MIN(Convergence,n02),d);
    EXPORT DistanceDelta(UNSIGNED n01=Convergence-1,UNSIGNED n02=Convergence,DATASET(lIterations) d=dIterations):=dDistanceDelta(MIN(Convergence-1,n01),MIN(Convergence,n02),d);

    // Quick re-directs to the Closest attribute specific to this module's
    // parameters
    EXPORT Allegiances(UNSIGNED n=Convergence):=PROJECT(Closest(Distances(d01,PROJECT(Result(n),TRANSFORM(RECORDOF(Result(n)),SELF.id:=LEFT.id+iOffset;SELF:=LEFT;)),fDist)),TRANSFORM(RECORDOF(Mat.Types.Element),SELF.y:=LEFT.y-iOffset;SELF:=LEFT;));
    EXPORT UNSIGNED Allegiance(UNSIGNED id,UNSIGNED n=Convergence):=Allegiances(n)(x=id)[1].y;
  END;

// When combining clusters how to compute the distances of the new clusters to each other
// Min-dist - minimum of the components
// Max-dist - maximum of the components
// ave-dist - average of the components
  EXPORT c_Method := ENUM( min_dist,max_dist,ave_dist );

  // Agglomerative (or Hierarchical clustering) - attempts to weld the clusters together bottom up
	// N is the number of steps to take

  EXPORT AggloN(DATASET(Types.NumericField) d,UNSIGNED4 N,DF.Default Dist=DF.QEuclidean, c_Method cm=c_Method.min_dist):= MODULE
    Distance:=Distances(d,d,Dist)(x<>y);
		dinit0 := DEDUP( d, ID, ALL );
		// To go around the loop this has to be a combined 'distance metric' / 'clusters so far' format
		ClusterRec := RECORD// Collect the full matrix of pair-pair distances
		  Types.t_RecordID ClusterId := dinit0.id;
			Types.t_RecordID Id := 0;
			Types.t_FieldReal value := 0;
			STRING Members := (STRING)dinit0.id;
			STRING newick := (STRING)dinit0.id;
		END;
		ConcatAll(DATASET(ClusterRec) s) := FUNCTION
			R := RECORD
			  STRING St;
			END;
			RETURN AGGREGATE(s,R,TRANSFORM(R,SELF.St := IF( RIGHT.St = '', LEFT.Members, RIGHT.St+' '+LEFT.Members)),TRANSFORM(R,SELF.St := IF( RIGHT1.St = '', RIGHT2.St, RIGHT1.St+' '+RIGHT2.St)))[1].St;
		END;
		ConcatAllnewick(DATASET(ClusterRec) s) := FUNCTION
			R := RECORD
			  STRING St;
			END;
			RETURN AGGREGATE(s,R,TRANSFORM(R,SELF.St := IF( RIGHT.St = '', LEFT.newick, RIGHT.St+' '+LEFT.newick)),TRANSFORM(R,SELF.St := IF( RIGHT1.St = '', RIGHT2.St, RIGHT1.St+' '+RIGHT2.St)))[1].St;
		END;
		dinit1 := TABLE(dinit0,ClusterRec);
		DistAsClus := PROJECT( Distance, TRANSFORM(ClusterRec, SELF.Members:='', SELF.newick:='newick', SELF.clusterid:=LEFT.y, SELF.id := LEFT.x, SELF := LEFT) );
		Dinit := dinit1+DistAsClus;
		Step(DATASET(ClusterRec) cd00) := FUNCTION
		  cd := cd00(Members='');
			cl := cd00(Members<>'');
		  // Find the best value for each id
			minx := TABLE(cd,{id,val := MIN(GROUP,value)},id);
			// Find the best value for each cluster
			miny := TABLE(cd,{clusterid,val := MIN(GROUP,value)},clusterid);
			// Find those entries that are best - only pick clusterid<id (so entries only found once) 
			xposs := JOIN(cd(clusterid<id),minx,LEFT.id=RIGHT.id AND LEFT.value=RIGHT.val,TRANSFORM(LEFT));
			// Make sure the other side is just as happy
			tojoin0 := JOIN(xposs,miny,LEFT.clusterid=RIGHT.clusterid AND LEFT.value=RIGHT.val,TRANSFORM(LEFT));
			// Now we have to avoid the transitive closure, no point in A->B if B->C
			// One option is to assert A->C; another is to break the A->B link
			tojoin := JOIN(tojoin0,tojoin0,LEFT.clusterid=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY);
			// Now we need to mutilate the distance table to reflect the new reality
			// We do this first by 'duping' the elements
			cd0 := JOIN(cd,tojoin,LEFT.id=RIGHT.id,TRANSFORM(ClusterRec,SELF.id:=IF(RIGHT.id<>0,RIGHT.clusterid,LEFT.id),SELF:=LEFT),LOOKUP,LEFT OUTER)(id<>clusterid);
			cd1 := JOIN(cd0,tojoin,LEFT.clusterid=RIGHT.id,TRANSFORM(ClusterRec,SELF.clusterid:=IF(RIGHT.id<>0,RIGHT.clusterid,LEFT.clusterid),SELF:=LEFT),LOOKUP,LEFT OUTER)(id<>clusterid);
			r1 := RECORD
			  cd1.id;
				cd1.clusterid;
				REAL8 MinV := MIN(GROUP,cd1.value);
				REAL8 MaxV := MAX(GROUP,cd1.value);
				REAL8 AveV := AVE(GROUP,cd1.value);
			END;
			cd2 := TABLE(cd1,r1,id,clusterid);
			cd3 := PROJECT(cd2,TRANSFORM(ClusterRec,SELF.Members:='',SELF.newick:='',SELF.value := CASE( cm,c_Method.min_dist => LEFT.MinV, c_Method.max_dist => LEFT.MaxV, LEFT.AveV ), SELF := LEFT ));
			// Now perform the actual clustering
			// First we flag those that will be growing clusters with a 1
			J1 := JOIN(cl,tojoin,LEFT.Clusterid=RIGHT.Clusterid,TRANSFORM(ClusterRec,SELF.id := IF ( RIGHT.ClusterID<>0, 1, 0 ),SELF := LEFT),LEFT OUTER,KEEP(1));
			// Those that will be collapsing get the cluster number in their ID slot
			J2 := JOIN(J1(id=0),tojoin,LEFT.Clusterid=RIGHT.id,TRANSFORM(ClusterRec,SELF.id := RIGHT.ClusterId,SELF.value:=RIGHT.value,SELF := LEFT),LEFT OUTER);
			// Those remaining inert will get a 0
			ClusterRec JoinCluster(J1 le,DATASET(ClusterRec) ri) := TRANSFORM
			  v:=REGEXREPLACE('^ +',REALFORMAT(ri[1].value,5,2),'');
			  left_newick:=IF(REGEXFIND('\\):[0-9]+\\.[0-9]+$',le.newick),le.newick,le.newick+':'+v);
			  c:=ConcatAllnewick(ri);
			  right_newick:=IF(REGEXFIND('\\):[0-9]+\\.[0-9]+$',c),c,c+':'+v);
			  SELF.clusterid := le.clusterid;
				SELF.Members := '{'+le.Members+'}{'+ConcatAll(ri)+'}';
			  SELF.newick := '('+left_newick+';'+right_newick+'):'+v;
			END;
			J3 := DENORMALIZE(J1(id=1),J2(id<>0),LEFT.ClusterId=RIGHT.id,GROUP,JoinCluster(LEFT,ROWS(RIGHT)));
			RETURN IF(~EXISTS(CD),CL,J3+J2(id=0)+cd3);
		END;
	SHARED res := LOOP(dinit,(COUNTER<=N) AND (COUNT(ROWS(LEFT))>1),Step(ROWS(LEFT)));
	EXPORT Dendrogram := TABLE(res(Members<>''),{ClusterId,Members});
	newick_dendrogram_rec := RECORD
	    Types.t_RecordID ClusterId:=res.ClusterId;
	    STRING newick := '('+REGEXREPLACE(';',res.newick,',')+');';
	END;
	EXPORT NewickDendrogram := TABLE(res((newick<>'') and (newick<>'newick')),newick_dendrogram_rec);
	EXPORT Distances := TABLE(res(Members=''),{ClusterId,Id,Value});
		NoBrace(STRING S) := Str.CleanSpaces(Str.SubstituteIncluded(S,'{}',' '));
    De := TABLE(Dendrogram,{ClusterId,Ids := NoBrace(Members)});
		Mat.Types.Element note(De le,UNSIGNED c) := TRANSFORM
		  SELF.y := le.clusterid;
			SELF.x := (UNSIGNED)Str.GetNthWord(le.Ids,c);
			SELF.value := 0; // Dendrogram does not return any cluster centroid distance measure
		END;
	EXPORT Clusters := NORMALIZE(De,Str.WordCount(LEFT.Ids),note(LEFT,COUNTER));
END;


//*********************Yinyang Performance Testing *******************88
EXPORT YinYangKMeans(DATASET(Types.NumericField) d01,DATASET(Types.NumericField) d02,UNSIGNED n=1,REAL nConverge=0.0,DF.Default fDist=DF.WEuclidean):=MODULE

// internal record structure of d02
// SHARED lIterations:=RECORD 
// TYPEOF(Types.NumericField.id) id;
// TYPEOF(Types.NumericField.number) number;
// SET OF TYPEOF(Types.NumericField.value) values;
// END;

// tempR := RECORD
// d01.id;
// d01.number;
// d01.value;
// TYPEOF(Types.NumericField.id) cid:=0 ;
// END;

//make sure all ids are different
iOffset:=IF(MAX(d01,id)>MIN(d02,id),MAX(d01,id),0);

//transfrom the d02 to internal data structure
d02Prep:=PROJECT(d02,TRANSFORM(lIterations,SELF.id:=LEFT.id+iOffset;SELF.values:=[LEFT.value];SELF:=LEFT;));
// d02Prep; 
dCentroid0 := PROJECT(d02Prep,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[1];SELF:=LEFT;));
// get Gt 
//running Kmeans on dCentroid
K := COUNT(DEDUP(d02,id));
t:= IF(K/10<1, 1, K/10);

//temporary solution to get dt is that dt = the first t cendroids of dCentroid 
temp := t * 2;
tempDt := dCentroid0[1..temp];
//transform ids of the centroids of dt starting from 1....
Types.NumericField transDt(Types.NumericField L, INTEGER c) := TRANSFORM
SELF.id := ROUNDUP(c/2);
SELF := L;
END;
dt := project(tempDt, transDt(LEFT, COUNTER));
nt := n;
tConverge := nConverge;

//run Kmeans on dCentroid to get Gt
KmeansDt :=KMeans(dCentroid0,dt,nt,tConverge);

Gt := TABLE(KmeansDt.Allegiances(), {x,y},y,x);//the assignment of each centroid to a group

//initialize ub and lb for each data point
//get ub0
dDistances0 := Distances(d01,dCentroid0);
dClosest0 := Closest(dDistances0);

// dClosest0;
ub0_ini := dClosest0; 

//get lbs0_ini
//lb of each group
lbs0_iniTemp := JOIN(dClosest0,dDistances0, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y,RIGHT ONLY);
lbs0_iniTemp1 := SORT(JOIN(lbs0_iniTemp, Gt, LEFT.y = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.y := RIGHT.y; SELF := LEFT;)),x, value);
lbs0_ini := DEDUP(lbs0_iniTemp1,x);

//initialize Vcount0
//Get V
lVcount := RECORD
TYPEOF(Types.NumericField.id)  id := dClosest0.y;
TYPEOF(Types.NumericField.number) c := COUNT(group);
TYPEOF(Types.NumericField.value)   inNum:= 0;
TYPEOF(Types.NumericField.number) outNum := 0;
END;


Vin0temp := TABLE(dClosest0, {y; UNSIGNED c := COUNT(GROUP);},y );
V0_ini := PROJECT(Vin0temp, TRANSFORM(Mat.Types.Element, SELF.x := LEFT.y; SELF.y := LEFT.c; SELF.value := 0; ));

//running Kmeans on d01
KmeansD01 := KMeans(d01,dCentroid0,1,nConverge);

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
// fDist :=DF.Euclidean;
dDistanceDelta(UNSIGNED n01=n-1,UNSIGNED n02=n,DATASET(lIterations) d):=FUNCTION
      iMax01:=MAX(dResult(n01,d),id);
      dDistance:=Distances(dResult(n01,d),PROJECT(dResult(n02,d),TRANSFORM(Types.NumericField,SELF.id:=LEFT.id+iMax01;SELF:=LEFT;)));
RETURN PROJECT(dDistance(x=y-iMax01),TRANSFORM({Types.NumericField AND NOT [number];},SELF.id:=LEFT.x;SELF:=LEFT;));
END;

//*********************************************************************************************************************************************************************

//set the result of Standard Kmeans to the result of first iteration
firstIte := KmeansD01.AllResults();
firstResult := PROJECT(firstIte,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[2];SELF:=LEFT;)); 

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

//********************************************start iterations*************************************************
lInput yyfIterate(DATASET(lInput) d,UNSIGNED c):=FUNCTION
			
			dAddedx := PROJECT(d(id = 1), TRANSFORM(lIterations , SELF.id := LEFT. x; SELF.number:= LEFT.y; SELF.values := LEFT.values;));
			iUb := TABLE(d(id = 2), {x;y;values;});
			iLbs := TABLE(d(id = 3), {x;y;values;});
			iV := TABLE(d(id = 4), {x;y;values;});
			
			ub0 := PROJECT(d(id = 2), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c]; SELF := LEFT;));
			// OUTPUT(ub0, NAMED('ub0'));
			lbs0 := PROJECT(d(id = 3), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c]; SELF := LEFT;));
			V0 := PROJECT(d(id = 4), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c]; SELF := LEFT;));
			
			//calculate the deltaC
			deltac := dDistanceDelta(c,c-1,dAddedx);			
			bConverged := IF(MAX(deltac,value)<=nConverge,TRUE,FALSE);
			dCentroid1 := PROJECT(dAddedx,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c+1];SELF:=LEFT;));

			//get deltaG1
			//group deltacg by Gt: first JOIN(deltaC with Gt) then group by Gt
			deltacg :=JOIN(deltac, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
			deltacGt := SORT(deltacg, y,value);
			deltaG1 := DEDUP(deltacGt,y, RIGHT); 

			//update ub0 and lbs0
			ub1_temp := JOIN(ub0, deltac, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;));
			lbs1_temp := JOIN(lbs0, deltaG1, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := LEFT.value - RIGHT.value; SELF := LEFT));
			
	
			// groupfilter1 testing on one time comparison
			groupFilterTemp := JOIN(ub0, lbs1_temp,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
			groupFilter1:= groupFilterTemp(value <0);
			
			pGroupFilter1 := groupFilter1;
			groupFilter1_Converge := IF(COUNT(pGroupFilter1)=0, TRUE, FALSE);
			unpGroupFilter1 := groupFilterTemp(value >=0);

			//localFilter
			changedSetTemp := JOIN(d01,pGroupFilter1, LEFT.id = RIGHT.x,TRANSFORM(LEFT));
			// changedSetTemp;
			dDistances1 := Distances(changedSetTemp,dCentroid1);
			dClosest1 := Closest(dDistances1); 

			//the dataset that change their best centroid = result of local fitler		
			localFilter1 := JOIN(dClosest1, ub0, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, TRANSFORM(LEFT));

			//localFilter1 is empty then it's converged. Or update the ub and lb.
			localFilter1_converge := IF(COUNT(localFilter1) =0, TRUE, FALSE);

			//update the Ub of data points who do not change their best centroid		
			unChangedUbSet := JOIN(ub0, localFilter1, LEFT.x = RIGHT.x, TRANSFORM(LEFT), LEFT ONLY);//map_set
			unChangedUbRolled := yyDistances(d01,dCentroid1,unChangedUbSet,fDist);
			
			//update the Ub of of data points who change their best centroid 		
			changedUb := LocalFilter1;
			unchangedUb:= unChangedUbRolled;
			//new Ub
			ub1:= SORT(changedUb + unchangedUb, x);
						
			//update lbs	
			changelbsTemp1 := JOIN(localFilter1,dDistances1, LEFT.x = RIGHT.x , TRANSFORM(RIGHT));
			changelbsTemp11 := SORT(JOIN(localFilter1,changelbsTemp1, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			//new lbs for data points who change their best centroid
			changelbs1 := DEDUP(changelbsTemp11,x );
			//new lbs
			lbs1 := JOIN(lbs1_temp, changelbs1,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value); SELF := LEFT;), LEFT OUTER );

			//update Vin and Vout
			//calculate the Vin
			VinSet1 := SORT(localFilter1,y);
			dClusterCountsVin1:=TABLE(VinSet1, {y; INTEGER c := COUNT(GROUP)},y);
			dClusteredVin1 := JOIN(d01, VinSet1, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			dRolledVin1:=ROLLUP(SORT(dClusteredVin1,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));

			VoutSet1 := JOIN(ub1_temp, localFilter1, LEFT.x = RIGHT.x, TRANSFORM(LEFT));
			dClusterCountsVout1:=TABLE(VoutSet1, {y; INTEGER c := COUNT(GROUP)},y);
			dClusteredVout1 := JOIN(d01,VoutSet1, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			dRolledVout1:=ROLLUP(SORT(dClusteredVout1,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));

			V1Temp :=JOIN(V0,dClusterCountsVin1, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y +RIGHT.c; SELF.value := LEFT.value + RIGHT.c; SELF := LEFT;), LEFT OUTER );
			V1:=JOIN(V1Temp,dClusterCountsVout1, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y -RIGHT.c;SELF.value := LEFT.value -RIGHT.c; SELF := LEFT;), LEFT OUTER);

			//let the old |c*V| to multiply old Vcount
			CV1 :=JOIN(dCentroid1, V0, LEFT.id = RIGHT.x, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value * RIGHT.y; SELF := LEFT), LEFT OUTER);


			//Add Vin
			CV1Vin := JOIN(CV1, dRolledVin1, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value + RIGHT.value; SELF := LEFT), LEFT OUTER);

			//Minus Vout
			CV1VinVout:= JOIN(CV1Vin, dRolledVout1, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value - RIGHT.value; SELF := LEFT), LEFT OUTER);
	
			//get the new C
			newC:=JOIN(CV1VinVout,V1,LEFT.id=RIGHT.x,TRANSFORM(RECORDOF(LEFT),SELF.value:=(LEFT.value/RIGHT.y);SELF:=LEFT;), LEFT OUTER);

			dAddedx1Temp := JOIN(dAddedx, newC, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
			dAddedx1 := PROJECT(dAddedx1Temp, TRANSFORM(lInput,SELF.id := 1;SELF.values:=LEFT.values;SELF.y := LEFT.number; SELF.x:=LEFT.id;));
			dAddedUb1 := JOIN(iUb, ub1, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 2;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			dAddedLbs1 := JOIN(iLbs, lbs1, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 3;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			dAddedV1 := JOIN(iV, V1, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 4;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));

			outputSet := dAddedx1+ dAddedUb1 + dAddedLbs1 +dAddedV1 ;

			RETURN MAP( MAX(deltac,value)<=nConverge => d(id=1), COUNT(groupFilter1)=0 => d(id=1),COUNT(localFilter1) =0 =>d(id=1) , outputSet );
END;

yyLoop :=LOOP(input0,n-1,yyfIterate(ROWS(LEFT),COUNTER));
yyResults := TABLE(yyLoop, {x, TYPEOF(Types.NumericField.number) number := y, values});
SHARED dIterations:=IF(iOffset>0,PROJECT(yyResults,TRANSFORM(lIterations,SELF.id:=LEFT.x-iOffset;SELF.number :=LEFT.number; SELF := LEFT;)),yyResults):INDEPENDENT;

// Show the fully traced result set
EXPORT lIterations AllResults:=dIterations;
END;

=======
﻿//-----------------------------------------------------------------------------
// Module used to cluster perform clustering on data in the NumericField
// format.  Includes functions for calculating distance using many different
// algorithms, determining centroid allegiance based on those distances, and
// performing K-Means calculations.
//-----------------------------------------------------------------------------

IMPORT * FROM $;
IMPORT Std.Str AS Str;
IMPORT ML.Mat;

EXPORT Cluster := MODULE
// Working structure for cluster distance logic
  SHARED ClusterPair:=RECORD
		Types.t_RecordID    id;
		Types.t_RecordID    clusterid;
		Types.t_FieldNumber number;
		Types.t_FieldReal   value01 := 0;
		Types.t_FieldReal   value02 := 0;
		Types.t_FieldReal   value03 := 0;
  END;
	
	 SHARED lIterations:=RECORD
    TYPEOF(Types.NumericField.id) id;
    TYPEOF(Types.NumericField.number) number;
    SET OF TYPEOF(Types.NumericField.value) values;
   END;
  
	// Compute the 'N^2' distance metric
	// The DF module contains the different distance functions
	// Note that this is a very flexible interface - it allows for a variety of 'speeds' of computation

	SHARED c_model := ENUM ( Dense = 1, SJoins = 2, Background = 4 );
  EXPORT DF := MODULE
	  // Each nested module is a 'control' interface for pairsB
		// Dense model - sparse vectors will be made dense - very simple handling of nulls - usually slow
		// Summary Joins - dimensions that are both non-null score directly; each combined score gets passed by the data from a join the summarizes either side
		// Background - a 'background' N^2 matrix is constructed from the summary joins - which is then merged with the dimension matched data
		// -- note Background REQUIRES that d02 fit in memory
		// 0 ==> score constructed only from co-populated dimensions and any EV1/2 stats computed
		// NAMING: A leading Q implies a 'quick' version of the result that probably shaves a corner or two
		//         A leading W implies a 'wide' version and is probably simple, unrestricted and painful
		//         No leading letter implies our 'best shot' at the 'correct' result
    EXPORT Default := MODULE,VIRTUAL
		  EXPORT UNSIGNED1 PModel := c_model.dense; // The process model for this distance metric
			EXPORT REAL8 EV1(DATASET(Types.NumericField) d) := 0; // An 'exotic' value which will be passed in at Comb time
			EXPORT REAL8 EV2(DATASET(Types.NumericField) d) := 0; // An 'exotic' value which will be passed in at Comb time
			EXPORT BOOLEAN JoinFilter(Types.t_FieldReal x,Types.t_FieldReal y,REAL8 ex1) := x<>0 OR y<>0; // If false - join value will not be computed
			EXPORT IV1(Types.t_FieldReal x,Types.t_FieldReal y) := x;
			EXPORT IV2(Types.t_FieldReal x,Types.t_FieldReal y) := y;
			EXPORT Types.t_FieldReal Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := 0; // The value1 - eventual result
			EXPORT Types.t_FieldReal Comb2(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := 0; // Scratchpad
			EXPORT Types.t_FieldReal Comb3(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := 0; // Scratchpad
// These can be though of as a 'turbo' or summary-join interface
// They will usually be used to prevent the need for dense computation
			EXPORT SummaryID1(DATASET(Types.NumericField) d) := d; // Used to create some form of summary by ID for dataset 1
			EXPORT SummaryID2(DATASET(Types.NumericField) d) := SummaryID1(d); // Used to create some form of summary by ID for dataset 2
			EXPORT Types.t_FieldReal Join11(ClusterPair im,Types.NumericField ri) := 0; // join 1 result 1
			EXPORT Types.t_FieldReal Join12(ClusterPair im,Types.NumericField ri) := 0;
			EXPORT Types.t_FieldReal Join13(ClusterPair im,Types.NumericField ri) := 0;
			EXPORT Types.t_FieldReal Join21(ClusterPair im,Types.NumericField ri) := 0;  // join 2 result 1
// This is the 'background' interface			
      EXPORT Types.t_FieldReal Background(Types.NumericField va1,Types.NumericField va2) := 0;// Compute background value from SI1/SI2 value
      EXPORT Types.t_FieldReal BackFront(Mat.Types.Element Back,ClusterPair Fro) := IF ( Fro.id>0, Fro.value01, Back.value );// Compute value given a background and a clusterpair <by default take front if possible>
	  END;

// These models compute a 'proper' Euclidean result but only for those vectors that have at least one dimension along
// which they both have a non-zero value. For very sparse vectors this will produce a result MUCH smaller than
// N^2 (and is correspondingly faster)		
    EXPORT QEuclideanSquared := MODULE(Default),VIRTUAL
		  EXPORT UNSIGNED1 PModel := c_model.SJoins;
			EXPORT SummaryID1(DATASET(Types.NumericField) d) := PROJECT(TABLE( d, { id, val := SUM(GROUP,value*value); }, id ),TRANSFORM(Types.NumericField,SELF.value:=LEFT.val,SELF.number:=0,SELF.id:=LEFT.id));
			EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := SUM(D,(Value01-Value02)*(Value01-Value02));
			EXPORT Comb2(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := SUM(D,Value01*Value01);
			EXPORT Comb3(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := SUM(D,Value02*Value02); // sum of all the contributing rhs
			EXPORT Join11(ClusterPair im,Types.NumericField ri) := im.value01 + ( ri.value-im.value02 ); // add in all of the lhs^2 that did not match
			EXPORT Join13(ClusterPair im,Types.NumericField ri) := im.value03; // keep the rhs^2
			EXPORT Join21(ClusterPair im,Types.NumericField ri) := im.value01 + ( ri.value-im.value03 ); // add in all of the rhs^2 that did not match
    END;
    EXPORT QEuclidean := MODULE(QEuclideanSquared)
			EXPORT Join21(ClusterPair im,Types.NumericField ri) := SQRT(im.value01 + ( ri.value-im.value03 )); // add in all of the rhs^2 that did not match
    END;

// These models compute a proper Euclidean result (the full N^2) - they do require that D02 be able to fit in memory
// However given this is N^2 - if N does not fit in memory - you are probably dead anyway
    EXPORT EuclideanSquared := MODULE(QEuclideanSquared),VIRTUAL // QEuclidean with a background added
		  EXPORT UNSIGNED1 PModel := c_model.Background; // We avoid the SJoins through cleverness in BackFront
      EXPORT Types.t_FieldReal Background(Types.NumericField va1,Types.NumericField va2) := va1.value+va2.value;
      EXPORT Types.t_FieldReal BackFront(Mat.Types.Element Back,ClusterPair Fro) := IF ( Fro.id>0, Back.value+Fro.value01-Fro.value02-Fro.value03, Back.value );
    END;
    EXPORT Euclidean := MODULE(EuclideanSquared)
      EXPORT Types.t_FieldReal BackFront(Mat.Types.Element Back,ClusterPair Fro) := SQRT(IF ( Fro.id>0, Back.value+Fro.value01-Fro.value02-Fro.value03, Back.value ));
    END;
    EXPORT Manhattan:=MODULE(Default),VIRTUAL
      EXPORT UNSIGNED1 PModel := c_model.Background;
      EXPORT SummaryID1(DATASET(Types.NumericField) d) := PROJECT(TABLE(d,{id,val:=SUM(GROUP,value);},id),TRANSFORM(Types.NumericField,SELF.value:=LEFT.val,SELF.number:=0,SELF.id:=LEFT.id));
      EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2):=SUM(d,ABS(Value01-Value02));
      EXPORT Comb2(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2):=SUM(D,Value01);
      EXPORT Comb3(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2):=SUM(D,Value02);
      EXPORT Types.t_FieldReal Background(Types.NumericField va1,Types.NumericField va2):=va1.value+va2.value;
      EXPORT Types.t_FieldReal BackFront(Mat.Types.Element Back,ClusterPair Fro):=IF(Fro.id>0,Back.value+Fro.value01-Fro.value02-Fro.value03,Back.value);
    END;
    EXPORT Cosine := MODULE(Default),VIRTUAL
      EXPORT UNSIGNED1 PModel := c_model.Background;
      EXPORT SummaryID1(DATASET(Types.NumericField) d) := PROJECT(TABLE(d,{id,val:=SQRT(SUM(GROUP,(value*value)));},id),TRANSFORM(Types.NumericField,SELF.value:=LEFT.val,SELF.number:=0,SELF.id:=LEFT.id));
      EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := SUM(d,Value01*Value02);
      EXPORT Types.t_FieldReal Background(Types.NumericField va1,Types.NumericField va2):=va1.value*va2.value;
      EXPORT Types.t_FieldReal BackFront(Mat.Types.Element Back,ClusterPair Fro):=IF(Fro.id>0,1.0-(Fro.Value01/Back.value),1.0);
    END;
    EXPORT Tanimoto := MODULE(Cosine),VIRTUAL
      EXPORT Types.t_FieldReal BackFront(Mat.Types.Element Back,ClusterPair Fro):=IF(Fro.id>0,1.0-(Fro.Value01/(Back.value-Fro.Value01)),1.0);
    END;
// These compute full Euclidean	the 'simple' way and have no obvious restrictions
// Expect to wait a while
    EXPORT WEuclideanSquared := MODULE(Default),VIRTUAL
			EXPORT IV1(Types.t_FieldReal x,Types.t_FieldReal y) := (x-y)*(x-y);
			EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := SUM(D,Value01); 
    END;
    EXPORT WEuclidean := MODULE(WEuclideanSquared)
			EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := SQRT( SUM(D,Value01) );
    END;
    EXPORT WManhattan := MODULE(Default),VIRTUAL
			EXPORT IV1(Types.t_FieldReal x,Types.t_FieldReal y) := ABS(x-y);
			EXPORT IV2(Types.t_FieldReal x,Types.t_FieldReal y) := 0;
			EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := SUM(D,Value01);
    END;
		EXPORT Maximum := MODULE(WManhattan)
			EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := MAX(D,Value01);
		END;
    EXPORT WCosine := MODULE(Default),VIRTUAL
			EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := 1-SUM(D,Value01*Value02)/( SQRT(SUM(D,Value01*Value01))*SQRT(SUM(D,Value02*Value02)));
    END;
    EXPORT WTanimoto := MODULE(Default),VIRTUAL
			EXPORT Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := 1-SUM(D,Value01*Value02)/( SQRT(SUM(D,Value01*Value01))*SQRT(SUM(D,Value02*Value02))-SUM(D,Value01*Value02));
    END;
		// Now for some quick and dirty functions
		// This attempts to approximate the missing values - it will have far few intermediates if the matrices were sparse
		EXPORT MissingAppx := MODULE(Default),VIRTUAL
		  EXPORT UNSIGNED1 Pmodel := 0;
			EXPORT REAL8 EV1(DATASET(Types.NumericField) d) := AVE(d,value); // Average value
			EXPORT REAL8 EV2(DATASET(Types.NumericField) d) := MAX(TABLE(d,{UNSIGNED C := COUNT(GROUP)},id),C);
			EXPORT BOOLEAN JoinFilter(Types.t_FieldReal x,Types.t_FieldReal y,REAL8 ex1) := (x<>0 OR y<>0) AND ABS(x-y)<ex1; // Only produce record if closer
			EXPORT Types.t_FieldReal IV1(Types.t_FieldReal x,Types.t_FieldReal y) := ABS(x-y);
			EXPORT Types.t_FieldReal Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := SUM(d,value01) + (ev2-COUNT(d))*ev1;
		END;

		// Co-occurences - only counts number of fields with exact matches
		// For this metric missing values are 'infinity'
		EXPORT CoOccur := MODULE(Default),VIRTUAL
		  EXPORT UNSIGNED1 Pmodel := 0;
			EXPORT REAL8 EV1(DATASET(Types.NumericField) d) := MAX(d,number);
			EXPORT BOOLEAN JoinFilter(Types.t_FieldReal x,Types.t_FieldReal y,REAL8 ex1) := x<>0 AND x=y;
			EXPORT Types.t_FieldReal IV1(Types.t_FieldReal x,Types.t_FieldReal y) := 1;
			EXPORT Types.t_FieldReal Comb1(DATASET(ClusterPair) d,REAL8 ev1,REAL8 ev2) := ev1 - COUNT(d);
		END;
	END;

// This is the 'distance computation engine'. It extremely configurable - see the 'Control' parameter
	EXPORT Distances(DATASET(Types.NumericField) d01,DATASET(Types.NumericField) d02,DF.Default Control = DF.WEuclidean) := FUNCTION
		// If we are in dense model then fatten up the records; otherwise zeroes not needed
		df1 := IF( Control.Pmodel & c_model.dense > 0, Utils.Fat(d01), d01(value<>0) );
		df2 := IF( Control.Pmodel & c_model.dense > 0, Utils.Fat(d02), d02(value<>0) );
		// Construct the summary records used by SJoins and Background processing models
		si1 := Control.SummaryID1(df1); // Summaries of each document by ID
		si2 := Control.SummaryID2(df2); // May be used by any summary joins features
		// Construct the 'background' matrix from the summary matrix
		bck := JOIN(si1,si2,LEFT.id<>RIGHT.id,TRANSFORM(Mat.Types.Element,SELF.x := LEFT.id, SELF.y := RIGHT.Id, SELF.value := Control.BackGround(LEFT,RIGHT)),ALL);
		// Create up to two 'aggregate' numbers that the models may use
		ex1 := Control.EV1(d01); 
		ex2 := Control.EV2(d01);
		// This is the principle N^2 join (although some join filters can improve on that)
		ClusterPair Take2(df1 le,df2 ri) := TRANSFORM
		  SELF.clusterid := ri.id;
			SELF.id := le.id;
			SELF.number := le.number;
			SELF.value01 := Control.IV1(le.value,ri.value);
			SELF.value02 := Control.IV2(le.value,ri.value);
		END;
		J := JOIN(df1,df2,LEFT.number=RIGHT.number AND LEFT.id<>RIGHT.id AND Control.JoinFilter(LEFT.value,RIGHT.value,ex1),Take2(LEFT,RIGHT),HASH); // numbers will be evenly distribute by definition
		// Take all of the values computed for each matching ID and combine them
		JG := GROUP(J,clusterid,id,ALL);
		ClusterPair roll(ClusterPair le, DATASET(ClusterPair) gd) := TRANSFORM
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
    Result:=IF(Control.PModel & c_model.Background>0,BF,ProAsDist);
    
    // If the d02 IDs were adjusted to avoid intersection, revert them back
    // to their original numbers before returning the results.
		RETURN Result;
	END;
	SHARED tempR := RECORD
		TYPEOF(Types.NumericField.id) id;
		TYPEOF(Types.NumericField.number) number;
		TYPEOF(Types.NumericField.value) value;
		TYPEOF(Types.NumericField.id) cid:=0 ;
	END;
	
	EXPORT yyDistances(DATASET(Types.NumericField) d01,DATASET(Types.NumericField) d02, DATASET(Mat.Types.Element) dMap ,DF.Default Control = DF.WEuclidean) := FUNCTION
			// If we are in dense model then fatten up the records; otherwise zeroes not needed
			df1 := IF( Control.Pmodel & c_model.dense > 0, Utils.Fat(d01), d01(value<>0) );
			df2 := IF( Control.Pmodel & c_model.dense > 0, Utils.Fat(d02), d02(value<>0) );
			//Mapped dataset
			dNew := JOIN(dMap, d01, LEFT.x = RIGHT.id, TRANSFORM(tempR, SELF.cid := LEFT.y; SELF := RIGHT; ));//dRelation

			// This is the principle N^2 join (although some join filters can improve on that)
			ClusterPair Take2(dNew le,df2 ri) := TRANSFORM
				SELF.clusterid := ri.id;
				SELF.id := le.id;
				SELF.number := le.number;
				SELF.value01 := Control.IV1(le.value,ri.value);
				SELF.value02 := Control.IV2(le.value,ri.value);
			END;
			// df1 : replace the id of  data points with the cid and df2 is just the cendroid set
			J := JOIN(dNew,df2,LEFT.number=RIGHT.number AND LEFT.cid = RIGHT.id ,Take2(LEFT,RIGHT),HASH); // numbers will be evenly distribute by definition

			// Take all of the values computed for each matching ID and combine them
			JG := GROUP(J,id,clusterid, ALL);
			// Create up to two 'aggregate' numbers that the models may use
			ex1 := Control.EV1(d01); 
			ex2 := Control.EV2(d01);
			ClusterPair  roll(ClusterPair le, DATASET(ClusterPair) gd) := TRANSFORM
				SELF.Value01 := Control.Comb1(gd,ex1,ex2);
				SELF.Value02 := Control.Comb2(gd,ex1,ex2); // These are really scratchpad
				SELF.Value03 := Control.Comb3(gd,ex1,ex2);
				SELF := le;
			END;
			
			rld := ROLLUP(JG,GROUP,roll(LEFT,ROWS(LEFT)));
			Result := PROJECT(rld,TRANSFORM(Mat.Types.Element,SELF.x := LEFT.id,SELF.y := LEFT.clusterid,SELF.Value := LEFT.Value01, SELF := LEFT));
    
			// If the d02 IDs were adjusted to avoid intersection, revert them back
			// to their original numbers before returning the results.
			RETURN Result;
	END; 
  
  //---------------------------------------------------------------------------
  // Closest takes a set of distances and returns a collapsed set containing
  // only the row for each id with the closest centroid
  //---------------------------------------------------------------------------
  EXPORT Closest(DATASET(Mat.Types.Element) dDistances):=DEDUP(SORT(DISTRIBUTE(dDistances,x),x,value,LOCAL),x,LOCAL);

  //---------------------------------------------------------------------------
  // Suite of functions to perform KMeans clustering.  User passes in the
  // following parameters:
  //   d01      : The Document dataset
  //   d02      : The Centroid dataset
  //   n        : The number of iterations to perform
  //   nConverge: [OPTIONAL] If the maximum distance moved by a centroid in
  //              any one iteration is below the threshold, stop iterating.
  //              Default is 0.
  //   fDist    : [OPTIONAL] The distance calculation to use when determining
  //              centroid allegiance.  Default is simple Euclidean.
  //---------------------------------------------------------------------------
  EXPORT KMeans(DATASET(Types.NumericField) d01,DATASET(Types.NumericField) d02,UNSIGNED n=1,REAL nConverge=0.0,DF.Default fDist=DF.WEuclidean):=MODULE
    SHARED iOffset:=IF(MAX(d01,id)>MIN(d02,id),MAX(d01,id),0);
    
    // For the internal storage of all iterations, we convert the VALUE field
    // in NumericField to a SET OF VALUES, where values[1] is the initial
    // location of the centroids, values[2] is after the first iteration, etc.
    // SHARED lIterations:=RECORD
      // TYPEOF(Types.NumericField.id) id;
      // TYPEOF(Types.NumericField.number) number;
      // SET OF TYPEOF(Types.NumericField.value) values;
    // END;
    
    // Function to pull iteration N from a table of type lIterations
    SHARED Types.NumericField dResult(UNSIGNED n=n,DATASET(lIterations) d):=PROJECT(d,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[n+1];SELF:=LEFT;));

    // Determine the delta along each axis between any two iterations
    Types.NumericField tGetDelta(Types.NumericField L,Types.NumericField R):=TRANSFORM
      SELF.id:=IF(L.id=0,R.id,L.id);
      SELF.number:=IF(L.number=0,R.number,L.number);
      SELF.value:=R.value-L.value;
    END;
    SHARED dDelta(UNSIGNED n01=n-1,UNSIGNED n02=n,DATASET(lIterations) d):=JOIN(dResult(n01,d),dResult(n02,d),LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,tGetDelta(LEFT,RIGHT));
    
    // Determine the distance delta between two iterations, using the distance
    // method specified by the user for this module
    SHARED dDistanceDelta(UNSIGNED n01=n-1,UNSIGNED n02=n,DATASET(lIterations) d):=FUNCTION
      iMax01:=MAX(dResult(n01,d),id);
      dDistances:=Distances(dResult(n01,d),PROJECT(dResult(n02,d),TRANSFORM(Types.NumericField,SELF.id:=LEFT.id+iMax01;SELF:=LEFT;)),fDist);
      RETURN PROJECT(dDistances(x=y-iMax01),TRANSFORM({Types.NumericField AND NOT [number];},SELF.id:=LEFT.x;SELF:=LEFT;));
    END;

    // Convert the input centroid dataset to our internal structure, then
    // iterate as many times as requested by the user.
    // NOTE: Values will stop being added once convergence is determined
    // to have been reached.
    d02Prep:=PROJECT(d02,TRANSFORM(lIterations,SELF.id:=LEFT.id+iOffset;SELF.values:=[LEFT.value];SELF:=LEFT;));
    fIterate(DATASET(lIterations) d,UNSIGNED c):=FUNCTION
      // Check the distance delta for the last two iterations.  If the highest
      // value is below the convergence threshold, then set bConverged to TRUE
      bConverged:=IF(c=1,FALSE,MAX(dDistanceDelta(c-1,c-2,d),value)<=nConverge);
      // set the current centroids to the results of the most recent iteration
      dCentroids:=PROJECT(d,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c];SELF:=LEFT;));
      // get all document-to-centroid distances, and determine centroid allegiance
      dDistances:=Distances(d01,dCentroids,fDist);
      dClosest:=Closest(dDistances);
      // Get a count of the number of documents allied to each centroid
      dClusterCounts:=TABLE(dClosest,{y;UNSIGNED c:=COUNT(GROUP);},y,FEW);
      // Join closest to the document set and replace the id with the centriod id
      dClustered:=SORT(DISTRIBUTE(JOIN(d01,dClosest,LEFT.id=RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;),HASH),id),RECORD,LOCAL);
      // Now roll up on centroid ID, summing up the values for each axis
      dRolled:=ROLLUP(dClustered,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;),id,number,LOCAL);
      // Join to cluster counts to calculate the new average on each axis
      dJoined:=JOIN(dRolled,dClusterCounts,LEFT.id=RIGHT.y,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value/RIGHT.c;SELF:=LEFT;),LOOKUP);
      // Find any centroids with no document allegiance and pass those through also
		  dPass:=JOIN(dCentroids,TABLE(dJoined,{id},id,LOCAL),LEFT.id=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);
      // Now join to the existing centroid dataset to add the new values to
      // the end of the values set.
      dAdded:=JOIN(d,dJoined+dPass,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=RIGHT;),RIGHT OUTER);
      // If the centroids have converged, simply pass the input dataset through
      // to the next iteration.  Otherwise perform an iteration.
      RETURN IF(bConverged,d,dAdded);
    END;
    dIterationResults:=LOOP(d02Prep,n,fIterate(ROWS(LEFT),COUNTER));
    SHARED dIterations:=IF(iOffset>0,PROJECT(dIterationResults,TRANSFORM(lIterations,SELF.id:=LEFT.id-iOffset;SELF:=LEFT;)),dIterationResults):INDEPENDENT;

    // Show the fully traced result set
    EXPORT lIterations AllResults:=dIterations;
    
    // The number of iterations upon which convergence was reached is simply
    // one less than the number of values in any of the dIterations rows
    EXPORT UNSIGNED Convergence:=COUNT(dIterations[1].values)-1;
    
    // Specific-instance exports for the SHARED attributes at the top of
    // the KMeans module (with d assumed to be the iterated results). 
    EXPORT Types.NumericField Result(UNSIGNED n=Convergence,DATASET(lIterations) d=dIterations):=dResult(MIN(Convergence,n),d);
    EXPORT Types.NumericField Delta(UNSIGNED n01=Convergence-1,UNSIGNED n02=Convergence,DATASET(lIterations) d=dIterations):=dDelta(MIN(Convergence-1,n01),MIN(Convergence,n02),d);
    EXPORT DistanceDelta(UNSIGNED n01=Convergence-1,UNSIGNED n02=Convergence,DATASET(lIterations) d=dIterations):=dDistanceDelta(MIN(Convergence-1,n01),MIN(Convergence,n02),d);

    // Quick re-directs to the Closest attribute specific to this module's
    // parameters
    EXPORT Allegiances(UNSIGNED n=Convergence):=PROJECT(Closest(Distances(d01,PROJECT(Result(n),TRANSFORM(RECORDOF(Result(n)),SELF.id:=LEFT.id+iOffset;SELF:=LEFT;)),fDist)),TRANSFORM(RECORDOF(Mat.Types.Element),SELF.y:=LEFT.y-iOffset;SELF:=LEFT;));
    EXPORT UNSIGNED Allegiance(UNSIGNED id,UNSIGNED n=Convergence):=Allegiances(n)(x=id)[1].y;
  END;

// When combining clusters how to compute the distances of the new clusters to each other
// Min-dist - minimum of the components
// Max-dist - maximum of the components
// ave-dist - average of the components
  EXPORT c_Method := ENUM( min_dist,max_dist,ave_dist );

  // Agglomerative (or Hierarchical clustering) - attempts to weld the clusters together bottom up
	// N is the number of steps to take

  EXPORT AggloN(DATASET(Types.NumericField) d,UNSIGNED4 N,DF.Default Dist=DF.QEuclidean, c_Method cm=c_Method.min_dist):= MODULE
    Distance:=Distances(d,d,Dist)(x<>y);
		dinit0 := DEDUP( d, ID, ALL );
		// To go around the loop this has to be a combined 'distance metric' / 'clusters so far' format
		ClusterRec := RECORD// Collect the full matrix of pair-pair distances
		  Types.t_RecordID ClusterId := dinit0.id;
			Types.t_RecordID Id := 0;
			Types.t_FieldReal value := 0;
			STRING Members := (STRING)dinit0.id;
			STRING newick := (STRING)dinit0.id;
		END;
		ConcatAll(DATASET(ClusterRec) s) := FUNCTION
			R := RECORD
			  STRING St;
			END;
			RETURN AGGREGATE(s,R,TRANSFORM(R,SELF.St := IF( RIGHT.St = '', LEFT.Members, RIGHT.St+' '+LEFT.Members)),TRANSFORM(R,SELF.St := IF( RIGHT1.St = '', RIGHT2.St, RIGHT1.St+' '+RIGHT2.St)))[1].St;
		END;
		ConcatAllnewick(DATASET(ClusterRec) s) := FUNCTION
			R := RECORD
			  STRING St;
			END;
			RETURN AGGREGATE(s,R,TRANSFORM(R,SELF.St := IF( RIGHT.St = '', LEFT.newick, RIGHT.St+' '+LEFT.newick)),TRANSFORM(R,SELF.St := IF( RIGHT1.St = '', RIGHT2.St, RIGHT1.St+' '+RIGHT2.St)))[1].St;
		END;
		dinit1 := TABLE(dinit0,ClusterRec);
		DistAsClus := PROJECT( Distance, TRANSFORM(ClusterRec, SELF.Members:='', SELF.newick:='newick', SELF.clusterid:=LEFT.y, SELF.id := LEFT.x, SELF := LEFT) );
		Dinit := dinit1+DistAsClus;
		Step(DATASET(ClusterRec) cd00) := FUNCTION
		  cd := cd00(Members='');
			cl := cd00(Members<>'');
		  // Find the best value for each id
			minx := TABLE(cd,{id,val := MIN(GROUP,value)},id);
			// Find the best value for each cluster
			miny := TABLE(cd,{clusterid,val := MIN(GROUP,value)},clusterid);
			// Find those entries that are best - only pick clusterid<id (so entries only found once) 
			xposs := JOIN(cd(clusterid<id),minx,LEFT.id=RIGHT.id AND LEFT.value=RIGHT.val,TRANSFORM(LEFT));
			// Make sure the other side is just as happy
			tojoin0 := JOIN(xposs,miny,LEFT.clusterid=RIGHT.clusterid AND LEFT.value=RIGHT.val,TRANSFORM(LEFT));
			// Now we have to avoid the transitive closure, no point in A->B if B->C
			// One option is to assert A->C; another is to break the A->B link
			tojoin := JOIN(tojoin0,tojoin0,LEFT.clusterid=RIGHT.id,TRANSFORM(LEFT),LEFT ONLY);
			// Now we need to mutilate the distance table to reflect the new reality
			// We do this first by 'duping' the elements
			cd0 := JOIN(cd,tojoin,LEFT.id=RIGHT.id,TRANSFORM(ClusterRec,SELF.id:=IF(RIGHT.id<>0,RIGHT.clusterid,LEFT.id),SELF:=LEFT),LOOKUP,LEFT OUTER)(id<>clusterid);
			cd1 := JOIN(cd0,tojoin,LEFT.clusterid=RIGHT.id,TRANSFORM(ClusterRec,SELF.clusterid:=IF(RIGHT.id<>0,RIGHT.clusterid,LEFT.clusterid),SELF:=LEFT),LOOKUP,LEFT OUTER)(id<>clusterid);
			r1 := RECORD
			  cd1.id;
				cd1.clusterid;
				REAL8 MinV := MIN(GROUP,cd1.value);
				REAL8 MaxV := MAX(GROUP,cd1.value);
				REAL8 AveV := AVE(GROUP,cd1.value);
			END;
			cd2 := TABLE(cd1,r1,id,clusterid);
			cd3 := PROJECT(cd2,TRANSFORM(ClusterRec,SELF.Members:='',SELF.newick:='',SELF.value := CASE( cm,c_Method.min_dist => LEFT.MinV, c_Method.max_dist => LEFT.MaxV, LEFT.AveV ), SELF := LEFT ));
			// Now perform the actual clustering
			// First we flag those that will be growing clusters with a 1
			J1 := JOIN(cl,tojoin,LEFT.Clusterid=RIGHT.Clusterid,TRANSFORM(ClusterRec,SELF.id := IF ( RIGHT.ClusterID<>0, 1, 0 ),SELF := LEFT),LEFT OUTER,KEEP(1));
			// Those that will be collapsing get the cluster number in their ID slot
			J2 := JOIN(J1(id=0),tojoin,LEFT.Clusterid=RIGHT.id,TRANSFORM(ClusterRec,SELF.id := RIGHT.ClusterId,SELF.value:=RIGHT.value,SELF := LEFT),LEFT OUTER);
			// Those remaining inert will get a 0
			ClusterRec JoinCluster(J1 le,DATASET(ClusterRec) ri) := TRANSFORM
			  v:=REGEXREPLACE('^ +',REALFORMAT(ri[1].value,5,2),'');
			  left_newick:=IF(REGEXFIND('\\):[0-9]+\\.[0-9]+$',le.newick),le.newick,le.newick+':'+v);
			  c:=ConcatAllnewick(ri);
			  right_newick:=IF(REGEXFIND('\\):[0-9]+\\.[0-9]+$',c),c,c+':'+v);
			  SELF.clusterid := le.clusterid;
				SELF.Members := '{'+le.Members+'}{'+ConcatAll(ri)+'}';
			  SELF.newick := '('+left_newick+';'+right_newick+'):'+v;
			END;
			J3 := DENORMALIZE(J1(id=1),J2(id<>0),LEFT.ClusterId=RIGHT.id,GROUP,JoinCluster(LEFT,ROWS(RIGHT)));
			RETURN IF(~EXISTS(CD),CL,J3+J2(id=0)+cd3);
		END;
	SHARED res := LOOP(dinit,(COUNTER<=N) AND (COUNT(ROWS(LEFT))>1),Step(ROWS(LEFT)));
	EXPORT Dendrogram := TABLE(res(Members<>''),{ClusterId,Members});
	newick_dendrogram_rec := RECORD
	    Types.t_RecordID ClusterId:=res.ClusterId;
	    STRING newick := '('+REGEXREPLACE(';',res.newick,',')+');';
	END;
	EXPORT NewickDendrogram := TABLE(res((newick<>'') and (newick<>'newick')),newick_dendrogram_rec);
	EXPORT Distances := TABLE(res(Members=''),{ClusterId,Id,Value});
		NoBrace(STRING S) := Str.CleanSpaces(Str.SubstituteIncluded(S,'{}',' '));
    De := TABLE(Dendrogram,{ClusterId,Ids := NoBrace(Members)});
		Mat.Types.Element note(De le,UNSIGNED c) := TRANSFORM
		  SELF.y := le.clusterid;
			SELF.x := (UNSIGNED)Str.GetNthWord(le.Ids,c);
			SELF.value := 0; // Dendrogram does not return any cluster centroid distance measure
		END;
	EXPORT Clusters := NORMALIZE(De,Str.WordCount(LEFT.Ids),note(LEFT,COUNTER));
END;


//*********************Yinyang Performance Testing *******************88
EXPORT YinYangKMeans(DATASET(Types.NumericField) d01,DATASET(Types.NumericField) d02,UNSIGNED n=1,REAL nConverge=0.0,DF.Default fDist=DF.WEuclidean):=MODULE

// internal record structure of d02
// SHARED lIterations:=RECORD 
// TYPEOF(Types.NumericField.id) id;
// TYPEOF(Types.NumericField.number) number;
// SET OF TYPEOF(Types.NumericField.value) values;
// END;

// tempR := RECORD
// d01.id;
// d01.number;
// d01.value;
// TYPEOF(Types.NumericField.id) cid:=0 ;
// END;

//make sure all ids are different
iOffset:=IF(MAX(d01,id)>MIN(d02,id),MAX(d01,id),0);

//transfrom the d02 to internal data structure
d02Prep:=PROJECT(d02,TRANSFORM(lIterations,SELF.id:=LEFT.id+iOffset;SELF.values:=[LEFT.value];SELF:=LEFT;));
// d02Prep; 
dCentroid0 := PROJECT(d02Prep,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[1];SELF:=LEFT;));
// get Gt 
//running Kmeans on dCentroid
K := COUNT(DEDUP(d02,id));
t:= IF(K/10<1, 1, K/10);

//temporary solution to get dt is that dt = the first t cendroids of dCentroid 
temp := t * 2;
tempDt := dCentroid0[1..temp];
//transform ids of the centroids of dt starting from 1....
Types.NumericField transDt(Types.NumericField L, INTEGER c) := TRANSFORM
SELF.id := ROUNDUP(c/2);
SELF := L;
END;
dt := project(tempDt, transDt(LEFT, COUNTER));
nt := n;
tConverge := nConverge;

//run Kmeans on dCentroid to get Gt
KmeansDt :=KMeans(dCentroid0,dt,nt,tConverge);

Gt := TABLE(KmeansDt.Allegiances(), {x,y},y,x);//the assignment of each centroid to a group

//initialize ub and lb for each data point
//get ub0
dDistances0 := Distances(d01,dCentroid0);
dClosest0 := Closest(dDistances0);

// dClosest0;
ub0_ini := dClosest0; 

//get lbs0_ini
//lb of each group
lbs0_iniTemp := JOIN(dClosest0,dDistances0, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y,RIGHT ONLY);
lbs0_iniTemp1 := SORT(JOIN(lbs0_iniTemp, Gt, LEFT.y = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.y := RIGHT.y; SELF := LEFT;)),x, value);
lbs0_ini := DEDUP(lbs0_iniTemp1,x);

//initialize Vcount0
//Get V
lVcount := RECORD
TYPEOF(Types.NumericField.id)  id := dClosest0.y;
TYPEOF(Types.NumericField.number) c := COUNT(group);
TYPEOF(Types.NumericField.value)   inNum:= 0;
TYPEOF(Types.NumericField.number) outNum := 0;
END;


Vin0temp := TABLE(dClosest0, {y; UNSIGNED c := COUNT(GROUP);},y );
V0_ini := PROJECT(Vin0temp, TRANSFORM(Mat.Types.Element, SELF.x := LEFT.y; SELF.y := LEFT.c; SELF.value := 0; ));

//running Kmeans on d01
KmeansD01 := KMeans(d01,dCentroid0,1,nConverge);

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
// fDist :=DF.Euclidean;
dDistanceDelta(UNSIGNED n01=n-1,UNSIGNED n02=n,DATASET(lIterations) d):=FUNCTION
      iMax01:=MAX(dResult(n01,d),id);
      dDistance:=Distances(dResult(n01,d),PROJECT(dResult(n02,d),TRANSFORM(Types.NumericField,SELF.id:=LEFT.id+iMax01;SELF:=LEFT;)));
RETURN PROJECT(dDistance(x=y-iMax01),TRANSFORM({Types.NumericField AND NOT [number];},SELF.id:=LEFT.x;SELF:=LEFT;));
END;

//*********************************************************************************************************************************************************************

//set the result of Standard Kmeans to the result of first iteration
firstIte := KmeansD01.AllResults();
firstResult := PROJECT(firstIte,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[2];SELF:=LEFT;)); 

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

//********************************************start iterations*************************************************
lInput yyfIterate(DATASET(lInput) d,UNSIGNED c):=FUNCTION
			
			dAddedx := PROJECT(d(id = 1), TRANSFORM(lIterations , SELF.id := LEFT. x; SELF.number:= LEFT.y; SELF.values := LEFT.values;));
			iUb := TABLE(d(id = 2), {x;y;values;});
			iLbs := TABLE(d(id = 3), {x;y;values;});
			iV := TABLE(d(id = 4), {x;y;values;});
			
			ub0 := PROJECT(d(id = 2), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c]; SELF := LEFT;));
			// OUTPUT(ub0, NAMED('ub0'));
			lbs0 := PROJECT(d(id = 3), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c]; SELF := LEFT;));
			V0 := PROJECT(d(id = 4), TRANSFORM(Mat.Types.Element, SELF.value := LEFT.values[c]; SELF := LEFT;));
			
			//calculate the deltaC
			deltac := dDistanceDelta(c,c-1,dAddedx);			
			bConverged := IF(MAX(deltac,value)<=nConverge,TRUE,FALSE);
			dCentroid1 := PROJECT(dAddedx,TRANSFORM(Types.NumericField,SELF.value:=LEFT.values[c+1];SELF:=LEFT;));

			//get deltaG1
			//group deltacg by Gt: first JOIN(deltaC with Gt) then group by Gt
			deltacg :=JOIN(deltac, Gt, LEFT.id = RIGHT.x, TRANSFORM(Mat.Types.Element,SELF.value := LEFT.value; SELF := RIGHT;));
			deltacGt := SORT(deltacg, y,value);
			deltaG1 := DEDUP(deltacGt,y, RIGHT); 

			//update ub0 and lbs0
			ub1_temp := JOIN(ub0, deltac, LEFT.y = RIGHT.id, TRANSFORM(Mat.Types.Element, SElF.value := LEFT.value + RIGHT.value; SELF := LEFT;));
			lbs1_temp := JOIN(lbs0, deltaG1, LEFT.y = RIGHT.y , TRANSFORM(Mat.Types.Element, SELF.value := LEFT.value - RIGHT.value; SELF := LEFT));
			
	
			// groupfilter1 testing on one time comparison
			groupFilterTemp := JOIN(ub0, lbs1_temp,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := RIGHT.value - LEFT.value; SELF := LEFT));
			groupFilter1:= groupFilterTemp(value <0);
			
			pGroupFilter1 := groupFilter1;
			groupFilter1_Converge := IF(COUNT(pGroupFilter1)=0, TRUE, FALSE);
			unpGroupFilter1 := groupFilterTemp(value >=0);

			//localFilter
			changedSetTemp := JOIN(d01,pGroupFilter1, LEFT.id = RIGHT.x,TRANSFORM(LEFT));
			// changedSetTemp;
			dDistances1 := Distances(changedSetTemp,dCentroid1);
			dClosest1 := Closest(dDistances1); 

			//the dataset that change their best centroid = result of local fitler		
			localFilter1 := JOIN(dClosest1, ub0, LEFT.x = RIGHT.x AND LEFT.y !=RIGHT.y, TRANSFORM(LEFT));

			//localFilter1 is empty then it's converged. Or update the ub and lb.
			localFilter1_converge := IF(COUNT(localFilter1) =0, TRUE, FALSE);

			//update the Ub of data points who do not change their best centroid		
			unChangedUbSet := JOIN(ub0, localFilter1, LEFT.x = RIGHT.x, TRANSFORM(LEFT), LEFT ONLY);//map_set
			unChangedUbRolled := yyDistances(d01,dCentroid1,unChangedUbSet,fDist);
			
			//update the Ub of of data points who change their best centroid 		
			changedUb := LocalFilter1;
			unchangedUb:= unChangedUbRolled;
			//new Ub
			ub1:= SORT(changedUb + unchangedUb, x);
						
			//update lbs	
			changelbsTemp1 := JOIN(localFilter1,dDistances1, LEFT.x = RIGHT.x , TRANSFORM(RIGHT));
			changelbsTemp11 := SORT(JOIN(localFilter1,changelbsTemp1, LEFT.x = RIGHT.x AND LEFT.y = RIGHT.y, RIGHT ONLY), x, value);
			//new lbs for data points who change their best centroid
			changelbs1 := DEDUP(changelbsTemp11,x );
			//new lbs
			lbs1 := JOIN(lbs1_temp, changelbs1,LEFT.x = RIGHT.x, TRANSFORM(Mat.Types.Element, SELF.value := IF( RIGHT.value = 0,LEFT.value, RIGHT.value); SELF := LEFT;), LEFT OUTER );

			//update Vin and Vout
			//calculate the Vin
			VinSet1 := SORT(localFilter1,y);
			dClusterCountsVin1:=TABLE(VinSet1, {y; INTEGER c := COUNT(GROUP)},y);
			dClusteredVin1 := JOIN(d01, VinSet1, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			dRolledVin1:=ROLLUP(SORT(dClusteredVin1,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));

			VoutSet1 := JOIN(ub1_temp, localFilter1, LEFT.x = RIGHT.x, TRANSFORM(LEFT));
			dClusterCountsVout1:=TABLE(VoutSet1, {y; INTEGER c := COUNT(GROUP)},y);
			dClusteredVout1 := JOIN(d01,VoutSet1, LEFT.id = RIGHT.x,TRANSFORM(Types.NumericField,SELF.id:=RIGHT.y;SELF:=LEFT;));
			dRolledVout1:=ROLLUP(SORT(dClusteredVout1,id, number),LEFT.number = RIGHT.number,TRANSFORM(Types.NumericField,SELF.value:=LEFT.value+RIGHT.value;SELF:=LEFT;));

			V1Temp :=JOIN(V0,dClusterCountsVin1, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y +RIGHT.c; SELF.value := LEFT.value + RIGHT.c; SELF := LEFT;), LEFT OUTER );
			V1:=JOIN(V1Temp,dClusterCountsVout1, LEFT.x = RIGHT.y, TRANSFORM(RECORDOF(LEFT), SELF.y := LEFT.y -RIGHT.c;SELF.value := LEFT.value -RIGHT.c; SELF := LEFT;), LEFT OUTER);

			//let the old |c*V| to multiply old Vcount
			CV1 :=JOIN(dCentroid1, V0, LEFT.id = RIGHT.x, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value * RIGHT.y; SELF := LEFT), LEFT OUTER);


			//Add Vin
			CV1Vin := JOIN(CV1, dRolledVin1, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value + RIGHT.value; SELF := LEFT), LEFT OUTER);

			//Minus Vout
			CV1VinVout:= JOIN(CV1Vin, dRolledVout1, LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(RECORDOF(LEFT), SELF.value := LEFT.value - RIGHT.value; SELF := LEFT), LEFT OUTER);
	
			//get the new C
			newC:=JOIN(CV1VinVout,V1,LEFT.id=RIGHT.x,TRANSFORM(RECORDOF(LEFT),SELF.value:=(LEFT.value/RIGHT.y);SELF:=LEFT;), LEFT OUTER);

			dAddedx1Temp := JOIN(dAddedx, newC, LEFT.id = RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(lIterations,SELF.values:=LEFT.values+[RIGHT.value];SELF:=LEFT;));
			dAddedx1 := PROJECT(dAddedx1Temp, TRANSFORM(lInput,SELF.id := 1;SELF.values:=LEFT.values;SELF.y := LEFT.number; SELF.x:=LEFT.id;));
			dAddedUb1 := JOIN(iUb, ub1, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 2;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			dAddedLbs1 := JOIN(iLbs, lbs1, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 3;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));
			dAddedV1 := JOIN(iV, V1, LEFT.x = RIGHT.x ,TRANSFORM(lInput,SELF.id := 4;SELF.values:=LEFT.values+[RIGHT.value];SELF.y := RIGHT.y; SELF:=LEFT;));

			outputSet := dAddedx1+ dAddedUb1 + dAddedLbs1 +dAddedV1 ;

			RETURN MAP( MAX(deltac,value)<=nConverge => d(id=1), COUNT(groupFilter1)=0 => d(id=1),COUNT(localFilter1) =0 =>d(id=1) , outputSet );
END;

yyLoop :=LOOP(input0,n-1,yyfIterate(ROWS(LEFT),COUNTER));
yyResults := TABLE(yyLoop, {x, TYPEOF(Types.NumericField.number) number := y, values});
SHARED dIterations:=IF(iOffset>0,PROJECT(yyResults,TRANSFORM(lIterations,SELF.id:=LEFT.x-iOffset;SELF.number :=LEFT.number; SELF := LEFT;)),yyResults):INDEPENDENT;

// Show the fully traced result set
EXPORT lIterations AllResults:=dIterations;
END;

>>>>>>> 3de2873bf7e1684cb50b7aa9a6f6bcd085d1c004
END;	