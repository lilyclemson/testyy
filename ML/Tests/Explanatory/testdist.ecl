IMPORT TS;
IMPORT ML.Mat;
IMPORT ML;
IMPORT ML.Cluster.DF as DF;
IMPORT ML.Types AS Types;
IMPORT ML.Utils;
IMPORT ML.MAT AS Mat;

lMatrix:={UNSIGNED id;REAL x;REAL y;};
dDocumentMatrix:=DATASET([
{1,0,1},
{2,0,2},
{3,0,3},
{4,0,4}
],lMatrix);

dCentroidMatrix:=DATASET([
{1,1,1},
{2,2,2},
{3,3,3},
{4,4,4}
],lMatrix);

ML.ToField(dDocumentMatrix,dDocuments);
ML.ToField(dCentroidMatrix,dCentroids);
   ClusterPair:=RECORD
		Types.t_RecordID    id;
		Types.t_RecordID    clusterid;
		Types.t_FieldNumber number;
		Types.t_FieldReal   value01 := 0;
		Types.t_FieldReal   value02 := 0;
		Types.t_FieldReal   value03 := 0;
  END;
d01 := dDocuments;
d02 := dCentroids;
dMaptest := DATASET([
{1,1,1,0,0,0},
{2,2,2,0,0,0},
{3,3,1,0,0,0},
{4,4,2,0,0,0}
],ClusterPair);;

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
  EXPORT testdist:= Distances(d01, d02, dMaptest);
