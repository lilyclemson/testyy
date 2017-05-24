EXPORT relation20network := MODULE

lData := RECORD
STRING	Pathway;
INTEGER	Nodes ;
INTEGER	Edges	;
INTEGER	Connected_Components	;
INTEGER	Network_Diameter;
INTEGER	Network_Radius;
INTEGER	Shortest_Path	; 
REAL	Characteristic_Path ; 
REAL	Avg_num_Neighbours;	
INTEGER	Isolated_Nodes;	
INTEGER	Number_of_Self_Loops;	
INTEGER	Multi_edge_Node_Pair	;
REAL	NeighborhoodConnectivity;
REAL	Outdegree;
REAL Stress;
INTEGER	SelfLoops;
REAL	PartnerOfMultiEdgedNodePairs;	 
REAl	EdgeCount;
REAL	BetweennessCentrality; 
REAL	Indegree;	
REAL	Eccentricity;	 
REAL	ClosenessCentrality; 
REAL	AverageShortestPathLength;	
REAL	ClusteringCoefficient	;
END;

lData1 := RECORD
INTEGER Pathway;
INTEGER	Nodes ;
INTEGER	Edges	;
INTEGER	Connected_Components	;
INTEGER	Network_Diameter;
INTEGER	Network_Radius;
INTEGER	Shortest_Path	; 
REAL	Characteristic_Path ; 
REAL	Avg_num_Neighbours;	
INTEGER	Isolated_Nodes;	
INTEGER	Number_of_Self_Loops;	
INTEGER	Multi_edge_Node_Pair	;
REAL	NeighborhoodConnectivity;
REAL	Outdegree;
REAL Stress;
INTEGER	SelfLoops;
REAL	PartnerOfMultiEdgedNodePairs;	 
REAl	EdgeCount;
REAL	BetweennessCentrality; 
REAL	Indegree;	
REAL	Eccentricity;	 
REAL	ClosenessCentrality; 
REAL	AverageShortestPathLength;	
REAL	ClusteringCoefficient	;
END;

lData1 changeFormat(lData ds, UNSIGNED c) := TRANSFORM
SELF.Pathway := c;
SELF := ds;
END;
dataTemp := DATASET('~::keggundirected.txt' ,lData,  CSV(HEADING(1)));
EXPORT input := SAMPLE(PROJECT(dataTemp,changeFormat(LEFT,COUNTER)),4);
//EXPORT input := PROJECT(dataTemp,changeFormat(LEFT,COUNTER));
END;


