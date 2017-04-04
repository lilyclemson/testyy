// K-MEANS EXAMPLE
//
// Presents K-Means clustering in a 2-dimensional space. 100 data points
// are initialized with random values on the x and y axes, and 4 centroids
// are initialized with values assigned to be regular but non-symmetrical.
//
// The sample code shows how to determine the new coordinates of the
// centroids after a user-defined set of iterations. Also shows how to
// determine the "allegiance" of each data point after those iterations.
//---------------------------------------------------------------------------

IMPORT ML;
IMPORT excercise.irisset as irisset;

lMatrix:={UNSIGNED id;REAL x;REAL y;};
 dDocumentMatrix := irisset.input;
 dCentroidMatrix := irisset.input[1..3];

ML.ToField(dDocumentMatrix,dDocuments);
ML.ToField(dCentroidMatrix,dCentroids);
                                                     
KMeans:=ML.Cluster_GF_t0.KMeans(dDocuments,dCentroids,30,0); 
OUTPUT(KMeans.Allresults, NAMED('KMeansAllresults'));                                       // The table that contains the results of each iteration
OUTPUT(KMeans.Convergence, NAMED('KMeansTotal_Iterations')); 
OUTPUT(KMeans.Allegiances(), NAMED('KMeansAllegiances'));



