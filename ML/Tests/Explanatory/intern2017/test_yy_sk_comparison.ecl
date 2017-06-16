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

//K-Means testing file: summer intern 2017

IMPORT ML;
IMPORT ML.Types;
IMPORT excercise.irisset as irisset;
IMPORT excercise.relation20network as kegg;
IMPORT excercise.uscensus as uscensus;

lMatrix:={UNSIGNED id;REAL x;REAL y;};

/**
//DP100
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
*/

// iris
// dDocumentMatrix := irisset.input;
// dCentroidMatrix := irisset.input[1..3];

//KEGG
dDocumentMatrix := kegg.input;
dCentroidMatrix := kegg.input[1..4];
 
//uscensus
// dDocumentMatrix := uscensus.input;
// dCentroidMatrix := uscensus.input[1..4];
 
ML.ToField(dDocumentMatrix,dDocuments);
ML.ToField(dCentroidMatrix,dCentroids);
                                                      
//#WORKUNIT('name', 'YinyangKMeans:USCensus:30:0.0'); 
//YinyangKMeans:=ML.onegroupfaster.YinyangKMeans(dDocuments,dCentroids,30,0);

//#WORKUNIT('name', 'YinyangKMeans:USCensus:30:0.3');  
//YinyangKMeans:=ML.onegroupfaster.YinyangKMeans(dDocuments,dCentroids,30,0.3);

//#WORKUNIT('name', 'YinyangKMeans:USCensus:30:0.6');  
//YinyangKMeans:=ML.onegroupfaster.YinyangKMeans(dDocuments,dCentroids,30,0.6); 

//#WORKUNIT('name', 'YinyangKMeans:USCensus:30:1.0');  
//YinyangKMeans:=ML.onegroupfaster.YinyangKMeans(dDocuments,dCentroids,30,1.0); 

// #WORKUNIT('name', 'YinyangKMeans:HTHOR:KEGG:30:0.3');
#WORKUNIT('name', 'Comparison:THOR:DP100:30:0.3');
n := 18;
nConverge := 0.3;
// YinyangKMeans:=ML.yinyang.drafts.multigroup_debug.YinyangKMeans(dDocuments,dCentroids,2,0.3);  
// YinyangKMeans:=ML.yinyang.drafts.v2combinev3.YinyangKMeans(dDocuments,dCentroids,16,0.3);
YinyangKMeans:=ML.yinyang.drafts.yinyangkmeansv4_test.YinyangKMeans(dDocuments,dCentroids,n,nConverge);
OUTPUT(YinyangKMeans.Allresults, NAMED('YinyangKMeansAllresults'));                                       
OUTPUT(YinyangKMeans.Convergence, NAMED('YinyangKMeans_Iterations')); 
//OUTPUT(KMeans.Allegiances(), NAMED('KMeansAllegiances'));

KMeans:=ML.yinyang.drafts.onegroupfaster_comp.KMeans(dDocuments,dCentroids,n,nConverge); 
OUTPUT(KMeans.Allresults, NAMED('KMeansAllresults'));
OUTPUT(KMeans.Convergence, NAMED('KMeansTotal_Iterations')); 
lCompare := RECORD
Types.NumericField.id;
Types.NumericField.number;
Boolean pass;
END;

result := JOIN(YinyangKMeans.Allresults,KMeans.Allresults,LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(lCompare, SELF.pass := IF(LEFT.values = RIGHT.values, TRUE, FALSE), SELF := LEFT;));
OUTPUT(result, NAMED('resultscomparison'));

// change the values to value
SHARED lIterations:=RECORD
	TYPEOF(Types.NumericField.id) id;
	TYPEOF(Types.NumericField.number) number;
	SET OF TYPEOF(Types.NumericField.value) values;
 END;
Types.NumericField normalizerst(lIterations L, UNSIGNED c) := TRANSFORM
  SELF.value := L.values[c];
	SELF := L;
END;

 //normalize result
rst_YinyangKMeans := NORMALIZE(YinyangKMeans.allresults, YinyangKMeans.Convergence, normalizerst(LEFT, COUNTER)); 
// rst := NORMALIZE(yinyangkmeans.allresults, iterations, TRANSFORM(Types.NumericField, SELF.value := LEFT.values[COUNTER],SELF := LEFT)); 
OUTPUT(rst_YinyangKMeans, NAMED('rst_YinyangKMeans'));

rst_KMeans := NORMALIZE(KMeans.allresults, KMeans.Convergence, normalizerst(LEFT, COUNTER)); 
// rst := NORMALIZE(yinyangkmeans.allresults, iterations, TRANSFORM(Types.NumericField, SELF.value := LEFT.values[COUNTER],SELF := LEFT)); 
OUTPUT(rst_KMeans, NAMED('rst_KMeans'));

// detail_result_comparison := JOIN(rst_YinyangKMeans,rst_KMeans,LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(lCompare, SELF.pass := IF(LEFT.value = RIGHT.value, TRUE, FALSE), SELF := LEFT;));
detail_result_comparison := JOIN(rst_YinyangKMeans,rst_KMeans,LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number AND LEFT.value = RIGHT.value, TRANSFORM(lCompare, SELF.pass := FALSE, SELF := LEFT;), FULL ONLY);
OUTPUT(detail_result_comparison, NAMED('detail_result_comparison'));