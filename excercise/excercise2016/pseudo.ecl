Variables:

iter : unsigned // equal to COUNTER, i.e. current iteration number

t : unsigned ,  1 <=1 <=K

n/nt: unsigned,

K: unsigned

nConverge/tConverge: real

x := RECORD

unsigned xid

unsigned number

real value

real ub 

real lbs[lb1...lbt] 

END;

centroid := RECORD

unsigned cid

unsigned number

real value

END;

groupt := RECORD

unsigned gid

DATASET centroid

END;

deltaV := RECORD

unsigned cid

unsigned number // the number of total members

unsigned newXid [] // the set of xid of newly joined data points

unsigned oldXid[] // the set of xid of data points that left this centroid 

END;







Functions:

//standard Kmeans 

Function SK(d01,d02,n,nConverge):

Return result

END;

//three conditions:1)initialize the ub 2) update ub after groupFilter 3) update ub after localFilter

Function updateUb(x,SK.AllResult==[], deltaC==[], groupFilter ==[], localFilter==[], iter==1) :

If iter ==1 then

ub <- updated from SK.AllResult

Elseif groupFilter == None then

ub <- ub + deltaC[ub]

Else then

ub <- updated from the result of localFilter

Return ub

END;

//2)initialize the lbs 2) update lbs after groupFilter 3) update lbs after localFilter

Function updateLbs(x,SK.AllResult==[], deltaC==[], groupFilter ==[], localFilter==[], iter==1):

If iter ==1 then

//ub <- updated from SK.AllResult
groupDis <- group(SK.AllResult , c.gid)// group by gid

For i in t

If x.ub not belongs to groupDis[i] then

lbs[i] <- Min(sort(groupDis[i]))

Else then

lbs[i] <- secondMin(sort(d01SKGroup[i]))

//update after passing groupFilter
Elseif groupFilter == None then

lbs <- lbs - deltaG

//update after passing localFilter

Else then

//lbs <- updated from the result of localFilter and groupFilter

Return lbs

END;

//record the membership change of each cluster. It's useful for the centroids update

Function updateDeltaV( deltaV, iter , xid, indicator == 1  ):

If iter ==1 then

deltaV'<- updated from SK.AllResult

Elseif indicator == 1 then

deltaV.newXid <- add new member

deltaV.number < deltaV.number + 1

Else then

deltaV.oldXid <- add old member

deltaV.number < deltaV.number - 1

END;

//update centroids of d02

Function updateCentroids(d02 , SK.AllResult) :

If iter == 1 then

d02' <- update as the SK does

Else then

For each c in do2

d02'.c <- (c - deltaV.oldXid + deltaV.newXid)/deltaV.number

Return d02'

END;

//calculate the drift of each centroid 

Function deltaC(d02', do2):

deltaC[] = distance(d02' , d02)

Return deltaC[]

END;

//calculate the max drift of centroids in each group

Function deltaG(deltaC[]):

deltaG <- MAX (group(deltaC[],c.gid))

Return deltaG[]

END;

//filter out the data points that do on change best centroid

Function groupFilter(d01, deltaC, deltaG):

For x in d01:

For Gj in x.lbs:

      x.ubNew <-  x.ub + deltaC[x.ub]

      x.lbNew <- lb – maxDrift[Gj]

If    x.ubNew <= x.lbNew then

coninue

Else then

      If    x.ub <= x.lbNew then

      coninue

      Else then

      group'.add(Gj)

x.ub  <- updateUb()

x.lbs <- updateLbs()

Return group'[]

END;

//filter out centroids that are impossible to be the new best centroid

Function localFilter(group',x):

x.ub <- updateUb()

x.lbs <- updateLbs(lFitler, Gt)

Else then

result <- group(d01SK, c.cid)

Return x'[]

//calculating the distance of two points

Function distance(d1, d2) : 

Return distValue




Function YinyangKmeans(K, d01, do2,n, nConverge):

//Step 1: Set t to a value no greater than k/10 and meeting the space constraint. Group the initial centers into t groups by running K-means on just those initial for five iterations to //produce reasonable  groups while incurring little overhead.

//set t

If k/10 <=1 then

t <- 1

else then

t<- K/10

//Set  d0t: the dataset of t centroids

If t =1 then

Gt <- dot

else then

dot <- getRandomCentroinds(t) // get random t centroids or other methods such as cross validation etc.

// Set nt : the maximum iterations

nt <- 5 // based on the paper or other methods such as cross validation etc.

//Set up tConverge: the threshold of converge

tConverge <- nConverge // equal to nConverge. May change it later

//running standard K-means on do2

Gt <- SK(d02,dot, t,nt=5, tConverge = nConverge)//get t groups of centroids

 

//Step 2: Run the standard K-means on the points for the first iteration. For each point //x, set the upper bound ub(x) = d(x; b(x)) and the lower bounds lb(x;Gi) as the shortest
//distance between x and all centers in Gi excluding b(x).

//Run SK on d01

d01SK<-SK(d01,d02,n,nConverge)

//initialize ub and lbs of each data point x of d01

For x in d01

x.ub <- updateUb(x,d01SK.AllResult):

x.lbs <- updateLbs(x,d01SK.AllResult)


//Step 3: Repeat until convergence:

If converge == False then
//3.1: Update centers by Equation 1, compute drift of each center (c), and record the //maximum drift for each group Gi

newCentroids <- updateCentroids(d01SK) // the updated dataset of all centroids

deltaCentroid <- deltaC(newCentroids,d02) // the distance change of all centroids

maxDrift <- maxDrift(deltaCentroid)// the max distance change of each group Gt

 

//3.2 Group filtering: For each point x, update the upper bound ub(x) and the group lower //bounds lb(x;Gi) with ub(x) + (b(x)) and lb(x;Gi) respectively. Assign the temporary global //lower bound as lb(x) = mint i=1 lb(x;Gi). If lb(x) ub(x), assign b0(x) with b(x). Otherwise, //tighten ub(x) = d(x; b(x)) and check the condition again. If it fails, find groups for which
//lb(x;Gi) < ub(x), and pass x and these groups to local filtering.

gFilter <- Function groupFilter(d01,Gt, maxDrift):

For x in d01:

For Gj in x.lbs:

x.ubNew <-  x.ub + deltaCentroid[x.ub]

x.lbNew <- lb – maxDrift[Gj]

If    x.ubNew <= x.lbNew then

coninue

Else then

x.ubNew <-  x.ub

If    x.ubNew <= x.lbNew then

coninue

Else then

gfilter <- gfilter.add(Gi)

If gfilter == None then

x.ub  <- updateUb(Gt,maxDrift)

x.lbs <- updateLbs(Gt, maxDrift)

Else then
//3.3 Local filtering: For each remaining point x, filter its remaining candidate centers with //the so-far-found second closest center, compute the distances from x to the centers
//that go through the filter to find out the new b(x), and update the group lower bound //lb(x;Gi) with the distance to the second closest center. For groups blocked by the group //filter, update the lower bounds lb(x;Gi) with lb(x;Gi). Update ub(x) with d(x; b(x)).

lFilter <- Function localFilter(gFilter):

x.ub <- updateUb(lFilter)

x.lbs <- updateLbs(lFitler, Gt)

Else then

result <- group(d01SK, c.cid)

return result


