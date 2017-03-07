EXPORT threed := MODULE

lData :=RECORD
    INTEGER  OSM_ID;
    REAL  LONGITUDE;
    REAL  LATITUDE;
    REAL  ALTITUDE;
END;

lData transDataset(lData l, INTEGER c):= TRANSFORM
		SELF.OSM_ID := c;
		SELF := l;
END;

dataTemp := DATASET('~::3dspatialnetwork.txt' ,lData,  CSV);
EXPORT input :=PROJECT(dataTemp, transDataset(LEFT, COUNTER));
// EXPORT input := TABLE(dataTemp, {LONGITUDE,LATITUDE,ALTITUDE});


END;