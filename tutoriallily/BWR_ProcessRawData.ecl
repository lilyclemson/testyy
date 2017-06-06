IMPORT Tutoriallily, Std;

Tutoriallily.Layout_People toUpperPlease(Tutoriallily.Layout_People pInput) := TRANSFORM
SELF.FirstName := Std.Str.ToUpperCase(pInput.FirstName); 
SELF.LastName := Std.Str.ToUpperCase(pInput.LastName); 
SELF.MiddleName := Std.Str.ToUpperCase(pInput.MiddleName); 
SELF.Zip := pInput.Zip; 
SELF.Street := pInput.Street; 
SELF.City := pInput.City; 
SELF.State := pInput.State; 
END;

OrigDataset := Tutoriallily.File_OriginalPerson;
UpperedDataset := PROJECT(OrigDataset, toUpperPlease(LEFT));
OUTPUT(UpperedDataset,,'~tutorial::lily::TutorialPerson',OVERWRITE);