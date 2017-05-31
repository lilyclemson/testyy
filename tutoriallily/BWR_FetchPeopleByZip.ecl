IMPORT Tutoriallily; 
ZipFilter :='33024'; 
FetchPeopleByZip :=
FETCH(Tutoriallily.File_TutorialPerson, Tutoriallily.IDX_PeopleByZIP(zip=ZipFilter), RIGHT.fpos);
OUTPUT(FetchPeopleByZip);