IMPORT tutoriallily; 
STRING10 ZipFilter := '' :STORED('ZIPValue'); 
resultSet :=
FETCH(tutoriallily.File_TutorialPerson, tutoriallily.IDX_PeopleByZIP(zip=ZipFilter), RIGHT.fpos);
OUTPUT(resultset)