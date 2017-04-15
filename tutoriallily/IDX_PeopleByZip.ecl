IMPORT tutoriallily;
EXPORT IDX_PeopleByZIP := 
INDEX(Tutoriallily.File_TutorialPerson,{zip,fpos},'~tutorial::lily::PeopleByZipINDEX');