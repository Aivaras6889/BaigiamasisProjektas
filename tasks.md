Galutinis atsiskaitymas
Due July 18, 2025 11:59 PM
Instructions
Nedarantiems savo įdėjos

Galutinis atsiskaitymas

 

Egzamino tikslas yra sukurti korektiška algoritmą, kuris kuo tobuliau sugebėtų išspręsti pagrindinį duomenų rinkinio uždavinį ir leistų naudotojams juo lengvai naudotis. Rinkinį rasite čia:
https://benchmark.ini.rub.de/gtsrb_dataset.html


Pagrindiniai Uždaviniai:

Turite paversti duomenų rinkinį į duomenų bazės rinkinį, tai nereiškia, kad reikia nuotraukas sudėti į duomenų bazę, tiesiog vietoje CSV, kuris yra šalia failų, reikėtų sukurti duomenų bazę, kuri saugotu csv esančius duomenis.
etapai: 
# Paversti duomenu rinkini i duomenu baze rinkini tik csv kuris salia failu 
-----------------------------------------------------------------------------
Atlikite įvairias ženklų transformacijas (kurios būtų naudingos, ištraukite reikalingus duomenis, šis duomenų rinkinys, jau turi papildomus rinkinius, kuriuose yra įvairūs papildomi elementai, jeigu ištrauksite patys, įvertinimas bus didesnis, bet išgauti papildomus parametrus būtina).
# Isgauti papildomus duomenys
----------------------------------------------------------------------------------
Komunikacija su duomenų baze privalo vykti per SQL Alchemy biblioteką
Naudotojas privalo galėti įdėti naują savo nuotrauką, su kelio ženklu. Turi būti galima įkelti nuotrauką tiek testavimo tiek treniravimo apmokymui.
# Komnunikacija su duomenu baze su sql_alchemy
-----------------------------------------------------------------------------
Turi būti minimali naudotojo sąsają, kad naudotojas galėtų paleisti modelio apmokymą įkėlęs norimas papildomas nuotraukas.
------------------------------------------------------------
Turi būti galimybė naudotojui neįkeliant nuotraukos (nei į testavimą nei treniravimą) gauti rezultatą. 
--------------------------------------------------------------------------
T.Y pasirenką pasirinkimą vienos nuotraukos nuspėjimas ir jam išvedamas būtent tos nuotraukos rezultatas.
--------------------------------------------------------------------------------
Galutiniai rezultatai taip pat saugomi duomenų bazėje.
--------------------------------------------------------------------------------
Privalo būti, bent vienas paprastas Modelis (toks, kaip KNN, SVM, Random forest, linear, logistic, Boosting, Arima, K-means ar kitas modelis (ne visi iš išvardytų modelių tinka šitam konkrečiam uždaviniui, įsitikinkite, kad jūsų uždavinys ir modelis yra suderinami))
-------------------------------------------------------------------------

Privalo būti bent vienas Neuroninis tinklas, kuris atliktų spėjimus, pavyzdžiui, RNN, CNN, FEED Forward neural network, Transformer.

Palyginti paprastą SL modelį ir neuroninį tinklą, kuris yra geresnis, padarykite išvadas.

Privaloma Neuroniniui tinklui atlikti, bent 20-30 įvairių pakeitimų (įskaitant su hyperparametrais), tai įeina sluoksnių, neuronų kiekio, learning rate, batch size, optimizer ir kitų hyperparametrų keitimas visi šie testai užrašomi išvadų lentelėje.

Privalote pateikti kuo daugiau metrikų, tinkančių uždaviniui

Privalote pateikti įvairius naudingus grafikus naudotojui paprašius (pavyzdžiui, kokių ženklų daugiausiai) arba sudėtingesnius grafikus, kaip modelio rezultatai atskiroms grupėms nuotraukų. (Čia gali dirbti jūsų fantazija, turi būti, bent 4 prasmingi grafikai).
Pagrindinės sąlygos

Neuroninio tinklo panaudojimas sudaro 3 balus (vadinasi, padarius tobulą neuroninį tinklą, gausite +3 balus)

Korektiškos išvados su įvairių hyperparametrų testavimo rezultatais ir tekstinėmis išvadomis sudarys 20% balo

Likusios dalys, kaip duomenų bazės panaudojimas, naudotojo įvestis, korektiškas duomenų paruošimas sudarys likusius 5 balus.
 

 

Papildomi reikalavimai

Privalomas Github naudojamas (atsiskaitomasis darbas turi būti įkeltas į Github, jo nuoroda pasidalinta su dėstytoju), taip, pat darbo metu privalote atnaujinti kodą (negalima, įkelti visko prieš atsiskaitymą, turėtų būti, bent 4 šakos ir bent 5 commitai (per visas šakas) šio punkto nevykdymas (-1 balas)

Privalomas tvarkingas projektas, su tinkama struktūra (klasės, metodai, funkcijos, modeliai viskas atskiruose failuose)
 

Bonus balai


 Įsiaiškinsite ir pamėginsite panaudoti transformerius (darbui su nuotraukomis, tinkami specifiniai transformeriai).

 https://en.wikipedia.org/wiki/Vision_transformer (+3 balai, už tvarkingą ir pilnai veikiantį sprendimą)

 Padarysite grafine sąsają (su tkinter, arba flask) ar kitu jums patinkančiu įrankiu. (+2 balai, už pilną ir užbaigtą sprendimą, turėtų būti, ne vienas puslapis ir įvairūs elementai, kaip nuotraukų įkėlimas ir t.t)
 
 Flaske padarykite galimybę keisti hyperparametrus (+ 0.5 balo)
 
 Padarykite, kad programa galėtų veikti su video (mašinos važiuojančios kelyje) ir galėtų realiu laiku atsinaujinti, kokius ženklus mato. (+1 balas)
 
 Panaudokite, daugiau nei tik pikselius iš nuotraukos (negalima konvertuoti į pikselius ir tiek, reikia panaudoti, bent vieną iš šių arba kitų jums žinomų parametrų Haar/Hog/Hue Histograms)
 

 

Papildomos sąlygos

Atsiskaitymo metu, taip pat būsite klausiami klausimų apie teoriją, 
šie klausimai nebūtinai bus iš to, ką panaudojate darbe, jie sudarys 16% galutinio atsiskaitymo 
(klausimai gali būti iš bet kurios temos, kurią praėjome), 
vienas klausimas 0.4 balo.

Negebant paaiškinti, kažkurios eilutės, kuri yra jūsų parašytame kode, balas mažinamas per 0.5, vadinasi, negebant paaiškinti 2 eilučių, netenkate 10% galutinio įvertinimo.

Akivaizdžiai aklai kopijuojant ChatGPT kodą ir nesuprantant, kas yra nukopijuota, maksimalus balas yra 5.

Galutinis įvertinimas, taip pat priklausys ir nuo jūsų galutinio modelio tikslumo (galbūt šis modelis nebus neuroninis tinklas). 

Įvertinus visus, bus imama kreivė t.y. jeigu geriausio studento rezultatas bus MSE = 1000 arba tikslumas = 90%, kiekvienas 10% nuokrypis, nuo geriausio rezultato, vertinamas, kaip -0.5 balo, vadinasi, jeigu geriausias studentas gaus MSE = 1000, o jūsų mse bus 1400 (40% didesnis, nei geriausio studento), maksimalus jūsų balas bus 8. Daugiausiai šitaip galima netekti iki 3 balų, didesnis skirtumas balo nebemažina.

//Kritinės klaidos, kaip ne to duomenų rinkinio naudojimas testavimui arba treniravimui ir panašūs dalykai (-2 balai)