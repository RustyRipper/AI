% siec semantyczna

element(kola).
element(szczotka).
element(czujnik).
element(zderzak).
element(pojemnik).
element(filtr).
element(wentylator).

posiada(pojemnik, filtr).
posiada(pojemnik, wentylator).

element_blokujacy(zderzak, obrys_robota).
element_blokujacy(szczotka, obrys_robota).
element_blokujacy(kola, obrys_robota).


% baza wiedzy

przyczyna(blad1, blokada).
przyczyna(blad1, kolo_zwisa).
przyczyna(blad2, szczotka_nie_obraca_sie).
przyczyna(blad5, kolo_zablokowane).
przyczyna(blad6, zabrudzony_czujnik).
przyczyna(blad6, utknal_w_ciemnosci).
przyczyna(blad6, kolo_zawisa).
przyczyna(blad8, wentylator_zacial_sie).
przyczyna(blad8, zapchany_filtr).
przyczyna(blad9, blokada).
przyczyna(blad9, zabrudzony_czujnik).
przyczyna(blad10, kolo_nie_obraca_sie).
przyczyna(blad11, szczotka_zaczepila_sie).
przyczyna(blad11, kolo_zaczepilo_sie).
przyczyna(blad14, zle_zamontowany_pojemnik).

powoduje(obrys_robota, blad1).
powoduje(szczotka, blad2).
powoduje(kola, blad5).
powoduje(czujnik, blad6).
powoduje(obrys_robota, blad6).
powoduje(wentylator, blad8).
powoduje(filtr, blad8).
powoduje(czujnik, blad9).
powoduje(obrys_robota, blad9).
powoduje(kola, blad10).
powoduje(szczotka, blad11).
powoduje(kola, blad11).
powoduje(pojemnik, blad14).

napraw(blokada, obrys_robota).
napraw(kolo_zwisa, kola).
napraw(szczotka_nie_obraca_sie, szczotka).
napraw(kolo_zablokowane, kola).
napraw(zabrudzony_czujnik, czujnik).
napraw(utknal_w_ciemnosci, obrys_robota).
napraw(wentylator_zacial_sie, wentylator).
napraw(zapchany_filtr, filtr).
napraw(kolo_nie_obraca_sie, kola).
napraw(szczotka_zaczepila_sie, szczotka).
napraw(kolo_zaczepilo_sie, kola).
napraw(zle_zamontowany_pojemnik, pojemnik).

% predykaty

powod_powodowany_przez_element(Element, Powod) :-
    przyczyna(Blad, Powod),
    (
    	(
    	element(Element),
    	powoduje(Element, Blad),
    	napraw(Powod, Element)
    	)
    ;
    	(
    	element_blokujacy(Element, NadElement),
    	powoduje(NadElement, Blad),
    	napraw(Powod, NadElement)
    	)
    ).

element_w_bledzie(Blad, Element, PodElement)  :-
	przyczyna(Blad, _),
    (
    	(
        element(Element),
        powoduje(Element, Blad)
        )
    ;
        (
        powoduje(Element, Blad),
        element_blokujacy(PodElement, Element)
        )
    ).

powoduje_wiecej_niz_jeden_element(Blad) :-
    powoduje(Element1, Blad),
    powoduje(Element2, Blad),
    Element1 \= Element2.