#Bütün tagleri çeken fonksiyon.
def getAllAtomicTags(root="[default]test"): #kök yol varsayılan değeri [default]
	#Bu kökten itibaren atomi tipte olan tüm tagleri toplayan fonksiyon.
    out = [] #sonuçları tutacak liste
    def walk(path):
        # path altında tarama yapar; klasör bulursa recursive olarak derine iner.
        try:
            results = system.tag.browse(path).getResults()
        except Exception as e:
            system.util.getLogger("getAllAtomicTags").warn(
                "Tarama hatası: path='{}', error={}".format(path, e)
            )
            return

        for r in results: #taramada dönen her bir sonucu sırayla dolaşır.
            tag_type = str(r["tagType"]) #sonuç öğesinin tag tipini alır. str ile string olmaya zorlar
            full     = r["fullPath"]   #tagin tam yolunu alır
            name     = r["name"]       #tagin kısa adı

            if "Folder" in tag_type:#bu sonuç bir klasör mü bakar
                # alt klasöre in
                walk(str(full)) #klasörse, bu klasörün tam yolunu stinge çeviren walk fonk teniden çağrılır.
            elif "Atomic" in tag_type: #klasör değilse, atomic tag mi diye bakılır.
            	#atomic doğrudan değer tutan tag tipi
                out.append({ #atomic tag bulduğunda sonuç listesine nesne eklenir.
                    "label": str(name), #kullanıcıya gösterilecek ad
                    "value": str(full) #programatik olarak kullanacak değer, tam yol.
                })

    walk(root) #tarama walk ı başlatır.
    return out #tümü tarandıktan sonra toplanan atomic taglerin listesini döndürür.