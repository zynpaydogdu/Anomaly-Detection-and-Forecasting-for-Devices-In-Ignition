def doGet(request, session):

    # Basit dahili sağlık kontrolü
    return {"contentType":"application/json",
            "status":200,
            "response": system.util.jsonEncode({"status":"ok","ts":system.date.now()})}