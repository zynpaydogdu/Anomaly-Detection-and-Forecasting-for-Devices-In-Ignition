def doGet(request, session):
	try:
		payload = system.util.jsonDecode(request['postData'])
		PY_BASE = "https://iot-primary-tmmt-dev.tmmt.com:8043"
		http = system.net.httpClient(connectTimeout=10000, readTimeout=60000)
		resp = http.post(PY_BASE + "/infer",headers={"Content-Type":"application/json"},data=system.util.jsonEncode(payload))
		return {"contentType":"application/json","status":resp.getStatusCode(),"response":resp.getBody()}
	except Exception as e:
		return {"contentType":"application/json","status":500,"response":system.util.jsonEncode({"status":"error","message":str(e)})}