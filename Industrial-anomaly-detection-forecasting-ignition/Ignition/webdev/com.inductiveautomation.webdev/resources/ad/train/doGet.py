def doGet(request, session):
	import system
	log = system.util.getLogger("WEBDEV-TRAIN-GET")
	log.info("WEBDEV GET ÇALIŞTI")
	return {
	"contentType": "text/plain",
	"status": 200,
	"body": "GET ÇALIŞTI"
	}