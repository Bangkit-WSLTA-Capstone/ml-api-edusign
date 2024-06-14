import json
from translation.translation import translate
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def translate_video(request):
    if(request.method == "POST"):
        request_body = request.body.decode('utf-8')
        data = json.loads(request_body)
        file_link = data.get('link')
        result = translate(file_link)
        return JsonResponse({"message": "Video translated succesfully", "result": result}, status=200)