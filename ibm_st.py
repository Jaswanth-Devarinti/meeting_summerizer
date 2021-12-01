import json
from os.path import join, dirname
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
  
   
# Insert API Key in place of 
# 'YOUR UNIQUE API KEY'
authenticator = IAMAuthenticator('GT7fppYOSTxHSkay3vt_IQSqc8Be46gfJ52WDFTLvYgc') 
service = SpeechToTextV1(authenticator = authenticator)
   
#Insert URL in place of 'API_URL' 
service.set_service_url('https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/30d9b4c1-020b-4e0d-9d35-47996a98f747')
   
# Insert local mp3 file path in
# place of 'LOCAL FILE PATH' 
def sp_to_txt(path,out_file,spkr):
    path = path.replace('\\','/')
    fh = open(out_file,'a+')
    with open(join(dirname('__file__'), path), 
            'rb') as audio_file:
        
            dic = json.loads(
                    json.dumps(
                        service.recognize(
                            audio=audio_file,
                            content_type='audio/wav',   
                            model='en-US_NarrowbandModel',
                        continuous=True).get_result(), indent=2))
    
    # Stores the transcribed text
    text = ""
    
    while bool(dic.get('results')):
        text = dic.get('results').pop().get('alternatives').pop().get('transcript')+text[:]
        
    fh.write('spkr'+str(spkr+1)+' : '+text+'\n')
    fh.close()