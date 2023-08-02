#from speakerDiarization import main,fmtTime
from pydub import AudioSegment
AudioSegment.converter = r"H:\Btech-Proj\Speaker_Diarization\ffmpeg\bin\ffmpeg.exe"

import os
#from ibm_st import sp_to_txt
from symbl_stt import sp_to_txt


def make_chunks(spkrs,meet_audio,out_file='transcribe.txt'):
    new_dict = {}
    for spkr in spkrs:
        for i in range(len(spkrs[spkr])):
            new_dict[spkrs[spkr][i]['start']] = [spkr,i]
    sort_new_dict = sorted(new_dict)

    audio = AudioSegment.from_wav(meet_audio)
    for i in sort_new_dict:
        spkr,ind = new_dict[i][0],new_dict[i][1]
        start,end = spkrs[spkr][ind]['start'],spkrs[spkr][ind]['stop']
        #print(start,end)
        chunk = audio[start:end]
        chunk_file = 'Chunks\chunk'+str(spkr)+str(ind)+'.wav'
        chunk.export(chunk_file,format='wav')
        sp_to_txt(chunk_file,out_file,spkr)
