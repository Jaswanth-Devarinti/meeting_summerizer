import symbl,json
 
def sp_to_txt(path,out_file,spkr):
    path = path.replace('\\','/')
    fh = open(out_file,'a+')
    conversation = symbl.Audio.process_file(file_path=path)
    response = conversation.get_messages()._messages
    text = ''
    for part in response:
        text = text+(str(part._text)+' ')
    fh.write('spkr'+str(spkr+1)+' : '+text+'\n')
    fh.close()
