import json,symbl
from os import path
import pickle
path = 'UK.wav'

query_parameters = {
    'name' : 'Test Meeting'
}
conversation = symbl.Audio.process_file(file_path=path,parameters=query_parameters)
#response = conversation.messages()._messages
pickle.dump(conversation,open('symb.pkl','wb'))
print(conversation.get_messages()._messages)