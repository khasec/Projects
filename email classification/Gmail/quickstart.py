from __future__ import print_function


import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import pandas as pd

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def main():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    print('New email? y/n')
    x = input()
    
    if os.path.exists('token.json') and not x == 'y':
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json',SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        # Call the Gmail API
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me',labelIds=['INBOX']).execute()
        messages = results.get('messages', [])
        messages = [service.users().messages().get(userId='me', id=msg['id'],format='raw').execute() for msg in messages]


        test= pd.read_csv("email.csv")
            
            
        a = list(test['Message'])
        

        for i in messages:

            data = {
                    'Category': [0],
                    'Message': [i['snippet']],   
                    }
            
            if data['Message'][0] not in a:

                string=data['Message'][0]
                print(f'{string} \n added to the dataset')

                df = pd.DataFrame(data)

                # append data frame to CSV file
                df.to_csv('email.csv', mode='a', index=False, header=False)

        
        

        
    except HttpError as error:
        # TODO(developer) - Handle errors from gmail API.
        print(f'An error occurred: {error}')


if __name__ == '__main__':
    main()

