import os
import yaml
import smtplib
from os.path import join
import getpass

project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

if cfg['specs']['send_finish_email']:
    gmail_user = 'luxiaoxuan555@gmail.com'
    gmail_password = 'wydky123'

    sent_from = gmail_user
    to = ['luxiaoxuan555@gmail.com']
    subject = 'Auto-EM Finish'
    msg = 'Subject:{}\n\n {}!'.format(subject, os.getcwd())

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(sent_from, to, msg)
        server.close()
    except:
        print('Something went wrong...')
        exit()

