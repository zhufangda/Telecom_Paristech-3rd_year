from py2ifttt import IFTTT

def notification(project, event, result):
    ifttt = IFTTT('bRXZ1sur_A6fFQb2O9S8Io', 'notificaiton_python')
    ifttt.notify(project, event, result)

    