import smtplib


def send_mail(From, To, password, message):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(From, password)

    server.sendmail(From, To, message)
    server.quit()

