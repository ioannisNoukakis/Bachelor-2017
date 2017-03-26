from unittest import TestCase
from mailsUtils import send_mail


class TestSend_mail(TestCase):
    def test_send_mail(self):
        send_mail("ioannisbachelorbot@gmail.com", "inoukakis@gmail.com", "<mdp>", "Test")
        self.assertEqual(True, True)
