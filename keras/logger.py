def info(header: str, msg):
    print("\e[31m" + header + "\e[39m", msg)


def error(header: str, msg):
    print("\e[32m" + header + "\e[39m", msg)
