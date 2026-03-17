from passlib.context import CryptContext
pwd = CryptContext(schemes=['bcrypt'], deprecated='auto')
print(pwd.hash('AdminReset2026!'))  # Example password, under 72 chars