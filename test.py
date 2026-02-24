"""
GigaChat SDK:
- credentials — постоянный ключ (Base64 от client_id:client_secret), по нему SDK сам получает временный токен через OAuth.
- access_token — уже полученный временный токен (JWT/JWE); передаётся напрямую, без вызова OAuth.
"""
from gigachat import GigaChat

# Временный токен (уже полученный, например из личного кабинета или своего OAuth-запроса)
token = """eyJjdHkiOiJqd3QiLCJlbmMiOiJBMjU2Q0JDLUhTNTEyIiwiYWxnIjoiUlNBLU9BRVAtMjU2In0.GlaTIACTwmphmWk65pSIKs9gqUUcKdp0pUR25yCvz...Ax9jNoefNOY31oi3z9PLm-EdHTjbyikrFVjmNVPcbeZtF8wwMJIB8DtC9uqZDrGY0-zU5wyA6QPZSrjhN0.0XjBsRk-PMK2eRfA0GhAs71urGpmkwoGqNbI1ZyIRAQ"""

giga = GigaChat(
    access_token=token,  # временный токен — не вызываем OAuth
    verify_ssl_certs=False,
)

response = giga.get_models()
print(response)
