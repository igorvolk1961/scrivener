REM Пример из документации GigaChat OAuth:
REM   curl -X POST "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
REM   -H "Content-Type: application/x-www-form-urlencoded"
REM   -H "Accept: application/json"
REM   -H "RqUID: <уникальный_идентификатор_запроса>"
REM   -H "Authorization: Basic authorization_key"
REM   --data-urlencode "scope=GIGACHAT_API_PERS"
REM Ниже — то же для Windows (-k отключает проверку SSL, RqUID = случайный).
curl -k -L -X POST "https://ngw.devices.sberbank.ru:9443/api/v2/oauth" ^
-H "Content-Type: application/x-www-form-urlencoded" ^
-H "Accept: application/json" ^
-H "RqUID: %RANDOM%%RANDOM%" ^
-H "Authorization: Basic M2RjNGFkZGEtOTA0MS00MzI0LTlmNzUtNzczNTIxNmQ0Zjk1OmFmNzE0NWQ3LWY5NDQtNGExNC05ZmZmLWEzYjE3Zjk5MjgwYw==" ^
--data-urlencode "scope=GIGACHAT_API_PERS"
pause
