@echo off
curl -k -L -X POST "https://gigachat.devices.sberbank.ru/api/v1/chat/completions" ^
-H "Content-Type: application/json" ^
-H "Accept: application/json" ^
-H "Authorization: Bearer eyJjdHkiOiJqd3QiLCJlbmMiOiJBMjU2Q0JDLUhTNTEyIiwiYWxnIjoiUlNBLU9BRVAtMjU2In0.GlaTIACTwmphmWk65pSIKs9gqUUcKdp0pUR25yCvz...Ax9jNoefNOY31oi3z9PLm-EdHTjbyikrFVjmNVPcbeZtF8wwMJIB8DtC9uqZDrGY0-zU5wyA6QPZSrjhN0.0XjBsRk-PMK2eRfA0GhAs71urGpmkwoGqNbI1ZyIRAQ" ^
-d "{\"model\": \"GigaChat-2-Max\", \"messages\": [{\"role\": \"user\", \"content\": \"Привет! Как дела?\"}], \"stream\": false, \"repetition_penalty\": 1}"
