# ML MIPT homework 1

# Окружение

Установка окружения

```bash
pip install -r requirements.txt
```

Установка гит хуков
```bash
pip install pre-commit
pre-commit install --install-hooks
```

Ручной запуск хуков
```bash
pre-commit run --show-diff-on-failure --color=always --all-files
```

Установка обновлений в перечне окружения requirements.in
```bash
pip install pip-tools
pip-compile requirements.in
```

Обновление пакетов в окружении
```bash
pip install pip-tools
pip-sync requirements.txt
```

# Запуск пайплайна

При первом запуске, используйте сперва:
```bash
dvc init
```
После этого воспользуйтесь командой:
```bash
dvc repro
```

# Тестирование

Для запуска тестов с покрытием в корне репозитория запустите:
```bash
pytest test --cov=src --cov-report html --cov-report term
```
После этого откройте в браузере файл:
```
htmlcov/index.html
```
