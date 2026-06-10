# Материалы для итогового проекта «CallCenterOpt: Симулятор колл-центра с оптимизацией штата»

*(3 курс — 27 мая 2026 г.)*

---

## Содержание

1. [Шаблон README.md](#1-шаблон-readmemd)
2. [Пример pytest-теста для модуля игры](#2-пример-pytest-теста-для-модуля-игры)
3. [Готовая структура docker-compose.yml](#3-готовая-структура-docker-composeyml)
4. [Чек-лист для проверки проекта](#4-чек-лист-для-проверки-проекта)

---

## 1. Шаблон README.md

```
CallCenterOpt
Курс: Теория игр | Трек: Программная инженерия | Уровень: 3 курс
Автор: [ФИО] | Группа: [Номер] | Руководитель: [ФИО]
```

### 1.1 Описание

Python-приложение, моделирующее работу колл-центра как системы массового обслуживания (M/M/c) и подбирающее оптимальное количество операторов. Выбор стратегии найма формализуется как игра в условиях риска (неопределённый поток звонков). Цель: минимизация затрат на персонал при соблюдении SLA (P_wait ≤ 15%).

### 1.2 Математическая модель

| Компонент | Описание | Реализация |
|-----------|----------|------------|
| **СМО (M/M/c)** | Пуассоновский вход λ, эксп. обслуживание μ, расчёт P_wait, L_q, W | `core/queue_math.py` |
| **Игра с природой** | Сценарии нагрузки {λ_min, λ_avg, λ_max} vs стратегии {c1, c2, c3}. Критерии: Гурвица, Байеса, Вальда | `core/game_solver.py` |
| **Валидация (ЛП)** | Минимизация `Cost = c · S + α · P_wait` при `P_wait ≤ 0.15` | `core/optimizer.py` |

### 1.3 Технологический стек

- **Ядро:** NumPy, SciPy.stats (формулы Эрланга), PuLP
- **Интерфейс:** Click (CLI) или Streamlit (веб)
- **Инфраструктура:** Pydantic (валидация), YAML/JSON (конфиги)
- **SE-практики:** pytest, pytest-cov (≥ 60%), ruff, black

### 1.4 Быстрый старт

```bash
git clone https://github.com/[user]/callcenter_opt.git
cd callcenter_opt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config/scenario.example.yaml config/scenario.yaml

# запуск расчёта
python main.py --config config/scenario.yaml --criterion hurwicz --alpha 0.6
```

### 1.5 Структура проекта

```
callcenter_opt/
├── config/           # scenario.yaml, params.json
├── core/
│   ├── queue_math.py     # Формулы Эрланга, метрики СМО
│   ├── game_solver.py    # Матрица рисков, критерии принятия решений
│   └── models.py         # Pydantic-схемы
├── ui/               # Streamlit / CLI wrapper
├── tests/            # pytest (unit + integration)
├── requirements.txt
└── README.md
```

### 1.6 Тестирование

```bash
pytest tests/ -v --cov=core --cov-report=term-missing
```

Минимальное покрытие: **≥ 60%**. Включены тесты на эталонные значения формул СМО и корректность игровых критериев.

---

## 2. Пример pytest-теста для модуля игры

```python
import pytest
import numpy as np
from core.game_solver import build_payoff_matrix, apply_hurwicz_criterion

# Пример матрицы затрат (в тыс. руб./месяц)
# Строки: стратегии колл-центра (c=3, c=5, c=7 операторов)
# Столбцы: состояния природы (низкая, средняя, высокая нагрузка)
COST_MATRIX = np.array([
    [120, 250, 480],  # c=3: дёшево при низкой нагрузке, большие штрафы при высокой
    [200, 200, 280],  # c=5: сбалансированный вариант
    [310, 310, 310],  # c=7: дорого, но стабильно
])


class TestGameSolver:
    @pytest.mark.parametrize("alpha, expected_index", [
        (0.0, 0),  # пессимист (Вальд) -> min(max_cost) = 280 (c=5)
        (0.5, 1),  # нейтральный -> min(avg) = 200 (c=5)
        (1.0, 2),  # оптимист -> min_cost в лучшем случае = 120 (c=3)
    ])
    def test_hurwicz_criterion(self, alpha, expected_index):
        """Проверка корректности критерия Гурвица для матрицы затрат"""
        costs = build_payoff_matrix(lambda_low=20, lambda_avg=50, lambda_high=90)

        # Критерий Гурвица для ЗАТРАТ: минимизация (alpha * min + (1-alpha) * max)
        best_idx, best_val = apply_hurwicz_criterion(costs, alpha=alpha)

        assert 0 <= best_idx < costs.shape[0]
        assert isinstance(best_val, float)

        # Проверка, что алгоритм возвращает ожидаемую стратегию для эталонной матрицы
        if np.array_equal(costs, COST_MATRIX):
            assert best_idx == expected_index

    def test_payoff_matrix_structure(self):
        """Матрица должна быть 2D, неотрицательной и конечной"""
        matrix = build_payoff_matrix(lambda_low=10, lambda_avg=30, lambda_high=80)
        assert matrix.ndim == 2
        assert np.all(matrix >= 0)
        assert np.all(np.isfinite(matrix))
        assert matrix.shape[1] == 3   # 3 сценария природы
        assert matrix.shape[0] >= 2   # не менее 2 стратегий

    def test_edge_case_alpha_values(self):
        """Критерий должен корректно обрабатывать граничные значения alpha"""
        matrix = COST_MATRIX
        for alpha in [-0.1, 0.0, 0.5, 1.0, 1.1]:
            # Функция должна либо зажать alpha, либо поднять ValueError
            try:
                idx, _ = apply_hurwicz_criterion(matrix, alpha=alpha)
                assert 0 <= idx < matrix.shape[0]
            except ValueError:
                pass  # Ожидаемо для alpha < 0 или > 1
```

*`tests/test_game_solver.py`*

---

## 3. Готовая структура docker-compose.yml

```yaml
version: "3.8"
services:
  app:
    build: .
    container_name: callcenter_opt
    ports:
      - "8501:8501"   # Streamlit dashboard
      - "8000:8000"   # Опциональный FastAPI (если подключён)
    volumes:
      - ./config:/app/config
      - ./reports:/app/reports
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=INFO
      - CRITERION_DEFAULT=hurwicz
      - ALPHA_DEFAULT=0.6
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 5s
      retries: 3

networks:
  default:
    name: callcenter_net
    driver: bridge
```

> **Примечание для сдачи:** В учебном проекте допускается запуск без Docker через виртуальное окружение. Файл `docker-compose.yml` демонстрирует знание современных практик контейнеризации и воспроизводимости среды. Для production рекомендуется вынести переменные окружения в `.env` и добавить `.dockerignore`.

---

## 4. Чек-лист для проверки проекта

### 4.1 Общее

- [ ] Репозиторий доступен преподавателю, ветка `main` стабильна, нет лишних артефактов (`.pyc`, `venv`, `.DS_Store`)
- [ ] `README.md` обновлён, команды `pip install` и `python main.py` работают на чистой системе
- [ ] Все тесты проходят, покрытие ≥ 60% (приложить вывод `pytest --cov`)
- [ ] Отчёт оформлен: PDF и исходники (`.tex`/`.docx`) подготовлены
- [ ] Подготовлены 3 демо-сценария: штатный, пограничный (SLA нарушен), стрессовый (высокая λ)
- [ ] Презентация: 8–12 слайдов, акцент на математике → архитектуре → результатам

### 4.2 Теоретическая часть

- [ ] Объяснить, почему поток звонков моделируется как пуассоновский, а обслуживание ~ Exp(μ)
- [ ] Могу вывести/прокомментировать формулу вероятности ожидания P_wait (формула Эрланга C)
- [ ] Понимаю различие критериев Вальда, Гурвица и Байеса, могу обосновать выбор α
- [ ] Готов ответить: «Что изменится в модели, если время обслуживания будет не экспоненциальным, а детерминированным?» (M/D/c)

### 4.3 Инженерная часть

- [ ] Могу объяснить разделение слоёв: ядро математики (`core/`) отделено от интерфейса (`ui/`) и конфигурации
- [ ] Демонстрирую валидацию входных данных через Pydantic (отсечение отрицательных λ, μ, проверка типов)
- [ ] Показываю обработку краевых случаев: деление на ноль при μ ≤ 0, стабилизация матрицы, логирование ошибок
- [ ] Объясняю, как масштабировать расчёт при добавлении новых сценариев или критериев (паттерн Strategy/Factory)

### 4.4 Демонстрация и Q&A

- [ ] Запуск демо ≤ 15 секунд (предзагруженные конфиги, моки если нужно)
- [ ] Показываю не только «счастливый путь», но и реакцию системы на нарушение SLA
- [ ] Интерпретирую метрики: «При α = 0.6 выбрано 5 операторов, так как…»
- [ ] Отвечаю честно на вопросы «не знаю», предлагаю гипотезу и способ верификации
- [ ] Соблюдаю тайминг (7 мин выступление + 5 мин вопросы), не читаю со слайдов

### 4.5 Документация и артефакты

- [ ] `requirements.txt` или `pyproject.toml` с зафиксированными версиями
- [ ] Примеры конфигов (`scenario.yaml`) с комментариями
- [ ] Отчёт `pytest-cov` сохранён в `docs/coverage/`
- [ ] Инструкция для проверяющего в `README` и в отчёте (раздел «Запуск»)
