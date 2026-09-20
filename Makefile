.PHONY: up down logs migrate api worker test lint demo

up:            ## start everything in docker
	docker compose up -d --build
down:
	docker compose down -v
logs:
	docker compose logs -f api worker

# ---- local dev (infra in docker, app on host) ----
infra:
	docker compose up -d db redis
migrate:
	alembic upgrade head
api:
	uvicorn app.main:app --reload --port 8000
worker:
	celery -A app.workers.celery_app:celery_app worker -Q documents --pool=threads --concurrency=4 --loglevel=INFO  # threads pool: prefork breaks on macOS spawn
test:
	pytest -q
lint:
	ruff check . && ruff format --check .
demo:
	python scripts/demo.py
eval:
	python scripts/eval.py --md eval-report.md
