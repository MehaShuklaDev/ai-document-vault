"""Celery application.

Settings chosen for a document pipeline where tasks are long, idempotent and
expensive:

* ``acks_late`` + ``reject_on_worker_lost`` – a crashed worker's task is redelivered.
* ``prefetch_multiplier=1`` – fair scheduling; one slow PDF does not hold 4 others hostage.
* soft/hard time limits – runaway extraction cannot wedge a worker forever.
* dedicated ``documents`` queue – lets you scale ingestion workers independently
  of any future queues (e.g. ``notifications``).
"""

from __future__ import annotations

from celery import Celery

from app.core.config import get_settings
from app.core.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)

celery_app = Celery("vault", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    task_time_limit=settings.celery_task_time_limit,
    task_default_queue="documents",
    task_routes={"app.workers.tasks.*": {"queue": "documents"}},
    result_expires=3600,
    broker_connection_retry_on_startup=True,
    worker_send_task_events=True,
    task_send_sent_event=True,
)
celery_app.autodiscover_tasks(["app.workers"])
