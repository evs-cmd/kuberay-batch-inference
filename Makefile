.PHONY: help deploy stop test verify job

deploy:  ## Install deps, start Ray + FastAPI, run test
	chmod +x scripts/*.sh
	bash scripts/deploy.sh

stop:  ## Stop Ray + FastAPI
	bash scripts/deploy.sh stop

verify:  ## Download + verify model (no server)
	python -m app.model_server

job:  ## Submit batch via Ray Job SDK
	ray job submit --address http://localhost:8265 --working-dir . -- python jobs/batch_job.py
