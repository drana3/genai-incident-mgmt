# GenAI Incident Management
A serverless AWS system to automate incident resolution using CrewAI, FastAPI, and CDK.

## Setup
1. Install Poetry: `curl -sSL https://install.python-poetry.org | python3 -`
2. Install deps: `poetry install`
3. Run tests: `poetry run pytest tests/ -v --cov=app`
4. Run app: `poetry run python app/main.py`
5. Deploy: `cd cdk && cdk deploy`