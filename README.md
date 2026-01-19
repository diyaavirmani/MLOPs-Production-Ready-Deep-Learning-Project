# MLOPs-Production-Ready-Deep-Learning-Project

## Workflows

### Development Workflow
1. Update `config.yaml`
2. Update `params.yaml`
3. Update the entity
4. Update the configuration manager in `src/config`
5. Update the components
6. Update the pipeline
7. Update the `main.py`
8. Update the `dvc.yaml`

## Project Structure
```
├── config/
│   ├── config.yaml
│   └── params.yaml
├── src/
│   └── cnnClassifier/
│       ├── components/
│       ├── config/
│       ├── entity/
│       ├── pipeline/
│       └── utils/
├── research/
├── artifacts/
└── main.py
```

## Setup

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Run Pipeline
```bash
python main.py
```
