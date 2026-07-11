# stock-lens — pipeline code

NYSE stock direction classification, sequence models, out-of-sample
evaluation, and a natural-language chatbot. Each file owns one
responsibility; `main.py` routes stages from the command line.

## File map

| File | Responsibility |
|---|---|
| `config.py` | paths, constants, feature lists, thresholds |
| `helpers.py` | logging, plot saving, section banners |
| `data_loading.py` | stage 1 — download and inspect raw NYSE data |
| `features.py` | stage 2 — indicators, fundamentals, labels, split |
| `enhanced_features.py` | stage 2b — lags, returns, relative features, horizons |
| `prep.py` | shared preprocessing + leak-safe held-out test evaluation |
| `classification.py` | stage 3 — random forest and xgboost baselines |
| `experiments.py` | stage 3b — strategy x horizon x granularity sweep |
| `sequence_models.py` | stage 4 — LSTM/GRU/TCN/CNN/transformer + model saving |
| `nlp_parser.py` | natural-language question parsing (ticker, date, model) |
| `predictors.py` | loading saved models, predicting, formatting answers |
| `chatbot.py` | stage 5 — NLP chatbot (CLI and Gradio) |
| `oos_evaluation.py` | stage 6 — yfinance 2017-2025 regime evaluation |
| `main.py` | command-line entry point |

## Setup

```bash
pip install -r requirements.txt

export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
export STOCK_LENS_BASE=/workspace/stock-lens-data   # where data/models/plots live
export ANTHROPIC_API_KEY=your_key                   # optional, adds LLM explanations
```

All outputs land under `$STOCK_LENS_BASE`:

```
stock-lens-data/
├── data/            downloaded csvs + master/train/val/test
├── models/          saved model files (.pkl, .pt)
└── observations/    plots, results csvs, run_log.txt
```

## Running

Stages depend on earlier outputs, so run in order the first time:

```bash
python main.py --stage 1     # download + inspect (~2 min)
python main.py --stage 2     # features + labels (~5 min)
python main.py --stage 2b    # enhanced features (~5 min)
python main.py --stage 3     # baselines, saves rf model (~15 min)
python main.py --stage 3b    # full experiment sweep (long — optional)
python main.py --stage 4     # sequence models, saves winner (GPU, ~30-60 min)
python main.py --stage 6     # out-of-sample eval on yfinance 2017-2025
```

## Chatbot

Ask questions in plain English — mention a ticker or company name,
optionally a date, optionally a model ("random forest" or "cnn"):

```bash
python main.py --stage 5 --chat-mode cli
# you> Will Apple go up tomorrow?
# you> predict MSFT on 2016-06-15 with random forest

python main.py --stage 5 --chat-mode gradio   # web UI with share link
```

Dataset covers 2010-2016, so "today"/"tomorrow" resolve to the latest
available trading day. Setting `ANTHROPIC_API_KEY` adds a plain-English
explanation to every answer; without it the chatbot still works.

## Notes

- Model selection always uses validation only; the test set is scored once.
- `STOCK_LENS_OOS_TICKERS=100` limits stage 6 to the first N tickers.
- Long stages survive disconnects with `nohup python main.py --stage 4 > run.log 2>&1 &`.
- Outputs are educational only and are not financial advice.
