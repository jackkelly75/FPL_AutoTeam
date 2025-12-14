# FPL-AutoTeam

* Add injurty
* Add desperation index to focus more on high scoring players to catch other players

I have no idea about the premier league or any football at all really. 

Import data from API and evaluate potential team points for upcoming fixtures. Will recommend trades to make maximising the points over the coming 7 weeks.

** Features**  

* Import data from FPL API and format to get game and player information
* **Note:** This is based on this years performance as was developed 12 weeks into the season. I will build some predictions for early season based on the previous year performance in the future.

**Missing**  
There are certain important features missing,





fpl-predictor/
│
├── fpl_predictor/                # Main package folder
│   ├── __init__.py               # Makes this a Python package
│   ├── fixture_transfer.py       # Functions for fixture & transfer logic
│   ├── import_data_func.py       # Functions for importing data
│   ├── review_points.py          # Functions for reviewing past weeks
│   ├── prediction.py             # Wraps logic into callable functions
│
├── scripts/
│   └── Transfers_main.py         # Example script that calls prediction functions
│
├── tests/
│   └── test_prediction.py        # Unit tests for your functions
│
├── README.md                     # Usage instructions
├── requirements.yaml / requirements.txt  # Dependencies
├── setup.py or pyproject.toml    # Packaging & installation
└── LICENSE                       # Open-source license
