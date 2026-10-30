# NBA Shot Chart Generator

An interactive Streamlit app that draws NBA.com-style zone shot charts for any player and season, colored by how their FG% compares to league average in each zone.

## Features

- Pick a data file, season, and player from the browser
- 14 court zones matching the NBA.com/Stats layout
- Each zone shows made/attempted and FG%, colored from −10% to +10% vs league average
- Overall FG% and **rFG%** (relative FG% — how much better or worse than league average the player shoots given where they take their shots)
- Optional **shrinkage** slider that pulls low-volume zones toward league average, so a 0/1 zone doesn't show up solid red
- Option to hide labels on zones below a minimum number of attempts
- Expandable zone-by-zone breakdown table
- Works with NBA or WNBA data

## Setup

```bash
pip install streamlit pandas numpy matplotlib pyarrow
```

## Usage

The repo already includes `data/shots_nba_2020_2025.parquet`, so you can run the app straight away:

```bash
streamlit run app.py
```

### Getting other data

`prep_data.py` downloads shot data from [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data), cleans it, and saves a parquet file into `data/`:

```bash
python prep_data.py                        # NBA 2020-2025
python prep_data.py --league wnba          # WNBA 2020-2025
python prep_data.py --start 2015 --end 2020
```

Any shotdetail CSV or parquet dropped into `data/` will show up in the app's file picker.

## Files

- `app.py` — the Streamlit app (court drawing, zone stats, chart)
- `prep_data.py` — downloads and prepares shot data
- `data/` — shot data files

## Data

Shot data comes from [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data). Backcourt heaves are dropped.
