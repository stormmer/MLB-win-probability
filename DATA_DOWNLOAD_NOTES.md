# Data Download Notes

## Current Status

The data download script successfully downloads:
- ✅ **Team batting statistics** (working)
- ✅ **Team pitching statistics** (working)
- ⚠️ **Schedule data** (may fail due to rate limiting)

## Schedule Data Issues

The `schedule_and_record` function from pybaseball may fail due to:
1. **Rate limiting** from baseball-reference.com
2. **Website blocking** automated requests
3. **API changes** in pybaseball

### Solutions

**Option 1: Retry with delays**
The script now includes 2-second delays between requests. If it still fails, try:
- Running the download again later
- Reducing the number of teams/seasons
- Using a VPN or different network

**Option 2: Manual download**
You can manually download schedule data from:
- https://www.baseball-reference.com/leagues/MLB/
- Save as CSV files in `data/raw/` with naming: `schedule_YYYY.csv`

**Option 3: Use alternative data source**
Consider using:
- MLB Stats API (requires API key)
- Retrosheet data
- Other baseball data sources

## Proceeding Without Schedule Data

If schedule data cannot be downloaded, you have a few options:

1. **Use sample/demo data**: Create a minimal dataset for testing
2. **Skip schedule features**: Build features only from team stats (limited functionality)
3. **Wait and retry**: Rate limits may reset after some time

## Next Steps

Even without schedule data, you can:
1. ✅ Use the team stats that were successfully downloaded
2. ✅ Test the feature engineering pipeline with sample data
3. ✅ Test the model training with synthetic game data

To proceed with limited functionality:
```powershell
# The dataset builder will warn about missing schedule data
# but you can still test other parts of the pipeline
.\run.ps1 build
```

