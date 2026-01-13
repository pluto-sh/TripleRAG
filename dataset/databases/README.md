# Database Files Directory

This directory should contain your SQLite database files.

## Expected Files

- `battery_supply_chain.db` - Main SQLite database file

## Database Schema

Please refer to `config/sql_schema.md` for the complete table structure.

## Example Table Structure

```sql
CREATE TABLE EnterpriseTradeData (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    enterprise TEXT NOT NULL,
    country TEXT,
    trade_type TEXT NOT NULL CHECK(trade_type IN ('production', 'purchase')),
    material TEXT NOT NULL,
    material_quantity TEXT,
    unit_price TEXT,
    trade_date DATE
);
```

## How to Import Your Database

1. Place your SQLite database file in this directory
2. Rename it to `battery_supply_chain.db` (or update the path in `config/config.py`)
3. Ensure the table structure matches the schema in `config/sql_schema.md`

## Verification

To verify your database is correctly set up:

```bash
sqlite3 battery_supply_chain.db "SELECT COUNT(*) FROM EnterpriseTradeData;"
```

This should return the total number of records in your database.
