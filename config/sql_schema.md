**SQLite Table Schema**

Table Overview: EnterpriseTradeData

Table Fields:
- EnterpriseTradeData: id, enterprise, country, trade_type, material, material_quantity, unit_price, trade_date

Field Descriptions:
- id: INTEGER PRIMARY KEY AUTOINCREMENT
- enterprise: TEXT NOT NULL - Enterprise name
- country: TEXT - Country/Region
- trade_type: TEXT NOT NULL CHECK(trade_type IN ('production', 'purchase')) - Trade type (production/purchase)
- material: TEXT NOT NULL - Material name (e.g., Nickel, Coal, Nickel Ore, etc.)
- material_quantity: TEXT - Material quantity
- unit_price: TEXT - Unit price
- trade_date: DATE - Trade date

Data Overview:
- Total records: 11,475 entries
- Primary materials: Nickel, Coal, Nickel Ore, Nickel Pig Iron, Ferronickel, Nickel Sulphate, Stainless Steel, Battery Materials, Cobalt, Cobalt Sulphate, etc.
