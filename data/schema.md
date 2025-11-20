# Crop Price Data Schema

## Primary Columns
- `date` (DATE): Trading date in YYYY-MM-DD format
- `state` (STRING): Indian state name
- `district` (STRING): District within state
- `market` (STRING): Market/mandi name
- `crop` (STRING): Crop type (rice, wheat, maize, pulses, cotton, sugarcane)
- `variety` (STRING): Crop variety (e.g., "PR 126", "Sharbati")

## Price Information
- `min_price` (FLOAT): Minimum price per quintal (INR)
- `max_price` (FLOAT): Maximum price per quintal (INR)
- `modal_price` (FLOAT): Most frequent price per quintal (INR)

## Supply & Weather
- `arrival_qty` (FLOAT): Quantity arrived in market (quintals)
- `acreage_estimate` (FLOAT): Estimated acreage for crop (hectares)
- `rainfall_mm` (FLOAT): Daily rainfall in mm
- `temp_c` (FLOAT): Average temperature in Celsius

## External Factors
- `govt_policy_flag` (INTEGER): 1 if government policy announcement, else 0
- `holiday_flag` (INTEGER): 1 if trading holiday, else 0