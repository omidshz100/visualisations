# Data Visualization Script

A Python script that performs comprehensive data visualization on sales datasets using pandas, matplotlib, and seaborn.

## Features

- **Automatic Data Loading**: Loads data from CSV file or creates sample dataset if file is missing
- **Data Verification**: Displays dataset overview including first few rows, shape, and statistics
- **Multiple Visualizations**:
  - Monthly sales bar chart with custom styling
  - Sales vs expenses scatter plot with correlation analysis
- **Professional Styling**: Uses seaborn color palettes and custom formatting
- **Error Handling**: Graceful fallback to sample data if CSV loading fails

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas matplotlib seaborn numpy
```

## Usage

### Option 1: Using the provided sample CSV

Simply run the script - it will automatically load the included `sales_data.csv`:

```bash
python data_visualization.py
```

### Option 2: Using your own CSV file

1. Replace `sales_data.csv` with your own CSV file (keep the same name)
2. Ensure your CSV has these columns:
   - `Month` (string): Month names
   - `Sales` (numeric): Sales figures
   - `Expenses` (numeric): Expense figures
3. Run the script:

```bash
python data_visualization.py
```

### Option 3: No CSV file

If no CSV file is found, the script automatically creates and uses a sample dataset.

## Output

The script generates:

1. **Console Output**:
   - Dataset loading confirmation
   - First 5 rows of data
   - Dataset statistics and information
   - Correlation coefficient between sales and expenses

2. **Visualizations**:
   - **Monthly Sales Bar Chart**: Shows sales trends across months with value labels
   - **Sales vs Expenses Scatter Plot**: Analyzes relationship with trend line and correlation

## Sample Data Format

Your CSV file should follow this format:

```csv
Month,Sales,Expenses
January,45000,32000
February,52000,35000
March,48000,33000
...
```

## Features Breakdown

### Data Loading (`load_data()`)
- Attempts to load from `sales_data.csv`
- Falls back to sample data if file is missing or corrupted
- Provides clear feedback about data source

### Data Display (`display_data_info()`)
- Shows first 5 rows for verification
- Displays dataset shape and column information
- Provides descriptive statistics

### Monthly Sales Chart (`create_monthly_sales_chart()`)
- Bar chart with viridis color palette
- Value labels on each bar
- Formatted y-axis (shows values in thousands)
- Rotated month labels for readability

### Sales vs Expenses Analysis (`create_sales_expenses_scatter()`)
- Scatter plot with plasma color palette
- Grid lines for better readability
- Trend line showing relationship direction
- Correlation coefficient display
- Formatted axes (values in thousands)

## Customization

You can easily modify:

- **Colors**: Change `sns.color_palette()` parameters
- **Figure Size**: Modify `plt.figure(figsize=(10, 6))`
- **Data Source**: Update the `csv_file` variable in `load_data()`
- **Sample Data**: Modify the dictionary in `create_sample_data()`

## Error Handling

The script includes robust error handling:
- Missing CSV file → Uses sample data
- Corrupted CSV file → Falls back to sample data
- Missing columns → Clear error message with requirements
- Import errors → Descriptive error messages

## Dependencies

- **pandas**: Data manipulation and CSV loading
- **matplotlib**: Core plotting functionality
- **seaborn**: Enhanced styling and color palettes
- **numpy**: Numerical operations for trend line calculation
- **os**: File system operations

## License

This script is provided as-is for educational and commercial use.