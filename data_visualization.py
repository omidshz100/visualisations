import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set the style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_data():
    """
    Load data from CSV file or create sample data if file doesn't exist
    """
    csv_file = 'sales_data.csv'
    
    # Check if CSV file exists
    if os.path.exists(csv_file):
        print(f"Loading data from {csv_file}...")
        try:
            df = pd.read_csv(csv_file)
            print("Data loaded successfully from CSV file!")
            return df
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            print("Creating sample dataset instead...")
            return create_sample_data()
    else:
        print(f"CSV file '{csv_file}' not found. Creating sample dataset...")
        return create_sample_data()

def create_sample_data():
    """
    Create a sample dataset with sales and expenses data
    """
    sample_data = {
        'Month': ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December'],
        'Sales': [45000, 52000, 48000, 61000, 55000, 67000,
                  72000, 69000, 58000, 63000, 71000, 78000],
        'Expenses': [32000, 35000, 33000, 42000, 38000, 45000,
                     48000, 46000, 39000, 43000, 47000, 52000]
    }
    
    df = pd.DataFrame(sample_data)
    print("Sample dataset created successfully!")
    return df

def display_data_info(df):
    """
    Display basic information about the dataset
    """
    print("\n" + "="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    print("\nDataset shape:", df.shape)
    print("\nColumn names:", list(df.columns))
    print("\nData types:")
    print(df.dtypes)
    
    print("\nBasic statistics:")
    print(df.describe())

def create_monthly_sales_chart(df):
    """
    Create a bar chart showing monthly sales
    """
    plt.figure(figsize=(10, 6))
    
    # Create bar chart with seaborn color palette
    bars = plt.bar(df['Month'], df['Sales'], color=sns.color_palette("viridis", len(df)))
    
    # Customize the chart
    plt.title('Monthly Sales', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Month', fontsize=12, fontweight='bold')
    plt.ylabel('Sales', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 500,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
    
    # Format y-axis to show values in thousands
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    plt.tight_layout()
    
    print("\nMonthly Sales bar chart prepared successfully!")

def create_sales_expenses_scatter(df):
    """
    Create a scatter plot analyzing the relationship between sales and expenses
    """
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(df['Sales'], df['Expenses'], 
                c=sns.color_palette("plasma", len(df)), 
                s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Customize the chart
    plt.title('Relationship between Sales and Expenses', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sales', fontsize=12, fontweight='bold')
    plt.ylabel('Expenses', fontsize=12, fontweight='bold')
    
    # Enable grid lines
    plt.grid(True, alpha=0.3)
    
    # Format axes to show values in thousands
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Add trend line
    z = np.polyfit(df['Sales'], df['Expenses'], 1)
    p = np.poly1d(z)
    plt.plot(df['Sales'], p(df['Sales']), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    plt.legend()
    
    # Add correlation coefficient
    correlation = df['Sales'].corr(df['Expenses'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    print(f"\nSales vs Expenses scatter plot prepared successfully!")
    print(f"Correlation coefficient: {correlation:.3f}")
    
    return correlation

def main():
    """
    Main function to execute the data visualization script
    """
    print("Data Visualization Script")
    print("========================")
    
    # Load the data
    df = load_data()
    
    # Verify required columns exist
    required_columns = ['Month', 'Sales', 'Expenses']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"\nError: Missing required columns: {missing_columns}")
        print("Please ensure your CSV file contains: Month, Sales, Expenses")
        return
    
    # Display dataset information
    display_data_info(df)
    
    # Create visualizations
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # Generate visualizations
    
    # Generate monthly sales bar chart
    print("\n1. Creating Monthly Sales Bar Chart...")
    create_monthly_sales_chart(df)
    
    # Generate sales vs expenses scatter plot
    print("\n2. Creating Sales vs Expenses Scatter Plot...")
    correlation = create_sales_expenses_scatter(df)
    
    # Display all charts at once
    print("\n3. Displaying all charts...")
    plt.show()
    
    print("\n" + "="*50)
    print("VISUALIZATION COMPLETE!")
    print("="*50)
    print("\nAll visualizations have been generated successfully.")
    print("Close the plot windows to continue or terminate the script.")

if __name__ == "__main__":
    main()