from flask import Flask, render_template, request
import pandas as pd
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
DATA_FOLDER = './data'
CLEANED_FOLDER = os.path.join(DATA_FOLDER, 'cleaned')

# Ensure the necessary folders exist
os.makedirs(CLEANED_FOLDER, exist_ok=True)
def shorten_text(text, max_words=15):
    if isinstance(text, str):
        words = text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words]) + "..."
    return text
def get_valid_files(folder, required_columns):
    valid_files = []
    required_columns = [col.lower() for col in required_columns]  
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            file_path = os.path.join(folder, file)
            try:
                df = pd.read_csv(file_path, nrows=1)
                file_columns = [col.lower() for col in df.columns] 
                if all(column in file_columns for column in required_columns):
                    valid_files.append(file)
            except Exception:
                pass
    return valid_files

def get_csv_files(folder):
    """Retrieve all CSV files in the specified folder."""
    return [file for file in os.listdir(folder) if file.endswith('.csv')]

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/clean-data', methods=['GET', 'POST'])
def clean_data():
    if request.method == 'POST':
        selected_file = request.form.get('datafile')
        if selected_file:
            filepath = os.path.join(DATA_FOLDER, selected_file)
            
            # Load and clean data
            df = pd.read_csv(filepath)
            df = df.drop_duplicates()  # Remove duplicates
            df = df.dropna()  # Remove missing values
            
            # Convert columns to appropriate types (example: 'Date' to datetime)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Shorten long text columns for display only
            display_df = df.copy()
            for col in display_df.select_dtypes(include='object').columns:
                display_df[col] = display_df[col].apply(lambda x: shorten_text(x))
            
            # Save cleaned data to a new file
            cleaned_filename = f"cleaned_{selected_file}"
            cleaned_filepath = os.path.join(CLEANED_FOLDER, cleaned_filename)
            df.to_csv(cleaned_filepath, index=False)
            
            # Render the cleaned data table with Bootstrap styles
            cleaned_html = display_df.head().to_html(
                classes="table table-striped table-hover table-bordered",
                index=False
            )

            return render_template(
                'clean_data.html',
                files=get_csv_files(DATA_FOLDER),
                cleaned_data=cleaned_html,
                file_url=f"/static/data/cleaned/{cleaned_filename}"  # Adjust the path if needed
            )
    
    return render_template('clean_data.html', files=get_csv_files(DATA_FOLDER))

@app.route('/categories-analysis', methods=['GET', 'POST'])
def categories_analysis():
    required_columns = ['Category', 'Qty', 'Amount', 'Order ID']
    valid_files = get_valid_files(CLEANED_FOLDER, required_columns)

    if request.method == 'POST':
        selected_file = request.form.get('datafile')
        if selected_file:
            file_path = os.path.join(CLEANED_FOLDER, selected_file)
            df = pd.read_csv(file_path)
            
            # Ensure data types for calculation
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')

            # Group data by 'Category' and calculate metrics
            category_metrics = df.groupby('Category').agg(
                total_revenue=('Amount', 'sum'),
                total_units=('Qty', 'sum'),
                avg_price_per_unit=('Amount', lambda x: x.sum() / max(1, x.count())),
                total_orders=('Order ID', 'nunique')
            ).reset_index()

            # Sort categories by total revenue
            category_metrics = category_metrics.sort_values(by='total_revenue', ascending=False)

            return render_template(
                'categories_analysis.html',
                valid_files=valid_files,
                selected_file=selected_file,
                analysis_results=category_metrics.to_html(index=False, classes='table table-striped')
            )

    return render_template('categories_analysis.html', valid_files=valid_files)
@app.route('/sales-analysis', methods=['GET', 'POST'])
def sales_analysis():
    required_columns = ['Date', 'Amount', 'Qty']
    valid_files = get_valid_files(CLEANED_FOLDER, required_columns)

    if request.method == 'POST':
        selected_file = request.form.get('datafile')
        if selected_file:
            file_path = os.path.join(CLEANED_FOLDER, selected_file)
            df = pd.read_csv(file_path)

            # Ensure correct data types
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')

            # Filter out invalid dates
            df = df.dropna(subset=['Date'])

            # Extract month and year for aggregation
            df['YearMonth'] = df['Date'].dt.to_period('M')

            # Aggregate data monthly
            monthly_data = df.groupby('YearMonth').agg(
                total_revenue=('Amount', 'sum'),
                total_units=('Qty', 'sum'),
                avg_price_per_unit=('Amount', lambda x: x.sum() / max(1, x.count()))
            ).reset_index()

            # Convert YearMonth back to datetime for plotting
            monthly_data['YearMonth'] = monthly_data['YearMonth'].dt.to_timestamp()

            # Plot the data
            plot_path = './static/sales_analysis.png'
            plt.figure(figsize=(10, 6))
            plt.plot(monthly_data['YearMonth'], monthly_data['total_revenue'], label='Total Revenue', marker='o')
            plt.plot(monthly_data['YearMonth'], monthly_data['total_units'], label='Total Units', marker='o')
            plt.plot(monthly_data['YearMonth'], monthly_data['avg_price_per_unit'], label='Avg Price Per Unit', marker='o')
            plt.legend()
            plt.title('Monthly Sales Analysis')
            plt.xlabel('Month')
            plt.ylabel('Values')
            plt.grid()
            plt.savefig(plot_path)
            plt.close()

            return render_template(
                'sales_analysis.html',
                valid_files=valid_files,
                selected_file=selected_file,
                plot_url=plot_path
            )

    return render_template('sales_analysis.html', valid_files=valid_files)

@app.route('/fulfilment-analysis', methods=['GET', 'POST'])
def fulfilment_analysis():
    required_columns = ['Fulfilment', 'Amount', 'Qty']
    valid_files = get_valid_files(CLEANED_FOLDER, required_columns)

    if request.method == 'POST':
        selected_file = request.form.get('datafile')
        if selected_file:
            file_path = os.path.join(CLEANED_FOLDER, selected_file)
            df = pd.read_csv(file_path)

            # Ensure correct data types
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')

            # Group data by 'Fulfilment' and calculate metrics
            fulfilment_metrics = df.groupby('Fulfilment').agg(
                total_revenue=('Amount', 'sum'),
                total_units=('Qty', 'sum')
            ).reset_index()

            # Sort by total revenue
            fulfilment_metrics = fulfilment_metrics.sort_values(by='total_revenue', ascending=False)

            return render_template(
                'fulfilment_analysis.html',
                valid_files=valid_files,
                selected_file=selected_file,
                analysis_results=fulfilment_metrics.to_html(index=False, classes='table table-striped')
            )

    return render_template('fulfilment_analysis.html', valid_files=valid_files)

@app.route('/top-products', methods=['GET', 'POST'])
def top_products():
    required_columns = ['SKU', 'ASIN', 'Amount']
    valid_files = get_valid_files(CLEANED_FOLDER, required_columns)

    if request.method == 'POST':
        selected_file = request.form.get('datafile')
        if selected_file:
            file_path = os.path.join(CLEANED_FOLDER, selected_file)
            df = pd.read_csv(file_path)

            # Ensure correct data types
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

            # Calculate total revenue for each product
            product_revenue = df.groupby(['SKU', 'ASIN']).agg(
                total_revenue=('Amount', 'sum')
            ).reset_index()

            # Sort by revenue and get top 5 products
            top_products = product_revenue.sort_values(by='total_revenue', ascending=False).head(5)

            return render_template(
                'top_products.html',
                valid_files=valid_files,
                selected_file=selected_file,
                top_products=top_products.to_html(index=False, classes='table table-striped')
            )

    return render_template('top_products.html', valid_files=valid_files)
@app.route('/b2b-analysis', methods=['GET', 'POST'])
def b2b_analysis():
    required_columns = ['B2B', 'Amount', 'Qty']
    valid_files = get_valid_files(CLEANED_FOLDER, required_columns)

    if request.method == 'POST':
        selected_file = request.form.get('datafile')
        if selected_file:
            file_path = os.path.join(CLEANED_FOLDER, selected_file)
            df = pd.read_csv(file_path)

            # Ensure correct data types
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')

            # Split data into B2B and non-B2B
            b2b_data = df[df['B2B'] == True]
            non_b2b_data = df[df['B2B'] == False]

            # Calculate metrics
            metrics = {
                'B2B': {
                    'total_revenue': b2b_data['Amount'].sum(),
                    'total_units': b2b_data['Qty'].sum(),
                    'avg_sale_value': b2b_data['Amount'].mean()
                },
                'Non-B2B': {
                    'total_revenue': non_b2b_data['Amount'].sum(),
                    'total_units': non_b2b_data['Qty'].sum(),
                    'avg_sale_value': non_b2b_data['Amount'].mean()
                }
            }

            # Convert to DataFrame for display
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index').reset_index()
            metrics_df.columns = ['Type', 'Total Revenue', 'Total Units', 'Average Sale Value']

            return render_template(
                'b2b_analysis.html',
                valid_files=valid_files,
                selected_file=selected_file,
                analysis_results=metrics_df.to_html(index=False, classes='table table-striped')
            )

    return render_template('b2b_analysis.html', valid_files=valid_files)

@app.route('/size-analysis', methods=['GET', 'POST'])
def size_analysis():
    required_columns = ['Size', 'Amount', 'Qty']
    valid_files = get_valid_files(CLEANED_FOLDER, required_columns)

    if request.method == 'POST':
        selected_file = request.form.get('datafile')
        if selected_file:
            file_path = os.path.join(CLEANED_FOLDER, selected_file)
            df = pd.read_csv(file_path)

            # Ensure correct data types
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')

            # Group data by Size and calculate metrics
            size_metrics = df.groupby('Size').agg(
                total_revenue=('Amount', 'sum'),
                total_units=('Qty', 'sum')
            ).reset_index()

            # Sort by revenue
            size_metrics = size_metrics.sort_values(by='total_revenue', ascending=False)

            return render_template(
                'size_analysis.html',
                valid_files=valid_files,
                selected_file=selected_file,
                analysis_results=size_metrics.to_html(index=False, classes='table table-striped')
            )

    return render_template('size_analysis.html', valid_files=valid_files)
@app.route('/courier-analysis', methods=['GET', 'POST'])
def courier_analysis():
    required_columns = ['Courier Status', 'Amount']
    valid_files = get_valid_files(CLEANED_FOLDER, required_columns)

    if request.method == 'POST':
        selected_file = request.form.get('datafile')
        if selected_file:
            file_path = os.path.join(CLEANED_FOLDER, selected_file)
            df = pd.read_csv(file_path)

            # Ensure correct data types
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

            # Group data by Courier Status
            courier_metrics = df.groupby('Courier Status').agg(
                total_revenue=('Amount', 'sum'),
                order_count=('Amount', 'count')
            ).reset_index()

            # Calculate delay impact (assuming "Delivered" means successful)
            delivered_revenue = courier_metrics.loc[
                courier_metrics['Courier Status'] == 'Delivered', 'total_revenue'
            ].sum()

            delayed_revenue = courier_metrics.loc[
                courier_metrics['Courier Status'] != 'Delivered', 'total_revenue'
            ].sum()

            # Recommendations and insights
            recommendations = {
                "delivered_revenue": delivered_revenue,
                "delayed_revenue": delayed_revenue,
                "delay_percentage": (delayed_revenue / (delayed_revenue + delivered_revenue)) * 100 if (delayed_revenue + delivered_revenue) > 0 else 0
            }

            return render_template(
                'courier_analysis.html',
                valid_files=valid_files,
                selected_file=selected_file,
                courier_results=courier_metrics.to_html(index=False, classes='table table-striped'),
                recommendations=recommendations
            )

    return render_template('courier_analysis.html', valid_files=valid_files)

@app.route('/currency-analysis', methods=['GET', 'POST'])
def currency_analysis():
    required_columns = ['Currency', 'Amount']
    valid_files = get_valid_files(CLEANED_FOLDER, required_columns)

    if request.method == 'POST':
        selected_file = request.form.get('datafile')
        if selected_file:
            file_path = os.path.join(CLEANED_FOLDER, selected_file)
            df = pd.read_csv(file_path)

            # Ensure correct data types
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

            # Group data by Currency and calculate metrics
            currency_metrics = df.groupby('currency').agg(
                total_revenue=('Amount', 'sum'),
                order_count=('Amount', 'count')
            ).reset_index()

            # Sort by total revenue
            currency_metrics = currency_metrics.sort_values(by='total_revenue', ascending=False)

            return render_template(
                'currency_analysis.html',
                valid_files=valid_files,
                selected_file=selected_file,
                analysis_results=currency_metrics.to_html(index=False, classes='table table-striped')
            )

    return render_template('currency_analysis.html', valid_files=valid_files)
@app.route('/business-insights')
def business_insights():
    # Categories Analysis
    categories_required_columns = ['Category', 'Qty', 'Amount', 'Order ID']
    categories_files = get_valid_files(CLEANED_FOLDER, categories_required_columns)
    if categories_files:
        file_path = os.path.join(CLEANED_FOLDER, categories_files[0])
        df = pd.read_csv(file_path)
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
        categories_metrics = df.groupby('Category').agg(
            total_revenue=('Amount', 'sum'),
            total_units=('Qty', 'sum')
        ).reset_index()
        top_category = categories_metrics.sort_values(by='total_revenue', ascending=False).iloc[0]
        categories_analysis = f"The top category is {top_category['Category']} with total revenue of {top_category['total_revenue']:.2f}."
    else:
        categories_analysis = "No valid data for category analysis."

    # Sales Trends Analysis
    sales_required_columns = ['Date', 'Amount', 'Qty']
    sales_files = get_valid_files(CLEANED_FOLDER, sales_required_columns)
    if sales_files:
        file_path = os.path.join(CLEANED_FOLDER, sales_files[0])
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['YearMonth'] = df['Date'].dt.to_period('M')
        sales_metrics = df.groupby('YearMonth').agg(
            total_revenue=('Amount', 'sum'),
            total_units=('Qty', 'sum')
        ).reset_index()
        peak_month = sales_metrics.sort_values(by='total_revenue', ascending=False).iloc[0]
        sales_trends = f"The peak sales month is {peak_month['YearMonth']} with total revenue of {peak_month['total_revenue']:.2f}."
    else:
        sales_trends = "No valid data for sales trend analysis."

    # Fulfilment Analysis
    fulfilment_required_columns = ['Fulfilment', 'Amount', 'Qty']
    fulfilment_files = get_valid_files(CLEANED_FOLDER, fulfilment_required_columns)
    if fulfilment_files:
        file_path = os.path.join(CLEANED_FOLDER, fulfilment_files[0])
        df = pd.read_csv(file_path)
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
        fulfilment_metrics = df.groupby('Fulfilment').agg(
            total_revenue=('Amount', 'sum')
        ).reset_index()
        top_fulfilment = fulfilment_metrics.sort_values(by='total_revenue', ascending=False).iloc[0]
        fulfilment_analysis = f"The best fulfilment method is {top_fulfilment['Fulfilment']} with total revenue of {top_fulfilment['total_revenue']:.2f}."
    else:
        fulfilment_analysis = "No valid data for fulfilment analysis."

    # Combine insights
    insights = {
        "categories_analysis": categories_analysis,
        "sales_trends": sales_trends,
        "fulfilment_analysis": fulfilment_analysis
    }

    # Recommendations based on insights
    recommendations = [
        "Focus marketing efforts on top-performing categories and products.",
        "Leverage seasonal trends by planning promotions in peak months.",
        "Expand best-performing fulfilment methods to improve delivery times and customer satisfaction."
    ]

    return render_template('business_insights.html', insights=insights, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
