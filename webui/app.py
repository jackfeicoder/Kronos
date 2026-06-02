import os
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
warnings.filterwarnings('ignore')

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

app = Flask(__name__)
CORS(app)

# Global variables to store models
tokenizer = None
model = None
predictor = None

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

def load_data_files():
    """Scan data directory and return available data files"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.feather')):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                })
    
    return data_files

def load_data_file(file_path):
    """Load data file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format"
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            # If column name is 'date', rename it to 'timestamps'
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            # If no timestamp column exists, create one
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns are numeric type
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume column (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Process amount column (optional, but not used for prediction)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows containing NaN values
        df = df.dropna()
        
        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0, stock_name=None, stock_code=None):
    """Create prediction chart"""
    # Use specified historical data start position, not always from the beginning of df
    if historical_start_idx + lookback + pred_len <= len(df):
        # Display lookback historical points + pred_len prediction points starting from specified position
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
        prediction_range = range(historical_start_idx+lookback, historical_start_idx+lookback+pred_len)
    else:
        # If data is insufficient, adjust to maximum available range
        available_lookback = min(lookback, len(df) - historical_start_idx)
        available_pred_len = min(pred_len, max(0, len(df) - historical_start_idx - available_lookback))
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
        prediction_range = range(historical_start_idx+available_lookback, historical_start_idx+available_lookback+available_pred_len)
    
    # Create chart
    fig = go.Figure()
    
    # Add historical data (candlestick chart)
    historical_x = historical_df['timestamps'].tolist() if 'timestamps' in historical_df.columns else historical_df.index.tolist()
    fig.add_trace(go.Candlestick(
        x=historical_x,
        open=historical_df['open'].tolist(),
        high=historical_df['high'].tolist(),
        low=historical_df['low'].tolist(),
        close=historical_df['close'].tolist(),
        name=f'历史数据 ({len(historical_df)}个交易日)',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    # 历史数据收盘价小字注记 (使用浅灰色细字体，避免全局缩小时重叠太乱，放大后非常清晰)
    fig.add_trace(go.Scatter(
        x=historical_x,
        y=historical_df['close'].tolist(),
        mode='text',
        text=[f"{c:.2f}" for c in historical_df['close'].tolist()],
        textposition='top center',
        textfont=dict(size=8, color='rgba(120, 120, 120, 0.7)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        # 直接使用 pred_df 已经生成好的 index 作为时间轴，确保交易日日历的连续性与准确性
        pred_timestamps = pred_df.index
        pred_x = pred_timestamps.tolist() if hasattr(pred_timestamps, 'tolist') else list(pred_timestamps)
        
        fig.add_trace(go.Candlestick(
            x=pred_x,
            open=pred_df['open'].tolist(),
            high=pred_df['high'].tolist(),
            low=pred_df['low'].tolist(),
            close=pred_df['close'].tolist(),
            name=f'预测走势 ({len(pred_df)}个交易日)',
            increasing_line_color='#66BB6A',
            decreasing_line_color='#FF7043'
        ))
        # 预测趋势收盘价小字注记 (使用稍醒目的深绿色粗体字)
        fig.add_trace(go.Scatter(
            x=pred_x,
            y=pred_df['close'].tolist(),
            mode='text',
            text=[f"{c:.2f}" for c in pred_df['close'].tolist()],
            textposition='top center',
            textfont=dict(size=9, color='#1B5E20', family='Arial-Bold'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add actual data for comparison (if exists)
    if actual_df is not None and len(actual_df) > 0:
        # 实际对照走势直接使用其自带的真实时间戳
        actual_timestamps = actual_df['timestamps'] if 'timestamps' in actual_df.columns else actual_df.index
        actual_x = actual_timestamps.tolist() if hasattr(actual_timestamps, 'tolist') else list(actual_timestamps)
        
        fig.add_trace(go.Candlestick(
            x=actual_x,
            open=actual_df['open'].tolist(),
            high=actual_df['high'].tolist(),
            low=actual_df['low'].tolist(),
            close=actual_df['close'].tolist(),
            name=f'实际走势 ({len(actual_df)}个交易日)',
            increasing_line_color='#FF9800',
            decreasing_line_color='#F44336'
        ))
        # 实际走势收盘价小字注记 (使用深橙色粗体字以作区分)
        fig.add_trace(go.Scatter(
            x=actual_x,
            y=actual_df['close'].tolist(),
            mode='text',
            text=[f"{c:.2f}" for c in actual_df['close'].tolist()],
            textposition='top center',
            textfont=dict(size=9, color='#E65100', family='Arial-Bold'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Update layout
    # 获取年份范围以便放在图表标题中
    start_year = ""
    end_year = ""
    if 'timestamps' in historical_df.columns and len(historical_df) > 0:
        start_year = pd.to_datetime(historical_df['timestamps'].iloc[0]).strftime('%Y年')
    if 'pred_timestamps' in locals() and len(pred_timestamps) > 0:
        end_year = pd.to_datetime(pred_timestamps[-1]).strftime('%Y年')
    elif 'timestamps' in historical_df.columns and len(historical_df) > 0:
        end_year = pd.to_datetime(historical_df['timestamps'].iloc[-1]).strftime('%Y年')

    year_range = f" {start_year}-{end_year}" if start_year and end_year else ""
    if start_year and end_year and start_year == end_year:
        year_range = f" {start_year}"

    if stock_name and stock_code:
        if str(stock_name).strip() != str(stock_code).strip():
            chart_title = f'{stock_name} ({stock_code}){year_range} - Kronos金融走势预测结果'
        else:
            chart_title = f'{stock_code}{year_range} - Kronos金融走势预测结果'
    elif stock_name:
        chart_title = f'{stock_name}{year_range} - Kronos金融走势预测结果'
    else:
        chart_title = f'Kronos金融走势预测结果{year_range}'
        
    chart_title += f' (历史:{len(historical_df)}天 + 预测:{len(pred_df)}天)'
    if actual_df is not None and len(actual_df) > 0:
        chart_title += f' (vs {len(actual_df)}天真实走势对照)'
        
    fig.update_layout(
        title=chart_title,
        xaxis_title='时间',
        yaxis_title='价格',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    # Ensure x-axis time continuity
    if 'timestamps' in historical_df.columns:
        # Get all timestamps and sort them
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=True,
                type='date',
                tickformat='%m月%d日'
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        # Return data information
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'Successfully loaded data, total {len(df)} rows'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform prediction"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        
        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        # Load data
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        if len(df) < lookback:
            return jsonify({'error': f'Insufficient data length, need at least {lookback} rows'}), 400
        
        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                # Use real Kronos model
                # Only use necessary columns: OHLCV, excluding amount
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                # Process time period selection
                start_date = data.get('start_date')
                
                if start_date:
                    # Custom time period - fix logic: use data within selected window
                    start_dt = pd.to_datetime(start_date)
                    
                    # Find data after start time
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]
                    
                    # Ensure sufficient data: lookback + pred_len
                    if len(time_range_df) < lookback + pred_len:
                        return jsonify({'error': f'Insufficient data from start time {start_dt.strftime("%Y-%m-%d %H:%M")}, need at least {lookback + pred_len} data points, currently only {len(time_range_df)} available'}), 400
                    
                    # Use first lookback data points within selected window for prediction
                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                    
                    # Use last pred_len data points within selected window as actual values
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
                    
                    # Calculate actual time period length
                    start_timestamp = time_range_df['timestamps'].iloc[0]
                    end_timestamp = time_range_df['timestamps'].iloc[lookback+pred_len-1]
                    time_span = end_timestamp - start_timestamp
                    
                    prediction_type = f"Kronos model prediction (within selected window: first {lookback} data points for prediction, last {pred_len} data points for comparison, time span: {time_span})"
                else:
                    # Use latest data
                    x_df = df.iloc[:lookback][required_cols]
                    x_timestamp = df.iloc[:lookback]['timestamps']
                    y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
                    prediction_type = "Kronos model prediction (latest data)"
                
                # Ensure timestamps are Series format, not DatetimeIndex, to avoid .dt attribute error in Kronos model
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count
                )
                
            except Exception as e:
                return jsonify({'error': f'Kronos model prediction failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400
        
        # Prepare actual data for comparison (if exists)
        actual_data = []
        actual_df = None
        
        if start_date:  # Custom time period
            # Fix logic: use data within selected window
            # Prediction uses first 400 data points within selected window
            # Actual data should be last 120 data points within selected window
            start_dt = pd.to_datetime(start_date)
            
            # Find data starting from start_date
            mask = df['timestamps'] >= start_dt
            time_range_df = df[mask]
            
            if len(time_range_df) >= lookback + pred_len:
                # Get last 120 data points within selected window as actual values
                actual_df = time_range_df.iloc[lookback:lookback+pred_len]
                
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        else:  # Latest data
            # Prediction uses first 400 data points
            # Actual data should be 120 data points after first 400 data points
            if len(df) >= lookback + pred_len:
                actual_df = df.iloc[lookback:lookback+pred_len]
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        
        # Create chart - pass historical data start position
        if start_date:
            # Custom time period: find starting position of historical data in original df
            start_dt = pd.to_datetime(start_date)
            mask = df['timestamps'] >= start_dt
            historical_start_idx = df[mask].index[0] if len(df[mask]) > 0 else 0
        else:
            # Latest data: start from beginning
            historical_start_idx = 0
        
        # 提取文件名作为图表的展示名称
        file_name = os.path.basename(file_path) if file_path else None
        chart_json = create_prediction_chart(
            df=df,
            pred_df=pred_df,
            lookback=lookback,
            pred_len=pred_len,
            actual_df=actual_df,
            historical_start_idx=historical_start_idx,
            stock_name=file_name
        )
        
        # Prepare prediction result data - fix timestamp calculation logic
        if 'timestamps' in df.columns:
            if start_date:
                # Custom time period: use selected window data to calculate timestamps
                start_dt = pd.to_datetime(start_date)
                mask = df['timestamps'] >= start_dt
                time_range_df = df[mask]
                
                if len(time_range_df) >= lookback:
                    # Calculate prediction timestamps starting from last time point of selected window
                    last_timestamp = time_range_df['timestamps'].iloc[lookback-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                    future_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
                else:
                    future_timestamps = []
            else:
                # Latest data: calculate from last time point of entire data file
                last_timestamp = df['timestamps'].iloc[-1]
                time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                future_timestamps = pd.date_range(
                    start=last_timestamp + time_diff,
                    periods=pred_len,
                    freq=time_diff
                )
        else:
            future_timestamps = range(len(df), len(df) + pred_len)
        
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': future_timestamps[i].isoformat() if i < len(future_timestamps) else f"T{i}",
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        # Save prediction results to file
        try:
            save_prediction_results(
                file_path=file_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'start_date': start_date if start_date else 'latest'
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")
        
        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else '')
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

def fetch_stock_data_from_pool(stock_code, start_date_str, end_date_str):
    """
    智能多源数据获取接口池，按优先级依次尝试获取数据。
    """
    import akshare as ak
    import requests
    import json
    import random
    import time
    import pandas as pd

    # 1. 尝试使用 Akshare SDK（智能分流）
    is_fund = stock_code.startswith(('1', '5'))
    df = None

    print(f"--- [API Pool] Fetching {stock_code} (IsFund: {is_fund}) ---")

    # 第一层：Akshare SDK
    try:
        if is_fund:
            print("Layer 1: Trying ak.fund_etf_hist_em...")
            df = ak.fund_etf_hist_em(symbol=stock_code, period="daily", start_date=start_date_str, end_date=end_date_str, adjust="qfq")
            if df is not None and not df.empty:
                df.rename(columns={
                    "日期": "timestamps",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount"
                }, inplace=True)
                print("Layer 1 (ak.fund_etf_hist_em) succeeded.")
        else:
            print("Layer 1: Trying ak.stock_zh_a_hist...")
            df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date_str, end_date=end_date_str, adjust="qfq")
            if df is not None and not df.empty:
                df.rename(columns={
                    "日期": "timestamps",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount"
                }, inplace=True)
                print("Layer 1 (ak.stock_zh_a_hist) succeeded.")
    except Exception as e:
        print(f"Layer 1 (Akshare preferred) failed: {e}")

    # 如果首选失败，尝试备用 Akshare 接口
    if df is None or df.empty:
        try:
            if is_fund:
                print("Layer 1 Fallback: Trying ak.stock_zh_a_hist...")
                df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date_str, end_date=end_date_str, adjust="qfq")
            else:
                print("Layer 1 Fallback: Trying ak.fund_etf_hist_em...")
                df = ak.fund_etf_hist_em(symbol=stock_code, period="daily", start_date=start_date_str, end_date=end_date_str, adjust="qfq")
            
            if df is not None and not df.empty:
                df.rename(columns={
                    "日期": "timestamps",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount"
                }, inplace=True)
                print("Layer 1 Fallback succeeded.")
        except Exception as e:
            print(f"Layer 1 Fallback failed: {e}")

    # 第二层：东方财富直连 HTTP 接口 (双市场自动轮询)
    if df is None or df.empty:
        print("Layer 2: Trying Eastmoney direct HTTP...")
        guess_market = '0' if stock_code.startswith(('0', '1', '2', '3')) else '1'
        fallback_market = '1' if guess_market == '0' else '0'

        for mkt in [guess_market, fallback_market]:
            try:
                print(f"Trying Eastmoney for market {mkt}...")
                secid = f"{mkt}.{stock_code}"
                url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
                params = {
                    'secid': secid,
                    'fields1': 'f1,f2,f3,f4,f5,f6',
                    'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                    'klt': '101',  # 日线
                    'fqt': '1',    # 前复权
                    'beg': start_date_str,
                    'end': end_date_str,
                    'lmt': '10000',
                    'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
                    'cb': f'jQuery{random.randint(1000000, 9999999)}_{int(time.time()*1000)}'
                }
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
                    'Referer': 'https://quote.eastmoney.com/',
                    'Accept': '*/*',
                }

                # 强制绕过代理
                import os
                old_no_proxy = os.environ.get("no_proxy")
                os.environ["no_proxy"] = "*"
                
                try:
                    response = requests.get(url, params=params, headers=headers, proxies={"http": None, "https": None}, timeout=8)
                finally:
                    if old_no_proxy is not None:
                        os.environ["no_proxy"] = old_no_proxy
                    else:
                        os.environ.pop("no_proxy", None)

                if response.status_code == 200:
                    text = response.text
                    if text.startswith('/**/'):
                        text = text[4:]
                    start_idx = text.find('(')
                    end_idx = text.rfind(')')
                    if start_idx != -1 and end_idx != -1:
                        json_str = text[start_idx + 1:end_idx]
                        resp_data = json.loads(json_str)
                    else:
                        resp_data = json.loads(text)
                        
                    if resp_data and resp_data.get('data') is not None:
                        klines = resp_data['data'].get('klines', [])
                        if klines:
                            stock_data = []
                            for kline in klines:
                                items = kline.split(',')
                                if len(items) >= 6:
                                    stock_data.append({
                                        'timestamps': items[0],
                                        'open': float(items[1]),
                                        'close': float(items[2]),
                                        'high': float(items[3]),
                                        'low': float(items[4]),
                                        'volume': float(items[5]),
                                        'amount': float(items[6]) if len(items) > 6 else 0,
                                    })
                            if stock_data:
                                df = pd.DataFrame(stock_data)
                                print(f"Layer 2 succeeded (market {mkt}).")
                                break
            except Exception as fe:
                print(f"Layer 2 failed for market {mkt}: {fe}")

    # 第三层：腾讯财经直连 HTTP 接口 (双市场自动轮询)
    if df is None or df.empty:
        print("Layer 3: Trying Tencent Finance direct HTTP...")
        guess_market = 'sz' if stock_code.startswith(('0', '1', '2', '3')) else 'sh'
        fallback_market = 'sh' if guess_market == 'sz' else 'sz'

        for mkt in [guess_market, fallback_market]:
            try:
                print(f"Trying Tencent for market {mkt}...")
                url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
                params = {
                    'param': f"{mkt}{stock_code},day,,,800,qfq"
                }
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
                    'Accept': '*/*',
                }

                # 强制绕过代理
                import os
                old_no_proxy = os.environ.get("no_proxy")
                os.environ["no_proxy"] = "*"
                
                try:
                    response = requests.get(url, params=params, headers=headers, proxies={"http": None, "https": None}, timeout=8)
                finally:
                    if old_no_proxy is not None:
                        os.environ["no_proxy"] = old_no_proxy
                    else:
                        os.environ.pop("no_proxy", None)

                if response.status_code == 200:
                    data = response.json()
                    symbol_key = f"{mkt}{stock_code}"
                    if 'data' in data and symbol_key in data['data']:
                        day_data = data['data'][symbol_key].get('day', [])
                        if day_data:
                            stock_data = []
                            for item in day_data:
                                if len(item) >= 6:
                                    stock_data.append({
                                        'timestamps': item[0],
                                        'open': float(item[1]),
                                        'close': float(item[2]),
                                        'high': float(item[3]),
                                        'low': float(item[4]),
                                        'volume': float(item[5]) * 100,
                                    })
                            if stock_data:
                                df = pd.DataFrame(stock_data)
                                print(f"Layer 3 succeeded (market {mkt}).")
                                break
            except Exception as te:
                print(f"Layer 3 failed for market {mkt}: {te}")

    # 4. 数据统一格式处理
    if df is not None and not df.empty:
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        df = df.sort_values("timestamps").reset_index(drop=True)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])

    return df

@app.route('/api/predict-online', methods=['POST'])
def predict_online():
    """Perform online real-time prediction using akshare data fetched in memory"""
    global predictor
    try:
        if not MODEL_AVAILABLE or predictor is None:
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400

        data = request.get_json()
        stock_code = data.get('stock_code')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 22))
        
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))

        if not stock_code:
            return jsonify({'error': 'Stock code cannot be empty'}), 400
        
        # Clean stock code (ensure 6 digits)
        stock_code = str(stock_code).strip()
        if len(stock_code) != 6 or not stock_code.isdigit():
            return jsonify({'error': 'Invalid stock code, must be 6 digits'}), 400

        print(f"Fetching online data for stock {stock_code}...")

        # Calculate date range
        end_dt = datetime.datetime.now()
        start_dt = end_dt - datetime.timedelta(days=900)
        start_date_str = start_dt.strftime("%Y%m%d")
        end_date_str = end_dt.strftime("%Y%m%d")

        # 从接口池获取数据
        df = fetch_stock_data_from_pool(stock_code, start_date_str, end_date_str)

        if df is None or df.empty:
            return jsonify({'error': f'Failed to fetch data for stock code {stock_code}. Please check your internet connection or try again later.'}), 400

        if len(df) < lookback:
            return jsonify({'error': f'Insufficient historical data. Stock {stock_code} only has {len(df)} trading days in the last 2.5 years, but lookback window requires {lookback} days.'}), 400

        # We take the latest lookback (400) trading days as input
        x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume"]].copy()
        x_timestamp = df.iloc[-lookback:]["timestamps"].reset_index(drop=True)

        # Generate future prediction timestamps
        # Using pd.bdate_range is a good proxy for A-share business days
        last_date = x_timestamp.iloc[-1]
        y_timestamp = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_len)
        y_timestamp = pd.Series(y_timestamp, name='timestamps')

        print(f"Running prediction: lookback={lookback}, pred_len={pred_len}, temperature={temperature}")
        
        # Run prediction
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count
        )

        # Fetch stock name for reporting purposes if available
        stock_name = stock_code
        try:
            # 1. 优先尝试从腾讯财经极简行情接口秒级获取名称
            guess_market = 'sh' if stock_code.startswith(('5', '6', '9')) else ('sz' if stock_code.startswith(('0', '1', '2', '3')) else 'bj')
            url = f"http://qt.gtimg.cn/q=s_{guess_market}{stock_code}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
            }
            import requests
            import os
            old_no_proxy = os.environ.get("no_proxy")
            os.environ["no_proxy"] = "*"
            try:
                response = requests.get(url, headers=headers, proxies={"http": None, "https": None}, timeout=3)
                if response.status_code == 200:
                    # 返回结果例如：v_s_sh512480="1~沪深300金融ETF~512480~2.062~0.010~0.49~1687445~34842~~45.92~GP-A";
                    text = response.text
                    parts = text.split('~')
                    if len(parts) > 1 and parts[1].strip():
                        stock_name = parts[1].strip()
                        print(f"Successfully fetched name for {stock_code} via Tencent API: {stock_name}")
            finally:
                if old_no_proxy is not None:
                    os.environ["no_proxy"] = old_no_proxy
                else:
                    os.environ.pop("no_proxy", None)
        except Exception as e:
            print(f"Failed to fetch stock name via Tencent API: {e}")

        # 2. 如果腾讯接口获取失败且依然是代码，则回退到 akshare 大表接口
        if stock_name == stock_code:
            try:
                import akshare as ak
                spot_df = ak.stock_zh_a_spot_em()
                match = spot_df[spot_df["代码"] == stock_code]
                if not match.empty:
                    stock_name = match.iloc[0]["名称"]
                else:
                    fund_df = ak.fund_etf_spot_em()
                    match = fund_df[fund_df["代码"] == stock_code]
                    if not match.empty:
                        stock_name = match.iloc[0]["名称"]
            except Exception:
                pass


        # Create chart using plotly
        chart_json = create_prediction_chart(
            df=df,
            pred_df=pred_df,
            lookback=lookback,
            pred_len=pred_len,
            actual_df=None,
            historical_start_idx=len(df) - lookback,
            stock_name=stock_name,
            stock_code=stock_code
        )

        # Format prediction results
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': y_timestamp[i].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
            })

        prediction_type = f"在线实时预测 ({stock_name} / {stock_code})"

        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': [],
            'has_comparison': False,
            'message': f'在线预测成功！已生成股票 {stock_name} ({stock_code}) 未来 {pred_len} 个交易日的趋势。'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'在线预测失败: {str(e)}'}), 500

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model"""
    global tokenizer, model, predictor
    
    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # Load tokenizer and model
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # Create predictor
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'description': model_config['description']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'name': predictor.model.__class__.__name__,
                    'device': str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })

if __name__ == '__main__':
    print("Starting Kronos Web UI...")
    print(f"Model availability: {MODEL_AVAILABLE}")
    if MODEL_AVAILABLE:
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Tip: Will use simulated data for demonstration")
    
    app.run(debug=True, host='0.0.0.0', port=7070)
