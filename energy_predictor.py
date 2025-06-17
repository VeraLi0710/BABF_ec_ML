
import gradio as gr
import pandas as pd
import numpy as np
import math
import joblib
import traceback
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from itertools import product
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load model and feature list with comprehensive error handling
# --- 以下是修改部分 ---

# 导入新的库
import onnxruntime as rt
# import numpy as np # (确保你的文件里已经导入了numpy)

# 加载特征列表 (这行代码和原来一样，只是换了位置)
feature_cols = joblib.load('gb_features.joblib')

# 加载 ONNX 模型并创建推理会话 (Session)
sess = rt.InferenceSession('gb_model.onnx')
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# 创建一个包装函数，让它用起来和原来的 model.predict 几乎一样
def predict_with_onnx(df_input):
    # 准备输入数据：确保列序正确，并转换为 float32 类型的 numpy 数组
    input_data = df_input[feature_cols].to_numpy().astype(np.float32)
    
    # 运行 ONNX 推理并返回结果
    result = sess.run([output_name], {input_name: input_data})[0]
    return result

# --- 修改完成 ---

# 【【【 最后一步，非常重要！！！】】】
# 在这个文件后面的代码中，所有原来写着 `model.predict(...)` 的地方,
# 全部都要修改为 `predict_with_onnx(...)`。
#
# 例如，你原来的代码可能是：
# pred = model.predict(some_dataframe)[0]
#
# 你需要把它改成：
# pred = predict_with_onnx(some_dataframe)[0]



# Carbon emission factors (kg CO2e per kWh) - UK BEIS 2024 data
CARBON_FACTORS = {
    'electricity': 0.193,  # kg CO2e/kWh - Grid electricity (2024)
    'gas': 0.182,         # kg CO2e/kWh - Natural gas (2024)
    'mixed': 0.187        # Average for mixed systems
}

# Renovation costs (UK market rates 2024)
RENOVATION_COSTS = {
    'wall_insulation': {
        'internal': 65,     # £55-75/m²
        'external': 120,    # £100-140/m²
        'cavity_fill': 22   # £18-26/m² - Cavity wall insulation fill
    },
    'roof_insulation': {
        'loft': 18,         # £15-22/m²
        'flat_roof': 80     # £70-90/m²
    },
    'glazing': {
        'double_glazing': 320,      # £280-360/m²
        'triple_glazing': 450,      # £400-500/m²
        'secondary_glazing': 150    # £130-170/m²
    },
    'heating_system': {
        # These costs are AFTER BUS grant (already deducted)
        'air_source_heat_pump': 6500,     # £14,000 - £7,500 BUS grant
        'ground_source_heat_pump': 12000, # £18,000 - £6,000 BUS grant
        'gas_boiler_upgrade': 2400,       # Standard cost
        'electric_boiler': 1800           # Standard cost
    }
}

# Energy costs from Ofgem Price Cap (October 2024)
ENERGY_COSTS = {
    'electricity': 0.285,  # £/kWh 
    'gas': 0.073,         # £/kWh
    'mixed': 0.18         # Average for mixed systems
}

# Government grants information
GOVERNMENT_GRANTS = {
    'BUS': {
        'air_source_heat_pump': 7500,
        'ground_source_heat_pump': 6000,
        'description': 'Boiler Upgrade Scheme (BUS) grants already deducted from heat pump costs above'
    },
    'ECO4': {
        'max_amount': 10000,
        'description': 'Energy Company Obligation - for eligible low-income households'
    }
}

def safe_float_conversion(value, default=0.0):
    """Safely convert value to float"""
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        print(f"Warning: Could not convert {value} to float, using default {default}")
        return default

def safe_int_conversion(value, default=1):
    """Safely convert value to int"""
    try:
        if value is None:
            return default
        return int(float(value))  # Convert through float first to handle "2.0" strings
    except (ValueError, TypeError):
        print(f"Warning: Could not convert {value} to int, using default {default}")
        return default

def get_energy_cost_per_kwh(fuel_type):
    """Get energy cost per kWh with comprehensive error handling"""
    try:
        if fuel_type is None:
            return ENERGY_COSTS['mixed']
        
        fuel_type_str = str(fuel_type).lower().strip()
        
        if 'electric' in fuel_type_str:
            return ENERGY_COSTS['electricity']
        elif 'gas' in fuel_type_str:
            return ENERGY_COSTS['gas']
        else:
            return ENERGY_COSTS['mixed']
    except Exception as e:
        print(f"Warning: Error getting energy cost for {fuel_type}: {e}")
        return ENERGY_COSTS['mixed']

def get_carbon_factor(fuel_type):
    """Get carbon emission factor per kWh"""
    try:
        if fuel_type is None:
            return CARBON_FACTORS['mixed']
        
        fuel_type_str = str(fuel_type).lower().strip()
        
        if 'electric' in fuel_type_str:
            return CARBON_FACTORS['electricity']
        elif 'gas' in fuel_type_str:
            return CARBON_FACTORS['gas']
        else:
            return CARBON_FACTORS['mixed']
    except Exception as e:
        print(f"Warning: Error getting carbon factor for {fuel_type}: {e}")
        return CARBON_FACTORS['mixed']

def calculate_actual_areas(total_floor_area, building_type='flat'):
    """Calculate actual renovation areas with comprehensive validation"""
    try:
        # Input validation and conversion
        floor_area = safe_float_conversion(total_floor_area, 60.0)
        
        if floor_area <= 0:
            print(f"Warning: Invalid floor area {floor_area}, using default 60")
            floor_area = 60.0
        
        if floor_area > 2000:  # Sanity check
            print(f"Warning: Very large floor area: {floor_area} m²")

        if building_type == 'flat':
            # Conservative calculation for flats
            wall_perimeter = math.sqrt(floor_area) * 2.5
            wall_area = wall_perimeter * 2.5 * 0.4  # Only 40% is external wall
            glazing_area = wall_area * 0.2  # 20% of wall area
            roof_area = floor_area
        else:  # house
            wall_area = floor_area * 1.8
            glazing_area = floor_area * 0.15
            roof_area = floor_area / math.cos(math.radians(30))
        
        # Ensure all areas are positive
        wall_area = max(wall_area, 1.0)
        glazing_area = max(glazing_area, 1.0)
        roof_area = max(roof_area, 1.0)
        
        return wall_area, glazing_area, roof_area
    
    except Exception as e:
        print(f"Error in calculate_actual_areas: {e}")
        # Return safe default values
        default_area = safe_float_conversion(total_floor_area, 60.0)
        return default_area * 0.3, default_area * 0.06, default_area

def determine_roof_type_from_location(floor_location, roof_type):
    """Determine final roof type based on floor location"""
    try:
        if floor_location == 'top floor':
            return str(roof_type) if roof_type else 'pitched'
        else:
            return 'another dwelling above'
    except Exception as e:
        print(f"Error in determine_roof_type_from_location: {e}")
        return 'another dwelling above'

def get_energy_prediction(total_floor_area, estimated_floor_count, epc_score,
                         wall_insulation, roof_type, roof_insulation, glazing_type,
                         built_form, main_heat_type, main_fuel_type, lookup_age_band, wall_type):
    """Energy prediction function with comprehensive error handling"""
    try:
        # Input validation and conversion
        floor_area = safe_float_conversion(total_floor_area, 60.0)
        floor_count = safe_float_conversion(estimated_floor_count, 2.0)
        epc = safe_float_conversion(epc_score, 50.0)
        
        # Validate ranges
        floor_area = max(10.0, min(1000.0, floor_area))
        floor_count = max(1.0, min(10.0, floor_count))
        epc = max(1.0, min(100.0, epc))
        
        # Ensure all string inputs are valid
        wall_insulation = str(wall_insulation) if wall_insulation else 'uninsulated'
        roof_type = str(roof_type) if roof_type else 'pitched'
        roof_insulation = str(roof_insulation) if roof_insulation else 'uninsulated'
        glazing_type = str(glazing_type) if glazing_type else 'single/partial'
        built_form = str(built_form) if built_form else 'mid-terrace'
        main_heat_type = str(main_heat_type) if main_heat_type else 'boiler'
        main_fuel_type = str(main_fuel_type) if main_fuel_type else 'mains gas'
        lookup_age_band = str(lookup_age_band) if lookup_age_band else '1950-1966'
        wall_type = str(wall_type) if wall_type else 'solid'
        
        # U-value calculations with error handling
        U_wall = 0.37 if wall_insulation == 'insulated' else 1.7
        
        if roof_insulation == 'insulated':
            U_roof = 0.25
        elif roof_type == 'flat':
            U_roof = 0.28
        else:
            U_roof = 2.3
        
        U_floor = 0.25
        
        if glazing_type == 'double/triple':
            U_glazing = 2.4
        elif glazing_type == 'secondary':
            U_glazing = 2.82
        else:
            U_glazing = 5.75
        
        # Area calculations
        if roof_type == 'pitched':
            roof_area = floor_area / math.cos(math.radians(30))
        elif roof_type == 'flat':
            roof_area = floor_area
        else:
            roof_area = 0
        
        wall_area = floor_area * 2.1
        floor_area_calc = floor_area
        glazing_area = floor_area * 0.18
        delta_T = 18

        Q_total = (U_wall * wall_area + U_roof * roof_area + U_floor * floor_area_calc + U_glazing * glazing_area) * delta_T

        # Create input data with safe conversions
        rowdict = {
            'epc_score': epc,
            'estimated_floor_count': floor_count,
            'wall_area': wall_area,
            'roof_area': roof_area,
            'floor_area': floor_area_calc,
            'glazing_area': glazing_area,
            'u_value_wall': U_wall,
            'u_value_roof': U_roof,
            'u_value_floor': U_floor,
            'u_value_glazing': U_glazing,
            'Q_total': Q_total,
            'wall_type': wall_type,
            'wall_insulation': wall_insulation,
            'roof_type': roof_type,
            'roof_insulation': roof_insulation,
            'glazing_type': glazing_type,
            'built_form': built_form,
            'main_heat_type': main_heat_type,
            'main_fuel_type': main_fuel_type,
            'lookup_age_band': lookup_age_band
        }
        
        # Create DataFrame and make prediction
        df_input = pd.DataFrame([rowdict])
        df_input = pd.get_dummies(df_input)
        
        # Ensure all required feature columns exist
        for col in feature_cols:
            if col not in df_input.columns:
                df_input[col] = 0
        
        # Select features in correct order
        df_input = df_input[feature_cols]
        
        # Make prediction
        pred =predict_with_onnx(df_input)[0]
        
        # Validate prediction result
        if pred < 0:
            print(f"Warning: Negative prediction {pred}, setting to 0")
            pred = 0
        elif pred > 100000:
            print(f"Warning: Very high prediction {pred}")
        
        # Convert Q_total from W to kWh (assuming full year operation)
        # Q_total (W) * 8760 hours / 1000 = kWh per year
        q_total_kwh = (Q_total * 8760) / 1000
        
        return float(pred), float(q_total_kwh)
    
    except Exception as e:
        print(f"Error in get_energy_prediction: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        # Return reasonable default values
        default_area = safe_float_conversion(total_floor_area, 60.0)
        default_consumption = default_area * 150  # 150 kWh/m² typical
        default_q_total = default_area * 50 * 8760 / 1000      # Convert to kWh
        return float(default_consumption), float(default_q_total)

def generate_optimization_strategies(
    lookup_age_band, total_floor_area, estimated_floor_count, epc_score, built_form,
    floor_location, roof_type, wall_type, wall_insulation, roof_insulation, 
    glazing_type, main_heat_type, main_fuel_type
):
    """Generate all possible renovation combinations and find optimal strategies"""
    try:
        print("🔍 Generating optimization strategies...")
        
        # Determine available options
        final_roof_type = determine_roof_type_from_location(floor_location, roof_type)
        final_roof_insulation = roof_insulation if floor_location == 'top floor' else 'another dwelling above'
        
        # Get original performance
        original_consumption, original_q_total = get_energy_prediction(
            total_floor_area, estimated_floor_count, epc_score,
            wall_insulation, final_roof_type, final_roof_insulation, glazing_type,
            built_form, main_heat_type, main_fuel_type, lookup_age_band, wall_type
        )
        
        # Calculate renovation areas
        wall_area, glazing_area, roof_area = calculate_actual_areas(total_floor_area, 'flat')
        
        # Define available options based on current state
        wall_options = ['no_change']
        if wall_insulation == 'uninsulated':
            if wall_type == 'solid':
                wall_options.extend(['internal', 'external'])
            elif wall_type == 'cavity':
                wall_options.extend(['cavity_fill', 'internal', 'external'])
        
        roof_options = ['no_change']
        if floor_location == 'top floor' and final_roof_insulation == 'uninsulated':
            if final_roof_type == 'pitched':
                roof_options.append('loft')
            elif final_roof_type == 'flat':
                roof_options.append('flat_roof')
        
        glazing_options = ['no_change']
        if glazing_type == 'single/partial':
            glazing_options.extend(['double_glazing', 'triple_glazing', 'secondary_glazing'])
        elif glazing_type == 'secondary':
            glazing_options.extend(['double_glazing', 'triple_glazing'])
        
        heating_options = ['no_change']
        if main_heat_type != 'heat pump':
            heating_options.extend(['air_source_heat_pump', 'ground_source_heat_pump'])
        if main_heat_type == 'boiler' and lookup_age_band in ['pre-1920', '1930-1949', '1950-1966', '1967-1982']:
            heating_options.append('gas_boiler_upgrade')
        if main_heat_type != 'boiler' or main_fuel_type != 'electricity':
            heating_options.append('electric_boiler')
        
        fuel_options = ['no_change']
        if main_fuel_type != 'mains gas':
            fuel_options.append('mains gas')
        if main_fuel_type != 'electricity':
            fuel_options.append('electricity')
        
        # Generate all combinations
        all_combinations = list(product(wall_options, roof_options, glazing_options, heating_options, fuel_options))
        
        strategies = []
        print(f"📊 Analyzing {len(all_combinations)} combinations...")
        
        for i, (wall_ren, roof_ren, glazing_ren, heating_ren, fuel_ren) in enumerate(all_combinations):
            if i % 20 == 0:
                print(f"   Progress: {i}/{len(all_combinations)}")
            
            # Skip no-change combination
            if all(opt == 'no_change' for opt in [wall_ren, roof_ren, glazing_ren, heating_ren, fuel_ren]):
                continue
            
            # Calculate costs and savings for this combination
            try:
                result = calculate_single_strategy(
                    lookup_age_band, total_floor_area, estimated_floor_count, epc_score, built_form,
                    floor_location, roof_type, wall_type, wall_insulation, roof_insulation,
                    glazing_type, main_heat_type, main_fuel_type,
                    wall_ren, roof_ren, glazing_ren, heating_ren, fuel_ren,
                    original_consumption, original_q_total, wall_area, glazing_area, roof_area
                )
                
                if result and result['annual_cost_savings'] > 0:
                    strategies.append(result)
                    
            except Exception as e:
                continue
        
        print(f"✅ Found {len(strategies)} viable strategies")
        
        # Sort strategies
        strategies_by_payback = sorted([s for s in strategies if s['payback_years'] < float('inf')], 
                                     key=lambda x: x['payback_years'])[:5]
        strategies_by_savings = sorted(strategies, key=lambda x: x['annual_cost_savings'], reverse=True)[:5]
        
        return strategies_by_payback, strategies_by_savings, original_consumption, original_q_total
        
    except Exception as e:
        print(f"Error in generate_optimization_strategies: {e}")
        return [], [], 0, 0

def calculate_single_strategy(
    lookup_age_band, total_floor_area, estimated_floor_count, epc_score, built_form,
    floor_location, roof_type, wall_type, wall_insulation, roof_insulation,
    glazing_type, main_heat_type, main_fuel_type,
    wall_renovation, roof_renovation, glazing_renovation, heating_renovation, fuel_change,
    original_consumption, original_q_total, wall_area, glazing_area, roof_area
):
    """Calculate costs and savings for a single strategy"""
    try:
        final_roof_type = determine_roof_type_from_location(floor_location, roof_type)
        final_roof_insulation = roof_insulation if floor_location == 'top floor' else 'another dwelling above'
        
        # Apply renovations
        new_wall_insulation = 'insulated' if wall_renovation != 'no_change' else wall_insulation
        new_roof_insulation = 'insulated' if roof_renovation != 'no_change' else final_roof_insulation
        
        # Glazing upgrades
        new_glazing_type = glazing_type
        if glazing_renovation in ['double_glazing', 'triple_glazing']:
            new_glazing_type = 'double/triple'
        elif glazing_renovation == 'secondary_glazing':
            new_glazing_type = 'secondary'
        
        # Heating system upgrades
        new_heat_type = main_heat_type
        new_fuel_type = main_fuel_type
        
        if heating_renovation in ['air_source_heat_pump', 'ground_source_heat_pump']:
            new_heat_type = 'heat pump'
            new_fuel_type = 'electricity'
        elif heating_renovation == 'gas_boiler_upgrade':
            new_heat_type = 'boiler'
            new_fuel_type = 'mains gas'
        elif heating_renovation == 'electric_boiler':
            new_heat_type = 'boiler'
            new_fuel_type = 'electricity'
        
        # Fuel change override
        if fuel_change != 'no_change':
            new_fuel_type = fuel_change
        
        # Get renovated performance
        renovated_consumption, renovated_q_total = get_energy_prediction(
            total_floor_area, estimated_floor_count, epc_score,
            new_wall_insulation, final_roof_type, new_roof_insulation, new_glazing_type,
            built_form, new_heat_type, new_fuel_type, lookup_age_band, wall_type
        )
        
        # Calculate costs
        total_cost = 0
        cost_breakdown = []
        
        # Wall insulation cost
        if wall_renovation != 'no_change' and wall_renovation in RENOVATION_COSTS['wall_insulation']:
            cost = wall_area * RENOVATION_COSTS['wall_insulation'][wall_renovation]
            total_cost += cost
            cost_breakdown.append(f'Wall ({wall_renovation})')
        
        # Roof insulation cost
        if (roof_renovation != 'no_change' and 
            floor_location == 'top floor' and 
            roof_renovation in RENOVATION_COSTS['roof_insulation']):
            cost = roof_area * RENOVATION_COSTS['roof_insulation'][roof_renovation]
            total_cost += cost
            cost_breakdown.append(f'Roof ({roof_renovation})')
        
        # Glazing cost
        if glazing_renovation != 'no_change' and glazing_renovation in RENOVATION_COSTS['glazing']:
            cost = glazing_area * RENOVATION_COSTS['glazing'][glazing_renovation]
            total_cost += cost
            cost_breakdown.append(f'Glazing ({glazing_renovation})')
        
        # Heating system cost
        if heating_renovation != 'no_change' and heating_renovation in RENOVATION_COSTS['heating_system']:
            cost = RENOVATION_COSTS['heating_system'][heating_renovation]
            total_cost += cost
            cost_breakdown.append(f'Heating ({heating_renovation})')
        
        # Calculate savings
        annual_savings = max(0, original_consumption - renovated_consumption)
        original_energy_cost = get_energy_cost_per_kwh(main_fuel_type)
        new_energy_cost = get_energy_cost_per_kwh(new_fuel_type)
        
        original_annual_cost = original_consumption * original_energy_cost
        new_annual_cost = renovated_consumption * new_energy_cost
        annual_cost_savings = original_annual_cost - new_annual_cost
        
        # Calculate carbon savings
        original_carbon_factor = get_carbon_factor(main_fuel_type)
        new_carbon_factor = get_carbon_factor(new_fuel_type)
        
        original_annual_carbon = original_consumption * original_carbon_factor
        new_annual_carbon = renovated_consumption * new_carbon_factor
        annual_carbon_savings = original_annual_carbon - new_annual_carbon
        
        # Calculate payback period
        if annual_cost_savings > 0:
            payback_years = total_cost / annual_cost_savings
        else:
            payback_years = float('inf')
        
        return {
            'combination': ' + '.join(cost_breakdown),
            'total_cost': total_cost,
            'annual_cost_savings': annual_cost_savings,
            'annual_carbon_savings': annual_carbon_savings,
            'payback_years': payback_years,
            'energy_savings': annual_savings,
            'renovated_consumption': renovated_consumption,
            'renovated_q_total': renovated_q_total,
            'details': {
                'wall': wall_renovation,
                'roof': roof_renovation,
                'glazing': glazing_renovation,
                'heating': heating_renovation,
                'fuel': fuel_change
            }
        }
        
    except Exception as e:
        return None

def create_optimization_chart(strategies_by_payback, strategies_by_savings):
    """Create optimization scatter plot using Plotly"""
    try:
        print("Creating optimization chart...")
        
        # 合并并去重策略
        all_strategies = {}
        
        # 添加最快回本策略
        for i, strategy in enumerate(strategies_by_payback[:5]):
            key = strategy['combination']
            if key not in all_strategies:
                all_strategies[key] = {
                    'combination': strategy['combination'],
                    'payback_years': strategy['payback_years'],
                    'annual_savings': strategy['annual_cost_savings'],
                    'annual_carbon_savings': strategy['annual_carbon_savings'],
                    'total_cost': strategy['total_cost'],
                    'category': 'Fastest Payback',
                    'rank_payback': i + 1,
                    'rank_savings': None
                }
        
        # 添加最高节省策略
        for i, strategy in enumerate(strategies_by_savings[:5]):
            key = strategy['combination']
            if key in all_strategies:
                all_strategies[key]['rank_savings'] = i + 1
                if all_strategies[key]['category'] == 'Fastest Payback':
                    all_strategies[key]['category'] = 'Both Top 5'
            else:
                all_strategies[key] = {
                    'combination': strategy['combination'],
                    'payback_years': strategy['payback_years'],
                    'annual_savings': strategy['annual_cost_savings'],
                    'annual_carbon_savings': strategy['annual_carbon_savings'],
                    'total_cost': strategy['total_cost'],
                    'category': 'Highest Savings',
                    'rank_payback': None,
                    'rank_savings': i + 1
                }
        
        # 转换为列表
        plot_data = list(all_strategies.values())
        
        if not plot_data:
            print("No data for chart")
            return None
        
        print(f"Chart data: {len(plot_data)} strategies")
        
        # 创建散点图
        fig = go.Figure()
        
        # 定义颜色和标记
        colors = {
            'Fastest Payback': '#FF6B6B',    # 红色
            'Highest Savings': '#4ECDC4',    # 青色
            'Both Top 5': '#45B7D1'          # 蓝色
        }
        
        symbols = {
            'Fastest Payback': 'circle',
            'Highest Savings': 'square',
            'Both Top 5': 'star'
        }
        
        # 按类别分组绘制
        for category in ['Fastest Payback', 'Highest Savings', 'Both Top 5']:
            category_data = [s for s in plot_data if s['category'] == category]
            
            if not category_data:
                continue
            
            # 创建简化标签
            labels = []
            for s in category_data:
                if s['rank_payback'] and s['rank_savings']:
                    labels.append(f"#{s['rank_payback']}/{s['rank_savings']}")
                elif s['rank_payback']:
                    labels.append(f"#{s['rank_payback']}")
                elif s['rank_savings']:
                    labels.append(f"#{s['rank_savings']}")
                else:
                    labels.append("")
            
            fig.add_trace(go.Scatter(
                x=[s['payback_years'] for s in category_data],
                y=[s['annual_savings'] for s in category_data],
                mode='markers+text',
                marker=dict(
                    size=[max(15, min(30, 15 + (s['total_cost'] / 500))) for s in category_data],
                    color=colors[category],
                    symbol=symbols[category],
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=12, color='black', family="Arial Black"),
                name=category,
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>' +
                    'Payback: %{x:.1f} years<br>' +
                    'Annual Savings: £%{y:,.0f}<br>' +
                    'Carbon Savings: %{customdata[2]:,.0f} kg CO2e/year<br>' +
                    'Investment: £%{customdata[1]:,.0f}<br>' +
                    '<extra></extra>'
                ),
                customdata=[[s['combination'][:40] + "..." if len(s['combination']) > 40 else s['combination'], 
                           s['total_cost'], s['annual_carbon_savings']] for s in category_data]
            ))
        
        # 添加理想区域
        max_savings = max([s['annual_savings'] for s in plot_data])
        fig.add_shape(
            type="rect",
            x0=0, x1=10, y0=max_savings * 0.5, y1=max_savings * 1.1,
            fillcolor="lightgreen",
            opacity=0.1,
            line=dict(width=0),
        )
        
        fig.add_annotation(
            x=5, y=max_savings * 0.8,
            text="🎯 Sweet Spot",
            showarrow=False,
            font=dict(size=14, color="green"),
            bgcolor="white",
            bordercolor="green",
            borderwidth=1
        )
        
        # 更新布局
        fig.update_layout(
            title={
                'text': '🏠 Retrofit Investment Strategy Analysis',
                'x': 0.5,
                'font': {'size': 20, 'family': 'Arial'}
            },
            xaxis=dict(
                title='⏱️ Payback Period (Years)',
                gridcolor='lightgray',
                showgrid=True,
                range=[0, max([s['payback_years'] for s in plot_data]) * 1.1],
                title_font=dict(size=14)
            ),
            yaxis=dict(
                title='💰 Annual Savings (£)',
                gridcolor='lightgray',
                showgrid=True,
                range=[0, max_savings * 1.1],
                title_font=dict(size=14)
            ),
            plot_bgcolor='white',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=12)
            ),
            hovermode='closest',
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        print("Chart created successfully")
        return fig
        
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

def predict_current_energy_and_show_options(
    lookup_age_band, total_floor_area, estimated_floor_count, epc_score, built_form,
    floor_location, roof_type, wall_type, wall_insulation, roof_insulation, 
    glazing_type, main_heat_type, main_fuel_type
):
    """Predict current energy consumption and show available renovation options"""
    try:
        print(f"Starting analysis with inputs: floor_area={total_floor_area}, epc={epc_score}")
        
        # Input validation - ensure all required parameters exist
        required_params = [
            ('lookup_age_band', lookup_age_band),
            ('built_form', built_form),
            ('wall_type', wall_type),
            ('wall_insulation', wall_insulation),
            ('glazing_type', glazing_type),
            ('main_heat_type', main_heat_type),
            ('main_fuel_type', main_fuel_type),
            ('floor_location', floor_location)
        ]
        
        for param_name, param_value in required_params:
            if not param_value:
                raise ValueError(f"Missing required parameter: {param_name}")
        
        # Determine final roof parameters
        final_roof_type = determine_roof_type_from_location(floor_location, roof_type)
        final_roof_insulation = roof_insulation if floor_location == 'top floor' else 'another dwelling above'
        
        print(f"Roof type determined: {final_roof_type}, insulation: {final_roof_insulation}")
        
        # Get energy prediction
        consumption, q_total = get_energy_prediction(
            total_floor_area, estimated_floor_count, epc_score,
            wall_insulation, final_roof_type, final_roof_insulation, glazing_type,
            built_form, main_heat_type, main_fuel_type, lookup_age_band, wall_type
        )
        
        print(f"Energy prediction completed: {consumption} kWh, {q_total} kWh")
        
        # Calculate costs and carbon emissions
        energy_cost = get_energy_cost_per_kwh(main_fuel_type)
        annual_cost = consumption * energy_cost
        
        carbon_factor = get_carbon_factor(main_fuel_type)
        annual_carbon = consumption * carbon_factor
        
        # Calculate renovation areas
        wall_area, glazing_area, roof_area = calculate_actual_areas(total_floor_area, 'flat')
        
        print(f"Areas calculated: wall={wall_area:.1f}, glazing={glazing_area:.1f}, roof={roof_area:.1f}")
        
        # Performance result display - 包含碳排放
        performance_result = f"""
## 🏠 Current Building Performance

| **Metric** | **Value** |
|------------|-----------|
| **Annual Energy Consumption** | {consumption:,.0f} kWh |
| **Annual Energy Bills** | £{annual_cost:,.0f} |
| **Annual Carbon Emissions** | {annual_carbon:,.0f} kg CO2e |
| **Total Heating Demand** | {q_total:,.0f} kWh/year |
| **Floor Area** | {safe_float_conversion(total_floor_area):.0f} m² |
| **EPC Score** | {safe_float_conversion(epc_score):.0f} |

**Current Systems:** {wall_type} walls ({wall_insulation}) • {final_roof_type} roof ({final_roof_insulation}) • {glazing_type} glazing • {main_heat_type} ({main_fuel_type})

**🌱 Carbon Impact:** Your property produces {annual_carbon/1000:.1f} tonnes of CO2 equivalent per year.
        """
        
        # Generate optimization strategies
        strategies_by_payback, strategies_by_savings, orig_consumption, orig_q_total = generate_optimization_strategies(
            lookup_age_band, total_floor_area, estimated_floor_count, epc_score, built_form,
            floor_location, roof_type, wall_type, wall_insulation, roof_insulation,
            glazing_type, main_heat_type, main_fuel_type
        )
        
        # Create optimization scatter plot
        optimization_chart = None
        chart_visible = False
        if strategies_by_payback or strategies_by_savings:
            optimization_chart = create_optimization_chart(strategies_by_payback, strategies_by_savings)
            if optimization_chart:
                chart_visible = True
                print("Chart will be displayed")
            else:
                print("Chart creation failed")
        
        # Determine available renovation options
        wall_options = ['no_change']
        if wall_insulation == 'uninsulated':
            if wall_type == 'solid':
                wall_options.extend(['internal', 'external'])
            elif wall_type == 'cavity':
                wall_options.extend(['cavity_fill', 'internal', 'external'])
        
        roof_options = ['no_change']
        if floor_location == 'top floor' and final_roof_insulation == 'uninsulated':
            if final_roof_type == 'pitched':
                roof_options.append('loft')
            elif final_roof_type == 'flat':
                roof_options.append('flat_roof')
        
        glazing_options = ['no_change']
        if glazing_type == 'single/partial':
            glazing_options.extend(['double_glazing', 'triple_glazing', 'secondary_glazing'])
        elif glazing_type == 'secondary':
            glazing_options.extend(['double_glazing', 'triple_glazing'])
        
        heating_options = ['no_change']
        if main_heat_type != 'heat pump':
            heating_options.extend(['air_source_heat_pump', 'ground_source_heat_pump'])
        if main_heat_type == 'boiler' and lookup_age_band in ['pre-1920', '1930-1949', '1950-1966', '1967-1982']:
            heating_options.append('gas_boiler_upgrade')
        if main_heat_type != 'boiler' or main_fuel_type != 'electricity':
            heating_options.append('electric_boiler')
        
        fuel_options = ['no_change']
        if main_fuel_type != 'mains gas':
            fuel_options.append('mains gas')
        if main_fuel_type != 'electricity':
            fuel_options.append('electricity')
        
        # Check if any renovation options are available
        show_options = any([
            len(wall_options) > 1,
            len(roof_options) > 1,
            len(glazing_options) > 1,
            len(heating_options) > 1,
            len(fuel_options) > 1
        ])
        
        print(f"Options available: wall={len(wall_options)}, roof={len(roof_options)}, glazing={len(glazing_options)}, heating={len(heating_options)}, fuel={len(fuel_options)}")
        print(f"Chart visible: {chart_visible}")
        
        return (
            performance_result,
            gr.Radio(choices=wall_options, value='no_change', visible=(len(wall_options) > 1)),
            gr.Radio(choices=roof_options, value='no_change', visible=(len(roof_options) > 1)),
            gr.Radio(choices=glazing_options, value='no_change', visible=(len(glazing_options) > 1)),
            gr.Radio(choices=heating_options, value='no_change', visible=(len(heating_options) > 1)),
            gr.Radio(choices=fuel_options, value='no_change', visible=(len(fuel_options) > 1)),
            gr.Markdown("## 🔧 Available Renovation Options", visible=show_options),
            gr.Button("💰 Calculate Renovation Analysis", visible=show_options),
            gr.Plot(value=optimization_chart, visible=chart_visible)
        )
    
    except Exception as e:
        error_msg = f"❌ Error in building analysis: {str(e)}\n\nPlease check all building parameters and try again."
        print(f"Error in predict_current_energy_and_show_options: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return (
            error_msg,
            gr.Radio(choices=['no_change'], value='no_change', visible=False),
            gr.Radio(choices=['no_change'], value='no_change', visible=False),
            gr.Radio(choices=['no_change'], value='no_change', visible=False),
            gr.Radio(choices=['no_change'], value='no_change', visible=False),
            gr.Radio(choices=['no_change'], value='no_change', visible=False),
            gr.Markdown(visible=False),
            gr.Button(visible=False),
            gr.Plot(visible=False)
        )

def calculate_renovation_analysis(
    lookup_age_band, total_floor_area, estimated_floor_count, epc_score, built_form,
    floor_location, roof_type, wall_type, wall_insulation, roof_insulation, 
    glazing_type, main_heat_type, main_fuel_type,
    wall_renovation, roof_renovation, glazing_renovation, heating_renovation, fuel_change
):
    """Calculate renovation costs and energy savings"""
    try:
        print(f"Starting renovation analysis...")
        
        # Handle None values from hidden radio buttons
        wall_renovation = wall_renovation or 'no_change'
        roof_renovation = roof_renovation or 'no_change'
        glazing_renovation = glazing_renovation or 'no_change'
        heating_renovation = heating_renovation or 'no_change'
        fuel_change = fuel_change or 'no_change'
        
        print(f"Renovations selected: wall={wall_renovation}, roof={roof_renovation}, glazing={glazing_renovation}, heating={heating_renovation}, fuel={fuel_change}")
        
        # Determine final roof parameters
        final_roof_type = determine_roof_type_from_location(floor_location, roof_type)
        final_roof_insulation = roof_insulation if floor_location == 'top floor' else 'another dwelling above'
        
        # Check if any renovations are selected
        no_renovations = all([
            wall_renovation == 'no_change',
            roof_renovation == 'no_change',
            glazing_renovation == 'no_change',
            heating_renovation == 'no_change',
            fuel_change == 'no_change'
        ])
        
        if no_renovations:
            return "⚠️ Please select at least one renovation option to see the analysis."
        
        # Calculate renovation areas
        wall_area, glazing_area, roof_area = calculate_actual_areas(total_floor_area, 'flat')
        
        # Get original performance
        original_consumption, original_q_total = get_energy_prediction(
            total_floor_area, estimated_floor_count, epc_score,
            wall_insulation, final_roof_type, final_roof_insulation, glazing_type,
            built_form, main_heat_type, main_fuel_type, lookup_age_band, wall_type
        )
        
        print(f"Original consumption: {original_consumption} kWh")
        
        # Apply renovations
        new_wall_insulation = 'insulated' if wall_renovation != 'no_change' else wall_insulation
        new_roof_insulation = 'insulated' if roof_renovation != 'no_change' else final_roof_insulation
        
        # Glazing upgrades
        new_glazing_type = glazing_type
        if glazing_renovation in ['double_glazing', 'triple_glazing']:
            new_glazing_type = 'double/triple'
        elif glazing_renovation == 'secondary_glazing':
            new_glazing_type = 'secondary'
        
        # Heating system upgrades
        new_heat_type = main_heat_type
        new_fuel_type = main_fuel_type
        
        if heating_renovation in ['air_source_heat_pump', 'ground_source_heat_pump']:
            new_heat_type = 'heat pump'
            new_fuel_type = 'electricity'
        elif heating_renovation == 'gas_boiler_upgrade':
            new_heat_type = 'boiler'
            new_fuel_type = 'mains gas'
        elif heating_renovation == 'electric_boiler':
            new_heat_type = 'boiler'
            new_fuel_type = 'electricity'
        
        # Fuel change override
        if fuel_change != 'no_change':
            new_fuel_type = fuel_change
        
        # Get renovated performance
        renovated_consumption, renovated_q_total = get_energy_prediction(
            total_floor_area, estimated_floor_count, epc_score,
            new_wall_insulation, final_roof_type, new_roof_insulation, new_glazing_type,
            built_form, new_heat_type, new_fuel_type, lookup_age_band, wall_type
        )
        
        print(f"Renovated consumption: {renovated_consumption} kWh")
        
        # Calculate costs
        total_cost = 0
        cost_breakdown = {}
        
        # Wall insulation cost
        if wall_renovation != 'no_change' and wall_renovation in RENOVATION_COSTS['wall_insulation']:
            cost = wall_area * RENOVATION_COSTS['wall_insulation'][wall_renovation]
            total_cost += cost
            cost_breakdown[f'Wall Insulation ({wall_renovation})'] = cost
        
        # Roof insulation cost
        if (roof_renovation != 'no_change' and 
            floor_location == 'top floor' and 
            roof_renovation in RENOVATION_COSTS['roof_insulation']):
            cost = roof_area * RENOVATION_COSTS['roof_insulation'][roof_renovation]
            total_cost += cost
            cost_breakdown[f'Roof Insulation ({roof_renovation})'] = cost
        
        # Glazing cost
        if glazing_renovation != 'no_change' and glazing_renovation in RENOVATION_COSTS['glazing']:
            cost = glazing_area * RENOVATION_COSTS['glazing'][glazing_renovation]
            total_cost += cost
            cost_breakdown[f'Glazing ({glazing_renovation})'] = cost
        
        # Heating system cost
        if heating_renovation != 'no_change' and heating_renovation in RENOVATION_COSTS['heating_system']:
            cost = RENOVATION_COSTS['heating_system'][heating_renovation]
            total_cost += cost
            cost_breakdown[f'Heating ({heating_renovation})'] = cost
        
        # Calculate savings
        annual_savings = max(0, original_consumption - renovated_consumption)
        original_energy_cost = get_energy_cost_per_kwh(main_fuel_type)
        new_energy_cost = get_energy_cost_per_kwh(new_fuel_type)
        
        original_annual_cost = original_consumption * original_energy_cost
        new_annual_cost = renovated_consumption * new_energy_cost
        annual_cost_savings = original_annual_cost - new_annual_cost
        
        # Calculate carbon savings
        original_carbon_factor = get_carbon_factor(main_fuel_type)
        new_carbon_factor = get_carbon_factor(new_fuel_type)
        
        original_annual_carbon = original_consumption * original_carbon_factor
        new_annual_carbon = renovated_consumption * new_carbon_factor
        annual_carbon_savings = original_annual_carbon - new_annual_carbon
        
        # Calculate payback period
        if annual_cost_savings > 0:
            payback_years = total_cost / annual_cost_savings
        else:
            payback_years = float('inf')
        
        # Calculate efficiency improvement
        if original_consumption > 0:
            efficiency_improvement = (annual_savings / original_consumption) * 100
        else:
            efficiency_improvement = 0
        
        print(f"Analysis complete: savings={annual_savings} kWh, cost_savings=£{annual_cost_savings:.0f}, carbon_savings={annual_carbon_savings:.0f} kg CO2e, payback={payback_years:.1f} years")
        
        # Generate results
        result_text = f"""
# 🏠 Renovation Analysis Results

## 📊 Performance Impact Summary
| **Metric** | **Current** | **After Renovation** | **Improvement** |
|------------|-------------|---------------------|-----------------|
| **Energy Consumption** | {original_consumption:,.0f} kWh | {renovated_consumption:,.0f} kWh | **{annual_savings:,.0f} kWh** ({efficiency_improvement:.1f}% ↓) |
| **Annual Energy Bills** | £{original_annual_cost:,.0f} | £{new_annual_cost:,.0f} | **£{annual_cost_savings:,.0f}** saved/year |
| **Carbon Emissions** | {original_annual_carbon:,.0f} kg CO2e | {new_annual_carbon:,.0f} kg CO2e | **{annual_carbon_savings:,.0f} kg CO2e** saved/year |
| **Total Heating Demand** | {original_q_total:,.0f} kWh/year | {renovated_q_total:,.0f} kWh/year | **{original_q_total - renovated_q_total:,.0f} kWh/year** ↓ |

## 💰 Investment Analysis
- **Total Renovation Investment**: £{total_cost:,.0f} *(one-time upfront cost)*
- **Annual Bill Savings**: £{annual_cost_savings:,.0f} *(recurring yearly savings)*
- **Simple Payback Period**: {payback_years:.1f} years
- **25-Year Total Savings**: £{annual_cost_savings * 25:,.0f}

## 🌱 Carbon Impact
- **Annual Carbon Reduction**: {annual_carbon_savings:,.0f} kg CO2e/year ({annual_carbon_savings/1000:.1f} tonnes/year)
- **25-Year Carbon Saved**: {annual_carbon_savings * 25 / 1000:.1f} tonnes CO2e
- **Equivalent to**: {annual_carbon_savings / 4600:.1f} cars removed from roads for a year

## 🔧 Selected Renovations & Costs
"""
        
        if cost_breakdown:
            for item, cost in cost_breakdown.items():
                percentage = (cost / total_cost * 100) if total_cost > 0 else 0
                result_text += f"- **{item}**: £{cost:,.0f} ({percentage:.1f}%)\n"
        else:
            result_text += "- No renovation costs calculated (free fuel switch only)\n"
        
        result_text += f"""

## 📏 Area Calculations Used:
- **External Wall Area**: {wall_area:.1f} m² (only exterior walls of your unit)
- **Window/Glazing Area**: {glazing_area:.1f} m² (20% of external wall area)
- **Roof Area**: {roof_area:.1f} m² (only if top floor)

## 🏛️ Government Support Information:
- **BUS Grant**: {GOVERNMENT_GRANTS['BUS']['description']}
- **ECO4**: Up to £{GOVERNMENT_GRANTS['ECO4']['max_amount']:,} for eligible households
- **Local Grants**: Check your council for additional support
"""
        
        # Show fuel cost impact if fuel type changes
        if new_fuel_type != main_fuel_type:
            result_text += f"""
## 💡 Fuel Cost Impact:
**Note**: Switching from {main_fuel_type} (£{original_energy_cost:.3f}/kWh) to {new_fuel_type} (£{new_energy_cost:.3f}/kWh)
"""
        
        # Investment recommendation
        if annual_cost_savings > 0:
            if payback_years <= 7:
                result_text += "\n## ✅ **Investment Recommendation: HIGHLY RECOMMENDED**\n**Excellent ROI** - Short payback period with strong long-term savings and carbon benefits."
            elif payback_years <= 15:
                result_text += "\n## ⚖️ **Investment Recommendation: RECOMMENDED**\n**Good ROI** - Reasonable payback period with solid financial and environmental benefits."
            else:
                result_text += "\n## ⚠️ **Investment Recommendation: CONSIDER CAREFULLY**\n**Extended Payback** - Prioritize highest-impact measures first for better ROI."
        else:
            result_text += "\n## ❌ **Investment Recommendation: NOT RECOMMENDED**\n**Negative savings** - This combination increases annual costs. Consider different options."
        
        return result_text
    
    except Exception as e:
        error_msg = f"❌ Error in renovation analysis: {str(e)}\n\nPlease check your selections and try again."
        print(f"Error in calculate_renovation_analysis: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return error_msg

def update_roof_type_visibility(floor_location):
    """Update roof type options visibility based on floor location"""
    try:
        if floor_location == 'top floor':
            return gr.Radio(
                choices=['pitched', 'flat', 'room in roof'],
                value='pitched',
                visible=True
            )
        else:
            return gr.Radio(
                choices=['another dwelling above'],
                value='another dwelling above',
                visible=False
            )
    except Exception as e:
        print(f"Error in update_roof_type_visibility: {e}")
        return gr.Radio(
            choices=['pitched'],
            value='pitched',
            visible=True
        )

def update_roof_insulation_visibility(floor_location):
    """Update roof insulation options visibility based on floor location"""
    try:
        if floor_location == 'top floor':
            return gr.Radio(
                choices=['insulated', 'uninsulated'],
                value='uninsulated',
                visible=True
            )
        else:
            return gr.Radio(
                choices=['another dwelling above'],
                value='another dwelling above',
                visible=False
            )
    except Exception as e:
        print(f"Error in update_roof_insulation_visibility: {e}")
        return gr.Radio(
            choices=['uninsulated'],
            value='uninsulated',
            visible=True
        )

# Create interface with comprehensive error handling
try:
    with gr.Blocks(theme=gr.themes.Soft(), title="🏠 Home Retrofit Calculator") as demo:
        gr.Markdown("# 🏠 Home Retrofit Calculator")
        gr.Markdown("**Find the best energy-saving upgrades for your property with real UK costs, payback analysis & carbon impact**")
        
        with gr.Row():
            # Left column - Building inputs
            with gr.Column(scale=1):
                gr.Markdown("## 🏢 Building Information")
                
                lookup_age_band = gr.Radio(
                    choices=['pre-1920', '1930-1949', '1950-1966', '1967-1982', '1983-1995', '1996-2011', '2012-onwards'],
                    label="📅 Construction Age Band",
                    value="1950-1966"
                )
                
                total_floor_area = gr.Number(
                    label="🏠 Total Floor Area (m²)",
                    value=60,
                    minimum=10,
                    maximum=1000
                )
                
                estimated_floor_count = gr.Number(
                    label="🏢 Above-ground Floors",
                    value=2,
                    minimum=1,
                    maximum=10
                )
                
                epc_score = gr.Number(
                    label="📊 EPC Score",
                    value=50,
                    minimum=1,
                    maximum=100
                )
                
                built_form = gr.Radio(
                    choices=['end-terrace', 'mid-terrace'],
                    label="🏘️ Built Form",
                    value="mid-terrace"
                )
                
                # Floor location and roof type
                floor_location = gr.Radio(
                    choices=['top floor', 'other floor'],
                    label="🏢 Floor Location",
                    value="top floor"
                )
                
                roof_type = gr.Radio(
                    choices=['pitched', 'flat', 'room in roof'],
                    label="🏠 Roof Type",
                    value="pitched"
                )
                
                # Wall information
                with gr.Row():
                    wall_type = gr.Radio(
                        choices=['solid', 'cavity'],
                        label="🧱 Wall Type",
                        value="solid"
                    )
                    wall_insulation = gr.Radio(
                        choices=['insulated', 'uninsulated'],
                        label="🧱 Wall Insulation",
                        value="uninsulated"
                    )
                
                # Roof insulation
                roof_insulation = gr.Radio(
                    choices=['insulated', 'uninsulated'],
                    label="🏠 Roof Insulation",
                    value="uninsulated"
                )
                
                # Other systems
                glazing_type = gr.Radio(
                    choices=['single/partial', 'double/triple', 'secondary'],
                    label="🪟 Glazing Type",
                    value="single/partial"
                )
                
                with gr.Row():
                    main_heat_type = gr.Radio(
                        choices=['boiler', 'communal', 'room/storage heaters', 'heat pump', 'other', 'no heating system'],
                        label="🔥 Main Heating",
                        value="boiler"
                    )
                    main_fuel_type = gr.Radio(
                        choices=['mains gas', 'electricity', 'other', 'no heating system'],
                        label="⚡ Main Fuel",
                        value="mains gas"
                    )
                
                analyze_btn = gr.Button(
                    "📊 Analyze Building Performance",
                    variant="primary",
                    size="lg"
                )
            
            # Right column - Results and options
            with gr.Column(scale=1):
                current_performance = gr.Markdown()
                
                # Optimization strategies chart
                optimization_chart = gr.Plot(label="📊 Retrofit Strategy Analysis", visible=False)
                
                # Renovation options (initially hidden)
                options_title = gr.Markdown("## 🔧 Available Renovation Options", visible=False)
                wall_renovation = gr.Radio(label="🧱 Wall Insulation Upgrade", visible=False)
                roof_renovation = gr.Radio(label="🏠 Roof Insulation Upgrade", visible=False)
                glazing_renovation = gr.Radio(label="🪟 Glazing Upgrade", visible=False)
                heating_renovation = gr.Radio(label="🔥 Heating System Upgrade", visible=False)
                fuel_change = gr.Radio(label="⚡ Fuel Type Change", visible=False)
                
                calculate_btn = gr.Button(
                    "💰 Calculate Renovation Analysis",
                    variant="secondary",
                    size="lg",
                    visible=False
                )
                
                # Results area
                renovation_results = gr.Markdown()
        
        # Event handlers with error handling
        def safe_update_roof_type(floor_location):
            try:
                return update_roof_type_visibility(floor_location)
            except Exception as e:
                print(f"Error updating roof type: {e}")
                return gr.Radio(choices=['pitched'], value='pitched', visible=True)
        
        def safe_update_roof_insulation(floor_location):
            try:
                return update_roof_insulation_visibility(floor_location)
            except Exception as e:
                print(f"Error updating roof insulation: {e}")
                return gr.Radio(choices=['uninsulated'], value='uninsulated', visible=True)
        
        # Connect event handlers
        floor_location.change(
            fn=safe_update_roof_type,
            inputs=[floor_location],
            outputs=[roof_type]
        )
        
        floor_location.change(
            fn=safe_update_roof_insulation,
            inputs=[floor_location],
            outputs=[roof_insulation]
        )
        
        analyze_btn.click(
            fn=predict_current_energy_and_show_options,
            inputs=[lookup_age_band, total_floor_area, estimated_floor_count, epc_score, built_form,
                    floor_location, roof_type, wall_type, wall_insulation, roof_insulation,
                    glazing_type, main_heat_type, main_fuel_type],
            outputs=[current_performance, wall_renovation, roof_renovation, glazing_renovation,
                    heating_renovation, fuel_change, options_title, calculate_btn, optimization_chart]
        )
        
        calculate_btn.click(
            fn=calculate_renovation_analysis,
            inputs=[lookup_age_band, total_floor_area, estimated_floor_count, epc_score, built_form,
                    floor_location, roof_type, wall_type, wall_insulation, roof_insulation,
                    glazing_type, main_heat_type, main_fuel_type,
                    wall_renovation, roof_renovation, glazing_renovation, heating_renovation, fuel_change],
            outputs=[renovation_results]
        )

    # Launch application with proper conditions
    if __name__ == "__main__":
        print("🚀 Launching Home Retrofit Calculator...")
        demo.launch(share=True, debug=False)  # Set debug=False for production
    else:
        print("✅ Application ready for launch")

except Exception as e:
    print(f"❌ Critical error in interface setup: {e}")
    print(f"Traceback: {traceback.format_exc()}")
    raise