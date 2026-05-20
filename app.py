import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="RegressionX", page_icon="📈", layout="wide")

# Custom CSS for dark theme look
st.markdown("""
<style>
    .metric-container {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #f8fafc;
    }
    .equation-box {
        background-color: rgba(0,0,0,0.3);
        padding: 15px;
        border-radius: 8px;
        font-family: monospace;
        color: #94a3b8;
        border: 1px dashed rgba(255,255,255,0.2);
        word-break: break-all;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 RegressionX Visualizer")
st.markdown("Upload a CSV, select variables, and interactively tweak the regression model!")

# Sidebar for inputs
with st.sidebar:
    st.header("Dataset Configuration")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Filter numerical columns only
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(num_cols) < 2:
        st.error("Dataset needs at least 2 numerical columns.")
        st.stop()
        
    df = df.dropna(subset=num_cols)
    
    with st.sidebar:
        st.markdown("---")
        target_var = st.selectbox("Target Variable (Y)", reversed(num_cols), index=0)
        indep_vars = st.multiselect(
            "Independent Variables (X)", 
            options=[c for c in num_cols if c != target_var],
            default=[num_cols[0]] if num_cols[0] != target_var else []
        )
        
    if not indep_vars:
        st.warning("Please select at least one independent variable.")
        st.stop()
        
    # Data preparation
    X = df[indep_vars]
    y = df[target_var]
    
    # Fit OLS
    model = LinearRegression()
    model.fit(X, y)
    
    true_intercept = model.intercept_
    true_coefs = model.coef_
    
    # Session state for user manual tweaks (sliders)
    st.sidebar.markdown("---")
    st.sidebar.header("Manual Tweaks")
    st.sidebar.markdown("Adjust coefficients to see how it affects the fit.")
    
    if st.sidebar.button("Reset to OLS Fit"):
        for i, var in enumerate(indep_vars):
            st.session_state[f'coef_{var}'] = float(true_coefs[i])
        st.session_state['intercept'] = float(true_intercept)
    
    # Setup sliders
    # Give a reasonable range based on true values
    def make_slider(label, key_name, true_val):
        if key_name not in st.session_state:
            st.session_state[key_name] = float(true_val)
            
        r = max(abs(true_val) * 2, 10.0)
        min_v = float(true_val - r)
        max_v = float(true_val + r)
        
        return st.sidebar.slider(
            label, 
            min_value=min_v, 
            max_value=max_v, 
            value=st.session_state[key_name],
            step=r/1000,
            key=key_name
        )

    user_intercept = make_slider("Intercept", 'intercept', true_intercept)
    user_coefs = []
    for i, var in enumerate(indep_vars):
        val = make_slider(f"Coef: {var}", f'coef_{var}', true_coefs[i])
        user_coefs.append(val)
        
    user_coefs = np.array(user_coefs)
    
    # Predictions and Metrics
    y_pred = user_intercept + np.dot(X, user_coefs)
    residuals = y - y_pred
    
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    # Equation string
    eq_str = f"Y = {user_intercept:.4f}"
    for i, var in enumerate(indep_vars):
        sign = "+" if user_coefs[i] >= 0 else "-"
        eq_str += f" {sign} {abs(user_coefs[i]):.4f}({var})"
        
    # Metrics display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{r2:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Mean Squared Error (MSE)</div>
            <div class="metric-value">{mse:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f"""
    <div class="equation-box">
        {eq_str}
    </div>
    """, unsafe_allow_html=True)
    st.write("---")

    # Plotting
    st.subheader("Prediction Fit")
    
    if len(indep_vars) == 1:
        # Simple Linear 2D
        x_col = indep_vars[0]
        x_min, x_max = df[x_col].min(), df[x_col].max()
        line_x = np.array([x_min, x_max])
        line_y = user_intercept + user_coefs[0] * line_x
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[x_col], y=y, mode='markers', name='Data', marker=dict(color='#3b82f6', opacity=0.8)))
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='Fit', line=dict(color='#ef4444', width=3)))
        
        fig.update_layout(xaxis_title=x_col, yaxis_title=target_var, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
    elif len(indep_vars) == 2:
        # 3D Plot toggle
        use_3d = st.checkbox("Toggle 3D Plot")
        if use_3d:
            x1_col, x2_col = indep_vars[0], indep_vars[1]
            x1, x2 = df[x1_col], df[x2_col]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=x1, y=x2, z=y, mode='markers', 
                marker=dict(size=4, color='#3b82f6', opacity=0.8),
                name='Data'
            ))
            
            # Surface
            x1_grid = np.linspace(x1.min(), x1.max(), 10)
            x2_grid = np.linspace(x2.min(), x2.max(), 10)
            X1, X2 = np.meshgrid(x1_grid, x2_grid)
            Z = user_intercept + user_coefs[0]*X1 + user_coefs[1]*X2
            
            fig.add_trace(go.Surface(x=X1, y=X2, z=Z, colorscale='Reds', opacity=0.6, showscale=False, name='Fit Plane'))
            
            fig.update_layout(scene=dict(xaxis_title=x1_col, yaxis_title=x2_col, zaxis_title=target_var), margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Predicted vs Actual
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_pred, y=y, mode='markers', name='Data', marker=dict(color='#3b82f6', opacity=0.8)))
            
            min_val = min(y_pred.min(), y.min())
            max_val = max(y_pred.max(), y.max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Fit', line=dict(color='#ef4444', dash='dash')))
            
            fig.update_layout(xaxis_title=f'Predicted {target_var}', yaxis_title=f'Actual {target_var}', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Predicted vs Actual for >2 vars
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pred, y=y, mode='markers', name='Data', marker=dict(color='#3b82f6', opacity=0.8)))
        
        min_val = min(y_pred.min(), y.min())
        max_val = max(y_pred.max(), y.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Fit', line=dict(color='#ef4444', dash='dash')))
        
        fig.update_layout(xaxis_title=f'Predicted {target_var}', yaxis_title=f'Actual {target_var}', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    # Residuals Plot
    st.write("---")
    st.subheader("Residuals")
    res_fig = go.Figure()
    res_fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals', marker=dict(color='#10b981', opacity=0.8)))
    res_fig.add_trace(go.Scatter(x=[y_pred.min(), y_pred.max()], y=[0, 0], mode='lines', name='Zero Line', line=dict(color='rgba(255,255,255,0.3)', dash='dot')))
    res_fig.update_layout(xaxis_title="Predicted Values", yaxis_title="Residuals", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(res_fig, use_container_width=True)

else:
    st.info("👈 Please upload a CSV dataset from the sidebar to begin.")
