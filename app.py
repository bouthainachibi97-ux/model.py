import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.optimize import minimize

# ---------------------------------------------------------
# 1. إعداد واجهة التطبيق
# ---------------------------------------------------------
st.set_page_config(
    page_title="Material Inverse Design Platform",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Inverse Design Framework for Advanced Materials")
st.write(
    "هذا النموذج يستخدم الذكاء الاصطناعي وخوارزميات البحث العكسي (Optimization) "
    "لاقتراح الهيكل البنائي الأمثل للمادة بناءً على الخصائص الميكانيكية والحرارية المطلوبة."
)

st.markdown("---")

# ---------------------------------------------------------
# 2. توليد بيانات محاكاة (Synthetic Dataset Generator)
# ---------------------------------------------------------
@st.cache_data
def generate_material_data(num_samples=1000):
    np.random.seed(42)
    
    # Structural Input Features (X)
    density = np.random.uniform(1.2, 2.5, num_samples)      # g/cm³
    layer_thickness = np.random.uniform(5, 50, num_samples)  # nm
    porosity = np.random.uniform(0.05, 0.40, num_samples)    # Ratio
    
    # Non-linear physical relations to simulate target performance (Y)
    # Young's Modulus (GPa) - affected by density and porosity
    youngs_modulus = 120 * (density ** 1.8) * ((1 - porosity) ** 2.5) + np.random.normal(0, 2, num_samples)
    
    # Thermal Conductivity (W/mK) - affected by layer thickness and density
    thermal_cond = 400 * (density / 2.0) * (1 / (1 + 0.02 * layer_thickness)) * (1 - porosity) + np.random.normal(0, 5, num_samples)
    
    # Tensile Strength (MPa)
    tensile_strength = 800 * (density ** 1.5) * (1 - 1.5 * porosity) + np.random.normal(0, 15, num_samples)

    X = np.column_stack((density, layer_thickness, porosity))
    Y = np.column_stack((youngs_modulus, thermal_cond, tensile_strength))
    
    return X, Y

X_data, Y_data = generate_material_data()

# ---------------------------------------------------------
# 3. تدريب نموذج الذكاء الاصطناعي (Forward ML Model)
# ---------------------------------------------------------
@st.cache_resource
def train_forward_model(X, Y):
    # Model to predict Properties (Y) from Microstructure (X)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model = MultiOutputRegressor(rf)
    model.fit(X, Y)
    return model

ml_model = train_forward_model(X_data, Y_data)

# ---------------------------------------------------------
# 4. خوارزمية التصميم العكسي (Inverse Design Solver)
# ---------------------------------------------------------
def inverse_design_solver(target_properties, model):
    # Target: [Young's Modulus, Thermal Cond, Tensile Strength]
    
    # Objective function: Minimize error between predicted & target properties
    def objective_function(x):
        x_input = np.array(x).reshape(1, -1)
        predicted_properties = model.predict(x_input)[0]
        # Normalized Root Mean Squared Error
        error = np.mean(((predicted_properties - target_properties) / target_properties) ** 2)
        return error

    # Bounds for inputs: Density [1.2, 2.5], Layer Thickness [5, 50], Porosity [0.05, 0.40]
    bounds = [(1.2, 2.5), (5.0, 50.0), (0.05, 0.40)]
    initial_guess = [1.85, 27.5, 0.22]

    # Optimization process
    result = minimize(objective_function, initial_guess, method='L-BFGS-B', bounds=bounds)
    
    optimal_microstructure = result.x
    achieved_properties = model.predict(optimal_microstructure.reshape(1, -1))[0]
    
    return optimal_microstructure, achieved_properties, result.fun

# ---------------------------------------------------------
# 5. واجهة التحكم والتفاعل (User Interface)
# ---------------------------------------------------------
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("🎯 1. إدخال الخصائص المستهدفة (Target Requirements)")
    
    target_E = st.slider("معامل المرونة - Young's Modulus (GPa)", 50.0, 350.0, 180.0, step=5.0)
    target_K = st.slider("الموصلية الحرارية - Thermal Conductivity (W/mK)", 50.0, 500.0, 250.0, step=10.0)
    target_S = st.slider("مقاومة الشد - Tensile Strength (MPa)", 200.0, 1200.0, 650.0, step=10.0)
    
    target_vec = np.array([target_E, target_K, target_S])
    
    run_button = st.button("🚀 تشغيل خوارزمية التصميم العكسي", type="primary")

with col_output:
    st.subheader("⚙️ 2. الهيكل البنائي الموصى به (Optimal Design)")
    
    if run_button:
        opt_micro, achieved_props, loss = inverse_design_solver(target_vec, ml_model)
        
        st.success("تم العثور على الهيكل الأمثل بنجاح!")
        
        # Display Recommended Parameters
        st.metric("الكثافة البنائية (Density)", f"{opt_micro[0]:.2f} g/cm³")
        st.metric("سُمْك الطبقات (Layer Thickness)", f"{opt_micro[1]:.1f} nm")
        st.metric("نسبة المسامية (Porosity)", f"{opt_micro[2]*100:.1f} %")
        
        st.markdown("---")
        st.subheader("📊 مقارنة الأداء (Target vs Achieved)")
        
        df_comp = pd.DataFrame({
            "Property": ["Young's Modulus (GPa)", "Thermal Cond. (W/mK)", "Tensile Strength (MPa)"],
            "Target": target_vec,
            "Achieved (ML)": np.round(achieved_props, 1)
        })
        st.dataframe(df_comp, use_container_width=True)
        
        # Visualizing comparison
        fig, ax = plt.subplots(figsize=(6, 3))
        x_indices = np.arange(len(df_comp))
        width = 0.35
        
        ax.bar(x_indices - width/2, df_comp["Target"], width, label='Target', color='#1f77b4')
        ax.bar(x_indices + width/2, df_comp["Achieved (ML)"], width, label='Achieved', color='#2ca02c')
        
        ax.set_xticks(x_indices)
        ax.set_xticklabels(["E (GPa)", "K (W/mK)", "Strength (MPa)"])
        ax.legend()
        ax.set_title("Target vs Predicted Performance")
        
        st.pyplot(fig)
    else:
        st.info("اضغط على زر التشغيل للبدء في استخراج البارامترات الهندسية للمادة.")
