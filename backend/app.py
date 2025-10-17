import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import streamlit.components.v1 as components
import json
import math


try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    QISKIT_INSTALLED = True
except ImportError:
    QISKIT_INSTALLED = False

\
st.set_page_config(
    page_title="Quantum Earth Digital Twin v2.3",
    layout="wide"
)


CITIES_DATABASE = {
    "Asia": {
        "Philippines": [
            {"name": "Manila", "lat": 14.5995, "lon": 120.9842},
            {"name": "Cebu", "lat": 10.3157, "lon": 123.8854},
            {"name": "Davao", "lat": 7.1907, "lon": 125.4553}
        ],
        "Japan": [
            {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
            {"name": "Osaka", "lat": 34.6937, "lon": 135.5023},
        ],
         "South Korea": [
            {"name": "Seoul", "lat": 37.5665, "lon": 126.9780}
        ]
    },
    "Europe": {
        "United Kingdom": [
            {"name": "London", "lat": 51.5072, "lon": -0.1276},
            {"name": "Manchester", "lat": 53.4808, "lon": -2.2426},
        ],
        "France": [
            {"name": "Paris", "lat": 48.8566, "lon": 2.3522}
        ]
    },
    "Americas": {
        "USA": [
            {"name": "New York", "lat": 40.7128, "lon": -74.0060},
            {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
        ],
        "Canada": [
            {"name": "Toronto", "lat": 43.6532, "lon": -79.3832}
        ]
    }
}




X_train = np.array(range(1, 31)).reshape(-1, 1)
y_train = 30 + (X_train.flatten() * 0.1) + np.random.normal(0, 1, size=30)
temp_model = LinearRegression().fit(X_train, y_train)


co2_X_train = np.array(range(1, 31)).reshape(-1, 1)
co2_y_train = 420 + (co2_X_train.flatten() * 0.5) + np.sin(co2_X_train.flatten()) * 5 + np.random.normal(0, 3, size=30)
co2_model = RandomForestRegressor(n_estimators=20, random_state=42).fit(co2_X_train, co2_y_train.ravel())


if QISKIT_INSTALLED:
    qiskit_simulator = AerSimulator()
    def run_qiskit_risk_assessment(temperature):
        norm_temp = np.clip((temperature - 20) / 35, 0, 1) 
        theta = norm_temp * math.pi
        qc = QuantumCircuit(1, 1)
        qc.ry(theta, 0)
        qc.measure(0, 0)
        compiled_circuit = transpile(qc, qiskit_simulator)
        job = qiskit_simulator.run(compiled_circuit, shots=100)
        result = job.result()
        counts = result.get_counts(compiled_circuit)
        prob_danger = counts.get('1', 0) / 100
        if prob_danger > 0.75: return 'Danger'
        elif prob_danger > 0.3: return 'Warning'
        else: return 'Normal'

# 4. Quantum-Inspired Fallback
def quantum_inspired_risk_assessment(temperature):
    norm_temp = np.clip((temperature - 20) / 35, 0, 1)
    p_high = 0.05 + (norm_temp * 0.9)
    p_medium = 0.10 + (norm_temp * 0.05)
    p_low = max(0, 1.0 - p_high - p_medium)
    p_total = p_low + p_medium + p_high
    return np.random.choice(['Normal', 'Warning', 'Danger'], p=[p_low/p_total, p_medium/p_total, p_high/p_total])

# 5. Quantum Optimization
def quantum_optimization_solar_placement(full_simulation_data):
    city_scores = {}
    for record in full_simulation_data:
        city = record['name']
        if city not in city_scores: city_scores[city] = 0
        if record['status'] == 'Normal': city_scores[city] += 1
    if not city_scores: return None
    return max(city_scores, key=city_scores.get)

# 6. "LLM" Story Generator with Anomaly Detection
def generate_climate_story(country, recommended_city, avg_temp, avg_co2, is_catastrophe, danger_days, temp_anomaly):
    anomaly_text = f"a significant **{temp_anomaly:+.1f}°C anomaly** compared to historical averages"
    if is_catastrophe:
        return f"""
        ### **AI Narrative Log: C-47**
        **Subject:** {country} - Catastrophic Cascade Failure
        **Projection Year:** 2070
        **Analysis:** The simulation reveals a grim outcome. With sustained temperatures showing **{anomaly_text}**, and CO₂ levels reaching a critical **{avg_co2:.0f} ppm**, quantum models predict a permanent 'Danger' state. The AI recommends immediate implementation of Protocol Phoenix.
        """
    if recommended_city:
        return f"""
        ### **AI Narrative Log: O-12**
        **Subject:** {country} - Emergence of a Green Oasis
        **Projection Year:** 2070
        **Analysis:** A hopeful future is projected. The quantum optimization algorithm identifies **{recommended_city}** as a Class-A sustainable energy site. Despite **{anomaly_text}**, this region's stable weather patterns make it a prime candidate for a global green energy hub.
        """
    else:
        return f"""
        ### **AI Narrative Log: F-05**
        **Subject:** {country} - A Region in Climatic Flux
        **Projection Year:** 2070
        **Analysis:** The simulation indicates significant volatility, showing **{anomaly_text}**. The model predicts **{danger_days} days** of 'Danger' level weather events, with CO₂ levels rising to **{avg_co2:.0f} ppm**. The AI strongly recommends a 400% increase in funding for adaptive infrastructure.
        """

# --- HTML & JS for the 3D Globe (Unchanged) ---
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Earth Globe</title>
    <style>
        body { margin: 0; background-color: #000; color: #fff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        canvas { display: block; }
        #infoBox {
            position: absolute;
            top: 10px;
            left: 10px;
            padding: 10px;
            background: rgba(0, 0, 0, 0.75);
            border-radius: 8px;
            border: 1px solid #444;
            display: none;
            width: 200px;
        }
        #infoBox h3 { margin: 0 0 5px 0; color: #00aaff; }
        #infoBox p { margin: 0 0 3px 0; font-size: 14px; }
    </style>
</head>
<body>
    <div id="infoBox"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        const citiesData = __CITIES_DATA_PLACEHOLDER__;
        const energyLayer = __ENERGY_LAYER_PLACEHOLDER__;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
        scene.add(ambientLight);
        const pointLight = new THREE.PointLight(0xffffff, 1.8, 100);
        pointLight.position.set(5, 3, 5);
        scene.add(pointLight);

        const earth = new THREE.Mesh(
            new THREE.SphereGeometry(1, 32, 32),
            new THREE.MeshStandardMaterial({
                map: new THREE.TextureLoader().load('https://unpkg.com/three-globe@2.24.4/example/img/earth-dark.jpg'),
                metalness: 0.3,
                roughness: 0.7
            })
        );
        scene.add(earth);

        const starVertices = [];
        for (let i = 0; i < 10000; i++) {
            const x = (Math.random() - 0.5) * 2000;
            const y = (Math.random() - 0.5) * 2000;
            const z = (Math.random() - 0.5) * 2000;
            starVertices.push(x, y, z);
        }
        const starGeometry = new THREE.BufferGeometry();
        starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starVertices, 3));
        const starMaterial = new THREE.PointsMaterial({ color: 0xffffff, size: 0.1 });
        const starfield = new THREE.Points(starGeometry, starMaterial);
        scene.add(starfield);

        const dataPointsGroup = new THREE.Group();
        earth.add(dataPointsGroup);

        const colorMap = { 'Normal': 0x00ff00, 'Warning': 0xffff00, 'Danger': 0xff0000 };
        const energyColorMap = {
            'Solar Potential': new THREE.Color(0xffff00),
            'Flood Risk': new THREE.Color(0x0000ff),
            'Agricultural Viability': new THREE.Color(0x00ff00)
        };

        function latLonToVector3(lat, lon, radius) {
            const phi = (90 - lat) * (Math.PI / 180);
            const theta = (lon + 180) * (Math.PI / 180);
            const x = -(radius * Math.sin(phi) * Math.cos(theta));
            const y = radius * Math.cos(phi);
            const z = radius * Math.sin(phi) * Math.sin(theta);
            return new THREE.Vector3(x, y, z);
        }
        
        while(dataPointsGroup.children.length > 0){ 
            dataPointsGroup.remove(dataPointsGroup.children[0]); 
        }

        if (Array.isArray(citiesData)) {
            citiesData.forEach(city => {
                let pointGeometry = new THREE.SphereGeometry(0.015, 16, 16);
                let pointMaterial;

                if (energyLayer !== 'Risk Assessment') {
                    const layerValue = city[energyLayer];
                    const baseColor = energyColorMap[energyLayer];
                    pointMaterial = new THREE.MeshBasicMaterial({
                        color: baseColor,
                        opacity: Math.max(0.2, layerValue / 100),
                        transparent: true
                    });
                } else {
                     pointMaterial = new THREE.MeshBasicMaterial({ color: colorMap[city.status] });
                }
                
                if (city.is_recommended) {
                    pointMaterial.color.set(0xffd700);
                    pointGeometry = new THREE.SphereGeometry(0.025, 16, 16);
                }

                const point = new THREE.Mesh(pointGeometry, pointMaterial);
                const position = latLonToVector3(city.lat, city.lon, 1);
                point.position.copy(position);
                point.userData = city;
                dataPointsGroup.add(point);
            });
        }

        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        const infoBox = document.getElementById('infoBox');

        function updateInfoBox(cityData) {
            if (cityData) {
                infoBox.style.display = 'block';
                let recommendedText = cityData.is_recommended ? '<p style="color: #ffd700;"><strong>[OPTIMAL SITE]</strong></p>' : '';
                let energyText = energyLayer !== 'Risk Assessment' ? `<p><strong>${energyLayer}:</strong> ${cityData[energyLayer]}/100</p>` : '';

                infoBox.innerHTML = `
                    <h3>${cityData.name}</h3>
                    ${recommendedText}
                    <p><strong>Status (Day 1):</strong> <span style="color: ${new THREE.Color(colorMap[cityData.status]).getStyle()}">${cityData.status}</span></p>
                    <p><strong>Temp (Day 1):</strong> ${cityData.temp.toFixed(1)}°C</p>
                    <p><strong>CO₂ (Day 1):</strong> ${cityData.co2_ppm.toFixed(0)} ppm</p>
                    ${energyText}
                `;
            } else {
                infoBox.style.display = 'none';
            }
        }

        window.addEventListener('click', (event) => {
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(dataPointsGroup.children);
            updateInfoBox(intersects.length > 0 ? intersects[0].object.userData : null);
        });

        camera.position.z = 3;
        const clock = new THREE.Clock();
        function animate() {
            requestAnimationFrame(animate);
            const elapsedTime = clock.getElapsedTime();
            if (energyLayer === 'Risk Assessment') {
                 dataPointsGroup.children.forEach(point => {
                    const scale = 1.0 + Math.sin(elapsedTime * 5 + point.position.x) * 0.3;
                    point.scale.set(scale, scale, scale);
                });
            }
            controls.update();
            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>
"""

# --- MAIN STREAMLIT APP LAYOUT ---
st.title("🌍 Quantum Earth Digital Twin v2.3")
st.markdown("With Anomaly Detection and Historical Comparison.")

# --- SIDEBAR (Unchanged) ---
with st.sidebar:
    st.header("Simulation Controls")
    selected_continent = st.selectbox("Select Continent:", list(CITIES_DATABASE.keys()))
    selected_country = st.selectbox("Select Country:", list(CITIES_DATABASE[selected_continent].keys()))
    st.divider()
    simulation_days = st.slider("Simulate Prediction for (Days):", 1, 30, 7)
    
    st.header("🤖 AI & Quantum Models")
    use_true_quantum = st.toggle("True Quantum Risk (Qiskit)", value=True, help="Uses a real Qiskit quantum circuit.")
    if not QISKIT_INSTALLED and use_true_quantum:
        st.error("Qiskit not found! Falling back to inspired model.")
    use_qml_regression = st.toggle("Simulated Quantum ML", value=False, help="Simulates faster prediction time for QML.")
    
    st.divider()
    st.header("⚡ Advanced Modes")
    catastrophe_mode = st.toggle("Global Catastrophe Mode")
    optimization_mode = st.toggle("Quantum Optimization Mode")
    scenario_mode = st.toggle("LLM Scenario Generator")

    run_button = st.button("🚀 Run New Simulation")

# --- APP LOGIC & STATE MANAGEMENT ---
# --- BUGFIX: Initialize chart data keys ---
for key in ['latest_data', 'full_sim_data', 'recommended_city', 'story', 'convergence_report', 'anomaly_score', 'temp_chart_data', 'co2_chart_data']:
    if key not in st.session_state: st.session_state[key] = None if key != 'latest_data' else []

if run_button:
    with st.spinner(f"Running simulation for {selected_country}..."):
        cities_to_simulate = CITIES_DATABASE[selected_continent][selected_country]
        full_simulation_results = []
        
        # --- NEW: Generate Historical Data ---
        historical_temps = (temp_model.predict(np.array(range(1, 31)).reshape(-1, 1)) - np.random.normal(1, 0.5, size=30))[:simulation_days]
        historical_co2 = (co2_model.predict(np.array(range(1, 31)).reshape(-1, 1)) - np.random.normal(5, 2, size=30))[:simulation_days]

        # Main simulation loop
        for i, day in enumerate(range(1, simulation_days + 1)):
            for city in cities_to_simulate:
                if catastrophe_mode: 
                    temp = 55.0 + np.random.normal(0, 2.0)
                    co2 = 900 + np.random.normal(0, 20)
                else: 
                    temp = temp_model.predict([[day]])[0] + np.random.normal(0, 1.5)
                    co2 = co2_model.predict([[day]])[0]

                risk = run_qiskit_risk_assessment(temp) if use_true_quantum and QISKIT_INSTALLED else quantum_inspired_risk_assessment(temp)
                
                full_simulation_results.append({
                    'day': day, 'name': city['name'], 
                    'status': risk, 'temp': temp, 'co2_ppm': co2,
                    'Solar Potential': np.random.randint(20, 100),
                    'Flood Risk': np.random.randint(5, 70),
                    'Agricultural Viability': np.random.randint(30, 95)
                })
        
        st.session_state['full_sim_data'] = full_simulation_results
        
        if optimization_mode: st.session_state['recommended_city'] = quantum_optimization_solar_placement(full_simulation_results)
        else: st.session_state['recommended_city'] = None
            
        df_full = pd.DataFrame(full_simulation_results)
        
        # --- NEW: Calculate Anomaly ---
        avg_forecast_temp = df_full['temp'].mean()
        avg_historical_temp = historical_temps.mean()
        st.session_state['anomaly_score'] = avg_forecast_temp - avg_historical_temp

        if scenario_mode:
            avg_co2 = df_full['co2_ppm'].mean()
            danger_days = len(df_full[df_full['status'] == 'Danger'])
            st.session_state['story'] = generate_climate_story(selected_country, st.session_state['recommended_city'], avg_forecast_temp, avg_co2, catastrophe_mode, danger_days, st.session_state['anomaly_score'])
        else:
            st.session_state['story'] = None

        day_1_data = [r for r in full_simulation_results if r['day'] == 1]
        for city_data in day_1_data:
            city_data['is_recommended'] = (city_data['name'] == st.session_state['recommended_city'])
        st.session_state['latest_data'] = day_1_data
        
        
        df_chart = pd.DataFrame({
            'day': range(1, simulation_days + 1),
            'Forecast': df_full.groupby('day')['temp'].mean(),
            'Historical Avg': historical_temps
        }).set_index('day')
        st.session_state['temp_chart_data'] = df_chart

        df_co2_chart = pd.DataFrame({
            'day': range(1, simulation_days + 1),
            'Forecast': df_full.groupby('day')['co2_ppm'].mean(),
            'Historical Avg': historical_co2
        }).set_index('day')
        st.session_state['co2_chart_data'] = df_co2_chart



col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 Simulation Results")
    
   
    if st.session_state.get('anomaly_score') is not None:
        st.metric(label="Temperature Anomaly", value=f"{st.session_state['anomaly_score']:.1f}°C",
                  help="Difference between the forecast and the historical average for this period.")

    if st.session_state.get('recommended_city'):
        st.metric(label="🏆 Optimal Site Recommendation", value=st.session_state['recommended_city'])

    
    if st.session_state.get('temp_chart_data') is not None:
        st.write(f"Full {simulation_days}-day forecast vs. Historical Average:")
        chart_type = st.radio("View Trend For:", ["Temperature (°C)", "CO₂ Levels (ppm)"], horizontal=True, key="chart_select")
        
        if chart_type == "Temperature (°C)":
            st.line_chart(st.session_state['temp_chart_data'])
        else:
            st.line_chart(st.session_state['co2_chart_data'])

    elif not st.session_state.get('full_sim_data'): 
        st.info("Select controls and run simulation.")
        
    if st.session_state.get('story'):
        st.divider()
        st.markdown(st.session_state['story'])


with col2:
    st.header("🛰️ Digital Twin Visualization")
    energy_layer = st.selectbox(
        "Select Visualization Layer:",
        ['Risk Assessment', 'Solar Potential', 'Flood Risk', 'Agricultural Viability']
    )
    
    cities_json = json.dumps(st.session_state.get('latest_data', []))
    html_with_data = html_template.replace("__CITIES_DATA_PLACEHOLDER__", cities_json).replace("__ENERGY_LAYER_PLACEHOLDER__", f"'{energy_layer}'")
    components.html(html_with_data, height=550, scrolling=False)








