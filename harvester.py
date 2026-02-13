import streamlit as st
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from threading import Thread
import queue
from collections import deque

# --- Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 256  # Reduced to 256 for ~5.8ms latency (lowest without dropouts)
CHUNK_TIME = CHUNK / RATE

# --- VIBRATION SENSITIVITY ---
WAVE_GAIN = 150.0  # Increased from 50.0 for much higher sensitivity
ENERGY_BOOST = 500.0  # Increased from 100.0 for stronger response to quiet sounds

# --- PHYSICAL CALIBRATION ---
WATT_FACTOR = 0.012  # Increased from 0.0024 (5x more sensitive)
INTENSITY_FACTOR = 12000.0  # Increased from 2400.0 (5x more sensitive)

# Global queue for audio data
audio_queue = queue.Queue(maxsize=1)  # Minimal queue size for lowest latency
is_running = False

# Total energy accumulator
total_energy_accumulated = 0.0
silence_threshold = 0.002  # Reduced from 0.01 - detects 5x quieter sounds
is_silent = False

# IoT Application Variables
capacitor_charge_level = 0.0  # In percentage (0-100%)
capacitor_max_charge = 1.0  # Maximum charge in Joules (adjustable)
data_packets_sent = 0
data_packet_energy_cost = 0.0001  # Energy cost per data packet in Joules
total_data_packets_available = 0  # Packets that can be sent with current charge

# Countdown timer variables
energy_rate_history = deque(maxlen=50)  # Track last 50 energy readings for rate calculation
last_packet_time = None
estimated_time_to_next_packet = None

def audio_capture_thread():
    """Background thread to capture audio data"""
    global is_running
    p = pyaudio.PyAudio()
    
    # Open stream with low latency settings
    stream = p.open(
        format=FORMAT, 
        channels=CHANNELS, 
        rate=RATE, 
        input=True, 
        frames_per_buffer=CHUNK,
        stream_callback=None,  # Using blocking mode for stability
        input_device_index=None  # Use default input device
    )
    
    while is_running:
        try:
            # Use non-blocking read with exception handling
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            data_int = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
            
            # Clear old data if queue is full to prevent latency buildup
            if audio_queue.full():
                try:
                    audio_queue.get_nowait()  # Remove oldest data
                except queue.Empty:
                    pass
            
            # Put new data in queue
            audio_queue.put(data_int)
            
        except Exception as e:
            print(f"Audio capture error: {e}")
            break
    
    stream.stop_stream()
    stream.close()
    p.terminate()

def process_audio_data(data_int, mic_gain=1.0):
    """Process audio data and return metrics"""
    # Apply microphone gain first
    data_int = data_int * mic_gain
    
    # Normalize
    data_norm = data_int / 32768.0
    vibration_wave = data_norm * WAVE_GAIN
    
    # Calculate Physics Metrics
    rms = np.sqrt(np.mean(data_norm**2))
    
    # Intensity Calculation (μW/m²)
    intensity_uw = (rms**2) * ENERGY_BOOST * INTENSITY_FACTOR
    
    # Energy Calculation (Joules)
    power_watts = (rms**2) * ENERGY_BOOST * WATT_FACTOR
    energy_joules = power_watts * CHUNK_TIME
    
    return vibration_wave, intensity_uw, energy_joules, rms

def calculate_time_to_next_packet(current_charge, target_charge, capacitor_max, energy_rate):
    """Calculate estimated time until next packet transmission"""
    if energy_rate <= 0:
        return None  # No energy being accumulated
    
    # Calculate energy needed to reach target charge
    current_energy = (current_charge / 100.0) * capacitor_max
    target_energy = (target_charge / 100.0) * capacitor_max
    energy_needed = target_energy - current_energy
    
    if energy_needed <= 0:
        return 0  # Already at or above target
    
    # Calculate time in seconds
    time_seconds = energy_needed / energy_rate
    return time_seconds

def format_time_countdown(seconds):
    """Format seconds into a readable countdown string"""
    if seconds is None:
        return "N/A"
    
    if seconds <= 0:
        return "Sending now!"
    
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def create_waveform_plot(vibration_wave):
    """Create waveform visualization"""
    fig, ax = plt.subplots(figsize=(12, 3))
    x_wave = np.arange(0, len(vibration_wave))
    ax.plot(x_wave, vibration_wave, color='#0066CC', lw=1.5)
    ax.set_title("Vibration Waveform (High Sensitivity)", fontsize=14, fontweight='bold', color='#000000')
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(0, len(vibration_wave))
    ax.set_xlabel("Sample", fontsize=11, color='#000000')
    ax.set_ylabel("Amplitude", fontsize=11, color='#000000')
    ax.set_facecolor('#FFFFFF')
    ax.grid(True, color='#CCCCCC', linestyle='--', alpha=0.7)
    ax.tick_params(colors='#000000', labelsize=10)
    fig.patch.set_facecolor('#FFFFFF')
    plt.tight_layout()
    return fig

def create_bar_plots(intensity_uw, energy_joules):
    """Create intensity and energy bar plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Intensity Plot
    ax1.bar(['Intensity'], [intensity_uw], color='#0099CC', width=0.5)
    ax1.set_title("Sound Intensity", fontsize=14, fontweight='bold', color='#000000')
    ax1.set_ylabel("Watts per m² (μW/m²)", fontsize=11, color='#000000')
    ax1.set_ylim(0, max(500, intensity_uw * 1.2))
    ax1.grid(True, axis='y', alpha=0.5, color='#CCCCCC')
    ax1.set_facecolor('#FFFFFF')
    ax1.tick_params(colors='#000000', labelsize=10)
    
    # Energy Plot
    ax2.bar(['Energy'], [energy_joules], color='#CC0066', width=0.5)
    ax2.set_title("Energy per Frame", fontsize=14, fontweight='bold', color='#000000')
    ax2.set_ylabel("Joules (J)", fontsize=11, color='#000000')
    ax2.set_ylim(0, max(0.0001, energy_joules * 1.2))
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax2.grid(True, axis='y', alpha=0.5, color='#CCCCCC')
    ax2.set_facecolor('#FFFFFF')
    ax2.tick_params(colors='#000000', labelsize=10)
    
    fig.patch.set_facecolor('#FFFFFF')
    plt.tight_layout()
    return fig

def create_capacitor_gauge(charge_level):
    """Create capacitor charge level gauge meter"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=charge_level,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "🔋 Capacitor Charge Level", 'font': {'size': 20, 'color': '#000000'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#000000'}},
        delta={'reference': 80, 'increasing': {'color': "#00CC00"}, 'decreasing': {'color': "#FF6600"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#000000"},
            'bar': {'color': "#0099CC"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#CCCCCC",
            'steps': [
                {'range': [0, 50], 'color': '#FFE6E6'},
                {'range': [50, 80], 'color': '#FFF4E6'},
                {'range': [80, 100], 'color': '#E6FFE6'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'color': "#000000", 'family': "Arial"},
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_packets_gauge(packets_available, packets_sent):
    """Create data packets gauge meter"""
    # Determine max range dynamically
    max_packets = max(100, packets_available + 20)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=packets_available,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"📦 Data Packets Available<br><sub>Sent: {packets_sent} packets</sub>", 
               'font': {'size': 18, 'color': '#000000'}},
        number={'font': {'size': 40, 'color': '#000000'}},
        gauge={
            'axis': {'range': [None, max_packets], 'tickwidth': 2, 'tickcolor': "#000000"},
            'bar': {'color': "#00CC66"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#CCCCCC",
            'steps': [
                {'range': [0, max_packets * 0.33], 'color': '#FFE6E6'},
                {'range': [max_packets * 0.33, max_packets * 0.66], 'color': '#FFF4E6'},
                {'range': [max_packets * 0.66, max_packets], 'color': '#E6FFE6'}
            ],
            'threshold': {
                'line': {'color': "green", 'width': 4},
                'thickness': 0.75,
                'value': max_packets * 0.5
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'color': "#000000", 'family': "Arial"},
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_countdown_gauge(time_seconds):
    """Create countdown timer gauge for next packet transmission"""
    # Convert time to display value (cap at max display time)
    max_display_time = 300  # 5 minutes max display
    display_value = min(time_seconds if time_seconds is not None else max_display_time, max_display_time)
    
    # Determine color based on time remaining
    if time_seconds is None:
        bar_color = "#CCCCCC"
    elif time_seconds <= 10:
        bar_color = "#00CC00"  # Green - sending soon
    elif time_seconds <= 60:
        bar_color = "#FFAA00"  # Orange - within a minute
    else:
        bar_color = "#0099CC"  # Blue - still charging
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "⏱️ Time to Next Packet", 'font': {'size': 20, 'color': '#000000'}},
        number={'suffix': "s" if time_seconds and time_seconds < 60 else "", 'font': {'size': 40, 'color': '#000000'}},
        gauge={
            'axis': {'range': [0, max_display_time], 'tickwidth': 2, 'tickcolor': "#000000"},
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#CCCCCC",
            'steps': [
                {'range': [0, 10], 'color': '#E6FFE6'},
                {'range': [10, 60], 'color': '#FFF4E6'},
                {'range': [60, max_display_time], 'color': '#FFE6E6'}
            ],
            'threshold': {
                'line': {'color': "green", 'width': 4},
                'thickness': 0.75,
                'value': 10
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'color': "#000000", 'family': "Arial"},
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def main():
    global is_running, total_energy_accumulated, is_silent, data_packets_sent
    global capacitor_charge_level, total_data_packets_available
    global energy_rate_history, last_packet_time, estimated_time_to_next_packet
    
    st.set_page_config(
        page_title="Acoustix Harvester",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎵 Real-Time Vibration Energy Harvesting System")
    #st.markdown("**High Sensitivity IoT Energy Harvesting Dashboard** - Detects even the quietest sounds with countdown timer")
    
    # Sidebar Controls
    st.sidebar.header("⚙️ Controls")

     # Control Buttons
    #st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    start_button = col1.button("▶️ Start", use_container_width=True, type="primary")
    stop_button = col2.button("⏸️ Stop", use_container_width=True)
    reset_button = st.sidebar.button("🔄 Reset All Data", use_container_width=True, type="secondary")

    # Handle button clicks
    if start_button and not is_running:
        is_running = True
        audio_thread = Thread(target=audio_capture_thread, daemon=True)
        audio_thread.start()
        st.sidebar.success("✅ System Started")
    
    if stop_button and is_running:
        is_running = False
        st.sidebar.warning("⏸️ System Stopped")
    
    if reset_button:
        total_energy_accumulated = 0.0
        data_packets_sent = 0
        capacitor_charge_level = 0.0
        total_data_packets_available = 0
        energy_rate_history.clear()
        last_packet_time = None
        estimated_time_to_next_packet = None
        st.sidebar.info("🔄 All data reset")
    
   
    
    # Performance info
    st.sidebar.info(
        "⚡ **High Sensitivity Mode Active**\n\n"
        f"• Audio buffer: {CHUNK} samples (~{CHUNK_TIME*1000:.1f}ms)\n"
        f"• Queue size: 1 (minimal buffering)\n"
        f"• Update rate: Real-time\n"
        f"• Sensitivity: 5x enhanced\n"
        f"• Silence threshold: {silence_threshold:.4f}\n\n"
        "🎤 Detects even whispers and quiet sounds!"
    )
    
    # Audio sensitivity controls
    with st.sidebar.expander("🎚️ Sensitivity Settings", expanded=False):
        mic_gain_input = st.slider("Microphone Input Gain", 1.0, 10.0, 3.0, 0.1,
                                   help="Direct amplification of microphone input (Higher = Pick up quieter sounds)")
        wave_gain_input = st.slider("Waveform Gain", 1.0, 300.0, WAVE_GAIN, 1.0,
                                    help="Amplification for waveform visualization (Higher = More sensitive)")
        energy_boost_input = st.slider("Energy Boost Factor", 1.0, 1000.0, ENERGY_BOOST, 10.0,
                                       help="Multiplier for energy calculations (Higher = More energy from quiet sounds)")
        silence_threshold_input = st.slider("Silence Threshold (RMS)", 0.0001, 0.05, silence_threshold, 0.0001,
                                           help="RMS value below which sound is considered silence (Lower = More sensitive)")
    
    # IoT Configuration
    with st.sidebar.expander("🔌 IoT Configuration", expanded=True):
        capacitor_max_charge_input = st.number_input("Capacitor Max Charge (J)", 
                                                     min_value=0.1, max_value=10.0, 
                                                     value=capacitor_max_charge, step=0.1,
                                                     help="Maximum energy storage capacity")
        data_packet_energy_cost_input = st.number_input("Packet Energy Cost (J)", 
                                                        min_value=0.00001, max_value=0.001, 
                                                        value=data_packet_energy_cost, 
                                                        step=0.00001, format="%.5f",
                                                        help="Energy required to transmit one data packet")
        transmission_threshold = st.slider("Auto-Transmit Threshold (%)", 
                                          min_value=50, max_value=100, 
                                          value=80, step=5,
                                          help="Capacitor charge level to trigger packet transmission")
    
    
    
    # Main Display Area
    st.markdown("---")

    # Main content area
    if not is_running:
        st.info("👆 Click 'Start Monitoring' in the sidebar to begin")
        st.stop()
    
    
    # 1. Real-time metrics (Live Metrics)
    metrics_container = st.container()
    
    with metrics_container:
        st.subheader("📊 Live Metrics")
        col1, col2, col3, col4 = st.columns(4)
        metric_intensity = col1.empty()
        metric_energy = col2.empty()
        metric_rms = col3.empty()
        metric_status = col4.empty()
    
    st.markdown("---")
    
    # 2. Waveform plot (Vibration Waveform)
    waveform_container = st.container()
    with waveform_container:
        st.subheader("🌊 Vibration Waveform")
        waveform_plot = st.empty()
    
    st.markdown("---")
    
    # 3. Bar plots (Intensity & Energy Metrics)
    bars_container = st.container()
    with bars_container:
        st.subheader("📊 Intensity & Energy Metrics")
        bar_plots = st.empty()
    
    st.markdown("---")
    
    # 4. Energy Accumulator Display (Total Energy Accumulated)
    accumulator_container = st.container()
    with accumulator_container:
        accumulator_col1, accumulator_col2 = st.columns([1, 2])
        with accumulator_col1:
            st.subheader("⚡ Total Energy Accumulated")
        total_energy_display = accumulator_col2.empty()
        silence_status_display = st.empty()
    
    st.markdown("---")
    
    # 5. IoT Application Display (IoT Application)
    iot_container = st.container()
    with iot_container:
        st.subheader("🔌 IoT Application: Energy Harvesting System")
        iot_col1, iot_col2, iot_col3, iot_col4 = st.columns(4)
        capacitor_display = iot_col1.empty()
        packets_sent_display = iot_col2.empty()
        packets_available_display = iot_col3.empty()
        countdown_display = iot_col4.empty()
        iot_status_display = st.empty()
        
        # Progress bar for capacitor
        capacitor_progress_bar = st.empty()
    
    st.markdown("---")
    
    # 6. Gauge meters for IoT visualization (Meters)
    gauge_container = st.container()
    with gauge_container:
        st.subheader("📈 System Meters")
        gauge_col1, gauge_col2 = st.columns(2)
        capacitor_gauge_display = gauge_col1.empty()
        packets_gauge_display = gauge_col2.empty()
    
    st.markdown("---")
    
    # Main update loop
    update_counter = 0
    start_time = time.time()
    
    while is_running:
        try:
            # Get audio data from queue (with minimal timeout for maximum responsiveness)
            data_int = audio_queue.get(timeout=0.001)  # 1ms timeout
            
            # Process audio data with microphone gain
            vibration_wave, intensity_uw, energy_joules, rms = process_audio_data(data_int, mic_gain_input)
            
            # Detect silence FIRST
            is_silent = rms < silence_threshold_input
            
            # Only accumulate energy when sound is detected (not silent)
            if not is_silent:
                total_energy_accumulated += energy_joules
                energy_rate_history.append(energy_joules)
            
            # Calculate average energy rate
            if len(energy_rate_history) > 0:
                avg_energy_per_frame = np.mean(energy_rate_history)
                energy_rate_per_second = avg_energy_per_frame / CHUNK_TIME
            else:
                energy_rate_per_second = 0
            
            # IoT Application Logic: Capacitor Charging and Data Packets
            # Convert accumulated energy to capacitor charge percentage
            capacitor_charge_level = min((total_energy_accumulated / capacitor_max_charge_input) * 100, 100.0)
            
            # Calculate available data packets based on current charge
            available_energy = total_energy_accumulated
            total_data_packets_available = int(available_energy / data_packet_energy_cost_input)
            
            # Calculate time to next packet
            if not is_silent and energy_rate_per_second > 0:
                estimated_time_to_next_packet = calculate_time_to_next_packet(
                    capacitor_charge_level, 
                    transmission_threshold, 
                    capacitor_max_charge_input, 
                    energy_rate_per_second
                )
            else:
                estimated_time_to_next_packet = None
            
            # Simulate automatic packet sending when capacitor reaches threshold
            if capacitor_charge_level >= transmission_threshold and total_data_packets_available > 0:
                # Send one packet
                data_packets_sent += 1
                total_energy_accumulated -= data_packet_energy_cost_input
                last_packet_time = time.time()
                
                # Recalculate after sending
                capacitor_charge_level = min((total_energy_accumulated / capacitor_max_charge_input) * 100, 100.0)
                total_data_packets_available = int(total_energy_accumulated / data_packet_energy_cost_input)
            
            # Update metrics (every frame)
            metric_intensity.metric(
                "Sound Intensity",
                f"{intensity_uw:.2f} μW/m²" if not is_silent else "0.00 μW/m²",
                delta=None
            )
            metric_energy.metric(
                "Energy",
                f"{energy_joules:.2e} J" if not is_silent else "0.00e+00 J",
                delta=None
            )
            metric_rms.metric(
                "RMS Amplitude",
                f"{rms:.4f}",
                delta=None
            )
            metric_status.metric(
                "Status",
                "🟢 Active" if not is_silent else "🔇 Silent",
                delta=None
            )
            
            # Update total energy display
            if is_silent:
                # When silent, show current accumulated total (no change)
                total_energy_display.metric(
                    "Total Energy",
                    f"{total_energy_accumulated:.6e} J" if total_energy_accumulated > 0 else "0.000000e+00 J",
                    delta=None,
                    help="Cumulative energy collected (only from detected sound)"
                )
                silence_status_display.info("🔇 **Silence Detected** - No energy being accumulated")
            else:
                # When sound detected, show accumulation
                total_energy_display.metric(
                    "Total Energy",
                    f"{total_energy_accumulated:.6e} J",
                    delta=f"+{energy_joules:.2e} J",
                    help="Cumulative energy collected (only from detected sound)"
                )
                silence_status_display.success("🔊 **Sound Detected** - Accumulating energy...")
            
            # Update IoT Application Display
            capacitor_display.metric(
                "🔋 Capacitor Charge",
                f"{capacitor_charge_level:.2f}%",
                delta=None,
                help="Current charge level of energy storage capacitor"
            )
            packets_sent_display.metric(
                "📤 Packets Sent",
                f"{data_packets_sent}",
                delta=None,
                help="Total data packets transmitted"
            )
            packets_available_display.metric(
                "📦 Packets Available",
                f"{total_data_packets_available}",
                delta=None,
                help="Data packets that can be sent with current charge"
            )
            
            # Display countdown timer
            countdown_text = format_time_countdown(estimated_time_to_next_packet)
            if estimated_time_to_next_packet is not None and estimated_time_to_next_packet <= 0:
                countdown_display.metric(
                    "⏱️ Next Packet",
                    "Sending!",
                    delta="Ready",
                    help="Time until next packet transmission"
                )
            elif estimated_time_to_next_packet is not None:
                countdown_display.metric(
                    "⏱️ Next Packet",
                    countdown_text,
                    delta=f"Rate: {energy_rate_per_second:.2e} J/s",
                    help="Estimated time until next packet transmission"
                )
            else:
                countdown_display.metric(
                    "⏱️ Next Packet",
                    "N/A",
                    delta="Waiting for sound",
                    help="Time until next packet transmission"
                )
            
            # Capacitor progress bar
            if capacitor_charge_level >= transmission_threshold:
                capacitor_progress_bar.progress(float(capacitor_charge_level) / 100.0, 
                                               text=f"⚡ Capacitor: {capacitor_charge_level:.1f}% - Transmitting packets!")
            elif capacitor_charge_level >= 50:
                capacitor_progress_bar.progress(float(capacitor_charge_level) / 100.0, 
                                               text=f"🔋 Capacitor: {capacitor_charge_level:.1f}% - Charging...")
            else:
                capacitor_progress_bar.progress(float(capacitor_charge_level) / 100.0, 
                                               text=f"🪫 Capacitor: {capacitor_charge_level:.1f}% - Low charge")
            
            # IoT Status message with countdown
            if is_silent:
                if capacitor_charge_level > 0:
                    iot_status_display.info(
                        f"🔇 **Silence Mode**: Capacitor holds {capacitor_charge_level:.2f}% charge. "
                        f"{total_data_packets_available} packets ready. {data_packets_sent} packets sent. "
                        f"⏱️ Waiting for sound to estimate next transmission..."
                    )
                else:
                    iot_status_display.warning("🔇 **Silence Mode**: No energy stored. Need sound to charge capacitor.")
            else:
                if estimated_time_to_next_packet is not None:
                    iot_status_display.success(
                        f"🔊 **Harvesting Energy**: Charging at {energy_rate_per_second:.2e} J/s. "
                        f"{capacitor_charge_level:.2f}% charged. "
                        f"⏱️ Next packet in ~{countdown_text}. Auto-transmit at {transmission_threshold}%."
                    )
                else:
                    iot_status_display.success(
                        f"🔊 **Harvesting Energy**: Charging capacitor... {capacitor_charge_level:.2f}% charged. "
                        f"Auto-transmit at {transmission_threshold}%."
                    )
            
            # Update plots (throttled - every 3 frames for better performance and lower latency)
            if update_counter % 3 == 0:
                # Waveform
                fig_wave = create_waveform_plot(vibration_wave)
                waveform_plot.pyplot(fig_wave)
                plt.close(fig_wave)
                
                # Bar plots
                fig_bars = create_bar_plots(intensity_uw, energy_joules)
                bar_plots.pyplot(fig_bars)
                plt.close(fig_bars)
                
                # Gauge meters (update every 5 frames for smooth animation)
                fig_capacitor_gauge = create_capacitor_gauge(capacitor_charge_level)
                capacitor_gauge_display.plotly_chart(fig_capacitor_gauge, use_container_width=True, key=f"capacitor_gauge_{update_counter}")
                
                fig_packets_gauge = create_packets_gauge(total_data_packets_available, data_packets_sent)
                packets_gauge_display.plotly_chart(fig_packets_gauge, use_container_width=True, key=f"packets_gauge_{update_counter}")
            
            update_counter += 1
            
            # No delay - maximum responsiveness
            
        except queue.Empty:
            # No data available, continue waiting
            continue
        except Exception as e:
            st.error(f"Error: {e}")
            break
    
    # Cleanup when stopped
    if not is_running:
        metric_status.metric("Status", "🔴 Stopped", delta=None)

if __name__ == "__main__":
    main()
